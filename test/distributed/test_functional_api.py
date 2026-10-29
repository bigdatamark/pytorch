# Owner(s): ["oncall: distributed"]
    # [MODULE DOC] Tests for the functional collectives API
    # (torch.distributed._functional_collectives): group expansion into tag, rank list and size,
    # process group tag lookup and creation, eager collectives on multi-threaded fake ranks,
    # meta and make_fx tracing, the lazy-wait view optimization of all_gather, torch.compile
    # with real or fake backends, and the autograd variants. Import-time side effects: exits
    # with status 0 if distributed is unavailable, builds the devices list and DEVICE string
    # from the available accelerator (hpu, xpu or cuda), selects BACKEND, and instantiates
    # device-type test classes for three backend-based classes via
    # instantiate_device_type_tests. Running as a script calls run_tests.

import sys
import unittest
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor as dt
from functorch import make_fx
from torch.distributed._local_tensor import LocalTensorMode
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.inductor_utils import HAS_GPU


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    MultiThreadedTestCase,
    requires_accelerator_dist_backend,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    skipIfHpu,
    TEST_CUDA,
    TEST_HPU,
    TEST_WITH_ROCM,
    TEST_XPU,
    TestCase,
)


# NOTE: Instructions for adding new device types to this test file
#
# This test file contains two types of tests:
# 1. Tests that run on both CPUs and accelerators
# 2. Tests that run only on accelerators
#
# We use two variables to manage device types:
# - `devices`: A list containing device types for both CPU and accelerator tests
# - `DEVICE`: A string containing only the accelerator type for accelerator-only tests
#
# To add a new device type:
# 1. Add a new `elif` statement in the if-else ladder below
# 2. Check for the presence of your device (e.g., TEST_NEW_DEVICE)
# 3. Append your device type to the `devices` list
# 4. Assign your device type string to `DEVICE`
#
# Example:
# elif TEST_NEW_DEVICE:
#     devices.append("new_device")
#     DEVICE = "new_device"

DEVICE = "cuda"
devices = ["cpu"]
if TEST_HPU:
    devices.append("hpu")
    DEVICE = "hpu"
elif TEST_XPU:
    devices.append("xpu")
    DEVICE = "xpu"
elif TEST_CUDA:
    devices.append("cuda")


def new_subgroups(group_size: int, pg_tag=None):
    # [FUNCTION DOC] HELPER: Splits the world into consecutive subgroups of group_size ranks
    # using the internal tagged new-group function with the optional pg_tag, for every subgroup
    # (all ranks create every group). Returns a tuple of the subgroup containing the calling
    # rank (None if none) and the list of all subgroups. Requires an initialized default group.
    world_size = dist.get_world_size()
    subgroups = []
    cur_subgroup = None

    for subgroup_id in range(world_size // group_size):
        start_rank = subgroup_id * group_size
        end_rank = start_rank + group_size
        ranks_in_subgroup = list(range(start_rank, end_rank))
        subgroup = c10d._new_group_with_tag(
            ranks=ranks_in_subgroup,
            pg_tag=pg_tag,
        )
        subgroups.append(subgroup)

        rank = dist.get_rank()
        if rank in ranks_in_subgroup:
            cur_subgroup = subgroup

    return cur_subgroup, subgroups


@skipIfHpu
class TestExpand(MultiThreadedTestCase):
    # [CLASS DOC] Tests for expanding a group specification (rank list, nested rank lists,
    # process group, device mesh, mesh plus dim tuple) into tag, rank set and group size with
    # _expand_group. Runs on 4 multi-threaded fake ranks; skipped on HPU.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 4, the number of threads the multithreaded test
        # case spawns.
        return 4

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and then spawns the rank threads.
        super().setUp()
        self._spawn_threads()

    def test_expand_1d_rank_list(self):
        # [FUNCTION DOC] Expands a flat rank list [0,1,2,3]: asserts empty tag, same rank list
        # and size 4, and that a supplied tag is returned instead of the empty one.
        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3])
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3], "bla")
        self.assertEqual("bla", tag)

    def test_expand_2d_rank_list(self):
        # [FUNCTION DOC] Expands a nested rank list [[0,1],[2,3]]: asserts empty tag, flattened
        # rank set and group size 2, honors a supplied tag, and that lists of unequal size raise
        # ValueError mentioning identical group sizes.
        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]])
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(2, group_size)

        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]], "blu")
        self.assertEqual("blu", tag)

        with self.assertRaisesRegex(ValueError, "group sizes must be identical"):
            ft_c._expand_group([[0], [1, 2, 3]])

    def test_expand_process_group(self):
        # [FUNCTION DOC] Expands the world group, a 2-rank subgroup from new_subgroups and a
        # single-rank group tagged my_pg; asserts tag equals the group tag (or the user supplied
        # tag), rank set equals the group ranks and size is correct.
        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD)
        self.assertEqual(c10d._get_group_tag(dist.group.WORLD), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD, "bla")
        self.assertEqual("bla", tag)

        my_pg, _ = new_subgroups(group_size=2)
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        self.assertEqual(c10d._get_group_tag(my_pg), tag)
        self.assertEqual(dist.get_process_group_ranks(my_pg), rankset)
        self.assertEqual(2, group_size)

        my_pg = None
        for i in range(dist.get_world_size()):
            group = c10d._new_group_with_tag([i], pg_tag="my_pg")
            if i == dist.get_rank():
                my_pg = group
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        self.assertEqual("my_pg", tag)
        self.assertEqual([dist.get_rank()], rankset)
        self.assertEqual(1, group_size)

        tag, rankset, group_size = ft_c._expand_group(my_pg, "bla")
        self.assertEqual("bla", tag)

    def test_expand_device_mesh(self):
        # [FUNCTION DOC] Expands a 1D CPU DeviceMesh of 4 ranks (twice) and asserts the tag is
        # that of the mesh group, the rank set is [0,1,2,3] and the size is 4.
        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, group_size = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, group_size = ft_c._expand_group(mesh)
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, group_size)

    def test_expand_device_mesh_tuple(self):
        # [FUNCTION DOC] Expands a 2x2 mesh: passing the mesh alone raises AssertionError
        # about only 1D meshes; passing mesh with dim 0 or dim 1 gives that dimension's group
        # tag, this rank's rank set (even or odd ranks for dim 0, adjacent pairs for dim 1)
        # and size 2.
        mesh = dt.DeviceMesh("cpu", torch.arange(4).view(2, 2))
        with self.assertRaisesRegex(AssertionError, "Only 1D mesh"):
            tag, rankset, group_size = ft_c._expand_group(mesh)

        tag, rankset, group_size = ft_c._expand_group((mesh, 0))
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        expected_rankset = [0, 2] if dist.get_rank() in [0, 2] else [1, 3]
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)

        tag, rankset, group_size = ft_c._expand_group((mesh, 1))
        expected_rankset = [0, 1] if dist.get_rank() in [0, 1] else [2, 3]
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=1)), tag)
        self.assertEqual(expected_rankset, rankset)
        self.assertEqual(2, group_size)


@skipIfHpu
class TestPgTag(MultiThreadedTestCase):
    # [CLASS DOC] Tests of process group tags: groups with the same ranks and tag are
    # deduplicated, untagged groups are distinct duplicates, and a tag plus rank set round-trips
    # to the group through the internal lookup. Runs on 4 multi-threaded ranks; skipped on HPU.
    # The bare string in the body documents the intended behavior and is not a docstring.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 4, the number of threads the multithreaded test
        # case spawns.
        return 4

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and then spawns the rank threads.
        super().setUp()
        self._spawn_threads()

    """
    The behavior we want is as follow:

    - rankset+tag will always result in the same PG.
    Do we enforce this by failing creation of new PGs or returning existing ones?
        Return existing one.

    - default tag gives existing behavior.
        This means we should create duplicates.
    - _expand_group on _default-tagged pg should always resolve to it
        This mean we can't depend on empty tag + rankset.
    """

    def test_pg_creation_with_tag(self):
        # [FUNCTION DOC] Creates subgroups with the same tag twice and asserts the same group
        # object is returned; a different tag or no tag gives different groups, and two untagged
        # creations also differ.
        my_group, _ = new_subgroups(group_size=2, pg_tag="blu")
        my_group2, _ = new_subgroups(group_size=2, pg_tag="blu")
        self.assertEqual(my_group, my_group2)

        my_group3, _ = new_subgroups(group_size=2, pg_tag="blu2")
        self.assertNotEqual(my_group, my_group3)

        my_group4, _ = new_subgroups(group_size=2)
        self.assertNotEqual(my_group, my_group4)

        my_group5, _ = new_subgroups(group_size=2)
        self.assertNotEqual(my_group4, my_group5)

    def test_pg_lookup_roundtrip(self):
        # [FUNCTION DOC] Creates tagged and untagged subgroups and asserts that expanding each
        # group and looking up the resulting tag and rank set via the internal finder returns
        # the identical group.
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="blu2")
        pg_notag0, _ = new_subgroups(group_size=2)
        pg_notag1, _ = new_subgroups(group_size=2)

        def roundtrip(pg):
            # [FUNCTION DOC] HELPER: Expands the group into tag and ranks and returns the
            # process group found by the internal lookup with that tag and rank set.
            tag, rankset, _ = ft_c._expand_group(pg)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        self.assertEqual(pg_tag0, roundtrip(pg_tag0))
        self.assertEqual(pg_tag1, roundtrip(pg_tag1))
        self.assertEqual(pg_notag0, roundtrip(pg_notag0))
        self.assertEqual(pg_notag1, roundtrip(pg_notag1))

    def test_pg_lookup_with_tag(self):
        # [FUNCTION DOC] Asserts that expanding a group with an explicit tag and looking it up
        # returns the group created with that tag; an untagged group given a tag resolves to the
        # tagged group, and an empty tag cannot erase an existing tag.
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="bla")
        pg_notag0, _ = new_subgroups(group_size=2)

        def roundtrip(pg, pg_tag):
            # [FUNCTION DOC] HELPER: Expands the group using the supplied tag override and
            # returns the group found by the internal lookup for the resulting tag and rank set.
            tag, rankset, _ = ft_c._expand_group(pg, pg_tag)
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        self.assertEqual(pg_tag0, roundtrip(pg_tag1, "blu"))
        self.assertEqual(pg_tag0, roundtrip(pg_notag0, "blu"))
        # Cannot erase the tag of a PG
        self.assertEqual(pg_tag0, roundtrip(pg_tag0, ""))

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/107278"
    )
    def test_find_or_create_pg(self):
        # [FUNCTION DOC] Skipped on Linux or ROCm (issue 107278), so effectively disabled on
        # those. Asserts the internal find-or-create call with tag blu and ranks 0 to 3 in
        # groups of 2 returns the same group as a subgroup created with tag blu.
        pg = c10d._find_or_create_pg_by_ranks_and_tag("blu", [0, 1, 2, 3], 2)
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        self.assertEqual(pg, pg_tag0)

    def test_find_root_pg(self):
        # [FUNCTION DOC] Asserts that looking up the empty tag with ranks 0 to 3 yields the
        # world group.
        pg = c10d._find_pg_by_ranks_and_tag("", [0, 1, 2, 3])
        self.assertEqual(dist.group.WORLD, pg)


@instantiate_parametrized_tests
@skipIfHpu
class TestTraceableCollectives(MultiThreadedTestCase):
    # [CLASS DOC] Eager-mode tests of the functional collectives (broadcast, all_reduce,
    # all_reduce_coalesced, all_gather, reduce_scatter and their coalesced forms) over 1D and 2D
    # device meshes on 4 multi-threaded ranks, parametrized over devices. Skipped on HPU.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 4, the number of threads the multithreaded test
        # case spawns.
        return 4

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and then spawns the rank threads.
        super().setUp()
        self._spawn_threads()

    @parametrize("device", devices)
    def test_broadcast(self, device):
        # [FUNCTION DOC] Broadcasts from rank 0 (ones) while other ranks hold zeros over a 1D
        # mesh and asserts every rank gets ones. Parametrized over devices; for a non-cpu device
        # it skips when fewer accelerators than world size exist and binds the rank to its
        # accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        if dist.get_rank() == 0:
            tensor = torch.ones([4], device=device)
        else:
            tensor = torch.zeros([4], device=device)

        mesh = dt.DeviceMesh(device, torch.arange(4))
        res = ft_c.broadcast(tensor, 0, mesh)
        self.assertEqual(res, torch.ones([4], device=device))

    @parametrize("device", devices)
    def test_all_reduce_eager(self, device):
        # [FUNCTION DOC] Eager all_reduce sum of ones over a 1D mesh gives 4s; over the
        # second dim of a 2x2 mesh gives 2s. Parametrized over devices; for a non-cpu device
        # it skips when fewer accelerators than world size exist and binds the rank to its
        # accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        tensor = torch.ones([4], device=device)
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_reduce(tensor, "sum", mesh)
        self.assertEqual(res, torch.tensor([4, 4, 4, 4], dtype=torch.float))

        mesh = dt.DeviceMesh(device, torch.arange(4).view(2, 2))
        res2 = ft_c.all_reduce(tensor, "sum", (mesh, 1))
        self.assertEqual(res2, torch.tensor([2, 2, 2, 2], dtype=torch.float))

    @parametrize("device", devices)
    def test_all_reduce_coalesced_eager(self, device):
        # [FUNCTION DOC] Runs all_reduce_coalesced on two tensors (ones and threes) over the
        # mesh and asserts each result is multiplied by 4. Parametrized over devices; for a non-
        # cpu device it skips when fewer accelerators than world size exist and binds the rank
        # to its accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        t0 = torch.ones([4], device=device)
        t1 = torch.ones([6], device=device) + 2
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_reduce_coalesced([t0, t1], "sum", mesh)
        self.assertEqual(res[0], t0 * 4)
        self.assertEqual(res[1], t1 * 4)

    @parametrize("device", devices)
    def test_all_gather_tensor(self, device):
        # [FUNCTION DOC] All-gathers a 3x3x3 ones tensor along dims 0, 1 and 2 over dim 0 of 1D
        # and 2D meshes and asserts the result is ones with that dim multiplied by the mesh
        # size. Parametrized over devices; for a non-cpu device it skips when fewer accelerators
        # than world size exist and binds the rank to its accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        # testing 1d/2d mesh
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_gather = [0, 1, 2]
            for dim in dims_to_gather:
                output_size = [3, 3, 3]
                output_size[dim] *= mesh.size(0)
                # each rank have its own tensor, all_gather gives a bigger tensor
                local_tensor = torch.ones([3, 3, 3], device=device)
                gathered_tensor = ft_c.all_gather_single(
                    local_tensor, gather_dim=dim, group=(mesh, 0)
                )
                self.assertEqual(gathered_tensor, torch.ones(output_size))

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced(self, device):
        # [FUNCTION DOC] Runs all_gather_single_coalesced on two tensors and asserts two results
        # whose size is multiplied by the world size and which keep the original values.
        # Parametrized over devices; for a non-cpu device it skips when fewer accelerators than
        # world size exist and binds the rank to its accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        tensors = [torch.ones([4], device=device), torch.ones([4], device=device) + 1]
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.all_gather_single_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()], device=device), res[0])
        self.assertEqual(
            torch.ones([4 * dist.get_world_size()], device=device) + 1, res[1]
        )

    @parametrize("device", devices)
    def test_reduce_scatter_tensor(self, device):
        # [FUNCTION DOC] Reduce-scatters a ones tensor along dims 0 and 1 over 1D and 2D meshes
        # and asserts the result is ones of the base shape scaled by the group size.
        # Parametrized over devices; for a non-cpu device it skips when fewer accelerators than
        # world size exist and binds the rank to its accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())

        # testing 1d/2d mesh
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        for mesh in [mesh_1d, mesh_2d]:
            dims_to_scatter = [0, 1]
            for dim in dims_to_scatter:
                group_size = mesh.size(0)
                input_size = [3, 3]
                output_size = [3, 3]
                output_size[dim] *= group_size
                input_tensor = torch.ones(output_size, device=device)
                res_num = 1 * group_size
                rs_tensor = ft_c.reduce_scatter_single(
                    input_tensor, "sum", scatter_dim=dim, group=(mesh, 0)
                )
                self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)

    @parametrize("device", devices)
    def test_reduce_scatter_into_tensor_coalesced(self, device):
        # [FUNCTION DOC] Runs reduce_scatter_single_coalesced on two int tensors with scatter
        # dims [0,0] and sum and asserts the results are 4 and 8. Parametrized over devices; for
        # a non-cpu device it skips when fewer accelerators than world size exist and binds the
        # rank to its accelerator index.
        if device != "cpu":
            if torch.accelerator.device_count() < self.world_size:
                self.skipTest("Not enough accelerator devices")
            torch.accelerator.set_device_index(dist.get_rank())
        tensors = [
            torch.ones([4], dtype=torch.int64, device=device),
            torch.ones([4], dtype=torch.int64, device=device) + 1,
        ]
        mesh = dt.DeviceMesh(device, torch.arange(4))

        res = ft_c.reduce_scatter_single_coalesced(tensors, "sum", [0, 0], mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.tensor([4], device=device), res[0])
        self.assertEqual(torch.tensor([8], device=device), res[1])


class TestMetaCollectives(TestCase):
    # [CLASS DOC] Single test checking the functional collectives with the meta device.
    def test_all_reduce(self):
        # [FUNCTION DOC] Asserts that all_reduce of a meta-device tensor with sum returns a
        # tensor of the same size, without a real group.
        x = torch.rand((2, 3, 4), device="meta")
        out = ft_c.all_reduce(x, "sum", "0")
        self.assertEqual(x.size(), out.size())


@skipIfHpu
class TestGradCollectives(MultiThreadedTestCase):
    # [CLASS DOC] Checks autograd through functional collectives on 2 multi-threaded ranks;
    # skipped on HPU.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 2, the number of threads the multithreaded test
        # case spawns.
        return 2

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and then spawns the rank threads.
        super().setUp()
        self._spawn_threads()

    def test_all_reduce(self):
        # [FUNCTION DOC] Runs all_reduce on a tensor that requires grad, adds another tensor,
        # backpropagates the sum and asserts x gets a gradient.
        x = torch.rand([4], requires_grad=True)
        y = torch.rand([4], requires_grad=True)
        out = ft_c.all_reduce(x, "sum", dist.group.WORLD)
        (out + y).sum().backward()
        self.assertIsNotNone(x.grad)


class TestMakeFx(TestCase):
    # [CLASS DOC] Tests tracing functional collectives with make_fx, using a single-process fake
    # group because make_fx is not thread-safe. setUp initializes the fake default group (rank
    # 0, world size 2) and tearDown asserts fx tracing is no longer active.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp, stores rank 0 and world size 2
        # and initializes a fake default process group. Side effect: global default
        # group is created.
        super().setUp()
        # make_fx is not thread-safe due to patching nd mutating global states
        # so create a fake_pg.
        self.rank = 0
        self.world_size = 2
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown and asserts that fx symbolic tracing is
        # not still active. Does not destroy the process group.
        super().tearDown()

        self.assertFalse(torch.fx._symbolic_trace.is_fx_tracing())

    def test_all_reduce_tracing(self):
        # [FUNCTION DOC] Traces all_reduce plus one with make_fx for the world group, a 1D mesh
        # and a mesh with dim, and uses FileCheck to assert an all_reduce and wait_tensor
        # appear for the group case and that no get_attr node appears before wait_tensor for
        # the mesh cases.
        def allred(input):
            # [FUNCTION DOC] HELPER: Traced function: sum all_reduce over the world group of the
            # input, plus one.
            return ft_c.all_reduce(input, "sum", group=dist.group.WORLD) + 1

        graph = make_fx(allred)(torch.rand(4))
        FileCheck().check("all_reduce").check("wait_tensor").run(str(graph.graph))

        mesh = dt.DeviceMesh("cpu", torch.arange(self.world_size))

        def allred_mesh(input):
            # [FUNCTION DOC] HELPER: Traced function: sum all_reduce over the captured mesh,
            # plus one.
            return ft_c.all_reduce(input, "sum", mesh) + 1

        mesh_graph = make_fx(allred_mesh)(torch.rand(4))
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_graph.graph)
        )

        def allred_mesh_dim(input):
            # [FUNCTION DOC] HELPER: Traced function: sum all_reduce over dim 0 of the captured
            # mesh, plus one.
            return ft_c.all_reduce(input, "sum", (mesh, 0)) + 1

        mesh_dim_graph = make_fx(allred_mesh_dim)(torch.rand(4))
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_dim_graph.graph)
        )


class TestAllGatherViewOptimization(TestCase):
    # [CLASS DOC] Verifies when all_gather_single delays the wait (returns an
    # AsyncCollectiveTensor) versus waits early. setUp creates a fake 4-rank group at rank 0 and
    # tearDown destroys it.
    """Validate that all_gather_tensor delays wait() when the view optimization
    applies and calls wait() early when the fallback path is needed."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp, sets world size 4, destroys any existing
        # default group and initializes a fake group as rank 0.
        super().setUp()
        self.world_size = 4
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=0,
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group and then runs the
        # base tearDown.
        dist.destroy_process_group()
        super().tearDown()

    def test_view_path_delays_wait(self):
        # [FUNCTION DOC] Gathers a tensor of shape 1x8 along dim 1 and asserts the result is an
        # AsyncCollectiveTensor, meaning the view optimization delayed the wait.
        # Shape [1, 8] with gather_dim=1: after all_gather -> [4, 8]
        # numel_between = prod(shape[1:1]) = 1, so view optimization applies.
        # Result should be an AsyncCollectiveTensor (wait delayed).
        t = torch.randn(1, 8)
        res = ft_c.all_gather_single(t, gather_dim=1, group=dist.group.WORLD)
        self.assertIsInstance(res, ft_c.AsyncCollectiveTensor)

    def test_fallback_path_waits_early(self):
        # [FUNCTION DOC] Gathers a tensor of shape 2x8 along dim 1 and asserts the result is not
        # an AsyncCollectiveTensor, meaning the wait was done early.
        # Shape [2, 8] with gather_dim=1: after all_gather -> [8, 8]
        # numel_between = prod(shape[1:1]) = 1 but shape[0]=8 != group_size=4,
        # so view optimization does NOT apply.
        # Result should be a plain tensor (wait called early).
        t = torch.randn(2, 8)
        res = ft_c.all_gather_single(t, gather_dim=1, group=dist.group.WORLD)
        self.assertNotIsInstance(res, ft_c.AsyncCollectiveTensor)


BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO

# Adding support for HCCL backend
# To add a different backend
# add an elif to the same chain with a conditional checking for the device type (along the lines of TEST_HPU or TEST_CUDA)
# And then set the BACKEND variable appropriately.
if TEST_HPU:
    BACKEND = dist.Backend.HCCL
elif TEST_XPU:
    BACKEND = dist.Backend.XCCL


# allows you to check for multiple accelerator irrespective of device type
# to add new device types to this check simply follow the same format
# and append an elif with the conditional and appropriate device count function for your new device
def exit_if_lt_x_accelerators(x):
    # [FUNCTION DOC] HELPER: If an accelerator is available and its device count is below x,
    # exits the process with the skip exit code registered for x GPUs. Otherwise returns. Side
    # effect: terminates the process.
    if torch.accelerator.is_available():
        if torch.accelerator.device_count() < x:
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)


def with_comms(func=None):
    # [FUNCTION DOC] Decorator for test methods of the distributed backend test classes.
    # Called with no argument (parenthesized use) it returns a partial that decorates later;
    # otherwise wraps func so that for NCCL or XCCL it exits with the skip code if too few
    # accelerators exist, sets the device keyword argument to DEVICE, creates a process group
    # via create_pg and stores it on the test, runs func and always destroys the process group
    # in a finally block.
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper described above; forwards extra arguments, overrides
        # the device keyword, and cleans up the process group even if the test fails.
        if (
            BACKEND == dist.Backend.NCCL or BACKEND == dist.Backend.XCCL
        ) and torch.accelerator.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        kwargs["device"] = DEVICE
        self.pg = self.create_pg(device=DEVICE)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class TestCollectivesWithDistributedBackend(DistributedTestBase):
    # [CLASS DOC] Tests of functional collectives with a real distributed backend through
    # DistributedTestBase and with_comms; instantiated per device for DEVICE only. Includes
    # coalesced all-gather, all_to_all_single variants and torch.compile tracing tests.
    @with_comms()
    def test_all_gather_into_tensor_coalesced(self, device):
        # [FUNCTION DOC] Exits if too few accelerators exist, then all-gathers two tensors with
        # all_gather_single_coalesced over a mesh and asserts two results with values scaled to
        # world size, compared against cpu tensors.
        exit_if_lt_x_accelerators(self.world_size)
        tensors = [
            torch.ones([4], device=device),
            torch.ones([4], device=device) + 1,
        ]
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))

        res = ft_c.all_gather_single_coalesced(tensors, mesh)
        self.assertEqual(2, len(res))
        self.assertEqual(torch.ones([4 * dist.get_world_size()]), res[0])
        self.assertEqual(torch.ones([4 * dist.get_world_size()]) + 1, res[1])

    @with_comms()
    def test_all_to_all_single(self, device):
        # [FUNCTION DOC] Runs all_to_all_single with per-rank split sizes on a 2D tensor over a
        # mesh and asserts the output has chunk i filled with value i plus one.
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), 5, device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_1d_input(self, device):
        # [FUNCTION DOC] Same as the 2D case but with a 1D input tensor: all_to_all_single with
        # equal output and input split sizes, asserting chunk i of the output holds i plus one.
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), device=device) * (rank + 1)
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @with_comms()
    def test_all_to_all_single_split_sizes_none(self, device):
        # [FUNCTION DOC] Runs all_to_all_single with no split sizes on a square tensor valued
        # rank plus one and asserts chunk i of the output equals i plus one.
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        rank = dist.get_rank()

        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        y = ft_c.all_to_all_single(
            x, output_split_sizes=None, input_split_sizes=None, group=mesh
        )
        expected = []
        for idx, tensor in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @with_comms()
    def test_tracing(self, device):
        # [FUNCTION DOC] Skipped without a GPU or an nccl/xccl backend. Compiles a function that
        # sums via all_reduce with fullgraph and runs it on a random tensor with the test
        # process group.
        def allreduce(t, pg):
            # [FUNCTION DOC] HELPER: Compiled function returning sum all_reduce of the tensor
            # over the given group.
            return ft_c.all_reduce(t, "sum", pg)

        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        compiled_allreduce(torch.randn(8, device=device), self.pg)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_tracing_with_fakepg(self, device=DEVICE):
        # [FUNCTION DOC] Skipped without a GPU. Exits if too few accelerators exist; defines
        # (but never calls) a compiled allreduce, initializes a fake group of world size 8 as
        # rank 0, runs the eager version on a random tensor and destroys the group. Not
        # decorated by with_comms and has a default argument for device.
        # KNOWN ISSUE: the compiled function is created but never executed, so tracing with the
        # fake group is not actually exercised (the variable is marked unused).
        exit_if_lt_x_accelerators(self.world_size)

        def allreduce(t, pg):
            # [FUNCTION DOC] HELPER: Function returning sum all_reduce of the tensor over
            # the group.
            return ft_c.all_reduce(t, "sum", pg)

        compiled_allreduce = torch.compile(allreduce, fullgraph=True)  # noqa: F841
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=8,
        )
        allreduce(torch.randn(8, device=device), pg=dist.group.WORLD)
        dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @with_comms()
    def test_tracing_with_dce_code(self, device):
        # [FUNCTION DOC] Skipped without a GPU or nccl/xccl backend. Returns immediately if
        # world size exceeds 2. Compiles a function that does permute_tensor with permutation
        # [1,0], waits if the result has a wait method, and returns it on rank 0 but five times
        # the input on other ranks; runs it and a barrier.
        if self.world_size > 2:
            return

        def func(batch, group, rank):
            # [FUNCTION DOC] HELPER: Function under compilation; permutes the batch among
            # ranks, waits for the result if async, returns the result on rank 0 and the scaled
            # input otherwise.
            ret = ft_c.permute_tensor(batch, [1, 0], group)
            if hasattr(ret, "wait"):
                ret = ret.wait()
            if rank == 0:
                return ret
            else:
                return batch * 5

        compiled_func = torch.compile(func)
        compiled_func(torch.ones((100,), device=device), self.pg, self.rank)
        dist.barrier()


class TestDistributedBackendCollectivesWithWorldSize4(
    TestCollectivesWithDistributedBackend
):
    # [CLASS DOC] Variant of the backend collectives tests with a world size of 4, adding a
    # permute_tensor test on sub-groups of a 2D mesh. Inherits all parent tests.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 4 as the world size.
        return 4

    @with_comms()
    def test_permute_tensor_with_sub_group(self, device):
        # [FUNCTION DOC] Exits if too few accelerators exist. Builds a 2D mesh (dp, tp), and for
        # each mesh dim permutes a tensor of two floats between the two ranks with permutation
        # [1,0]; asserts the received tensor equals the peer rank's data with a descriptive
        # failure message.
        exit_if_lt_x_accelerators(self.world_size)
        mesh_dim_names = ["dp", "tp"]

        mesh_2d = dt.init_device_mesh(
            device, (2, self.world_size // 2), mesh_dim_names=mesh_dim_names
        )

        for mesh_name in mesh_dim_names:
            mesh = mesh_2d[mesh_name]
            rank = mesh.get_local_rank()

            # rank0: [0., 1.], rank1: [2., 3.]
            send_tensor = torch.arange(2, dtype=torch.float32, device=device) + 2 * rank
            recvd_tensor = ft_c.permute_tensor(send_tensor, [1, 0], group=mesh)

            # rank0: [2., 3.], rank1: [0., 1.]
            expected = torch.arange(2, dtype=torch.float32, device=device) + 2 * (
                (rank - 1 + 2) % 2
            )
            self.assertEqual(
                recvd_tensor,
                expected,
                msg=lambda msg: f"{msg}\nExpected {expected} on {self.rank=} (local_rank={rank}), "
                f"but received {recvd_tensor} instead.",
            )


class TestFunctionalAutograd(TestCase):
    # [CLASS DOC] Autograd tests of the functional collectives on a fake 2-rank group using
    # LocalTensorMode. setUp initializes the fake default group at rank 0, tearDown destroys it.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp, sets world size 2 and initializes the fake
        # default process group at rank 0.
        super().setUp()
        self.world_size = 2
        dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group and runs the base tearDown.
        dist.destroy_process_group()
        super().tearDown()

    def test_all_to_all_single(self) -> None:
        # [FUNCTION DOC] Under LocalTensorMode, runs all_to_all_single_autograd and asserts
        # output shape and values (twice the ones input), a grad_fn, requires_grad, and after
        # backward a gradient of 2s.
        with LocalTensorMode(frozenset(range(self.world_size))):
            group = dist.group.WORLD.group_name

            t = torch.ones((self.world_size, 2), requires_grad=True)

            def my_func(t: torch.Tensor, world_size: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Doubles the tensor, checks it requires grad (raising
                # AssertionError otherwise), applies all_to_all_single_autograd with unit splits
                # and adds zero to the result.
                sizes = [1] * world_size
                t = t * 2
                if not t.requires_grad:
                    raise AssertionError("Expected t.requires_grad to be True")
                out = ft_c.all_to_all_single_autograd(t, sizes, sizes, group)
                out = out + 0
                return out

            out = my_func(t, self.world_size)
            self.assertEqual(out.shape, t.shape)
            self.assertEqual(out, torch.full_like(t, 2.0))
            self.assertIsNotNone(out.grad_fn)
            self.assertTrue(out.requires_grad)
            loss = out.sum()
            loss.backward()
            self.assertEqual(t.grad, torch.full_like(t, 2.0))

    def test_all_gather_tensor(self) -> None:
        # [FUNCTION DOC] Under LocalTensorMode, for gather dims 0, 1 and 2 runs
        # all_gather_single_autograd and asserts the output is ones with the gathered dim
        # doubled and that backward gives a gradient filled with world size.
        with LocalTensorMode(frozenset(range(self.world_size))):
            group = dist.group.WORLD.group_name

            def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Raises AssertionError if the input does not require
                # grad; otherwise gathers it along dim via all_gather_single_autograd and scales
                # by one.
                if not t.requires_grad:
                    raise AssertionError("Expected t.requires_grad to be True")
                out = ft_c.all_gather_single_autograd(
                    t * 1.0,
                    gather_dim=dim,
                    group=group,
                )
                out = out * 1.0
                return out

            dims_to_gather = [0, 1, 2]
            for dim in dims_to_gather:
                output_size = [3, 3, 3]
                output_size[dim] *= self.world_size
                local_tensor = torch.ones([3, 3, 3], requires_grad=True)
                gathered_tensor = my_func(local_tensor, dim)
                self.assertEqual(gathered_tensor, torch.ones(output_size))

                gathered_tensor.sum().backward()
                self.assertEqual(
                    local_tensor.grad,
                    torch.full((3, 3, 3), fill_value=float(self.world_size)),
                )

    def test_reduce_scatter_tensor(self) -> None:
        # [FUNCTION DOC] Under LocalTensorMode, for scatter dims 0 and 1 runs
        # reduce_scatter_single_autograd on ones and asserts the output is ones scaled by group
        # size and that backward gives ones as the input gradient.
        with LocalTensorMode(frozenset(range(self.world_size))):
            group = dist.group.WORLD.group_name

            def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Raises AssertionError if the input does not require
                # grad; otherwise applies reduce_scatter_single_autograd with sum on input_tensor
                # taken from the enclosing test scope, assigned in the loop before each call.
                # KNOWN ISSUE: the argument t is only checked for requires_grad; the result is
                # computed from the caller-scope tensor, so the argument is otherwise unused.
                if not t.requires_grad:
                    raise AssertionError("Expected t.requires_grad to be True")
                rs_tensor = (
                    ft_c.reduce_scatter_single_autograd(
                        input_tensor * 1.0, "sum", scatter_dim=dim, group=group
                    )
                    * 1.0
                )
                return rs_tensor

            dims_to_scatter = [0, 1]
            for dim in dims_to_scatter:
                group_size = self.world_size
                input_size = [3, 3]
                output_size = [3, 3]
                output_size[dim] *= group_size
                input_tensor = torch.ones(output_size, requires_grad=True)
                rs_tensor = my_func(input_tensor, dim)
                res_num = 1 * group_size
                self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)
                rs_tensor.sum().backward()
                self.assertEqual(
                    input_tensor.grad,
                    torch.full(output_size, fill_value=1.0),
                )


class TestFunctionalAutogradWithDistributedBackend(DistributedTestBase):
    # [CLASS DOC] Autograd test of the functional collectives using a real distributed backend
    # through with_comms; instantiated per device for DEVICE only.
    @with_comms()
    def test_all_to_all_single(self, device) -> None:
        # [FUNCTION DOC] Runs all_to_all_single_autograd with unit splits on a doubled ones
        # tensor using the test process group; asserts output shape and values, a grad_fn,
        # requires_grad, and a backward gradient of 2s.
        group = self.pg
        t = torch.ones((self.world_size, 2), requires_grad=True, device=device)

        sizes = [1] * self.world_size
        if not t.requires_grad:
            raise AssertionError("Expected t.requires_grad to be True")
        out = ft_c.all_to_all_single_autograd(t * 2, sizes, sizes, group) + 0

        self.assertEqual(out.shape, t.shape)
        self.assertEqual(out, torch.full_like(t, 2.0))
        self.assertIsNotNone(out.grad_fn)
        self.assertTrue(out.requires_grad)
        loss = out.sum()
        loss.backward()
        self.assertEqual(t.grad, torch.full_like(t, 2.0))


# Update the supported devices in DEVICE
instantiate_device_type_tests(
    TestCollectivesWithDistributedBackend, globals(), only_for=DEVICE, allow_xpu=True
)
instantiate_device_type_tests(
    TestDistributedBackendCollectivesWithWorldSize4,
    globals(),
    only_for=DEVICE,
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestFunctionalAutogradWithDistributedBackend,
    globals(),
    only_for=DEVICE,
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
