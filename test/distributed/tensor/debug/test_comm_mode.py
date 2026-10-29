# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for CommDebugMode, which counts collective ops issued while active. Most
# tests run on a fake process group (world size 2, rank 1) and check per-op counts for
# functional collectives, DTensor redistribution, and (with NCCL/XCCL) raw c10d collectives.
# A second class checks that module forward hooks are removed after the mode exits. Defines
# module-level aliases for the c10d_functional and c10d op namespaces and the device type.
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor._redistribute import use_min_cost_redistribution_plan
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore


c10d_functional = torch.ops.c10d_functional
c10d_ops = torch.ops.c10d
device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class TestCommMode(TestCase):
    # [CLASS DOC] Tests CommDebugMode using a two-rank fake process group created per test
    # (setUp) and destroyed afterwards (tearDown). Attributes: world_size, device_type,
    # world_pg (the default group).
    def tearDown(self):
        # [FUNCTION DOC] Runs the base teardown, then destroys the default process group.
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        # [FUNCTION DOC] Sets world_size to 2, initializes a fake-backend process group as rank
        # 1 with a FakeStore, and records the device type and the default process group.
        super().setUp()
        self.world_size = 2
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=1, world_size=self.world_size, store=store
        )
        self.device_type = device_type
        self.world_pg = dist.distributed_c10d._get_default_group()

    def checksAssert(self, comm_mode, key, expected_value, expected_total_value):
        # [FUNCTION DOC] HELPER: Asserts that the mode's total collective count equals
        # expected_total_value and that the count recorded for the op key equals
        # expected_value. Returns nothing.
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), expected_total_value)
        self.assertEqual(comm_counts[key], expected_value)

        return

    def test_comm_mode(self):
        # [FUNCTION DOC] Runs a wrapper model (all_gather_single, reduce_scatter_single, MLP,
        # all_reduce) under CommDebugMode and asserts a total of 3 collectives, one each of
        # all_reduce, all_gather_into_tensor and reduce_scatter_tensor.
        world_pg = self.world_pg

        class WrapperModel(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an MLPModule with functional collectives.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Builds the inner MLPModule on the given device.
                super().__init__()
                self.model = MLPModule(device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies all_gather_single, reduce_scatter_single, the
                # MLP, then all_reduce (sum) on the world group; returns the all_reduce result.
                x = funcol.all_gather_single(x, 0, world_pg)
                x = funcol.reduce_scatter_single(x, "sum", 0, world_pg)
                out = self.model(x)
                return funcol.all_reduce(out, "sum", world_pg)

        model = WrapperModel(self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            model(torch.randn(20, 10, device=self.device_type))

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 3)
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 1)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 1)

    def test_comm_mode_coalesced(self):
        # [FUNCTION DOC] Same as test_comm_mode but the final collective is a coalesced
        # all_reduce; asserts total 3 with one each of all_reduce_coalesced,
        # all_gather_into_tensor and reduce_scatter_tensor.
        world_pg = self.world_pg

        class WrapperModelCoalesced(nn.Module):
            # [CLASS DOC] HELPER: Like the previous wrapper but ends with all_reduce_coalesced.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Builds the inner MLPModule on the given device.
                super().__init__()
                self.model = MLPModule(device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies all_gather_single, reduce_scatter_single, the
                # MLP, then all_reduce_coalesced on a one-element list; returns that result.
                x = funcol.all_gather_single(x, 0, world_pg)
                x = funcol.reduce_scatter_single(x, "sum", 0, world_pg)
                out = self.model(x)
                return funcol.all_reduce_coalesced([out], "sum", world_pg)

        model = WrapperModelCoalesced(self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            model(torch.randn(20, 10, device=self.device_type))

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 3)
        self.assertEqual(comm_counts[c10d_functional.all_reduce_coalesced], 1)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 1)

    def test_comm_mode_with_dtensor(self):
        # [FUNCTION DOC] Multiplies two row-sharded DTensors under CommDebugMode with the
        # min-cost redistribution plan and asserts exactly one collective was issued, an
        # all_gather_into_tensor, with no all_reduce or reduce_scatter_tensor.
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of x and y.
            return torch.mm(x, y)

        comm_mode = CommDebugMode()
        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with comm_mode, use_min_cost_redistribution_plan():
            f(x_dtensor, y_dtensor)

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 0)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 0)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_comm_mode_with_c10d(self):
        # [FUNCTION DOC] Requires an nccl or xccl backend (decorator); returns early with no
        # accelerator. Runs each raw c10d collective (all_reduce, all_gather, broadcast,
        # gather, reduce, scatter, coalesced variants, reduce_scatter, all_to_all) under the
        # mode and asserts each registers exactly one count under its c10d op key.
        if not torch.accelerator.is_available():
            return

        inp = torch.rand(2, 8, 16).to(device_type)
        all_gather_out = inp.new_empty(self.world_size * 2, 8, 16)

        comm_mode = CommDebugMode()

        # tests c10d all_reduce tracing
        with comm_mode:
            dist.all_reduce(inp)

        self.checksAssert(comm_mode, c10d_ops.allreduce_, 1, 1)

        # tests c10d all_gather_into_tensor tracing
        with comm_mode:
            dist.all_gather_single(all_gather_out, inp)

        self.checksAssert(comm_mode, c10d_ops._allgather_base_, 1, 1)

        # tests c10d reduce_scatter tracing
        with comm_mode:
            dist.reduce_scatter_single(inp, all_gather_out)

        self.checksAssert(comm_mode, c10d_ops._reduce_scatter_base_, 1, 1)

        # tests c10d broadcast tracing
        with comm_mode:
            dist.broadcast(inp, 0)

        self.checksAssert(comm_mode, c10d_ops.broadcast_, 1, 1)

        # tests c10d gather tracing
        with comm_mode:
            dist.gather(inp, None, 0)

        self.checksAssert(comm_mode, c10d_ops.gather_, 1, 1)

        # tests c10d reduce tracing
        with comm_mode:
            dist.reduce(inp, 0)

        self.checksAssert(comm_mode, c10d_ops.reduce_, 1, 1)

        # tests c10d scatter tracing
        with comm_mode:
            dist.scatter(inp, None, 0)

        self.checksAssert(comm_mode, c10d_ops.scatter_, 1, 1)

        # tests c10d all_gather tracing
        output_list = []

        with comm_mode:
            dist.all_gather(output_list, inp, None)

        self.checksAssert(comm_mode, c10d_ops.allgather_, 1, 1)

        # tests c10d allgather_coalesced_ tracing
        output_list = [[inp.new_empty(inp.shape)] for _ in range(self.world_size)]

        with comm_mode:
            dist.all_gather_coalesced(output_list, [inp], None)

        self.checksAssert(comm_mode, c10d_ops.allgather_coalesced_, 1, 1)

        # tests c10d allgather_into_tensor_coalesced_ tracing
        with comm_mode, dist._coalescing_manager():
            dist.all_gather_single(all_gather_out, inp)

        self.checksAssert(comm_mode, c10d_ops.allgather_into_tensor_coalesced_, 1, 1)

        # tests c10d allreduce_coalesced
        with comm_mode:
            dist.all_reduce_coalesced(inp)

        self.checksAssert(comm_mode, c10d_ops.allreduce_coalesced_, 1, 1)

        # tests c10d reduce_scatter_
        with comm_mode:
            dist.reduce_scatter(inp, [inp.clone() for _ in range(self.world_size)])

        self.checksAssert(comm_mode, c10d_ops.reduce_scatter_, 1, 1)

        # tests c10d reduce_scatter_tensor_coalesced
        with comm_mode, dist._coalescing_manager():
            dist.reduce_scatter_single(inp, all_gather_out)

        self.checksAssert(comm_mode, c10d_ops.reduce_scatter_tensor_coalesced_, 1, 1)

        # tests c10d alltoall_
        with comm_mode:
            dist.all_to_all(
                [torch.empty_like(inp) for _ in range(self.world_size)],
                [inp.clone() for _ in range(self.world_size)],
            )

        self.checksAssert(comm_mode, c10d_ops.alltoall_, 1, 1)

        # tests c10d alltoall_base_
        with comm_mode:
            dist.all_to_all_single(inp, inp)

        self.checksAssert(comm_mode, c10d_ops.alltoall_base_, 1, 1)


class TestCommModeModuleReuse(TestCase):
    # [CLASS DOC] Tests for CommDebugMode hook handling that need no process group.
    """Tests that do not require a process group."""

    def test_comm_mode_repeated_module_forward(self):
        # [FUNCTION DOC] Runs a Linear twice inside the mode, then once outside, and asserts
        # no forward or forward-pre hooks remain on the module.
        m = torch.nn.Linear(2, 2)
        with CommDebugMode():
            m(torch.ones(1, 2))
            m(torch.ones(1, 2))

        # After exiting, normal forward should still work
        m(torch.ones(1, 2))

        # No hooks should remain on the module from CommDebugMode
        self.assertEqual(len(m._forward_hooks), 0)
        self.assertEqual(len(m._forward_pre_hooks), 0)

    def test_comm_mode_many_forwards(self):
        # [FUNCTION DOC] Runs a Linear ten times inside the mode and asserts no forward hook
        # remains afterwards, and that a plain forward still works.
        m = torch.nn.Linear(2, 2)
        with CommDebugMode():
            for _ in range(10):
                m(torch.ones(1, 2))

        m(torch.ones(1, 2))
        self.assertEqual(len(m._forward_hooks), 0)

    def test_comm_mode_nested_modules_reuse(self):
        # [FUNCTION DOC] Runs a two-layer MLP twice inside the mode and once outside, then
        # asserts both child Linear modules have no leftover forward hooks.
        class TwoLayerMLP(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module with two Linear layers and a ReLU between.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates fc1 and fc2 as 2x2 Linear layers.
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc2 applied to relu of fc1 of x.
                return self.fc2(torch.relu(self.fc1(x)))

        m = TwoLayerMLP()
        with CommDebugMode():
            m(torch.ones(1, 2))
            m(torch.ones(1, 2))

        m(torch.ones(1, 2))
        self.assertEqual(len(m.fc1._forward_hooks), 0)
        self.assertEqual(len(m.fc2._forward_hooks), 0)


if __name__ == "__main__":
    run_tests()
