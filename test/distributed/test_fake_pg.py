# Owner(s): ["oncall: distributed"]
    # [MODULE DOC] Tests for the fake process group backend (backend name fake) of
    # torch.distributed. They check that collectives (all_reduce, all_gather, reduce_scatter,
    # scatter, gather, all_to_all, send/recv, broadcast) run without real communication, that
    # outputs get copy semantics even when inputs require grad, that argument validation errors
    # are raised, and that FSDP, tensor parallel, tracing and split_group work on a fake world.
    # Import-time side effects: exits the process with status 0 if torch.distributed is
    # unavailable, and computes the module-level device_type from get_devtype. Running the file
    # as a script calls run_tests.

import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C._distributed_c10d import FakeProcessGroup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import HAS_ACCELERATOR
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

device_type = get_devtype().type


class TestFakePG(TestCase):
    # [CLASS DOC] Test case for FakeProcessGroup and the fake backend. Each test initializes a
    # default process group with the fake backend (varying rank and world size, optionally with
    # a FakeStore or HashStore) and calls collectives or higher-level wrappers on it. tearDown
    # destroys the default group so tests do not leak state. Parametrized tests are expanded by
    # instantiate_parametrized_tests at module end.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown, then destroys the default process
        # group, ignoring the AssertionError raised when no group was initialized (for example
        # in tests that never created one).
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def test_invalid_rank(self):
        # [FUNCTION DOC] Checks that init_process_group with the fake backend, rank 3 and world
        # size 2 raises a RuntimeError whose message says the rank must be below the world size.
        with self.assertRaisesRegex(
            RuntimeError, "Cannot init process group where rank .* >= world_size"
        ):
            dist.init_process_group(backend="fake", rank=3, world_size=2)

    def test_all_reduce(self):
        # [FUNCTION DOC] Initializes a 2-rank fake group as rank 1 and runs all_reduce on a 3x3
        # tensor in place; asserts only that the shape is unchanged.
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_set_timeout(self):
        # [FUNCTION DOC] Creates a FakeProcessGroup directly via the internal constructor and
        # calls set_timeout on it; asserts nothing and passes if no exception is raised (setting
        # a timeout is a no-op on the fake group).
        # FakeProcessGroup used to inherit Backend's default set_timeout, which
        # raised "does not support setting timeout". Setting a timeout is a
        # no-op for a group that does no real communication, but it must not
        # raise.
        backend = FakeProcessGroup._create_internal(0, world_size=2)
        backend.set_timeout(timedelta(seconds=42))

    def test_set_timeout_via_dist(self):
        # [FUNCTION DOC] Initializes a fake default group with a FakeStore and calls the dist
        # level set_timeout with 30 seconds; passes if it does not raise, which covers
        # forwarding through ProcessGroup to a backend that has null options.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        # Forwards through ProcessGroup to the fake backend; must not raise (the
        # default init path leaves the backend options null).
        dist.set_timeout(timedelta(seconds=30))

    def test_add_ephemeral_timeout_is_noop(self):
        # [FUNCTION DOC] Checks that adding an ephemeral timeout is a no-op that does not raise,
        # both on a directly created FakeProcessGroup and through the internal helper that
        # applies it to all process groups of an initialized fake world.
        backend = FakeProcessGroup._create_internal(0, world_size=2)
        backend._add_ephemeral_timeout(timedelta(seconds=42))

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=42))

    def test_allgather(self):
        # [FUNCTION DOC] Runs all_gather as rank 1 of 2 into a list of two empty tensors and
        # asserts each output keeps shape 3x3.
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out_tensor in output_tensors:
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_allgather_into_tensor_requires_grad(self):
        # [FUNCTION DOC] Regression test: all_gather_single into a flat output of size 4 times
        # world size from an input that requires grad must succeed (it used to hit an autograd
        # safety check because the fake backend writes into chunk views of the output). Asserts
        # the output shape.
        # Regression test: fake PG's _allgather_base uses chunk() which
        # produces multi-output views. Writing into those views inplace when
        # the input requires grad used to trip an autograd safety check.
        dist.init_process_group(backend="fake", rank=0, world_size=4)
        world_size = dist.get_world_size()

        input_tensor = torch.randn(4, requires_grad=True)
        output_tensor = torch.empty(4 * world_size)
        dist.all_gather_single(output_tensor, input_tensor)
        self.assertEqual(output_tensor.shape, (4 * world_size,))

    def test_reduce_scatter(self):
        # [FUNCTION DOC] Runs reduce_scatter as rank 1 of 2 with a list of two 3x3 tensors and
        # asserts the output shape stays 3x3.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_construct_fsdp(self):
        # [FUNCTION DOC] Skipped without an accelerator. Initializes a fake 2-rank group with a
        # FakeStore and constructs an FSDP wrapper around a small linear layer on the
        # accelerator; passes if construction does not raise.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device=device_type))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_fake_e2e(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Wraps a small three-layer
        # module in FSDP with use_orig_params on a fake 2-rank group, then runs one forward
        # pass, a backward pass and an Adam step; passes if nothing raises.
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        my_module = nn.Sequential(
            nn.Linear(2, 3, device=device_type),
            nn.ReLU(),
            nn.Linear(3, 2, device=device_type),
        )
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fake_pg_tracing(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Traces a function calling the
        # functional all_gather_single on the default fake group with make_fx and uses FileCheck
        # to assert that the graph contains an all_gather node followed by a wait_tensor node.
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            # [FUNCTION DOC] HELPER: Traced function that all-gathers its input along dim 0 over
            # the default group via the functional collectives API and returns the result.
            return funcol.all_gather_single(tensor, 0, default_pg)

        gm = make_fx(allgather_fn)(torch.randn(2, 2, device=device_type))
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        # [FUNCTION DOC] Runs broadcast on a fake 2-rank group as rank 0 with src equal to the
        # rank and with src not equal to the rank; asserts the tensor shape is unchanged in
        # both cases.
        dist.init_process_group(backend="fake", rank=0, world_size=2)

        # src == rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=0)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        # [FUNCTION DOC] Runs scatter as rank 0, once as the source with a list of two tensors
        # and once as a non-source with no list and src 1; asserts the output shape is unchanged
        # each time.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # src == rank
        output = torch.ones(3, 3)
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        dist.scatter(output, to_scatter)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.scatter(output, None, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        # [FUNCTION DOC] Runs all_to_all on a fake 2-rank group with two 3x3 inputs and outputs;
        # asserts the output list length is 2 and each shape is 3x3.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        # [FUNCTION DOC] Runs all_to_all_single with explicit equal split sizes on 4x3 tensors;
        # asserts the output shape is unchanged.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(4, 3)
        in_tensor = torch.ones(4, 3)
        output_split = [2, 2]
        input_split = [2, 2]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (4, 3))

    def test_send(self):
        # [FUNCTION DOC] Sends a 3x3 tensor to rank 1 on a fake group as rank 0; asserts only
        # that the call returns and the tensor shape is intact.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        tensor = torch.ones(3, 3)
        dist.send(tensor, 1)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    def test_recv(self):
        # [FUNCTION DOC] Receives into a 3x3 tensor from rank 1 on a fake group as rank 0;
        # asserts only that the call returns and the shape is intact.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.ones(3, 3)
        dist.recv(output, 1)
        self.assertEqual(tuple(output.shape), (3, 3))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_tp_fake_e2e(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. On a fake world of 4 ranks
        # builds a 2x2 device mesh (dp, tp), applies tensor parallel plans (sequence-parallel
        # layouts and plain colwise/rowwise) to an MLP module, wraps it in FSDP over the dp
        # mesh, and runs 10 forward/backward/Adam steps per plan with seeded random inputs;
        # passes if nothing raises.
        world_size = 4
        tp_size = 2

        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        device_mesh = init_device_mesh(
            device_type, (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
        )

        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        pairwise_parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        for parallel_plan in [sequence_parallelize_plan, pairwise_parallelize_plan]:
            my_module = parallelize_module(
                MLPModule(device=device_type),
                device_mesh["tp"],
                parallel_plan,
            )

            sharded_module = FSDP(
                my_module, use_orig_params=True, device_mesh=device_mesh["dp"]
            )
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

            for i in range(10):
                dp_rank = dist.get_rank()
                torch.manual_seed(i + dp_rank)
                input = torch.randn(20, 10, device=f"{device_type}:{dp_rank}")
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs reduce_scatter with inputs equal
        # to the rank index times ones and asserts the output equals the input slot at this
        # rank, i.e. the fake group copies the local slot rather than reducing.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * r for r in range(2)]
        output = torch.empty(3, 3)
        dist.reduce_scatter(output, to_reduce_scatter)
        self.assertEqual(output, to_reduce_scatter[rank])

    def test_reduce_scatter_requires_grad(self):
        # [FUNCTION DOC] Runs reduce_scatter as rank 1 with inputs that require grad; asserts
        # the output equals the input at index 1 and that the output does not require grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output = torch.empty(3, 3)
        dist.reduce_scatter(output, inputs)
        self.assertEqual(output, inputs[1])
        self.assertFalse(output.requires_grad)

    @parametrize("rank", [0, 1])
    def test_scatter_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Rank 0 scatters a list and asserts it
        # receives the first element; rank 1 scatters with no list and asserts its output tensor
        # is left unchanged.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        if rank == 0:
            to_scatter = [torch.ones(3, 3) * r for r in range(2)]
            output = torch.empty(3, 3)
            dist.scatter(output, to_scatter)
            self.assertEqual(output, to_scatter[0])
        else:
            output = torch.ones(3, 3) * 5
            dist.scatter(output, None, src=0)
            self.assertEqual(output, torch.ones(3, 3) * 5)

    def test_scatter_requires_grad(self):
        # [FUNCTION DOC] Scatters from rank 0 a list of inputs that require grad; asserts the
        # output equals the first input and does not require grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output = torch.empty(3, 3)
        dist.scatter(output, inputs)
        self.assertEqual(output, inputs[0])
        self.assertFalse(output.requires_grad)

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_base_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Calls the base reduce-scatter on a
        # 6x2 input into a 3x2 output and asserts the output equals the chunk of the input at
        # this rank.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_buf = torch.arange(12.0).reshape(6, 2)
        out_buf = torch.empty(3, 2)
        dist._reduce_scatter_base(out_buf, in_buf)
        self.assertEqual(out_buf, in_buf.chunk(2)[rank])

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_tensor_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Calls reduce_scatter_tensor on a 4x2
        # input into a 2x2 output and asserts the output equals the chunk of the input at
        # this rank.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(2, 2)
        dist.reduce_scatter_tensor(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[rank])

    def test_reduce_scatter_base_requires_grad(self):
        # [FUNCTION DOC] As rank 1, calls the base reduce-scatter with an input that requires
        # grad and asserts the output equals the input chunk at index 1.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(2, 1)
        dist._reduce_scatter_base(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[1])

    def test_reduce_scatter_tensor_coalesced_requires_grad(self):
        # [FUNCTION DOC] As rank 1, calls reduce_scatter_tensor inside a coalescing manager with
        # an input that requires grad; asserts the output equals the input chunk at index 1.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(2, 1)
        with dist._coalescing_manager():
            dist.reduce_scatter_tensor(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[1])

    @parametrize("rank", [0, 1])
    def test_allgather_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_gather of a constant tensor
        # valued 42 and asserts every gathered output equals the input.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_tensor = torch.ones(3, 3) * 42
        output_tensors = [torch.empty(3, 3) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out in output_tensors:
            self.assertEqual(out, input_tensor)

    def test_allgather_requires_grad(self):
        # [FUNCTION DOC] As rank 1, all_gathers an input that requires grad; asserts every
        # output equals the input and none requires grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        input_tensor = torch.ones(3, 3).requires_grad_(True)
        output_tensors = [torch.empty(3, 3) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out in output_tensors:
            self.assertEqual(out, input_tensor)
            self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_gather_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Rank 0 gathers into a list and asserts
        # each entry equals the input; rank 1 gathers with no list and destination 0, which must
        # simply not raise.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_tensor = torch.ones(3, 3) * 42
        if rank == 0:
            gather_list = [torch.empty(3, 3) for _ in range(2)]
            dist.gather(input_tensor, gather_list)
            for out in gather_list:
                self.assertEqual(out, input_tensor)
        else:
            dist.gather(input_tensor, None, dst=0)

    def test_gather_requires_grad(self):
        # [FUNCTION DOC] As rank 0, gathers an input that requires grad into a list; asserts
        # each output equals the input and does not require grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        input_tensor = torch.ones(3, 3).requires_grad_(True)
        gather_list = [torch.empty(3, 3) for _ in range(2)]
        dist.gather(input_tensor, gather_list)
        for out in gather_list:
            self.assertEqual(out, input_tensor)
            self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_allgather_coalesced_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_gather_coalesced of three
        # tensors into two output lists and asserts each output entry equals the
        # corresponding input.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        inputs = [torch.ones(3, 3) * i for i in range(3)]
        output_lists = [[torch.empty(3, 3) for _ in inputs] for _ in range(2)]
        dist.all_gather_coalesced(output_lists, inputs)
        for output_list in output_lists:
            for i, out in enumerate(output_list):
                self.assertEqual(out, inputs[i])

    def test_allgather_coalesced_requires_grad(self):
        # [FUNCTION DOC] As rank 1, runs all_gather_coalesced on inputs that require grad;
        # asserts each output equals its input and does not require grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output_lists = [[torch.empty(3, 3) for _ in inputs] for _ in range(2)]
        dist.all_gather_coalesced(output_lists, inputs)
        for output_list in output_lists:
            for i, out in enumerate(output_list):
                self.assertEqual(out, inputs[i])
                self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_alltoall_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_to_all with two tensors and
        # asserts each output equals the input at the same index.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_list = [torch.ones(3, 3) * i for i in range(2)]
        output_list = [torch.empty(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        for i in range(2):
            self.assertEqual(output_list[i], input_list[i])

    def test_alltoall_requires_grad(self):
        # [FUNCTION DOC] As rank 1, runs all_to_all on inputs that require grad and asserts each
        # output equals the input at the same index.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        input_list = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output_list = [torch.empty(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        for i in range(2):
            self.assertEqual(output_list[i], input_list[i])

    @parametrize("rank", [0, 1])
    def test_alltoall_base_copy_semantics(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_to_all_single with default
        # splits on a 4x2 tensor and asserts the output equals the input.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(4, 2)
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor)

    @parametrize("rank", [0, 1])
    def test_alltoall_base_split_sizes_repeat(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_to_all_single with output
        # splits 2 and 6 and input splits 1 and 3; asserts the output has the input data
        # repeated cyclically to fill the larger slot, matching a hard-coded expected tensor.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        # Slot 1 (size 6) is larger than the input buffer (size 4),
        # so the input is repeated to fill it.
        in_tensor = torch.arange(4.0).reshape(4, 1)
        out_tensor = torch.empty(8, 1)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[2, 6],
            input_split_sizes=[1, 3],
        )
        expected = torch.tensor(
            [[0.0], [1.0], [0.0], [1.0], [2.0], [3.0], [0.0], [1.0]]
        )
        self.assertEqual(out_tensor, expected)

    @parametrize("rank", [0, 1])
    def test_alltoall_base_split_sizes_truncate(self, rank):
        # [FUNCTION DOC] Parametrized over rank 0 and 1. Runs all_to_all_single with output
        # splits 1 and 3 and input splits 4 and 4; asserts each output slot holds the input
        # truncated to the slot size, matching a hard-coded expected tensor.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        # Each output slot is smaller than the input buffer, so input is
        # truncated to slot size.
        in_tensor = torch.arange(8.0).reshape(8, 1)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[1, 3],
            input_split_sizes=[4, 4],
        )
        expected = torch.tensor([[0.0], [0.0], [1.0], [2.0]])
        self.assertEqual(out_tensor, expected)

    def test_alltoall_base_split_size_validation(self):
        # [FUNCTION DOC] Checks that all_to_all_single raises RuntimeError for three bad cases:
        # a first dimension not divisible by group size, a split list whose length differs from
        # group size, and split sizes that do not sum to the dim 0 size. Each is matched by
        # error message.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        with self.assertRaisesRegex(
            RuntimeError, "does not divide equally across group size"
        ):
            dist.all_to_all_single(torch.empty(3, 2), torch.ones(3, 2))

        with self.assertRaisesRegex(
            RuntimeError, "Number of tensor splits not equal to group size"
        ):
            dist.all_to_all_single(
                torch.empty(4, 2),
                torch.ones(4, 2),
                output_split_sizes=[4],
                input_split_sizes=[4],
            )

        with self.assertRaisesRegex(
            RuntimeError, "Split sizes doesn't match total dim 0 size"
        ):
            dist.all_to_all_single(
                torch.empty(4, 2),
                torch.ones(4, 2),
                output_split_sizes=[1, 1],
                input_split_sizes=[2, 2],
            )

    def test_alltoall_base_empty_input_with_output(self):
        # [FUNCTION DOC] Checks that all_to_all_single with an empty input but a non-empty
        # output split raises a RuntimeError mentioning an empty input buffer.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        with self.assertRaisesRegex(RuntimeError, "inputBuffer is empty"):
            dist.all_to_all_single(
                torch.empty(2, 1),
                torch.empty(0, 1),
                output_split_sizes=[1, 1],
                input_split_sizes=[0, 0],
            )

    def test_alltoall_list_size_validation(self):
        # [FUNCTION DOC] Checks that all_to_all with one output tensor and two input tensors
        # raises a RuntimeError saying the list size does not match the world size.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        with self.assertRaisesRegex(RuntimeError, "does not match world size"):
            dist.all_to_all(
                [torch.empty(3, 3)],
                [torch.ones(3, 3), torch.ones(3, 3)],
            )

    def test_alltoall_base_requires_grad(self):
        # [FUNCTION DOC] Regression test: all_to_all_single with an input requiring grad must
        # not fail, in both the default-split and the uneven split case, because the fake
        # backend writes outputs below autograd. Only checks that no exception is raised.
        # Real backends write into output from C++ kernels autograd never sees.
        # Without AutoDispatchBelowAutograd, copy_ into a narrow()/chunk() view
        # would fail when input requires grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(out_tensor, in_tensor)

        in_tensor2 = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor2 = torch.empty(8, 1)
        dist.all_to_all_single(
            out_tensor2,
            in_tensor2,
            output_split_sizes=[2, 6],
            input_split_sizes=[1, 3],
        )

    def test_error_on_collective(self):
        # [FUNCTION DOC] First runs all_reduce on a default fake group without error. Then
        # destroys the group, re-initializes it with FakeProcessGroup options having
        # error_on_collective set, and asserts that all_reduce, all_gather, broadcast and
        # barrier each raise a RuntimeError containing the fake process group collective error
        # text. Re-imports FakeStore and FakeProcessGroup locally.
        from torch.testing._internal.distributed.fake_pg import FakeStore

        # Test with error_on_collective=False (default behavior)
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # These should work normally
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor)
        self.assertEqual(tuple(tensor.shape), (3, 3))

        dist.destroy_process_group()

        # Test with error_on_collective=True
        from torch._C._distributed_c10d import FakeProcessGroup

        options = FakeProcessGroup.Options()
        options.error_on_collective = True

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=2, store=store, pg_options=options
        )

        # These should now raise errors
        tensor = torch.ones(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.all_reduce(tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            output_tensors = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(output_tensors, tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.broadcast(tensor, src=0)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.barrier()

    def test_fake_process_group_direct_usage_error(self):
        # [FUNCTION DOC] Checks that constructing FakeProcessGroup directly with rank and world
        # size raises a TypeError saying no constructor is defined; the dispatch-mode block
        # inside the assertion is never reached.
        class SimpleTensorMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Minimal TorchDispatchMode that forwards every call unchanged.
            # Used only inside the enclosing test.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that defaults kwargs to an empty
                # mapping and calls the intercepted function with the same arguments,
                # returning its result.
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with self.assertRaisesRegex(TypeError, r"No constructor defined"):
            fake_pg = FakeProcessGroup(rank=0, world_size=3)

            with SimpleTensorMode():
                tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                dist.all_reduce(tensor, group=fake_pg)

    def test_fake_process_group_proper_usage_dispatch(self):
        # [FUNCTION DOC] Initializes a fake 3-rank group, then runs a tensor creation and
        # all_reduce under a recording dispatch mode; asserts the recorded op names include the
        # lift_fresh op and the c10d allreduce_ op, showing the fake collective dispatches
        # through the c10d op.
        class SimpleTensorMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: TorchDispatchMode that appends the string form of each
            # dispatched function to its ops list and then runs it unchanged.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the recorded op list as empty. Does not
                # call the base class initializer.
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Records the string form of the function in the ops
                # list, defaults kwargs to an empty mapping, and returns the result of calling
                # the function.
                self.ops.append(str(func))
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=3)

        with SimpleTensorMode() as mode:
            tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            dist.all_reduce(tensor)

        op_names = [str(op) for op in mode.ops]
        self.assertIn("aten.lift_fresh.default", op_names)
        self.assertIn("c10d.allreduce_.default", op_names)

    def test_reduce_scatter_wrong_input_list_size(self):
        # [FUNCTION DOC] Checks that reduce_scatter with a one-element input list on a 2-rank
        # fake group raises a RuntimeError saying the list size must equal the world size.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        output = torch.empty(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "invalid input tensor list size, must be world size"
        ):
            dist.reduce_scatter(output, [torch.ones(3, 3)])

    def test_scatter_wrong_input_list_size(self):
        # [FUNCTION DOC] Checks that scatter as rank 0 with a one-element input list on a 2-rank
        # fake group raises a RuntimeError about an incorrect input list size.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.empty(3, 3)
        with self.assertRaisesRegex(RuntimeError, "Incorrect input list size"):
            dist.scatter(output, [torch.ones(3, 3)])

    def test_scatter_non_root_rejects_input_list(self):
        # [FUNCTION DOC] As rank 1 with root 0, calls the process group scatter directly with
        # a non-empty input list and asserts a RuntimeError saying non-root ranks require
        # empty input.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.ScatterOptions()
        opts.rootRank = 0
        output = torch.empty(3, 3)
        inputs = [[torch.ones(3, 3), torch.ones(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "requires empty input on non-root"):
            pg.scatter([output], inputs, opts)

    def test_gather_non_root_rejects_output_list(self):
        # [FUNCTION DOC] As rank 1 with root 0, calls the process group gather directly with
        # a non-empty output list and asserts a RuntimeError saying non-root ranks require
        # empty output.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.GatherOptions()
        opts.rootRank = 0
        output = [[torch.empty(3, 3), torch.empty(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "requires empty output on non-root"):
            pg.gather(output, [torch.ones(3, 3)], opts)

    def test_gather_invalid_root_rank(self):
        # [FUNCTION DOC] Calls the process group gather directly with root rank 3 on a 2-rank
        # group and asserts a RuntimeError mentioning an invalid root rank.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.GatherOptions()
        opts.rootRank = 3
        output = [[torch.empty(3, 3), torch.empty(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            pg.gather(output, [torch.ones(3, 3)], opts)

    def test_reduce_scatter_wrong_output_list_size(self):
        # [FUNCTION DOC] Calls the process group reduce_scatter directly with two outputs but
        # one input list and asserts a RuntimeError saying input and output lists must have the
        # same length.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.ReduceScatterOptions()
        outputs = [torch.empty(3, 3), torch.empty(3, 3)]
        inputs = [[torch.ones(3, 3), torch.ones(3, 3)]]
        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.reduce_scatter(outputs, inputs, opts)

    def test_reduce_scatter_base_wrong_input_size(self):
        # [FUNCTION DOC] Calls the base reduce-scatter with an input the same size as the output
        # (not world size times larger) and asserts a RuntimeError about the input size.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_buf = torch.empty(3, 2)
        in_buf = torch.ones(3, 2)
        with self.assertRaisesRegex(
            RuntimeError,
            "input tensor must be the same size as output size times world size",
        ):
            dist._reduce_scatter_base(out_buf, in_buf)

    def test_allgather_coalesced_wrong_inner_list_size(self):
        # [FUNCTION DOC] Calls all_gather_coalesced with an output list containing an inner list
        # of the wrong length (one list has one entry, the other is empty) and asserts a
        # RuntimeError mentioning an invalid output size.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = [[torch.empty(3, 3)], []]
        inputs = [torch.ones(3, 3)]
        with self.assertRaisesRegex(RuntimeError, "invalid output size"):
            dist.all_gather_coalesced(output, inputs)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    @parametrize("rank", [0, 1, 2, 3])
    def test_split_group(self, rank):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Parametrized over ranks 0 to
        # 3 of a fake world of 4 with device_id set. Asserts the parent backend supports
        # splitting, calls split_group with unsorted interleaved groups [2,0] and [3,1], and
        # checks the new group size, that its rank equals the position of this rank in its
        # listed subgroup, and that the global rank mapping maps back to the original rank.
        # Finally runs all_reduce on the split group.
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=rank,
            world_size=world_size,
            store=store,
            device_id=torch.device(device_type, 0),
        )

        parent_pg = dist.distributed_c10d._get_default_group()
        self.assertTrue(
            parent_pg._get_backend(torch.device(device_type)).supports_splitting
        )

        # Interleaved, unsorted subgroups: each rank shares a group with the
        # rank two away, and is not always listed first. split_group preserves
        # the order ranks are listed in, so the child rank is the position
        # within the subgroup as given -- which here differs from the parent
        # rank, making this a meaningful check of rank assignment.
        split_ranks = [[2, 0], [3, 1]]
        new_pg = dist.split_group(split_ranks=split_ranks)
        self.assertIsNotNone(new_pg)

        my_group = next(g for g in split_ranks if rank in g)
        self.assertEqual(new_pg.size(), len(my_group))
        self.assertEqual(new_pg.rank(), my_group.index(rank))
        # Independent cross-check: the child rank must map back to this
        # process's original global rank via the world's rank mapping, which is
        # built separately from the C++ backend's rank.
        self.assertEqual(
            dist.distributed_c10d.get_global_rank(new_pg, new_pg.rank()), rank
        )

        # Collectives on the split group should still work.
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor, group=new_pg)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_backend_filter(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Calls split_group with
        # several backend filters on a fake world: bare fake keeps all parent device types,
        # cpu:fake keeps only cpu, a cpu plus accelerator filter keeps both, and mps:fake,
        # cpu:gloo and gloo are rejected with ValueError (not present in the parent, or backend
        # mismatch). Regression test for filter validation of a bare fake backend string.
        # A fake world's backend string is the bare name "fake", which
        # BackendConfig expands to every device fake supports. split_group's
        # filter validation used to re-expand it through
        # Backend.default_device_backend_map -- where fake is only the default
        # for hpu -- so every device-qualified filter was rejected as "not
        # present in the parent" and the bare "fake" filter selected hpu alone.
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=2,
            store=store,
            device_id=torch.device(device_type, 0),
        )
        parent_pg = dist.distributed_c10d._get_default_group()
        parent_devices = {d.type for d in parent_pg._device_types}
        self.assertIn(device_type, parent_devices)

        # A bare filter naming the parent's backend keeps every parent device.
        full = dist.split_group(split_ranks=[[0, 1]], backend="fake")
        self.assertEqual({d.type for d in full._device_types}, parent_devices)

        # A device-qualified filter keeps exactly the named devices. The C++
        # split additionally requires the filter to keep the parent's default
        # backend device, which for an all-fake parent is cpu.
        cpu_only = dist.split_group(split_ranks=[[0, 1]], backend="cpu:fake")
        self.assertEqual({d.type for d in cpu_only._device_types}, {"cpu"})

        pair = dist.split_group(
            split_ranks=[[0, 1]], backend=f"cpu:fake,{device_type}:fake"
        )
        self.assertEqual({d.type for d in pair._device_types}, {"cpu", device_type})

        # Filters that genuinely do not match the parent are still rejected.
        with self.assertRaisesRegex(ValueError, "is not present in the parent"):
            dist.split_group(split_ranks=[[0, 1]], backend="mps:fake")
        with self.assertRaisesRegex(ValueError, "Backend mismatch"):
            dist.split_group(split_ranks=[[0, 1]], backend="cpu:gloo")
        with self.assertRaisesRegex(ValueError, "is not present in the parent"):
            dist.split_group(split_ranks=[[0, 1]], backend="gloo")

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_non_member(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. As rank 0 of 4, splits into
        # the single group [1,2,3] and asserts the result is the NON_GROUP_MEMBER sentinel
        # because this rank is in no group.
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=4,
            store=store,
            device_id=torch.device(device_type, 0),
        )

        # Rank 0 is in none of the splits, so it gets the non-member sentinel.
        new_pg = dist.split_group(split_ranks=[[1, 2, 3]])
        self.assertIs(new_pg, dist.GroupMember.NON_GROUP_MEMBER)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_store_not_retained(self):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Regression test that
        # initializes the fake group in a helper so the Python reference to the FakeStore is
        # dropped, then calls split_group; asserts the split group has size 2, which only works
        # if the store clone is a real C++ method.
        # split_group clones the parent's store, and the process group holds the
        # store only at the C++ level. FakeStore.clone() must therefore be a
        # real C++ method: a pure-Python Store.clone() override would be garbage
        # collected once the caller drops its reference, and the split would
        # fail with "pure virtual function Store::clone".
        def setup():
            # [FUNCTION DOC] HELPER: Initializes a 2-rank fake default group as rank 0 with a
            # fresh FakeStore created inline and device_id set, so no Python reference to the
            # store survives after return.
            dist.init_process_group(
                backend="fake",
                rank=0,
                world_size=2,
                store=FakeStore(),
                device_id=torch.device(device_type, 0),
            )

        setup()  # the only Python reference to the store is dropped here
        new_pg = dist.split_group(split_ranks=[[0, 1]])
        self.assertIsNotNone(new_pg)
        self.assertEqual(new_pg.size(), 2)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    @parametrize("rank", [0, 1, 2, 3])
    def test_split_group_consistent_naming_after_partial_split(self, rank):
        # [FUNCTION DOC] Skipped on HPU or without an accelerator. Regression test for issue
        # 190396, parametrized over ranks 0 to 3 of a fake world of 4. Records the world group
        # counter, does a partial split [0,1,2] and asserts the counter advances by one on every
        # rank (non-member rank 3 gets a non-process-group result), then does a full split
        # [[0,2],[1,3]] and asserts the counter advances again. Finally asserts the new group
        # name equals the hex sha1 digest of the joined member ranks and the counter value
        # before the second split.
        # Regression test for https://github.com/pytorch/pytorch/issues/190396.
        #
        # _hash_ranks_to_str previously used len(_world.pg_names) as the
        # uniqueness suffix. Non-member ranks of a partial split don't register
        # the PG, so their counter stayed lower than member ranks. Subsequent
        # splits then computed different names for the same communicator on
        # different ranks, causing inconsistent teardown ordering and (for NCCL)
        # circular ncclCommFinalize waits that deadlock destroy_process_group.
        #
        # The fix uses _world.group_count as the salt. _process_group_name now
        # increments group_count on BOTH paths so it advances on every rank that
        # reaches it, including non-members, keeping it collective-consistent.
        # This test verifies group_count advances consistently even for non-member
        # ranks and that PG names are computed from it correctly.
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=rank,
            world_size=world_size,
            store=store,
            device_id=torch.device(device_type, 0),
        )
        import hashlib as _hashlib

        from torch.distributed import distributed_c10d

        # group_count starts at 1 (the default PG consumed count 0).
        count_after_init = distributed_c10d._world.group_count

        # Partial split: ranks 0,1,2 are members; rank 3 is not.
        partial = dist.split_group(split_ranks=[[0, 1, 2]])

        # group_count must advance on ALL ranks, including non-member rank 3,
        # because _process_group_name is called before the member check.
        self.assertEqual(distributed_c10d._world.group_count, count_after_init + 1)
        if rank == 3:
            self.assertNotIsInstance(partial, dist.ProcessGroup)
        else:
            self.assertIsInstance(partial, dist.ProcessGroup)

        # Full split: all ranks are members.
        full = dist.split_group(split_ranks=[[0, 2], [1, 3]])
        self.assertEqual(distributed_c10d._world.group_count, count_after_init + 2)
        self.assertIsInstance(full, dist.ProcessGroup)

        # PG name must use count_after_init+1 as the group_count salt
        # (the value at the time of the second split_group call, before it
        # was incremented). Co-participants of [0,2] and [1,3] each compute
        # the same name because group_count is consistent across all ranks.
        pg_name = distributed_c10d._world.pg_names[full]
        my_group = [0, 2] if rank in [0, 2] else [1, 3]
        rank_join = "_".join(map(str, my_group))
        expected = _hashlib.sha1(
            f"{rank_join}_{count_after_init + 1}".encode(), usedforsecurity=False
        ).hexdigest()
        self.assertEqual(pg_name, expected)


instantiate_parametrized_tests(TestFakePG)

if __name__ == "__main__":
    run_tests()
