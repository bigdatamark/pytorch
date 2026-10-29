# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the c10d integration with TorchComms. Multi-process device-parametrized
# collective tests (cpu, cuda, xpu) check results of every collective and new_group variants;
# cuda-only classes check init auto-qualification of backends and subgroups on a mixed-backend
# parent; single-process unit tests with heavy patching check backend routing, device and
# environment handling in the process group helper, group-name hashing, finalize deduplication on
# destroy, and default backend type selection. Import-time side effects: instantiates the
# device-specific test classes into the module namespace. Running as a script calls run_tests.

import datetime
import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import C10dTorchCommsTestBase
from torch.testing._internal.common_utils import (
    find_free_port,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsBasic(C10dTorchCommsTestBase):
    # [CLASS DOC] Continuous multi-process collective tests over the default process group built
    # by the TorchComms test base (skipped when TorchComms is not installed). Every rank
    # contributes rank plus one as a float tensor and results are compared with values computed
    # from the world size. Instantiated per device type for cpu, cuda and xpu. Class attribute
    # REDUCE_OPS lists the parametrized reduce operations.
    REDUCE_OPS = [
        subtest(dist.ReduceOp.SUM, name="SUM"),
        subtest(dist.ReduceOp.AVG, name="AVG"),
        subtest(dist.ReduceOp.MIN, name="MIN"),
        subtest(dist.ReduceOp.MAX, name="MAX"),
        subtest(dist.ReduceOp.PRODUCT, name="PRODUCT"),
    ]

    @property
    def _rank_value(self):
        # [FUNCTION DOC] HELPER: Property returning this rank's contribution, rank plus one.
        return self.rank + 1

    def _requires_cuda(self):
        # [FUNCTION DOC] HELPER: Returns true when the device type of the variant is not cuda;
        # tests call it and return early because skipping in these workers would poison the whole
        # class.
        """Return True when the test variant is NOT cuda.

        MultiProcContinuousTest workers wrap unittest.SkipTest as RuntimeError,
        so @onlyCUDA / self.skipTest() poison the entire class.  Tests that
        need NCCL should call this and ``return`` early instead.
        """
        return self.device_type != "cuda"

    def _skip_if_product_overflows(self, op):
        # [FUNCTION DOC] HELPER: Skips the test when the product reduction would overflow float32
        # exactness, which happens for world sizes above 12.
        if op == dist.ReduceOp.PRODUCT and self.world_size > 12:
            self.skipTest(
                f"world_size={self.world_size} > 12: PRODUCT is world_size! "
                "and only up to 12! is exactly representable in float32"
            )

    def _expected_reduce_result(self, op):
        # [FUNCTION DOC] HELPER: Returns the expected scalar for reducing rank plus one across all
        # ranks: the sum, the average, one for min, world size for max, or the factorial of world
        # size for product. Raises ValueError for an unsupported operation.
        """Return the expected scalar result for a rank+1 input reduced across all ranks."""
        total = sum(range(1, self.world_size + 1))
        if op == dist.ReduceOp.SUM:
            return total
        elif op == dist.ReduceOp.AVG:
            return total / self.world_size
        elif op == dist.ReduceOp.MIN:
            return 1
        elif op == dist.ReduceOp.MAX:
            return self.world_size
        elif op == dist.ReduceOp.PRODUCT:
            result = 1
            for i in range(1, self.world_size + 1):
                result *= i
            return result
        raise ValueError(f"Unsupported op: {op}")

    @parametrize("op", REDUCE_OPS)
    def test_allreduce(self, op):
        # [FUNCTION DOC] All-reduces a one-element tensor with each reduce op and checks the
        # result against the expected value.
        self._skip_if_product_overflows(op)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, op=op, group=self.pg)
        self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        # [FUNCTION DOC] All-gathers one-element tensors into a list and checks it holds one
        # through world size in order.
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in gather_list], expected)

    def test_all_gather_into_tensor(self):
        # [FUNCTION DOC] Gathers each rank's value into a single preallocated tensor with the
        # all-gather-single call and checks the values are one through world size.
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        output_tensor = torch.empty(self.world_size, dtype=torch.float32)
        dist.all_gather_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_broadcast(self):
        # [FUNCTION DOC] Broadcasts from rank 0 and checks every rank ends with rank 0's value,
        # one.
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=self.pg)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        # [FUNCTION DOC] Gathers values to rank 0 (only rank 0 supplies the output list) and
        # checks rank 0 sees one through world size.
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = None
        if self.rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, gather_list=gather_list, dst=0, group=self.pg)
        if self.rank == 0:
            expected = list(range(1, self.world_size + 1))
            self.assertEqual([t.item() for t in gather_list], expected)

    def test_scatter(self):
        # [FUNCTION DOC] Scatters from rank 0 a list holding the value i for destination i, and
        # checks each rank receives its own rank number.
        if self.rank == 0:
            scatter_list = [
                torch.tensor([i], dtype=torch.float32) for i in range(self.world_size)
            ]
        else:
            scatter_list = None
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0, group=self.pg)
        self.assertEqual(tensor.item(), self.rank)

    @parametrize("op", REDUCE_OPS)
    def test_reduce(self, op):
        # [FUNCTION DOC] Reduces to rank 0 with each op and checks the expected value on rank 0
        # only.
        self._skip_if_product_overflows(op)
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.reduce(input_tensor, dst=0, op=op, group=self.pg)
        if self.rank == 0:
            self.assertEqual(input_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter(self, op):
        # [FUNCTION DOC] Reduce-scatters a list of identical one-element inputs with each op and
        # checks each rank's output equals the expected reduction.
        self._skip_if_product_overflows(op)
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter_tensor(self, op):
        # [FUNCTION DOC] Reduce-scatters a single tensor filled with the rank value with each op
        # and checks the output equals the expected reduction.
        self._skip_if_product_overflows(op)
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter_single(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    def test_all_to_all(self):
        # [FUNCTION DOC] All-to-all with a list of per-peer one-element tensors, and checks each
        # rank receives one through world size (the value of each sender).
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.world_size)
        ]
        dist.all_to_all(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single(self):
        # [FUNCTION DOC] All-to-all-single with equal splits on a tensor filled with the rank
        # value and checks the output holds one through world size.
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty([self.world_size], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single_with_split_sizes(self):
        # [FUNCTION DOC] All-to-all-single with unequal splits: each rank sends rank plus one
        # elements to every peer, filled with rank plus destination, and receives sender plus one
        # elements from each sender. Checks each received section equals sender plus own rank.
        # Each rank sends (rank + 1) elements to every other rank,
        # so rank r's input_split_sizes are all (rank + 1).
        input_split_sizes = [self._rank_value] * self.world_size
        # Rank r receives (sender_rank + 1) elements from each sender,
        # so output_split_sizes[i] = i + 1.
        output_split_sizes = [i + 1 for i in range(self.world_size)]

        input_tensor = torch.empty(sum(input_split_sizes), dtype=torch.float32)
        offset = 0
        for dst in range(self.world_size):
            input_tensor[offset : offset + input_split_sizes[dst]].fill_(
                self.rank + dst
            )
            offset += input_split_sizes[dst]

        output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.pg,
        )

        # Verify: section from sender i should contain value (i + rank)
        offset = 0
        for src in range(self.world_size):
            section = output_tensor[offset : offset + output_split_sizes[src]]
            expected = torch.full_like(section, src + self.rank)
            self.assertTrue(
                torch.equal(section, expected),
                lambda msg: f"{msg}\nMismatch in section from rank {src}: got {section}, expected {expected}",
            )
            offset += output_split_sizes[src]

    def test_send_recv(self):
        # [FUNCTION DOC] Ring send and receive: each rank sends its rank to the next rank and
        # receives from the previous one, with even ranks sending first and odd ranks receiving
        # first to avoid deadlock. Checks the received value is the previous rank.
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank + self.world_size - 1) % self.world_size
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            dist.send(send_tensor, dst=send_rank, group=self.pg)
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
        else:
            # Odd ranks: receive first, then send
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
            dist.send(send_tensor, dst=send_rank, group=self.pg)
        # Each rank receives the rank number of the sender
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        # [FUNCTION DOC] Runs a barrier on the group; passes if it completes.
        dist.barrier(group=self.pg)
        # If we reach this point, the barrier succeeded without deadlock
        self.assertTrue(True)

    def test_monitored_barrier(self):
        # [FUNCTION DOC] Cpu variant only (returns early otherwise): checks the default group
        # backend is gloo and that the monitored barrier with a 30 second timeout and wait for all
        # ranks completes.
        # monitored_barrier is gloo-only; the default PG is gloo only on the
        # cpu variant (nccl on cuda/xpu), so run there. This drives the full
        # dist.monitored_barrier dispatch onto BackendWrapper::monitoredBarrier
        # (the torchcomms reimplementation of ProcessGroupGloo::monitoredBarrier).
        if self.device_type != "cpu":
            return
        self.assertEqual(dist.get_backend(self.pg), dist.Backend.GLOO)
        # All ranks check in -> the health-checking barrier returns cleanly.
        dist.monitored_barrier(
            group=self.pg,
            timeout=datetime.timedelta(seconds=30),
            wait_all_ranks=True,
        )

    def test_new_group_delegates_to_split_group(self):
        # [FUNCTION DOC] Creates a subgroup of the first half of the ranks (the new_group call is
        # routed to split_group under TorchComms). Members check the group ranks and an all-reduce
        # of the sum of their values; non-members check they received the non-member sentinel.
        # Under torchcomms, `new_group` routes through `split_group`. The
        # resulting subgroup must contain the requested ranks and be usable
        # for collectives.
        subg_ranks = list(range(self.world_size // 2))
        ng = dist.new_group(ranks=subg_ranks)

        if self.rank in subg_ranks:
            self.assertEqual(dist.get_process_group_ranks(ng), subg_ranks)
            tensor = torch.tensor([self._rank_value], dtype=torch.float32)
            dist.all_reduce(tensor, group=ng)
            self.assertEqual(tensor.item(), sum(r + 1 for r in subg_ranks))
        else:
            self.assertIs(ng, dist.GroupMember.NON_GROUP_MEMBER)

    def test_new_group_backend_none_narrows_to_default_device(self):
        # [FUNCTION DOC] Creates a group over all ranks with a backend of none (inherits the
        # default device) and checks an all-reduce gives the sum of all values.
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend=None)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_bare_default_backend_is_auto_qualified(self):
        # [FUNCTION DOC] Cuda variant only: creates a group with the bare backend name nccl and
        # checks an all-reduce sum works.
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend="nccl")
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_qualified_backend_passes_through(self):
        # [FUNCTION DOC] Cuda variant only: creates a group with the device-qualified backend name
        # for nccl and checks an all-reduce sum works.
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks, backend="cuda:nccl")
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_with_pg_options(self):
        # [FUNCTION DOC] Cuda variant only: creates a group with NCCL options (high priority
        # stream, cluster size 2, max CTAs 16) and checks an all-reduce sum works.
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 16
        ng = dist.new_group(ranks=ranks, pg_options=opts)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_sequential_pg_options_produce_distinct_groups(self):
        # [FUNCTION DOC] Cuda variant only: creates two groups over the same ranks with different
        # NCCL options and checks they get different group names.
        if self._requires_cuda():
            return
        ranks = list(range(self.world_size))
        opts_a = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts_a.config.cga_cluster_size = 2
        opts_b = dist.ProcessGroupNCCL.Options()
        opts_b.config.cga_cluster_size = 4
        g_a = dist.new_group(ranks=ranks, pg_options=opts_a)
        g_b = dist.new_group(ranks=ranks, pg_options=opts_b)
        self.assertNotEqual(g_a.group_name, g_b.group_name)


devices = ["cpu", "cuda", "xpu"]
instantiate_device_type_tests(
    TestC10dTorchCommsBasic, globals(), only_for=devices, allow_xpu=True
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsInitAutoQualify(C10dTorchCommsTestBase):
    # [CLASS DOC] Cuda-only multi-process tests where the default group is initialized with the
    # bare backend name nccl and a device id, checking that init auto-qualifies it to cover both
    # cpu and cuda. The initialization override sets environment variables itself and leaves the
    # default group in a class attribute.
    """Verify init_process_group auto-qualifies bare backends under torchcomms.

    Overrides ``_init_pg`` to pass ``device_id`` with bare ``"nccl"`` —
    the auto-qualify logic in ``init_process_group`` should expand it to
    ``"cpu:gloo,cuda:nccl"`` so both CPU and CUDA backends are available.
    """

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Worker-side initialization: enables the torchcomms config flag,
        # sets the master address, a free master port, rank, world size, store path and local rank
        # environment variables, binds the cuda device for the rank, and initializes the default
        # group with backend nccl, a file store and the device id. Stores the group in a class
        # attribute and sets the default device to the rank's cuda device.
        torch.distributed.config.use_torchcomms = True
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = rdvz_file
        os.environ["LOCAL_RANK"] = str(rank)

        store = dist.FileStore(rdvz_file, world_size)
        device_id = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            device_id=device_id,
        )
        cls.pg = dist.distributed_c10d._get_default_group()
        torch.set_default_device(device_id)

    @property
    def _rank_value(self):
        # [FUNCTION DOC] HELPER: Property returning rank plus one.
        return self.rank + 1

    def test_default_pg_has_cpu_backend(self):
        # [FUNCTION DOC] Checks the default group has a backend registered for cpu.
        default_pg = dist.distributed_c10d._get_default_group()
        cpu_be = default_pg._get_backend(torch.device("cpu"))
        self.assertIsNotNone(cpu_be)

    def test_default_pg_has_cuda_backend(self):
        # [FUNCTION DOC] Checks the default group has a backend registered for cuda.
        default_pg = dist.distributed_c10d._get_default_group()
        cuda_be = default_pg._get_backend(torch.device("cuda"))
        self.assertIsNotNone(cuda_be)

    def test_allreduce_on_auto_qualified_pg(self):
        # [FUNCTION DOC] All-reduces rank plus one on the default group and checks the sum.
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=self.pg)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_new_group_from_auto_qualified_parent(self):
        # [FUNCTION DOC] Creates a new group over all ranks from the auto-qualified parent and
        # checks an all-reduce sum.
        ranks = list(range(self.world_size))
        ng = dist.new_group(ranks=ranks)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, group=ng)
        self.assertEqual(tensor.item(), sum(range(1, self.world_size + 1)))

    def test_bound_device_id_is_set(self):
        # [FUNCTION DOC] Checks the default group has a bound device and that its type is cuda.
        default_pg = dist.distributed_c10d._get_default_group()
        self.assertIsNotNone(default_pg.bound_device_id)
        self.assertEqual(default_pg.bound_device_id.type, "cuda")

    def test_monitored_barrier_on_multi_backend_group(self):
        # [FUNCTION DOC] Checks the backend of the device-bound default group is not plain gloo
        # but mentions gloo, and that the monitored barrier (30 second timeout, wait for all
        # ranks) is still accepted and completes.
        # A device-bound world reports a multi-backend string from get_backend
        # (e.g. "cpu:gloo,cuda:nccl"), which is NOT equal to Backend.GLOO. The
        # relaxed monitored_barrier guard must still accept it under torchcomms
        # (because it has a CPU-capable backend) and dispatch to the gloo CPU
        # backend's BackendWrapper::monitoredBarrier, rather than rejecting it
        # for not being a plain gloo group.
        backend = dist.get_backend(self.pg)
        self.assertNotEqual(backend, dist.Backend.GLOO)
        self.assertIn("gloo", str(backend))
        # All ranks check in -> the barrier returns cleanly. Reaching past this
        # call proves the guard accepted the group and the barrier ran.
        dist.monitored_barrier(
            group=self.pg,
            timeout=datetime.timedelta(seconds=30),
            wait_all_ranks=True,
        )

    def test_new_group_nccl_lazy_builds_per_peer_group(self):
        # [FUNCTION DOC] Creates an nccl-lazy group over all ranks with local synchronization,
        # checks its ranks, and does a ring send and receive of cuda tensors over it (even ranks
        # send first), checking each rank receives the previous rank's value.
        # Passing backend="nccl-lazy" builds a per-peer, lazily-initialized
        # group (a dedicated comm + stream per send/recv peer) usable for P2P.
        # use_local_synchronization makes it members-only. The parent must be
        # device-bound (it is, in this class).
        ranks = list(range(self.world_size))
        g = dist.new_group(
            ranks=ranks, backend="nccl-lazy", use_local_synchronization=True
        )
        self.assertEqual(dist.get_process_group_ranks(g), ranks)
        # P2P round-trip over the lazy group: 0 -> 1 -> ... -> 0
        dev = torch.device(f"cuda:{self.rank}")
        send_to = (self.rank + 1) % self.world_size
        recv_from = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(
                torch.full((4,), float(self.rank), device=dev), dst=send_to, group=g
            )
            r = torch.empty(4, device=dev)
            dist.recv(r, src=recv_from, group=g)
        else:
            r = torch.empty(4, device=dev)
            dist.recv(r, src=recv_from, group=g)
            dist.send(
                torch.full((4,), float(self.rank), device=dev), dst=send_to, group=g
            )
        self.assertEqual(r[0].item(), float(recv_from))

    def test_non_torchcomms_backend_falls_through_to_c10d(self):
        # [FUNCTION DOC] Registers (if absent) a custom backend whose creator builds a gloo group,
        # creates a group with it, and checks the cpu backend is a real gloo group and not a
        # TorchComms wrapper. Then checks an all-reduce sum. Side effect: the backend stays
        # registered after the test.
        # Under torchcomms, a backend TorchComms does not own (registered the way
        # mooncake registers a custom c10d backend) must route through the normal
        # ProcessGroup path, not new_comm. Use a gloo-backed stand-in.
        def _creator(store, grank, gsize, timeout):
            # [FUNCTION DOC] HELPER: Backend creator returning a gloo process group built from the
            # given store, rank, size and timeout.
            return dist.ProcessGroupGloo(store, grank, gsize, timeout)

        name = "tc_gate_stub"
        if name not in dist.Backend.backend_list:
            dist.Backend.register_backend(name, _creator, devices=["cpu", "cuda"])

        ranks = list(range(self.world_size))
        g = dist.new_group(ranks=ranks, backend=name)
        be = g._get_backend(torch.device("cpu"))
        # The c10d creator ran (real ProcessGroupGloo), not a TorchComms wrapper.
        self.assertNotIn("BackendWrapper", type(be).__name__)
        t = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(t, group=g)
        self.assertEqual(t.item(), sum(range(1, self.world_size + 1)))


instantiate_device_type_tests(
    TestC10dTorchCommsInitAutoQualify, globals(), only_for=["cuda"]
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsMixedBackends(C10dTorchCommsTestBase):
    # [CLASS DOC] Cuda-only multi-process tests for subgroup creation when the default group mixes
    # cuda nccl with a cpu fake backend; the subgroup is built directly without a parent split.
    """Verify subgroup creation from a mixed-backend parent under torchcomms.

    The parent PG mixes ``cuda:nccl`` with ``cpu:fake``. Under torchcomms,
    ``new_group`` builds each subgroup directly via ``new_comm`` (a members-only
    store rendezvous), so a non-splittable device backend in the parent (here
    ``cpu:fake``) is no obstacle: members construct the subgroup and non-members
    return ``NON_GROUP_MEMBER`` without any parent split.
    """

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Worker-side initialization like the auto-qualify class but with
        # the backend string cuda nccl plus cpu fake; stores the default group in a class
        # attribute and sets the default device to the rank's cuda device.
        torch.distributed.config.use_torchcomms = True
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = rdvz_file
        os.environ["LOCAL_RANK"] = str(rank)

        store = dist.FileStore(rdvz_file, world_size)
        device_id = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        dist.init_process_group(
            backend="cuda:nccl,cpu:fake",
            world_size=world_size,
            rank=rank,
            store=store,
            device_id=device_id,
        )
        cls.pg = dist.distributed_c10d._get_default_group()
        torch.set_default_device(device_id)

    @property
    def _rank_value(self):
        # [FUNCTION DOC] HELPER: Property returning rank plus one.
        return self.rank + 1

    def test_mixed_backend_subgroup(self):
        # [FUNCTION DOC] Creates a subgroup of the first half of the ranks. Members check the
        # group ranks and an all-reduce sum over their values; non-members check they get the
        # non-member sentinel.
        subg_ranks = list(range(self.world_size // 2))
        ng = dist.new_group(ranks=subg_ranks)

        if self.rank in subg_ranks:
            self.assertEqual(dist.get_process_group_ranks(ng), subg_ranks)
            tensor = torch.tensor([self._rank_value], dtype=torch.float32)
            dist.all_reduce(tensor, group=ng)
            self.assertEqual(tensor.item(), sum(r + 1 for r in subg_ranks))
        else:
            self.assertIs(ng, dist.GroupMember.NON_GROUP_MEMBER)


instantiate_device_type_tests(
    TestC10dTorchCommsMixedBackends, globals(), only_for=["cuda"]
)


class TestTorchCommsHandlesBackend(TestCase):
    # [CLASS DOC] Single-process unit tests of the routing helper that decides whether TorchComms
    # handles a backend string. The helper's availability flag and its built and registered
    # lookups are patched, so these run without TorchComms installed.
    """Unit-test the pure routing logic of ``_torchcomms_handles_backend``.

    This decides whether a backend is one TorchComms can construct (so it goes
    through ``new_comm`` / ``split_group``) versus a custom c10d plugin such as
    ``mooncake`` that must fall through to the normal ProcessGroup path. The
    function depends only on ``_TORCHCOMM_AVAILABLE`` and the two
    ``is_backend_*`` callables, all of which we patch -- so these tests run even
    when torchcomms is not installed.
    """

    def _patch(self, available=True, built=(), registered=()):
        # [FUNCTION DOC] HELPER: Returns a patcher for the c10d module that sets the availability
        # flag and replaces the built and registered lookups with membership checks in the given
        # collections (attributes are created if missing).
        built, registered = set(built), set(registered)
        return mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=available,
            _torchcomms_is_backend_built=lambda name: name in built,
            _torchcomms_is_backend_registered=lambda name: name in registered,
            create=True,
        )

    def test_unavailable_returns_false(self):
        # [FUNCTION DOC] Checks the helper returns false for both a named backend and none when
        # TorchComms is unavailable.
        with self._patch(available=False, built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl"))
            self.assertFalse(c10d._torchcomms_handles_backend(None))

    def test_none_backend_inherits_parent(self):
        # [FUNCTION DOC] Checks that a backend of none (inherit the parent) is handled.
        with self._patch(built=[]):
            self.assertTrue(c10d._torchcomms_handles_backend(None))

    def test_builtin_backend(self):
        # [FUNCTION DOC] Checks that a backend reported as built is handled.
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl"))

    def test_registered_backend(self):
        # [FUNCTION DOC] Checks that a backend reported as registered is handled.
        with self._patch(registered=["myadapter"]):
            self.assertTrue(c10d._torchcomms_handles_backend("myadapter"))

    def test_custom_backend_falls_through(self):
        # [FUNCTION DOC] Checks that backends neither built nor registered fall through (not
        # handled).
        with self._patch(built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("mooncake"))
            self.assertFalse(c10d._torchcomms_handles_backend("ucc"))

    def test_case_insensitive(self):
        # [FUNCTION DOC] Checks the lookup is case insensitive for an upper case name.
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("NCCL"))

    def test_nccl_lazy_is_own_backend(self):
        # [FUNCTION DOC] Checks nccl-lazy is handled only when nccl-lazy itself is built, not when
        # only nccl is built or nothing is.
        # nccl-lazy is a distinct built torchcomms backend, not aliased to nccl:
        # it is handled iff nccl-lazy itself is built/registered.
        with self._patch(built=["nccl-lazy"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl-lazy"))
        with self._patch(built=["nccl"]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl-lazy"))
        with self._patch(built=[]):
            self.assertFalse(c10d._torchcomms_handles_backend("nccl-lazy"))

    def test_qualified_all_handled(self):
        # [FUNCTION DOC] Checks a device-qualified comma-separated string is handled when all its
        # backends are built.
        with self._patch(built=["gloo", "nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("cpu:gloo,cuda:nccl"))

    def test_qualified_one_unhandled_falls_through(self):
        # [FUNCTION DOC] Checks a qualified string is not handled when one of its backends is
        # neither built nor registered.
        with self._patch(built=["gloo"]):
            self.assertFalse(c10d._torchcomms_handles_backend("cpu:gloo,cuda:mooncake"))

    def test_empty_parts_are_skipped(self):
        # [FUNCTION DOC] Checks empty and whitespace-padded entries in a comma-separated string
        # are ignored.
        with self._patch(built=["nccl"]):
            self.assertTrue(c10d._torchcomms_handles_backend("nccl,"))
            self.assertTrue(c10d._torchcomms_handles_backend(" nccl , "))


class TestC10dTorchCommsNewGroupHelper(TestCase):
    # [CLASS DOC] Single-process unit tests of the TorchComms branches in the c10d process group
    # creation helper: the device passed to the comm constructor, the rank and size environment
    # variables around that call, and non-member handling. TorchComms parts are patched so
    # TorchComms need not be installed. TIMEOUT is a 30 second timeout shared by the tests.
    """Unit-test the TorchComms-specific branches of ``_new_process_group_helper``.

    These cover the three subgroup-init fixes: the device handed to ``new_comm``
    carries this rank's device index (``device_id``) rather than a device-type-only
    device; ``TORCHCOMM_RANK``/``TORCHCOMM_SIZE`` are seeded from the group's
    rank/size around the ``new_comm`` call and restored afterwards; and a
    non-member of a subgroup must NOT issue a no-color parent split under
    TorchComms. Everything TorchComms-specific is patched (``create=True``), so
    these run even when TorchComms is not installed.
    """

    TIMEOUT = datetime.timedelta(seconds=30)

    def _drive_member(self, *, backend, device_id, group_rank, group_size):
        # [FUNCTION DOC] HELPER: Drives the member path of the process group helper on the
        # default-group route with TorchComms forced on and every backend handled. The comm
        # constructor is replaced by a fake that records the backend string, device and the rank
        # and size environment values, then raises a sentinel error that the helper expects.
        # Returns the recorded values.
        """Drive the members path down to ``new_comm``.

        ``new_comm`` is mocked to capture its device argument and the live env,
        then abort with a sentinel so we never touch real comm machinery. Uses
        the default-group path (``global_ranks_in_group == []``) so no
        initialized world is required. Returns the captured dict.
        """
        captured = {}

        def fake_new_comm(backend_str, device, name=None, store=None, hints=None):
            # [FUNCTION DOC] HELPER: Fake comm constructor recording its backend string,
            # device and the current rank and size environment variables into the enclosing
            # dict, then raising RuntimeError with a stop message.
            captured["backend_str"] = backend_str
            captured["device"] = device
            captured["rank_env"] = os.environ.get("TORCHCOMM_RANK")
            captured["size_env"] = os.environ.get("TORCHCOMM_SIZE")
            raise RuntimeError("stop-after-new_comm")

        with mock.patch.multiple(
            c10d,
            _use_torchcomms_enabled=lambda: True,
            _torchcomms_handles_backend=lambda b: True,
            new_comm=fake_new_comm,
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop-after-new_comm"):
                c10d._new_process_group_helper(
                    group_size=group_size,
                    group_rank=group_rank,
                    global_ranks_in_group=[],
                    backend=backend,
                    store=dist.HashStore(),
                    group_name=c10d.GroupName(self.id()),
                    timeout=self.TIMEOUT,
                    device_id=device_id,
                )
        return captured

    def test_new_comm_gets_indexed_device_id(self):
        # [FUNCTION DOC] Checks the device handed to the comm constructor is the bound cuda device
        # with its index, not a device-type-only device.
        # A subgroup's group-local rank differs from the rank's physical device,
        # so new_comm must receive device_id (WITH index), not a device-type-only
        # device. group_rank (2) deliberately differs from the device index (3).
        cap = self._drive_member(
            backend="nccl",
            device_id=torch.device("cuda:3"),
            group_rank=2,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cuda:3"))

    def test_new_comm_device_type_mismatch_not_overridden(self):
        # [FUNCTION DOC] Checks that with the gloo backend and a cuda device id, the device stays
        # cpu (type mismatch is not overridden).
        # gloo maps to the cpu device; device_id is a cuda device, so the type
        # guard must leave cpu alone rather than substituting cuda:3.
        cap = self._drive_member(
            backend="gloo",
            device_id=torch.device("cuda:3"),
            group_rank=1,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cpu"))

    def test_new_comm_without_device_id_keeps_type_only_device(self):
        # [FUNCTION DOC] Checks that without a device id the device stays the type-only cuda
        # device.
        # World-group path: no device_id bound, so the guard must not fire and
        # new_comm gets the device-type-only device from the backend map.
        cap = self._drive_member(
            backend="nccl",
            device_id=None,
            group_rank=0,
            group_size=4,
        )
        self.assertEqual(cap["device"], torch.device("cuda"))

    def test_torchcomm_rank_size_seeded_from_group(self):
        # [FUNCTION DOC] Checks the rank and size environment variables seen inside the comm
        # constructor equal the group rank and size.
        cap = self._drive_member(
            backend="nccl",
            device_id=torch.device("cuda:1"),
            group_rank=2,
            group_size=7,
        )
        self.assertEqual(cap["rank_env"], "2")
        self.assertEqual(cap["size_env"], "7")

    def test_torchcomm_rank_size_restored_after_call(self):
        # [FUNCTION DOC] Checks the rank and size environment variables are unset after the call
        # if they were unset before, and restored to their prior values otherwise. Saves and
        # restores the real environment around the test.
        saved = {k: os.environ.get(k) for k in ("TORCHCOMM_RANK", "TORCHCOMM_SIZE")}
        try:
            for k in ("TORCHCOMM_RANK", "TORCHCOMM_SIZE"):
                os.environ.pop(k, None)
            # Unset before -> unset after.
            self._drive_member(
                backend="nccl",
                device_id=torch.device("cuda:0"),
                group_rank=0,
                group_size=2,
            )
            self.assertIsNone(os.environ.get("TORCHCOMM_RANK"))
            self.assertIsNone(os.environ.get("TORCHCOMM_SIZE"))
            # Set before -> original values restored after.
            os.environ["TORCHCOMM_RANK"] = "99"
            os.environ["TORCHCOMM_SIZE"] = "77"
            self._drive_member(
                backend="nccl",
                device_id=torch.device("cuda:0"),
                group_rank=0,
                group_size=2,
            )
            self.assertEqual(os.environ["TORCHCOMM_RANK"], "99")
            self.assertEqual(os.environ["TORCHCOMM_SIZE"], "77")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _drive_non_member(self, *, torchcomms_enabled):
        # [FUNCTION DOC] HELPER: Drives the non-member path: fakes an initialized, device-bound
        # default group whose rank is not in the subgroup ranks, with TorchComms toggled by the
        # flag. Returns the helper result and the mock that stands for the split source.
        """Drive the non-member early-return path of a subgroup.

        The default group is faked as initialized and device-bound with this
        rank (0) absent from the requested subgroup, so ``_new_process_group_helper``
        takes the ``NON_GROUP_MEMBER`` branch. Returns (result, split_source_mock).
        """
        split_src = mock.MagicMock()
        default = mock.MagicMock()
        default.rank.return_value = 0
        default.bound_device_id = torch.device("cuda:0")
        with mock.patch.multiple(
            c10d,
            _use_torchcomms_enabled=lambda: torchcomms_enabled,
            is_initialized=lambda: True,
            _get_default_group=lambda: default,
            _get_split_source=lambda pg: split_src,
        ):
            res = c10d._new_process_group_helper(
                group_size=2,
                group_rank=0,
                global_ranks_in_group=[1, 2],  # rank 0 is NOT a member
                backend="nccl",
                store=dist.HashStore(),
                group_name=c10d.GroupName(self.id()),
                timeout=self.TIMEOUT,
            )
        return res, split_src

    def test_non_member_skips_nocolor_split_under_torchcomms(self):
        # [FUNCTION DOC] Checks that under TorchComms a non-member gets the non-member sentinel
        # result and the no-color split is not performed.
        res, split_src = self._drive_non_member(torchcomms_enabled=True)
        self.assertEqual(res, (dist.GroupMember.NON_GROUP_MEMBER, None))
        split_src.perform_nocolor_split.assert_not_called()

    def test_non_member_performs_nocolor_split_without_torchcomms(self):
        # [FUNCTION DOC] Checks that without TorchComms a non-member still gets the sentinel and
        # the no-color split is performed once on the bound device.
        # Contrast: with TorchComms disabled the NCCL-style path still requires
        # non-members to issue the no-color split to stay in sync.
        res, split_src = self._drive_non_member(torchcomms_enabled=False)
        self.assertEqual(res, (dist.GroupMember.NON_GROUP_MEMBER, None))
        split_src.perform_nocolor_split.assert_called_once_with(torch.device("cuda:0"))


class TestC10dGroupNameHashSalt(TestCase):
    # [CLASS DOC] Single-process tests that process group names hashed from ranks are salted with
    # the group counter, which advances in lockstep on every rank, and not with the number of
    # registered groups, which differs between members and non-members. Saves and restores global
    # world state.
    """Unit-test the collective-consistent group-name hash salt.

    ``_hash_ranks_to_str`` / ``_process_group_name`` are generic c10d helpers
    (not TorchComms-specific), but the divergence they can produce was surfaced
    by the TorchComms subgroup work. After an earlier asymmetric-membership
    ``new_group`` (e.g. ``new_group([0])``), member and non-member ranks hold
    different ``len(_world.pg_names)`` because only members register a PG.
    Salting the hash with that length makes ranks compute DIFFERENT names for the
    same later group; backends that use the group name as a rendezvous store
    prefix (e.g. Gloo split's ``connectFullMesh``) then key off mismatched
    prefixes and deadlock. The salt must instead be ``_world.group_count``, which
    every rank advances in lockstep -- ``_process_group_name`` increments it
    before the membership check, on BOTH the hashed and non-hashed paths.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves the world's group count and a copy of the group-name
        # mapping.
        super().setUp()
        self._saved_group_count = c10d._world.group_count
        self._saved_pg_names = dict(c10d._world.pg_names)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the saved group count and group-name mapping, then runs
        # the base tearDown.
        c10d._world.group_count = self._saved_group_count
        c10d._world.pg_names.clear()
        c10d._world.pg_names.update(self._saved_pg_names)
        super().tearDown()

    def _register_dummy_pgs(self, n):
        # [FUNCTION DOC] HELPER: Registers the given number of dummy entries in the world
        # group-name mapping to simulate extra groups.
        # Simulate a rank that registered extra PGs (asymmetric membership).
        for i in range(n):
            c10d._world.pg_names[object()] = c10d.GroupName(f"dummy_{i}")

    def test_hash_salt_uses_group_count(self):
        # [FUNCTION DOC] Checks the same ranks hash to different names under different group
        # counts.
        # Same ranks, different group_count -> different name (salt is live).
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 5
        name_a = c10d._hash_ranks_to_str(ranks)
        c10d._world.group_count = 6
        name_b = c10d._hash_ranks_to_str(ranks)
        self.assertNotEqual(name_a, name_b)

    def test_hash_independent_of_pg_names_length(self):
        # [FUNCTION DOC] Regression check: the hashed name does not change with the number of
        # registered group names.
        # ROOT-CAUSE regression: the hash must NOT depend on len(_world.pg_names),
        # the quantity that diverges across ranks under asymmetric membership.
        ranks = [0, 1, 2, 3]
        c10d._world.group_count = 9
        c10d._world.pg_names.clear()
        name_few = c10d._hash_ranks_to_str(ranks)
        self._register_dummy_pgs(5)
        name_many = c10d._hash_ranks_to_str(ranks)
        self.assertEqual(name_few, name_many)

    def test_two_asymmetric_ranks_agree_on_name(self):
        # [FUNCTION DOC] Checks that two ranks with equal group counts but different numbers of
        # registered groups compute the same hashed group name.
        # The deadlock scenario end-to-end through _process_group_name: two ranks
        # that made the SAME sequence of group-creation calls (equal group_count)
        # but registered a DIFFERENT number of PGs must compute the same name.
        ranks = [0, 1, 2, 3]
        # Rank that was a member of earlier subgroups: more pg_names.
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        self._register_dummy_pgs(2)
        name_member = c10d._process_group_name(ranks, use_hashed_name=True)
        # Rank that was a non-member: fewer pg_names, but same group_count because
        # every rank advances it in lockstep regardless of membership.
        c10d._world.group_count = 3
        c10d._world.pg_names.clear()
        name_non_member = c10d._process_group_name(ranks, use_hashed_name=True)
        self.assertEqual(name_member, name_non_member)

    def test_process_group_name_increments_on_hashed_path(self):
        # [FUNCTION DOC] Checks the group counter is incremented by one when the name is hashed.
        c10d._world.group_count = 10
        c10d._process_group_name([0, 1], use_hashed_name=True)
        self.assertEqual(c10d._world.group_count, 11)

    def test_process_group_name_increments_on_nonhashed_path(self):
        # [FUNCTION DOC] Checks the group counter is incremented by one when the name is not
        # hashed.
        c10d._world.group_count = 20
        c10d._process_group_name([0, 1], use_hashed_name=False)
        self.assertEqual(c10d._world.group_count, 21)

    def test_nonhashed_name_is_group_count_before_increment(self):
        # [FUNCTION DOC] Checks the non-hashed name is the group counter value before the
        # increment, and the counter then advances.
        c10d._world.group_count = 42
        name = c10d._process_group_name([0, 1], use_hashed_name=False)
        self.assertEqual(name, "42")
        self.assertEqual(c10d._world.group_count, 43)

    def test_repeated_hashed_same_ranks_get_distinct_names(self):
        # [FUNCTION DOC] Checks that two hashed names over the same ranks differ because the
        # counter advances.
        # Two distinct PGs over identical ranks must get distinct names; the salt
        # advancing on the hashed path (fixed by this commit) is what
        # disambiguates them. Before the fix the hashed path left group_count
        # unchanged, so back-to-back splits over the same ranks collided.
        c10d._world.group_count = 0
        first = c10d._process_group_name([2, 3], use_hashed_name=True)
        second = c10d._process_group_name([2, 3], use_hashed_name=True)
        self.assertNotEqual(first, second)


class _FakeComm:
    # [CLASS DOC] HELPER: Fake comm that counts finalize calls and raises RuntimeError on any call
    # after the first, mimicking a non-idempotent finalize.
    """Stand-in for a TorchComm whose ``finalize`` is not idempotent.

    Mirrors the real comm: the second ``finalize`` raises, exactly the failure
    ``destroy_process_group`` must avoid when a comm is shared across device
    types.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the finalize call counter to zero.
        self.finalize_calls = 0

    def finalize(self):
        # [FUNCTION DOC] HELPER: Counts the call and raises RuntimeError if this is not the first
        # call.
        self.finalize_calls += 1
        if self.finalize_calls > 1:
            raise RuntimeError("already finalized")


class _FakeBackendWrapper:
    # [CLASS DOC] HELPER: Stand-in for the TorchComms backend wrapper that just holds a comm and
    # returns it.
    def __init__(self, comm):
        # [FUNCTION DOC] HELPER: Stores the comm.
        self._comm = comm

    def get_comm(self):
        # [FUNCTION DOC] HELPER: Returns the stored comm.
        return self._comm


class _FakeOtherBackend:
    # [CLASS DOC] HELPER: Marker backend that is not a wrapper and must be left alone.
    """A c10d backend that is not a _BackendWrapper (must be left untouched)."""


class TestC10dTorchCommsDestroyDedup(TestCase):
    # [CLASS DOC] Single-process unit tests that destroying a process group finalizes each shared
    # comm exactly once, using fake wrappers and comms and a mocked group. Saves the global world
    # containers and restores them afterwards. _WORLD_ATTRS lists the saved containers.
    """Unit-test the comm-deduplication in ``destroy_process_group``.

    A gloo subgroup reports both ``cuda`` and ``cpu`` device types backed by the
    same _BackendWrapper/comm; the destroy loop must finalize each comm exactly
    once because ``finalize()`` is not idempotent. Everything TorchComms-specific
    is patched (``create=True``), so these run even when TorchComms is not
    installed.
    """

    _WORLD_ATTRS = (
        "pg_map",
        "pg_names",
        "pg_group_ranks",
        "pg_backend_config",
        "pg_to_tag",
        "tags_to_pg",
        "pg_coalesce_state",
        "comms",
    )

    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves copies of the world containers, the default group and the
        # group counter.
        super().setUp()
        self._saved_world = {
            attr: getattr(c10d._world, attr).copy() for attr in self._WORLD_ATTRS
        }
        self._saved_default_pg = c10d._world.default_pg
        self._saved_group_count = c10d._world.group_count

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores each saved container's contents, the default group and
        # the group counter, then runs the base tearDown.
        for attr, value in self._saved_world.items():
            container = getattr(c10d._world, attr)
            container.clear()
            if isinstance(value, dict):
                container.update(value)
            else:
                container.extend(value)
        c10d._world.default_pg = self._saved_default_pg
        c10d._world.group_count = self._saved_group_count
        super().tearDown()

    def _register_pg(self, device_backends, *, name=None):
        # [FUNCTION DOC] HELPER: Creates a mock group with the given device-to-backend mapping and
        # name (defaulting to the test id), registers it in the world maps for groups, names,
        # ranks, backend config and tags, and returns it.
        """Register a fake PG whose backends are keyed by device type."""
        pg = mock.MagicMock()
        pg._device_types = list(device_backends)
        pg._get_backend.side_effect = lambda dev: device_backends[dev]
        group_name = c10d.GroupName(self.id() if name is None else name)
        pg.group_name = group_name

        c10d._world.pg_map[pg] = (None, None)
        c10d._world.pg_names[pg] = group_name
        c10d._world.pg_group_ranks[pg] = {}
        c10d._world.pg_backend_config[pg] = ""
        c10d._world.pg_to_tag[pg] = None
        return pg

    def _drive_destroy(self, device_backends):
        # [FUNCTION DOC] HELPER: Registers a fake group and destroys it with the TorchComms flag,
        # wrapper class and unregistration patched; returns the fake group.
        """Register a fake subgroup PG and run ``destroy_process_group`` on it."""
        pg = self._register_pg(device_backends)

        with mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=True,
            _BackendWrapper=_FakeBackendWrapper,
            _unregister_process_group=lambda group_name: None,
            create=True,
        ):
            c10d.destroy_process_group(pg)
        return pg

    def test_shared_comm_finalized_once(self):
        # [FUNCTION DOC] Checks that one wrapper shared for cuda and cpu finalizes its comm once,
        # shuts the group down once and removes it from the world.
        # gloo scenario: one _BackendWrapper/comm shared across cuda and cpu.
        comm = _FakeComm()
        wrapper = _FakeBackendWrapper(comm)
        pg = self._drive_destroy({"cuda": wrapper, "cpu": wrapper})
        self.assertEqual(comm.finalize_calls, 1)
        pg.shutdown.assert_called_once()
        self.assertNotIn(pg, c10d._world.pg_map)

    def test_same_comm_distinct_wrappers_finalized_once(self):
        # [FUNCTION DOC] Checks that distinct wrappers sharing one comm finalize it once.
        # Dedup keys on comm identity, not wrapper identity: two distinct
        # _BackendWrapper objects sharing one comm still finalize it once.
        comm = _FakeComm()
        pg = self._drive_destroy(
            {"cuda": _FakeBackendWrapper(comm), "cpu": _FakeBackendWrapper(comm)}
        )
        self.assertEqual(comm.finalize_calls, 1)
        pg.shutdown.assert_called_once()

    def test_distinct_comms_each_finalized_once(self):
        # [FUNCTION DOC] Checks that two different comms are each finalized once.
        comm_a, comm_b = _FakeComm(), _FakeComm()
        self._drive_destroy(
            {"cuda": _FakeBackendWrapper(comm_a), "cpu": _FakeBackendWrapper(comm_b)}
        )
        self.assertEqual(comm_a.finalize_calls, 1)
        self.assertEqual(comm_b.finalize_calls, 1)

    def test_non_backendwrapper_backend_is_skipped(self):
        # [FUNCTION DOC] Checks a backend that is not a wrapper is skipped and the other comm is
        # finalized once.
        # A non-_BackendWrapper backend must not be finalized; the wrapped comm
        # still finalizes exactly once.
        comm = _FakeComm()
        self._drive_destroy(
            {"cuda": _FakeBackendWrapper(comm), "cpu": _FakeOtherBackend()}
        )
        self.assertEqual(comm.finalize_calls, 1)

    def test_destroy_subgroup_keeps_other_live_comms_tracked(self):
        # [FUNCTION DOC] Registers a world group and two subgroups with distinct comms, destroys
        # one subgroup, and checks only its comm was finalized and removed from the tracked comms.
        # Then destroys the whole world and checks the other comm was finalized once and the comm
        # list is empty.
        world_pg = self._register_pg({}, name="world_pg")
        comm_a, comm_b = _FakeComm(), _FakeComm()
        subgroup_a = self._register_pg(
            {"cuda": _FakeBackendWrapper(comm_a)}, name="subgroup_a"
        )
        subgroup_b = self._register_pg(
            {"cuda": _FakeBackendWrapper(comm_b)}, name="subgroup_b"
        )
        c10d._world.default_pg = world_pg
        c10d._world.comms.extend([comm_a, comm_b])

        with mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=True,
            _BackendWrapper=_FakeBackendWrapper,
            _unregister_process_group=lambda group_name: None,
            _unregister_all_process_groups=lambda: None,
            _update_default_pg=lambda pg: setattr(c10d._world, "default_pg", pg),
            create=True,
        ):
            c10d.destroy_process_group(subgroup_a)
            self.assertEqual(comm_a.finalize_calls, 1)
            self.assertEqual(comm_b.finalize_calls, 0)
            self.assertNotIn(subgroup_a, c10d._world.pg_map)
            self.assertIn(subgroup_b, c10d._world.pg_map)
            self.assertEqual(c10d._world.comms, [comm_b])

            c10d.destroy_process_group()
            self.assertEqual(comm_a.finalize_calls, 1)
            self.assertEqual(comm_b.finalize_calls, 1)
            self.assertEqual(c10d._world.comms, [])


class TestC10dTorchCommsBackendConfig(TestCase):
    # [CLASS DOC] Tests that the default backend type for a device-qualified backend config
    # follows whether TorchComms is enabled and the backend is registered. The tests replace
    # module-level functions by hand and restore them in a finally block.
    def test_registered_device_qualified_torchcomms_backend_uses_custom_type(self):
        # [FUNCTION DOC] With TorchComms enabled and the backend registered (not built), checks
        # the default backend type is the custom type.
        backend_name = "tc_test_backend"
        self.assertNotIn(backend_name, dist.Backend.backend_type_map)

        orig_use = c10d._use_torchcomms_enabled
        orig_registered = c10d._torchcomms_is_backend_registered
        orig_built = c10d._torchcomms_is_backend_built
        c10d._use_torchcomms_enabled = lambda: True
        c10d._torchcomms_is_backend_registered = lambda backend: backend == backend_name
        c10d._torchcomms_is_backend_built = lambda backend: False
        try:
            backend_config = dist.BackendConfig(f"cuda:{backend_name}")
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(backend_config),
                dist.ProcessGroup.BackendType.CUSTOM,
            )
        finally:
            c10d._use_torchcomms_enabled = orig_use
            c10d._torchcomms_is_backend_registered = orig_registered
            c10d._torchcomms_is_backend_built = orig_built

    def test_unregistered_device_qualified_backend_with_torchcomms_keeps_gloo_type(
        self,
    ):
        # [FUNCTION DOC] With TorchComms enabled but the backend not registered or built, checks
        # the default backend type is gloo.
        backend_name = "tc_test_backend"
        self.assertNotIn(backend_name, dist.Backend.backend_type_map)

        orig_use = c10d._use_torchcomms_enabled
        orig_registered = c10d._torchcomms_is_backend_registered
        orig_built = c10d._torchcomms_is_backend_built
        c10d._use_torchcomms_enabled = lambda: True
        c10d._torchcomms_is_backend_registered = lambda backend: False
        c10d._torchcomms_is_backend_built = lambda backend: False
        try:
            backend_config = dist.BackendConfig(f"cuda:{backend_name}")
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(backend_config),
                dist.ProcessGroup.BackendType.GLOO,
            )
        finally:
            c10d._use_torchcomms_enabled = orig_use
            c10d._torchcomms_is_backend_registered = orig_registered
            c10d._torchcomms_is_backend_built = orig_built

    def test_registered_device_qualified_backend_without_torchcomms_keeps_gloo_type(
        self,
    ):
        # [FUNCTION DOC] With TorchComms disabled even though the backend is registered and built,
        # checks the default backend type is gloo.
        backend_name = "tc_test_backend"
        self.assertNotIn(backend_name, dist.Backend.backend_type_map)

        orig_use = c10d._use_torchcomms_enabled
        orig_registered = c10d._torchcomms_is_backend_registered
        orig_built = c10d._torchcomms_is_backend_built
        c10d._use_torchcomms_enabled = lambda: False
        c10d._torchcomms_is_backend_registered = lambda backend: backend == backend_name
        c10d._torchcomms_is_backend_built = lambda backend: True
        try:
            backend_config = dist.BackendConfig(f"cuda:{backend_name}")
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(backend_config),
                dist.ProcessGroup.BackendType.GLOO,
            )
        finally:
            c10d._use_torchcomms_enabled = orig_use
            c10d._torchcomms_is_backend_registered = orig_registered
            c10d._torchcomms_is_backend_built = orig_built


if __name__ == "__main__":
    run_tests()
