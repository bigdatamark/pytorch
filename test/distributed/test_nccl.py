# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for NCCL collectives exposed by torch.cuda.nccl (broadcast, reduce,
# all_reduce, all_gather, reduce_scatter, unique_id and input validation) and for NCCL-backed
# symmetric memory (torch.distributed._symmetric_memory): allocation, rendezvous, barrier,
# signals, put/get, reduce_scatter_offset, all_to_all_nd, CUDA graph capture and mem pools.
# Import-time effects: counts GPUs (nGPUs), replaces TestCase with NoTest when CUDA is
# unavailable (printing a skip message), builds datatypes (float, plus bfloat16 when NCCL
# 2.10+ or ROCm) and broadcast_dtypes (float8 variants, ROCm-specific ones on ROCm), and
# instantiates TestNCCL as CUDA device tests.

import os
import sys

import torch
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as c10d
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    load_tests,
    NoTest,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


datatypes = [torch.float]
if (
    TEST_CUDA and c10d.is_nccl_available() and nccl.version() >= (2, 10)
) or TEST_WITH_ROCM:
    datatypes.append(torch.bfloat16)

# Broadcast (and alltoall) support float8, while reduce and allreduce do not support float8 currently
broadcast_dtypes = (
    datatypes + [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
    if TEST_WITH_ROCM
    else [torch.float8_e4m3fn, torch.float8_e5m2]
)


class TestNCCL(TestCase):
    # [CLASS DOC] Device-generic tests (instantiated for CUDA only) of the torch.cuda.nccl
    # module-level collectives acting on lists of per-GPU tensors; most need more than one GPU
    # and are skipped on Windows.
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        # [FUNCTION DOC] nccl.unique_id returns a bytes object longer than one byte.
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*broadcast_dtypes)
    def test_broadcast(self, device, dtype):
        # [FUNCTION DOC] Broadcasts a random 128-element tensor from GPU 0 to zero tensors on
        # every other GPU, using a list, then a tuple, then a non-zero root (the last GPU,
        # regression for #179908); each time every GPU's tensor must equal the source.
        # Parametrized over the float8 broadcast dtypes.
        # KNOWN ISSUE: the device argument is shadowed by the loop variables.
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with tuple
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tuple(tensors))
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with a non-zero root (regression test for #179908)
        root = nGPUs - 1
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [
            expected.cuda(device)
            if device == root
            else torch.zeros(128, dtype=dtype, device=device)
            for device in range(nGPUs)
        ]

        nccl.broadcast(tensors, root=root)
        for i in range(nGPUs):
            self.assertEqual(tensors[i], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce(self, device, dtype):
        # [FUNCTION DOC] Reduces random per-GPU tensors with nccl.reduce, given as list and as
        # tuple; checks the root (first) tensor equals the sum of all inputs.
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

        # Test with tuple
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tuple(tensors))

        self.assertEqual(tensors[0], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_reduce(self, device, dtype):
        # [FUNCTION DOC] all_reduce over per-GPU tensors passed as list, tuple and set; every
        # resulting tensor must equal the sum of the inputs.
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        tensors = tuple(cpu_tensors[i].cuda(i) for i in range(nGPUs))
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with set.
        tensors = {cpu_tensors[i].cuda(i) for i in range(nGPUs)}
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_collective_errors(self, device):
        # [FUNCTION DOC] Passing a single tensor instead of a collection to all_reduce,
        # reduce, broadcast, all_gather and reduce_scatter raises TypeError matching the
        # message that inputs should be a collection of tensors.
        t = torch.rand(10).cuda(0)
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.broadcast(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_gather(t, t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce_scatter(t, t)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        # [FUNCTION DOC] all_gather of per-GPU 128-element tensors into per-GPU output
        # buffers, as lists and as tuples; every output must equal the concatenation
        # of all inputs.
        cpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)]
        expected = torch.cat(cpu_inputs, 0)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(tuple(inputs), tuple(outputs))

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        # [FUNCTION DOC] reduce_scatter of per-GPU inputs (32 times GPU count elements) into
        # per-GPU outputs of 32 elements, as lists and tuples; output i must equal slice i of
        # the elementwise sum.
        in_size = 32 * nGPUs
        out_size = 32

        cpu_inputs = [
            torch.zeros(in_size).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(in_size, dtype=dtype)
        for t in cpu_inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])

        # Test with tuple
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(tuple(inputs), tuple(outputs))

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class NCCLSymmetricMemoryTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process tests (MultiProcContinuousTest) of the NCCL symmetric
    # memory backend and related ops. Requires CUDA peer-to-peer access; individual tests
    # skip for ROCm, Windows, fewer than two GPUs and missing NCCL versions. Process groups
    # come from _init_pg.
    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Property: the CUDA device for this process's rank.
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Classmethod run in each worker: requires a rendezvous file,
        # sets LOCAL_RANK in the environment, selects the CUDA device for the rank, creates
        # a file store, initializes an NCCL process group with eager device binding and
        # stores the default group in the class attribute pg. Raises AssertionError if the
        # file path is None.
        # Eager NCCL communicator init via device_id, so symm_mem rendezvous
        # does not require a separate warm-up collective.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_alloc(self):
        # [FUNCTION DOC] Selects the NCCL symmetric memory backend, initializes the
        # communicator with an all_reduce (otherwise the test hangs, per a TODO), then
        # allocates symmetric tensors and rendezvouses them on the world group, both inside a
        # helper and directly.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        def foo():
            # [FUNCTION DOC] HELPER: Allocates a symmetric tensor and performs a
            # rendezvous on it.
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_many_allocations(self):
        # [FUNCTION DOC] Allocates 256 symmetric single-element tensors, rendezvouses every
        # sixteenth (checking the handle's rank and world size) and again a second time, then
        # runs a one-shot all_reduce on the last tensor filled with the rank and expects the
        # sum of ranks.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensors = [
            symm_mem.empty(1, dtype=torch.float, device=self.device) for _ in range(256)
        ]

        # Rendezvous a subset twice so the repeated lookup path is covered
        # while many allocations are still live.
        sampled_tensors = tensors[::16]
        for tensor in sampled_tensors:
            handle = symm_mem.rendezvous(tensor, group=group_name)
            self.assertEqual(handle.rank, self.rank)
            self.assertEqual(handle.world_size, self.world_size)
        for tensor in sampled_tensors:
            symm_mem.rendezvous(tensor, group=group_name)

        result = torch.ops.symm_mem.one_shot_all_reduce(
            tensors[-1].fill_(self.rank), "sum", group_name
        )
        self.assertEqual(
            result, torch.full_like(result, (self.world_size - 1) * self.world_size / 2)
        )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_world(self):
        # [FUNCTION DOC] Rendezvous on the world group; checks handle rank and world size,
        # fills the local buffer with the rank, barriers, and verifies the next peer's buffer
        # (obtained from the handle) contains the peer's rank.
        symm_mem.set_backend("NCCL")
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier()

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_barrier(self):
        # [FUNCTION DOC] Handle barrier test: fills a symmetric tensor with the rank,
        # rendezvouses, runs the handle barrier, checks every peer's buffer holds the peer
        # rank, then barriers again.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        numel = 64
        t = symm_mem.empty(numel, dtype=torch.float32, device=self.device).fill_(
            self.rank
        )
        handle = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(handle.rank, self.rank)
        self.assertEqual(handle.world_size, self.world_size)

        handle.barrier()
        for peer in range(self.world_size):
            buf = handle.get_buffer(peer, (numel,), torch.float32)
            self.assertTrue(buf.eq(peer).all())
        handle.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_barrier_channel_out_of_bounds(self):
        # [FUNCTION DOC] Computes the maximum barrier channel count from the signal pad size
        # and world size; using that channel raises RuntimeError matching maximum supported
        # channel, while the channel below it succeeds (regression for #191618).
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        num_slots = handle.signal_pad_size // 4
        max_channel = num_slots // self.world_size

        # check_channel() is shared with the CUDA backend; an over-capacity
        # channel must be rejected host-side before the kernel launch (#191618).
        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            handle.barrier(channel=max_channel)
        handle.barrier(channel=max_channel - 1)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_signal_rank_out_of_bounds(self):
        # [FUNCTION DOC] Calling put_signal with a destination rank or wait_signal with a
        # source rank of minus one or equal to the world size raises RuntimeError matching
        # a must-be-in-range message.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        for bad_rank in (-1, self.world_size):
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                handle.put_signal(dst_rank=bad_rank)
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                handle.wait_signal(src_rank=bad_rank)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_subgroup(self):
        # [FUNCTION DOC] Same as the world rendezvous test, but rendezvous and barrier use a
        # subgroup containing all ranks created with new_group.
        symm_mem.set_backend("NCCL")

        subgroup = c10d.new_group(list(range(self.world_size)))

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=subgroup)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier(group=subgroup)

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective(self):
        # [FUNCTION DOC] Fills a symmetric tensor with the rank, rendezvouses, all_reduces it
        # (checking the sum of ranks), then checks the one-shot all_reduce of another rank-
        # filled symmetric tensor matches that result.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        c10d.all_reduce(out)
        torch.cuda.synchronize()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        self.assertEqual(out, res)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective_cuda_graph(self):
        # [FUNCTION DOC] Captures a regular all_reduce and a one-shot symmetric all_reduce
        # into CUDA graphs, replays and checks the sum of ranks; then three more times refills
        # inputs with an offset, replays both graphs and checks both outputs equal the
        # expected offset-based sum.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        graph_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_all_reduce):
            c10d.all_reduce(out)
        graph_all_reduce.replay()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        graph_one_shot_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_one_shot_all_reduce):
            res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        graph_one_shot_all_reduce.replay()
        self.assertEqual(out, res)

        for repeat in range(3):
            offset = 13 + repeat
            inp.fill_(self.rank + offset)
            out.fill_(self.rank + offset)
            res.fill_(0.0)
            expected_sum = float(
                self.world_size * offset + self.world_size * (self.world_size - 1) / 2
            )
            graph_all_reduce.replay()
            graph_one_shot_all_reduce.replay()
            self.assertEqual(out, torch.full_like(out, expected_sum))
            self.assertEqual(res, out)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_tensor_creation_and_collective_cuda_graph(self):
        # [FUNCTION DOC] Warms up symmetric allocation plus one_shot_all_reduce_out on a side
        # stream three times, then captures allocation, fill and the collective in a CUDA
        # graph on that stream; replays and verifies the sum, then replays with updated
        # offsets three more times and verifies each result.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        capture_offset = torch.tensor(1, dtype=dtype, device=self.device)
        capture_stream = torch.cuda.Stream(device=self.device)

        # Warm up on the same stream that will be captured. This establishes
        # NCCL window metadata for the symmetric block that capture reuses.
        with torch.cuda.stream(capture_stream):
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            out = torch.empty(numel, dtype=dtype, device=self.device)
            for warmup_idx in range(3):
                offset = 1 + warmup_idx
                capture_offset.fill_(self.rank + offset)
                inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
                out = torch.empty(numel, dtype=dtype, device=self.device)
                inp.fill_(capture_offset)
                out.fill_(0.0)
                torch.ops.symm_mem.one_shot_all_reduce_out(inp, "sum", group_name, out)
                expected_sum = float(
                    self.world_size * offset
                    + self.world_size * (self.world_size - 1) / 2
                )
                self.assertEqual(out, torch.full_like(out, expected_sum))
            del inp, out
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        offset = 13
        capture_offset.fill_(self.rank + offset)
        with torch.cuda.graph(graph, stream=capture_stream):
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            out = torch.empty(numel, dtype=dtype, device=self.device)
            inp.fill_(capture_offset)
            out.fill_(0.0)
            torch.ops.symm_mem.one_shot_all_reduce_out(inp, "sum", group_name, out)

        graph.replay()
        expected_sum = float(
            self.world_size * offset + self.world_size * (self.world_size - 1) / 2
        )
        self.assertEqual(out, torch.full_like(out, expected_sum))

        for repeat in range(3):
            offset = 20 + repeat
            capture_offset.fill_(self.rank + offset)
            graph.replay()
            expected_sum = float(
                self.world_size * offset + self.world_size * (self.world_size - 1) / 2
            )
            self.assertEqual(out, torch.full_like(out, expected_sum))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_put(self):
        # [FUNCTION DOC] Rank 1 calls nccl_put_with_signal into rank 0 with signal value 5
        # while rank 0 waits for the signal and expects rank 1's tensor (ones); then rank 1
        # doubles its tensor and does a plain nccl_put and rank 0 expects twos. Barriers
        # separate the phases.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        signal_val = 5
        c10d.barrier()

        if self.rank == 1:
            torch.ops.symm_mem.nccl_put_with_signal(tensor, signal_val, 0)
        elif self.rank == 0:
            torch.ops.symm_mem.nccl_wait_for_signal(tensor, signal_val)
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        c10d.barrier()
        if self.rank == 1:
            tensor *= 2
            torch.ops.symm_mem.nccl_put(tensor, 0)
            c10d.barrier()
        else:
            c10d.barrier()
        if self.rank == 0:
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device) * 2
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_handle_signal(self):
        # [FUNCTION DOC] Pairs ranks: odd ranks put_signal to the previous even rank and
        # synchronize, even ranks with an odd successor wait_signal for it; passes if no hang
        # or error occurs; barriers before and after.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        handle = symm_mem.rendezvous(tensor, group=group_name)

        channel = 0
        world_size = handle.world_size

        c10d.barrier()

        # Pair up ranks: odd ranks send to even ranks
        # This allows the test to work with any number of GPUs
        if self.rank % 2 == 1:
            # Odd rank: send signal to previous even rank
            dst_rank = self.rank - 1
            handle.put_signal(dst_rank=dst_rank, channel=channel)
            torch.cuda.synchronize()
        elif self.rank % 2 == 0 and self.rank + 1 < world_size:
            # Even rank: wait for signal from next odd rank (if it exists)
            src_rank = self.rank + 1
            # wait_signal blocks until the signal arrives
            # If this completes without hanging, the test passes
            handle.wait_signal(src_rank=src_rank, channel=channel)
            torch.cuda.synchronize()

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_get(self):
        # [FUNCTION DOC] Rank 0 does nccl_get from rank 1 into its tensor (after barriers) and
        # expects ones (the data from rank 1); other ranks only barrier.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        c10d.barrier()
        if self.rank == 0:
            torch.ops.symm_mem.nccl_get(tensor, 1)
            # TODO: remove after we have wait_signal
            c10d.barrier()
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        else:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory device API support from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_get(self):
        # [FUNCTION DOC] Tests symm_mem.get: rank 0 copies the whole buffer from peer 1; then
        # copies with an element offset into a sub-region, fills only a view leaving the tail
        # untouched, and checks ValueError for a non-contiguous destination, negative offset
        # (non-negative message), offset past the buffer (exceeds) and an invalid peer.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        # Full-buffer get from a peer's allocation.
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            dst = torch.empty_like(src)
            symm_mem.get(dst, hdl, peer=1)
            torch.testing.assert_close(dst, torch.ones_like(dst))

        c10d.barrier()

        # Offset get: copy a sub-region of the peer's allocation.
        src_base = symm_mem.empty(2 * numel, dtype=dtype, device=self.device)
        src_base.copy_(
            torch.arange(2 * numel, dtype=dtype, device=self.device)
            + self.rank * 2 * numel
        )
        hdl = symm_mem.rendezvous(src_base, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            offset = numel // 2
            dst = torch.empty(numel, dtype=dtype, device=self.device)
            symm_mem.get(dst, hdl, peer=1, offset=offset)
            expected = (
                torch.arange(offset, offset + numel, dtype=dtype, device=self.device)
                + 2 * numel
            )
            torch.testing.assert_close(dst, expected)

            # Filling a sub-region: pass a view; the rest of dst is untouched.
            larger_dst = torch.full((numel + 1,), -1, dtype=dtype, device=self.device)
            symm_mem.get(larger_dst[:numel], hdl, peer=1, offset=offset)
            self.assertEqual(larger_dst[:numel], expected)
            self.assertEqual(larger_dst[numel], -1)

            noncontig_dst = torch.empty(2 * numel, dtype=dtype, device=self.device)[::2]
            with self.assertRaisesRegex(ValueError, "contiguous"):
                symm_mem.get(noncontig_dst, hdl, peer=1)

            with self.assertRaisesRegex(ValueError, "non-negative"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=-1,
                )

            with self.assertRaisesRegex(ValueError, "exceeds"):
                symm_mem.get(
                    torch.empty(1, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=hdl.buffer_size // dst.element_size(),
                )

            with self.assertRaisesRegex(ValueError, "invalid peer"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=hdl.world_size,
                )

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("experts_per_rank", [1, 2])
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset(self, experts_per_rank: int, dim: int):
        # [FUNCTION DOC] Parametrized over experts per rank (1, 2) and the split dimension (0,
        # 1). Fills a symmetric buffer of expert blocks with values based on rank and expert
        # index, calls reduce_scatter_offset with round-robin destination ranks and per-block
        # end offsets, and checks each owned output equals the reduced sum for that expert and
        # the source buffer is unchanged.
        """reduce_scatter_offset: each expert gradient is reduced to its
        destination rank and written to a separate contiguous tensor; the source
        Grouped GEMM buffer is left unmodified."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        n_experts = experts_per_rank * self.world_size

        # dim=1: experts laid out as column blocks [rows, n_experts * cols]
        # dim=0: experts laid out as row blocks    [n_experts * rows, cols]
        if dim == 1:
            buf = symm_mem.empty(
                rows, n_experts * cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[:, i * cols : (i + 1) * cols] = float((self.rank + 1) * (i + 1))
        else:
            buf = symm_mem.empty(
                n_experts * rows, cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[i * rows : (i + 1) * rows, :] = float((self.rank + 1) * (i + 1))
        symm_mem.rendezvous(buf, group=group_name)

        # Round-robin: expert i is reduced to rank i % world_size.
        dst_ranks = [i % self.world_size for i in range(n_experts)]
        n_owned = sum(r == self.rank for r in dst_ranks)
        out = [
            torch.zeros(rows, cols, dtype=torch.float, device=self.device)
            for _ in range(n_owned)
        ]
        block_size = cols if dim == 1 else rows
        offsets = [i * block_size for i in range(1, n_experts + 1)]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        # out[j] corresponds to expert (rank + j * world_size); expected value is
        # (expert_idx + 1) * sum(r + 1 for r in range(world_size)).
        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(n_owned):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )
        # Source buffer must be unmodified.
        for i in range(n_experts):
            if dim == 1:
                src_slice = buf[:, i * cols : (i + 1) * cols]
            else:
                src_slice = buf[i * rows : (i + 1) * rows, :]
            self.assertEqual(
                src_slice,
                torch.full(
                    (rows, cols),
                    float((self.rank + 1) * (i + 1)),
                    dtype=torch.float,
                    device=self.device,
                ),
                msg=lambda msg: f"{msg}\nrank {self.rank}: source buffer block {i} should be unchanged",
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset_uneven(self, dim: int):
        # [FUNCTION DOC] Parametrized over dim. Like the previous test but with uneven blocks
        # (16 and 48 along the dimension) so each rank owns two outputs of different shapes;
        # verifies each output holds the reduced sum for its expert.
        """reduce_scatter_offset with uneven block sizes: j=0 and j=1 own blocks
        of different sizes, verifying that out[j] shapes differ across j."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        # j=0 blocks have size_0 along dim; j=1 blocks have size_1 along dim.
        # Arrange blocks as [size_0] * world_size + [size_1] * world_size so
        # that round-robin assigns each rank exactly one block of each size.
        size_0, size_1 = 16, 48
        block_sizes = [size_0] * self.world_size + [size_1] * self.world_size
        offsets = []
        total = 0
        for sz in block_sizes:
            total += sz
            offsets.append(total)

        n_experts = 2 * self.world_size
        if dim == 1:
            buf = symm_mem.empty(rows, total, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[:, pos : pos + sz] = float((self.rank + 1) * (i + 1))
                pos += sz
        else:
            buf = symm_mem.empty(total, cols, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[pos : pos + sz, :] = float((self.rank + 1) * (i + 1))
                pos += sz
        symm_mem.rendezvous(buf, group=group_name)

        dst_ranks = [i % self.world_size for i in range(n_experts)]
        if dim == 1:
            out = [
                torch.zeros(rows, size_0, dtype=torch.float, device=self.device),
                torch.zeros(rows, size_1, dtype=torch.float, device=self.device),
            ]
        else:
            out = [
                torch.zeros(size_0, cols, dtype=torch.float, device=self.device),
                torch.zeros(size_1, cols, dtype=torch.float, device=self.device),
            ]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(2):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 28, 0), "nccl_all_to_all_nd requires nccl 2.28")
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "scatter_gather,out_2d,input_3d",
        [
            ((1, 0), False, False),
            ((1, 0), True, False),
            ((1, 0), False, True),
            ((1, 0), True, True),
            ((0, 1), False, False),
            ((0, 1), True, False),
            ((0, 1), False, True),
            ((0, 1), True, True),
        ],
    )
    def test_all_to_all_nd(self, scatter_gather, out_2d, input_3d):
        # [FUNCTION DOC] Parametrized over (scatter_dim, gather_dim) in (1,0) or (0,1), 2-D
        # output and 3-D input flags. Fills a symmetric buffer with the rank, runs
        # all_to_all_nd into an output buffer and checks each peer j's block equals j, using a
        # column-block layout for (1,0) and a row-block layout otherwise.
        """all_to_all_nd: (1,0)/(0,1); 3-D input [rows,G,loc] or [G,loc,cols] where supported."""
        scatter_dim, gather_dim = scatter_gather
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        p = self.world_size
        dtype = torch.float

        if scatter_dim == 1 and gather_dim == 0:
            local_cols = 4
            rows = 8
            if input_3d:
                buf = symm_mem.empty(
                    rows, p, local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    rows, p * local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(p * rows, local_cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(p, rows, local_cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(p, rows, local_cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[j],
                    torch.full(
                        (rows, local_cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[{j}] should be peer {j}'s column block",
                )
        else:
            local_rows = 4
            cols = 4
            if input_3d:
                buf = symm_mem.empty(
                    p, local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    p * local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(local_rows, p * cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(local_rows, p, cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(local_rows, p, cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[:, j, :],
                    torch.full(
                        (local_rows, cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[:, {j}, :] should be peer {j}'s row block",
                )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_put_wait_signal(self):
        # [FUNCTION DOC] Pairs ranks: odd ranks put_signal their source buffer into the
        # previous even rank's destination buffer; even ranks wait_signal and check the
        # destination now holds the sender's rank value; barrier at the end.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Use this barrier to make sure all ranks are initialized.
        c10d.barrier()
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        hdl = symm_mem.rendezvous(dst, group=group_name)

        # Pair ranks: odd ranks send to previous even ranks.
        if self.rank % 2 == 1:
            dst_rank = self.rank - 1
            symm_mem.put_signal(src, hdl, dst_rank)
        elif self.rank % 2 == 0 and self.rank + 1 < self.world_size:
            src_rank = self.rank + 1
            symm_mem.wait_signal(hdl, src_rank)
            self.assertEqual(dst, torch.full_like(dst, float(src_rank)))

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_mempool_tensor_factory(self):
        # [FUNCTION DOC] Creates a tensor inside the symmetric memory pool (mem pool tensor
        # factory), rendezvouses it and checks a one-shot all_reduce returns the arange
        # times world size.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            tensor = torch.arange(numel, dtype=dtype, device=self.device)

        # Rendezvous should not error out
        symm_mem.rendezvous(tensor, group=group_name)
        tensor = torch.ops.symm_mem.one_shot_all_reduce(tensor, "sum", group_name)
        expected = (
            torch.arange(numel, dtype=dtype, device=self.device) * self.world_size
        )
        self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_mempool_compute_ops(self):
        # [FUNCTION DOC] Computes a matrix multiply inside the symmetric memory pool,
        # then checks a one-shot all_reduce of the result equals the plain product
        # times world size.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        dim = 1024
        w = torch.ones(dim, dim, dtype=dtype, device=self.device)
        x = torch.ones(1, dim, dtype=dtype, device=self.device)

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            y = torch.mm(x, w)

        # One-shot all-reduce should not error out
        y = torch.ops.symm_mem.one_shot_all_reduce(y, "sum", group_name)
        expected = torch.mm(x, w) * self.world_size
        self.assertEqual(y, expected)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_mempool_recycled_barrier(self):
        # [FUNCTION DOC] Regression test: allocates from the memory pool, does a bounded-
        # timeout barrier and peer buffer check, frees, allocates again so the block is
        # recycled, and repeats the round-trip, ensuring a recycled signal pad does not
        # deadlock the barrier.
        # Regression test for the signal-pad pollution bug: ncclMemAlloc does
        # not zero memory, so a MemPool-recycled NCCL symmetric allocation could
        # start the CAS-based barrier() protocol from a non-zero signal pad and
        # deadlock. alloc() zeros the pad up front. Allocate from the SymmMem
        # MemPool, run a barrier / buffer round-trip, free, then allocate the
        # same size again (recycling the freed block) and confirm the round-trip
        # still works on the recycled region.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        mempool = symm_mem.get_mem_pool(self.device)
        numel, dtype = 1024, torch.float

        def barrier_roundtrip():
            # [FUNCTION DOC] HELPER: Allocates a tensor from the pool, rendezvouses, fills
            # with the rank, barriers with a 60 s timeout, verifies each peer's buffer holds
            # the peer rank, barriers again and returns the tensor and handle.
            with torch.cuda.use_mem_pool(mempool):
                t = torch.empty(numel, dtype=dtype, device=self.device)
            hdl = symm_mem.rendezvous(t, group=group_name)
            t.fill_(self.rank)
            # Bounded barriers so a polluted-pad regression fails cleanly
            # instead of hanging.
            hdl.barrier(timeout_ms=60000)
            for peer in range(self.world_size):
                buf = hdl.get_buffer(peer, (numel,), dtype)
                self.assertTrue(buf.eq(peer).all())
            hdl.barrier(timeout_ms=60000)
            return t, hdl

        t1, hdl1 = barrier_roundtrip()
        del hdl1, t1
        t2, hdl2 = barrier_roundtrip()
        del hdl2, t2

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("NCCL_NVLS_ENABLE", "1") == "0",
        "NCCL_NVLS_ENABLE=0",
    )
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version(
        (2, 29), "NCCL Symmetric Memory multicast support from nccl 2.29"
    )
    def test_multicast_ptr(self) -> None:
        # [FUNCTION DOC] Rendezvouses a one-element symmetric tensor; if the device supports
        # multicast the handle's multicast pointer is nonzero, otherwise zero.
        """
        Get the multicast pointer
        """
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensor = symm_mem.empty(1, device=self.device)
        handle = symm_mem.rendezvous(tensor, group_name)
        if _SymmetricMemory.has_multicast_support(DeviceType.CUDA, self.device.index):
            self.assertNotEqual(handle.multicast_ptr, 0)
        else:
            self.assertEqual(handle.multicast_ptr, 0)


@requires_cuda_p2p_access()
class NCCLSymmetricMemoryNccl2Test(MultiProcContinuousTest):
    # [CLASS DOC] Runs the symmetric memory rendezvous against a process group using
    # the nccl2 backend (name in backend_name); a subclass overrides the name. Requires
    # CUDA peer-to-peer access.
    """NCCL symmetric memory over an nccl2-backed process group.

    Same flow as NCCLSymmetricMemoryTest, but the process group uses the in-tree
    torchcomms "nccl2" backend. Regression test for nccl2 publishing its host
    ncclComm_t into NCCLDevCommManager (via the comm-registration hook) so that
    NCCLSymmetricMemory can resolve it by group name -- before that, rendezvous
    on an nccl2 group raised "NCCL host communicator for group ... not found".
    """

    backend_name = "nccl2"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Property: the CUDA device for this process's rank.
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # [FUNCTION DOC] HELPER: Classmethod run in each worker: same setup as the base
        # symmetric memory test but initializes the backend named by backend_name; raises
        # AssertionError if the rendezvous file path is None.
        # Eager communicator init via device_id, mirroring
        # NCCLSymmetricMemoryTest.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend=cls.backend_name,
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous(self):
        # [FUNCTION DOC] Checks the default group backend and symmetric memory backend
        # names, publishes the communicator with an all_reduce, rendezvouses a rank-filled
        # tensor (checking rank and world size) and verifies a one-shot all_reduce equals
        # the sum of ranks.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Confirm the intended path: a torchcomms PG + the NCCL symm-mem backend.
        self.assertEqual(c10d.get_backend(c10d.group.WORLD), self.backend_name)
        self.assertEqual(symm_mem.get_backend(self.device), "NCCL")
        # Publish the communicator before rendezvous looks it up.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float, device=self.device).fill_(self.rank)
        handle = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(handle.rank, self.rank)
        self.assertEqual(handle.world_size, self.world_size)

        # Exercise an actual symm-mem collective end to end.
        result = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        self.assertEqual(
            result, torch.full_like(result, (self.world_size - 1) * self.world_size / 2)
        )


class NCCLSymmetricMemoryNcclLazyTest(NCCLSymmetricMemoryNccl2Test):
    # [CLASS DOC] Same tests as the nccl2 variant, but with the backend name set to nccl-lazy.
    backend_name = "nccl-lazy"


instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
