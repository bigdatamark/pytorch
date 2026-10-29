# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_nvshmem.py
# [MODULE DOC] Multi-process GPU tests for the NVSHMEM backend of symmetric memory.
# Covers allocation and rendezvous, MemPool-backed tensors, remote tensor access, put/get
# and signaling ops, all-to-all variants (fixed, variable, 2D, 2D with offsets),
# dispatch-then-combine round trips (world group and subgroups), and tile reductions.
# Defines skip decorators (NVSHMEM, NVLS, CUDA 13.2) and sets module-level device_type to
# cuda; importing has no side effects beyond that. Runs via run_tests when executed.


import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
)


# Decorator
def requires_nvshmem():
    # [FUNCTION DOC] HELPER: Returns a skip decorator (pass-in-sandcastle variant) that
    # skips when the platform lacks symmetric memory support or NVSHMEM is unavailable.
    return skip_but_pass_in_sandcastle_if(
        not PLATFORM_SUPPORTS_SYMM_MEM or not symm_mem.is_nvshmem_available(),
        "test_nvshmem requires NVSHMEM with SymmMem platform support, skipping tests",
    )


def has_nvls_support():
    # [FUNCTION DOC] HELPER: Reports whether NVLink SHARP multicast is usable. Returns
    # False if CUDA or NVSHMEM is unavailable, or if the NVSHMEM_DISABLE_NVLS environment
    # variable equals 1. Otherwise sets the symmetric memory backend to NVSHMEM (global
    # side effect) and returns whether the CUDA device 0 has multicast support.
    if not torch.cuda.is_available() or not symm_mem.is_nvshmem_available():
        return False

    if os.environ.get("NVSHMEM_DISABLE_NVLS", "0") == "1":
        return False

    # Set NVSHMEM as SymmMem backend before running the check
    symm_mem.set_backend("NVSHMEM")
    from torch._C._autograd import DeviceType
    from torch._C._distributed_c10d import _SymmetricMemory

    return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)


def requires_nvls():
    # [FUNCTION DOC] HELPER: Returns a skip decorator that skips when has_nvls_support is
    # false. Note has_nvls_support runs at decoration time and may set the NVSHMEM backend.
    """Skip test if NVLS (NVLink SHARP) is not available."""
    return skip_but_pass_in_sandcastle_if(
        not has_nvls_support(),
        "Test requires NVLink SHARP support",
    )


def skip_if_cuda_13_2():
    # [FUNCTION DOC] HELPER: Returns a skip decorator that skips when the torch CUDA
    # version is exactly 13.2, where dispatch-combine tests hang or fail.
    """Skip dispatch-combine tests that hang/fail on CUDA 13.2 (B200/sm100).

    See https://github.com/pytorch/pytorch/issues/191201
    """
    return skip_but_pass_in_sandcastle_if(
        _get_torch_cuda_version() == (13, 2),
        "Dispatch-combine hangs/fails on CUDA 13.2, "
        "see https://github.com/pytorch/pytorch/issues/191201",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process tests of NVSHMEM-backed symmetric memory basics: allocation,
    # rendezvous, MemPool tensors, handle offsets, remote tensor access, put/get ops.
    # Skipped unless NVSHMEM and CUDA P2P access are available. One process per rank.
    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and
        # selects NVSHMEM as the symmetric memory backend (global state change).
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property returning the CUDA device whose index is this rank.
        return torch.device(device_type, self.rank)

    def test_alloc(self) -> None:
        # [FUNCTION DOC] Allocates a symmetric tensor and rendezvouses on the world group,
        # both inside a nested function and at test level, checking neither raises.
        self._init_device()

        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        def foo():
            # [FUNCTION DOC] HELPER: Allocates a symmetric tensor and rendezvouses on it.
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    def test_alloc_without_device_context(self) -> None:
        # [FUNCTION DOC] Sets the NVSHMEM backend without calling set_device first, allocates
        # a symmetric tensor on this rank's device, checks its device, and rendezvouses.
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        self.assertEqual(out.device, self.device)
        symm_mem.rendezvous(out, group=group_name)

    def test_mempool_tensor_factory(self) -> None:
        # [FUNCTION DOC] Creates a tensor under a MemPool using the symmetric memory
        # allocator (rank 0 holds arange, others zeros), rendezvouses, broadcasts from
        # rank 0 with the NVSHMEM op, and checks every rank holds arange.
        """
        Test the effectiveness of MemPool on tensor factory ops.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        src_rank = 0

        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            if self.rank == src_rank:
                tensor = torch.arange(numel, dtype=dtype, device=self.device)
            else:
                tensor = torch.zeros(numel, dtype=dtype, device=self.device)

        symm_mem.rendezvous(tensor, group=group_name)
        torch.ops.symm_mem.nvshmem_broadcast(tensor, src_rank, group_name)
        self.assertEqual(tensor, torch.arange(numel, dtype=dtype, device=self.device))

    def test_mempool_tensor_w_collective(self) -> None:
        # [FUNCTION DOC] Creates a ones tensor under a symmetric-memory MemPool, rendezvouses,
        # runs a regular all_reduce, and checks each element equals the world size.
        # KNOWN ISSUE: the docstring is copied from the factory test and says "tensor
        # factory ops" although this test exercises a collective.
        """
        Test the effectiveness of MemPool on tensor factory ops.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            tensor = torch.ones(numel, dtype=dtype, device=self.device)

        symm_mem.rendezvous(tensor, group=group_name)
        dist.all_reduce(tensor)
        self.assertEqual(
            tensor, torch.ones(numel, dtype=dtype, device=self.device) * self.world_size
        )

    def test_mempool_compute_ops(self) -> None:
        # [FUNCTION DOC] Computes an add and a matmul under a symmetric-memory MemPool so the
        # result is a symmetric tensor, broadcasts it from rank 0 with the NVSHMEM op
        # (without an explicit rendezvous), and checks it equals the rank-0 result.
        """
        Apply MemPool context to a compute op that creates input to collective.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        dim = 1024
        w = torch.ones(dim, dim, dtype=dtype, device=self.device)
        x0 = torch.ones(1, dim, dtype=dtype, device=self.device)

        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            x = x0 + self.rank
            y = torch.mm(x, w)

        # y should be a symm tensor
        torch.ops.symm_mem.nvshmem_broadcast(y, 0, group_name)
        expected = torch.mm(x0, w)
        self.assertEqual(y, expected)

    def test_handle_offset(self) -> None:
        # [FUNCTION DOC] Allocates two tensors consecutively in one MemPool and checks the
        # rendezvous handles report offset 0 for the first and the first tensor's storage
        # byte size for the second.
        """
        Test if handle offset is correctly set.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            x0 = torch.empty(numel, dtype=dtype, device=self.device)
            x1 = torch.empty_like(x0)

        hdl0 = symm_mem.rendezvous(x0, group=group_name)
        hdl1 = symm_mem.rendezvous(x1, group=group_name)
        self.assertEqual(hdl0.offset, 0)
        self.assertEqual(hdl1.offset, x0.untyped_storage().nbytes())

    def test_get_remote_tensor(self) -> None:
        # [FUNCTION DOC] Each rank gets a tensor view of the next rank's buffer through the
        # handle, copies its own rank-filled data into it with a plain copy, barriers, and
        # checks its own buffer holds the previous rank's id (modulo world size).
        """
        Get a remote tensor and use regular aten ops to write to it.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            # src data stores my rank
            x = torch.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
            y = torch.empty_like(x)

        hdl_y = symm_mem.rendezvous(y, group=group_name)
        peer = (self.rank + 1) % self.world_size  # Shifting pattern
        y_remote = hdl_y.get_remote_tensor(peer, y.size(), y.dtype)
        y_remote.copy_(x)
        dist.barrier()
        # Expecting data from -1 rank
        expected = torch.empty(numel, dtype=dtype, device=self.device).fill_(
            (self.rank - 1) % self.world_size
        )
        self.assertEqual(y, expected)

    def test_get_remote_tensors(self) -> None:
        # [FUNCTION DOC] Fills a symmetric tensor with the rank, fetches the per-peer remote
        # tensors via the get_remote_tensors op, barriers, and checks the tensor from peer
        # i holds the value i.
        """
        Get all remote tensors
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        my_tensor = symm_mem.empty(1, device=self.device).fill_(self.rank)
        remote_tensors = torch.ops.symm_mem.get_remote_tensors(my_tensor, group_name)
        dist.barrier()

        for peer, tensor in enumerate(remote_tensors):
            self.assertEqual(tensor, peer)

    def test_multicast_ptr(self) -> None:
        # [FUNCTION DOC] Rendezvouses a symmetric tensor and checks the handle's multicast
        # pointer is nonzero when NVLS is supported and zero otherwise.
        """
        Get the multicast pointer
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        tensor = symm_mem.empty(1, device=self.device)
        handle = symm_mem.rendezvous(tensor, group_name)
        if has_nvls_support():
            self.assertNotEqual(handle.multicast_ptr, 0)
        else:
            self.assertEqual(handle.multicast_ptr, 0)

    def test_nvshmem_put(self) -> None:
        # [FUNCTION DOC] Rank 0 puts its tensor to rank 1 with a signal value 5 on the signal
        # pad; rank 1 waits for that signal then checks its tensor equals zeros (rank 0's
        # data, since each tensor is filled with its rank). Other ranks do nothing further.
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        signal_pad = hdl.get_signal_pad(self.rank)
        signal_val = 5

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_put_with_signal(
                tensor, signal_pad, signal_val, 1
            )
        elif self.rank == 1:
            torch.ops.symm_mem.nvshmem_wait_for_signal(signal_pad, signal_val, 0)
            torch.testing.assert_close(
                tensor, torch.zeros(numel, dtype=dtype, device=self.device)
            )

    def test_nvshmem_get(self) -> None:
        # [FUNCTION DOC] Rank 0 gets rank 1's tensor (filled with ones) into its own with the
        # NVSHMEM get op, barriers, and checks it equals ones; other ranks just barrier.
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_get(tensor, 1)
            # TODO: remove after we have wait_signal
            dist.barrier()
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        else:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            dist.barrier()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM, "nvshmem_get_out not yet implemented for ROCm"
    )
    def test_get(self) -> None:
        # [FUNCTION DOC] Tests symm_mem.get (skipped on ROCm). Rank 0 gets a full buffer from
        # peer 1, then a sub-region at an element offset, then fills a view of a larger
        # destination leaving the rest untouched. Also checks ValueError for a
        # non-contiguous destination, negative offset, offset past the buffer, bad peer.
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        # Full-buffer get from a peer's allocation.
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        dist.barrier()

        if self.rank == 0:
            dst = torch.empty_like(src)
            symm_mem.get(dst, hdl, peer=1)
            torch.testing.assert_close(dst, torch.ones_like(dst))

        dist.barrier()

        # Offset get: copy a sub-region of the peer's allocation.
        src_base = symm_mem.empty(2 * numel, dtype=dtype, device=self.device)
        src_base.copy_(
            torch.arange(2 * numel, dtype=dtype, device=self.device)
            + self.rank * 2 * numel
        )
        hdl = symm_mem.rendezvous(src_base, group=group_name)
        dist.barrier()

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

        dist.barrier()


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMAll2AllTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process tests of the NVSHMEM all-to-all ops: fixed-size, variable
    # (vdev), 2D expert-major (with major alignment), and 2D with input offsets.
    # Skipped unless NVSHMEM and CUDA P2P access are available. One process per rank.
    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and
        # selects NVSHMEM as the symmetric memory backend (global state change).
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property returning the CUDA device whose index is this rank.
        return torch.device(device_type, self.rank)

    def test_nvshmem_all_to_all(self) -> None:
        # [FUNCTION DOC] Each rank fills an input with its rank, runs the NVSHMEM all_to_all
        # op, and checks the output has numel_per_peer elements from each rank in rank order.
        self._init_device()

        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel_per_peer = 10
        numel = self.world_size * numel_per_peer
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        torch.ops.symm_mem.nvshmem_all_to_all(inp, out, group_name)

        expected = torch.cat(
            [
                torch.empty(numel_per_peer, dtype=dtype, device=self.device).fill_(i)
                for i in range(self.world_size)
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_all_to_all_vdev(self) -> None:
        # [FUNCTION DOC] Uses random per-peer input splits, derives expected output splits
        # via all_to_all_single, runs all_to_all_vdev on symmetric buffers sized for the worst
        # case, and checks input splits unchanged, output splits, exclusive-scan output
        # offsets, and received data against a regular all_to_all_single.
        self._init_device()

        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        # Number of elements for a peer is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (self.world_size,), device=self.device)
        inp_numel = inp_splits.sum().item()
        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        out_numel = out_splits.sum().item()

        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * self.world_size
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        in_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        out_splits_offsets = symm_mem.empty(
            (2, self.world_size), dtype=torch.int64, device=self.device
        )
        # Row 0 is input splits
        in_splits.copy_(inp_splits)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev(
            inp, out, in_splits, out_splits_offsets, group_name
        )

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_splits, inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(out_splits_offsets[0], out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `all_to_all_vdev` is exclusive scan
        self.assertEqual(out_splits_offsets[1][0], 0)
        torch.testing.assert_close(out_splits_offsets[1][1:], out_offsets[:-1])

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        dist.all_to_all_single(
            expected, inp[:inp_numel], out_splits.tolist(), inp_splits.tolist()
        )
        torch.testing.assert_close(out[:out_numel], expected)

    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_all_to_all_vdev_2d(self, align: int) -> None:
        # [FUNCTION DOC] Parametrized by major_align. With 8 experts per rank and random
        # splits, runs all_to_all_vdev_2d and checks input splits unchanged, output splits
        # in expert-major order, offsets as an exclusive scan with each expert's total padded
        # up to the alignment (minimum one block, padding absorbed by the last split), and
        # each received chunk against a reference built from all_to_all_single.
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        # Number of experts per rank
        ne = 8
        nsplits = ne * self.world_size

        # Number of elements for an expert is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)

        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        # We do a .t() here because there is a rank-major to expert-major shuffle
        out_splits_t = out_splits.reshape(self.world_size, ne).t()

        # Actual number of input elements
        inp_numel = inp_splits.sum().item()
        # Actual number of output elements
        out_numel = out_splits.sum().item()
        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        in_splits = symm_mem.empty(
            nsplits, dtype=torch.int64, device=self.device
        ).copy_(inp_splits)
        # 2 rows: output splits, output offsets
        # Initializing all values to -1 to check if they are updated
        out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp, out, in_splits, out_splits_offsets, group_name, major_align=align
        )
        received_out_splits = out_splits_offsets[0]
        received_out_offsets = out_splits_offsets[1]

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_splits, inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(received_out_splits, out_splits_t.reshape(-1))

        # Check output offsets (row 2)
        out_split_list = out_splits_t.tolist()
        for i in range(ne):
            expert_sum = 0
            for j in range(self.world_size):
                expert_sum += out_split_list[i][j]
            # Align up expert_sum
            expert_sum_aligned = (expert_sum + align - 1) // align * align
            # If 0, make it at least `align` (bc cutlass currently does not support empty bins)
            expert_sum_aligned = max(expert_sum_aligned, align)
            # last element absorbs the padding
            out_split_list[i][-1] += expert_sum_aligned - expert_sum

        out_splits_padded = torch.tensor(out_split_list, device=self.device).reshape(-1)
        out_offsets = torch.cumsum(out_splits_padded, dim=0)  # inclusive scan
        # Make it exclusive scan because that's what `all_to_all_vdev_2d` returns
        out_offsets = torch.cat(
            [torch.zeros(1, device=self.device), out_offsets[:-1]]
        ).to(torch.int64)
        torch.testing.assert_close(received_out_offsets, out_offsets)

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        inp_splits_rank = inp_splits.reshape(self.world_size, ne).sum(1)
        out_splits_rank = out_splits.reshape(self.world_size, ne).sum(1)
        dist.all_to_all_single(
            expected,
            inp[:inp_numel],
            out_splits_rank.tolist(),
            inp_splits_rank.tolist(),
        )
        # We still need to shuffle `expected`
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        result_list = []
        for j in range(ne):
            for i in range(self.world_size):
                chunk_id = i * ne + j
                offset = out_offsets[chunk_id]
                chunk = expected[offset - out_splits[chunk_id] : offset]
                result_list.append(chunk)

        # Do a chunk-wise comparison
        for c, chunk in enumerate(result_list):
            start = received_out_offsets[c].item()
            split = received_out_splits[c].item()
            received_chunk = out[start : start + split]
            torch.testing.assert_close(received_chunk, chunk)

    def test_all_to_all_vdev_2d_offset(self) -> None:
        # [FUNCTION DOC] Runs all_to_all_vdev_2d_offset with inputs laid out at padded fixed
        # offsets. Checks input splits and offsets unchanged, output splits (from a
        # transposed all_to_all_single), exclusive-scan output offsets, and the received data
        # against a reference built by stripping padding and using dist.all_to_all.
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        # Number of experts per rank
        ne = 8
        nsplits = ne * self.world_size

        # Number of elements for an expert is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)
        # Each split up align to k, as the offset, i.e. [0, k, 2k, 3k, ...]
        inp_offsets = torch.arange(
            0, k * nsplits, k, dtype=torch.int64, device=self.device
        )

        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        # Remember that we up-align each input split to k?
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        # 2 rows: input splits, input offsets
        in_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        )
        # 2 rows: output splits, output offsets
        # Initializing all values to -1 to check if they are updated
        out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)

        # Row 0 is input splits
        in_splits_offsets[0].copy_(inp_splits)
        # Row 1 is input offsets
        in_splits_offsets[1].copy_(inp_offsets)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            inp, out, in_splits_offsets, out_splits_offsets, group_name
        )
        received_out_splits = out_splits_offsets[0]
        received_out_offsets = out_splits_offsets[1]

        # Check input splits and offsets -- should not change
        torch.testing.assert_close(in_splits_offsets[0], inp_splits)
        torch.testing.assert_close(in_splits_offsets[1], inp_offsets)

        # Check output splits (row 1)
        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        # First need to transpose the input splits
        inp_splits_t = inp_splits.reshape(ne, self.world_size).t().contiguous()
        dist.all_to_all_single(out_splits, inp_splits_t)
        torch.testing.assert_close(received_out_splits, out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `all_to_all_vdev_2d_offset` is exclusive scan
        self.assertEqual(received_out_offsets[0], 0)
        torch.testing.assert_close(received_out_offsets[1:], out_offsets[:-1])

        # Check data
        # Let's "squeeze" the padding out of the input data first
        inp_chunks = []  # (ne, nranks)
        for i in range(ne):
            inp_chunks_e = []  # (nranks,)
            for j in range(self.world_size):
                chunk_id = i * self.world_size + j
                offset = in_splits_offsets[1][chunk_id]
                chunk = inp[offset : offset + inp_splits[chunk_id]]
                inp_chunks_e.append(chunk)
            inp_chunks.append(inp_chunks_e)

        # Transpose the 2D input chunks
        inp_chunks_t = list(zip(*inp_chunks))
        # Now it is (nranks, ne), concatenate the e's
        inp_chunks_t = [torch.cat(row) for row in inp_chunks_t]

        # Create empty output tensors -- each tensor is data to be received from a peer
        out_splits = out_splits.reshape(self.world_size, ne)
        # Sum the split sizes of all experts, per peer
        receive_size_per_peer = out_splits.sum(1)
        out_chunks = []  # (nranks,)
        for i in range(self.world_size):
            out_chunks.append(
                torch.empty(
                    receive_size_per_peer[i].item(), dtype=dtype, device=self.device
                )
            )

        # All-to-all
        dist.all_to_all(out_chunks, inp_chunks_t)

        # Concatenate the output chunks received from all peers
        out_expected = torch.cat(out_chunks)
        # Actual number of output elements
        out_numel = out_splits.sum().item()
        self.assertEqual(out_expected.shape[0], out_numel)

        # Check data
        torch.testing.assert_close(out_expected, out[:out_numel])


# Help function used by multiple tests
def dispatch_then_combine(device, align: int, group) -> None:
    # [FUNCTION DOC] HELPER: Shared body for dispatch-combine tests. On the given group,
    # allocates symmetric buffers, shuffles random tokens with all_to_all_vdev_2d (major
    # alignment align), then combines them back with all_to_all_vdev_2d_offset using the
    # shuffle's splits/offsets. Asserts combined data, splits and exclusive-scan offsets
    # equal the originals. Synchronizes CUDA and barriers on the group before and after.
    """
    Shuffle the tokens, then combine them, and check if the combined data is
    exactly the same as the original input data
    """
    group_name = group.group_name

    dtype = torch.float
    # Number of experts per rank
    ne = 8
    nsplits = ne * group.size()

    # Number of elements for an expert is random between [0, k)
    k = 10
    inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=device)

    # Actual number of input elements
    inp_numel = inp_splits.sum().item()
    # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
    max_inp_numel = k * nsplits
    # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
    overflow_factor = group.size()  # worst case: one rank receives all data
    max_out_numel = max_inp_numel * overflow_factor

    # Buffers for shuffle
    inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=device).copy_(
        torch.randn(max_inp_numel, dtype=dtype, device=device)
    )
    out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
    in_splits = symm_mem.empty(nsplits, dtype=torch.int64, device=device).copy_(
        inp_splits
    )
    # 2 rows: output splits, output offsets
    # Initializing all values to -1 to check if they are updated
    out_splits_offsets = symm_mem.empty(
        (2, nsplits), dtype=torch.int64, device=device
    ).fill_(-1)

    # Buffers for combine
    combine_out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
    # 2 rows: output splits, output offsets
    # Initializing all values to -1 to check if they are updated
    combine_out_splits_offsets = symm_mem.empty(
        (2, nsplits), dtype=torch.int64, device=device
    ).fill_(-1)

    # Wait for all ranks to finish tensor allocation before accessing them
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # Shuffle the tokens
    torch.ops.symm_mem.all_to_all_vdev_2d(
        inp, out, in_splits, out_splits_offsets, group_name, major_align=align
    )

    # Combine the tokens
    # `out_splits_offsets` from shuffle is exactly the `input_splits_offsets` for combine
    # `out` data from shuffle is exactly the `input` data for combine
    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
        out, combine_out, out_splits_offsets, combine_out_splits_offsets, group_name
    )

    # Assert the combined data is exactly the same as the original input data
    torch.testing.assert_close(combine_out[:inp_numel], inp[:inp_numel])

    # Assert the combined out splits are exactly the same as the original input splits
    torch.testing.assert_close(combine_out_splits_offsets[0], inp_splits)

    # Assert the combined out offsets are exactly the same as the original input offsets
    inp_offsets = torch.cumsum(inp_splits, dim=0)  # inclusive scan
    # Make it exclusive scan because that's what `all_to_all_vdev_2d_offset` returns
    inp_offsets = torch.cat([torch.zeros(1, device=device), inp_offsets[:-1]]).to(
        torch.int64
    )
    torch.testing.assert_close(combine_out_splits_offsets[1], inp_offsets)

    # Wait for all ranks to finish accessing tensors before freeing them
    dist.barrier(group=group)
    torch.cuda.synchronize(device)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class DispatchCombineTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test running dispatch-then-combine over the world group,
    # parametrized by output alignment. Skipped without NVSHMEM/P2P or on CUDA 13.2.
    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and
        # selects NVSHMEM as the symmetric memory backend (global state change).
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property returning the CUDA device whose index is this rank.
        return torch.device(device_type, self.rank)

    @skip_if_cuda_13_2()
    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_dispatch_combine(self, align: int) -> None:
        # [FUNCTION DOC] Seeds the RNG by rank, initializes the device, and runs the
        # dispatch-then-combine round trip over the world group with the given alignment.
        """
        Test dispatch-and-combine over World group
        """
        torch.manual_seed(42 + self.rank)
        self._init_device()
        dispatch_then_combine(self.device, align, dist.group.WORLD)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class DispatchCombineInSubgroups(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test running dispatch-then-combine concurrently on subgroups
    # of a device mesh. Needs at least 4 GPUs; skipped on CUDA 13.2.
    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and
        # selects NVSHMEM as the symmetric memory backend (global state change).
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property returning the CUDA device whose index is this rank.
        return torch.device(device_type, self.rank)

    @skip_if_cuda_13_2()
    @skip_if_lt_x_gpu(4)
    def test_dispatch_combine_subgroup(self) -> None:
        # [FUNCTION DOC] Builds a 2 x (world/2) device mesh named dp and ep, takes the ep
        # subgroup of this rank, and runs the dispatch-then-combine round trip on it with
        # alignment 8.
        """
        Test dispatch-and-combine over concurrent subgroups
        """
        torch.manual_seed(42 + self.rank)
        self._init_device()
        # Test on two concurrent subgroups
        ngroups = 2
        subgroup_size = self.world_size // ngroups
        dm = init_device_mesh(
            device_type, (ngroups, subgroup_size), mesh_dim_names=("dp", "ep")
        )
        subgroup = dm.get_group("ep")
        dispatch_then_combine(self.device, align=8, group=subgroup)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMTileCommTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process tests of NVSHMEM tile reduction ops (single-root and
    # multi-root), parametrized by tile size and dtype. Require NVLS support.
    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and
        # selects NVSHMEM as the symmetric memory backend (global state change).
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property returning the CUDA device whose index is this rank.
        return torch.device(device_type, self.rank)

    @requires_nvls()
    @parametrize("tile_size", [32, 128, 512])
    @parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
    def test_tile_reduce(self, tile_size: int, dtype: torch.dtype) -> None:
        # [FUNCTION DOC] Fills a 1024 square symmetric input with the rank, reduces the
        # square sub-tile at offset tile_size into the root's output tile (tile_reduce op, root 0),
        # and checks the root's tile holds the sum of ranks and everything else is zero.
        # Raises AssertionError if tile_size exceeds the full size.
        full_size = 1024
        if not (tile_size <= full_size):
            raise AssertionError(
                f"Expected tile_size <= full_size, got {tile_size} vs {full_size}"
            )

        self._init_device()
        group_name = dist.group.WORLD.group_name

        full_inp = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(self.rank)
        full_out = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(0)

        slice_ut = slice(tile_size, 2 * tile_size)
        inp_tile = full_inp[slice_ut, slice_ut]
        out_tile = full_out[slice_ut, slice_ut]

        # Reduce the tile
        root = 0
        torch.ops.symm_mem.tile_reduce(inp_tile, out_tile, root, group_name)

        # Check data
        expected = torch.zeros_like(full_out)
        expected_tile = expected[slice_ut, slice_ut]
        if self.rank == root:
            expected_tile.fill_(self.world_size * (self.world_size - 1) / 2)

        torch.testing.assert_close(full_out, expected)

    @requires_nvls()
    @parametrize("tile_size", [32, 128, 512])
    @parametrize(
        "root_ratio", [1, 2]
    )  # 1: all ranks are roots, 2: half of ranks are roots
    @parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
    def test_multi_root_tile_reduce(
        self, tile_size: int, root_ratio: int, dtype: torch.dtype
    ) -> None:
        # [FUNCTION DOC] Splits a 2048 square symmetric input into a grid of tiles (two
        # columns, world/2 rows), picks the first world/root_ratio ranks as active roots, and
        # calls multi_root_tile_reduce so each root reduces its own tile. Checks active roots'
        # tiles hold the sum of ranks and all else is zero. Raises AssertionError if the tile
        # grid does not fit in the full size.
        full_size = 2048
        num_slices_col = 2  # number of tiles on column dimension
        num_slices_row = (
            self.world_size // num_slices_col
        )  # number of tiles on row dimension
        if not (tile_size * num_slices_col <= full_size):
            raise AssertionError(
                f"tile_size * num_slices_col > full_size: "
                f"{tile_size * num_slices_col} vs {full_size}"
            )
        if not (tile_size * num_slices_row <= full_size):
            raise AssertionError(
                f"tile_size * num_slices_row > full_size: "
                f"{tile_size * num_slices_row} vs {full_size}"
            )

        self._init_device()
        group_name = dist.group.WORLD.group_name

        full_inp = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(self.rank)
        full_out = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(0)

        # Get range of each slice in terms of element indices
        slices_row = [
            slice(s * tile_size, (s + 1) * tile_size) for s in range(num_slices_row)
        ]
        slices_col = [
            slice(s * tile_size, (s + 1) * tile_size) for s in range(num_slices_col)
        ]

        # Active roots, can be a subset of all ranks
        num_active_roots = self.world_size // root_ratio
        active_roots = list(range(num_active_roots))

        # Map rank to slice indices (e.g. rank 0 -> (0, 0), rank 1 -> (0, 1), rank 2 -> (1, 0), rank 3 -> (1, 1))
        map_rank_to_slices = lambda r: (  # noqa: E731
            slices_row[r // num_slices_col],
            slices_col[r % num_slices_col],
        )
        # Populate input tiles
        input_tiles_ij = [map_rank_to_slices(r) for r in active_roots]
        input_tiles = [
            full_inp[slice_i, slice_j] for (slice_i, slice_j) in input_tiles_ij
        ]
        # My output tile (i.e. the one that I will reduce)
        out_tile_ij = map_rank_to_slices(self.rank)
        out_tile = full_out[out_tile_ij[0], out_tile_ij[1]]

        # Reduce the tiles
        torch.ops.symm_mem.multi_root_tile_reduce(
            input_tiles, out_tile, active_roots, group_name
        )

        # Check data
        expected = torch.zeros_like(full_out)
        expected_tile = expected[out_tile_ij[0], out_tile_ij[1]]
        if self.rank in active_roots:
            expected_tile.fill_(self.world_size * (self.world_size - 1) / 2)
        torch.testing.assert_close(full_out, expected)


if __name__ == "__main__":
    run_tests()
