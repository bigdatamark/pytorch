# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process NCCL tests for copy-engine (zero-CTA policy) collectives,
# which need NCCL 2.28 or newer, CUDA peer-to-peer access and symmetric-memory buffers.
# Tests run all-gather and all-to-all over symmetric tensors and compare to the regular
# kernel-based result. At import time, if c10d or NCCL is unavailable, prints a message to
# stderr and exits the process with status 0 (skipping everything).
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# NCCL Copy Engine Collectives
# Requires NCCL 2.28+ for CE collectives
# Requires Symmetric Memory allocation and registration
@requires_nccl_version((2, 28), "Need NCCL 2.28+ for CE collectives")
@requires_cuda_p2p_access()
class NCCLCopyEngineCollectives(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test case using the NCCL backend configured with the
    # zero-CTA policy so collectives run on copy engines instead of SM kernels. Provides
    # a device property, an init helper and two tests (all-gather, all-to-all). Skipped
    # unless NCCL >= 2.28 and CUDA p2p access exist.
    @classmethod
    def backend_str(cls) -> str | None:
        # [FUNCTION DOC] HELPER: Selects the nccl backend for the test process group.
        return "nccl"

    @classmethod
    def opts(cls) -> dist.ProcessGroupNCCL.Options | None:
        # [FUNCTION DOC] HELPER: Builds NCCL process group options whose config CTA policy
        # is set to zero, which enables copy-engine collectives. Returns the options.
        # Enable Zero-CTA policy for CE collectives
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        return opts

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property giving the cuda device indexed by this rank.
        return torch.device("cuda", self.rank)

    def _init(self):
        # [FUNCTION DOC] HELPER: Selects the NCCL symmetric-memory backend, binds the
        # process to its device and runs a one-element all_reduce so the NCCL communicator
        # exists (otherwise the tests hang). Creates, but does not start, a CPU+CUDA
        # profiler. Returns the world group name and the profiler.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        dist.all_reduce(torch.ones(1, device=self.device))
        group_name = dist.group.WORLD.group_name

        # Prepare a profiler
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        return group_name, prof

    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_ce_allgather(self):
        # [FUNCTION DOC] Computes a reference all_gather_single on ordinary buffers, then
        # repeats it with symmetric-memory (rendezvoused) buffers twice: asynchronously
        # with an explicit wait, and on a side stream ordered after the current stream.
        # Runs under the profiler and steps it. Asserts both copy-engine outputs equal the
        # reference. Skipped with fewer than two GPUs or on ROCm 7.14 and later.
        group_name, prof = self._init()
        dtype = torch.float
        numel = 1024 * 1024 * 32

        # Regular implementation
        inp_golden = torch.randn(numel, dtype=dtype, device=self.device)
        out_golden = torch.empty(
            numel * self.world_size, dtype=dtype, device=self.device
        )

        # Copy engine implementation
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).copy_(inp_golden)
        out = symm_mem.empty(numel * self.world_size, dtype=dtype, device=self.device)
        out2 = symm_mem.empty(numel * self.world_size, dtype=dtype, device=self.device)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)

        current_stream = torch.cuda.current_stream()
        # Create a side stream
        stream = torch.cuda.Stream()

        with prof:
            # SM
            dist.all_gather_single(out_golden, inp_golden)
            # CE + async
            work = dist.all_gather_single(out, inp, async_op=True)
            work.wait()
            # CE + side stream
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                dist.all_gather_single(out2, inp)

            prof.step()

        self.assertEqual(out, out_golden)
        self.assertEqual(out2, out_golden)

        # if self.rank == 0:
        #     prof.export_chrome_trace("test_ce_allgather.json")

    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_ce_alltoall(self):
        # [FUNCTION DOC] Same structure as the all-gather test but for all_to_all_single:
        # input and output both hold a multiple of world size elements; a reference result
        # from ordinary buffers is compared with copy-engine results from symmetric buffers
        # (async with wait, and on a side stream). Skipped with fewer than two GPUs or on
        # ROCm 7.14 and later.
        group_name, prof = self._init()
        dtype = torch.float
        numel = 1024 * 1024 * self.world_size

        # Regular implementation
        inp_golden = torch.randn(numel, dtype=dtype, device=self.device)
        out_golden = torch.empty(numel, dtype=dtype, device=self.device)

        # Copy engine implementation
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).copy_(inp_golden)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)

        current_stream = torch.cuda.current_stream()
        # Create a side stream
        stream = torch.cuda.Stream()

        with prof:
            # SM
            dist.all_to_all_single(out_golden, inp_golden)
            # CE + async
            work = dist.all_to_all_single(out, inp, async_op=True)
            work.wait()
            # CE + side stream
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                dist.all_to_all_single(out2, inp)
            prof.step()

        self.assertEqual(out, out_golden)
        self.assertEqual(out2, out_golden)


if __name__ == "__main__":
    run_tests()
