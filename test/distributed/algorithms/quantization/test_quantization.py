# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process tests for quantized collectives (auto_quantize wrapping all_gather,
# all_to_all and all_to_all_single with FP16 and BFP16 quantization types). Import-time effects:
# exits with status 0 if torch.distributed is unavailable or under dev-asan builds; reads the
# required BACKEND environment variable (KeyError if unset) and defines the test class only when
# it is gloo or nccl. Module-level fixtures setUpModule and tearDownModule disable and restore
# TF32 matmul precision. The test class also reads WORLD_SIZE from the environment.

import os
import sys

import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


_PRIOR_FP32_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] Module fixture: saves the current CUDA matmul fp32 precision setting in a
    # module global and then disables TF32 for matmul, so numerics are exact during the tests.
    global _PRIOR_FP32_PRECISION
    # Snapshot fp32_precision (not allow_tf32) so tearDownModule restores the
    # exact original; writing allow_tf32 back can't reproduce the "none" default.
    _PRIOR_FP32_PRECISION = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    # [FUNCTION DOC] Module fixture: restores the saved CUDA matmul fp32 precision, if one was
    # saved, and clears the saved value.
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        torch.backends.cuda.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    # [FUNCTION DOC] HELPER: Returns a tensor of the given shape (or length) and dtype filled
    # with value, defaulting the value to the size argument itself; moves it to the CUDA device
    # with device_id when one is given, else leaves it on the CPU.
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype).fill_(value).cuda(device_id)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = os.environ["BACKEND"]
if BACKEND == "gloo" or BACKEND == "nccl":

    class DistQuantizationTests(MultiProcessTestCase):
        # [CLASS DOC] Multi-process test case (MultiProcessTestCase) spawning world_size
        # processes, each initialising its own process group over a file store, that checks
        # quantized collectives produce the exact expected tensors for the all_gather, all_to_all
        # and all_to_all_single variants. Defined only when BACKEND is gloo or nccl.
        def setUp(self):
            # [FUNCTION DOC] Runs the base setup, spawns the worker processes, then enters (and
            # never exits) a cudnn flags context enabling cudnn and disabling TF32.
            # KNOWN ISSUE: the context manager enter is called without a matching exit, so the
            # cudnn TF32 setting stays changed afterwards.
            super().setUp()
            self._spawn_processes()
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

        def tearDown(self):
            # [FUNCTION DOC] Runs the base teardown and deletes the file used for the file store,
            # ignoring any OS error if it is already gone.
            super().tearDown()
            try:
                os.remove(self.file_name)
            except OSError:
                pass

        @property
        def op_timeout_sec(self):
            # [FUNCTION DOC] Returns 1, a timeout in seconds for operations.
            # KNOWN ISSUE: not read anywhere in this file.
            return 1

        @property
        def world_size(self):
            # [FUNCTION DOC] Returns the number of ranks from the WORLD_SIZE environment variable
            # as an integer (KeyError if unset).
            return int(os.environ["WORLD_SIZE"])

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_fp16(self):
            # [FUNCTION DOC] Initialises a gloo process group on a file store and runs the
            # all_gather check with float32 tensors and FP16 quantization over all ranks. Needs
            # gloo; skipped unless the backend is gloo.
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            group = list(range(self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16
            )

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_bfp16(self):
            # [FUNCTION DOC] Same as the FP16 all_gather test with BFP16 quantization (gloo
            # only).
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            group = list(range(self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.BFP16
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        def test_all_to_all_fp16(self):
            # [FUNCTION DOC] Initialises an nccl process group, creates a group of all ranks and
            # a rank-to-GPU map, and runs the all_to_all check on CUDA with FP16 quantization.
            # Needs nccl, at least world-size GPUs and is skipped on ROCm and for non-nccl
            # backends.
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        def test_all_to_all_bfp16(self):
            # [FUNCTION DOC] Same as the FP16 all_to_all test with BFP16 quantization.
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_single_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_all_to_all_single_fp16(self):
            # [FUNCTION DOC] Initialises an nccl process group and runs the all_to_all_single
            # check on CUDA with FP16 quantization. Needs nccl and at least world-size GPUs;
            # skipped for non-nccl backends.
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_single_bfp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_all_to_all_single_bfp16(self):
            # [FUNCTION DOC] Same as the FP16 all_to_all_single test with BFP16 quantization.
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )

        def _test_all_gather(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=None,
        ):
            # [FUNCTION DOC] HELPER: For each destination rank in the group, builds a tensor of
            # shape (dest+1) by (dest+1) filled with this rank's id, output tensors filled with
            # -1 and expected tensors filled with each rank id; moves them to this rank's GPU if
            # cuda is set. Runs all_gather wrapped by auto_quantize with the given qtype and no
            # loss tracking, and asserts every gathered tensor equals the expected one.
            for dest in group:
                tensor = _build_tensor([dest + 1, dest + 1], rank, dtype=dtype)
                tensors = [
                    _build_tensor([dest + 1, dest + 1], -1, dtype=dtype) for i in group
                ]
                expected_tensors = [
                    _build_tensor([dest + 1, dest + 1], i, dtype=dtype) for i in group
                ]
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                    tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
                allgather = quant.auto_quantize(dist.all_gather, qtype, quant_loss=None)
                allgather(tensors, tensor, group=group_id, async_op=False)

                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        def _test_all_to_all(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=None,
        ):
            # [FUNCTION DOC] HELPER: Does nothing if group_id is None. Otherwise builds, per
            # peer, input tensors with a row count of peer index plus one filled with this rank
            # id, output tensors with rank plus one rows, and expected tensors filled with the
            # peer id; moves them to this rank's GPU if cuda is set. Runs all_to_all wrapped by
            # auto_quantize with the given qtype and asserts the outputs equal the expected
            # tensors.
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                in_tensors = [
                    torch.ones([in_splits[i], size], dtype=dtype) * rank
                    for i, _ in enumerate(group)
                ]
                out_tensors = [
                    torch.ones([(rank + 1), size], dtype=dtype) for _ in group
                ]
                expected_tensors = [
                    torch.ones([rank + 1, size], dtype=dtype) * i for i in group
                ]
                if cuda:
                    in_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in in_tensors]
                    expected_tensors = [
                        t.cuda(rank_to_GPU[rank][0]) for t in expected_tensors
                    ]
                    out_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in out_tensors]
                quantize_alltoall = quant.auto_quantize(
                    dist.all_to_all, qtype, quant_loss=None
                )
                quantize_alltoall(out_tensors, in_tensors, group=group_id)
                for t1, t2 in zip(out_tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        def _test_all_to_all_single(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=DQuantType.FP16,
        ):
            # [FUNCTION DOC] HELPER: Does nothing if group_id is None. Otherwise builds one flat
            # input of summed split rows filled with this rank id, an output with rank plus one
            # rows per peer and the expected concatenated tensor. If cuda is set, moves them to
            # the rank's GPU, runs all_to_all_single wrapped by auto_quantize with the given
            # qtype and split sizes and asserts the output equals the expected tensor.
            # KNOWN ISSUE: the collective call and the assertion are inside the cuda branch, so
            # with cuda false the function checks nothing.
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                out_splits = [rank + 1 for _ in group]
                in_tensor = torch.ones([sum(in_splits), size], dtype=dtype) * rank
                out_tensor = torch.ones([(rank + 1) * size, size], dtype=dtype)
                expected_tensor = torch.cat(
                    [torch.ones([rank + 1, size], dtype=dtype) * i for i in group]
                )
                if cuda:
                    rank_to_GPU = rank_to_GPU[rank][0]
                    in_tensor = in_tensor.cuda(rank_to_GPU)
                    expected_tensor = expected_tensor.cuda(rank_to_GPU)
                    out_tensor = out_tensor.cuda(rank_to_GPU)
                    quantize_alltoall_single = quant.auto_quantize(
                        dist.all_to_all_single, qtype, quant_loss=None
                    )
                    quantize_alltoall_single(
                        out_tensor,
                        in_tensor,
                        out_splits=out_splits,
                        in_splits=in_splits,
                        group=group_id,
                    )
                    self.assertEqual(out_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
