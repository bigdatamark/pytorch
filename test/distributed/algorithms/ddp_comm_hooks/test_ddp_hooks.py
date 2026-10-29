# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Two-process tests for the built-in DDP communication hooks (allreduce, fp16
# compress, quantize per tensor, quantize per channel, noop) that compare gradients with and
# without the hook, and a test that the last-bucket flag is seen only on the final bucket.
# Import-time effects: exits the process with status 0 if torch.distributed is unavailable or
# under dev-asan builds; computes the module global device_type (current available accelerator
# type or cpu).

import sys

import torch
import torch.distributed as dist
from torch import nn


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.algorithms.ddp_comm_hooks import (
    DDPCommHookType,
    register_ddp_comm_hook,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if TEST_WITH_DEV_DBG_ASAN:
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)


device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)


def gpus_for_rank(world_size):
    # [FUNCTION DOC] HELPER: Splits the visible accelerator devices evenly among world_size
    # processes (integer share per process, extra devices unused) and returns a list with one
    # list of device indices per rank.
    visible_devices = list(range(torch.accelerator.device_count()))
    gpus_per_process = torch.accelerator.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class Task(nn.Module):
    # [CLASS DOC] HELPER: Module with a single seeded random 40 by 20 parameter (the seed is
    # reset to 0 at construction) whose forward multiplies the input elementwise by the
    # parameter.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 (side effect) and creates the random
        # parameter p.
        super().__init__()
        torch.manual_seed(0)
        self.p = nn.Parameter(torch.randn(40, 20))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the parameter multiplied elementwise by x.
        return self.p * x


class TestDdpCommHook(nn.Module):
    # [CLASS DOC] HELPER: Wrapper model with one Task submodule; forward raises its input to the
    # power one plus rank, so each rank produces different gradients. Despite the Test prefix it
    # is a plain module, not a test case.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the Task submodule t0.
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        # [FUNCTION DOC] HELPER: Raises x elementwise to the power of one plus rank and returns
        # the output of the Task submodule on it.
        return self.t0(x ** (1 + rank))


class DistributedDataParallelCommHookTest(DistributedTestBase):
    # [CLASS DOC] Two-rank test case (DistributedTestBase) comparing DDP gradients with each
    # built-in communication hook against gradients with no hook.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns 2, the number of ranks used.
        return 2

    def _local_model(self):
        # [FUNCTION DOC] HELPER: Returns a fresh TestDdpCommHook model on the CPU.
        # KNOWN ISSUE: not used anywhere in this file.
        local_model = TestDdpCommHook().cpu()

        return local_model

    def _get_grads(self, process_group, hook_type=None):
        # [FUNCTION DOC] HELPER: Wraps a new model on this rank's first assigned device in DDP
        # with the given process group, optionally registers the named built-in comm hook type
        # using the process group as hook state, runs forward and backward and returns the
        # resulting gradient of the parameter.
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            TestDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        # Register DDP Communication Hook if defined
        if hook_type is not None:
            register_ddp_comm_hook(
                comm_hook_type=hook_type, model=gpu_model, state=process_group
            )

        return self._run_and_get_grads(gpu_model)

    def _run_and_get_grads(self, model):
        # [FUNCTION DOC] HELPER: Seeds the RNG with 2020, makes a random 40 by 20 input, runs the
        # model with this rank, backpropagates the mean of the output and returns the gradient of
        # the model's first parameter.
        torch.manual_seed(2020)
        input = torch.randn(40, 20)
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        # The only layer
        param = next(model.parameters())
        return param.grad

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook(self):
        # [FUNCTION DOC] Creates a process group and checks the gradient with the allreduce hook
        # equals the no-hook gradient to a relative tolerance of 1e-5 with zero absolute
        # tolerance. Needs an accelerator backend and 2 GPUs.
        """
        This unit test verifies the ``allreduce`` hook registered case gives same result
        with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.ALLREDUCE)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_fp16compress_hook(self):
        # [FUNCTION DOC] Same comparison for the fp16 compress hook, allowing a relative
        # tolerance of 1e-5 and absolute tolerance of 1e-4.
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.FP16_COMPRESS)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_tensor_hook(self):
        # [FUNCTION DOC] Same comparison for the quantize-per-tensor hook, with relative 1e-5 and
        # absolute 1e-4 tolerances.
        """
        This unit test verifies the ``quantize per tensor`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_TENSOR)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_channel_hook(self):
        # [FUNCTION DOC] Same comparison for the quantize-per-channel hook, with relative 1e-5
        # and absolute 1e-4 tolerances.
        """
        This unit test verifies the ``quantize per channel`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(
            process_group, DDPCommHookType.QUANTIZE_PER_CHANNEL
        )

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_noop_hook(self):
        # [FUNCTION DOC] Checks the noop hook: the hook gradient is divided by the world size and
        # all-reduced across the group, and then must equal the no-hook gradient (relative 1e-5,
        # absolute 0). Mutates the hook gradient in place.
        """
        This unit test verifies the ``noop`` hook registered case and a subsequent allreduce
        gives same result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.NOOP)
        # Apply a subsequent allreduce to average grads.
        hook_grads.div_(self.world_size)
        dist.all_reduce(hook_grads, group=process_group)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_is_last_hook(self):
        # [FUNCTION DOC] Registers a custom comm hook on a DDP model of 11 large bias-free linear
        # layers that records each bucket's is-last flag in a list and returns the bucket buffer
        # unchanged. After one forward and backward asserts the last recorded flag is true and
        # none of the earlier flags are.
        process_group = self.create_pg(device_type)

        def hook(flags, bucket):
            # [FUNCTION DOC] HELPER: Comm hook that appends whether the bucket is the last one to
            # the flags list and returns an already completed future holding the bucket buffer
            # (no communication).
            flags.append(bucket.is_last())
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        flags = []
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = nn.Sequential(
            nn.Linear(2, 4000, bias=False),
            *[nn.Linear(4000, 4000, bias=False) for _ in range(10)],
        )
        gpu_model = DistributedDataParallel(
            model.to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        gpu_model.register_comm_hook(state=flags, hook=hook)
        input = torch.randn(10, 2)
        gpu_model(input).sum().backward()
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))


if __name__ == "__main__":
    if torch.cuda._initialized:
        raise AssertionError(
            "test_distributed must not have initialized CUDA context on main process"
        )

    run_tests()
