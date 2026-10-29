# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Runs the shared torch.distributed.nn functional collective tests from
# test_c10d_spawn against the NCCL backend, plus NCCL-only ops (reduce_scatter, all_gather_base)
# and non-contiguous-gradient checks. Defines TestDistributedNNFunctionsNccl unless running under
# dev-dbg-asan. Needs NCCL and at least 2 GPUs. Import-time side effect: sets NO_NCCL from whether
# ProcessGroupNCCL exists. Running as a script calls run_tests.


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsNccl(TestDistributedNNFunctions):
        # [CLASS DOC] NCCL variant of the shared distributed.nn functional test case. Inherits the
        # 2-process spawn setup and the _test_* helpers; the common-op tests only pick the "nccl"
        # backend, and the tests after them cover ops with direct NCCL coverage. All need NCCL, 2+
        # GPUs and torch.distributed.nn (skipped-but-passing in sandcastle otherwise).
        # Test Common Ops First.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            # [FUNCTION DOC] Shared helper for nn broadcast on "nccl": init a 2-rank group on cuda
            # devices, check forward values and gradients.
            self._test_broadcast("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            # [FUNCTION DOC] Shared helper for nn reduce on "nccl": init a 2-rank group on cuda
            # devices, check forward values and gradients.
            self._test_reduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            # [FUNCTION DOC] Shared helper for nn all_reduce on "nccl": init a 2-rank group on
            # cuda devices, check forward values and gradients.
            self._test_allreduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather(self):
            # [FUNCTION DOC] Shared helper for nn all_gather on "nccl": init a 2-rank group on
            # cuda devices, check forward values and gradients.
            self._test_all_gather("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            # [FUNCTION DOC] Shared helper for nn all_to_all on "nccl": init a 2-rank group on
            # cuda devices, check forward values and gradients.
            self._test_all_to_all("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            # [FUNCTION DOC] Shared helper for nn all_to_all_single on "nccl": init a 2-rank group
            # on cuda devices, check forward values and gradients.
            self._test_all_to_all_single("nccl")

        # Test Ops only supported in NCCL.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter(self):
            # [FUNCTION DOC] Initializes a 2-rank NCCL group from a file store, then calls the nn
            # reduce_scatter on two per-rank inputs (ones plus rank, ones plus rank plus one).
            # Checks the output equals a value computed from world size and rank, backpropagates
            # the sine of the output, and checks each input gradient equals the cosine of the
            # expected summed value.
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + self.rank
            x1 = torch.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.empty_like(x0)
            expected = (
                1 + self.world_size
            ) * self.world_size / 2 + self.world_size * self.rank
            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            self.assertEqual(y, torch.ones(5, 5, device=device) * expected)
            z = y.sin().sum()
            z.backward()
            expected_0 = (1 + self.world_size) * self.world_size / 2
            expected_1 = expected_0 + self.world_size
            x_s_0 = (expected_0 * torch.ones(5, 5, device=device)).cos()
            x_s_1 = (expected_1 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s_0)
            self.assertEqual(x1.grad, x_s_1)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter_non_contiguous(self):
            # [FUNCTION DOC] Initializes a 2-rank NCCL group, runs the nn reduce_scatter on random
            # inputs, and backpropagates through an identity op that returns a transposed
            # gradient. Only checks that the backward pass completes; no value is asserted.
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                # [CLASS DOC] HELPER: autograd function that is the identity going forward and
                # hands back a transposed (non-contiguous) gradient going backward.
                @staticmethod
                def forward(ctx, input):
                    # [FUNCTION DOC] HELPER: identity forward; returns the input unchanged.
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    # [FUNCTION DOC] HELPER: returns a cloned, transposed (non-contiguous) copy of
                    # the incoming gradient so the upstream collective backward must handle
                    # non-contiguous gradients.
                    return grad_output.clone().transpose(0, 1)

            x0 = torch.rand(5, 5, device=device, requires_grad=True)
            x1 = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.empty(5, 5, device=device)

            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_reduce_non_contiguous(self):
            # [FUNCTION DOC] Initializes a 2-rank NCCL group, runs the nn all_reduce on a random
            # input, and backpropagates through an op returning a transposed gradient. Only
            # checks that the backward pass completes; no value is asserted.
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                # [CLASS DOC] HELPER: autograd function that is the identity going forward and
                # hands back a transposed (non-contiguous) gradient going backward.
                @staticmethod
                def forward(ctx, input):
                    # [FUNCTION DOC] HELPER: identity forward; returns the input unchanged.
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    # [FUNCTION DOC] HELPER: returns a cloned, transposed (non-contiguous) copy of
                    # the incoming gradient so the upstream collective backward must handle
                    # non-contiguous gradients.
                    return grad_output.clone().transpose(0, 1)

            x = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.distributed.nn.all_reduce(x)
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather_base(self):
            # [FUNCTION DOC] Initializes a 2-rank NCCL group and calls the private nn functional
            # all_gather_base into a preallocated output of world-size stacked blocks. Checks the
            # output shape and that each block holds that rank's input, then backpropagates the
            # sine of the summed blocks and checks the input gradient.
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )

            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True

            output = torch.empty(5 * self.world_size, 5, device=device)
            output = torch.distributed.nn.functional._all_gather_base(output, x)
            self.assertEqual(output.size(), torch.Size((5 * self.world_size, 5)))

            for idx in range(self.world_size):
                self.assertEqual(
                    output[5 * idx : 5 * (idx + 1)],
                    torch.ones(5, 5, device=device) + idx,
                )

            y = torch.sum(output.view(self.world_size, 5, 5), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()
