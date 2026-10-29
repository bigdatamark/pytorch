# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Runs the shared torch.distributed.nn functional collective tests from
# test_c10d_spawn against the UCC backend. Defines TestDistributedNNFunctionsUcc (only when not
# running under dev-dbg-asan) whose tests each call a shared helper with the backend name "ucc".
# Needs UCC support and at least 2 GPUs. Import-time side effect: computes NO_UCC from whether
# ProcessGroupUCC exists. Running the file as a script calls run_tests.


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_UCC = not hasattr(c10d, "ProcessGroupUCC")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        # [CLASS DOC] UCC variant of the shared distributed.nn functional test case. Inherits
        # world size, process spawning and the _test_* helpers from the base class; each test
        # here only selects the "ucc" backend. Skipped without UCC, with fewer than 2 GPUs, or
        # when torch.distributed.nn is unavailable (skipped-but-passing in sandcastle).
        # Test Common Ops First.
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            # [FUNCTION DOC] Shared helper for nn broadcast on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            self._test_broadcast("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            # [FUNCTION DOC] Shared helper for nn reduce on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            self._test_reduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            # [FUNCTION DOC] Shared helper for nn all_reduce on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            self._test_allreduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self):
            # [FUNCTION DOC] Shared helper for nn all_gather on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            # Also skipped in sandcastle (illegal memory access on the first equality check).
            self._test_all_gather("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            # [FUNCTION DOC] Shared helper for nn all_to_all on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            self._test_all_to_all("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            # [FUNCTION DOC] Shared helper for nn all_to_all_single on "ucc":
            # init a 2-rank group on cuda devices, check forward values and gradients.
            self._test_all_to_all_single("ucc")


if __name__ == "__main__":
    run_tests()
