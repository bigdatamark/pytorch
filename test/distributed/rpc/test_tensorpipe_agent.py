#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Generates the CPU RPC test suites for the TensorPipe agent. At import time it
# exits with status 0 when torch.distributed is unavailable; otherwise, unless running in CI
# on a machine with CUDA (where the CPU jobs already cover it), it injects into the module
# globals the test classes built by generate_tests from TensorPipeRpcAgentTestFixture and the
# generic plus TensorPipe-specific test lists.
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import IS_CI, run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_TESTS,
    TENSORPIPE_TESTS,
)


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since they wouldn't provide additional test signal.
if not (IS_CI and torch.cuda.is_available()):
    globals().update(
        generate_tests(
            "TensorPipe",
            TensorPipeRpcAgentTestFixture,
            GENERIC_TESTS + TENSORPIPE_TESTS,
            __name__,
        )
    )


if __name__ == "__main__":
    run_tests()
