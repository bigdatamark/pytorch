#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Generates the RPC test suites for the faulty (fault-injection) RPC agent. At
# import time it exits with status 0 when torch.distributed is unavailable; otherwise, unless
# running in CI on a machine with CUDA, it injects into the module globals the test classes
# built by generate_tests from FaultyRpcAgentTestFixture and FAULTY_AGENT_TESTS.
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import IS_CI, run_tests
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    FAULTY_AGENT_TESTS,
    generate_tests,
)


# On CircleCI these tests are already run on CPU jobs, thus to save resources do
# not run them on GPU jobs, since they wouldn't provide additional test signal.
if not (IS_CI and torch.cuda.is_available()):
    globals().update(
        generate_tests(
            "Faulty",
            FaultyRpcAgentTestFixture,
            FAULTY_AGENT_TESTS,
            __name__,
        )
    )


if __name__ == "__main__":
    run_tests()
