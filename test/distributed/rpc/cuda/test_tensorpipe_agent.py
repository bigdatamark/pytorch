#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Generates the CUDA-enabled RPC test suites for the TensorPipe agent. At import
# time it exits the process (status 0) when torch.distributed is unavailable, disables the
# CUDA allocator expandable_segments setting when CUDA is present, and injects into the module
# globals the test classes built by generate_tests from TensorPipeRpcAgentTestFixture and the
# generic plus TensorPipe-specific CUDA test lists. Run directly it calls run_tests.
import sys

import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_CUDA_TESTS,
    TENSORPIPE_CUDA_TESTS,
)


if torch.cuda.is_available():
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_CUDA_TESTS + TENSORPIPE_CUDA_TESTS,
        __name__,
    )
)


if __name__ == "__main__":
    run_tests()
