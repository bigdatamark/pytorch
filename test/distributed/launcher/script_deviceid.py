# Owner(s): ["oncall: r2p"]

# This is a helper script for
# test_run.py::ElasticLaunchTest::test_virtual_local_rank. It prints out the
# generated inductor output for a simple function.

# [MODULE DOC] Helper script for test_run.py::ElasticLaunchTest::test_virtual_local_rank.
# At import time it initializes an nccl process group, binds the process to the CUDA device
# given by LOCAL_RANK, compiles a trivial function with torch.compile, and prints the
# inductor generated code (captured from the codecache output_code_log debug message) so the
# test can compare generated code across ranks. Finally destroys the process group.
# All work happens at module level; there is no main guard.
# KNOWN ISSUE: the fallback for a missing LOCAL_RANK is the string cuda:0, which cannot be
# converted to an integer, so the script raises ValueError if LOCAL_RANK is unset.

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch._inductor import codecache


@torch.compile
def myfn(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: compiled test function; returns the input tensor added to itself.
    return x + x


dist.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", "cuda:0"))
torch.cuda.set_device(local_rank)


def print_output_code(original_fn):
    # [FUNCTION DOC] HELPER: given the original log function, returns a replacement wrapper to be
    # used as a mock side effect. The original function is accepted but never called.
    # KNOWN ISSUE: because the original is never invoked, the real debug logging is suppressed.
    def wrapper(msg, *args, **kwargs):
        # [FUNCTION DOC] HELPER: log replacement. When extra arguments exist and the message text
        # contains Output code:, prints the first argument (the generated code); otherwise does
        # nothing. Always returns None.
        # Check if this is the "Output code:" message
        if args and "Output code:" in msg:
            print(args[0])

    return wrapper


x = torch.rand(2, 2, device="cuda")

with patch.object(
    codecache.output_code_log,
    "debug",
    side_effect=print_output_code(codecache.output_code_log.debug),
):
    y = myfn(x)

dist.destroy_process_group()
