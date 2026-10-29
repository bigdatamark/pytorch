# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Entry point that runs the shared distributed test suite
# (DistributedTest._DistTestBase) under multiprocessing spawn for one backend.
# Import-time behavior: exits with status 0 if torch.distributed is unavailable or the
# dev-asan build is used; raises RuntimeError unless the BACKEND, WORLD_SIZE and TEMP_DIR
# environment variables are all set; only defines the test class when BACKEND is one of
# gloo, nccl or ucc (otherwise prints a message and no tests are collected).
# Also defines module-level fixtures that disable TF32 matmul around the whole module.

import os
import sys

import torch
import torch.distributed as dist


_PRIOR_FP32_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] HELPER: Module fixture. Saves the current CUDA matmul fp32_precision in the
    # module global _PRIOR_FP32_PRECISION, then turns off allow_tf32 for CUDA matmul.
    global _PRIOR_FP32_PRECISION
    # Snapshot fp32_precision (not allow_tf32) so tearDownModule restores the
    # exact original; writing allow_tf32 back can't reproduce the "none" default.
    _PRIOR_FP32_PRECISION = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    # [FUNCTION DOC] HELPER: Module fixture. If a precision was saved, restores the CUDA matmul
    # fp32_precision to it and clears the saved global; otherwise does nothing.
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        torch.backends.cuda.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed.distributed_test import (
    DistributedTest,
    TestDistBackend,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

_allowed_backends = ("gloo", "nccl", "ucc")
if (
    "BACKEND" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or "TEMP_DIR" not in os.environ
):
    # TODO can we actually have `run_tests.py` emit the complete instructions when it prints a repro command?
    raise RuntimeError(
        "Missing expected env vars for `test_distributed_spawn.py`.  Please ensure to specify the following:\n"
        f"'BACKEND' = one of {_allowed_backends}\n"
        f"'WORLD_SIZE' = int >= 2\n"
        "'TEMP_DIR' specifying a directory containing a barrier file named 'barrier'.\n\n"
        f"e.g.\ntouch /tmp/barrier && TEMP_DIR=/tmp BACKEND='nccl' WORLD_SIZE=2 python {__file__}",
    )

BACKEND = os.environ["BACKEND"]

if BACKEND in _allowed_backends:

    class TestDistBackendWithSpawn(TestDistBackend, DistributedTest._DistTestBase):
        # [CLASS DOC] Concrete test class: combines the backend base class with the shared
        # distributed test cases, so each case runs on WORLD_SIZE spawned processes.
        def setUp(self):
            # [FUNCTION DOC] Runs the base setup, spawns the worker processes, then enters (and
            # never exits) a cudnn flags context with cudnn enabled and TF32 disabled.
            # KNOWN ISSUE: the cudnn flags context is entered but never exited, so it leaks.
            super().setUp()
            self._spawn_processes()
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

else:
    print(f"Invalid backend {BACKEND}. Tests will not be run!")


if __name__ == "__main__":
    run_tests()
