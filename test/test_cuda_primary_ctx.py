# Owner(s): ["module: cuda"]
# [MODULE DOC] Tests that common tensor operations do not create CUDA primary contexts on
# devices other than the one in use. Must run in a fresh process where no context exists.
# Import-time: prints a message and swaps TestCase for NoTest when CUDA is unavailable.

import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_utils import NoTest, run_tests, skipIfRocm, TestCase


# NOTE: this needs to be run in a brand new process

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaPrimaryCtx(TestCase):
    # [CLASS DOC] Test case checking, through the private has-primary-context query, which
    # devices have a primary context after various operations. CTX_ALREADY_CREATED_ERR_MSG is
    # the failure message used when setUp finds a pre-existing context.
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_cuda_primary_ctx.py must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, then asserts that no device has a primary
        # context yet, failing with CTX_ALREADY_CREATED_ERR_MSG otherwise.
        super().setUp()
        for device in range(torch.cuda.device_count()):
            # Ensure context has not been created beforehand
            self.assertFalse(
                torch._C._cuda_hasPrimaryContext(device),
                TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    @skipIfRocm(
        msg="last checked in ROCm 7, HIP runtime doesn't create context for hipSetDevice()"
    )
    def test_set_device_0(self):
        # [FUNCTION DOC] Checks that device 0 has no context, sets the current device to 0,
        # and checks a context now exists (CUDA 12 creates it eagerly). Skipped on ROCm.
        # In CUDA 12 the behavior of cudaSetDevice has changed. It eagerly creates context on target.
        # The behavior of `torch.cuda.set_device(0)` should also create context on the device 0.
        # Initially, we should not have any context on device 0.
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        torch.cuda.set_device(0)
        # Now after the device was set, the context should present in CUDA 12.
        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_str_repr(self):
        # [FUNCTION DOC] Creates a tensor on device 1, calls str and repr on it, and checks a
        # context exists only on device 1 before and after. Needs multiple GPUs.
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        str(x)
        repr(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy(self):
        # [FUNCTION DOC] Creates a tensor on device 1, copies it into a CPU tensor, and checks
        # a context exists only on device 1 before and after. Needs multiple GPUs.
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        y = torch.randn(1, device="cpu")
        y.copy_(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_pin_memory(self):
        # [FUNCTION DOC] Creates a tensor on device 1, then runs several pinned-memory
        # operations (is_pinned query, pin_memory, and creation with the pin_memory option via
        # randn, zeros and empty), checking after each that only device 1 has a primary
        # context. Needs multiple GPUs.
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertFalse(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu").pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertTrue(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.zeros(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.empty(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = x.pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))


if __name__ == "__main__":
    run_tests()
