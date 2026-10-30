# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for IR node reuse in the lazy TorchScript backend. Import-time side effects:
# initializes the TS lazy backend and turns on the global reuse-IR config flag for the whole
# process. Each test runs the same arithmetic eagerly (CPU, or CUDA when the LTC_TS_CUDA
# environment variable is set) and on the lazy device with a step marker per iteration, compares
# results, and checks that a metrics counter shows enough reused IR nodes.

import os
import unittest

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ir_cache
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch._lazy.ts_backend.init()
torch._lazy.config.set_reuse_ir(True)


def get_test_device():
    # [FUNCTION DOC] HELPER: Returns the string cuda if the LTC_TS_CUDA environment variable is
    # present, otherwise cpu; this is the eager reference device.
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIr(TestCase):
    # [CLASS DOC] Test case for IR reuse; skipped on Windows by the decorator. Tests clear metrics
    # and the IR cache when they finish so counters do not leak between tests.
    def testAdd(self):
        # [FUNCTION DOC] Adds two random tensors into an accumulator ten times eagerly and on lazy
        # (one step per iteration), asserts the results match, asserts the reused AddTensor node
        # counter is at least 14, then resets metrics and the IR cache.
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for _ in range(10):
            z += x + y

        for _ in range(10):
            z_lazy += x_lazy + y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 14:
            raise AssertionError(
                f"Expected at least 14 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSub(self):
        # [FUNCTION DOC] Like the add test but the loop adds for the first five iterations and
        # subtracts for the last five, both eagerly and lazily. Asserts matching results and at
        # least 8 reused AddTensor nodes, then resets metrics and the IR cache.
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 8:
            raise AssertionError(
                f"Expected at least 8 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSubFallback(self):
        # [FUNCTION DOC] Same as the add-then-subtract test but with the subtract op forced to fall
        # back to eager via the lazy config. Asserts matching results and at least 8 reused
        # AddTensor nodes, resets metrics and the IR cache, and finally clears the forced fallback.
        # KNOWN ISSUE: the forced fallback is only cleared at the end, so a failing assertion
        #     earlier leaves it set for later tests.
        torch._lazy.config.set_force_fallback("aten::sub")
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 8:
            raise AssertionError(
                f"Expected at least 8 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_force_fallback("")

    def testBatchNorm(self):
        # [FUNCTION DOC] Runs two batch norm entry points (the native op and the legit variant) ten
        # times on a random 16x3x224x224 input with weight and bias, eagerly and on lazy with a
        # step per iteration, asserts both outputs match the eager ones and that the reused
        # NativeBatchNorm node counter is at least 7, then resets metrics and the IR cache.
        device = get_test_device()
        x = torch.randn(16, 3, 224, 224, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)

        for _ in range(10):
            # BatchNorm2d does extra checks on dimensions which SymInts don't support yet
            # so we call `torch.ops.aten.native_batch_norm` to bypass the checks.
            z, _, _ = torch.ops.aten.native_batch_norm(
                x, weight, bias, None, None, True, 0.1, 1e-5
            )
            z_legit, _, _ = torch.ops.aten._native_batch_norm_legit(
                x, weight, bias, True, 0.1, 1e-5
            )

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)
        for _ in range(10):
            z_lazy, _, _ = torch.ops.aten.native_batch_norm(
                x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5
            )
            z_legit_lazy, _, _ = torch.ops.aten._native_batch_norm_legit(
                x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5
            )
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        torch.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::NativeBatchNorm")
        if reused < 7:
            raise AssertionError(
                f"Expected at least 7 reused NativeBatchNorm nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()


if __name__ == "__main__":
    run_tests()
