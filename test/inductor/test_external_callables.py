# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests the external_matmul Inductor option: user-supplied Python callables that
# write a matmul result into an output buffer are offered as autotuning choices under
# max_autotune, and compiled results are compared against a compiled baseline without them.
# Defines the module, three stand-in callables, and the test class; at import time it picks the
# accelerator device type, falling back to cpu.
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TEST_XPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class MatMulModule(torch.nn.Module):
    # [CLASS DOC] Small module with one parameter, an identity matrix scaled by two (128 by 128,
    # trainable); forward multiplies the input by it.
    def __init__(self):
        # [FUNCTION DOC] Creates the parameter holding twice the 128x128 identity matrix.
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.eye(128, 128) * 2, requires_grad=True)

    def forward(self, x):
        # [FUNCTION DOC] Returns the matrix product of the input and the stored parameter.
        return torch.matmul(x, self.matrix)


# torch.add performs better than torch.mm and got chosen during tuning
def matmul_cpu(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    # [FUNCTION DOC] External matmul stand-in: writes the elementwise sum of the two inputs into
    # the out tensor (not a matmul). Valid only because the tests use twice the identity, for
    # which sum and product coincide. Returns nothing.
    torch.add(a, b, out=out)


def matmul_dup(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    # [FUNCTION DOC] Duplicate of the cpu stand-in with a different name: writes the elementwise
    # sum of a and b into out; used to test registering the same callable twice.
    torch.add(a, b, out=out)


def matmul_cuda(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    # [FUNCTION DOC] Accelerator variant of the stand-in: writes the elementwise sum of a and b
    # into out; used by the GPU test. Returns nothing.
    torch.add(a, b, out=out)


class TestInductorExternalCallable(TestCase):
    # [CLASS DOC] Test case that saves the Inductor config before the class runs and restores it
    # after each test, so options passed to compile do not leak.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup: calls the parent setup then stores a snapshot of the
        # Inductor config on the class.
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        # [FUNCTION DOC] Runs parent teardown then reloads the config snapshot saved at class
        # setup.
        super().tearDown()
        config.load_config(self._saved_config)

    def test_matmul_cpu(self):
        # [FUNCTION DOC] Compiles the module with max_autotune and the cpu stand-in as the
        # external matmul, and checks the output on identity-times-two input matches a compiled
        # baseline with only max_autotune, using assert_close.
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_cpu]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cpu}) failed",
        )

    def test_matmul_dup(self):
        # [FUNCTION DOC] Same as the cpu test but passes the same callable twice in the
        # external_matmul list (comment says only the first is registered); checks output still
        # matches the baseline compiled without it.
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        # This should only register the first external call
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_dup, matmul_dup]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_dup}) failed",
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (7, 0),
        "Triton does not support device capability < 7.0",
    )
    def test_matmul_cuda(self):
        # [FUNCTION DOC] Moves input and module to the current accelerator device, compiles with
        # max_autotune and the accelerator stand-in, and compares with a baseline compiled
        # without it. Skipped without CUDA or XPU, or when CUDA capability is below 7.0.
        device = torch.device(device_type)
        x = (torch.eye(128, 128) * 2).to(device=device)
        opt_fn = torch.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_cuda]},
        )
        opt_fn_golden = torch.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cuda}) failed",
        )


if __name__ == "__main__":
    run_tests()
