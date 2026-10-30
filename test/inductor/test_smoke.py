# Owner(s): ["module: inductor"]
# [MODULE DOC] Smoke tests for torch.compile on a small MLP, the compile decorator, and option
# validation. GPU tests are skipped without Triton/GPU support. The main block runs only on
# Linux with a GPU that is non-CUDA/Triton or a CUDA device of major version <= 5.
import logging
import unittest

import torch
import torch._logging
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)


class MLP(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny model with two linear layers (1->6, 6->1), each followed by
    # relu.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two linear layers l1 and l2.
        super().__init__()
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        # [FUNCTION DOC] HELPER: Applies l1 then relu, then l2 then relu, and returns the
        # result.
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


def _test_f(x):
    # [FUNCTION DOC] HELPER: Trivial function returning the elementwise square of its input.
    return x * x


class SmokeTest(TestCase):
    # [CLASS DOC] Smoke tests for compile entry points; no instance state.
    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_mlp(self):
        # [FUNCTION DOC] Enables DEBUG logging for dynamo, inductor and aot, compiles the MLP
        # on the GPU and runs it three times on a one-element tensor, then resets logging to
        # defaults. Passes if nothing raises. Skipped when no GPU/Triton.
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )

        mlp = torch.compile(MLP().to(GPU_TYPE))
        for _ in range(3):
            mlp(torch.randn(1, device=GPU_TYPE))

        # set back to defaults
        torch._logging.set_logs()

    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_compile_decorator(self):
        # [FUNCTION DOC] Uses the compile decorator on a sin-plus-min function and, with the
        # reduce-overhead mode, on a square function; runs each three times on GPU tensors.
        # Passes if nothing raises. Skipped when no GPU/Triton.
        @torch.compile
        def foo(x):
            # [FUNCTION DOC] HELPER: Compiled function returning sin of input plus its
            # minimum.
            return torch.sin(x) + x.min()

        @torch.compile(mode="reduce-overhead")
        def bar(x):
            # [FUNCTION DOC] HELPER: Compiled (reduce-overhead mode) function squaring its
            # input.
            return x * x

        for _ in range(3):
            foo(torch.full((3, 4), 0.7, device=GPU_TYPE))
            bar(torch.rand((2, 2), device=GPU_TYPE))

    def test_compile_invalid_options(self):
        # [FUNCTION DOC] Checks that compiling with an unrecognized mode string raises
        # RuntimeError.
        with self.assertRaises(RuntimeError):
            torch.compile(_test_f, mode="ha")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_LINUX and HAS_GPU:
        if (not HAS_CUDA_AND_TRITON) or torch.cuda.get_device_properties(0).major <= 5:
            run_tests()
