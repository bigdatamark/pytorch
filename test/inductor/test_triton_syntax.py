# Owner(s): ["module: inductor"]
# [MODULE DOC] Regression test that inductor generates syntactically valid Triton code for a
# square root followed by scaling, based on a Newton-Schulz orthogonalization routine (linked
# issue number 142328). Only runs the tests when a GPU is available.

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


class TestTritonSyntacticallyValid(TestCase):
    # [CLASS DOC] Single GPU test that compiles a Newton-Schulz based function and runs it on
    # gradients, failing if the generated Triton code is invalid.
    @requires_gpu()
    def test_triton_sqrt(self):
        # [FUNCTION DOC] Builds a two-layer bias-free linear model on the GPU, runs a forward
        # pass and backward pass to get weight gradients, and runs an inductor-compiled scaled
        # Newton-Schulz function on both gradients with six steps. Passes if compilation and
        # execution raise no error; no values are compared.
        # https://github.com/pytorch/pytorch/issues/142328
        import math

        import torch.nn as nn

        def newtonschulz5(G, steps: int, eps=1e-7):
            assert len(G.shape) == 2  # noqa: S101
            # [FUNCTION DOC] HELPER: Newton-Schulz iteration approximating orthogonalization of
            # a two-dimensional matrix. Raises an assertion error if the input is not
            # two-dimensional. Casts to bfloat16 when the CUDA device supports it without
            # emulation, else float16, normalizes by the norm plus a small epsilon (modifying
            # the cast copy in place), transposes if there are more rows than columns, applies
            # the given number of quintic iterations and transposes back. Returns the result.
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.to(
                torch.bfloat16
                if torch.cuda.is_bf16_supported(including_emulation=False)
                else torch.float16
            )
            X /= X.norm() + eps  # ensure top singular value <= 1
            if G.size(0) > G.size(1):
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            if G.size(0) > G.size(1):
                X = X.T
            return X

        @torch.compile(backend="inductor")
        def scaled_newton_schulz(G, steps: int):
            shape = G.shape
            # [FUNCTION DOC] HELPER: Compiled with the inductor backend. Flattens the gradient
            # to a matrix, runs the Newton-Schulz routine, restores shape and dtype, and
            # multiplies by the square root of the larger of one and the ratio of the first
            # dimension to the element count of the first row. Returns the scaled tensor.
            dtype = G.dtype
            G = G.reshape(shape[0], -1)
            G = newtonschulz5(G, steps)
            G = G.reshape(shape).type(dtype)
            G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
            return G

        model = nn.Sequential(
            nn.Linear(16, 16, bias=False),
            nn.Linear(16, 32, bias=False),
        ).to(device=torch.device(GPU_TYPE))

        loss = model(torch.randn(4, 16, device=torch.device(GPU_TYPE))).sum()
        loss.backward()

        scaled_newton_schulz(model[0].weight.grad, 6)
        scaled_newton_schulz(model[1].weight.grad, 6)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
