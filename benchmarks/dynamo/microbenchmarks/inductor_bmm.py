# [MODULE DOC] Script comparing batched matmul timings (CUDA float16) between eager PyTorch and
# two Inductor compiled variants, on attention shapes from BERT, hf_GPT2 and hf_Albert; prints a
# semicolon separated table of milliseconds. Defines two Dynamo-optimized functions at import.
# KNOWN ISSUE: the Inductor triton config has no option for choosing the bmm implementation, so
#     the toggle used here likely has no effect and the two Inductor columns may measure the
#     same code.
from benchmark_helper import time_with_torch_timer

import torch
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_bmm(a, b):
    # [FUNCTION DOC] Batched matmul of a and b, compiled by Dynamo with the Inductor backend and
    # no graph breaks allowed. Meant to run while the Triton bmm toggle is off.
    return torch.bmm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bmm(a, b):
    # [FUNCTION DOC] Same computation as the aten variant, a separate compiled function meant to
    # be first called while the Triton bmm toggle is on. Returns the batched matmul.
    return torch.bmm(a, b)


def torch_bmm(a, b):
    # [FUNCTION DOC] Eager batched matmul of a and b; the baseline. Returns the result.
    return torch.bmm(a, b)


def test_total_time(shapes):
    # [FUNCTION DOC] For each pair of operand shapes, create random float16 CUDA inputs, warm up
    # each compiled variant under its toggle setting, then time eager and both compiled variants
    # with the Timer helper and print the mean milliseconds of each, after the shape text on the
    # same line.
    print("shape; torch bmm; inductor aten bmm; inductor triton bmm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.triton.use_bmm = False
        inductor_aten_bmm(a, b)

        config.triton.use_bmm = True
        inductor_triton_bmm(a, b)

        torch_ms = time_with_torch_timer(torch_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = False
        ind_aten_ms = time_with_torch_timer(inductor_aten_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = True
        ind_triton_ms = time_with_torch_timer(inductor_triton_bmm, (a, b)).mean * 1000

        print(torch_ms, ind_aten_ms, ind_triton_ms, sep="; ")


if __name__ == "__main__":
    shapes = [
        # BERT (all)
        ([192, 128, 64], [192, 64, 128]),
        ([192, 128, 128], [192, 128, 64]),
        # hf_GPT2 (all)
        ([12, 1024, 1024], [12, 1024, 64]),
        ([12, 1024, 64], [12, 64, 1024]),
        # hf_Albert (all)
        ([12, 512, 64], [12, 64, 512]),
        ([12, 512, 512], [12, 512, 64]),
    ]

    test_total_time(shapes)
