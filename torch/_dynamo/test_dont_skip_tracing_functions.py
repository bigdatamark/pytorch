# [MODULE DOC]
# Test-only helper functions (f1..f6) used to exercise dont_skip_tracing. Lives in
# torch/_dynamo so trace rules skip it by default; a special rule un-skips it when
# dont_skip_tracing is active. No import-time side effects beyond importing torch.
"""
Functions used to test torch._dynamo.dont_skip_tracing.
This file is located in torch/_dynamo so that it is skipped by trace rules.
There is a special rule in trace_rules that doesn't skip this file when
dont_skip_tracing is active.
"""

import torch


def f1(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the input tensor plus one.
    return x + 1


def f2(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the input tensor plus one.
    return x + 1


def f3(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Calls f2 on the input and returns the result (tests nested calls).
    return f2(x)


def f4(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Applies f5 with 1, then f6 wrapped in dont_skip_tracing,
    # then f5 with 8; returns the tensor. Under compilation adds 1, 2, 4 and 8 in
    # total (f6 contains a graph break).
    x = f5(x, 1)
    x = torch._dynamo.dont_skip_tracing(f6)(x)
    x = f5(x, 8)
    return x


def f5(x: torch.Tensor, n: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns x plus n when torch.compiler.is_compiling() is true,
    # otherwise returns x unchanged. Used to detect whether a frame was traced.
    if torch.compiler.is_compiling():
        return x + n
    return x


def f6(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Applies f5 with 2, forces a graph break, then applies f5
    # with 4; returns the tensor.
    x = f5(x, 2)
    torch._dynamo.graph_break()
    x = f5(x, 4)
    return x
