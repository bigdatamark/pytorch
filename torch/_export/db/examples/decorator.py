# [MODULE DOC] ExportDB example: a decorator wrapping forward, which is inlined
# during tracing. Defines example_args and model.
# mypy: allow-untyped-defs
import functools

import torch

def test_decorator(func):
    # [FUNCTION DOC] HELPER: decorator factory. Returns a wrapper (preserving
    # func metadata) that calls func with the same arguments and adds one to its result.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: calls the wrapped function and returns its result plus one.
        return func(*args, **kwargs) + 1

    return wrapper

class Decorator(torch.nn.Module):
    # [CLASS DOC] Stateless example module whose forward is decorated.
    """
    Decorators calls are inlined into the exported function during tracing.
    """

    @test_decorator
    def forward(self, x, y):
        # [FUNCTION DOC] Returns x plus y; the decorator adds one more to the result.
        return x + y

example_args = (torch.randn(3, 2), torch.randn(3, 2))
model = Decorator()
