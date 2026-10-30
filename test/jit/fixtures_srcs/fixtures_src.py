# [MODULE DOC] Source modules for operator-versioning (upgrader) tests. Each class is a small
# torch module exercising one operator whose schema changed (division, linspace, logspace, gelu,
# random) and whose version is encoded in the class name. They are instantiated in generate_models
# and exported as old-format model files. Importing has no side effects.
from typing import Union

import torch


class TestVersionedDivTensorExampleV7(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that computes true division of two inputs three ways (operator,
    # function, method) to cover the div.Tensor operator.
    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: Returns a tuple of three results: the division operator, the
        # torch division function and the tensor division method, all applied to a and b.
        result_0 = a / b
        result_1 = torch.div(a, b)
        result_2 = a.div(b)
        return result_0, result_1, result_2


class TestVersionedLinspaceV7(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the linspace operator with and without a steps
    # argument.
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        # [FUNCTION DOC] HELPER: Builds a five-step linspace from a to b, and one using the
        # default step count. Returns both tensors as a tuple.
        c = torch.linspace(a, b, steps=5)
        d = torch.linspace(a, b)
        return c, d


class TestVersionedLinspaceOutV7(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the out-variant of the linspace operator.
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: Fills and returns the caller-provided out tensor with linspace
        # values between a and b using default steps.
        return torch.linspace(a, b, out=out)


class TestVersionedLogspaceV8(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the logspace operator with and without a steps
    # argument.
    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        # [FUNCTION DOC] HELPER: Builds a five-step logspace from a to b, and one using the
        # default step count. Returns both tensors as a tuple.
        c = torch.logspace(a, b, steps=5)
        d = torch.logspace(a, b)
        return c, d


class TestVersionedLogspaceOutV8(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the out-variant of the logspace operator.
    def forward(
        self,
        a: Union[int, float, complex],
        b: Union[int, float, complex],
        out: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: Fills and returns the caller-provided out tensor with logspace
        # values between a and b using default steps.
        return torch.logspace(a, b, out=out)


class TestVersionedGeluV9(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the gelu operator through the private nn binding.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the gelu activation of x via the private torch nn
        # binding.
        return torch._C._nn.gelu(x)


class TestVersionedGeluOutV9(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the out-variant of the gelu operator.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Allocates a zeros tensor shaped like x, passes it as the out
        # argument of gelu applied to x, and returns the result of that call.
        out = torch.zeros_like(x)
        return torch._C._nn.gelu(x, out=out)


class TestVersionedRandomV10(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the in-place random_ operator with a range.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Makes a zeros tensor like x and fills it in place with random
        # integers drawn from the range from zero up to ten, then returns it.
        out = torch.zeros_like(x)
        return out.random_(0, 10)


class TestVersionedRandomFuncV10(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the non-underscore random method with a range.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Makes a zeros tensor like x and returns the result of calling its
        # random method with the range zero to ten.
        out = torch.zeros_like(x)
        return out.random(0, 10)


class TestVersionedRandomOutV10(torch.nn.Module):
    # [CLASS DOC] HELPER: Module exercising the out-variant of the random operator with a range.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Rebinds x to a zeros tensor like x, creates another zeros tensor
        # as out, calls the random method on x with the range zero to ten and out as the output,
        # and returns out.
        x = torch.zeros_like(x)
        out = torch.zeros_like(x)
        x.random(0, 10, out=out)
        return out
