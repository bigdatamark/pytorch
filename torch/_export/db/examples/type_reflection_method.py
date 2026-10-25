# [MODULE DOC] ExportDB example: calling type on a custom object and then a
# class method of the result. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class A:
    # [CLASS DOC] HELPER: plain Python class exposing one class method.
    @classmethod
    def func(cls, x):
        # [FUNCTION DOC] HELPER: returns one plus x.
        return 1 + x

class TypeReflectionMethod(torch.nn.Module):
    # [CLASS DOC] Stateless example module that reflects on an object's type.
    # KNOWN ISSUE: the docstring says this pattern is not allowed, yet the case
    # is registered with the default supported level.
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """

    def forward(self, x):
        # [FUNCTION DOC] Creates an instance of A and returns the class method
        # func (looked up through type) applied to x, i.e. one plus x.
        a = A()
        return type(a).func(x)


example_args = (torch.randn(3, 4),)
tags = {"python.builtin"}
model = TypeReflectionMethod()
