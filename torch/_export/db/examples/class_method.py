# [MODULE DOC] ExportDB example: calling a classmethod through the instance,
# the class attribute and type(self). Defines example_args and model.
# mypy: allow-untyped-defs
import torch

class ClassMethod(torch.nn.Module):
    # [CLASS DOC] Example module holding a Linear(4, 2) layer and a classmethod
    # that adds one; export inlines the class method calls.
    """
    Class methods are inlined during tracing.
    """

    @classmethod
    def method(cls, x):
        # [FUNCTION DOC] Returns x plus one.
        return x + 1

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the module with a Linear layer (4 in, 2 out).
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        # [FUNCTION DOC] Applies the linear layer to x, then returns the product
        # of the class method applied three ways (via self, via the class, via type).
        x = self.linear(x)
        return self.method(x) * self.__class__.method(x) * type(self).method(x)

example_args = (torch.randn(3, 4),)
model = ClassMethod()
