# [MODULE DOC] ExportDB example: a method decorated with assume_constant_result
# so that non-traceable code is treated as a constant. Defines example_args,
# tags and model for the case registry.
# mypy: allow-untyped-defs
import torch
import torch._dynamo as torchdynamo


class AssumeConstantResult(torch.nn.Module):
    # [CLASS DOC] Stateless example module whose helper get_item is marked
    # constant-result, then used to slice the input.
    """
    Applying `assume_constant_result` decorator to burn make non-traceable code as constant.
    """

    @torchdynamo.assume_constant_result
    def get_item(self, y):
        # [FUNCTION DOC] Converts tensor y to integer and returns its Python
        # scalar value; tracing treats the result as a constant.
        return y.int().item()

    def forward(self, x, y):
        # [FUNCTION DOC] Returns x sliced along its first dimension up to the
        # constant integer obtained from y.
        return x[: self.get_item(y)]

example_args = (torch.randn(3, 2), torch.tensor(4))
tags = {"torch.escape-hatch"}
model = AssumeConstantResult()
