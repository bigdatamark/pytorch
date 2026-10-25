# [MODULE DOC] ExportDB example: a loop with a constant trip count that gets
# unrolled. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class StaticForLoop(torch.nn.Module):
    # [CLASS DOC] Stateless example module with a fixed-length comprehension.
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns a list of ten tensors, each x plus the loop index (0 to 9).
        # constant
        ret = [i + x for i in range(10)]
        return ret

example_args = (torch.randn(3, 2),)
tags = {"python.control-flow"}
model = StaticForLoop()
