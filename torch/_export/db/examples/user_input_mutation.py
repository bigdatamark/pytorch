# [MODULE DOC] ExportDB example: in-place mutation of a user input. Defines
# example_args, tags and model.
# mypy: allow-untyped-defs
import torch


class UserInputMutation(torch.nn.Module):
    # [CLASS DOC] Stateless example module that mutates its input tensor.
    """
    Directly mutate user input in forward
    """

    def forward(self, x):
        # [FUNCTION DOC] Side effect: multiplies x in place by 2. Returns cosine
        # of the mutated x.
        x.mul_(2)
        return x.cos()


example_args = (torch.randn(3, 2),)
tags = {"torch.mutation"}
model = UserInputMutation()
