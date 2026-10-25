# [MODULE DOC] ExportDB example: a module that calls a custom autograd Function
# (backward is not tracked by Dynamo). Defines example_args and model.
# mypy: allow-untyped-defs
import torch

class MyAutogradFunction(torch.autograd.Function):
    # [CLASS DOC] HELPER: custom autograd Function whose forward clones the input
    # and whose backward deliberately returns the incoming gradient plus one.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: returns a clone of x; ctx is unused.
        return x.clone()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: returns grad_output plus one (not a true
        # gradient of a clone; demonstrates untracked backward logic).
        return grad_output + 1

class AutogradFunction(torch.nn.Module):
    # [CLASS DOC] Stateless example module applying MyAutogradFunction in forward.
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    def forward(self, x):
        # [FUNCTION DOC] Applies the custom autograd function to x and returns the result.
        return MyAutogradFunction.apply(x)

example_args = (torch.randn(3, 2),)
model = AutogradFunction()
