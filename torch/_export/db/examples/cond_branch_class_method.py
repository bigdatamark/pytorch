# [MODULE DOC] ExportDB example: passing module methods as the branch functions
# of cond. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class MySubModule(torch.nn.Module):
    # [CLASS DOC] HELPER: submodule whose forward delegates to foo, used as a branch.
    def foo(self, x):
        # [FUNCTION DOC] HELPER: returns the elementwise cosine of x.
        return x.cos()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: returns foo applied to x.
        return self.foo(x)

class CondBranchClassMethod(torch.nn.Module):
    # [CLASS DOC] Example module holding a MySubModule and using its forward and
    # its own bar method as the two cond branches.
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables


    This example demonstrates using class method in cond().

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the MySubModule stored as the attribute subm.
        super().__init__()
        self.subm = MySubModule()

    def bar(self, x):
        # [FUNCTION DOC] Returns the elementwise sine of x (the false branch).
        return x.sin()

    def forward(self, x):
        # [FUNCTION DOC] Conditionally applies cos (submodule forward) when the
        # first dimension of x is at most 2, otherwise sin (bar); returns the result.
        return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

example_args = (torch.randn(3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchClassMethod()
