# [MODULE DOC] ExportDB example: cond whose branches are nested functions that
# themselves define inner functions. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondBranchNestedFunction(torch.nn.Module):
    # [CLASS DOC] Stateless example module with nested-function cond branches.
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables

    This example demonstrates using nested function in cond().

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        # [FUNCTION DOC] Runs cond on whether the first dimension of x is below
        # 10: true branch gives x plus x, false branch gives x minus x. Returns it.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: true branch; returns x plus x via an inner function.
            def inner_true_fn(y):
                # [FUNCTION DOC] HELPER: adds the enclosing x to y.
                return x + y

            return inner_true_fn(x)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: false branch; returns x minus x via an inner function.
            def inner_false_fn(y):
                # [FUNCTION DOC] HELPER: subtracts y from the enclosing x.
                return x - y

            return inner_false_fn(x)

        return cond(x.shape[0] < 10, true_fn, false_fn, [x])

example_args = (torch.randn(3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchNestedFunction()
