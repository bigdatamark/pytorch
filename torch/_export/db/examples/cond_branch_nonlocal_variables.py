# [MODULE DOC] ExportDB example: rewriting cond branches to take captured
# values as explicit operands instead of closure variables. Defines
# example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondBranchNonlocalVariables(torch.nn.Module):
    # [CLASS DOC] Stateless example module passing a tensor and a scalar tensor
    # as explicit cond operands rather than capturing them.
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
    - both branches must take the same args, which must also match the branch args passed to cond.
    - both branches must return a single tensor
    - returned tensor must have the same tensor metadata, e.g. shape and dtype
    - branch function can be free function, nested function, lambda, class methods
    - branch function can not have closure variables
    - no inplace mutations on inputs or global variables

    This example demonstrates how to rewrite code to avoid capturing closure variables in branch functions.

    The code below will not work because capturing closure variables is not supported.
    ```
    my_tensor_var = x + 100
    my_primitive_var = 3.14

    def true_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y + my_tensor_var + my_primitive_var

    def false_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y - my_tensor_var - my_primitive_var

    return cond(x.shape[0] > 5, true_fn, false_fn, [x])
    ```

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        # [FUNCTION DOC] Computes x plus 100 and a scalar 3.14, then runs cond on
        # whether the first dimension of x exceeds 5, passing x, the shifted
        # tensor and the scalar (as a tensor) as operands. Returns the sum of
        # the three (true) or x minus the other two (false).
        my_tensor_var = x + 100
        my_primitive_var = 3.14

        def true_fn(x, y, z):
            # [FUNCTION DOC] HELPER: true branch; returns x plus y plus z.
            return x + y + z

        def false_fn(x, y, z):
            # [FUNCTION DOC] HELPER: false branch; returns x minus y minus z.
            return x - y - z

        return cond(
            x.shape[0] > 5,
            true_fn,
            false_fn,
            [x, my_tensor_var, torch.tensor(my_primitive_var)],
        )

example_args = (torch.randn(6),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchNonlocalVariables()
