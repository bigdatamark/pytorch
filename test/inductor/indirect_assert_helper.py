# [MODULE DOC] Standalone script run in a subprocess by the inductor indirect-indexing assert
# tests. Defines small functions that index tensors with index values that may be out of bounds.
# When run as a main program it takes four arguments: function name, number of dims (2 or 3),
# dynamic-shape flag and one-size flag (flags are the strings True or False). It validates them
# (raising AssertionError on bad values; the one-size case is allowed only for first_arg), builds a
# float tensor x of shape 3x2 (or 3x2x4; first dim 1 in the one-size case) and an index tensor y
# holding 0..3 on the GPU device type, compiles the chosen function with torch.compile, runs it
# and prints the result. The store function is run for its in-place effect and x is printed.
import sys

import torch
from torch.testing._internal.inductor_utils import GPU_TYPE


def first_arg(x, y):
    # [FUNCTION DOC] Indexes x along its first dimension with y. Returns the selected slices.
    return x[y]


def second_arg(x, y):
    # [FUNCTION DOC] Indexes x along its second dimension with y. Returns the selected slices.
    return x[:, y]


def same_pm_one(x, y):
    # [FUNCTION DOC] Indexes x in its first two dimensions with y plus one and y minus one.
    # Returns the selected elements; the offsets can push indices out of bounds.
    return x[y + 1, y - 1]


def same_pp_one(x, y):
    # [FUNCTION DOC] Indexes x in its first two dimensions with y plus one for both. Returns
    # the selected elements; the offset can push indices out of bounds.
    return x[y + 1, y + 1]


def gather(x, y):
    # [FUNCTION DOC] Gathers along dim 0 of x using an index built from y: the first rows of y
    # (one per row of x) minus one, reshaped and expanded to the shape of x. Returns the
    # gathered tensor. The minus one can produce a negative index.
    index_shape = [x.shape[0]] + [1] * (x.dim() - 1)
    index = (y[: x.shape[0]] - 1).view(index_shape).expand_as(x)
    return torch.gather(x, 0, index)


def gather_generated_index(x, y):
    # [FUNCTION DOC] Gathers along dim 0 of x using an index generated on-device: a range over
    # the size of dim 0 minus one, reshaped and expanded to the shape of x. Argument y is
    # unused. Returns the gathered tensor; the first index is negative.
    index_shape = [x.shape[0]] + [1] * (x.dim() - 1)
    index = (
        (torch.arange(x.shape[0], device=x.device) - 1).view(index_shape).expand_as(x)
    )
    return torch.gather(x, 0, index)


def cross_entropy_loss(x, y):
    # [FUNCTION DOC] Computes cross entropy of x against a target taken from the first rows of
    # y minus one (so a target can be negative). For inputs with more than two dims the target
    # is reshaped and expanded to match the trailing dims. Returns the loss tensor.
    target = y[: x.shape[0]] - 1
    if x.dim() > 2:
        target = target.view(x.shape[0], *[1] * (x.dim() - 2))
        target = target.expand(x.shape[0], *x.shape[2:])
    return torch.nn.functional.cross_entropy(x, target)


def cross_entropy_loss_generated_target(x, y):
    # [FUNCTION DOC] Computes cross entropy of x against a target generated on-device as a
    # range over the batch size minus one (first target is negative). For inputs with more
    # than two dims the target is reshaped and expanded to the trailing dims. Argument y is
    # unused. Returns the loss.
    target = torch.arange(x.shape[0], device=x.device) - 1
    if x.dim() > 2:
        target = target.view(x.shape[0], *[1] * (x.dim() - 2))
        target = target.expand(x.shape[0], *x.shape[2:])
    return torch.nn.functional.cross_entropy(x, target)


def store(x, y, z):
    # [FUNCTION DOC] Writes z into x at positions given by y plus one in the first two dims,
    # in place. Returns nothing; the offset can push indices out of bounds.
    x[y + 1, y + 1] = z


def upper1(x):
    # [FUNCTION DOC] Indexes x with a generated index tensor of the values 0..3. Out of bounds
    # when the first dim of x is smaller than 4 (the upper bound case). Returns the selected
    # slices.
    b = torch.arange(4, device=x.device)
    return x[b]


def lower1(x):
    # [FUNCTION DOC] Indexes x with a scalar int64 tensor holding minus 4. Out of bounds below
    # when the first dim of x is smaller than 4 (the lower bound case). Returns the selected
    # slice.
    b = x.new_full((), -4, dtype=torch.int64)
    return x[b]


def upper2(x):
    # [FUNCTION DOC] Indexes x with a scalar int64 tensor holding 4, which is out of bounds
    # above for the shapes used by the driver (the upper bound case). Returns the selected
    # slice.
    b = x.new_full((), 4, dtype=torch.int64)
    return x[b]


def lower2(x):
    # [FUNCTION DOC] Indexes x with a zero scalar int64 tensor minus 4, giving index minus 4,
    # out of bounds below for the shapes used by the driver (lower bound case). Returns the
    # slice.
    b = x.new_zeros((), dtype=torch.int64)
    return x[b - 4]


if __name__ == "__main__":
    fns = [
        name
        for name, obj in locals().items()
        if callable(obj) and obj.__module__ == __name__
    ]

    _, fn_name, dims, dyn_shape, one_size = sys.argv
    if fn_name not in fns:
        raise AssertionError(f"Unknown function: {fn_name}")
    if one_size not in ("True", "False"):
        raise AssertionError(f"one_size must be 'True' or 'False', got {one_size}")
    one_size = one_size == "True"
    if dims not in ("2", "3"):
        raise AssertionError(f"dims must be '2' or '3', got {dims}")
    shape_x = [3, 2, 4] if dims == "3" else [3, 2]
    if one_size:
        if fn_name != "first_arg":
            raise AssertionError(
                "only first_arg can be tested for a special case of 1-size tensor"
            )
        shape_x[0] = 1
    if dyn_shape not in ("True", "False"):
        raise AssertionError(f"dyn_shape must be 'True' or 'False', got {dyn_shape}")
    dynamic_shapes = dyn_shape == "True"

    x = torch.randn(shape_x, device=GPU_TYPE)
    y = torch.arange(4, device=GPU_TYPE)
    fn = vars()[fn_name]
    fn = torch.compile(dynamic=dynamic_shapes)(fn)
    if fn_name == "store":
        shape = (y.numel(),) + x.shape[2:]
        z = torch.randn(shape, device=GPU_TYPE)
        fn(x, y, z)
        # On Windows, Python will optimize away a function call if its updated value is not used.
        # Touch the memory of x so that the fn(x, y, z) will not be optimized away
        print(x)
    elif fn_name in ("upper1", "upper2", "lower1", "lower2"):
        print(fn(x))
    else:
        print(fn(x, y))
