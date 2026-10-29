# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Database of custom operator test cases for OpInfo-based tests. Defines
# custom ops in the _torch_testing namespace (NumPy-based cube, multiply, multiply by scalar, sort,
# take, nonzero, view copy, cat, split copy, split copy with int, non-maximum suppression) with
# fake, autograd and vmap registrations where supported, sample input generators, and the list
# custom_op_db of OpInfo entries. Also registers mechanical source0 to source5 ops. Import-time
# side effects: registers all these operators and fake kernels in the global op registry.
import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch.testing._internal.autograd_function_db import (
    sample_inputs_numpy_cube,
    sample_inputs_numpy_mul,
    sample_inputs_numpy_mul_scalar,
    sample_inputs_numpy_sort,
    sample_inputs_numpy_take,
)
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403

# Note: [custom op db]
#
# This is a collection of custom operator test cases written as OpInfos
# so they can easily be consumed by OpInfo-based tests to check if subsystems
# support them correctly.

def to_numpy(tensor):
    # [FUNCTION DOC] HELPER: Converts a tensor to a NumPy array by moving it to CPU first (shares
    # memory for CPU input).
    return tensor.cpu().numpy()

@torch.library.custom_op("_torch_testing::numpy_cube", mutates_args=())
def numpy_cube(x: Tensor) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_cube) computing the elementwise cube of x
    # with NumPy. Returns a pair: the cubes and the elementwise derivative (three times the
    # square), both on the device of x. Does not mutate inputs. Result dtype follows NumPy's
    # conversion of the input.
    x_np = to_numpy(x)
    dx = torch.tensor(3 * x_np ** 2, device=x.device)
    return torch.tensor(x_np ** 3, device=x.device), dx

@numpy_cube.register_fake
def _(x):
    # [FUNCTION DOC] HELPER: Fake (shape-only) implementation for numpy_cube: returns two clones
    # of the input.
    return x.clone(), x.clone()

def numpy_cube_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_cube: saves the input x and the derivative
    # output for backward.
    x, = inputs
    _cube, dx = output
    ctx.save_for_backward(x, dx)

def numpy_cube_backward(ctx, grad_out, grad_dx):
    # [FUNCTION DOC] HELPER: Backward for numpy_cube: grad is the output grad times the saved
    # derivative plus six times the derivative-output grad times x, using the custom multiply op so
    # it is itself differentiable.
    x, dx = ctx.saved_tensors
    grad_x = numpy_mul(grad_out, dx) + 6 * numpy_mul(grad_dx, x)
    return grad_x

numpy_cube.register_autograd(numpy_cube_backward, setup_context=numpy_cube_setup_context)

def numpy_cube_vmap(info, in_dims, x):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_cube: applies the op directly to the batched input
    # and reports that both outputs are batched along the same dimension as the input.
    result = numpy_cube(x)
    return result, (in_dims[0], in_dims[0])

numpy_cube.register_vmap(numpy_cube_vmap)

@torch.library.custom_op("_torch_testing::numpy_mul", mutates_args=())
def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_mul) multiplying x and y elementwise via
    # NumPy. Returns a new tensor on the device of x. Inputs are not mutated.
    return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

@numpy_mul.register_fake
def _(x, y):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_mul: raises AssertionError if the
    # devices of x and y differ, else returns a contiguous tensor with the broadcast result shape.
    if x.device != y.device:
        raise AssertionError(f"x.device={x.device} != y.device={y.device}")
    return (x * y).contiguous()

def numpy_mul_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_mul: saves both inputs for backward.
    ctx.save_for_backward(*inputs)

def numpy_mul_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_mul: returns grad times y for x and grad times x
    # for y, using None for any input that does not need a gradient.
    x, y = ctx.saved_tensors
    grad_x = grad_out * y if ctx.needs_input_grad[0] else None
    grad_y = grad_out * x if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

numpy_mul.register_autograd(numpy_mul_backward, setup_context=numpy_mul_setup_context)

def numpy_mul_vmap(info, in_dims, x, y):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_mul: moves each batch dimension to the last
    # position (or adds a trailing size-one dimension when unbatched), multiplies, and moves the
    # batch dimension to the front. Returns the result and output batch dimension 0.
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x * y
    result = result.movedim(-1, 0)
    return result, 0

numpy_mul.register_vmap(numpy_mul_vmap)

@torch.library.custom_op("_torch_testing::numpy_mul_scalar", mutates_args=())
def numpy_mul_scalar(x: Tensor, *, scalar: float) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_mul_scalar) multiplying x by the keyword-only
    # float scalar via NumPy. Returns a new tensor on the device of x.
    return torch.tensor(to_numpy(x) * scalar, device=x.device)

@numpy_mul_scalar.register_fake
def _(x, *, scalar):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_mul_scalar: returns a contiguous tensor
    # of the product shape.
    return (x * scalar).contiguous()

def numpy_mul_scalar_setup_context(ctx, inputs, keyword_only_inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_mul_scalar: stores the keyword-only scalar on
    # the context.
    ctx.scalar = keyword_only_inputs["scalar"]

def numpy_mul_scalar_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_mul_scalar: returns the output grad times the
    # stored scalar.
    grad_x = grad_out * ctx.scalar
    return grad_x

numpy_mul_scalar.register_autograd(numpy_mul_scalar_backward, setup_context=numpy_mul_scalar_setup_context)

def numpy_mul_scalar_vmap(info, in_dims, x, *, scalar):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_mul_scalar: moves the batch dimension to last (or
    # adds a trailing one), multiplies by the scalar, moves the batch dimension to the front, and
    # returns it with out dim 0.
    x_bdim, = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    result = x * scalar
    result = result.movedim(-1, 0)
    return result, 0

numpy_mul_scalar.register_vmap(numpy_mul_scalar_vmap)

@torch.library.custom_op("_torch_testing::numpy_sort", mutates_args=())
def numpy_sort(x: Tensor, dim: int) -> tuple[Tensor, Tensor, Tensor]:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_sort) sorting x along dim with NumPy. Returns
    # three tensors on the device of x: the sorted values, the sorting indices, and the inverse
    # permutation indices (the argsort of the indices).
    device = x.device
    x = to_numpy(x)
    ind = np.argsort(x, axis=dim)
    ind_inv = np.argsort(ind, axis=dim)
    result = np.take_along_axis(x, ind, axis=dim)
    return (
        torch.tensor(result, device=device),
        torch.tensor(ind, device=device),
        torch.tensor(ind_inv, device=device),
    )

@numpy_sort.register_fake
def _(x, dim):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_sort: returns empty tensors shaped like
    # x (values with the dtype of x, both index outputs with dtype long).
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)

def numpy_sort_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_sort: stores dim, saves the two index outputs
    # for backward, and marks the index outputs as non-differentiable.
    _out, ind, ind_inv = output
    ctx.dim = inputs[1]
    ctx.save_for_backward(ind, ind_inv)
    ctx.mark_non_differentiable(ind, ind_inv)

def numpy_sort_backward(ctx, grad_out, grad_ind, grad_ind_inv):
    # [FUNCTION DOC] HELPER: Backward for numpy_sort: gathers the output grad with the inverse
    # permutation indices through the take op; the dim input gets no gradient. Index-output grads
    # are ignored.
    ind, ind_inv = ctx.saved_tensors
    return numpy_take(grad_out, ind_inv, ind, ctx.dim), None

numpy_sort.register_autograd(numpy_sort_backward, setup_context=numpy_sort_setup_context)

def numpy_sort_vmap(info, in_dims, x, dim):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_sort: moves the batch dimension to the front,
    # wraps a negative dim relative to the logical (unbatched) rank, sorts along dim shifted by
    # one, and reports out dim 0 for all three outputs.
    x_bdim, _ = in_dims
    x = x.movedim(x_bdim, 0)
    dim = dim if dim >= 0 else dim + x.dim() - 1
    result = numpy_sort(x, dim + 1)
    return result, (0, 0, 0)

numpy_sort.register_vmap(numpy_sort_vmap)

@torch.library.custom_op("_torch_testing::numpy_take", mutates_args=())
def numpy_take(x: Tensor, ind: Tensor, ind_inv: Tensor, dim: int) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_take) gathering values of x along dim using
    # the index tensor ind via NumPy. Returns a new tensor on the device of x. The ind_inv argument
    # is not used in the forward; it exists so backward can apply the inverse permutation.
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

@numpy_take.register_fake
def _(x, ind, ind_inv, dim):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_take: raises AssertionError if ind or
    # ind_inv is on another device than x or is not of dtype long; else returns an empty tensor
    # shaped like x.
    if x.device != ind.device:
        raise AssertionError(f"x.device={x.device} != ind.device={ind.device}")
    if x.device != ind_inv.device:
        raise AssertionError(f"x.device={x.device} != ind_inv.device={ind_inv.device}")
    if ind.dtype != torch.long:
        raise AssertionError(f"ind.dtype must be torch.long, got {ind.dtype}")
    if ind_inv.dtype != torch.long:
        raise AssertionError(f"ind_inv.dtype must be torch.long, got {ind_inv.dtype}")
    return torch.empty_like(x)

def numpy_take_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_take: stores dim and saves the index tensors
    # for backward.
    _x, ind, ind_inv, dim = inputs
    ctx.dim = dim
    ctx.save_for_backward(ind, ind_inv)

def numpy_take_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_take: takes the output grad with the roles of ind
    # and ind_inv swapped to undo the gather; returns that for x and None for the other inputs.
    ind, ind_inv = ctx.saved_tensors
    grad_x = numpy_take(grad_out, ind_inv, ind, ctx.dim)
    return grad_x, None, None, None

numpy_take.register_autograd(numpy_take_backward, setup_context=numpy_take_setup_context)

def numpy_take_vmap(info, in_dims, x, ind, ind_inv, dim):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_take: wraps a negative dim relative to the logical
    # rank, expands the unbatched inputs to the batch size or moves their batch dimension to the
    # front, applies the op along dim shifted by one, and returns the result with out dim 0.
    x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

    # wrap dim
    logical_dim = x.dim() if x_bdim is None else x_bdim - 1
    dim = dim if dim >= 0 else dim + logical_dim

    def expand_bdim(x, x_bdim):
        # [FUNCTION DOC] HELPER: Nested helper for numpy_take_vmap: expands an unbatched tensor to
        # a leading batch size dimension, or moves an existing batch dimension to the front.
        if x_bdim is None:
            return x.expand(info.batch_size, *x.shape)
        return x.movedim(x_bdim, 0)

    x = expand_bdim(x, x_bdim)
    ind = expand_bdim(ind, ind_bdim)
    ind_inv = expand_bdim(ind_inv, ind_inv_bdim)

    return numpy_take(x, ind, ind_inv, dim + 1), 0

numpy_take.register_vmap(numpy_take_vmap)

@torch.library.custom_op("_torch_testing::numpy_nonzero", mutates_args=())
def numpy_nonzero(x: Tensor) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_nonzero) returning the indices of nonzero
    # entries of x as a matrix with one row per nonzero element and one column per dimension.
    # Data-dependent output size. Raises RuntimeError (not supported) if there are at most
    # one row.
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)

@numpy_nonzero.register_fake
def _(x):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_nonzero: returns an empty long tensor
    # whose row count is a fresh unbacked symbolic integer and whose column count is the rank of x.
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    shape = [i0, x.dim()]
    result = x.new_empty(shape, dtype=torch.long)
    return result

def sample_inputs_numpy_nonzero(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: OpInfo sample generator: yields one sample, a 10-element tensor of
    # values in a positive range with a random subset zeroed (masked by random zeros and ones),
    # without gradient tracking of the masking step.
    # KNOWN ISSUE: nothing guarantees at least two nonzero entries, so the op may raise for
    # some samples.
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shape = 10
    result = make_arg(shape, low=0.9, high=2)
    mask = make_tensor(shape, low=0, high=2, device=device, dtype=torch.long)
    with torch.no_grad():
        result *= mask

    yield SampleInput(result, args=())

def numpy_nonzero_vmap(info, in_dims, x):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_nonzero: always raises NotImplementedError, since
    # the output is data-dependent.
    raise NotImplementedError("Operator is data-dependent and cannot be vmapped.")

numpy_nonzero.register_vmap(numpy_nonzero_vmap)

@torch.library.custom_op("_torch_testing::numpy_view_copy", mutates_args=())
def numpy_view_copy(x: Tensor, shape: Sequence[int]) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_view_copy) reshaping x to the given shape via
    # NumPy and returning a copy (not a view) on the device of x.
    return torch.tensor(np.copy(to_numpy(x).reshape(shape)), device=x.device)

@numpy_view_copy.register_fake
def _(x, shape) -> Tensor:
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_view_copy: returns a cloned, reshaped,
    # cloned tensor.
    return x.clone().view(shape).clone()

def numpy_view_copy_setup_context(ctx, inputs, output) -> None:
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_view_copy: stores the input shape on
    # the context.
    ctx.x_shape = inputs[0].shape

def numpy_view_copy_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_view_copy: reshapes the output grad back to the
    # input shape using the same op; the shape argument gets no gradient.
    return torch.ops._torch_testing.numpy_view_copy(grad_out, ctx.x_shape), None

numpy_view_copy.register_autograd(numpy_view_copy_backward, setup_context=numpy_view_copy_setup_context)

def numpy_view_copy_vmap(info, in_dims, x, shape):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_view_copy: moves the batch dimension to the front,
    # reshapes to the batch size followed by the target shape, and returns the result with out dim
    # 0.
    x_bdim, _ = in_dims
    x = x.movedim(x_bdim, 0)
    x_shape = x.shape[0]
    batch_shape = (x_shape, *shape)
    result = numpy_view_copy(x, batch_shape)
    return result, 0

numpy_view_copy.register_vmap(numpy_view_copy_vmap)

def sample_inputs_numpy_view_copy(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: OpInfo sample generator: yields one sample, a 2 by 3 by 4 tensor
    # reshaped to 2 by 12.
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    result = make_arg(2, 3, 4, low=0.9, high=2)
    yield SampleInput(result, args=([2, 12],))

@torch.library.custom_op('_torch_testing::numpy_cat', mutates_args=())
def numpy_cat(xs: Sequence[Tensor], dim: int) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_cat) concatenating a sequence of tensors
    # along dim via NumPy. Raises AssertionError if the sequence is empty or the tensors differ in
    # device or dtype. Returns a new tensor on the device of the first tensor.
    if len(xs) == 0:
        raise AssertionError("xs must not be empty")
    if not all(x.device == xs[0].device for x in xs):
        raise AssertionError("All tensors must be on the same device")
    if not all(x.dtype == xs[0].dtype for x in xs):
        raise AssertionError("All tensors must have the same dtype")
    np_xs = [to_numpy(x) for x in xs]
    np_out = np.concatenate(np_xs, axis=dim)
    return torch.tensor(np_out, device=xs[0].device)

@numpy_cat.register_fake
def _(xs, dim):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_cat: performs the same emptiness, device
    # and dtype checks (raising AssertionError) and returns the regular concatenation.
    if len(xs) == 0:
        raise AssertionError("xs must not be empty")
    if not all(x.device == xs[0].device for x in xs):
        raise AssertionError("All tensors must be on the same device")
    if not all(x.dtype == xs[0].dtype for x in xs):
        raise AssertionError("All tensors must have the same dtype")
    return torch.cat(xs, dim=dim)

def numpy_cat_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_cat: records the size of each input along dim
    # and the dim.
    xs, dim = inputs
    ctx.dim_sizes = [x.shape[dim] for x in xs]
    ctx.dim = dim

def numpy_cat_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_cat: splits the output grad at the cumulative sizes
    # along dim with the split-copy op and returns the pieces for the sequence and None for dim.
    dim_sizes = ctx.dim_sizes
    dim = ctx.dim

    splits = list(np.cumsum(dim_sizes)[:-1])
    grad_xs = torch.ops._torch_testing.numpy_split_copy(grad_out, splits, dim)
    return grad_xs, None

numpy_cat.register_autograd(numpy_cat_backward, setup_context=numpy_cat_setup_context)

def numpy_cat_vmap(info, in_dims, x, dim):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_cat: applies the op directly and returns the batch
    # dimensions given for the input as the output batch dimension.
    # KNOWN ISSUE: the first input is a sequence of tensors, so its batch dimension spec is a
    # sequence and the dim argument is not shifted; the rule looks wrong for batched inputs.
    x_bdim, = in_dims
    result = numpy_cat(x, dim)
    return result, x_bdim

numpy_cat.register_vmap(numpy_cat_vmap)

def sample_inputs_numpy_cat(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: OpInfo sample generator: yields one sample, three tensors with 2, 4
    # and 5 rows (3 by 4 trailing shape) to be concatenated along dimension 0.
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    r0 = make_arg(2, 3, 4, low=0.9, high=2)
    r1 = make_arg(4, 3, 4, low=0.9, high=2)
    r2 = make_arg(5, 3, 4, low=0.9, high=2)
    yield SampleInput([r0, r1, r2], args=(0,))

@torch.library.custom_op('_torch_testing::numpy_split_copy', mutates_args=())
def numpy_split_copy(x: Tensor, splits: Sequence[int], dim: int) -> List[Tensor]:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_split_copy) splitting x at the given indices
    # along dim via NumPy. Returns a list of new tensors (copies) on the device and with the dtype
    # of x.
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs]

@numpy_split_copy.register_fake
def _(x, splits, dim):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_split_copy: returns clones of the pieces
    # of a regular split.
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)]

def numpy_split_copy_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup for numpy_split_copy: stores dim on the context.
    _, _, dim = inputs
    ctx.dim = dim

def numpy_split_copy_backward(ctx, grad_out):
    # [FUNCTION DOC] HELPER: Backward for numpy_split_copy: concatenates the list of grads along
    # dim with the cat op; returns that for x and None for splits and dim.
    result = torch.ops._torch_testing.numpy_cat(grad_out, dim=ctx.dim)
    return result, None, None

numpy_split_copy.register_autograd(numpy_split_copy_backward, setup_context=numpy_split_copy_setup_context)

def numpy_split_copy_vmap(info, in_dims, x, splits, dim):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_split_copy: moves the batch dimension to the
    # front, splits along dim shifted by one, and returns the list result with out dim 0.
    x_bdim, _ , _ = in_dims
    x = x.movedim(x_bdim, 0)
    result = numpy_split_copy(x, splits, dim + 1)
    return result, 0

numpy_split_copy.register_vmap(numpy_split_copy_vmap)

def sample_inputs_numpy_split_copy(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: OpInfo sample generator: yields one sample, a 2 by 9 tensor split at
    # indices 1, 3 and 6 along dimension 1.
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    x = make_arg(2, 9, low=0.9, high=2)
    yield SampleInput(x, args=([1, 3, 6], 1))

@torch.library.custom_op('_torch_testing::numpy_split_copy_with_int', mutates_args=())
def numpy_split_copy_with_int(x: Tensor, splits: Sequence[int], dim: int) -> tuple[List[Tensor], int]:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_split_copy_with_int) like numpy_split_copy
    # but also returns the number of split indices as a plain integer. Returns a pair of the
    # list of tensors and the count.
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs], len(splits)

@numpy_split_copy_with_int.register_fake
def _(x, splits, dim):
    # [FUNCTION DOC] HELPER: Fake implementation: clones of the pieces of a regular split, plus the
    # number of splits.
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)], len(splits)

def numpy_split_copy_with_int_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup: stores dim on the context.
    _, _, dim = inputs
    ctx.dim = dim

def numpy_split_copy_with_int_backward(ctx, grad_out, _):
    # [FUNCTION DOC] HELPER: Backward: concatenates the list of grads along dim; the integer output
    # grad is ignored. Returns the result for x and None for splits and dim.
    return torch.ops._torch_testing.numpy_cat(grad_out, dim=ctx.dim), None, None

numpy_split_copy_with_int.register_autograd(
    numpy_split_copy_with_int_backward,
    setup_context=numpy_split_copy_with_int_setup_context)

def numpy_split_copy_with_int_vmap(info, in_dims, x, splits, dim):
    # [FUNCTION DOC] HELPER: Vmap rule: moves the batch dimension to the front, splits along dim
    # shifted by one, and reports out dims of 0 for every piece and None for the integer.
    x_bdim, _ , _ = in_dims
    x = x.movedim(x_bdim, 0)
    result, len_split = numpy_split_copy_with_int(x, splits, dim + 1)
    return (result, len_split), ([0 for _ in range(len(result))], None)

numpy_split_copy_with_int.register_vmap(numpy_split_copy_with_int_vmap)

@torch.library.custom_op("_torch_testing::numpy_nms", mutates_args=())
def numpy_nms(boxes: Tensor, scores: Tensor, iou_threshold: Number) -> Tensor:
    # [FUNCTION DOC] Custom op (_torch_testing::numpy_nms) non-maximum suppression with NumPy,
    # adapted from fast-rcnn. Inputs: boxes (N by 4 corner coordinates), scores (N) and an overlap
    # threshold. Raises AssertionError on device mismatch, bad shapes, or fewer than two kept
    # boxes. Returns a long tensor of kept box indices in descending score order (data-dependent
    # output size).
    # Adapted from Ross Girshick's fast-rcnn implementation at
    # https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    if boxes.device != scores.device:
        raise AssertionError(f"boxes.device={boxes.device} != scores.device={scores.device}")
    device = boxes.device

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)

    N = boxes.shape[0]
    if boxes.shape != (N, 4):
        raise AssertionError(f"boxes.shape must be (N, 4), got {boxes.shape}")
    if scores.shape != (N,):
        raise AssertionError(f"scores.shape must be (N,), got {scores.shape}")

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    result = torch.tensor(np.stack(keep), device=device)
    # Needed for data-dependent condition :(
    if result.size(0) < 2:
        raise AssertionError(f"result.size(0) must be >= 2, got {result.size(0)}")
    return result

@numpy_nms.register_fake
def _(boxes, scores, iou_threshold):
    # [FUNCTION DOC] HELPER: Fake implementation for numpy_nms: repeats the device and shape checks
    # (AssertionError) and returns an empty int64 tensor whose length is a fresh unbacked symbolic
    # integer length.
    if boxes.device != scores.device:
        raise AssertionError(f"boxes.device={boxes.device} != scores.device={scores.device}")
    N = boxes.shape[0]
    if boxes.shape != (N, 4):
        raise AssertionError(f"boxes.shape must be (N, 4), got {boxes.shape}")
    if scores.shape != (N,):
        raise AssertionError(f"scores.shape must be (N,), got {scores.shape}")

    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    result = boxes.new_empty([i0], dtype=torch.int64)
    return result

def numpy_nms_vmap(info, in_dims, boxes, scores, iou_threshold):
    # [FUNCTION DOC] HELPER: Vmap rule for numpy_nms: always raises NotImplementedError
    # (data-dependent).
    raise NotImplementedError("Operator is data-dependent and cannot be vmapped.")

numpy_nms.register_vmap(numpy_nms_vmap)

def sample_inputs_numpy_nms(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: OpInfo sample generator: yields one sample of 64 random boxes (corner
    # plus size up to four) inside a 28 by 28 area, random scores, and a random scalar overlap
    # threshold. Gradient tracking is applied to boxes and scores per requires_grad.
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    N = 64
    xs = make_arg([N], low=0, high=28)
    dx = make_arg([N], low=0, high=4)
    ys = make_arg([N], low=0, high=28)
    dy = make_arg([N], low=0, high=4)
    boxes = torch.stack([xs, ys, xs + dx, ys + dy], dim=1).requires_grad_(requires_grad)
    scores = make_arg([N], low=0, high=1, requires_grad=requires_grad)
    iou_threshold = make_arg([], low=0, high=1).item()

    yield SampleInput(boxes, args=(scores, iou_threshold))

custom_op_db = [
    OpInfo(
        'NumpyCubeCustomOp',
        op=numpy_cube._opoverload,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulCustomOp',
        op=numpy_mul._opoverload,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulScalarCustomOp',
        op=numpy_mul_scalar._opoverload,
        sample_inputs_func=sample_inputs_numpy_mul_scalar,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortCustomOp',
        op=numpy_sort._opoverload,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyTakeCustomOp',
        op=numpy_take._opoverload,
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyNonzeroCustomOp',
        op=numpy_nonzero._opoverload,
        sample_inputs_func=sample_inputs_numpy_nonzero,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpyNMSCustomOp',
        op=torch.ops._torch_testing.numpy_nms,
        sample_inputs_func=sample_inputs_numpy_nms,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpyViewCopyCustomOp',
        op=torch.ops._torch_testing.numpy_view_copy,
        sample_inputs_func=sample_inputs_numpy_view_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        supports_out=False,
    ),
    OpInfo(
        'NumpyCatCustomOp',
        op=torch.ops._torch_testing.numpy_cat,
        sample_inputs_func=sample_inputs_numpy_cat,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpySplitCopyCustomOp',
        op=torch.ops._torch_testing.numpy_split_copy,
        sample_inputs_func=sample_inputs_numpy_split_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpySplitCopyWithIntCustomOp',
        op=torch.ops._torch_testing.numpy_split_copy_with_int,
        sample_inputs_func=sample_inputs_numpy_split_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs)[0],
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
]


# ==============================================================
# some mechanical test cases
# ==============================================================

lib = torch.library.Library("_torch_testing", "FRAGMENT")

lib.define("source0(Tensor x) -> Tensor")

@torch.library.register_fake("_torch_testing::source0", lib=lib)
def _(x):
    # [FUNCTION DOC] HELPER: Fake implementation registered for source0: returns a clone of x.
    return x.clone()

lib.define("source1(Tensor x) -> Tensor")

def source1_fake(x):
    # [FUNCTION DOC] HELPER: Fake implementation for source1, registered by direct call: returns a
    # clone of x.
    return x.clone()

torch.library.register_fake("_torch_testing::source1", source1_fake, lib=lib)

lib.define("source2(Tensor x) -> Tensor")

@torch.library.register_fake("_torch_testing::source2", lib=lib)
def _(x):
    # [FUNCTION DOC] HELPER: Fake implementation registered for source2: returns a clone of x.
    return x.clone()

lib.define("source3(Tensor x) -> Tensor")

def source3_fake(x):
    # [FUNCTION DOC] HELPER: Fake implementation for source3, registered by direct call: returns a
    # clone of x.
    return x.clone()

torch.library.register_fake("_torch_testing::source3", source3_fake, lib=lib)


@torch.library.custom_op("_torch_testing::source4", mutates_args=())
def source4(x: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Custom op (_torch_testing::source4) returning a clone of x.
    return x.clone()

@source4.register_fake
def _(x):
    # [FUNCTION DOC] HELPER: Fake implementation for source4: returns a clone of x.
    return x.clone()

@torch.library.custom_op("_torch_testing::source5", mutates_args=())
def source5(x: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Custom op (_torch_testing::source5) returning a clone of x.
    return x.clone()

def source5_fake(x):
    # [FUNCTION DOC] HELPER: Fake implementation for source5, registered by direct call: returns a
    # clone of x.
    return x.clone()

source5.register_fake(source5_fake)
