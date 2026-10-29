# mypy: ignore-errors

# [MODULE DOC]
# Test database of torch.autograd.Function subclasses wrapped as OpInfo entries, so that
# OpInfo-based tests can check whether a subsystem (vmap, forward AD, functorch, etc.)
# supports autograd.Function. Defines numpy-backed and pure-torch Function classes covering
# different save/mark_dirty/non-differentiable/vmap/jvp patterns, sample input generators, and
# the list autograd_function_db of OpInfo objects. Imports numpy. Building the list at import
# time constructs the OpInfo objects; no other side effects.

import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and
import numpy as np

# Note: [autograd.Function db]
#
# This is a collection of autograd.Function test cases written as OpInfos
# so they can easily be consumed by OpInfo-based tests to check if a subsystem
# supports autograd.Function.
#
# Axes:
# - saves {output, input, intermediate, non-tensor}
# - {inputs, output} x {single tensor, tensors, arbitrary objects}
# - Uses {mark_dirty, mark_non_differentiable, once_differentiable}


def to_numpy(tensor):
    # [FUNCTION DOC] HELPER: Returns the tensor as a numpy array after moving it to the CPU.
    return tensor.cpu().numpy()


class NumpyCube(torch.autograd.Function):
    # [CLASS DOC] Cube op implemented with numpy. Forward returns the cube and its
    # derivative (three times the square) as two outputs. Saves tensors for backward and
    # forward AD, and provides backward, a manual vmap rule and a jvp rule. Second-order
    # differentiable via NumpyMul.
    @staticmethod
    def forward(input):
        # [FUNCTION DOC] Computes the cube of the input via numpy on CPU and returns a pair: the
        # cube and the derivative (three times the square), both as new tensors on the
        # input's device.
        input_np = to_numpy(input)
        dinput = torch.tensor(3 * input_np ** 2, device=input.device)
        return torch.tensor(input_np ** 3, device=input.device), dinput

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Saves the input and the derivative output for both backward and
        # forward AD.
        ctx.save_for_backward(inputs[0], output[1])
        ctx.save_for_forward(inputs[0], output[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        # [FUNCTION DOC] Returns the input gradient given the gradient of the cube output and of
        # the saved derivative output, combining them with the saved derivative and 6 times the
        # saved input.
        input, dinput = ctx.saved_tensors
        return NumpyMul.apply(grad_output, dinput) + 6 * NumpyMul.apply(grad_saved, input)

    @staticmethod
    def vmap(info, in_dims, input):
        # [FUNCTION DOC] Batching rule: applies the function to the (already batched) input and
        # reports that both outputs are batched along the same dimension as the input.
        result = NumpyCube.apply(input)
        return result, (in_dims[0], in_dims[0])

    @staticmethod
    def jvp(ctx, input_tangent):
        # [FUNCTION DOC] Forward-mode rule: returns the tangents of the two outputs computed from
        # the saved derivative and saved input.
        input, dinput = ctx.saved_tensors
        return NumpyMul.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input)


class CubeGenVmap(torch.autograd.Function):
    # [CLASS DOC] Cube op in plain torch with an auto-generated vmap rule. Forward returns the
    # cube and its derivative as two outputs; defines backward and jvp.
    # KNOWN ISSUE: backward and jvp are not consistent with the cube/derivative math used by the
    # numpy variant (backward uses 6 times the derivative instead of using grad_saved), and jvp
    # calls NumpyMul and MulGenVmap, mixing in numpy-backed ops.
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        # [FUNCTION DOC] Returns the cube of x and its derivative (three times the square) as
        # two outputs.
        return x ** 3, 3 * x ** 2

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] Saves the input and the derivative output for backward and forward AD.
        ctx.save_for_backward(inputs[0], outputs[1])
        ctx.save_for_forward(inputs[0], outputs[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        # [FUNCTION DOC] Returns the input gradient from the saved derivative and grad_output.
        # grad_saved is ignored.
        # KNOWN ISSUE: the grad_saved term should involve the saved input, not the derivative.
        _input, dinput = ctx.saved_tensors
        result = grad_output * dinput + 6 * dinput
        return result

    @staticmethod
    def jvp(ctx, input_tangent):
        # [FUNCTION DOC] Forward-mode rule: returns tangents for both outputs from the
        # saved values.
        input, dinput = ctx.saved_tensors
        return MulGenVmap.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input)


def sample_inputs_numpy_cube(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator for the cube ops: yields a single sample with
    # a one-element tensor with values in a positive range, no extra arguments.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(1, low=0.8, high=2), args=())


class NumpyCubeNotComposable(torch.autograd.Function):
    # [CLASS DOC] Numpy-backed cube that stores a numpy array on ctx instead of saving
    # tensors, with a once_differentiable backward, so it cannot be composed for double
    # backward or forward AD.
    @staticmethod
    def forward(input):
        # [FUNCTION DOC] Returns the cube of the input (computed in numpy) and the numpy array of
        # the input.
        input_np = to_numpy(input)
        return torch.tensor(input_np ** 3, device=input.device), input_np

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Stores the numpy input array and the input device on ctx as
        # plain attributes.
        _, input_np = output
        ctx.input_np = input_np
        ctx.device = inputs[0].device

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, grad_saved):
        # [FUNCTION DOC] Returns the derivative (three times the square) as a tensor on the
        # stored device.
        # KNOWN ISSUE: grad_output is ignored, so the result is not scaled by the
        # incoming gradient.
        result_np = 3 * (ctx.input_np ** 2)
        return torch.tensor(result_np, device=ctx.device)


class NumpyMul(torch.autograd.Function):
    # [CLASS DOC] Elementwise product of two tensors computed with numpy. Saves inputs for
    # backward and forward AD, with manual backward, vmap and jvp rules.
    @staticmethod
    def forward(x, y):
        # [FUNCTION DOC] Returns the elementwise product of x and y computed via numpy, as a
        # new tensor.
        return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Saves both inputs for backward and forward AD.
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns gradients for x and y (the other input times grad_output),
        # computing each only if needed according to needs_input_grad, else None.
        x, y = ctx.saved_tensors
        gx = None
        if ctx.needs_input_grad[0]:
            gx = NumpyMul.apply(grad_output, y)
        gy = None
        if ctx.needs_input_grad[1]:
            gy = NumpyMul.apply(grad_output, x)
        return gx, gy

    @staticmethod
    def vmap(info, in_dims, x, y):
        # [FUNCTION DOC] Batching rule: moves each input's batch dimension to the last position
        # (adding a singleton dimension when unbatched), multiplies, and moves the batch back to
        # dim 0.
        x_bdim, y_bdim = in_dims
        x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
        y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
        result = NumpyMul.apply(x, y)
        result = result.movedim(-1, 0)
        return result, 0

    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        # [FUNCTION DOC] Forward-mode rule: product rule using the saved inputs and the
        # two tangents.
        x, y = ctx.saved_tensors
        return x_tangent * y + y_tangent * x

def sample_inputs_numpy_mul(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: yields one sample multiplying a vector by a
    # matrix (broadcasting) with values in a positive range.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # Broadcasting
    yield SampleInput(make_arg(4, low=0.9, high=2), args=(make_arg(3, 4, low=0.9, high=2),))

def sample_inputs_numpy_mul_scalar(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: yields one sample with a vector and a scalar
    # keyword argument named scalar.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(4, low=0.9, high=2), args=(), kwargs={"scalar": 3.14})

class MulGenVmap(torch.autograd.Function):
    # [CLASS DOC] Elementwise product in plain torch with auto-generated vmap rule; defines
    # backward and jvp via the product rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        # [FUNCTION DOC] Returns x times y.
        return x * y

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] Saves both inputs for backward and forward AD.
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns gradients for x and y using the other saved input, only
        # where needed.
        x, y = ctx.saved_tensors
        gx = None
        if ctx.needs_input_grad[0]:
            gx = MulGenVmap.apply(grad_output, y)
        gy = None
        if ctx.needs_input_grad[1]:
            gy = MulGenVmap.apply(grad_output, x)
        return gx, gy

    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        # [FUNCTION DOC] Forward-mode rule: product rule using the saved inputs and tangents.
        x, y = ctx.saved_tensors
        return x_tangent * y + y_tangent * x


class NumpyExp_(torch.autograd.Function):
    # [CLASS DOC] In-place exponential implemented with numpy that mutates and returns its input,
    # marking it dirty. Saves the output for backward and forward AD; has vmap and jvp rules.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] Computes the exponential in place on the input's numpy view (mutating it)
        # and returns the same tensor. Only valid for CPU or shared-memory tensors.
        x_np = to_numpy(x)
        np.exp(x_np, x_np)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Marks the input dirty (modified in place) and saves the output for
        # backward and forward AD.
        x, = inputs
        ctx.mark_dirty(x)
        ctx.save_for_backward(output)
        ctx.save_for_forward(output)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns the gradient: grad_output times the saved output
        # (the exponential).
        output, = ctx.saved_tensors
        return NumpyMul.apply(grad_output, output)

    @staticmethod
    def vmap(info, in_dims, x):
        # [FUNCTION DOC] Batching rule: applies the function in place to the batched input and
        # returns it with the input's batch dimension.
        NumpyExp_.apply(x)
        return x, in_dims[0]

    @staticmethod
    def jvp(ctx, x_tangent):
        # [FUNCTION DOC] Forward-mode rule: multiplies the tangent in place by the saved output
        # and returns it (mutates the tangent).
        # Doesn't call numpy operations because I didn't want to write NumpyMul_
        output, = ctx.saved_tensors
        x_tangent.mul_(output)
        return x_tangent

class NumpySort(torch.autograd.Function):
    # [CLASS DOC] Sort along a dimension implemented with numpy, returning three
    # outputs (the values, the sort indices and the inverse permutation); the indices
    # are marked non-differentiable.
    # KNOWN ISSUE: forward returns a copy of the unsorted input as the first output, not the
    # sorted values, which appears inconsistent with the backward and jvp rules.
    @staticmethod
    def forward(x, dim):
        # [FUNCTION DOC] Computes the sort indices and inverse permutation along dim in numpy
        # and returns tensors on the input device: the input values, the indices, and the
        # inverse indices.
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        return (
            torch.tensor(x, device=device),
            torch.tensor(ind, device=device),
            torch.tensor(ind_inv, device=device),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Marks the index outputs non-differentiable, saves them for backward and
        # forward AD, and stores dim on ctx.
        _x, dim = inputs
        _, ind, ind_inv = output
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # [FUNCTION DOC] Returns the gradient for x by gathering grad_output with the inverse
        # permutation (via NumpyTake), and None for dim. Index-output gradients are ignored.
        ind, ind_inv = ctx.saved_tensors
        return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

    @staticmethod
    def vmap(info, in_dims, x, dim):
        # [FUNCTION DOC] Batching rule: moves the batch dimension to the front, normalizes a
        # negative dim and applies sort along the shifted dim; all three outputs are batched at
        # dim 0.
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 0)
        # wrap dim
        dim = dim if dim >= 0 else dim + x.dim() - 1
        return NumpySort.apply(x, dim + 1), (0, 0, 0)

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: gathers the tangent with the saved indices; None for
        # the two index outputs.
        ind, ind_inv = ctx.saved_tensors
        return NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim), None, None

class SortGenVmap(torch.autograd.Function):
    # [CLASS DOC] Sort in plain torch with auto-generated vmap rule, returning sorted values and
    # the index and inverse-index tensors (marked non-differentiable).
    generate_vmap_rule = True

    @staticmethod
    def forward(x, dim):
        # [FUNCTION DOC] Returns the values sorted along dim, the sort indices and the
        # inverse permutation.
        ind = torch.argsort(x, dim=dim)
        ind_inv = torch.argsort(ind, axis=dim)
        result = torch.take_along_dim(x, ind, dim=dim)
        return result, ind, ind_inv

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] Marks the index outputs non-differentiable, saves them for backward and
        # forward AD, and stores dim on ctx.
        x, dim = inputs
        _, ind, ind_inv = outputs
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # [FUNCTION DOC] Returns the gradient for x by gathering grad_output with the inverse
        # permutation, and None for dim.
        ind, ind_inv = ctx.saved_tensors
        return TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim), None

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: gathers the tangent with the saved indices; None for
        # the index outputs.
        ind, ind_inv = ctx.saved_tensors
        return TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim), None, None


def sample_inputs_numpy_sort(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: yields one sample with a matrix and sort
    # dim 1.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(3, 5), args=(1,))


def sample_inputs_numpy_take(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: builds a matrix, sorts it along dim 1
    # with NumpySort and yields a sample of the matrix with the resulting indices, inverse
    # indices and dim.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor = make_arg(3, 5)
    dim = 1
    _, ind, ind_inv = NumpySort.apply(tensor, 1)
    yield SampleInput(tensor, args=(ind, ind_inv, dim))


class NumpyTake(torch.autograd.Function):
    # [CLASS DOC] Gather along a dimension (take along axis) implemented with numpy, taking the
    # data, the indices, the inverse indices and dim; has backward, vmap and jvp rules.
    @staticmethod
    def forward(x, ind, ind_inv, dim):
        # [FUNCTION DOC] Gathers elements of x along dim using ind via numpy and returns a new
        # tensor on the same device. ind_inv is unused here.
        device = x.device
        x = to_numpy(x)
        ind = to_numpy(ind)
        return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Saves the index tensors for backward and forward AD and stores dim
        # on ctx.
        _x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns the gradient for x by gathering grad_output with the inverse
        # indices and None for the other arguments.
        ind, ind_inv = ctx.saved_tensors
        result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
        return result, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, ind, ind_inv, dim):
        # [FUNCTION DOC] Batching rule: normalizes dim, expands or moves every operand so the
        # batch is at dim 0, applies the gather along the shifted dim, and reports output batch
        # dim 0.
        x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

        # wrap dim
        logical_dim = x.dim() if x_bdim is None else x_bdim - 1
        dim = dim if dim >= 0 else dim + logical_dim

        def expand_bdim(x, x_bdim):
            # [FUNCTION DOC] HELPER: Returns x with its batch dimension at 0: expanded to the
            # batch size if unbatched, otherwise moved from x_bdim.
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        x = expand_bdim(x, x_bdim)
        ind = expand_bdim(ind, ind_bdim)
        ind_inv = expand_bdim(ind_inv, ind_inv_bdim)

        return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

    @staticmethod
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: gathers the tangent with the saved indices. Raises an
        # error if the index tangents are not None.
        if ind_tangent is not None:
            raise AssertionError("Expected ind_tangent to be None")
        if ind_inv_tangent is not None:
            raise AssertionError("Expected ind_inv_tangent to be None")
        ind, ind_inv = ctx.saved_tensors
        return NumpyTake.apply(x_tangent, ind, ind_inv, ctx.dim)

class TakeGenVmap(torch.autograd.Function):
    # [CLASS DOC] Gather along a dimension in plain torch with auto-generated vmap rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(x, ind, ind_inv, dim):
        # [FUNCTION DOC] Gathers x along dim with index tensor ind; ind_inv is unused.
        return torch.take_along_dim(x, ind, dim)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] Saves the index tensors for backward and forward AD and stores dim
        # on ctx.
        _x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns the gradient for x by gathering grad_output with the inverse
        # indices, and None for the other arguments.
        ind, ind_inv = ctx.saved_tensors
        result = TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim)
        return result, None, None, None

    @staticmethod
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: gathers the tangent with the saved indices.
        ind, ind_inv = ctx.saved_tensors
        return TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim)

class Select(torch.autograd.Function):
    # [CLASS DOC] Indexing op (select by idx) with ctx-stored shape and index, with backward, vmap
    # and jvp rules.
    @staticmethod
    def forward(x, idx):
        # [FUNCTION DOC] Returns x indexed by idx.
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Stores the shape of x and the index on ctx.
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns a zero tensor of the input shape with grad_output scattered into
        # the selected position, and None for idx.
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    def vmap(info, in_dims, x, idx):
        # [FUNCTION DOC] Batching rule: moves the batch dimension to position 1 so idx indexes the
        # first logical dimension, applies the op, and reports output batch dim 0.
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 1)
        return Select.apply(x, idx), 0

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: applies the same indexing to the tangent.
        return Select.apply(x_tangent, ctx.idx)

class SelectGenVmap(torch.autograd.Function):
    # [CLASS DOC] Indexing op (select by idx) with auto-generated vmap rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(x, idx):
        # [FUNCTION DOC] Returns x indexed by idx.
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] Stores the shape of x and the index on ctx.
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns a zero tensor of the input shape with grad_output placed at the
        # index, and None for idx.
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: applies the same indexing to the tangent.
        return SelectGenVmap.apply(x_tangent, ctx.idx)


def sample_inputs_select(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: yields one sample with a matrix and index 2.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(3, 5), args=(2,))

class ScaleGradGenVmap(torch.autograd.Function):
    # [CLASS DOC] Identity-like op (clone) whose backward and jvp scale by the class attribute
    # scale (a constant); auto-generated vmap rule.
    generate_vmap_rule = True
    scale = 3.14

    @staticmethod
    def forward(x):
        # [FUNCTION DOC] Returns a clone of x.
        return x.clone()

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] No-op: nothing is saved.
        pass

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns grad_output multiplied by the class-level scale constant.
        return grad_output * ScaleGradGenVmap.scale

    @staticmethod
    def jvp(ctx, x_tangent):
        # [FUNCTION DOC] Forward-mode rule: returns the tangent multiplied by the
        # class-level scale.
        return x_tangent * ScaleGradGenVmap.scale

class ZeroGradientsGenVmap(torch.autograd.Function):
    # [CLASS DOC] Op returning clones of two inputs whose backward and jvp intentionally return
    # zero tensors (not None) to test handling of zero gradients; auto-generated vmap rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        # [FUNCTION DOC] Returns clones of x and y.
        return x.clone(), y.clone()

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # [FUNCTION DOC] No-op: nothing is saved.
        pass

    @staticmethod
    def backward(ctx, gx, gy):
        # [FUNCTION DOC] Returns zero tensors as gradients; the first intentionally has extra
        # leading dimensions (too-large gradient) to test shape handling.
        # Intentionally returning torch.zeros instead of zeros_like or new_zeros.
        # Also intentionally not None.
        return (
            # Intentionally too-large gradient
            torch.zeros(3, 4, *gx.shape, dtype=gx.dtype, device=gx.device),
            torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device),
        )

    @staticmethod
    def jvp(ctx, gx, gy):
        # [FUNCTION DOC] Forward-mode rule: returns zero tangents shaped like the inputs.
        # Intentionally returning torch.zeros instead of zeros_like or new_zeros.
        # Also intentionally not None.
        return (
            torch.zeros(gx.shape, dtype=gx.dtype, device=gx.device),
            torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device),
        )


def sample_inputs_forward_default_args(opinfo, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] HELPER: Sample input generator: yields one sample with a matrix and no
    # other args.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(3, 5))


class ForwardHasDefaultArgs(torch.autograd.Function):
    # [CLASS DOC] Indexing op whose forward has a default value for idx, to test handling of
    # default arguments in forward; has backward, vmap and jvp rules.
    @staticmethod
    def forward(x, idx=(2,)):
        # [FUNCTION DOC] Returns x indexed by idx (default a one-element tuple holding 2).
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] Stores the shape of x and the index on ctx.
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns a zero tensor of the input shape with grad_output placed at the
        # index, and None for idx.
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return result, None

    @staticmethod
    def vmap(info, in_dims, x, idx):
        # [FUNCTION DOC] Batching rule: moves the batch dimension to position 1, applies the op,
        # and reports output batch dim 0.
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 1)
        return ForwardHasDefaultArgs.apply(x, idx), 0

    @staticmethod
    def jvp(ctx, x_tangent, _):
        # [FUNCTION DOC] Forward-mode rule: applies the same indexing to the tangent.
        return ForwardHasDefaultArgs.apply(x_tangent, ctx.idx)


autograd_function_db = [
    OpInfo(
        'NumpyCubeAutogradFunction',
        op=NumpyCube.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyExpMarkDirtyAutogradFunction',
        op=lambda x: NumpyExp_.apply(x.clone()),
        inplace_variant=NumpyExp_.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulAutogradFunction',
        op=NumpyMul.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyCubeNotComposableAutogradFunction',
        op=lambda x: NumpyCubeNotComposable.apply(x)[0],
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortAutogradFunction',
        op=NumpySort.apply,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        gradcheck_wrapper=lambda y, ind: y,
    ),
    OpInfo(
        'NumpyTakeAutogradFunction',
        op=NumpyTake.apply,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=False,
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'SelectAutogradFunction',
        op=Select.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_select,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'CubeGenVmapAutogradFunction',
        op=CubeGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'MulGenVmapAutogradFunction',
        op=MulGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'SortGenVmapAutogradFunction',
        op=SortGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        gradcheck_wrapper=lambda y, ind: y,
    ),
    OpInfo(
        'SelectGenVmapAutogradFunction',
        op=SelectGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_select,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'ScaleGradGenVmapAutogradFunction',
        op=ScaleGradGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'ZeroGradientsGenVmapAutogradFunction',
        op=ZeroGradientsGenVmap.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'ForwardHasDefaultArgsAutogradFunction',
        op=ForwardHasDefaultArgs.apply,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_forward_default_args,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
]
