# [MODULE DOC] Support for torch.autograd.Function under functorch transforms (grad, jvp, vmap,
# functionalize). Defines the custom_function_call higher-order operator with per-transform
# dispatch rules, helpers that generate single-level or vmapped autograd.Function subclasses,
# ctx wrapper classes, gradient reduction helpers, the autograd_function_apply higher-order
# operator used by Dynamo, and a Dynamo tracing helper. Import-time side effects: instantiates
# and registers the custom_function_call and autograd_function_apply operators and their py_impl rules.
from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    current_level,
    TransformType,
)
from torch._functorch.apis import vmap
from torch._functorch.utils import enable_single_level_autograd_function
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    restore_vmap,
    unwrap_batched,
    wrap_batched,
)
from torch._ops import HigherOrderOperator
from torch.autograd.forward_ad import _set_fwd_grad_enabled


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._functorch.pyfunctorch import FuncTorchInterpreter, VmapInterpreter

_P = ParamSpec("_P")
_R = TypeVar("_R")


# autograd.Function technically runs before the regular PyTorch dispatcher.
# This is how features like autocast and torch_dispatch (e.g. PythonTLSSnapshot)
# work with it. One day we might decide to change this, but until then,
# we need to give the illusion that autograd.Function runs before those things.
#
# We do this by using creating a custom HigherOrderOperator that only functorch
# dispatches specially.
class CustomFunctionHigherOrderOperator(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator named custom_function_call that lets functorch transforms
    # intercept autograd.Function calls. Holds no extra state; transform-specific behavior is
    # registered through py_impl on the single instance custom_function_call.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name custom_function_call.
        super().__init__("custom_function_call")

    def __call__(
        self,
        autograd_function: type[torch.autograd.Function],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Any:
        # [FUNCTION DOC] Invoke the operator on an autograd.Function class. If any functorch transform
        # is active, dispatch through the higher-order-operator machinery (transform-specific rules);
        # otherwise directly call the function's apply with args and kwargs. Returns the function's outputs.
        # When custom_function_call is done dispatching through functorch,
        # it should just invoke the autograd.Function. This is consistent
        # with the autograd.Function behavior of being invoked before the
        # PyTorch dispatcher.
        #
        # This will lead us into trouble later down the line, but this is
        # pre-existing. There is an invariant that a function traced by
        # make_fx should have the same behavior when provided the same
        # Tensor. However, make_fx sees autograd.Function as a composite
        # (because autograd.Function happens before the Python dispatch key)
        # and only traces the forward pass.
        if torch._C._are_functorch_transforms_active():
            # pyrefly: ignore [missing-attribute]
            return super().__call__(autograd_function, *args, **kwargs)
        return autograd_function.apply(*args, **kwargs)


# "custom_function_call"
# This is the mechanism for an autograd.Function that works with functorch transforms.
# It wraps an autograd.Function; interactions with functorch transforms are defined
# via PyDispatcher and HigherOrderOperator rather than through the traditional PyTorch
# dispatcher.
custom_function_call = CustomFunctionHigherOrderOperator()


# The grad rule for custom_function_call is to construct a new _SingleLevelFunction
# (autograd.Function that only works with a single layer (level) of functorch) that:
# - unwraps the inputs
# - redispatches to custom_function_call
# - wraps the outputs
# and whose backward pass calls the original autograd.Function's backward.
#
# Why do we need to redispatch to custom_function_call?
# -----------------------------------------------------
# This is consistent with how ATen operators work with functorch's grad transform:
# they always redispatch to the original operator.
# Consider torch.sin, and let's say we do grad0(grad1(torch.sin))(x)
#
# grad1 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin (*)
# - rewrap the outputs on the return
#
# On the redispatch in (*), grad0 will:
# - set up the autograd graph
# - unwrap the inputs
# - redispatch to at::sin
# - rewrap the outputs on the return
#
# To "set up the autograd graph", we generate a _SingleLevelFunction
# and apply it.
@custom_function_call.py_impl(TransformType.Grad)
@custom_function_call.py_impl(TransformType.Jvp)
def custom_function_call_grad(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
) -> Any:
    # [FUNCTION DOC] Grad/Jvp rule for custom_function_call. Builds a single-level autograd.Function
    # wrapping autograd_function for the interpreter's level and applies it to the operands with
    # single-level autograd-function mode enabled. Returns the (wrapped) outputs.
    Generated = generate_single_level_function(interpreter, autograd_function)
    with enable_single_level_autograd_function():
        # pyrefly: ignore [missing-attribute]
        flat_out = Generated.apply(*operands)
    return flat_out


def generate_single_level_function(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
) -> type[torch.autograd.function._SingleLevelFunction]:
    # [FUNCTION DOC] Dynamically creates a _SingleLevelFunction subclass (named after the original
    # function plus Generated) for one functorch level. Its forward unwraps inputs at that level,
    # re-enters custom_function_call with grad mode and forward-grad enabled and the interpreter
    # lowered, then rewraps outputs preserving identity of returned inputs. setup_context, backward and
    # jvp delegate to the original function. Returns the new class; no global side effects.
    level = interpreter.level()

    def forward(*operands: Any) -> Any:
        # [FUNCTION DOC] HELPER: Unwrap tensor operands for this level, run the original function via
        # custom_function_call under enabled grad modes, and wrap tensor outputs back (keeping identity of
        # outputs that are the unwrapped inputs, needed for mark_dirty). Returns the wrapped outputs.
        unwrapped_operands = pytree.tree_map_only(
            torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands
        )
        # Both enable_grad() and _set_fwd_grad_enabled() are necessary no matter
        # the transform. _SingleLevelFunction will turn off both fwd and bwd
        # gradient computation and we need to turn it back on here.
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            unwrapped_output = custom_function_call(
                autograd_function, *unwrapped_operands
            )

        # See NOTE [mark_dirty object identity check]
        def wrap_fn(output: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Wrap a tensor output for grad at the enclosing level.
            return _wrap_for_grad(output, level)

        return wrap_outputs_maintaining_identity(
            unwrapped_output, unwrapped_operands, operands, wrap_fn
        )

    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forwards to the original autograd.Function's setup_context.
        return autograd_function.setup_context(ctx, inputs, output)

    # backward is only used if the transform is TransformType.Grad
    def backward(ctx: Any, *grads: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forwards to the original function's backward; returns its gradients.
        result = autograd_function.backward(ctx, *grads)
        return result

    # jvp is only used if the transform is TransformType.Jvp
    def jvp(ctx: Any, *tangents: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forwards to the original function's jvp; returns output tangents.
        result = autograd_function.jvp(ctx, *tangents)
        return result

    # This is the sequence of magic words to dynamically generate a Subclass with
    # a given name. A Tensor's .grad_fn field has a class name that is the original
    # autograd.Function's name + Backward, so we do this to generate some
    # meaningful name.
    name = f"{autograd_function.__name__}Generated"
    Generated = type(
        name,
        (torch.autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
        },
    )
    return Generated


# wrap_outputs_maintaining_identity handles outputs from the vmap,
# backward (vjp), and jvp staticmethod. The way it distinguishes
# between the vmap case and the {backward, jvp} case is if the out_dims
# are specified or not.
#
# NB: we cannot use out_dims=None as the deciding factor. This because
# out_dims=None can still happen in the vmap staticmethod! What the
# user is saying in that case is that their output does not have a
# dimension that is being vmapped over, which is valid.
NO_OUT_DIMS = "not specified"


# NOTE [mark_dirty object identity check]
# autograd.Function's ctx.mark_dirty expect a returned input
# to have the same object identity as the input.
# Mode-only functorch will greatly simplify this logic.
def wrap_outputs_maintaining_identity(
    outputs: Any,
    unwrapped_inputs: Any,
    orig_inputs: Any,
    wrap_fn: Callable[..., Any],
    out_dims: Any = NO_OUT_DIMS,
) -> Any:
    # [FUNCTION DOC] Wraps the tensors in outputs using wrap_fn, except that any output that is
    # literally (same object as) one of the unwrapped inputs is replaced by the corresponding original
    # input, preserving identity for mark_dirty. Non-tensor outputs pass through. If out_dims is
    # specified (vmap case) it is broadcast to the output structure and wrap_fn receives the per-output
    # dim; raises RuntimeError if out_dims is incompatible with the output structure. Returns the outputs
    # in their original pytree structure.
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)

    unwrapped_input_to_orig_input = {
        id(unwrapped): orig
        for unwrapped, orig in zip(flat_unwrapped_inputs, flat_orig_inputs)
    }

    flat_outputs, spec = pytree.tree_flatten(outputs)
    result = []

    out_dims_specified = out_dims != NO_OUT_DIMS

    flat_out_dims = None
    if out_dims_specified:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        # _broadcast_to_and_flatten returns None if it is unable to broadcast.
        # TODO: update following link from master to stable once that's out
        if flat_out_dims is None:
            raise RuntimeError(
                f"The autograd.Function's vmap staticmethod returned an "
                f"incompatible (output, out_dims) tuple. "
                f"Expected out_dims={out_dims} "
                f"to be compatible with the structure of `output`. "
                f"out_dims has structure {pytree.tree_flatten(out_dims)[1]} "
                f"but output has structure {spec}. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )

    for i, output in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            if flat_out_dims is None:
                raise AssertionError(
                    "flat_out_dims must not be None when out_dims is specified"
                )
            result.append(wrap_fn(output, flat_out_dims[i]))
        else:
            result.append(wrap_fn(output))

    return pytree.tree_unflatten(result, spec)


# NOTE: [functorch vjp and autograd interaction]
# There's an edge case with the functorch vjp and autograd interaction
# that will eventually be fixed by mode-only functorch.
# The TL;DR is that there's no way to unwrap a dead GradTensorWrapper,
# so we (the framework) need to do it manually. Regular PyTorch operators
# automatically do so this is consistent.
#
# class MyExp(torch.autograd.Function):
#     @staticmethod
#     def forward(x):
#         return x.exp()
#
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         y = output
#         ctx.save_for_backward(y)
#
#     @staticmethod
#     def backward(gy):
#         y, = ctx.saved_tensors()
#         return MyMul.apply(gy, y)
#
# x = torch.randn([], requires_grad=True)
# gy = torch.randn([], requires_grad=True)
# _, vjp_fn = vjp(MySin.apply, x)
# result = vjp_fn(gy)
#
# MyMul is an autograd.Function that is not shown here.
# It saves a `y` for backward (since gy requires grad).
#
# in vjp_fn(gy), we get:
# > MyMul.apply(gy, GradTensorWrapper(y, level=dead))
# Because the y that is saved for backward by MyExp is a GradTensorWrapper
# but is now dead since we are outside the vjp context.
#
# PyTorch dispatcher operations, upon seeing a dead GradTensorWrapper,
# will automatically unwrap the GradTensorWrapper when applied.
# But since autograd.Function technically sits above the regular PyTorch
# dispatcher, it doesn't get this treatment. So we manually do
# the unwrapping to be consistent with regular PyTorch dispatcher operations.


class VmapInfo(NamedTuple):
    # [CLASS DOC] Immutable record passed as the first argument to an autograd.Function's vmap
    # staticmethod: batch_size is the size of the vmapped dimension, randomness is the vmap randomness mode string.
    batch_size: int
    randomness: str


def has_overridden_vmap_rule(
    autograd_function: type[torch.autograd.Function],
) -> bool:
    # [FUNCTION DOC] HELPER: True if the class defines its own vmap staticmethod (differs from the base
    # torch.autograd.Function.vmap).
    return autograd_function.vmap is not torch.autograd.Function.vmap


def validate_vmap_returns_tuple_of_two_elements(result: Any) -> None:
    # [FUNCTION DOC] HELPER: Raises RuntimeError unless result is a tuple of exactly two elements
    # (output, out_dims). Returns None.
    base_error_msg = (
        "Expected the vmap staticmethod to have two returns, an output "
        "and out_dims with pytree structure compatible with the output. "
    )
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f"Got a {type(result)} instead")
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f"Got {len(result)} returns instead")


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(
    interpreter: VmapInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC] Vmap rule for custom_function_call. Raises NotImplementedError if any kwarg holds
    # a tensor. If the function sets generate_vmap_rule, it must not also override vmap (RuntimeError
    # on ambiguity) and the rule is auto-generated; otherwise the function must override vmap
    # (RuntimeError if not), and the user's vmap staticmethod is run through the helper. Returns outputs.
    if any(
        isinstance(val, torch.Tensor)
        for val in torch.utils._pytree.tree_flatten(kwargs)[0]
    ):
        raise NotImplementedError(
            f"Run vmap on autograd.Function with kwarg-only Tensor args. "
            f"Please do not pass kwarg-only Tensors to autograd.Function. "
            f"Got: {kwargs}"
        )

    if autograd_function.generate_vmap_rule:
        if has_overridden_vmap_rule(autograd_function):
            # TODO: Update link to stable once that's out
            # https://github.com/pytorch/pytorch/issues/92029
            raise RuntimeError(
                f"You tried to vmap over {autograd_function.__name__}, but "
                f"it has both generate_vmap_rule=True and an overridden vmap "
                f"staticmethod. Please set generate_vmap_rule=False or delete "
                f"the overridden vmap staticmethod to avoid ambiguity. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )
        return custom_function_call_vmap_generate_rule(
            interpreter, autograd_function, *operands
        )

    if not has_overridden_vmap_rule(autograd_function):
        # TODO: Update link to stable once that's out
        # https://github.com/pytorch/pytorch/issues/92029
        raise RuntimeError(
            f"You tried to vmap over {autograd_function.__name__}, but "
            f"it does not have vmap support. Please override and implement the "
            f"vmap staticmethod or set generate_vmap_rule=True. "
            f"For more details, please see "
            f"https://pytorch.org/docs/main/notes/extending.func.html"
        )

    return custom_function_call_vmap_helper(
        interpreter, autograd_function.vmap, autograd_function, *operands, **kwargs
    )


def custom_function_call_vmap_helper(
    interpreter: VmapInterpreter,
    vmap_function: Callable[..., Any],
    op: Any,
    *operands: Any,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC] Runs a user vmap rule. Unwraps batched operands at this level to get per-operand
    # batch dims. If no operand is batched here, lowers to the next level and calls the op unbatched.
    # Otherwise calls vmap_function with a VmapInfo, the in_dims and unwrapped operands (levels lowered),
    # validates it returned (output, out_dims), and rewraps outputs as batched tensors at this level
    # (out_dim None means unbatched), preserving identity of returned inputs. op is either an
    # autograd.Function class or a library-registered op; the latter lowers by excluding the batched dispatch key.
    current_level = interpreter.level()
    info = VmapInfo(
        batch_size=interpreter.batch_size(),
        randomness=interpreter.randomness(),
    )
    # We're either in the autograd.Function case (vmap staticmethod)
    # or the torch.library.register_vmap case.
    autograd_function_case = isinstance(op, torch.autograd.function.FunctionMeta)

    def lower_to_next() -> Any:
        # [FUNCTION DOC] HELPER: Returns a context manager that moves dispatch below this vmap level:
        # the interpreter's lower guard for autograd.Function, or a guard excluding the batched dispatch key.
        if autograd_function_case:
            return interpreter.lower()
        else:
            return torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.FuncTorchBatched)
            )

    unwrapped_operands, in_dims = unwrap_batched(operands, current_level)
    # If none of the tensors are batched at the current level, then we skip the
    # current level. This saves the user from needing to handle this case in
    # their vmap staticmethod (and is consistent with our C++ batching rule API)
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with lower_to_next():
            if autograd_function_case:
                return custom_function_call(op, *operands)
            else:
                return op(*operands, **kwargs)

    with lower_to_next():
        result = vmap_function(info, in_dims, *unwrapped_operands, **kwargs)
    validate_vmap_returns_tuple_of_two_elements(result)
    unwrapped_output, out_dims = result

    # See NOTE [mark_dirty object identity check]
    def wrap_fn(output: torch.Tensor, out_dim: int | None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns output unchanged if out_dim is None, else adds a batch dim at out_dim for the current level.
        return (
            output
            if out_dim is None
            else _add_batch_dim(output, out_dim, current_level)
        )

    return wrap_outputs_maintaining_identity(
        unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims
    )


def unpack_outputs(outputs: tuple[Any, ...]) -> tuple[Any, Any]:
    # [FUNCTION DOC] HELPER: Splits a vmapped forward's flat result into (outputs, out_dims). The last
    # element is out_dims; if it is a tuple the remaining elements are the outputs (as a tuple), else
    # the outputs are the single first element.
    out_dims = outputs[-1]
    if isinstance(out_dims, tuple):
        outputs = outputs[:-1]
    else:
        outputs = outputs[0]
    return outputs, out_dims


def custom_function_call_vmap_generate_rule(
    interpreter: VmapInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
) -> Any:
    # [FUNCTION DOC] Auto-generated vmap rule (generate_vmap_rule=True). Unwraps batched operands,
    # builds a vmapped autograd.Function class for the observed in_dims, batch size and randomness,
    # runs it below this level via custom_function_call, unpacks (outputs, out_dims) and wraps the
    # outputs as batched tensors at this level.
    unwrapped_operands, in_dims = unwrap_batched(operands, interpreter.level())
    vmapped_function = vmapify_autograd_function(
        autograd_function,
        in_dims,
        interpreter.batch_size(),
        interpreter.randomness(),
    )
    with interpreter.lower():
        outputs = custom_function_call(vmapped_function, *unwrapped_operands)

    if not isinstance(outputs, tuple):
        raise AssertionError(f"expected outputs to be a tuple, got {type(outputs)}")
    outputs, out_dims = unpack_outputs(outputs)
    return wrap_batched(outputs, out_dims, interpreter.level())


@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
    generate_vmap_rule: bool,
    *operands: Any,
) -> Any:
    # [FUNCTION DOC] Functionalize rule for custom_function_call: not implemented; always raises RuntimeError.
    # KNOWN ISSUE: the signature takes a generate_vmap_rule argument before operands, unlike other rules.
    raise RuntimeError("NYI: Functionalize rule for custom_function_call")


def vmapify_autograd_function(
    autograd_function: type[torch.autograd.Function],
    in_dims: Any,
    batch_size: int,
    randomness: str,
) -> type[torch.autograd.Function]:
    # [FUNCTION DOC] Builds a new autograd.Function subclass (named Vmapped plus the original name,
    # generate_vmap_rule set) whose forward, setup_context, backward and jvp run the original
    # function's corresponding methods under vmap with the given in_dims, batch_size and randomness.
    # Per-class state (input shapes, saved-tensor batch dims, out_dims) is stored on ctx keyed by the
    # generated class id. Gradients are reduced via reductify. Returns the class.
    def forward(*operands: Any) -> Any:
        # [FUNCTION DOC] HELPER: Runs the original forward under vmap; returns the outputs with out_dims
        # appended as the last element (flattened if outputs is a tuple).
        outputs, out_dims = restore_vmap(
            autograd_function.forward, in_dims, batch_size, randomness
        )(*operands)
        if isinstance(outputs, torch.Tensor):
            return outputs, out_dims
        else:
            return *outputs, out_dims

    def setup_context(ctx: Any, inputs: Any, outputs: Any) -> None:
        # [FUNCTION DOC] HELPER: Runs the original setup_context under vmap with a ctx wrapper that
        # unwraps saved tensors and records their batch dims. Stores on ctx (keyed by generated class id)
        # the input shapes, saved-tensor bdims and out_dims, creating those mappings if absent.
        outputs, out_dims = unpack_outputs(outputs)
        key = id(Generated)

        def inner(inputs: Any, outputs: Any) -> None:
            # [FUNCTION DOC] HELPER: Body executed under vmap: calls the original setup_context with a
            # saving-aware ctx wrapper and records input shapes and saved-tensor bdims on the real ctx.
            # wrapped_ctx.save_for_backward will:
            # - unwrap batchedtensors into (tensor, bdim)
            # - save_for_backward(*unwrapped_tensors)
            # - assign the bdims to wrapped_ctx._pt_saved_tensors_bdims
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)

            # input_shapes are used for reductify later to reduce expanded gradients
            # to the correct shape.
            # See NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
            # for more details
            input_shapes = tuple(
                inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs
            )
            if not hasattr(ctx, "_pt_input_shapes"):
                # pyrefly: ignore [implicit-any]
                ctx._pt_input_shapes = {}
            ctx._pt_input_shapes.update({key: input_shapes})

            if not hasattr(ctx, "_pt_saved_tensors_bdims_stack"):
                # pyrefly: ignore [implicit-any]
                ctx._pt_saved_tensors_bdims_stack = {}
            ctx._pt_saved_tensors_bdims_stack.update(
                {key: (wrapped_ctx._pt_saved_tensors_bdims)}
            )

        # See NOTE: [Why do we need to run setup_context under a vmap?]
        restore_vmap(
            inner,
            (in_dims, out_dims),
            batch_size,
            randomness,
        )(inputs, outputs)

        if not hasattr(ctx, "_pt_out_dims"):
            # pyrefly: ignore [implicit-any]
            ctx._pt_out_dims = {}
        ctx._pt_out_dims.update({key: out_dims})

    def jvp(ctx: Any, *tangents: Any) -> Any:
        # [FUNCTION DOC] HELPER: Runs the original jvp under vmap using the saved tensors and their
        # recorded batch dims, with None tangents given a None in_dim; reduces the result to out_dims
        # and returns it with a trailing None (for the extra out_dims output).
        key = id(Generated)

        def jvp_no_context(saved_tensors: Any, tangents: Any) -> Any:
            # [FUNCTION DOC] HELPER: Calls the original jvp with a ctx whose saved_tensors are overridden.
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)

        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        out_tangents, out_tangents_dims = restore_vmap(
            jvp_no_context,
            (ctx._pt_saved_tensors_bdims_stack[key], tangent_in_dims),
            batch_size,
            randomness,
        )(ctx.saved_tensors, tangents)

        result = reductify(
            out_tangents, out_tangents_dims, ctx._pt_out_dims[key], batch_size
        )
        if isinstance(result, torch.Tensor):
            return result, None
        else:
            return *result, None

    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Runs the original backward under vmap. Drops the gradient for the extra
        # out_dims output, uses the stored out_dims as grad in_dims (None where a grad is None), then
        # reduces resulting grads to the input bdims and recorded input shapes via reductify. Returns the tuple of grads.
        key = id(Generated)
        grad_outputs_ = grad_outputs[:-1]
        grad_outputs_in_dims = ctx._pt_out_dims[key]

        if not isinstance(grad_outputs_in_dims, tuple):
            grad_outputs_in_dims = (grad_outputs_in_dims,)

        grad_outputs_in_dims = tuple(
            in_dim if grad_output is not None else None
            for grad_output, in_dim in zip(grad_outputs_, grad_outputs_in_dims)
        )

        def backward_no_context(inputs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Unpacks (saved_tensors, grad_outputs) and calls the original backward with a ctx overriding saved_tensors.
            saved_tensors, grad_outputs = inputs
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)

        grad_ins, grad_ins_dims = restore_vmap(
            backward_no_context,
            ((ctx._pt_saved_tensors_bdims_stack[key], grad_outputs_in_dims),),
            batch_size,
            randomness,
        )((ctx.saved_tensors, grad_outputs_))
        result = reductify(
            grad_ins, grad_ins_dims, in_dims, batch_size, ctx._pt_input_shapes[key]
        )
        return result

    name = f"Vmapped{autograd_function.__name__}"
    Generated = type(
        name,
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
            "generate_vmap_rule": True,
        },
    )

    return Generated


# tangents might be None, so we need to replace
# the corresponding in_dims with None.
def get_tangents_in_dims(input_dims: Any, tangents: tuple[Any, ...]) -> Any:
    # [FUNCTION DOC] HELPER: Flattens input_dims and tangents, replaces the in_dim with None wherever
    # the matching tangent is None, and returns the result in the original input_dims structure.
    flat_in_dims, spec = pytree.tree_flatten(input_dims)
    flat_tangents = pytree.arg_tree_leaves(*tangents)
    result = [
        None if tangent is None else in_dim
        for in_dim, tangent in zip(flat_in_dims, flat_tangents)
    ]
    return pytree.tree_unflatten(result, spec)


# NOTE: [Why do we need to run setup_context under a vmap?]
# Consider the following autograd.Function
#
# class Sum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return x.sum()
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        ctx.x_shape = inputs[0]
#    @staticmethod
#    def backward(ctx, gy):
#        return gy.expand(ctx.x_shape)
#
# x = torch.randn(B, 4)
# in_dims = 0
# vmap(Sum.apply, in_dims)(x)
#
# Let's assume for a moment that we didn't vmap setup_context in VmappedSum:
#
# class VmappedSum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return vmap(Sum.forward, in_dims)(x)
#
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        Sum.setup_context(ctx, inputs, outputs)
#
#    @staticmethod
#    def backward(ctx, gy):
#        def backward_no_context(gy):
#            return gy.expand(ctx.x_shape)
#
#        dims = (0,)
#        gx = vmap(backward_no_context, dims)(gy)
#        return gx
#
# We end up saving [B, 4] as x_shape. In the backward, gy has shape [B],
# and we're doing:
#
# def backward_no_context(gy):
#     return gy.expand([B, 4])
#
# gx = vmap(backward_no_context, dims)(gy: "Tensor[B]")
#
# This gives us the wrong result (gx has shape [B, B, 4], but it should
# have shape [4]). Performing vmap over setup_context means the shape
# saved has shape [4] and leads to a correct result shape for gx.


# Wraps a ctx object. Forwards all attr accesses to the underlying object
# except for the attrs in _pt_attrs
class WrappedCtx:
    # [CLASS DOC] HELPER: Proxy around an autograd ctx. Attribute reads and writes are forwarded to the
    # inner ctx (_pt_inner_ctx) except names in _pt_reserved_attrs, which live on the wrapper itself.
    # Invariant: the user's ctx must not already have the reserved attribute names.
    _pt_reserved_attrs: tuple[str, ...] = ("_pt_reserved_attrs", "_pt_inner_ctx")

    def __init__(self, ctx: Any) -> None:
        # [FUNCTION DOC] HELPER: Stores ctx as the inner ctx. If ctx is not itself a WrappedCtx, raises
        # RuntimeError when it already has any reserved attribute name.
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                raise RuntimeError(
                    f"PyTorch reserves the {reserved_attrs} field on ctx. "
                    "Please name your fields on ctx something else to avoid name "
                    "collision."
                )
        self._pt_inner_ctx = ctx

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] HELPER: Called for attributes not found on the wrapper; reads from the inner ctx.
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # [FUNCTION DOC] HELPER: Reserved names are stored on the wrapper's own dict; all other names are set on the inner ctx.
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        return setattr(self._pt_inner_ctx, name, value)


# Wraps ctx to create a new ctx object that overrides saved_tensors.
class CtxWithSavedTensors(WrappedCtx):
    # [CLASS DOC] HELPER: ctx wrapper whose saved_tensors property returns a caller-supplied sequence
    # (_pt_new_saved_tensors) instead of the real ctx's saved tensors; everything else is forwarded.
    _pt_reserved_attrs = ("_pt_new_saved_tensors", *WrappedCtx._pt_reserved_attrs)

    def __init__(self, ctx: Any, new_saved_tensors: Sequence[torch.Tensor]) -> None:
        # [FUNCTION DOC] HELPER: Wraps ctx and records the replacement saved tensors.
        super().__init__(ctx)
        self._pt_new_saved_tensors = new_saved_tensors

    @property
    def saved_tensors(self) -> Sequence[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Returns the replacement saved tensors.
        return self._pt_new_saved_tensors


class CtxCustomSave(WrappedCtx):
    # [CLASS DOC] HELPER: ctx wrapper used in setup_context under vmap. save_for_backward and
    # save_for_forward unwrap batched tensors at the current level, save the unwrapped tensors on the
    # real ctx and remember their batch dims in _pt_saved_tensors_bdims (overwritten on each save call).
    _pt_reserved_attrs = (
        "_pt_saved_tensors_bdims",
        "_pt_current_level",
        *WrappedCtx._pt_reserved_attrs,
    )

    def __init__(self, ctx: Any, current_level: int) -> None:
        # [FUNCTION DOC] HELPER: Wraps ctx, with empty saved bdims and the vmap level to unwrap at.
        super().__init__(ctx)
        self._pt_saved_tensors_bdims: tuple[Any, ...] = ()
        self._pt_current_level = current_level

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Unwraps batched tensors, saves them for backward on the inner ctx and records their batch dims.
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_backward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims

    def save_for_forward(self, *tensors: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Unwraps batched tensors, saves them for forward-mode AD on the inner ctx and records their batch dims.
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims


def reductify(
    grad_input: torch.Tensor | tuple[torch.Tensor, ...],
    grad_input_bdim: int | tuple[int, ...],
    input_bdim: int | tuple[int, ...],
    batch_size: int,
    target_shape_without_bdim_to_reduce_to: Any = None,
) -> tuple[Any, ...]:
    # [FUNCTION DOC] Normalizes grad_input and both bdim arguments to tuples, then applies
    # reductify_leaf elementwise (zipped with the optional per-input target shapes, defaulting to
    # none). Returns a tuple of reduced gradients.
    if not isinstance(grad_input, tuple):
        grad_input = (grad_input,)
    if not isinstance(grad_input_bdim, tuple):
        grad_input_bdim = (grad_input_bdim,)
    if not isinstance(input_bdim, tuple):
        input_bdim = (input_bdim,)

    if target_shape_without_bdim_to_reduce_to is None:
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    result = tuple(
        reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape)
        for gi, gi_bdim, i_bdim, maybe_ishape in zip(
            grad_input,
            grad_input_bdim,
            input_bdim,
            target_shape_without_bdim_to_reduce_to,
        )
    )
    return result


def reductify_leaf(
    grad_input: torch.Tensor | None,
    grad_input_bdim: int | None,
    input_bdim: int | None,
    batch_size: int,
    target_shape_without_bdim_to_reduce_to: Any = None,
) -> torch.Tensor | None:
    # [FUNCTION DOC] Reconciles one gradient with its input's batch dim. None stays None; if neither is
    # batched return as is; if only the grad is batched sum over its bdim. If the input is batched but
    # the grad is not, expand the grad along a new dim of batch_size at the input bdim. If a target
    # shape is given, sum the grad down to that shape under vmap; else just move the grad's bdim to the input's.
    if grad_input is None:
        return None

    if grad_input_bdim is None and input_bdim is None:
        return grad_input

    if grad_input_bdim is not None and input_bdim is None:
        return grad_input.sum(grad_input_bdim)

    # NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
    # For reverse-mode AD,
    # given a grad_input and input, it is valid for the user to return a
    # grad_input that has a broadcasted shape when compared to the input.
    # In this situation, autograd automatically reduces the grad_input to
    # the shape of the input.
    #
    # However, when input_bdim is not None, we have problems.
    #
    # [example 1]
    # grad_input: Tensor[3, 4], input: Tensor[B, 4]
    # We can expand grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # [example 2]
    # grad_input: Tensor[3, B, 4], input: Tensor[B, 4]
    # We can swizzle grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # This means that we need to also reduce the grad_input to the shape of the
    # input. This behavior is controlled by the `target_shape_without_bdim_to_reduce_to` flag;
    # if not-None then we do the reducing manually, otherwise, we do not do a reduction.
    if input_bdim is None:
        raise AssertionError("input_bdim must not be None")

    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size
        grad_input = grad_input.expand(new_shape)
        grad_input_bdim = input_bdim

    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(
            torch.Tensor.sum_to_size,
            in_dims=(grad_input_bdim, None),
            out_dims=input_bdim,
        )(grad_input, target_shape_without_bdim_to_reduce_to)

    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)
    return grad_input


def autograd_function_forward_rewritten(
    original_forward: Callable[_P, _R],
    original_setup_context: Callable[..., Any],
) -> Callable[..., _R]:
    # [FUNCTION DOC] Combines a forward without ctx and a setup_context into a single forward taking
    # ctx first (the legacy style). Returns the combined function.
    def new_forward(ctx: Any, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Runs the original forward, then setup_context with ctx, the args tuple and the output; returns the output.
        output = original_forward(*args, **kwargs)
        original_setup_context(ctx, args, output)
        return output

    return new_forward


class AutogradFunctionApply(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator autograd_function_apply used by Dynamo to represent an
    # autograd.Function as a forward graph module and a backward graph module. Stateless itself;
    # each call builds a fresh autograd.Function template.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name autograd_function_apply.
        super().__init__("autograd_function_apply")

    def __call__(
        self,
        fwd: torch.fx.GraphModule,
        bwd: torch.fx.GraphModule,
        *fwd_args: Any,
        **fwd_kwargs: Any,
    ) -> Any:
        # [FUNCTION DOC] Runs fwd/bwd graph modules as a real autograd.Function. fwd_kwargs must provide
        # non_differentiable_idx and saved_for_backward_idx (KeyError if missing) and may provide
        # dirty_idx. Forward interprets fwd to get (output, saved_values), tagging proxies of saved
        # values not from save_for_backward with node metadata; setup_context marks dirty and
        # non-differentiable outputs; backward interprets bwd on grads plus saved values. Returns the
        # outputs of applying the template to fwd_args.
        saved_values: Iterable[Any] | None = None
        dirty_idx = fwd_kwargs.get("dirty_idx", [])
        dirty_idx_set = set(dirty_idx)
        non_differentiable_idx = fwd_kwargs["non_differentiable_idx"]
        saved_for_backward_idx = fwd_kwargs["saved_for_backward_idx"]

        class ApplyTemplate(torch.autograd.Function):
            # [CLASS DOC] HELPER: Per-call autograd.Function whose forward/backward execute the captured
            # fwd and bwd graph modules; shares saved_values with the enclosing call via closure.
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interprets fwd on args, stores the saved values in the enclosing
                # scope, flags proxies of non-save_for_backward saved tensors in node meta, returns output.
                nonlocal saved_values

                # The Interpreter here is required to propagate metadata
                # from the dynamo graph body to the local_map graph body.
                # This is required for fx_traceback.annotate for work.
                output, saved_values = torch.fx.Interpreter(fwd).run(*args)

                # See Note [Activations with no version counter checks in eager]
                # Mark tensors that came from ctx.save_for_backward with metadata.
                # This allows AOT autograd to distinguish between tensors saved via
                # save_for_backward vs those stashed directly on ctx (e.g., ctx.x = x).
                from torch.fx.experimental.proxy_tensor import _get_proxies

                for idx, t in enumerate(saved_values):
                    if idx not in saved_for_backward_idx:
                        for proxy in _get_proxies(t):
                            proxy.node.meta["saved_tensor_with_no_vc_check"] = True

                return output

            @staticmethod
            def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
                # [FUNCTION DOC] HELPER: Calls mark_dirty and mark_non_differentiable on ctx for the outputs at the recorded indices (when any).
                def selected_outputs(idx: Iterable[int]) -> list[Any]:
                    # [FUNCTION DOC] HELPER: Returns the outputs at positions in idx (a single non-sequence output counts as position 0).
                    idx_set = set(idx)
                    if isinstance(output, (tuple, list)):
                        return [x for i, x in enumerate(output) if i in idx_set]
                    return [output] if 0 in idx_set else []

                # If users call ctx.mark_dirty() in the original fwd function.
                if len(dirty_idx) > 0:
                    ctx.mark_dirty(*selected_outputs(dirty_idx_set))

                # If users call ctx.mark_non_differentiable() in the original fwd function.
                if len(non_differentiable_idx) > 0:
                    ctx.mark_non_differentiable(
                        *selected_outputs(non_differentiable_idx)
                    )

            @staticmethod
            def backward(ctx: Any, *grad: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interprets bwd on the incoming grads followed by the saved
                # values from forward; raises AssertionError if forward has not stored saved values.
                # The Interpreter here is required to propagate metadata
                # from the dynamo graph body to the local_map graph body.
                # This is required for fx_traceback.annotate for work.

                if saved_values is None:
                    raise AssertionError("saved_values must not be None")
                return torch.fx.Interpreter(bwd).run(*grad, *saved_values)

        return ApplyTemplate.apply(*fwd_args)


autograd_function_apply = AutogradFunctionApply()


class DynamoAutogradFunctionTraceHelper:
    # [CLASS DOC] Namespace (no state) of static helpers that re-express parts of autograd.Function's
    # C++ forward behavior in Python so Dynamo can trace them.
    @staticmethod
    def fwd_trace_helper(orig_fwd: Callable[_P, Any]) -> Callable[_P, Any]:
        # [FUNCTION DOC] Wraps a forward function. The wrapper runs it under no_grad, then for each
        # tensor output that is the same object as an input and not marked dirty on ctx (first arg),
        # returns a view_as of it, mirroring the C++ behavior; other outputs pass through. A single
        # tensor returns as such; otherwise a tuple.
        # KNOWN ISSUE: per the TODO, requires_grad changes from mark_non_differentiable are not replicated.
        # autograd.Function forward does more than just running the forward method. Most
        # of this logic is in C++. Here, we rewrite that functionality in python and let
        # Dynamo trace it.
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            # [FUNCTION DOC] HELPER: Runs orig_fwd under no_grad and converts pass-through inputs to views as described above.
            with torch.no_grad():
                outs = orig_fwd(*args, **kwargs)

            ctx = args[0]
            dirty_tensors = getattr(ctx, "dirty_tensors", None)

            def is_dirty_tensor(tensor: torch.Tensor) -> bool:
                # [FUNCTION DOC] HELPER: True if tensor is (by identity) among ctx.dirty_tensors; False if ctx has none.
                if dirty_tensors is None:
                    return False
                return any(
                    tensor is dirty_tensor
                    for dirty_tensor in dirty_tensors
                    if isinstance(dirty_tensor, torch.Tensor)
                )

            def is_tensor_arg(tensor: torch.Tensor) -> bool:
                # [FUNCTION DOC] HELPER: True if tensor is (by identity) one of the positional args.
                return any(
                    arg is tensor for arg in args if isinstance(arg, torch.Tensor)
                )

            # Handle the case where if the input is passed on directly to the output, we call view_as
            # Refer to https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/custom_function.cpp#L254
            if isinstance(outs, torch.Tensor):
                if is_tensor_arg(outs) and not is_dirty_tensor(outs):
                    return outs.view_as(outs)
                else:
                    return outs

            new_outs = []
            for out in outs:
                if isinstance(out, torch.Tensor):
                    if is_tensor_arg(out) and not is_dirty_tensor(out):
                        new_outs.append(out.view_as(out))
                    else:
                        new_outs.append(out)
                else:
                    new_outs.append(out)
            return tuple(new_outs)

            # TODO - there is missing functionality here, where
            # autograd.Function overwrites the requires_grad_ of the output
            # tensors depending on the `mark_non_differentiable`. Currently,
            # this is handled hackily in Dynamo, where we just overwrite the
            # variable trackers requires_grad. Refer to the function -
            # overwrite_tensor_vt_requires_grad

        return inner
