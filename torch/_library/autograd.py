# mypy: allow-untyped-defs
# [MODULE DOC] Autograd integration for custom operators: builds an Autograd-key kernel
# from user-supplied backward / setup_context functions (make_autograd_impl) and provides
# supports_tensorlist, which lets an autograd.Function accept and return lists of tensors
# by flattening them with pytree. No import-time side effects beyond definitions.
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch import _C, _ops, autograd, Tensor
from torch.utils import _pytree

from . import utils


class InfoProtocol(Protocol):
    # [CLASS DOC] HELPER: Structural type for objects holding an optional backward function
    # and an optional setup_context function for an operator.
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


@dataclasses.dataclass
class Info:
    # [CLASS DOC] HELPER: Dataclass implementing InfoProtocol; stores the backward and
    # setup_context callables (each may be None) registered for an operator.
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    # [FUNCTION DOC] Builds and returns an autograd kernel (taking a dispatch keyset plus
    # op args) for op. It creates a dynamic autograd.Function whose forward redispatches
    # below autograd and calls info's setup_context (with defaults filled in), and whose
    # backward calls info's backward and validates the gradient count. Out-variant ops
    # raise if any input requires grad; otherwise grad-mode and requires-grad inputs
    # choose between the Function and a plain no-grad redispatch. Tensor-list args wrap
    # the Function with supports_tensorlist.
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"

    schema = op._schema
    is_out_op = utils.is_out(op)
    has_kwarg_only_args = utils.has_kwarg_only_args(op._schema)
    num_positional_args = sum(not a.kwarg_only for a in schema.arguments)
    has_tensorlist_like_args = any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    )

    @dataclass
    class Metadata:
        # [CLASS DOC] HELPER: Extra trailing forward argument carrying the dispatch keyset
        # and the keyword-only args, which autograd.Function cannot take as kwargs.
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    def redispatch_no_grad(keyset, args, kwargs):
        # [FUNCTION DOC] HELPER: Redispatches op below the autograd key with autograd
        # disabled for the call, restricting keyset to keys after autograd; returns result.
        with _C._AutoDispatchBelowAutograd():
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    def forward(ctx, *args):
        # [FUNCTION DOC] HELPER: Forward of the generated Function. The last positional arg
        # is a Metadata object; the rest are op args. Redispatches the op below autograd,
        # then, if a setup_context function exists, fills default args back in and calls it
        # (with keyword-only inputs separately when the schema has any). Returns the result.
        metadata = args[-1]
        args = args[:-1]

        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            if info._setup_context_fn:
                # The Dispatcher will remove args that are equal to their default
                # values from (args, kwargs). We're going to add it back so that
                # the user can access them.
                #
                # This is OK to do: The Dispatcher removed the args for serialization
                # FC/BC reasons (that is, a graph will not store args that are equal
                # to their default values), but that doesn't matter here. If the user
                # adds a new default arg, then they must update
                # their setup_context (along with the rest of their operator
                # registrations)
                args, kwargs = utils.fill_defaults(op._schema, args, kwargs)

                if has_kwarg_only_args:
                    info._setup_context_fn(
                        ctx=ctx, inputs=args, keyword_only_inputs=kwargs, output=result
                    )
                else:
                    info._setup_context_fn(ctx=ctx, inputs=args, output=result)
            return result

    def backward(ctx, *grads):
        # [FUNCTION DOC] HELPER: Backward of the generated Function. Temporarily drops the
        # trailing metadata entry from needs_input_grad and calls the user backward. Raises
        # RuntimeError if the number of returned grads is neither the forward input count
        # nor the positional arg count, if extra grads are non-None, or if no backward was
        # registered. Returns the grads plus a trailing None for the metadata input.
        if info._backward_fn:
            try:
                prev_needs_input_grad = ctx.needs_input_grad
                ctx.needs_input_grad = prev_needs_input_grad[:-1]
                result = info._backward_fn(ctx, *grads)
            finally:
                ctx.needs_input_grad = prev_needs_input_grad
            num_actual_inputs = len(prev_needs_input_grad) - 1
            valid_return_counts = {num_actual_inputs, num_positional_args}
            actual = len(result) if isinstance(result, tuple) else 1
            if actual not in valid_return_counts:
                expected = (
                    str(num_actual_inputs)
                    if num_actual_inputs == num_positional_args
                    else f"{num_actual_inputs} or {num_positional_args}"
                )
                raise RuntimeError(
                    f"The backward formula for {op} returned an incorrect "
                    f"number of gradients (expected {expected}, got {actual}). "
                    f"Expected one gradient for each forward input, or for "
                    f"each positional input to the operator. Use None for "
                    f"inputs that do not require a gradient."
                )
            if isinstance(result, tuple):
                extra_grads = result[num_actual_inputs:]
                if any(grad is not None for grad in extra_grads):
                    raise RuntimeError(
                        f"The backward formula for {op} returned a non-None "
                        f"gradient for an input that was not passed to the "
                        f"operator. Defaulted inputs that were not passed "
                        f"through autograd must return None."
                    )
                if has_tensorlist_like_args:
                    result = result[:num_actual_inputs]
                return (*result, None)
            return result, None
        raise RuntimeError(
            f"Trying to backward through {op} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    Generated = type(
        name,
        (autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    if has_tensorlist_like_args:
        Generated = supports_tensorlist(Generated)

    # The dispatcher passes any keyword-only-args as kwargs and the
    # rest of the args (even if specified as kwargs) as args.
    def autograd_impl(keyset, *args, **keyword_only_args):
        # [FUNCTION DOC] HELPER: The returned Autograd kernel. For out ops, raises
        # RuntimeError if grad is enabled and any input requires grad, else redispatches.
        # Otherwise applies the generated Function when grad is enabled and a positional
        # arg requires grad, else redispatches without grad. Returns the op result.
        if is_out_op:
            if _C.is_grad_enabled() and _C._any_requires_grad(
                *args, **keyword_only_args
            ):
                raise RuntimeError(
                    f"{op._opname}(): functions with out=... arguments don't "
                    "support automatic differentiation, but one of the arguments "
                    "requires grad."
                )
            return redispatch_no_grad(keyset, args, keyword_only_args)

        if _C.is_grad_enabled() and _C._any_requires_grad(*args):
            result = Generated.apply(*args, Metadata(keyset, keyword_only_args))  # type: ignore[attr-defined]
        else:
            result = redispatch_no_grad(keyset, args, keyword_only_args)
        return result

    return autograd_impl


def supports_tensorlist(cls: Any) -> Any:
    # [FUNCTION DOC] Class decorator: patches forward, backward and apply of an
    # autograd.Function class (in place, and returns it) so list-of-tensor inputs and
    # outputs work. apply flattens inputs with pytree and passes a spec metadata object as
    # the last arg; forward unflattens, runs the original and flattens outputs; backward
    # unflattens grads and checks the structure of returned grads matches the inputs.
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    @dataclass
    class TensorListMetadata:
        # [CLASS DOC] HELPER: Carries the input pytree spec (set at apply time) and the
        # output spec and tuple-ness of the result (filled in during forward).
        input_spec: _pytree.TreeSpec
        output_spec: _pytree.TreeSpec | None = None
        result_is_tuple: bool | None = None

    def new_forward(ctx, *args):
        # [FUNCTION DOC] HELPER: Replacement forward. Raises NotImplementedError if the last
        # arg is not TensorListMetadata, and RuntimeError if ctx already has _pt_metadata.
        # Unflattens args, calls the original forward, records output spec / tuple-ness in
        # the metadata, stores metadata on ctx, and returns the flattened output tuple.
        metadata = args[-1]
        args = args[:-1]
        if not isinstance(metadata, TensorListMetadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        args = _pytree.tree_unflatten(list(args), metadata.input_spec)
        result = orig_forward(ctx, *args)
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        flat_result, output_spec = _pytree.tree_flatten(result, not_list_of_tensor)
        metadata.output_spec = output_spec

        if hasattr(ctx, "_pt_metadata"):
            raise RuntimeError(
                "Please don't set ctx._pt_metadata; PyTorch uses it to store info"
            )
        ctx._pt_metadata = metadata

        return tuple(flat_result)

    def new_backward(ctx, *grads):
        # [FUNCTION DOC] HELPER: Replacement backward. Raises NotImplementedError if ctx has
        # no stored metadata. Unflattens grads, temporarily restructures needs_input_grad,
        # calls the original backward, flattens its result (None entries treated as
        # tensors), raises RuntimeError if the structure differs from the inputs, and
        # returns flat grads plus a trailing None for the metadata input.
        if not hasattr(ctx, "_pt_metadata"):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.backward directly. "
                "This will automatically get called by PyTorch autograd. "
                "Please file an issue if you need this."
            )

        metadata = ctx._pt_metadata
        grads = _pytree.tree_unflatten(list(grads), metadata.output_spec)

        # If the user's input is ([x, y, z], w),
        # then needs_input_grad is (bool, bool, bool, bool, bool).
        # We need to
        # 1. get rid of the additional bool (which comes from the extra
        # `metadata input`)
        # 2. _pytree.tree_unflatten to get the right structure.
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            ctx.needs_input_grad = _pytree.tree_unflatten(
                list(ctx.needs_input_grad[:-1]), metadata.input_spec
            )
            grad_inputs = orig_backward(ctx, *grads)
        finally:
            ctx.needs_input_grad = prev_needs_input_grad

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        # Assume that any Nones in the backward are Tensors.
        # If the forward has an arg that is [1, 2, 3], the backward should
        # return None as the grad.
        # If the forward has an arg that is [tensor, tensor], the backward
        # may return [None, None], [grad, None], [None, grad], or [grad, grad].
        flat_grad_inputs, grad_inputs_spec = _pytree.tree_flatten(
            grad_inputs, not_list_of_optional_tensor
        )
        if grad_inputs_spec != metadata.input_spec:
            raise RuntimeError(
                f"Expected the return from backward to be of the same structure "
                f"as the inputs. Got: {grad_inputs_spec} (return from backward), "
                f"{metadata.input_spec} (inputs)"
            )
        return tuple(flat_grad_inputs + [None])

    def new_apply(*args):
        # [FUNCTION DOC] HELPER: Replacement apply. Flattens args (lists of tensors stay
        # leaves of the spec), calls the original apply with metadata, unflattens the output
        # and unwraps a single-element tuple if the original result was not a tuple.
        flat_args, input_spec = _pytree.tree_flatten(args, is_leaf=not_list_of_tensor)
        metadata = TensorListMetadata(input_spec)
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        if metadata.output_spec is None:
            raise AssertionError("metadata.output_spec must not be None")
        result = _pytree.tree_unflatten(list(result), metadata.output_spec)
        if not metadata.result_is_tuple:
            if not isinstance(result, tuple):
                raise AssertionError(f"result must be tuple, got {type(result)}")
            if len(result) != 1:
                raise AssertionError(
                    f"result tuple must have length 1, got {len(result)}"
                )
            return result[0]
        return result

    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    return cls


def not_list_of_tensor(tree):
    # [FUNCTION DOC] Pytree is_leaf predicate: False for tuples, False for a list whose
    # elements are all Tensors (treated as a leaf is NOT wanted here, so it is traversed
    # as a unit by the caller's convention), True for other values and mixed lists.
    # KNOWN ISSUE: name reads as "is not a list of tensors" but it is used as is_leaf, so
    # a pure tensor list returns False, meaning it is not a leaf and is split into tensors.
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    return True


def not_list_of_optional_tensor(tree):
    # [FUNCTION DOC] Same as not_list_of_tensor but list elements may also be None: False
    # for tuples and lists containing only Tensors/None, True otherwise. Used as the
    # pytree leaf predicate when flattening gradients.
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(l is not None and not isinstance(l, Tensor) for l in tree)
    return True
