# mypy: allow-untyped-defs
# [MODULE DOC] Implements effect-token threading for side-effectful ops. Defines the
# with_effects higher-order operator (token, op, args -> new token plus op results), helpers
# to look up/register an op's effect type in the library registry, and handle_effects which
# wraps an effectful op call with the proper token during functionalization/tracing.
# Import-time side effects: registers ORDERED effects for aten::_print,
# aten::_linalg_check_errors, torchbind calls, hop print and invoke_leaf_function, clears the
# effect for the profiler record-function-exit op, creates the global with_effects operator,
# and registers its dense, fake, proxy, functionalize implementations and autograd fallthroughs.
from typing import Any, cast, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch._higher_order_ops.print import print as hop_print
from torch._higher_order_ops.schema import HopSchema
from torch._higher_order_ops.torchbind import call_torchbind
from torch._higher_order_ops.utils import register_fake
from torch._library.custom_ops import CustomOpDef
from torch._library.effects import EffectType
from torch._library.utils import RegistrationHandle
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


_op_identifier = Union[
    str,
    "torch._ops.OpOverload",
    "torch._library.custom_ops.CustomOpDef",
    "torch._ops.HigherOrderOperator",
]
OpType = Union["torch._ops.HigherOrderOperator", "torch._ops.OpOverload"]

_EffectType = EffectType


def _get_op_qualname(op: _op_identifier) -> str:
    # [FUNCTION DOC] HELPER: Maps an op identifier to its qualified name string: the name of
    # an OpOverload, namespace::name for a HigherOrderOperator, the qualname of a CustomOpDef,
    # or the string itself. Raises ValueError for any other input.
    """Convert an op identifier to a qualified string key."""
    if isinstance(op, torch._ops.OpOverload):
        return op._name
    elif isinstance(op, torch._ops.HigherOrderOperator):
        return f"{op.namespace}::{op.name()}"
    elif isinstance(op, CustomOpDef):
        return op._qualname
    elif isinstance(op, str):
        return op

    raise ValueError(f"Invalid operator input {op}")


def _register_effectful_op(
    op: _op_identifier, effect: EffectType | None
) -> RegistrationHandle:
    # [FUNCTION DOC] HELPER: Looks up the op's entry in the global simple registry and registers
    # the given effect type (or None to clear) on it. Mutates global registry state; returns
    # the registration handle that can be used to undo it.
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    handle = entry.effect.register(effect)
    return handle


def _get_effect(op: _op_identifier) -> _EffectType | None:
    # [FUNCTION DOC] HELPER: Returns the effect type registered for the op in the global simple
    # registry, or None if it has none.
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    return entry.effect.effect


_register_effectful_op("aten::_print", _EffectType.ORDERED)
# _linalg_check_errors has no tensor output, but it raises on invalid linalg
# results and must be preserved by functionalization/DCE.
_register_effectful_op("aten::_linalg_check_errors", _EffectType.ORDERED)
_register_effectful_op("profiler::_record_function_exit._RecordFunction", None)
_register_effectful_op(call_torchbind, _EffectType.ORDERED)
_register_effectful_op(hop_print, _EffectType.ORDERED)
_register_effectful_op(invoke_leaf_function, _EffectType.ORDERED)


class WithEffects(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator named with_effects. Takes an effect token, an op and
    # that op's arguments, runs the op, and returns a new token followed by the op results so
    # that data dependence through tokens keeps side-effectful ops ordered. Stateless; one
    # global instance with_effects exists.
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name with_effects.
        super().__init__("with_effects")

    def __call__(
        self,
        token,
        op: OpType,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        # [FUNCTION DOC] Validates the call then dispatches. Raises AssertionError if op is
        # not a HigherOrderOperator or OpOverload, if the op has aliasing in its schema, or
        # if kwargs is not a dict. Otherwise dispatches through the base operator and returns
        # a tuple of the new token followed by the op's outputs.
        if not isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload)):
            raise AssertionError(
                f"op must be HigherOrderOperator or OpOverload, got {type(op)}"
            )
        if has_aliasing(op):
            raise AssertionError("Ops with aliasing is not supported")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(token, op, *args, **kwargs)


with_effects = WithEffects()


def has_aliasing(op: OpType):
    # [FUNCTION DOC] Returns True if any argument or return in the op's schema carries alias
    # info (always False for a HigherOrderOperator, which has no schema checked here).
    # NOT FOR PUBLIC USE
    if isinstance(op, torch._ops.HigherOrderOperator):
        return False

    for arg in op._schema.arguments:
        if arg.alias_info is not None:
            return True
    for arg in op._schema.returns:
        if arg.alias_info is not None:
            return True
    return False


def has_effects(op) -> bool:
    # [FUNCTION DOC] True if op is a HigherOrderOperator or OpOverload, has no aliasing, and
    # has a registered effect type.
    return (
        isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        and not has_aliasing(op)
        and _get_effect(op) is not None
    )


def new_token_tensor() -> torch.Tensor:
    # [FUNCTION DOC] Creates a fresh empty tensor used as an effect token.
    return torch.tensor([])


@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Eager implementation: runs the op, creates a new token, and returns a
    # tuple of the new token followed by the op's outputs. A tuple output is spread into the
    # result; any other output (including a list) is appended as a single element, matching
    # the op schema. The input token is ignored.
    out = op(*args, **kwargs)
    new_token = new_token_tensor()
    # [NOTE: with_effects return type]
    # Note that we should only do *out for tuple type, but not list type.
    # This is to match the schema of the op.
    # For tuple output, the length of schema output is the same as the length of out.
    # For list output, the length of schema output is 1 (e.g. Tensor[]) regardless of the
    # length of the list.
    if isinstance(out, tuple):
        return (new_token, *out)
    return (new_token, out)


@register_fake(with_effects, skip_cache=True)
def with_effects_fake(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Fake tensor implementation (caching skipped): delegates to the dense
    # implementation on fake inputs and returns its (token, outputs) tuple.
    result = with_effects_dense(token, op, *args, **kwargs)
    return result


@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Proxy-tracing implementation: computes real outputs with proxy tracing
    # disabled, marks the op as having a side effect (so dead-code elimination keeps it, a
    # global graph-node registry mutation), emits a call_function node for with_effects with
    # the token, op and unwrapped proxy args/kwargs, and returns the outputs tracked to it.
    with disable_proxy_modes_tracing():
        out = with_effects(token, op, *args, **kwargs)

    proxy_token = mode.tracer.unwrap_proxy(token)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    from torch.fx.node import has_side_effect

    # To avoid the being DCEed by graph.eliminate_dead_code if they.
    # don't have output or their outputs are not used.
    has_side_effect(op)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        with_effects,
        (proxy_token, op, *proxy_args),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


with_effects.fallthrough(DispatchKey.AutogradCPU)
with_effects.fallthrough(DispatchKey.AutogradCUDA)


@with_effects.py_functionalize_impl
def with_effects_functional(
    ctx,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Functionalization implementation: unwraps tensors in token/args/kwargs,
    # redispatches with_effects to the next key and wraps the result back. No mutation
    # handling is needed because the operator is already functional.
    # with_effects is already functional, so just re-emit it.
    unwrapped_token, unwrapped_args, unwrapped_kwargs = ctx.unwrap_tensors(
        [token, args, kwargs]
    )
    with ctx.redispatch_to_next():
        result = with_effects(unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)


_EFFECTFUL_HOPS_WITH_SCHEMA = {hop_print, invoke_leaf_function}


def _get_schema(op, args, kwargs: dict | None = None) -> torch.FunctionSchema:
    # [FUNCTION DOC] HELPER: Returns the schema for an op call: the overload's schema, the
    # schema of the bound torchbind method (found from the first two args), or a generated
    # schema for the effectful HOPs with schemas. Raises RuntimeError for any other op.
    if isinstance(op, torch._ops.OpOverload):
        return op._schema
    elif op == call_torchbind:
        return getattr(args[0], args[1]).schema
    elif op in _EFFECTFUL_HOPS_WITH_SCHEMA:
        extra_kwargs = kwargs or {}
        return op.gen_schema(*args, **extra_kwargs)
    else:
        raise RuntimeError(f"Unable to get schema for op {op}")


def handle_effects(
    allow_token_discovery: bool,
    tokens: dict[_EffectType, torch.Tensor],
    op: OpType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    # [FUNCTION DOC] Calls an effectful op through with_effects while threading its effect
    # token. Finds the effect key of op (AssertionError if None). If tokens has no token for
    # the key: raises AssertionError when discovery is not allowed; otherwise creates one,
    # and under an active proxy mode adds a tangents_token placeholder input to the graph
    # being traced. Runs with_effects redispatched below functionalization, normalizes the
    # outputs according to the schema returns (none -> None, one -> unwrapped single value,
    # many -> tuple; AssertionError on count mismatch), stores the new token back into the
    # tokens mapping (mutation), and returns the wrapped outputs.
    """
    Args:
        allow_token_discovery: Whether or not we are discovering tokens. If this
        is true, we will create a token for every side effect type seen that
        does not have a token assigned yet.  If this is false, the tokens
        should've all been created ahead of time, so we will error if there is
        no token mapping to every effect type.

        tokens: Map of effect type to tokens. This is to chain operators of the
        same effects together so that they do not get reordered in later
        optimization passes.
    """

    # Get a token. We can't do `tokens.get(op, torch.tensor([]))` because
    # this will create an empty tensor during proxy mode tracing if the token
    # doesn't exist. But the tokens should always exist during proxy mode tracing.
    key = _get_effect(op)
    if key is None:
        raise AssertionError(f"effect key must not be None for op {op}")
    if key not in tokens:
        if not allow_token_discovery:
            raise AssertionError(
                f"Could not find a token for effect {key} which came from the function {op}"
            )
        proxy_tensor_mode = cast(
            "ProxyTorchDispatchMode | None",
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY),
        )
        if proxy_tensor_mode is not None:
            # If we discovered a new token during tracing, we are in backward.
            # Then we patch the graph, adding additional tangents_token as input to the joint graph.
            tracer = proxy_tensor_mode.tracer

            from torch.fx.experimental.proxy_tensor import (
                disable_proxy_modes_tracing,
                track_tensor_tree,
            )

            with disable_proxy_modes_tracing():
                token_tensor = new_token_tensor()

            token_proxy = proxy_tensor_mode.tracer.create_proxy(
                "placeholder", "tangents_token", (), {}, name="tangents_token"
            )
            track_tensor_tree(token_tensor, token_proxy, constant=None, tracer=tracer)

            tokens[key] = token_tensor
        else:
            tokens[key] = new_token_tensor()

    token = tokens[key]

    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    unwrapped_token = ctx.unwrap_tensors([token])[0]
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        (new_token, *unwrapped_outs) = with_effects(
            unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs
        )

    schema = _get_schema(op, unwrapped_args, unwrapped_kwargs)

    if isinstance(schema, HopSchema):
        if len(schema.returns) == 0:
            unwrapped_outs = ()
        else:
            if len(unwrapped_outs) != len(schema.returns):
                raise AssertionError(
                    f"expected {len(schema.returns)} outputs but got {len(unwrapped_outs)}"
                )
            unwrapped_outs = tuple(unwrapped_outs)
    elif len(schema.returns) == 0:
        if unwrapped_outs[0] is not None:
            raise AssertionError(f"expected no outputs but got {unwrapped_outs[0]}")
        unwrapped_outs = None  # type: ignore[assignment]
    elif len(schema.returns) == 1:
        if len(unwrapped_outs) != 1:
            raise AssertionError(f"expected 1 output but got {len(unwrapped_outs)}")
        unwrapped_outs = unwrapped_outs[0]
    else:
        if len(unwrapped_outs) != len(schema.returns):
            raise AssertionError(
                f"expected {len(schema.returns)} outputs but got {len(unwrapped_outs)}"
            )

    # Add the newly created token into the tokens map for a following call to
    # use this token.
    wrapped_token = ctx.wrap_tensors(new_token)
    if not isinstance(wrapped_token, torch.Tensor):
        raise AssertionError(
            f"expected wrapped_token to be torch.Tensor, got {type(wrapped_token)}"
        )
    tokens[key] = wrapped_token

    # pyrefly: ignore [bad-argument-type]
    return ctx.wrap_tensors(unwrapped_outs)
