# [MODULE DOC]
# Defines the print higher order operator, a functional graph-friendly print
# that formats a string with positional/keyword args at execution time. Provides
# proxy tracing, fake, eager, DTensor (per-rank local print) and functionalization
# (effect-token) implementations.
# Import-time side effects: creates the module-level singleton named print
# (shadowing the builtin within this module) and registers autograd fallthroughs
# for CPU and CUDA.
import builtins

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


class Print(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator "print". Stateless; calling it prints the
    # format string formatted with the given args and kwargs and returns None.
    """
    print(format_str, *args, **kwargs) -> None

    This Higher Order Operator (HOP) provides a functional version of print for use in PyTorch graphs.
    It supports the calling conventions of print(format_str.format(*args, **kwargs)):

    1. Format string with keyword arguments (named placeholders):
       torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
       Output: "moo 1 2"

    2. Format string with positional arguments (positional placeholders):
       torch._higher_order_ops.print("moo {} {}", 1, 2)
       Output: "moo 1 2"

    3. Mixed positional and keyword arguments:
       torch._higher_order_ops.print("moo {} {y}", 1, y=2)
       Output: "moo 1 2"

    4. DTensor support:
       DTensor args are unwrapped to local tensors via to_local() (no collective).
       Each rank prints its own local view, prefixed with [rank N].
       For the global view of a sharded tensor, call full_tensor() before
       passing to print.

       dt = DTensor.from_local(local_shard, mesh, [Shard(0)])
       torch._higher_order_ops.print("activations: {}", dt)
       # Output: [rank 0] activations: tensor([0., 1.])

    This HOP enables printing without causing graph break.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the operator under the name "print".
        super().__init__("print")

    def __call__(self, format_str: str, *args: object, **kwargs: object) -> None:
        # [FUNCTION DOC] Dispatches a print. Args: format_str (must be a str),
        # positional and keyword format arguments. Raises AssertionError if
        # format_str is not a string. Returns None (result of dispatch).
        if not isinstance(format_str, str):
            raise AssertionError(f"format_str must be a string, got {type(format_str)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(format_str, *args, **kwargs)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, format_str: str, *args: object, **kwargs: object
    ) -> torch.FunctionSchema:
        # [FUNCTION DOC] Builds the HOP function schema: an argument for the
        # format string, one positional argument per extra arg (arg0, arg1, ...),
        # one keyword-only argument per kwarg, plus the pytree spec of the call.
        # Returns the generated FunctionSchema.
        # KNOWN ISSUE: adds the first element of format_str (its first character
        # when it is a string) as the example value for the format_str argument
        # rather than the string itself.
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("format_str", format_str[0])

        # Add each positional arg
        for i, value in enumerate(args):
            schema_gen.add_arg(f"arg{i}", value)

        # Add each kwarg as a keyword-only argument
        for key, value in kwargs.items():
            schema_gen.add_arg(key, value, kw_only=True)

        schema_gen.add_schema_tree_spec(format_str, *args, **kwargs)

        return schema_gen.gen_schema()


print = Print()


@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode, format_str: str, *args: object, **kwargs: object
) -> None:
    # [FUNCTION DOC] Proxy-tracing implementation: unwraps proxies for the args and
    # kwargs and records a call_function node for print in the tracer graph. The
    # format string is passed through as a constant. Returns None. Side effect:
    # adds a node to the graph.
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]
    mode.tracer.create_proxy(
        "call_function", print, (format_str, *proxy_args), proxy_kwargs
    )


@register_fake(print, skip_cache=True)
# pyre-ignore
def print_fake_tensor_mode(format_str: str, *args: object, **kwargs: object):
    # [FUNCTION DOC] HELPER: Fake tensor implementation (not cached); does nothing
    # and returns None.
    return None


@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_impl(format_str: str, *args: object, **kwargs: object) -> None:
    # [FUNCTION DOC] Eager implementation: converts FX immutable dict/list values
    # in args and kwargs into ordinary dict/list, formats the string with them and
    # writes it via the builtin print (to avoid recursing into the HOP). Side
    # effect: writes a line to standard output.
    # Ensure all immutable_dict/list in args and kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args, new_kwargs = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        (args, kwargs),
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    #  Use built-in print to avoid recursion with the HOP print
    builtins.print(format_str.format(*new_args, **new_kwargs))


print.fallthrough(torch._C.DispatchKey.AutogradCPU)
print.fallthrough(torch._C.DispatchKey.AutogradCUDA)


def _register_dtensor_impl() -> None:
    # [FUNCTION DOC] HELPER: Lazily registers the DTensor dispatch implementation of
    # print (imports DTensor to avoid an import-time dependency). Side effect:
    # mutates the print operator's implementation table. Not called in this module.
    from torch.distributed.tensor import DTensor

    @print.py_impl(DTensor)  # pyrefly: ignore [missing-attribute]
    # pyre-ignore
    def print_dtensor(format_str: str, *args: object, **kwargs: object) -> None:
        # [FUNCTION DOC] Prints with DTensor args replaced by their local shards
        # (no collective). When a process group is initialized with world size
        # above 1, the format string is prefixed with the rank. Side effect:
        # prints a line from each rank.
        # Unwrap DTensors to local tensors via to_local() — no collective is
        # introduced so there is no OOM or performance risk.  Every rank prints
        # its own local view (including Replicate, where to_local() already
        # holds the full tensor).
        #
        # The output is prefixed with [rank N] so users can identify which rank
        # produced each line.
        #
        # If the user needs the global view of a sharded tensor, they can call
        # full_tensor() explicitly before passing it to print.
        import torch.distributed as dist

        local_args = pytree.tree_map_only(DTensor, DTensor.to_local, args)
        local_kwargs = pytree.tree_map_only(DTensor, DTensor.to_local, kwargs)
        if dist.is_initialized() and dist.get_world_size() > 1:
            format_str = f"[rank {dist.get_rank()}] {format_str}"
        print(  # pyrefly: ignore [no-matching-overload]
            format_str, *local_args, **local_kwargs
        )


@print.py_functionalize_impl
def print_func(ctx, format_str: str, *args: object, **kwargs: object):
    # [FUNCTION DOC] Functionalization implementation: routes print through the
    # effects machinery (handle_effects) using the context's token state so the
    # print is ordered as a side effect. Returns the handler's result.
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        print,  # type: ignore[arg-type]
        (format_str, *args),
        kwargs,  # type: ignore[arg-type]
    )
