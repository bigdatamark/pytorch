# [MODULE DOC] FX transformers that canonicalize call sites. NormalizeArgs rewrites
# arguments of function and module calls to match the target signature (keyword form
# with defaults filled in); NormalizeOperators remaps torch arithmetic and comparison
# functions to the equivalent python operator functions. No import-time side effects.
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.fx
import torch.fx as fx
from torch.fx import Proxy, Transformer
from torch.fx.node import Argument, map_aggregate, Node, Target
from torch.fx.operator_schemas import (
    create_type_hint,
    normalize_function,
    normalize_module,
)

from .schema_type_annotation import AnnotateTypesWithSchema


class NormalizeArgs(Transformer):
    # [CLASS DOC] Transformer that rewrites each call_function and call_module node so
    # its args and kwargs follow the target's signature (all keyword when
    # normalize_to_only_use_kwargs is set). Holds node_map from new proxies to original
    # nodes and the normalize flag. Copies meta and type from the original node to the
    # new one. Falls back to the unmodified call when normalization is not possible.
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """

    def __init__(
        self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = True
    ) -> None:
        # [FUNCTION DOC] Initializes the transformer over the given graph module, with an
        # empty proxy-to-node map, and stores the flag controlling kwargs-only output.
        super().__init__(module)
        self.node_map: dict[Proxy, Node] = {}
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs

    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Executes one node during transformation. Resolves its args from
        # the environment, computes type hints for positional and keyword arguments
        # (node arguments use the node's own type metadata, others their runtime type),
        # and for call_function nodes goes through this class's call_function with the
        # hints; other ops use the base behavior. For non-output nodes it records the
        # result in node_map and copies the original meta and type onto the new node.
        # Returns the resulting proxy. Raises an error if the positional type structure
        # is not a tuple.
        # KNOWN ISSUE: get_type returns the current node's type metadata for every
        # Node-valued argument rather than that argument's own type.
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        def get_type(arg: object) -> Any:
            # [FUNCTION DOC] HELPER: Returns the type metadata of the enclosing node
            # when arg is a Node, otherwise the runtime type of arg.
            if isinstance(arg, fx.Node):
                return n.meta.get("type")
            return type(arg)

        arg_types = map_aggregate(n.args, get_type)
        if not isinstance(arg_types, tuple):
            raise AssertionError(f"Expected tuple, got {type(arg_types)}")
        arg_types = tuple(create_type_hint(i) for i in arg_types)
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}
        if n.op == "call_function":
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)
        else:
            out = super().run_node(n)
        if n.op != "output":
            self.node_map[out] = n
            out.node.meta = n.meta
            out.node.type = n.type
        return out

    def call_function(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
        arg_types: tuple[Any, ...] | None = None,
        kwarg_types: dict[str, Any] | None = None,
    ) -> Proxy:
        # [FUNCTION DOC] Normalizes a function call using the optional type hints to pick
        # an overload. If normalization succeeds, emits a new call_function proxy with
        # the normalized args and kwargs; otherwise delegates to the base transformer
        # unchanged. Raises an error if target is not callable.
        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")
        new_args_and_kwargs = normalize_function(
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            arg_types,  # type: ignore[arg-type]
            kwarg_types,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return self.tracer.create_proxy(
                "call_function", target, new_args, new_kwargs
            )
        else:
            return super().call_function(target, args, kwargs)

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] Normalizes a module call using the submodule's forward
        # signature; on success delegates to the base call_module with the normalized
        # args and kwargs, otherwise with the originals. Raises an error if target is
        # not a string.
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        new_args_and_kwargs = normalize_module(
            self.module,
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return super().call_module(target, new_args, new_kwargs)
        else:
            return super().call_module(target, args, kwargs)


class NormalizeOperators(AnnotateTypesWithSchema):
    # [CLASS DOC] Schema-annotating transformer that remaps torch binary functions
    # (add, mul, sub, div, floor_divide, remainder and the six comparisons) to the
    # matching python operator functions via the class-level mapping
    # binary_magic_method_remap. Other calls pass through.
    # KNOWN ISSUE: the class docstring is truncated mid-sentence, and the mapping is
    # keyed by torch functions while the docstring says operators map to torch ops.
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """

    binary_magic_method_remap: dict[
        Callable[[Any, Any], Any], Callable[[Any, Any], Any]
    ] = {
        torch.add: operator.add,
        torch.mul: operator.mul,
        torch.sub: operator.sub,
        torch.div: operator.truediv,
        torch.floor_divide: operator.floordiv,
        torch.remainder: operator.mod,
        torch.eq: operator.eq,
        torch.ne: operator.ne,
        torch.lt: operator.lt,
        torch.le: operator.le,
        torch.gt: operator.gt,
        torch.ge: operator.ge,
    }

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] If target is a key of the remap table and exactly two positional
        # args are given, emits the mapped operator function with those two args and no
        # kwargs; otherwise delegates unchanged. Raises an error if target is not
        # callable.
        # Normalize operators according to the magic methods implemented on tensors here:
        # https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137

        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")

        if target in self.binary_magic_method_remap:
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args

            return super().call_function(
                target=self.binary_magic_method_remap[target],
                args=(lhs, rhs),
                kwargs={},
            )

        return super().call_function(target, args, kwargs)
