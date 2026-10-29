# [MODULE DOC] Operator support predicates used to decide whether a backend can
# handle an FX node (for example when partitioning a graph). Contains the
# abstract OperatorSupportBase, the dtype-table based OperatorSupport, combinators
# (create_op_support, chain, any_chain), and OpSupports with common atomic
# predicates. No import-time side effects.
import abc
import typing as t

import torch
import torch.fx
from torch.fx._compatibility import compatibility

from .shape_prop import TensorMetadata
from .tools_common import CALLABLE_NODE_OPS, get_node_target


__all__ = [
    "OperatorSupportBase",
    "OperatorSupport",
    "create_op_support",
    "chain",
    "OpSupports",
    "any_chain",
]

# fx.Node.target typename, as returned by `get_node_target()`
TargetTypeName = str

# Arguments' dtypes for a given node, see `OperatorSupport`
SupportedArgumentDTypes = (
    tuple[
        t.Sequence[t.Sequence[torch.dtype]],
        dict[str, t.Sequence[torch.dtype]],
    ]
    | None
)

SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]


@compatibility(is_backward_compatible=False)
class OperatorSupportBase(abc.ABC):
    """Interface for determining if a fx.Node is supported by a backend"""
    # [CLASS DOC] Abstract interface with a single method telling whether a node
    # is supported by some backend. Stateless; subclasses implement the method.

    @abc.abstractmethod
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # [FUNCTION DOC] Abstract: given the submodule mapping and a node, return
        # whether the node is supported. Base version raises NotImplementedError.
        raise NotImplementedError


@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """
    # [CLASS DOC] Table-driven support check. State: _support_dict mapping the
    # node target type name (from get_node_target) to either None (any dtype
    # accepted) or a pair of positional dtype lists and keyword dtype lists.
    # A target missing from the table is unsupported.

    _support_dict: SupportDict

    def __init__(self, support_dict: SupportDict | None = None) -> None:
        # [FUNCTION DOC] Store the support table (empty mapping if not given).
        self._support_dict = support_dict or {}

    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        # [FUNCTION DOC] Decide support for a node. Nodes whose op is not a callable
        # kind (placeholder, output, get_attr) are always supported. Otherwise the
        # target name is looked up in the table: missing means unsupported, None
        # means supported. Then each positional tensor-node argument's dtype must
        # be in the allowed list for its position (None entries and non-node args
        # are skipped, extra table entries beyond actual args are ignored), and
        # each listed keyword argument present as a node must have an allowed dtype.
        # Returns True only if all checks pass. Reads node metadata; no mutation.
        if node.op not in CALLABLE_NODE_OPS:
            return True

        target = get_node_target(submodules, node)

        # Target not found in _support_dict meaning that we don't support this op at all
        if target not in self._support_dict:
            return False

        # The rule for target is None meaning that we accept any dtype
        if self._support_dict[target] is None:
            return True

        args_dtypes, kwargs_dtypes = self._support_dict[target]  # type: ignore[misc]

        # Check args dtypes
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None indicates we don't care about the dtype of args[i]
            if dtypes is None:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            arg_dtype = _get_arg_dtype(node.args[i])  # type: ignore[arg-type]
            if arg_dtype not in dtypes:
                return False

        # Check kwargs dtypes
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            kwarg_dtype = _get_arg_dtype(node.kwargs[k])  # type: ignore[arg-type]
            if kwarg_dtype not in dtypes:
                return False

        return True


# ======================================================================
# Functional interfaces and utils for defining basic operator support logic
# and composing them into more complex ones
# ======================================================================

IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]


@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    """Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance

    `IsNodeSupported` has the same call signature as
    `OperatorSupportBase.is_node_supported`
    """
    # [FUNCTION DOC] Adapt a plain predicate function with the support signature
    # into an OperatorSupportBase object that delegates to it. Returns the object.

    class FunctionalOperatorSupport(OperatorSupportBase):
        # [CLASS DOC] HELPER: Support object delegating to the captured predicate.
        def is_node_supported(
            self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
        ) -> bool:
            # [FUNCTION DOC] HELPER: Call the captured predicate and return its result.
            return is_node_supported(submodules, node)

    return FunctionalOperatorSupport()


@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns False if
    any of it reports False.
    """
    # [FUNCTION DOC] Build a support object that is true only if every given
    # support object accepts the node (logical AND, short-circuits; true when
    # given none).

    def _chain(submods: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: True if all wrapped supports accept the node.
        return all(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_chain)


@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns True if
    any of it reports True.
    """
    # [FUNCTION DOC] Build a support object that is true if at least one given
    # support object accepts the node (logical OR, short-circuits; false when
    # given none).

    def _any_chain(
        submods: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if any wrapped support accepts the node.
        return any(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_any_chain)


@compatibility(is_backward_compatible=False)
class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """
    # [CLASS DOC] Namespace of class-method factories for atomic support objects
    # (decline_if_input_dtype, decline_if_node_in_names). Holds no state.

    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        """Report a node as non-supported, if any of its arguments is of dtype"""
        # [FUNCTION DOC] Build a support object rejecting any node that has an
        # input node of the given dtype. Returns the support object.

        def _decline_if_input_dtype(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            # [FUNCTION DOC] HELPER: False if any input node has the dtype, else
            # True. Raises if an input node lacks dtype metadata.
            for arg in node.all_input_nodes:
                arg_dtype = _get_arg_dtype(arg)
                if arg_dtype == dtype:
                    return False
            return True

        return create_op_support(_decline_if_input_dtype)

    @classmethod
    def decline_if_node_in_names(cls, disallow_set: set[str]) -> OperatorSupportBase:
        """
        If a node has a name that is in the disallow set, reported it as non-supported.
        """
        # [FUNCTION DOC] Build a support object rejecting nodes whose name is in the
        # given set of names. Returns the support object.

        def _decline_if_node_in_names(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            # [FUNCTION DOC] HELPER: True if the node name is not in the disallow set.
            return node.name not in disallow_set

        return create_op_support(_decline_if_node_in_names)


def _get_arg_dtype(arg: torch.fx.Node) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Return the dtype of a node: from the tensor_meta entry
    # of its metadata when it is a TensorMetadata, else the type entry of the
    # metadata. Raises AssertionError if arg is not a Node; raises KeyError if
    # neither metadata entry exists.
    if not isinstance(arg, torch.fx.Node):
        raise AssertionError(f"Expected torch.fx.Node, got {type(arg)}")
    tensor_meta = arg.meta.get("tensor_meta")  # type: ignore[union-attr]
    dtype = (
        tensor_meta.dtype
        if isinstance(tensor_meta, TensorMetadata)
        else arg.meta["type"]
    )
    return dtype
