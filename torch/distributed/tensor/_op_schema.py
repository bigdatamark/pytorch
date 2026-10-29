# mypy: allow-untyped-defs
# [MODULE DOC]
# Core data structures describing DTensor operator schemas and sharding
# strategies: OpSpec, StrategyType/OpStrategy/TupleStrategy, RuntimeSchemaInfo,
# OpSchema, OutputSharding and OpInfo, plus type aliases (ArgsType, KwargsType,
# PlacementList, OutputSpecType) and small helpers. They are consumed by sharding
# propagation, strategy selection, redistribution planning and dispatch caching.
# Import-time side effects: registers TupleStrategy as a pytree node (ignoring
# the error if already registered); prefers the C++ pytree and falls back to
# the Python pytree.
"""
DTensor operator schema definitions and utilities.

This module defines the core data structures and utilities for describing and managing
distributed tensor operations in PyTorch's DTensor system. It provides the foundational
schema types used for sharding propagation, operator strategy selection, and distributed
execution planning.

Key components:
- OpSpec: Describes acceptable sharding placements for operations
- OpStrategy: Represents the possible sharding strategies for an operator
- TupleStrategy: Container for multiple strategies when ops have tuple/list of tensors input
- OpSchema: Describes operator input/output schemas with DTensorSpecs
- OutputSharding: Manages output sharding specifications and redistribution
- RuntimeSchemaInfo: Runtime execution metadata for operators
- OpInfo: Complete runtime operator execution information

These schema definitions enable the DTensor system to:
1. Propagate tensor sharding information to the operator outputs
2. Greedily select sharding strategies for distributed operations
3. Plan and execute tensor redistributions when needed
4. Cache sharding decisions for performance optimization
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from typing_extensions import deprecated

import torch
from torch._C import (
    _DTensor_OpSchema_post_init,
    _DTensor_OpSchema_recompute_comparison_key,
)
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import Placement


try:
    from torch.utils._cxx_pytree import (
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )
except ImportError:
    from torch.utils._pytree import (  # type: ignore[no-redef, assignment]
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )


# Common type aliases
ArgsType = tuple[object, ...]
KwargsType = dict[str, object]

PlacementList = list[Placement | None]

# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type should
# be the same set of possibilities.
OutputSpecType = DTensorSpec | Sequence[DTensorSpec | None] | None


def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    # [FUNCTION DOC] HELPER: Creates an uninitialized strided tensor from the
    # tensor_meta (shape, stride, dtype) of a DTensorSpec, on the mesh's device
    # type. Raises AssertionError if tensor_meta is missing. Should be called
    # under fake mode so no real memory is used.
    """
    This is used to propagate tensor metadata, must be under fake mode
    """
    if arg.tensor_meta is None:
        raise AssertionError("DTensorSpec does not contain tensor_meta.")
    return torch.empty_strided(
        arg.tensor_meta.shape,
        arg.tensor_meta.stride,
        dtype=arg.tensor_meta.dtype,
        device=arg.mesh.device_type,
    )


def _pretty_print_spec(spec: object) -> str:
    # [FUNCTION DOC] HELPER: Formats a spec for display: None gives the text
    # None, a DTensorSpec gives the concatenated placement strings, and a
    # sequence gives a parenthesized comma-separated list of recursively formatted
    # items. Raises RuntimeError for any other type.
    if spec is None:
        return "None"
    elif isinstance(spec, DTensorSpec):
        return "".join([str(p) for p in spec.placements])
    elif isinstance(spec, Sequence):
        return "(" + ", ".join([_pretty_print_spec(s) for s in spec]) + ")"
    else:
        raise RuntimeError(f"Unknown spec type to print: spec={spec}")


@dataclass
class OpSpec:
    # [CLASS DOC]
    # Dataclass for one acceptable sharding choice of an op: output_specs (a
    # DTensorSpec, a tuple of optional DTensorSpecs, or None for non-tensor
    # results), optional input_specs, and optional redistribute_cost (per tensor
    # input, the cost of converting each producer strategy's output to the
    # required input spec). All specs must share one DeviceMesh. Provides
    # output_spec / mesh (cached), input_spec(index), string formatting, and
    # hash/equality based on output_specs and input_specs only (cost ignored).
    """
    An OpSpec describes an acceptable sharding placements of an operation, with the
    specified DTensorSpecs for both the output and the inputs.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].

    note: we MUST produce an DTensorSpec for every output that is a Tensor.  None
    entries only occur for non-Tensor outputs (e.g., operators that return Optional[Tensor],
    or non-Tensor outputs.)

    invariant: the DeviceMesh on all DTensorSpec must be the same
    """

    # output_specs and input_specs are related: for this op, given these input_specs,
    # this is the way the output would look
    # Note: output_specs can be None for ops that don't return tensors (e.g., _linalg_check_errors)
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...] | None
    input_specs: Sequence[DTensorSpec] | None = None

    """
    redistribute_cost tells how expensive it is to redistribute a given input into the
    placement specified in this OpSpec.

    outer list: one entry (list) per (tensor) input in the op's arg schema
    inner list: one entry (cost value) per possible sharding spec for that input

    Example:
    -------
    another_op() -> tensor_a   # another_op produces the output that becomes our first input
    my_op(tensor_a)

    Let's assume this OpSpec's input_specs are [Replicate()],
    but another_op() supports 2 strategies (OpSpecs) which produce outputs of
       Replicate()
       Shard(0)

    In this example, redistribute_costs would look like this
    [
        # one row representing "my_op's first input" (tensor_a)
        [
            # two entries, one for each strategies supported by another_op
            0.0,  # cost of redistributing tensor_a from 'Replicate()'
            K,    # cost of redistributing tensor_a from 'Shard(0)'
        ],
    """
    redistribute_cost: list[list[float]] | None = None

    @cached_property
    def output_spec(self) -> DTensorSpec:
        # [FUNCTION DOC]
        # Cached property returning output_specs when it is a single DTensorSpec;
        # raises ValueError if it is a tuple or None.
        """
        This function requires that the strategy have exactly one DTensorSpec as the
        output spec. If the output_specs is a tuple, we throw an exception.
        """
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs
        else:
            raise ValueError(
                f"function output_spec expects a single DTensorSpec but got: {self.output_specs}"
            )

    @cached_property
    def mesh(self):
        # [FUNCTION DOC]
        # Cached property giving the DeviceMesh of this strategy: taken from the
        # single output spec, else the first DTensorSpec in an output tuple, else
        # the first non-None input spec. Raises ValueError for an unexpected
        # output_specs type and AssertionError if no spec is available.
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs.mesh
        elif isinstance(self.output_specs, tuple):
            for out_spec in self.output_specs:
                if isinstance(out_spec, DTensorSpec):
                    return out_spec.mesh
            # all entries are None — fall through to input_specs
        elif self.output_specs is not None:
            raise ValueError(
                f"output_specs expects DTensorSpec, tuple of DTensorSpec, or None but got: {self.output_specs}"
            )
        # output_specs is None or tuple-of-all-None — try input_specs
        if self.input_specs is not None:
            for spec in self.input_specs:
                if spec is not None:
                    return spec.mesh
        raise AssertionError(
            "Cannot determine mesh: no non-None DTensorSpec in output_specs or input_specs"
        )

    def input_spec(self, index: int = 0) -> DTensorSpec:
        # [FUNCTION DOC]
        # Returns the input DTensorSpec at the given index (default 0). Raises
        # AssertionError if input_specs is None or the index is out of range.
        if self.input_specs is None:
            raise AssertionError("input_specs of OpSpec is None!")
        if len(self.input_specs) <= index:
            raise AssertionError(
                f"Invalid index {index} for input_specs of length "
                f"{len(self.input_specs)}: {self.input_specs}"
            )
        return self.input_specs[index]

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Human-readable text: input placements, an arrow, then output placements
        # (the input part and arrow are omitted when input_specs is None).
        if self.input_specs is not None:
            input_specs_str = f"{_pretty_print_spec(self.input_specs)} -> "
        else:
            input_specs_str = ""
        output_spec_str = _pretty_print_spec(self.output_specs)
        return f"{input_specs_str}{output_spec_str}"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash combining the hash of output_specs (None, single spec, or tuple)
        # and of input_specs (0 if None or empty); redistribute_cost is excluded.
        if self.output_specs is None:
            output_hash = hash(None)
        elif isinstance(self.output_specs, DTensorSpec):
            output_hash = hash(self.output_specs)
        else:
            output_hash = hash(tuple(self.output_specs))
        input_hash = hash(tuple(self.input_specs)) if self.input_specs else 0
        return hash((output_hash, input_hash))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equal when other is an OpSpec with equal output_specs and input_specs
        # (redistribute_cost is ignored).
        if not isinstance(other, OpSpec):
            return False
        return (
            self.output_specs == other.output_specs
            and self.input_specs == other.input_specs
        )


class StrategyType:
    # [CLASS DOC]
    # Marker base class for op strategies; subclasses are OpStrategy (list of
    # OpSpec choices for one tensor result) and TupleStrategy (independent child
    # strategies). Has no state or methods.
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """


class OpStrategy(StrategyType):
    # [CLASS DOC]
    # A StrategyType holding the list of candidate OpSpec sharding strategies
    # (attribute strategies) for an op producing a single tensor; all must share
    # one DeviceMesh. Convenience properties (mesh, mesh_shape, ndim, shape,
    # tensor_meta) read from the first strategy's output spec; hash and equality
    # are based on the strategies list.
    """
    OpStrategy that consists of a list of sharding strategies associated with the op,
    where each strategy is an OpSpec that describes the acceptable input/output sharding.

    invariant: the DeviceMesh on all OpSpec must be the same
    """

    def __init__(self, strategies: list[OpSpec]) -> None:
        # [FUNCTION DOC] Stores the list of OpSpec strategies (no copy).
        super().__init__()
        self.strategies: list[OpSpec] = strategies

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Text listing all strategies, suffixed with the mesh shape when it can be
        # determined (AssertionError from mesh lookup is swallowed).
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        try:
            mesh_shape = self.mesh_shape
        except AssertionError:
            return f"OpStrategy[{strategy_list_str}]"
        return f"OpStrategy[{strategy_list_str}] @ mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        # [FUNCTION DOC]
        # Returns the largest num_shards among the output specs of all strategies.
        # Requires every strategy to have a single DTensorSpec output.
        """
        Returns the max number of shards across all OpSpecs
        """
        return max(strategy.output_spec.num_shards for strategy in self.strategies)

    @property
    def mesh(self):
        # [FUNCTION DOC] Property: DeviceMesh of the first strategy.
        return self.strategies[0].mesh

    @property
    def mesh_shape(self):
        # [FUNCTION DOC] Property: shape of the first strategy's DeviceMesh.
        return self.strategies[0].mesh.shape

    @property
    def ndim(self):
        # [FUNCTION DOC] Property: tensor rank of the first strategy's output spec.
        return self.strategies[0].output_spec.ndim

    @property
    def shape(self):
        # [FUNCTION DOC] Property: global tensor shape of the first strategy's output spec.
        return self.strategies[0].output_spec.shape

    @property
    def tensor_meta(self) -> TensorMeta:
        # [FUNCTION DOC] Property: TensorMeta of the first strategy's output spec;
        # raises AssertionError if it is missing.
        # TODO upstream this assert to DTensorSpec itself and fill any missing TensorMetas
        if self.strategies[0].output_spec.tensor_meta is None:
            raise AssertionError
        return self.strategies[0].output_spec.tensor_meta

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of the tuple of strategies.
        return hash(tuple(self.strategies))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is an OpStrategy with equal strategies lists.
        return isinstance(other, OpStrategy) and self.strategies == other.strategies


class TupleStrategy(StrategyType):
    # [CLASS DOC]
    # A StrategyType holding a sequence of child strategies (attribute children)
    # for ops, such as foreach ops, whose tuple/list elements are sharded
    # independently. Registered as a pytree node (flatten to children). Offers a
    # deprecated alias property for children, child_mesh lookup, and hash/equality
    # based on children.
    """
    TupleStrategy is a special case for operators that are fundamentally compound or batched such that some subset
    of the inputs and outputs are completely unrelated to some other subset.

    Generally, foreach_* ops are the most common use-case for TupleStrategy, because they accept lists of inputs,
    but operate independently on each input or tuple of zipped inputs.

    For example, [out_a, out_b] = torch.foreach_add([a,  b], scalar): input a's sharding only affects out_a's sharding,
    independent of b and out_b.

    An example of an operator that should NOT use TupleStrategy is torch.split.  It produces a List[Tensor]
    as its output, but the sharding decision of one output is bound together with the decision
    of each other output and the common input.
    """

    def __init__(
        self,
        children: Sequence[StrategyType],
    ) -> None:
        # [FUNCTION DOC] Stores the sequence of child strategies (no copy).
        super().__init__()
        self.children: Sequence[StrategyType] = children

    @property
    @deprecated(
        "TupleStrategy.childs is deprecated, use TupleStrategy.children instead.",  # codespell:ignore childs
        category=FutureWarning,
    )
    def childs(self) -> Sequence[StrategyType]:  # codespell:ignore childs
        # [FUNCTION DOC] Deprecated property alias returning children; emits a
        # FutureWarning on access.
        """
        Alias for children, to maintain backward compatibility.
        """
        return self.children

    def child_mesh(self, index: int) -> DeviceMesh:
        # [FUNCTION DOC] Returns the DeviceMesh of the child at index; raises
        # AssertionError if that child is not an OpStrategy.
        op_strategy = self.children[index]
        if not isinstance(op_strategy, OpStrategy):
            raise AssertionError
        return op_strategy.mesh

    def __str__(self) -> str:
        # [FUNCTION DOC] Text of the form TupleStrategy(...) listing the children.
        child_strategies_str = ", ".join(
            [f"{str(strat)}" for idx, strat in enumerate(self.children)]
        )
        return f"TupleStrategy({child_strategies_str})"

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of the tuple of children.
        return hash(tuple(self.children))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a TupleStrategy with equal children.
        return isinstance(other, TupleStrategy) and self.children == other.children


try:
    register_pytree_node(
        TupleStrategy,
        lambda node: (node.children, None),
        lambda children, _: TupleStrategy(tuple(children)),
    )
except ValueError:
    # already registered TupleStrategy, skip
    pass


@dataclass
class RuntimeSchemaInfo:
    # [CLASS DOC]
    # Dataclass of per-op runtime hints: static_argnum (index from which args are
    # hashed into the sharding cache key, default 100 meaning effectively none),
    # static_kwargkey (kwarg names that affect sharding propagation), and
    # needs_pytree (whether eager execution must flatten/unflatten args).
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """

    # This static_argnum records static arg "starting index" for ops that have non-tensor
    # args/kwargs which would affect sharding propagation results. All args starting from
    # this index would be hashed to our sharding cache.
    # Note that only a few ops need this information, e.g. view, transpose, var.dim, etc.
    static_argnum: int = 100
    # This static_kwargkey records static kwarg names which would affect sharding prop
    static_kwargkey: list[str] | None = None
    # each op can decide if it wants to use pytree flatten/unflatten during operator
    # eager execution, by default we don't need to do flatten/unflatten, only if the
    # op indicate it needs to, this is to accelerate eager performance.
    needs_pytree: bool = False


@dataclass
class OpSchema:
    # [CLASS DOC]
    # Read-only (by convention) dataclass describing one intercepted op call for
    # sharding propagation: the op overload, args_schema and kwargs_schema (DTensor
    # args replaced by DTensorSpec or OpStrategy, other args kept in order),
    # optional RuntimeSchemaInfo, and a lazily computed comparison key used for
    # hashing/equality and sharding cache lookup. Provides helpers to view spec
    # args, detect inplace/out ops, rewrap suggestions, and compare schemas.
    """
    OpSchema is a data class that describes an operator input schemas, it includes
    DTensorSpecs/OpStrategies (instead of DTensor) and non-tensor args/kwargs (positional
    order preserved). It is mainly used by the DTensor's dispatching logic to perform various
    actions (i.e. sharding propagation, caching sharding decisions, redistribute, etc.)

    NOTE: this must be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec or OpStrategy
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec or OpStrategy
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: RuntimeSchemaInfo | None = None

    _comparison_key: tuple[object, ...] | None = None

    @property
    def args_spec(self) -> tuple[DTensorSpec, ...]:
        # [FUNCTION DOC]
        # Property returning the DTensorSpecs found among the positional args.
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, DTensorSpec))

    @property
    def args_strategy(self) -> tuple[OpStrategy, ...]:
        # [FUNCTION DOC] Property: the OpStrategy items among the positional args
        # (flattened by pytree first when schema_info.needs_pytree), skipping
        # non-strategy values.
        # filter out non-relevant values from args schema to get a clean OpStrategy list
        # separate with args_spec for the ease of type annotation
        # TODO: see if we should merge this with args_spec
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, OpStrategy))

    @property
    def kwargs_strategy(self) -> tuple[OpStrategy, ...]:
        # [FUNCTION DOC] Property: the OpStrategy items among the kwargs values
        # (pytree-flattened when schema_info.needs_pytree).
        # returns OpStrategy items from kwargs_schema.
        kwargs_vals = (
            tree_leaves(self.kwargs_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.kwargs_schema.values()
        )
        return tuple(item for item in kwargs_vals if isinstance(item, OpStrategy))

    @property
    def args_meta(self) -> tuple[TensorMeta | Any, ...]:
        # [FUNCTION DOC] Property: copy of args_schema with each OpStrategy replaced
        # by its TensorMeta and each TupleStrategy by a tuple of converted children,
        # recursing into lists and tuples (type preserved); other values unchanged.
        # Used for single-dim strategy functions that must not see specs or meshes.
        # Used for calling single_dim strategy functions, which aren't allowed to see DTensorSpecs/Meshes
        # like args_spec, but has OpStrategy replaced with corresponding TensorMeta,
        # and TupleStrategy replaced with tuple of TensorMeta
        # preserves the original pytree structure
        # example:
        # args_schema = (OpStrategy1, TupleStrategy([OpStrategy2, OpStrategy3]), OpStrategy4)
        # args_meta: (TensorMeta1, (TensorMeta2, TensorMeta3), TensorMeta4)

        def convert_to_meta(item):
            # [FUNCTION DOC] HELPER: Recursively replaces strategies with tensor
            # metadata (see args_meta); other values are returned as is.
            if isinstance(item, OpStrategy):
                return item.tensor_meta
            elif isinstance(item, TupleStrategy):
                return tuple(convert_to_meta(child) for child in item.children)
            elif isinstance(item, (list, tuple)):
                converted = [convert_to_meta(child) for child in item]
                return type(item)(converted)
            else:
                return item

        return tuple(convert_to_meta(arg) for arg in self.args_schema)

    @property
    def kwargs_meta(self) -> dict[str, object]:
        # [FUNCTION DOC] Property: like args_meta but for kwargs_schema, returning a
        # mapping from kwarg name to converted value.
        # like args_meta, but for kwargs

        def convert_to_meta(item):
            # [FUNCTION DOC] HELPER: Recursively replaces strategies with tensor
            # metadata (see args_meta); other values are returned as is.
            if isinstance(item, OpStrategy):
                return item.tensor_meta
            elif isinstance(item, TupleStrategy):
                return tuple(convert_to_meta(child) for child in item.children)
            elif isinstance(item, (list, tuple)):
                converted = [convert_to_meta(child) for child in item]
                return type(item)(converted)
            else:
                return item

        return {
            key: convert_to_meta(value) for key, value in self.kwargs_schema.items()
        }

    def __repr__(self) -> str:
        # [FUNCTION DOC] Debug text showing the op, str of each positional arg
        # schema, and the kwargs_schema.
        args_schema = ", ".join([str(arg_schema) for arg_schema in self.args_schema])
        return (
            f"OpSchema(op={self.op},"
            f" args_schema=({args_schema}),"
            f" kwargs_schema={self.kwargs_schema})"
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] Readable text: op name, formatted args (specs as-is,
        # single-strategy OpStrategy via its output placements, others via str),
        # and the mesh of the last mesh-bearing arg. Raises AssertionError if an
        # OpStrategy arg has other than one strategy or a TupleStrategy's first
        # child is not an OpStrategy.
        # KNOWN ISSUE: the returned text has an unbalanced closing parenthesis.
        args_schema: list[str] = []
        device_mesh = None

        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_schema.append(str(arg))
                device_mesh = arg.mesh
            elif isinstance(arg, OpStrategy):
                if len(arg.strategies) != 1:
                    raise AssertionError
                args_schema.append(_pretty_print_spec(arg.strategies[0].output_specs))
                device_mesh = arg.mesh
            elif isinstance(arg, TupleStrategy):
                first_op_strategy = arg.children[0]
                if not isinstance(first_op_strategy, OpStrategy):
                    raise AssertionError
                device_mesh = first_op_strategy.mesh
                args_schema.append(str(arg))
            else:
                args_schema.append(str(arg))

        return f"{self.op}({', '.join(args_schema)}) on {device_mesh})"

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Dataclass hook delegating to the C++ post-init helper,
        # which initializes derived state such as the comparison key.
        _DTensor_OpSchema_post_init(self)

    def arg_type_tensor_or_tensor_list_like(self, arg: object) -> bool:
        # [FUNCTION DOC] True if arg is a DTensorSpec, or a list whose elements are
        # all DTensorSpec or None; False otherwise.
        is_tensor = isinstance(arg, DTensorSpec)
        if is_tensor:
            return True

        if not isinstance(arg, list):
            return False

        return all(isinstance(e, DTensorSpec) or e is None for e in arg)

    def return_type_tuple_tensor_like(self) -> bool:
        # [FUNCTION DOC] True if the op schema has more than one return and the
        # first return is a Tensor.
        # all dispatch ops could only return Tuple[Tensor] or have None/ints/floats
        # in the tuple, but the first element must be a Tensor, so this check is enough
        return_types = self.op._schema.returns
        return len(return_types) > 1 and isinstance(
            return_types[0].type, torch.TensorType
        )

    def return_type_list_tensor_like(self) -> bool:
        # [FUNCTION DOC] True if the op schema has exactly one return and it is a
        # list type.
        # returns True if the return type is a List
        return_types = self.op._schema.returns
        return len(return_types) == 1 and isinstance(
            return_types[0].type, torch.ListType
        )

    def return_type_tensor(self) -> bool:
        # [FUNCTION DOC] True if the op schema has at least one return and the
        # first return is a Tensor.
        return_types = self.op._schema.returns
        # all dispatch ops only return Tensor or Tuple[Tensor] for tensor like
        # return types, so this check is enough for tensor like types
        return len(return_types) > 0 and isinstance(
            return_types[0].type, torch.TensorType
        )

    def get_mesh_from_args(self, validate: bool = True) -> DeviceMesh:
        # [FUNCTION DOC] Finds the DeviceMesh from the first DTensorSpec or
        # OpStrategy in args_schema, scanning into lists, tuples and
        # TupleStrategy children. Raises ValueError if none is found. When
        # validate is True, every later top-level spec/strategy arg must use the
        # same mesh, else RuntimeError (cross-mesh op). Nested elements are not
        # validated.
        """
        This util can be used to get a mesh from the OpSchema that contains multiple
        DTensors as arguments. When `validate` is True, it will try to validate that all the
        arguments have the same mesh to avoid unexpected cross mesh errors.

        NOTE: this util currently does not handle TupleStrategy when `validate=True`,
        this is because for TupleStrategy there could be different types of checks, i.e.:
            - for stack and cat like op, we need to check within a TupleStrategy is every
              input is on the same mesh
            - for foreach like ops we need to check "zipped" inputs are on the same mesh
              for each index.
        """
        mesh = None
        # Scan all args to find the first DTensorSpec/OpStrategy (not just the first arg)
        for arg in self.args_schema:
            if isinstance(arg, (DTensorSpec, OpStrategy)):
                mesh = arg.mesh
                break
            elif isinstance(arg, (list, tuple, TupleStrategy)):
                # Scan all elements in the list/tuple, not just the first one,
                # to handle cases like List[Optional[Tensor]] where first elem may be None
                elems = arg.children if isinstance(arg, TupleStrategy) else arg
                for elem in elems:
                    if isinstance(elem, (DTensorSpec, OpStrategy)):
                        mesh = elem.mesh
                        break
                if mesh is not None:
                    break
        if mesh is None:
            raise ValueError(f"Cannot find device mesh from args for op : {self.op}.")

        if validate:
            for arg in self.args_schema[1:]:
                if isinstance(arg, (DTensorSpec, OpStrategy)) and arg.mesh != mesh:
                    raise RuntimeError(
                        f"DTensor does not support cross-mesh operation on {self.op}! "
                        f"Got meshes: {mesh} {arg.mesh}. "
                        f"Please make sure all the arguments have the same DeviceMesh."
                    )

        return mesh

    def is_inplace_op(self) -> bool:
        # [FUNCTION DOC] Heuristic: True if the op schema name ends with an
        # underscore.
        # simple analysis of function schema to determine
        # if this is an inplace variant, it might not
        # be entirely correct, but it's good enough for now.
        return self.op._schema.name[-1] == "_"

    def is_out_variant_op(self) -> bool:
        # [FUNCTION DOC] True if any argument of the op schema is an out argument.
        # simple analysis of function schema to determine
        # if this is an out variant.
        return any(argument.is_out for argument in self.op._schema.arguments)

    def is_view_op(self) -> bool:
        # [FUNCTION DOC] True if the op schema reports itself as a view op.
        return self.op._schema._is_view_op()

    def _recompute_comparison_key(self) -> None:
        # [FUNCTION DOC] HELPER: Delegates to the C++ helper to rebuild the cached
        # comparison key used for hashing and equality after args change.
        _DTensor_OpSchema_recompute_comparison_key(self)

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of the precomputed comparison key.
        return hash(self._comparison_key)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is an OpSchema with the same op, the same
        # number of positional args, and equal comparison keys.
        # early return checks
        if not isinstance(other, OpSchema):
            return False

        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        return self._comparison_key == other._comparison_key

    def gen_fake_args(self) -> ArgsType:
        # [FUNCTION DOC] Returns args_schema with every DTensorSpec replaced by an
        # empty strided tensor built from its tensor_meta (AssertionError if
        # missing); intended for use under fake mode.
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.args_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def gen_fake_kwargs(self) -> KwargsType:
        # [FUNCTION DOC] Same as gen_fake_args but for kwargs_schema.
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.kwargs_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        # [FUNCTION DOC] HELPER: Mutates this schema (whose args hold the suggested
        # DTensorSpecs) so its args_schema follows the origin schema layout: each
        # DTensorSpec position in the origin (pytree-flattened if needed) takes the
        # next suggested spec in order and other args are copied. kwargs_schema is
        # taken from the origin and the comparison key is recomputed.
        suggestion_args_spec = self.args_spec
        new_arg_schema: list[object] = []
        idx_of_args_spec = 0
        if (
            origin_schema.schema_info is not None
            and origin_schema.schema_info.needs_pytree
        ):
            args_schema: Sequence[Any] = tree_leaves(origin_schema.args_schema)
        else:
            args_schema = origin_schema.args_schema
        for arg in args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema
        self._recompute_comparison_key()


@dataclass
class OutputSharding:
    # [CLASS DOC]
    # Result of sharding propagation: output_spec (DTensorSpec, sequence of
    # optional specs, or None) on success, or a redistribute_schema describing the
    # input specs to redistribute to first (with needs_redistribute and
    # use_val_from_redistribute_schema flags). The mesh property derives the
    # DeviceMesh from the output spec.
    """
    OutputSharding is a data class that is used by the sharding propagation,
    it could set the output_spec upon successful propagation. If needs_redistribute
    is set to True, a redistribute_schema would be returned together to indicate
    the input arguments needs to be redistributed before the op execution.

    NOTE: the redistribute_schema generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensorSpecs
    """

    # specifies the output sharding pattern
    output_spec: OutputSpecType
    # schema for redistribution if needed
    redistribute_schema: OpSchema | None = None
    # flag indicating if inputs need redistribution
    needs_redistribute: bool = False
    # flag to use values from `redistribute_schema`
    use_val_from_redistribute_schema: bool = False

    @cached_property
    def mesh(self):
        # [FUNCTION DOC] Cached property: mesh of the single output spec, or of the
        # first element when output_spec is a tuple. Raises ValueError if the spec
        # (or first tuple element) is not a DTensorSpec, including when None or a
        # non-tuple sequence.
        if isinstance(self.output_spec, DTensorSpec):
            return self.output_spec.mesh
        elif isinstance(self.output_spec, tuple):
            out_spec = self.output_spec[0]
            if isinstance(out_spec, DTensorSpec):
                return out_spec.mesh
            else:
                raise ValueError(f"Unknown output spec type: {type(out_spec)}")
        else:
            raise ValueError(f"Unknown output spec type: {type(self.output_spec)}")


@dataclass
class OpInfo:
    # [CLASS DOC]
    # Dataclass bundling runtime execution info for one DTensor op call:
    # compute_mesh (first mesh among args), schema (OpSchema, None on the C++
    # cached fast path), flat_args_schema, local_args, local_kwargs, optional
    # args_tree_spec, and the output_sharding once propagation has run.
    """
    All Runtime Op execution info are packed here
    """

    # The first compute device mesh recorded from args
    # NOTE: one op could have multiple meshes from its args. We just record the first
    # mesh here to check if current rank should participate in computation or not.
    compute_mesh: DeviceMesh

    # compete runtime operator infos
    # NOTE: schema can be None due to C++ fast path optimization. When the C++
    # dispatch layer (dispatchDTensorOp in python_variable.cpp) finds a cached
    # sharding decision, it skips creating the full OpSchema to reduce CPU overhead.
    # In this case, OpInfo is created with create_schema=False, setting schema to None.
    # The operator information is still available through output_sharding.redistribute_schema
    # when redistribution is needed.
    schema: OpSchema | None
    flat_args_schema: list[object]
    local_args: Sequence[object]
    local_kwargs: dict[str, object]
    args_tree_spec: TreeSpec | None = None

    # the output sharding info
    output_sharding: OutputSharding | None = None
