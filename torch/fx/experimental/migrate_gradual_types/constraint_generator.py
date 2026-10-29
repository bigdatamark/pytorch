# [MODULE DOC] Constraint generation for gradual type migration on fx graphs.
# Holds a global table mapping call targets (functions, method names, module classes) to inference rules, each producing constraints
# over tensor and dimension variables for one node; ConstraintGenerator drives the rules over a whole graph.
# Also defines MAX_TENSOR_RANK (4) and re-exported helper builders listed in __all__. Import-time side effect: registers all rules in the global table
# (registering a duplicate target raises an error).
import operator
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias, TypeVar
from typing_extensions import ParamSpec

import torch
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import (
    ApplyBroadcasting,
    BinConstraintD,
    BinConstraintT,
    BVar,
    CalcConv,
    CalcMaxPool,
    CalcProduct,
    CanReshape,
    Conj,
    Constraint,
    DGreatestUpperBound,
    Disj,
    DVar,
    F,
    GetItem,
    GetItemTensor,
    IndexSelect,
    T,
    TGreatestUpperBound,
    Transpose,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_consistency,
    op_div,
    op_eq,
    op_gt,
    op_leq,
    op_lt,
    op_matching,
    op_mul,
    op_neq,
    op_precision,
    op_sub,
)
from torch.fx.experimental.migrate_gradual_types.util import (
    gen_bvar,
    gen_dvar,
    gen_nat_constraints,
    gen_tensor_dims,
    gen_tvar,
)
from torch.fx.graph import Graph
from torch.fx.node import Node, Target
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


_T = TypeVar("_T")
_P = ParamSpec("_P")

_SymbolDict: TypeAlias = dict[Node, TVar | DVar | BVar]

_INFERENCE_RULES: dict[Target, Callable[..., tuple[list[Constraint], int]]] = {}

MAX_TENSOR_RANK = 4

__all__ = [
    "ConstraintGenerator",
    "adaptive_inference_rule",
    "add_layer_norm_constraints",
    "add_linear_constraints",
    "arange_inference_rule",
    "assert_inference_rule",
    "batchnorm_inference_rule",
    "bmm_inference_rule",
    "broadcasting_inference_rule",
    "conv2d_inference_rule",
    "cumsum_inference_rule",
    "embedding_inference_rule",
    "embedding_inference_rule_functional",
    "eq_inference_rule",
    "equality_inference_rule",
    "expand_inference_rule",
    "flatten_inference_rule",
    "full_inference_rule",
    "gen_broadcasting_constraints",
    "gen_embedding_rules",
    "gen_layer_norm_constraints",
    "generate_flatten_constraints",
    "get_attr_inference_rule",
    "getitem_inference_rule",
    "gt_inference_rule",
    "index_select_inference_rule",
    "layer_norm_functional",
    "layer_norm_inference_rule",
    "linear_constraints",
    "linear_inference_rule",
    "lt_inference_rule",
    "masked_fill_inference_rule",
    "maxpool_inference_rule",
    "neq_inference_rule",
    "range_check",
    "register_inference_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "size_inference_rule",
    "tensor_inference_rule",
    "torch_dim_inference_rule",
    "torch_linear_inference_rule",
    "transpose_inference_rule",
    "type_inference_rule",
    "view_inference_rule",
]


def register_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] HELPER: Decorator factory that registers an inference rule for a call target (function, method name or module class).
    # The returned decorator stores the function in the global rule table and returns it unchanged.
    # Raises RuntimeError at decoration time if the target already has a registered rule.
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Decorator body: adds fn to the global inference rule table under call_target and returns fn.
        # Raises RuntimeError if call_target is already registered. Side effect: mutates the global rule table.
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f"Inference rule already registered for {call_target}!")
        _INFERENCE_RULES[call_target] = fn  # pyrefly: ignore[unsupported-operation]
        return fn

    return register


def generate_flatten_constraints(
    start_dim: int, end_dim: int, input: TVar, flattened: TVar, n: int, counter: int
) -> tuple[Conj, int]:
    # [FUNCTION DOC] Builds the constraint set for flattening a rank-n input into a tensor between start_dim and end_dim.
    # Creates n fresh dimension variables, requires the input to equal a tensor of those dims, adds a
    # product constraint relating the flattened tensor to the dims, and adds natural-number constraints.
    # A start_dim of -1 means n; other negative values use their absolute value; a negative end_dim is
    # normalized from the end and then made exclusive. Returns (conjunction, updated counter).
    d, counter = gen_tensor_dims(n, counter)
    c1 = BinConstraintT(input, TensorType(d), op_eq)
    start_dim = n if start_dim == -1 else abs(start_dim)
    end_dim = n + end_dim + 1 if end_dim < 0 else end_dim + 1
    c2 = CalcProduct(start_dim, end_dim, flattened, d)
    nat_constraints = gen_nat_constraints(d)
    return Conj([c1, c2, *nat_constraints]), counter


@register_inference_rule(getattr)
def get_attr_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for attribute access on a tensor node.
    # Only the device attribute is supported: output shape equals the input shape (equality constraint).
    # Raises AssertionError if the arguments are not a node and a string, NotImplementedError for any other attribute.
    # Side effect: records a fresh tensor variable for the node in the symbol mapping.
    """
    If the attribute is "device" then the tensor shape is preserved
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], str):
        raise AssertionError(f"Expected str, got {type(n.args[1])}")
    output, counter = gen_tvar(counter)
    symbols[n] = output

    input = symbols[n.args[0]]
    attr = n.args[1]

    if attr == "device":
        return [BinConstraintT(input, output, op_eq)], counter
    else:
        raise NotImplementedError("Not yet implemented")


@register_inference_rule(torch.bmm)
def bmm_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for batched matrix multiplication.
    # Produces one disjunction of four cases: all of input1, input2 and output are Dyn; only input1 is Dyn
    # (output batch/last dim taken from input2, middle dim Dyn); only input2 is Dyn (symmetric); or both are
    # rank-3 tensors with consistent batch dims, output batch as the greatest upper bound of the two batch dims,
    # and output shape batch x rows(input1) x cols(input2). Raises AssertionError if args are not nodes.
    """
    Constraints that match the input to a size 3 tensor
    and switch the dimensions according to the rules
    of batch multiplication
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    bmm_output, counter = gen_tvar(counter)
    symbols[n] = bmm_output

    bmm_input1 = symbols[n.args[0]]
    bmm_input2 = symbols[n.args[1]]

    dims_input1, counter = gen_tensor_dims(3, counter)
    dims_input2, counter = gen_tensor_dims(3, counter)

    inputs_dyn = Conj(
        [
            BinConstraintT(bmm_input1, Dyn, op_eq),
            BinConstraintT(bmm_input2, Dyn, op_eq),
            BinConstraintT(bmm_output, Dyn, op_eq),
        ]
    )

    input1_dyn = Conj(
        [
            BinConstraintT(bmm_input1, Dyn, op_eq),
            BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
            BinConstraintT(
                bmm_output, TensorType([dims_input2[0], Dyn, dims_input2[2]]), op_eq
            ),
        ]
    )

    input2_dyn = Conj(
        [
            BinConstraintT(bmm_input2, Dyn, op_eq),
            BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
            BinConstraintT(
                bmm_output, TensorType([dims_input1[0], dims_input1[1], Dyn]), op_eq
            ),
        ]
    )

    consistency_constraints = [
        BinConstraintD(dims_input1[0], dims_input2[0], op_consistency)
    ]

    batch_size, counter = gen_dvar(counter)

    inputs_are_tensors = Conj(
        [
            BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
            BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
            BinConstraintT(
                bmm_output,
                TensorType([batch_size, dims_input1[1], dims_input2[2]]),
                op_eq,
            ),
            *consistency_constraints,
            DGreatestUpperBound(batch_size, dims_input1[0], dims_input2[0]),
        ]
    )

    return [Disj([inputs_dyn, input1_dyn, input2_dyn, inputs_are_tensors])], counter


@register_inference_rule("index_select")
def index_select_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for index_select.
    # Arg 0 is the input tensor node, arg 1 the integer dimension, arg 2 the index tensor node.
    # The index tensor must be a rank-1 tensor or Dyn; for each possible input rank up to the maximum an
    # IndexSelect constraint is generated that replaces the selected dimension by the index length (or Dyn).
    # Returns a single disjunction over the two cases. Raises AssertionError on wrongly typed arguments.
    """
    We constrain the second argument to a vector or Dyn.
    The output replaces the input with the shape of the vector
    at the position given by the index (first argument)
    """
    # print(n.args)
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], int):
        raise AssertionError(f"Expected int, got {type(n.args[1])}")
    if not isinstance(n.args[2], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[2])}")

    index_select, counter = gen_tvar(counter)
    symbols[n] = index_select

    dims, counter = gen_tensor_dims(1, counter)

    # equality constraint
    is_size_1 = BinConstraintT(symbols[n.args[2]], TensorType(dims), op_eq)
    is_dyn = BinConstraintT(symbols[n.args[2]], Dyn, op_eq)

    c2 = Conj(
        [
            is_size_1,
            Disj(
                [
                    IndexSelect(
                        i + 1,
                        symbols[  # pyrefly: ignore[bad-argument-type, bad-index]
                            n.args[0]
                        ],
                        dims[0],
                        n.args[1],
                        index_select,
                    )
                    for i in range(MAX_TENSOR_RANK)
                ]
            ),
        ]
    )
    c3 = Conj(
        [
            is_dyn,
            Disj(
                [
                    IndexSelect(
                        i + 1,
                        symbols[  # pyrefly: ignore[bad-argument-type, bad-index]
                            n.args[0]
                        ],
                        Dyn,
                        n.args[1],
                        index_select,
                    )
                    for i in range(MAX_TENSOR_RANK)
                ]
            ),
        ]
    )

    return [Disj([c2, c3])], counter


@register_inference_rule("expand")
def expand_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for expand.
    # Treats expand like a broadcast between the input and a synthetic tensor whose shape is given by the
    # remaining args (ints or dimension-variable nodes). Adds broadcasting constraints, an equality fixing the
    # output rank to the number of size args, natural-number constraints and non-negativity of node sizes.
    # Raises AssertionError for non-node input or size args that are neither node nor int (nodes must map to dimension variables).
    # KNOWN ISSUE: the constraints parameter is shadowed by the broadcasting result, so the caller list is ignored.
    """
    We generate the exact constraints as we do for tensor additions but we constraint
    the rank of this expression to be equal to len(n.args[1:]) so that only
    those cases get considered for the output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # define the output for expand
    expand, counter = gen_tvar(counter)
    symbols[n] = expand

    # since we do not have two nodes here, we will construct an argument variable
    e1 = symbols[n.args[0]]
    e2, counter = gen_tvar(counter)

    e2_nat_constraints = []
    for arg in n.args[1:]:
        if not isinstance(arg, (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(arg)}")
        if isinstance(arg, Node):
            if not isinstance(symbols[arg], DVar):
                raise AssertionError(f"Expected DVar, got {type(symbols[arg])}")
            e2_nat_constraints.append(BinConstraintD(0, symbols[arg], op_leq))

    e2_constraint = BinConstraintT(
        e2,
        TensorType(
            [
                arg
                if isinstance(arg, int)
                else symbols[arg]  # pyrefly: ignore[bad-index]
                for arg in n.args[1:]
            ]
        ),
        op_eq,
    )

    constraints, counter = gen_broadcasting_constraints(
        e1,  # pyrefly: ignore[bad-argument-type]
        e2,
        symbols,
        counter,
        expand,
    )

    # constraint the output size
    dims, counter = gen_tensor_dims(len(n.args[1:]), counter)
    nat_constraints = gen_nat_constraints(dims)
    c = [
        BinConstraintT(expand, TensorType(dims), op_eq),
        *nat_constraints,
        e2_constraint,
        *e2_nat_constraints,
    ]
    constraints += c

    return constraints, counter


@register_inference_rule(torch.nn.functional.gelu)
@register_inference_rule(torch.nn.functional.dropout)
@register_inference_rule(torch.nn.functional.softmax)
@register_inference_rule("detach")
@register_inference_rule("to")
@register_inference_rule("int")
@register_inference_rule("long")
@register_inference_rule("contiguous")
@register_inference_rule(torch.ones)
@register_inference_rule(torch.zeros)
def equality_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for shape-preserving or shape-creating ops (gelu, dropout, softmax, detach, to, int, long, contiguous, ones, zeros).
    # If arg 0 is a node with a tensor variable, output equals input. If it is a node with dimension variables, all
    # args must be dimension variables and the output is a tensor of those dims. If arg 0 is a tuple (at most 4 entries),
    # the output is a tensor whose dims are the symbols of the entries. Otherwise raises NotImplementedError.
    # KNOWN ISSUE: tuple entries and dimension args are looked up in the symbol mapping, so plain int sizes would fail.
    """
    We generate the constraint: input = output
    """
    output, counter = gen_tvar(counter)
    symbols[n] = output

    if isinstance(n.args[0], Node):
        input = symbols[n.args[0]]
        if isinstance(input, TVar):
            return [BinConstraintT(input, output, op_eq)], counter

        # then we have dimension variables
        else:
            for arg in n.args:
                if not isinstance(symbols[arg], DVar):  # pyrefly: ignore[bad-index]
                    raise AssertionError(
                        f"Expected DVar, got {type(symbols[arg])}"  # pyrefly: ignore[bad-index]
                    )
        my_size = [symbols[arg] for arg in n.args]  # pyrefly: ignore[bad-index]
        return [BinConstraintT(output, TensorType(my_size), op_eq)], counter

    elif isinstance(n.args[0], tuple):
        # then the tuple is the size
        if len(n.args[0]) > 4:
            raise AssertionError(f"Expected len <= 4, got {len(n.args[0])}")
        my_size = [symbols[arg] for arg in n.args[0]]  # pyrefly: ignore[bad-index]
        return [BinConstraintT(output, TensorType(my_size), op_eq)], counter
    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule("transpose")
def transpose_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for transpose (modeled as two index selects).
    # Args: input node, two integer dims. Result is a disjunction: input and output both Dyn, or a Transpose
    # constraint for each possible rank up to the maximum. Raises AssertionError on wrong argument types or if the input is not a tensor variable.
    """
    Can be considered as a sequence of two index selects, so we generate constraints accordingly
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], int):
        raise AssertionError(f"Expected int, got {type(n.args[1])}")
    if not isinstance(n.args[2], int):
        raise AssertionError(f"Expected int, got {type(n.args[2])}")

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    if not isinstance(from_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(from_arg)}")

    # input and output are dyn
    is_dyn = Conj(
        [BinConstraintT(from_arg, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)]
    )

    # or input is a tensor and we actually do the replacement
    c3 = Disj(
        [
            Transpose(i + 1, from_arg, n.args[1], n.args[2], output)
            for i in range(MAX_TENSOR_RANK)
        ]
    )

    return [Disj([is_dyn, c3])], counter


@register_inference_rule("type_as")
def type_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for type_as.
    # Requires both args to be nodes with tensor variables; the first must be consistent with the second and the output equals the second.
    # KNOWN ISSUE: docstring says input = output, but the output is constrained to equal the type_as argument (arg 1), not the input.
    # Raises AssertionError on wrong argument kinds.
    """
    We generate the constraint: input = output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    to_arg = symbols[n.args[1]]

    if not isinstance(from_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(from_arg)}")
    if not isinstance(to_arg, TVar):
        raise AssertionError(f"Expected TVar, got {type(to_arg)}")

    return [
        BinConstraintT(from_arg, to_arg, op_consistency),
        BinConstraintT(output, to_arg, op_eq),
    ], counter


@register_inference_rule("masked_fill_")
def masked_fill_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for masked_fill_.
    # Handled like a tensor addition: when both args are tensor variables, returns broadcasting constraints with a fresh output variable.
    # Raises NotImplementedError if either symbol is not a tensor variable, AssertionError if args are not nodes.
    """
    Similar to addition. For now we implement the constraints when
    the argument is a boolean tensor. There is also a case for when
    it is a condition. We will leave this out for now.
    """

    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")

    # We will retrieve the type variables from the symbol table
    # and confirm they are tensor variables

    e1 = symbols[n.args[0]]
    e2 = symbols[n.args[1]]

    if isinstance(e1, TVar) and isinstance(e2, TVar):
        masked_fill_tensor, counter = gen_tvar(counter)
        symbols[n] = masked_fill_tensor
        return gen_broadcasting_constraints(
            e1, e2, symbols, counter, masked_fill_tensor
        )
    else:
        raise NotImplementedError("Not yet implemented")


@register_inference_rule(torch.nn.functional.embedding)
def embedding_inference_rule_functional(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the functional embedding call.
    # Arg 0 is the index input, arg 1 the weight. Constrains the weight to a rank-2 tensor with fresh dims (static shape, no matching)
    # and delegates to the shared embedding rule using the second weight dim as embedding size. Returns weight equality plus those constraints.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    embedding_dim_weights = symbols[n.args[1]]  # pyrefly: ignore[bad-index]

    # will treat this as a static shape. So we will not use matching.
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(
        embedding_dim_weights, TensorType(weight_dims), op_eq
    )
    embedding_dim = weight_dims[1]
    constraints, counter = gen_embedding_rules(n, symbols, embedding_dim, counter)
    return [equality_constraint] + constraints, counter


@register_inference_rule(torch.nn.modules.sparse.Embedding)
def embedding_inference_rule(
    n: Node,
    module_instance: torch.nn.Embedding,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the Embedding module.
    # Output shape equals the input shape with the module's embedding_dim appended. Delegates to gen_embedding_rules.
    # Raises AssertionError if arg 0 is not a node.
    """
    The output shape differs from the input shape in the last dimension
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_embedding_rules(n, symbols, module_instance.embedding_dim, counter)


def gen_embedding_rules(
    n: Node, symbols: _SymbolDict, embedding_dim: int | DVar, counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] HELPER: Shared embedding constraint builder.
    # Creates the output variable; result is a disjunction of: input and output both Dyn, or input a tensor of rank 1 to
    # (maximum rank minus one) with the output being the same dims plus embedding_dim at the end, with natural-number constraints. Returns (constraints, counter).
    embedding_output, counter = gen_tvar(counter)
    symbols[n] = embedding_output
    embedding_input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(embedding_input, Dyn, op_eq)
    output_dyn = BinConstraintT(embedding_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])
    c2 = []

    for i in range(1, MAX_TENSOR_RANK):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)

        # we consider all tensor sizes and append embedding_dim to the end of the output dimension in all cases
        c_tensor_i = Conj(
            [
                BinConstraintT(embedding_input, TensorType(new_dims), op_eq),
                BinConstraintT(
                    embedding_output, TensorType(new_dims + [embedding_dim]), op_eq
                ),
            ]
            + nat_constraints
        )
        c2.append(c_tensor_i)

    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.tensor)
def tensor_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for tensor creation: produces no constraints (scalars unsupported) and leaves the counter unchanged.
    """
    If the tensor is a scalar, we will skip it since we
    do not support scalars yet. We will add support in the future
    if it's needed. For our examples so far, scalars are not needed.
    """
    return [], counter  # pyrefly: ignore[implicit-any]


@register_inference_rule("reshape")
@register_inference_rule("view")
def view_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for reshape and view methods.
    # Builds the target shape from the remaining args (node symbols or ints); each -1 becomes a fresh dimension variable.
    # Adds an equality for the output type, a CanReshape constraint from the source, and not-Dyn constraints for every target dim.
    # KNOWN ISSUE: the extra stride condition for view (see TODO) is not implemented.
    """
    Similar to reshape but with an extra condition on the strides
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    my_view, counter = gen_tvar(counter)
    symbols[n] = my_view

    src_var = symbols[n.args[0]]
    t2 = [
        symbols[elem] if isinstance(elem, Node) else elem for elem in n.args[1:]
    ]  # target shape
    t2_type = []
    num_constraints = []

    for t in t2:
        if t == -1:
            var, counter = gen_dvar(counter)
            t2_type.append(var)
            # pyrefly: ignore [bad-argument-type]
            num_constraints.append(BinConstraintD(var, Dyn, op_neq))

        else:
            # pyrefly: ignore [bad-argument-type]
            num_constraints.append(BinConstraintD(t, Dyn, op_neq))
            t2_type.append(t)  # type: ignore[arg-type]

    t2_type = TensorType(t2_type)  # type: ignore[assignment]

    c1 = BinConstraintT(my_view, t2_type, op_eq)
    c2 = CanReshape(src_var, t2_type)  # pyrefly: ignore[bad-argument-type]

    # TODO: add the extra check mentioned here:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

    return [c1, c2] + num_constraints, counter  # type: ignore[operator]


@register_inference_rule("size")
def size_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for size.
    # With one arg, the result equals the input tensor variable. With two args where the second is an int, the result is a dimension
    # variable: either the input is Dyn and the dim is Dyn, or a GetItem constraint per possible rank with a natural-number result.
    # Raises NotImplementedError for a non-int index or any other argument count (the TODO notes the Dyn case is under review).
    """
    The constraint is just lhs = rhs.
    Ex: size = input_ids.size()
    """

    if len(n.args) == 1:
        # generate the new variable
        size, counter = gen_tvar(counter)
        symbols[n] = size
        input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
        c = BinConstraintT(input, size, op_eq)
        return [c], counter

    elif len(n.args) == 2:
        # TODO: review this rule; should input = dyn; output = dyn be included here?
        if isinstance(n.args[1], int):
            # generate the new variable
            size_index, counter = gen_dvar(counter)
            symbols[n] = size_index
            input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
            c2 = [
                GetItem(
                    i + 1,
                    n.args[1],
                    size_index,
                    input,  # pyrefly: ignore[bad-argument-type]
                )
                for i in range(MAX_TENSOR_RANK)
            ]
            c3 = BinConstraintD(0, size_index, op_leq)

            input_dyn = BinConstraintT(input, Dyn, op_eq)
            output_dyn = BinConstraintD(size_index, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])

            return [Disj([c1, Conj([Disj(c2), c3])])], counter

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


def range_check(i: int, n: int) -> T | F:
    # [FUNCTION DOC] HELPER: Returns a true or false constraint object stating whether index i is valid for a list of size n.
    # For non-negative i it is true when i is less than n.
    # KNOWN ISSUE: for negative i it is true only when i >= n, which can never hold for positive n (should compare with minus n); also the docstring says Boolean but constraint objects are returned.
    """
    Checks if an index i is within range of a size n list
    Args:
        i: index
        n: list size

    Returns: Boolean
    """
    if i >= 0:
        return T() if i < n else F()
    else:
        return T() if i >= n else F()


@register_inference_rule(torch.cumsum)
def cumsum_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for cumsum.
    # Output shape equals input shape. Result is a disjunction: both Dyn, or for each rank up to the maximum, both equal the same dims with
    # a range check on the dim argument (positional or keyword) and natural-number constraints.
    # Raises AssertionError if the input is not a node or the dim is not an int.
    """
    Input and output shapes should be equal
    We should verify that the index is valid
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_1 = n.args[1] if len(n.args) > 1 else n.kwargs["dim"]
    if not isinstance(arg_1, int):
        raise AssertionError(f"Expected int, got {type(arg_1)}")

    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)
    c1 = Conj([input_dyn, output_dyn])
    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims), op_eq),
                BinConstraintT(output, TensorType(new_dims), op_eq),
            ]
            + [range_check(arg_1, i)]
            + nat_constraints
        )

        c2.append(c_tensor_i)
    dyn_or_tensor = Disj([c1, Disj(c2)])
    return [dyn_or_tensor], counter


@register_inference_rule(_assert_is_none)
def assert_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the internal assert-is-none call: produces no constraints.
    # Raises AssertionError if the node has any users.
    if len(n.users) != 0:
        raise AssertionError(f"Expected no users, got {len(n.users)}")
    return [], counter  # pyrefly: ignore[implicit-any]


@register_inference_rule(operator.getitem)
def getitem_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for indexing with the getitem operator.
    # Int index: output is a fresh dimension variable; Dyn input gives Dyn output, otherwise GetItem constraints per rank plus non-negativity.
    # Tuple index: output is a fresh tensor variable; Dyn input gives Dyn output, otherwise GetItemTensor constraints per rank.
    # If the indexed node has no symbol in the tuple case, returns no constraints (TODO in code). The int case requires a tensor variable (AssertionError otherwise).
    # Raises RuntimeError for other index kinds.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # dimension output case
    if isinstance(n.args[1], int):
        # create and store the new dimension variable
        get_item_output, counter = gen_dvar(counter)
        symbols[n] = get_item_output

        # retrieve arg variables
        get_item_arg = symbols[n.args[0]]
        if not isinstance(get_item_arg, TVar):
            raise AssertionError(f"Expected TVar, got {type(get_item_arg)}")

        # if the input is dynamic, we accept any index and return
        # a dynamic dimension as output
        input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
        output_dyn = BinConstraintD(get_item_output, Dyn, op_eq)
        c1 = Conj([input_dyn, output_dyn])

        # if the input is a tensor,
        # generate a getItem constraint which will be expanded based on the
        # tensor dimension.

        c2 = [
            GetItem(i + 1, n.args[1], get_item_output, get_item_arg)
            for i in range(MAX_TENSOR_RANK)
        ]

        # since the output is a dimension, we make sure it's a natural number
        # added as a conjunction to the disjunction of c2
        c3 = BinConstraintD(0, get_item_output, op_leq)
        return [Disj([c1, Conj([Disj(c2), c3])])], counter

    # tensor output case
    elif isinstance(n.args[1], tuple):
        # create and store the new tensor variable
        get_item_output, counter = gen_tvar(counter)  # type: ignore[arg-type,assignment]
        symbols[n] = get_item_output

        # retrieve arg variables
        if n.args[0] in symbols:
            get_item_arg = symbols[n.args[0]]
            if not isinstance(get_item_arg, TVar):
                raise AssertionError(f"Expected TVar, got {type(get_item_arg)}")

            input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
            output_dyn = BinConstraintT(get_item_output, Dyn, op_eq)  # type: ignore[assignment]
            c1 = Conj([input_dyn, output_dyn])

            c2 = [
                GetItemTensor(i + 1, n.args[1], get_item_output, get_item_arg)  # type: ignore[misc]
                for i in range(MAX_TENSOR_RANK)
            ]
        else:
            # TODO: we should figure out why there is a key-error here.
            return [], counter  # pyrefly: ignore[implicit-any]

        return [Disj([c1, *c2])], counter

    else:
        raise RuntimeError("Method not yet implemented")


@register_inference_rule(operator.gt)
def gt_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for greater-than.
    # Node vs node: tensor variables give broadcasting constraints with a fresh output; dimension variables give a boolean variable equal to the comparison
    # (for flow analysis only); otherwise RuntimeError (sort mismatch). Node vs int: dimension variable gives the boolean comparison; a tensor variable
    # with an int triggers a warning, the node's symbol is replaced by a fresh dimension variable and the comparison is built.
    # Other cases raise NotImplementedError. Side effect: may overwrite a symbol and emit a warning.
    # KNOWN ISSUE: a bare statement reading the symbol after reassignment has no effect.
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    # We make sure this node will not be used again. We do not
    # generate a constraint about that node. Only about the operands.

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            gt_tensor, counter = gen_tvar(counter)
            symbols[n] = gt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, gt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        elif isinstance(e1, TVar) and isinstance(e2, int):
            # then we made the wrong assumption about the argument being a tensor
            # so we should fix the assumption
            warnings.warn(
                f"Made the wrong assumption for node {n}. Correctness not guaranteed."
            )

            new_e1, counter = gen_dvar(counter)
            symbols[n.args[0]] = new_e1
            symbols[n.args[0]]

            gt_constraint = BinConstraintD(new_e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(operator.eq)
def eq_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for equality comparison.
    # Node vs node: tensor variables give broadcasting constraints with a fresh output; dimension variables give a boolean variable equal to the comparison
    # (flow analysis only); otherwise RuntimeError. Node vs int: only dimension variables are supported. Other cases raise NotImplementedError.
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            eq_tensor, counter = gen_tvar(counter)
            symbols[n] = eq_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, eq_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return [equality_constraint], counter
        else:
            raise NotImplementedError("Method not yet implemented")
    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(operator.ne)
def neq_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for not-equal between a tensor node and a tuple of three or four sizes (work in progress; only the false-condition case works).
    # Builds a boolean variable equal to: the input has that rank and at least one dimension pair is non-Dyn and different.
    # Other tuple lengths raise NotImplementedError; arg types are checked with AssertionError.
    # KNOWN ISSUE: in the size 4 case the fourth dimension check tests the third input dim against Dyn instead of the fourth.
    # KNOWN ISSUE: a boolean variable is generated twice per branch (first one unused), the size 3 case generates four dims but uses three, and the node itself gets no symbol.
    """
    Translates to inconsistent in gradual types.
    To prove inequality, we should prove that
    tensors are either different sizes or
    disagree on at least one dimension

    This is a WIP (works when the condition
    is false. We are working on making this operation work
    when the condition is true as well)
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], tuple):
        raise AssertionError(f"Expected tuple, got {type(n.args[1])}")

    # implementing for size 3 and 4
    if len(n.args[1]) == 3:
        if not isinstance(n.args[1][0], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][0])}")
        if not isinstance(n.args[1][1], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][1])}")
        if not isinstance(n.args[1][2], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][2])}")

        lhs = symbols[n.args[0]]

        b, counter = gen_tensor_dims(4, counter)
        input_is_size3 = BinConstraintT(lhs, TensorType([b[0], b[1], b[2]]), op_eq)

        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]

        # dimensions not equal
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b[0], op_neq)
        neq_2 = BinConstraintD(d2, b[1], op_neq)
        neq_3 = BinConstraintD(d3, b[2], op_neq)

        # dimensions inconsistent
        dims_inconsistent1 = Conj(
            [BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b[0], Dyn, op_neq), neq_1]
        )
        dims_inconsistent2 = Conj(
            [BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b[1], Dyn, op_neq), neq_2]
        )
        dims_inconsistent3 = Conj(
            [BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b[2], Dyn, op_neq), neq_3]
        )

        dims_inconsistent = Disj(
            [dims_inconsistent1, dims_inconsistent2, dims_inconsistent3]
        )

        # we are covering size 3 and 4 only for now
        ne_constraint = Conj([input_is_size3, dims_inconsistent])

        my_ne, counter = gen_bvar(counter)
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)

    elif len(n.args[1]) == 4:
        if not isinstance(n.args[1][0], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][0])}")
        if not isinstance(n.args[1][1], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][1])}")
        if not isinstance(n.args[1][2], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][2])}")
        if not isinstance(n.args[1][3], (Node, int)):
            raise AssertionError(f"Expected Node or int, got {type(n.args[1][3])}")

        lhs = symbols[n.args[0]]

        b1, counter = gen_dvar(counter)
        b2, counter = gen_dvar(counter)
        b3, counter = gen_dvar(counter)
        b4, counter = gen_dvar(counter)

        input_is_size4 = BinConstraintT(lhs, TensorType([b1, b2, b3, b4]), op_eq)

        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]
        d4 = n.args[1][3] if isinstance(n.args[1][3], int) else symbols[n.args[1][3]]

        # dimensions not equal
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b1, op_neq)
        neq_2 = BinConstraintD(d2, b2, op_neq)
        neq_3 = BinConstraintD(d3, b3, op_neq)
        neq_4 = BinConstraintD(d4, b4, op_neq)

        # dimensions to inconsistent
        dims_inconsistent1 = Conj(
            [BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b1, Dyn, op_neq), neq_1]
        )
        dims_inconsistent2 = Conj(
            [BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b2, Dyn, op_neq), neq_2]
        )
        dims_inconsistent3 = Conj(
            [BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_3]
        )
        dims_inconsistent4 = Conj(
            [BinConstraintD(d4, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_4]
        )

        dims_inconsistent = Disj(
            [
                dims_inconsistent1,
                dims_inconsistent2,
                dims_inconsistent3,
                dims_inconsistent4,
            ]
        )

        ne_constraint = Conj([input_is_size4, dims_inconsistent])

        my_ne, counter = gen_bvar(counter)

        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)

    else:
        raise NotImplementedError("Method not yet implemented")

    return [equality_constraint], counter


@register_inference_rule(operator.lt)
def lt_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for less-than; mirrors the greater-than rule without the tensor-with-int repair.
    # Node vs node: tensors give broadcasting constraints with a fresh output; dimension variables give a boolean variable equal to the comparison.
    # Node vs int requires a dimension variable. Otherwise RuntimeError or NotImplementedError.
    if not isinstance(n.args[0], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[0])}")
    if not isinstance(n.args[1], (Node, int)):
        raise AssertionError(f"Expected Node or int, got {type(n.args[1])}")

    # We make sure this node will not be used again. We do not
    # generate a constraint about that node. Only about the operands.

    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            lt_tensor, counter = gen_tvar(counter)
            symbols[n] = lt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, lt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # This is meant to be used for flow analysis only
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError("Sort Mismatch")

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # This is meant to be used for flow analysis only
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter
        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        raise NotImplementedError("Method not yet implemented")


@register_inference_rule(torch.full)
def full_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for full.
    # Arg 0 is an iterable of sizes (ints or nodes with symbols); the output is a tensor with those dims. Raises AssertionError if arg 0 is not iterable.
    full, counter = gen_tvar(counter)
    symbols[n] = full
    res = []

    if not isinstance(n.args[0], Iterable):
        raise AssertionError(f"Expected Iterable, got {type(n.args[0])}")
    for arg in n.args[0]:
        dim = (
            arg if isinstance(arg, int) else symbols[arg]  # pyrefly: ignore[bad-index]
        )
        res.append(dim)
    c = BinConstraintT(full, TensorType(list(res)), op_eq)  # type: ignore[arg-type]
    return [c], counter


# TODO normalize index
@register_inference_rule(torch.arange)
def arange_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for arange with a single end argument (start 0, step 1).
    # Output is a rank-1 tensor with a fresh length dimension. Either some bound is Dyn and the length is Dyn, or none is Dyn and the length equals
    # (end - start) divided by step. Raises NotImplementedError for other argument counts.
    # KNOWN ISSUE: the end argument is looked up in the symbol mapping, so a literal int end would fail; the TODO about index normalization is open.
    start = 0
    step = 1

    if len(n.args) == 1:
        end = symbols[n.args[0]]  # pyrefly: ignore[bad-index]
    else:
        raise NotImplementedError("Not yet implemented")

    # int((end - start) / step)
    d1, counter = gen_dvar(counter)
    size_constraint = BinConstraintD(
        d1, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq
    )
    arange, counter = gen_tvar(counter)
    symbols[n] = arange

    # either a parameter is a number or it is Dyn
    c1 = Disj(
        [
            BinConstraintD(end, Dyn, op_eq),
            BinConstraintD(start, Dyn, op_eq),
            BinConstraintD(step, Dyn, op_eq),
        ]
    )
    c2 = BinConstraintD(d1, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    c11 = Conj(
        [
            BinConstraintD(end, Dyn, op_neq),
            BinConstraintD(start, Dyn, op_neq),
            BinConstraintD(step, Dyn, op_neq),
        ]
    )
    c22 = BinConstraintD(d1, Dyn, op_neq)
    both_numbers = Conj([c11, c22, size_constraint])

    return [
        BinConstraintT(arange, TensorType([d1]), op_eq),
        Disj([both_dyn, both_numbers]),
    ], counter


def gen_broadcasting_constraints(
    e1: TVar, e2: TVar, symbols: _SymbolDict, counter: int, output_var: TVar
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] HELPER: Builds the three constraints for broadcasting two tensor variables into output_var.
    # Creates two extra tensor variables for the broadcast operands; returns a greatest-upper-bound constraint on the output, an ApplyBroadcasting
    # constraint and a consistency constraint between the operands, plus the updated counter.
    # additional vars that don't correspond to expressions
    e11, counter = gen_tvar(counter)
    e22, counter = gen_tvar(counter)

    # generate constraints
    c1 = TGreatestUpperBound(output_var, e11, e22)
    c2 = ApplyBroadcasting(e11, e22, e1, e2)
    c3 = BinConstraintT(e11, e22, op_consistency)
    return [c1, c2, c3], counter


@register_inference_rule(operator.mul)
@register_inference_rule(torch.ne)
@register_inference_rule("ne")
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def broadcasting_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:  # pyrefly: ignore[bad-return]
    # [FUNCTION DOC] Inference rule for add, mul and ne (function and method forms).
    # Node-node: both tensor variables give broadcasting constraints with a fresh output, otherwise NotImplementedError.
    # Node-number: a tensor variable gives output equal input; a dimension variable gives output equal to the arithmetic result (add or mul) and non-negative.
    # Number-node: symmetric. Both scalar raises NotImplementedError.
    # KNOWN ISSUE: op_code stays unset for ne, and in the node-number branch a symbol that is neither tensor nor dimension falls through and returns nothing.
    op_code = None
    if n.target is operator.add or n.target is torch.add:
        op_code = op_add
    elif n.target is operator.mul:
        op_code = op_mul

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(symbols[n.args[0]], TVar) and isinstance(
            symbols[n.args[1]], TVar
        ):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            e2 = symbols[n.args[1]]

            return gen_broadcasting_constraints(
                e1,  # pyrefly: ignore[bad-argument-type]
                e2,  # pyrefly: ignore[bad-argument-type]
                symbols,
                counter,
                my_output,
            )
        else:
            raise NotImplementedError("Method not yet implemented")

    elif isinstance(n.args[0], Node) and isinstance(n.args[1], (int, float)):
        if isinstance(symbols[n.args[0]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            return [BinConstraintT(my_output, e1, op_eq)], counter
        elif isinstance(symbols[n.args[0]], DVar):
            my_output, counter = gen_dvar(counter)  # type: ignore[arg-type,assignment]
            symbols[n] = my_output
            e1 = symbols[n.args[0]]

            # we will propagate the runtime value here since this is regular addition
            c = Conj(
                [
                    BinConstraintD(
                        my_output, BinConstraintD(e1, n.args[1], op_code), op_eq
                    ),
                    BinConstraintD(0, my_output, op_leq),
                ]
            )
            return [c], counter

    elif isinstance(n.args[1], Node) and isinstance(n.args[0], (int, float)):
        if isinstance(symbols[n.args[1]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]
            return [BinConstraintT(my_output, e2, op_eq)], counter
        elif isinstance(symbols[n.args[1]], DVar):
            my_output, counter = gen_dvar(counter)  # type: ignore[arg-type,assignment]
            symbols[n] = my_output
            e2 = symbols[n.args[1]]

            # we will propagate the runtime value here since this is regular addition
            c = Conj(
                [
                    BinConstraintD(
                        my_output, BinConstraintD(e2, n.args[0], op_code), op_eq
                    ),
                    BinConstraintD(0, my_output, op_leq),
                ]
            )
            return [c], counter

        else:
            raise NotImplementedError("Method not yet implemented")

    else:
        # TODO generate add constraints for scalar addition
        raise NotImplementedError("Addition not yet implemented")


@register_inference_rule(torch.flatten)
def flatten_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for flatten.
    # Defaults start_dim 1 and end_dim -1, overridden by int args 1 and 2. Result is a disjunction: input and output Dyn, or flatten constraints for each rank up to the maximum.
    # Raises AssertionError on wrongly typed args.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    flattened, counter = gen_tvar(counter)
    symbols[n] = flattened

    input = symbols[n.args[0]]

    # set the default start and end dims
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    c1 = BinConstraintT(input, Dyn, op_eq)
    c2 = BinConstraintT(flattened, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    const = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        c, counter = generate_flatten_constraints(
            start_dim,
            end_dim,
            input,  # pyrefly: ignore[bad-argument-type]
            flattened,
            i,
            counter,
        )
        const.append(c)

    return [Disj([both_dyn, *const])], counter


@register_inference_rule(torch.nn.functional.layer_norm)
def layer_norm_functional(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for functional layer_norm: delegates to the shared layer-norm constraint builder using arg 1 as the normalized shape.
    # Raises AssertionError if arg 0 is not a node.
    """
    We generate the constraint: input = output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_layer_norm_constraints(
        n,
        n.args[1],  # pyrefly: ignore[bad-argument-type]
        symbols,
        counter,
    )


@register_inference_rule(torch.nn.LayerNorm)
def layer_norm_inference_rule(
    n: Node,
    module_instance: torch.nn.LayerNorm,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the LayerNorm module: output shape equals input shape and the trailing input dims must be consistent with the module normalized_shape.
    # Raises AssertionError if arg 0 is not a node.
    """
    Input and output shapes should be equal.
    Input should be consistent with the normalized_shape
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return gen_layer_norm_constraints(
        n, module_instance.normalized_shape, symbols, counter
    )


def gen_layer_norm_constraints(
    n: Node, normalized_shape: Sequence[int], symbols: _SymbolDict, counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] HELPER: Shared layer-norm builder. Disjunction of: input and output Dyn, or for each rank both equal the same dims, trailing dims consistent with normalized_shape
    # (see add_layer_norm_constraints) and natural-number constraints. Returns (constraints, counter).
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims_rhs)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims_rhs), op_eq),
                BinConstraintT(output, TensorType(new_dims_rhs), op_eq),
            ]
            + add_layer_norm_constraints(new_dims_rhs, list(normalized_shape))
            + nat_constraints
        )
        c2.append(c_tensor_i)
    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.nn.Dropout)
@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(
    n: Node,
    module_instance: torch.nn.Module,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for ReLU and Dropout modules: output equals input, which must be a tensor variable (AssertionError otherwise).
    """
    Input and output shapes should be equal.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    if not isinstance(input, TVar):
        raise AssertionError(f"Expected TVar, got {type(input)}")
    return [BinConstraintT(input, output, op_eq)], counter


@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(
    n: Node,
    module_instance: torch.nn.Linear,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the Linear module: delegates to linear_constraints with the module in_features and out_features.
    # Raises AssertionError if arg 0 is not a node.
    """
    Input and output sizes should be the same except for the last dimension
    If the input is Dyn, then so should the output
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    return linear_constraints(
        n, module_instance.in_features, module_instance.out_features, symbols, counter
    )


@register_inference_rule("dim")
def torch_dim_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the dim method. Output is a dimension variable: Dyn if the input is Dyn, otherwise the rank (1 to maximum) matching an input tensor of that rank.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    my_dim, counter = gen_dvar(counter)
    symbols[n] = my_dim
    input = symbols[n.args[0]]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintD(my_dim, Dyn, op_eq)

    c1 = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)

        c_tensor_i = Conj(
            [
                BinConstraintT(input, TensorType(new_dims_rhs_1), op_eq),
                BinConstraintD(my_dim, i, op_eq),
            ]
        )
        c1.append(c_tensor_i)

    return [Disj([Conj([input_dyn, output_dyn]), Disj(c1)])], counter


@register_inference_rule(torch._C._nn.linear)
def torch_linear_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for the functional linear call. Constrains the weight (arg 1) to a rank-2 tensor with fresh dims and uses the
    # second dim as in_features and the first as out_features in linear_constraints. Returns the weight equality plus those constraints.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(
        symbols[n.args[1]],  # pyrefly: ignore[bad-index]
        TensorType(weight_dims),
        op_eq,
    )
    constraints, counter = linear_constraints(
        n, weight_dims[1], weight_dims[0], symbols, counter
    )
    return [equality_constraint] + constraints, counter


def linear_constraints(
    n: Node,
    in_features: int | DVar,
    out_features: int | DVar,
    symbols: _SymbolDict,
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] HELPER: Shared linear builder. Disjunction of: input and output Dyn, or for each rank input and output are tensors whose leading dims
    # are equal, last input dim is consistent with in_features, last output dim equals out_features, plus natural-number constraints.
    linear_output, counter = gen_tvar(counter)
    symbols[n] = linear_output
    linear_input = symbols[n.args[0]]  # pyrefly: ignore[bad-index]

    input_dyn = BinConstraintT(linear_input, Dyn, op_eq)
    output_dyn = BinConstraintT(linear_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj(
            [
                BinConstraintT(linear_input, TensorType(new_dims_rhs_1), op_eq),
                BinConstraintT(linear_output, TensorType(new_dims_rhs_2), op_eq),
            ]
            + add_linear_constraints(
                new_dims_rhs_1, new_dims_rhs_2, in_features, out_features
            )
            + nat_constraints
        )
        c2.append(c_tensor_i)
    return [Disj([c1, Disj(c2)])], counter


def add_layer_norm_constraints(
    input_dim: list[DVar], normalized_dim: list[int]
) -> list[Constraint]:
    # [FUNCTION DOC] HELPER: Returns consistency constraints between the trailing input dims and the normalized dims, aligned from the end.
    # Returns a single false constraint if normalized_dim is longer than input_dim.
    """
    The constraints say that the type has the form: ``[*, 1024, 1024]``
    while the normalized_dim have the form ``[1024, 1024]``.

    Args:
        input_dim: Input shape of layer norm
        normalized_dim: normalized_dim parameter of the module instance
    """

    # in this case we return false since there's a pattern mismatch
    if len(normalized_dim) > len(input_dim):
        return [F()]

    else:
        constraints: list[Constraint] = []
        for i, n in zip(reversed(input_dim), reversed(normalized_dim)):
            constraints.append(BinConstraintD(i, n, op_consistency))
        return constraints


def add_linear_constraints(
    dims1: list[DVar],
    dims2: list[DVar],
    in_features: int | DVar,
    out_features: int | DVar,
) -> list[Constraint]:
    # [FUNCTION DOC] HELPER: Per-dimension constraints for linear: all leading dims of the two lists are equal, last dim of dims1 is consistent with in_features,
    # last dim of dims2 equals out_features. Raises AssertionError if the lists differ in length.
    if len(dims1) != len(dims2):
        raise AssertionError(f"Expected same length, got {len(dims1)} vs {len(dims2)}")
    constraints: list[Constraint] = []
    for i in range(len(dims1)):
        if i == len(dims1) - 1:
            constraints.append(BinConstraintD(dims1[i], in_features, op_consistency))
            constraints.append(BinConstraintD(dims2[i], out_features, op_eq))
        else:
            constraints.append(BinConstraintD(dims1[i], dims2[i], op_eq))

    return constraints


@register_inference_rule(torch.reshape)
def reshape_inference_rule(
    n: Node, symbols: _SymbolDict, constraints: list[Constraint], counter: int
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for torch.reshape. The target shape (arg 1) maps -1 to Dyn; output equals that tensor type and a CanReshape constraint links the source.
    # Raises AssertionError if arg 0 is not a node.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    my_reshape, counter = gen_tvar(counter)
    symbols[n] = my_reshape

    src_var = symbols[n.args[0]]
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])  # type: ignore[union-attr]
    c1 = BinConstraintT(my_reshape, t2_type, op_eq)  # type: ignore[union-attr]
    c2 = CanReshape(src_var, t2_type)  # pyrefly: ignore[bad-argument-type]

    return [c1, c2], counter


@register_inference_rule(BatchNorm2d)
def batchnorm_inference_rule(
    n: Node,
    module_instance: BatchNorm2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for BatchNorm2d. Input must match a rank-4 tensor of fresh dims (or all Dyn) and output equals input; adds natural-number constraints.
    # The module instance is unused.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # generate the new variable
    batchnorm_output, counter = gen_tvar(counter)
    symbols[n] = batchnorm_output
    batchnorm_input = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    c1 = BinConstraintT(batchnorm_input, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(batchnorm_input, batchnorm_output, op_eq)
    return [c1, c2, *nat_constraints], counter


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptive_inference_rule(
    n: Node,
    module_instance: torch.nn.AdaptiveAvgPool2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for AdaptiveAvgPool2d. Input matches a rank-4 tensor; output keeps the first two dims and uses the module output_size for the last two. Adds natural-number constraints.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    avg_pool, counter = gen_tvar(counter)

    symbols[n] = avg_pool
    input_var = symbols[n.args[0]]

    # dim vars
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(
        avg_pool,
        TensorType(
            [
                d1,
                d2,
                module_instance.output_size[  # pyrefly: ignore[bad-index, unsupported-operation]
                    0
                ],
                module_instance.output_size[  # pyrefly: ignore[bad-index, unsupported-operation]
                    1
                ],
            ]
        ),
        op_eq,
    )

    return [c1, c2, *nat_constraints], counter


@register_inference_rule(Conv2d)
def conv2d_inference_rule(
    n: Node,
    module_instance: Conv2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for Conv2d. Input matches a rank-4 tensor, in_channels must be consistent with the second dim, and a CalcConv constraint
    # carries output channels, kernel, padding, stride and dilation of the module. Adds natural-number constraints.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    my_conv, counter = gen_tvar(counter)
    symbols[n] = my_conv
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    # c1 = Matching(input_var, TensorType([d1, d2, d3, d4]))
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    # c2 = DConsistency(module_instance.in_channels, d2)
    c2 = BinConstraintD(module_instance.in_channels, d2, op_consistency)

    c3 = CalcConv(
        my_conv,
        input_var,  # pyrefly: ignore[bad-argument-type]
        module_instance.out_channels,
        module_instance.kernel_size,  # pyrefly: ignore[bad-argument-type]
        module_instance.padding,  # pyrefly: ignore[bad-argument-type]
        module_instance.stride,  # pyrefly: ignore[bad-argument-type]
        module_instance.dilation,  # pyrefly: ignore[bad-argument-type]
        [d1, d2, d3, d4],
    )

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, c3, *nat_constraints], counter


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool_inference_rule(
    n: Node,
    module_instance: torch.nn.MaxPool2d,
    symbols: _SymbolDict,
    constraints: list[Constraint],
    counter: int,
) -> tuple[list[Constraint], int]:
    # [FUNCTION DOC] Inference rule for MaxPool2d. Input matches a rank-4 tensor and a CalcMaxPool constraint carries kernel, padding, stride and dilation. Adds natural-number constraints.
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    maxpool, counter = gen_tvar(counter)
    symbols[n] = maxpool
    input_var = symbols[n.args[0]]

    # dim vars
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    c2 = CalcMaxPool(
        maxpool,
        input_var,  # pyrefly: ignore[bad-argument-type]
        module_instance.kernel_size,
        module_instance.padding,
        module_instance.stride,
        module_instance.dilation,
        [d1, d2, d3, d4],
    )

    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    return [c1, c2, *nat_constraints], counter


class ConstraintGenerator:
    # [CLASS DOC] Walks an fx graph and produces gradual-typing constraints using the registered inference rules.
    # State: traced (module), traced_params (mapping of parameter name to tensor), constraints (list passed to rules; never populated),
    # symbol_dict (mapping from node to its type or dimension variable, filled while generating) and graph (taken from the traced module when it has one, else the given graph).
    # Key methods: generate_constraints (whole graph) and generate_constraints_node (one node).
    # KNOWN ISSUE: the constraints list is passed to rules but nothing appends to it, despite the docstring of generate_constraints.
    def __init__(self, traced: torch.nn.Module, graph: Graph | None = None) -> None:
        # [FUNCTION DOC] Stores the traced module, a mapping of its named parameters, an empty constraints list and symbol dictionary, and the graph
        # (the module's own graph attribute if present, else the graph argument).
        self.traced = traced  # traced or tracer.root
        self.traced_params = dict(self.traced.named_parameters())
        self.constraints = []
        self.symbol_dict = {}
        self.graph = traced.graph if hasattr(traced, "graph") else graph

    def generate_constraints(self, counter: int = 0) -> tuple[Conj, int]:
        # [FUNCTION DOC] Iterates over every node of the graph in order, collecting constraints from generate_constraints_node and threading the counter.
        # Returns (conjunction of all constraints, final counter). Side effect: fills symbol_dict.
        # KNOWN ISSUE: docstring claims self.constraints is populated, but it is not.
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        graph = self.graph

        all_constraints: list[Constraint] = []

        # pyrefly: ignore [missing-attribute]
        for n in graph.nodes:
            (constraints, counter) = self.generate_constraints_node(n, counter)
            all_constraints += constraints

        return Conj(all_constraints), counter

    def generate_constraints_node(
        self, n: Node, counter: int
    ) -> tuple[list[Constraint], int]:
        # [FUNCTION DOC] Generates constraints for one node by its op kind.
        # placeholder: fresh tensor variable; declared type (or Dyn, or the example value shape for parameters) is a precision bound of the variable, and its rank is at most the maximum.
        # call_function and call_method: dispatches on target in the rule table; call_module: dispatches on module class; get_attr: tensor parameters with nonzero rank get an equality to their static shape, others produce nothing;
        # output: nothing. Raises RuntimeError if no rule is registered, NotImplementedError for unknown ops, AssertionError if a parameter placeholder lacks example_value.
        # KNOWN ISSUE: docstring lists only reshape, add and conv2d as supported; many more rules exist.
        """
        Generate constraints the given node:
        Currently supported operations:
        - Reshape
        - Add
        - conv2d
        """

        if n.op == "placeholder":
            x, counter = gen_tvar(counter)
            self.symbol_dict[n] = x

            my_type = n.type

            if n.type != Dyn and (not isinstance(n.type, TensorType)):
                if n.type == torch.nn.parameter.Parameter:
                    # since we have a parameter, the shape must be static
                    if "example_value" not in n.meta:
                        raise AssertionError("example_value not in n.meta")
                    my_type = TensorType(n.meta["example_value"].size())
                else:
                    my_type = Dyn

            c1 = BinConstraintT(my_type, x, op_precision)
            c2 = BinConstraintT(x, MAX_TENSOR_RANK, op_leq)
            return [c1, c2], counter

        elif n.op == "call_function":
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](
                    n, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "call_module":
            module_instance = self.traced.get_submodule(
                n.target  # pyrefly: ignore[bad-argument-type]
            )
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](
                    n, module_instance, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for class {type(module_instance)}!"
                )

        elif n.op == "call_method":
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](
                    n, self.symbol_dict, self.constraints, counter
                )
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "get_attr":
            t = self.traced_params.get(  # pyrefly: ignore[no-matching-overload]
                n.target, None
            )

            if isinstance(t, torch.Tensor):
                if len(t.shape) > 0:
                    res = list(t.shape)
                    attr_type = TensorType(res)
                    output, counter = gen_tvar(counter)
                    self.symbol_dict[n] = output
                    return [BinConstraintT(output, attr_type, op_eq)], counter
                else:
                    # scalar?
                    return [], counter  # pyrefly: ignore[implicit-any]
            else:
                return [], counter  # pyrefly: ignore[implicit-any]

        elif n.op == "output":
            return [], counter  # pyrefly: ignore[implicit-any]

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")
