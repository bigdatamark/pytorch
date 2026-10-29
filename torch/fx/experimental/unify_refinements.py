# [MODULE DOC] Glue between the FX gradual type checker's refinement step and
# the unification library: collects equality constraints produced by Refine,
# unifies them into a most general unifier (substitution), and rewrites the
# node types of a graph with that substitution. Public functions are listed in
# __all__. No import-time side effects beyond imports.
from typing import Any

import torch
import torch.fx
from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.unification import unify, Var  # type: ignore[attr-defined]
from torch.fx.tensor_type import TensorType


__all__ = [
    "check_for_type_equality",
    "convert_eq",
    "infer_symbolic_types",
    "infer_symbolic_types_single_pass",
    "substitute_all_types",
    "substitute_solution_one_type",
    "unify_eq",
]


def infer_symbolic_types_single_pass(traced: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Run one round of refinement on the traced module, unify the
    # collected equality constraints, and rewrite the graph's node types in
    # place with the resulting substitution. Returns nothing.
    """
    Calls our symbolic inferencer once.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)


def infer_symbolic_types(traced: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Run refinement, unification and type substitution twice on
    # the traced module (the second pass resolves information such as
    # broadcasting that one pass misses), then ask the second refiner to compute
    # its symbolic relations. Mutates node types of the graph in place.
    """
    Calls our symbolic inferencer twice.
    This is useful when one pass is not enough
    to infer all the information such as the case
    for broadcasting.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)

    r.symbolic_relations()


def convert_eq(list_of_eq: list[Equality]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    # [FUNCTION DOC] Split a list of equality constraints into a tuple of all
    # left-hand sides and a tuple of all right-hand sides (same order), the
    # form unify expects.
    """
    Convert equality constraints in the right format
    to be used by unification library.
    """
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return tuple(lhs), tuple(rhs)


def unify_eq(list_of_eq: list[Equality]) -> Any:
    # [FUNCTION DOC] Unify the left-hand sides of all constraints with the
    # right-hand sides. Returns the substitution mapping (variables to values),
    # or False if the constraints cannot be unified.
    """
    Apply unification to a set of
    equality constraints
    """
    lhs, rhs = convert_eq(list_of_eq)
    return unify(lhs, rhs)


def substitute_solution_one_type(mapping: dict[object, object], t: object) -> Any:
    # [FUNCTION DOC] Return type t with the substitution applied. A variable is
    # replaced by its mapped value if any. For a tensor type, each dimension that
    # is a key in the mapping is replaced (one level only, no recursion into the
    # replacement). Lists and tuples are processed element by element
    # recursively into a new container of the same kind. Anything else is
    # returned unchanged. No mutation of inputs.
    """
    Apply the most general unifier to a type
    """
    if isinstance(t, Var):
        if t in mapping:
            return mapping[t]
        else:
            return t

    elif isinstance(t, TensorType):
        new_type = []
        for typ in t.dims:
            if typ in mapping:
                new_type.append(mapping[typ])
            else:
                new_type.append(typ)
        return TensorType(tuple(new_type))

    elif isinstance(t, list):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return new_type

    elif isinstance(t, tuple):
        new_type = []
        for typ in t:
            new_type.append(substitute_solution_one_type(mapping, typ))
        return tuple(new_type)

    else:
        return t


def substitute_all_types(graph: torch.fx.Graph, mapping: dict[object, object]) -> None:
    # [FUNCTION DOC] First collapse chains in mapping in place: repeat until no
    # value changes, replacing a value that is itself a key by that key's value.
    # Then set every graph node's type to the substituted type. Mutates both the
    # mapping and the node types.
    # KNOWN ISSUE: if unification failed, mapping is False (from unify_eq) and
    # iterating it raises TypeError; a cyclic mapping would loop forever.
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping:
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True

    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)


def check_for_type_equality(g1: torch.fx.Graph, g2: torch.fx.Graph) -> bool:
    # [FUNCTION DOC] True if the nodes of g1 and g2, compared pairwise in order,
    # all have equal types. Extra nodes in the longer graph are ignored.
    """
    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    """
    for n, m in zip(g1.nodes, g2.nodes):
        if n.type != m.type:
            return False
    return True
