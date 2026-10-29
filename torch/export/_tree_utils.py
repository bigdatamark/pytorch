# [MODULE DOC]
# Small helpers for comparing and normalizing pytree specs used by export.
# Provides reorder_kwargs (put user kwargs into the order recorded in an exported
# program's in_spec) and is_equivalent (recursive TreeSpec comparison with a
# caller-supplied type/context equivalence predicate). No import-time side effects.
from collections.abc import Callable
from typing import Any

from torch.utils._pytree import Context, TreeSpec


def reorder_kwargs(user_kwargs: dict[str, Any], spec: TreeSpec) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Returns a new mapping holding user_kwargs reordered to follow the key order
    # stored in the kwargs child of spec (an in_spec shaped like (args, kwargs)).
    # Raises AssertionError if spec is not a 2-child tuple or its second child is
    # not a mapping; raises ValueError if the key sets differ. No side effects.
    """Reorder user-provided kwargs to match the order in `spec`. `spec` is
    expected to be the in_spec of an exported program, i.e. the spec that
    results from flattening `(args, kwargs)`.

    We need this to provide consistent input ordering, such so that users can
    pass in foo(a=a, b=b) OR foo(b=b, a=a) and receive the same result.
    """
    # Make sure that the spec is actually shaped like (args, kwargs)
    if spec.type is not tuple:
        raise AssertionError(f"Expected spec type to be tuple, but got {spec.type}")
    if spec.num_children != 2:
        raise AssertionError(
            f"Expected spec to have 2 children, but got {spec.num_children}"
        )
    kwargs_spec = spec.child(1)
    if kwargs_spec.type is not dict:
        raise AssertionError(
            f"Expected kwargs_spec type to be dict, but got {kwargs_spec.type}"
        )

    if set(user_kwargs) != set(kwargs_spec.context):
        raise ValueError(
            f"Ran into a kwarg keyword mismatch: "
            f"Got the following keywords {list(user_kwargs)} but expected {kwargs_spec.context}"
        )

    reordered_kwargs = {}
    for kw in kwargs_spec.context:
        reordered_kwargs[kw] = user_kwargs[kw]

    return reordered_kwargs


def is_equivalent(
    spec1: TreeSpec,
    spec2: TreeSpec,
    equivalence_fn: Callable[[type | None, Context, type | None, Context], bool],
) -> bool:
    # [FUNCTION DOC]
    # Recursively checks two TreeSpecs: applies equivalence_fn to each node's type
    # and context pair, requires equal child counts, then compares children
    # pairwise. Returns True only if every node pair passes, else False. Pure.
    """Customizable equivalence check for two TreeSpecs.

    Arguments:
        spec1: The first TreeSpec to compare
        spec2: The second TreeSpec to compare
        equivalence_fn: A function to determine the equivalence of two
            TreeSpecs by examining their types and contexts. It will be called like:

                equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context)

            This function will be applied recursively to all children.

    Returns:
        True if the two TreeSpecs are equivalent, False otherwise.
    """
    if not equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context):
        return False

    # Recurse on children
    if spec1.num_children != spec2.num_children:
        return False

    for child_spec1, child_spec2 in zip(spec1.children(), spec2.children()):
        if not is_equivalent(child_spec1, child_spec2, equivalence_fn):
            return False

    return True
