# [MODULE DOC] Flattening of pytrees guided by a precomputed TreeSpec, used by FX/export for
# containers where the spec is trusted. Holds two registries (container type to flatten
# function, and container type to exact-match check function), the registration helpers,
# and the tree_flatten_spec entry point. Import-time side effect: registers the built-in
# dict, list, tuple, every torch return-type and the namedtuple factory.
from collections import namedtuple
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import NamedTuple

import torch.return_types
from torch.utils._pytree import PyTree, tree_flatten, TreeSpec


FlattenFnSpec = Callable[[PyTree, TreeSpec], list[Any]]
FlattenFnExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

# Keep deprecated alias for backward compatibility
FlattenFuncSpec = FlattenFnSpec  # deprecated
FlattenFuncExactMatchSpec = FlattenFnExactMatchSpec  # deprecated

SUPPORTED_NODES: dict[type[Any], FlattenFnSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], FlattenFnExactMatchSpec | None] = {}

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFnSpec,
    flatten_fn_exact_match_spec: FlattenFnExactMatchSpec | None = None,
) -> None:
    # [FUNCTION DOC] Record, in the two module-level registries, the function that extracts the
    # direct children of a container of type cls according to a spec, and an optional check
    # that the container exactly matches a spec. Overwrites any previous registration for
    # cls. Returns nothing; mutates global state.
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec


def _deregister_pytree_flatten_spec(
    cls: type[Any],
) -> None:
    # [FUNCTION DOC] HELPER: Remove cls from both registries. Raises a key error if cls was not
    # registered. Mutates global state.
    del SUPPORTED_NODES[cls]
    del SUPPORTED_NODES_EXACT_MATCH[cls]


def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
) -> list[Any]:
    # [FUNCTION DOC] Flatten a pytree into a list of leaves following the given spec. A leaf spec
    # yields a one-element list holding the input. If the spec's container type is
    # registered, the registered function extracts children by the spec (so missing extra
    # entries are tolerated) and each child is flattened recursively against the matching
    # child spec, concatenating results in order. Otherwise the standard flattener is used
    # and its spec must equal the given spec, else a RuntimeError is raised asking for a bug
    # report. No side effects.
    # KNOWN ISSUE: pairing children with child specs stops at the shorter sequence, so a
    # mismatch in child count is silently ignored on the registered path.
    if spec.is_leaf():
        return [pytree]
    # I guess these exist for BC, FC reasons.
    # In general, we should be able to directly
    # use pytree tree flattener to flatten them,
    # as export serializes the pytree separately.
    # Will remove it in follow up PR.
    if spec.type in SUPPORTED_NODES:
        flatten_fn_spec = SUPPORTED_NODES[spec.type]
        child_pytrees = flatten_fn_spec(pytree, spec)
        result: list[Any] = []
        for child, child_spec in zip(child_pytrees, spec.children()):
            flat = tree_flatten_spec(child, child_spec)
            result += flat
        return result
    flat_result, real_spec = tree_flatten(pytree)
    if spec != real_spec:
        raise RuntimeError(
            f"Real spec {real_spec} of object {pytree} is different from expected spec {spec}. "
            f"Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml"
        )
    return flat_result


def _dict_flatten_spec(d: dict[_K, _V], spec: TreeSpec) -> list[_V]:
    # [FUNCTION DOC] HELPER: Return the mapping's values in the order of the keys stored in the
    # spec context. Raises a key error if a key is missing.
    return [d[k] for k in spec.context]


def _list_flatten_spec(d: list[_T], spec: TreeSpec) -> list[_T]:
    # [FUNCTION DOC] HELPER: Return the first num_children items of the list, by position.
    # Raises an index error if the list is shorter.
    return [d[i] for i in range(spec.num_children)]


def _tuple_flatten_spec(d: tuple[_T, ...], spec: TreeSpec) -> list[_T]:
    # [FUNCTION DOC] HELPER: Return the first num_children items of the tuple as a list.
    # Raises an index error if the tuple is shorter.
    return [d[i] for i in range(spec.num_children)]


def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> list[Any]:
    # [FUNCTION DOC] HELPER: Return the first num_children fields of the named tuple, by
    # position, as a list. Raises an index error if it is shorter.
    return [d[i] for i in range(spec.num_children)]


def _dict_flatten_spec_exact_match(d: dict[_K, _V], spec: TreeSpec) -> bool:
    # [FUNCTION DOC] HELPER: True when the mapping has exactly as many entries as the spec has
    # children (keys are not compared).
    return len(d) == spec.num_children


def _list_flatten_spec_exact_match(d: list[_T], spec: TreeSpec) -> bool:
    # [FUNCTION DOC] HELPER: True when the list length equals the spec's child count.
    return len(d) == spec.num_children


def _tuple_flatten_spec_exact_match(d: tuple[_T, ...], spec: TreeSpec) -> bool:
    # [FUNCTION DOC] HELPER: True when the tuple length equals the spec's child count.
    return len(d) == spec.num_children


def _namedtuple_flatten_spec_exact_match(d: NamedTuple, spec: TreeSpec) -> bool:
    # [FUNCTION DOC] HELPER: True when the named tuple length equals the spec's child count.
    return len(d) == spec.num_children


register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(
    tuple,
    _tuple_flatten_spec,
    _tuple_flatten_spec_exact_match,
)
for return_type in torch.return_types.all_return_types:
    register_pytree_flatten_spec(
        return_type,
        _tuple_flatten_spec,
        _tuple_flatten_spec_exact_match,
    )
register_pytree_flatten_spec(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten_spec,
    _namedtuple_flatten_spec_exact_match,
)
