# Copyright (c) Meta Platforms, Inc. and affiliates
# ruff: noqa: F821
# flake8: noqa: F821
# [MODULE DOC] Utilities for walking and editing nested state dicts addressed by object paths (tuples of string keys and
# integer indices): traverse_state_dict (current) and traverse_state_dict_v_2_3 (legacy), set_element, get_element and
# print_tensor. Defines the type aliases PATH_ITEM, OBJ_PATH, STATE_DICT_ITEM, CONTAINER_TYPE and TypeVar T.
# Exports: traverse_state_dict, set_element, get_element, print_tensor. No import-time side effects.
from collections.abc import Callable, Collection, Mapping, MutableMapping
from typing import cast, TypeVar
from typing_extensions import TypeIs

import torch
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.tensor import DTensor


PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
T = TypeVar("T")

STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> TypeIs[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Default traversal predicate: true when the value is a tensor.
    return isinstance(value, torch.Tensor)


# TODO: update docstring for traverse.py
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    # [FUNCTION DOC] Walks state_dict and calls visitor with (path, value) for each leaf. Mappings are always descended
    # into (keys converted to strings); lists are descended into only if they contain tensors, mappings or lists
    # that keep_traversing/terminal checks say need visiting, otherwise the whole list is passed to the visitor as a leaf.
    # Other values are leaves. Deletes the inner helper closures at the end to break reference cycles (avoids leaks in async save).
    # KNOWN ISSUE: the docstring says lists or tuples are treated as containers, but the terminal check only recognizes lists, so tuples are always visited as leaves.
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """

    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        # [FUNCTION DOC] HELPER: True if the value is not a mapping and is either a non-list or a list that holds no mappings, no non-terminal lists and no entries accepted by keep_traversing.
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # [FUNCTION DOC] HELPER: Recursive step: mappings descend by key, terminal values go to the visitor with their path, lists/tuples descend by index.
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)

    # release reference cycle to prevent memory leaks in async_save
    del _traverse_obj, _is_terminal


def traverse_state_dict_v_2_3(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    # [FUNCTION DOC] Legacy (pre-2.4) traversal kept to reconstruct old checkpoints. Calls visitor for every value
    # that is terminal, where a mapping or list is terminal when none of its entries are containers needing
    # traversal or accepted by keep_traversing; so a dict of non-tensors is visited as one leaf. Otherwise descends
    # into mappings and lists. Deletes helper closures at the end to avoid reference cycles.
    """
    Traversal is short-circuited when it finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """

    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        # [FUNCTION DOC] HELPER: True if the value is a non-container, or a mapping/list whose entries are all terminal and rejected by keep_traversing.
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # [FUNCTION DOC] HELPER: Recursive step: calls the visitor on terminal values, otherwise descends into mappings by key and lists by index.
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)

    # release reference cycle to prevent memory leaks in async_save
    del _traverse_obj, _is_terminal


def set_element(
    root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM
) -> None:
    # [FUNCTION DOC] Sets value in root_dict at the given path, creating intermediate containers: a mapping if the
    # next path element is a string, a list if it is an integer. Lists are padded with None as needed. Mutates root_dict in place; returns None.
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: list[STATE_DICT_ITEM], idx: int) -> None:
        # [FUNCTION DOC] HELPER: Pads the list with None until index idx is valid.
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val = cast(STATE_DICT_ITEM, {} if type(key) is str else [])

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            # pyrefly: ignore [bad-argument-type]
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) is int:
        extend_list(cast(list[STATE_DICT_ITEM], cur_container), key)

    cur_container[key] = value


def get_element(
    root_dict: STATE_DICT_TYPE,
    path: OBJ_PATH,
    default_value: T | None = None,
) -> T | None:
    # [FUNCTION DOC] Returns the value at path inside root_dict, or default_value if any step is missing
    # (wrong container type, missing key, or index out of range).
    # KNOWN ISSUE: the list bound check rejects only indices greater than the length, so an index equal to the length passes and raises IndexError instead of returning the default.
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
    cur_value = cast(CONTAINER_TYPE, root_dict)
    for part in path:
        if type(part) is int:
            if not isinstance(cur_value, list) or len(cur_value) < part:
                return default_value
        elif not isinstance(cur_value, Mapping) or part not in cur_value:
            return default_value

        cur_value = cast(CONTAINER_TYPE, cur_value[part])
    return cast(T | None, cur_value)


def _print_nested(
    value: STATE_DICT_ITEM,
    prefix: str = "",
    print_fun: Callable[[str], None] = print,
) -> None:
    # [FUNCTION DOC] HELPER: Prints a one-line description via print_fun of a value: size for ShardedTensor (recursing into
    # local shards), DTensor (recursing into the local tensor) and plain tensors, or the type for anything else.
    if type(value) is ShardedTensor:
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")
        for shard in value.local_shards():
            _print_nested(
                shard.tensor,
                f"{shard.metadata.shard_offsets} ",
                print_fun=print_fun,
            )
    elif type(value) is (DTensor):
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")
        # TODO: add local offset for _local_tensor in print_nested.
        _print_nested(
            value._local_tensor,
            print_fun=print_fun,
        )
    elif isinstance(value, torch.Tensor):
        print_fun(f"{prefix} Tensor size: {value.size()}")
    else:
        print_fun(f"{prefix} Type: {type(value)}")


def print_tensor(
    path: OBJ_PATH,
    value: STATE_DICT_ITEM,
    print_fun: Callable[[str], None] = print,
) -> None:
    # [FUNCTION DOC] Visitor callback for traverse_state_dict that prints the path and a description of the value
    # (see _print_nested) using print_fun, which defaults to the builtin print.
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be changed by passing a different ``print_fun`` callable.
    """
    _print_nested(value, prefix=str(path), print_fun=print_fun)
