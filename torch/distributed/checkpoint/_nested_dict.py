# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] Flatten and unflatten of nested state dicts. flatten_state_dict turns nested dicts/lists into a
# single-level dict with dot-joined keys plus a mapping from new keys to original object paths;
# unflatten_state_dict restores the nesting from that mapping. Defines the FLATTEN_MAPPING alias.
# No import-time side effects.

from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

from . import _version
from ._traverse import (
    OBJ_PATH,
    set_element,
    STATE_DICT_ITEM,
    traverse_state_dict,
    traverse_state_dict_v_2_3,
)


"""
TODO:
Need to add ability to handle tuple, OrderedDict, NamedTuple.
Update mappings from dict to a class.
Change set_element to recreate the right type for tuple, OrderedDict, and NamedTuple.
"""


FLATTEN_MAPPING = dict[str, OBJ_PATH]


# TODO: Update Docstring for nested_dict.py
def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    # [FUNCTION DOC] Flattens state_dict via a traversal that calls a collector for each leaf. Returns the flat dict
    # (keys are the path elements joined by dots) and a mapping from flat key to original path.
    # Uses the legacy 2.3 traversal when the derived version in the _version module is "2_3",
    # otherwise the current traversal. Raises ValueError on duplicate flattened keys.
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # [FUNCTION DOC] HELPER: Leaf callback: records the value under the dotted key and the path in the mapping; raises ValueError if the key already exists.
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    # We started to flatten dictionary since v2.4. But in order to not break
    # the checkpoints that were saved before v2.4, we need to keep the old
    # traversal so that we can reconstruct those checkpoints.
    use_v_2_3 = (
        _version._derived_version is not None and _version._derived_version == "2_3"
    )
    if use_v_2_3:
        traverse_state_dict_v_2_3(state_dict, flat_copy)
    else:
        traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    # [FUNCTION DOC] Rebuilds a nested dict from the flat state_dict by placing each value at the path given by mapping for its key. Raises KeyError if a key is not in the mapping.
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        set_element(nested, mapping[key], value)
    return nested
