# mypy: allow-untyped-defs
# [MODULE DOC]
# Helpers that configure a whole DataPipe graph: apply sharding to sharding DataPipes,
# toggle shuffling, and seed random DataPipes. Graphs come from traverse_dps. Exports
# apply_sharding, apply_shuffle_settings, apply_shuffle_seed (deprecated),
# apply_random_seed and get_all_graph_pipes.
import inspect
import warnings
from typing import Any
from typing_extensions import deprecated

import torch
from torch.utils.data.datapipes.iter.sharding import (
    _ShardingIterDataPipe,
    SHARDING_PRIORITIES,
)
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps


__all__ = [
    "apply_random_seed",
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]


def get_all_graph_pipes(graph: DataPipeGraph) -> list[DataPipe]:
    # [FUNCTION DOC] Returns a flat list of every DataPipe in the graph, depth first, each
    # distinct DataPipe (by id) listed once.
    return _get_all_graph_pipes_helper(graph, set())


def _get_all_graph_pipes_helper(
    graph: DataPipeGraph, id_cache: set[int]
) -> list[DataPipe]:
    # [FUNCTION DOC] HELPER: Recursive part of get_all_graph_pipes; skips ids already in the
    # shared cache, records new ids, and returns the DataPipe followed by its subgraph's.
    results: list[DataPipe] = []
    for dp_id, (datapipe, sub_graph) in graph.items():
        if dp_id in id_cache:
            continue
        id_cache.add(dp_id)
        results.append(datapipe)
        results.extend(_get_all_graph_pipes_helper(sub_graph, id_cache))
    return results


def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    # [FUNCTION DOC] HELPER: True if the DataPipe is a sharding base-class instance or has a
    # bound method named apply_sharding.
    return isinstance(datapipe, _ShardingIterDataPipe) or (
        hasattr(datapipe, "apply_sharding")
        and inspect.ismethod(datapipe.apply_sharding)
    )


def apply_sharding(
    datapipe: DataPipe,
    num_of_instances: int,
    instance_id: int,
    sharding_group=SHARDING_PRIORITIES.DEFAULT,
) -> DataPipe:
    # [FUNCTION DOC] Walks the graph and calls apply_sharding on each sharding DataPipe with
    # the instance count, instance id and group (the group is passed only if the method
    # accepts three parameters, for backward compatibility). Mutates the DataPipes and
    # returns the same input DataPipe. Raises RuntimeError if two sharding DataPipes lie on
    # one branch.
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.

    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None) -> None:
        # [FUNCTION DOC] HELPER: Recursive walk carrying the sharding DataPipe applied higher in
        # the branch; applies sharding to each sharding DataPipe found and raises RuntimeError
        # if one was already applied on the path.
        for dp, sub_graph in graph.values():
            applied = None
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError(
                        "Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                        f"Sharding already applied to {prev_applied} while trying to apply to {dp}"
                    )
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(
                        num_of_instances, instance_id, sharding_group=sharding_group
                    )
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe


def _is_shuffle_datapipe(datapipe: DataPipe) -> bool:
    # [FUNCTION DOC] HELPER: True if the DataPipe has bound methods set_shuffle and set_seed.
    return (
        hasattr(datapipe, "set_shuffle")
        and hasattr(datapipe, "set_seed")
        and inspect.ismethod(datapipe.set_shuffle)
        and inspect.ismethod(datapipe.set_seed)
    )


def apply_shuffle_settings(datapipe: DataPipe, shuffle: bool | None = None) -> DataPipe:
    # [FUNCTION DOC] With shuffle None, returns the input unchanged. Otherwise calls
    # set_shuffle on every shuffle-capable DataPipe in the graph. If shuffling is requested
    # and none exist, warns and appends a shuffle step to the end of the pipeline, returning
    # that new DataPipe. Mutates the DataPipes.
    r"""
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    if shuffle is None:
        return datapipe

    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task.",
            stacklevel=2,
        )
        datapipe = datapipe.shuffle()
        shufflers = [
            datapipe,
        ]

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe


@deprecated(
    "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases. "
    "Please use `apply_random_seed` instead.",
    category=FutureWarning,
)
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    # [FUNCTION DOC] Deprecated alias of apply_random_seed (emits a FutureWarning when called).
    return apply_random_seed(datapipe, rng)


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    # [FUNCTION DOC] HELPER: True if the DataPipe has a bound method set_seed.
    return hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed)


def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    # [FUNCTION DOC] Finds each distinct DataPipe in the graph that has set_seed and calls it
    # with a fresh 64-bit integer drawn from the given generator (advancing it; a None
    # generator uses the global torch RNG). Returns the same input DataPipe.
    r"""
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    # Using a set to track id of DataPipe to prevent setting randomness per DataPipe more than once.
    # And, `id` is used in case of unhashable DataPipe
    cache = set()
    random_datapipes = []
    for pipe in all_pipes:
        if id(pipe) in cache:
            continue
        if _is_random_datapipe(pipe):
            random_datapipes.append(pipe)
            cache.add(id(pipe))

    for pipe in random_datapipes:
        random_seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=rng).item()
        )
        pipe.set_seed(random_seed)

    return datapipe
