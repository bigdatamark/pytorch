# mypy: allow-untyped-defs
# [MODULE DOC]
# Sharding support for iterable DataPipes: the SHARDING_PRIORITIES enum, a base class
# declaring apply_sharding, and ShardingFilterIterDataPipe (functional name
# sharding_filter) which keeps every n-th element for the current worker. Registers the
# functional name at import time.
from collections.abc import Sized
from enum import IntEnum
from typing import NoReturn

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = [
    "SHARDING_PRIORITIES",
    "ShardingFilterIterDataPipe",
]


class SHARDING_PRIORITIES(IntEnum):
    # [CLASS DOC]
    # Integer enum naming sharding levels (DEFAULT, DISTRIBUTED, MULTIPROCESSING). The
    # numeric value gives the order in which groups are combined; lower values are the
    # outer level.
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


class _ShardingIterDataPipe(IterDataPipe):
    # [CLASS DOC] HELPER: Abstract base for sharding DataPipes; declares the
    # apply_sharding interface that graph utilities look for.
    def apply_sharding(
        self,
        num_of_instances: int,
        instance_id: int,
        sharding_group: SHARDING_PRIORITIES,
    ) -> NoReturn:
        # [FUNCTION DOC] Abstract interface: subclasses apply sharding with the given number
        # of instances, instance id and group. This base version always raises
        # NotImplementedError.
        raise NotImplementedError


@functional_datapipe("sharding_filter")
class ShardingFilterIterDataPipe(_ShardingIterDataPipe):
    # [CLASS DOC]
    # Iterable DataPipe that yields only the elements whose position modulo
    # num_of_instances equals instance_id. State: the source, a mapping from sharding group
    # to (num_of_instances, instance_id), an optional group filter, and the combined
    # num_of_instances/instance_id derived from the groups (default 1 and 0, no sharding).
    # Invariant: DEFAULT and non-DEFAULT groups cannot be mixed.
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).

    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(
        self, source_datapipe: IterDataPipe, sharding_group_filter=None
    ) -> None:
        # [FUNCTION DOC] Stores the source and optional group filter, starts with no
        # groups (a single instance, id 0), and computes the combined values.
        super().__init__()
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: dict[int, tuple[int, int]] = {}
        self.num_of_instances = 1
        self.instance_id = 0
        self._update_num_of_instances()

    def apply_sharding(
        self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT
    ):
        # [FUNCTION DOC] Records (num_of_instances, instance_id) for the given sharding group
        # (replacing any earlier entry) and recomputes the combined sharding. Raises
        # ValueError if instance_id is not smaller than num_of_instances, and RuntimeError if
        # this would mix the DEFAULT group with other groups.
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    def _update_num_of_instances(self) -> None:
        # [FUNCTION DOC] HELPER: Combines the groups (only the filtered one if a group filter
        # is set) into one effective num_of_instances and instance_id, treating the
        # highest-priority-value group as the innermost level (a mixed-radix combination).
        sorted_sharding_groups = [
            self.groups[key]
            for key in sorted(self.groups.keys())
            if self.sharding_group_filter is None or key == self.sharding_group_filter
        ]

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances

    def __iter__(self):
        # [FUNCTION DOC] Yields only source elements whose zero-based position modulo
        # num_of_instances equals instance_id.
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of elements in this shard: source length divided by the
        # instance count, plus one if instance_id is below the remainder. Raises TypeError
        # if the source has no length.
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // self.num_of_instances + (
                1
                if (
                    self.instance_id < len(self.source_datapipe) % self.num_of_instances
                )
                else 0
            )
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
