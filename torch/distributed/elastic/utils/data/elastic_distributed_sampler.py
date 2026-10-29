#!/usr/bin/env python3

# [MODULE DOC] Defines ElasticDistributedSampler, a DistributedSampler variant that begins sampling at a given dataset offset so training can resume mid-epoch after elastic restarts. No import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterator, Sized
from typing import cast, TypeVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


T = TypeVar("T")

__all__ = ["ElasticDistributedSampler"]


class ElasticDistributedSampler(DistributedSampler[T]):
    # [CLASS DOC] DistributedSampler subclass that samples only dataset indices from start_index onward, split across replicas.
    # State: start_index, num_samples (per-replica count, rounded up), total_size (num_samples times replicas), plus inherited epoch/rank/num_replicas. Assumes a constant-size dataset.
    """
    Sampler that restricts data loading to a subset of
    the dataset for elastic training.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        start_index (optional):  Which index of the dataset to start sampling from
    """

    def __init__(
        self,
        dataset: Dataset[T],
        num_replicas: int | None = None,
        rank: int | None = None,
        start_index: int = 0,
    ):
        # [FUNCTION DOC] Initialize the base sampler, then compute per-replica num_samples as the ceiling of remaining items (dataset length minus start_index) over num_replicas and total_size.
        # Raises TypeError if the dataset is not Sized and ValueError if start_index is not less than the dataset length.
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be an instance of collections.abc.Sized")

        # Cast to Sized for mypy

        sized_dataset = cast(Sized, dataset)

        if start_index >= len(sized_dataset):
            raise ValueError(
                f"Start index {start_index} should be less than dataset size {len(sized_dataset)}"
            )

        self.start_index = start_index
        sized_dataset = cast(Sized, self.dataset)
        self.num_samples = math.ceil(
            float(len(sized_dataset) - self.start_index) / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T]:
        # [FUNCTION DOC] Return an iterator of indices for this rank. Builds a permutation of the indices from start_index on, seeded deterministically by the current epoch, pads by repeating from the beginning to reach total_size, then takes every num_replicas-th element starting at rank.
        # Raises AssertionError if internal length checks fail. Does not honor the base class shuffle flag (always shuffles).
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        sized_dataset = cast(Sized, self.dataset)
        indices = (
            torch.randperm(len(sized_dataset) - self.start_index, generator=g)
            .add(self.start_index)
            .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise AssertionError

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError

        return iter(indices)

    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of samples for this rank (num_samples).
        return self.num_samples
