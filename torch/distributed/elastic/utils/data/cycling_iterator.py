#!/usr/bin/env python3

# [MODULE DOC] Defines CyclingIterator, an iterator wrapper that replays a generator-produced iterator for a fixed number of epochs. No import-time side effects.
from collections.abc import Callable, Iterator
from typing import TypeVar
from typing_extensions import Self


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_T = TypeVar("_T")

__all__ = ["CyclingIterator"]


class CyclingIterator(Iterator[_T]):
    # [CLASS DOC] Iterator that chains n epochs of the iterator produced by generator_fn(epoch).
    # State: _n (number of cycles), _epoch (current cycle index, starting at start_epoch), _generator_fn, _iter (current underlying iterator). Exhausting one underlying iterator creates the next until n cycles are consumed.
    """
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    """

    def __init__(
        self,
        n: int,
        generator_fn: Callable[[int], Iterator[_T]],
        start_epoch: int = 0,
    ):
        # [FUNCTION DOC] Store n, the generator function and start epoch, and eagerly create the first underlying iterator by calling the generator with the start epoch.
        self._n = n
        self._epoch = start_epoch
        self._generator_fn = generator_fn
        self._iter = generator_fn(self._epoch)

    def __iter__(self) -> Self:
        # [FUNCTION DOC] Return self.
        return self

    def __next__(self) -> _T:
        # [FUNCTION DOC] Return the next item of the current underlying iterator. At its end, if more cycles remain, advance the epoch, build a new iterator via the generator and continue; otherwise re-raise StopIteration.
        # # KNOWN ISSUE: the comparison uses the epoch counter against n - 1, so a nonzero start_epoch still counts from zero to n, yielding fewer than n cycles.
        try:
            return next(self._iter)
        except StopIteration as eod:  # eod == end of data
            if self._epoch < self._n - 1:
                self._epoch += 1
                self._iter = self._generator_fn(self._epoch)
                return self.__next__()
            else:
                raise eod
