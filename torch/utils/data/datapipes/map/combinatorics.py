# mypy: allow-untyped-defs
# [MODULE DOC]
# ShufflerIterDataPipe: an iterable DataPipe that yields the items of a MapDataPipe in
# random order by shuffling its indices. Import-time side effect: registers the functional
# name shuffle on MapDataPipe (not via the decorator).
import random
from collections.abc import Iterator
from typing import TypeVar

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["ShufflerIterDataPipe"]


_T_co = TypeVar("_T_co", covariant=True)


# @functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    # [CLASS DOC]
    # Iterable DataPipe over a MapDataPipe that visits the chosen indices in shuffled order.
    # State: source, index list, enabled flag, optional pending seed, a private random
    # generator and the remaining shuffled indices (consumed during iteration). reset
    # reshuffles; shuffling can be disabled to iterate in the original index order.
    r"""
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even though this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """

    datapipe: MapDataPipe[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random

    def __init__(
        self,
        datapipe: MapDataPipe[_T_co],
        *,
        indices: list | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the source and the index list (all positions 0..len-1 of the
        # source when indices is None), enables shuffling, clears the seed and creates a
        # random generator. The not-yet-shuffled index list is used until reset runs.
        super().__init__()
        self.datapipe = datapipe
        # pyrefly: ignore [bad-argument-type]
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self._enabled = True
        self._seed = None
        self._rng = random.Random()
        self._shuffled_indices: list = self.indices

    def set_shuffle(self, shuffle=True):
        # [FUNCTION DOC] Enables or disables shuffling for later iterations; returns self.
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        # [FUNCTION DOC] Sets the seed used by the next reset (it is cleared once used);
        # returns self.
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC] When disabled, yields source items in index-list order. Otherwise
        # repeatedly removes an index from the end of the shuffled list and yields the source
        # item for it, so the list is consumed (a fresh shuffle comes from reset).
        if not self._enabled:
            for idx in self.indices:
                yield self.datapipe[idx]
        else:
            while self._shuffled_indices:
                idx = self._shuffled_indices.pop()
                yield self.datapipe[idx]

    def reset(self) -> None:
        # [FUNCTION DOC] Prepares a new epoch: when enabled and no seed is set, draws a
        # random seed from torch's generator (advancing global torch RNG state); seeds the
        # private generator, clears the stored seed, and builds a freshly shuffled copy of
        # the index list.
        if self._enabled and self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._rng.seed(self._seed)
        self._seed = None
        self._shuffled_indices = self._rng.sample(self.indices, len(self.indices))

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the length of the source datapipe.
        # KNOWN ISSUE: ignores a custom indices list, so the length can differ from the
        # number of items actually yielded.
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getstate__(self):
        # [FUNCTION DOC] Pickling support: returns a tuple with the source, indices, flags,
        # seed, the generator state, remaining shuffled indices and iterator bookkeeping,
        # passed through the class-level getstate hook if set.
        state = (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC] Restores the fields from the tuple made by __getstate__ and
        # rebuilds the private random generator from the saved generator state.
        (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            rng_state,
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


MapDataPipe.register_datapipe_as_function("shuffle", ShufflerIterDataPipe)
