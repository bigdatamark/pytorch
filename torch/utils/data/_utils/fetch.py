# mypy: allow-untyped-defs
# [MODULE DOC] Fetcher classes used by the DataLoader iterators to pull samples from a dataset
# and collate them. Provides a base fetcher plus iterable-style and map-style variants shared by
# single- and multi-process loading. No import-time side effects.
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

from typing import NoReturn


class _BaseDatasetFetcher:
    # [CLASS DOC] HELPER: Abstract base holding the dataset, the auto_collation flag, the
    # collate_fn and the drop_last flag. Subclasses implement fetch.
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        # [FUNCTION DOC] HELPER: Stores the dataset, auto_collation flag, collate_fn and
        # drop_last flag as attributes. No other side effects.
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index) -> NoReturn:
        # [FUNCTION DOC] HELPER: Abstract; always raises NotImplementedError.
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    # [CLASS DOC] HELPER: Fetcher for iterable-style datasets. Holds a single iterator created
    # from the dataset at construction and an ended flag set once the iterator is exhausted.
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base fields, creates the dataset iterator
        # and sets the ended flag to false.
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        # [FUNCTION DOC] HELPER: Returns the collated next sample or batch. With auto_collation
        # the index argument only gives the batch size: that many items are drawn from the
        # iterator, stopping early (and marking ended) on exhaustion. Raises StopIteration if
        # already ended, if no items were obtained, or if drop_last is set and the batch is
        # short. Without auto_collation, one item is drawn. Advances the shared iterator.
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    # [CLASS DOC] HELPER: Fetcher for map-style datasets, which are indexed by key.
    def fetch(self, possibly_batched_index):
        # [FUNCTION DOC] HELPER: Returns the collated data for the given index. With
        # auto_collation the argument is a list of indices: uses the dataset bulk getter if it
        # exists and is truthy, otherwise looks up each index. Without it, looks up the single
        # index. Errors from the dataset propagate.
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
