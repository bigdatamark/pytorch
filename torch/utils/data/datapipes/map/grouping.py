# mypy: allow-untyped-defs
# [MODULE DOC]
# BatcherMapDataPipe (functional name batch): a MapDataPipe whose items are fixed-size
# batches of consecutive source items. Registers the functional name at import time.
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe


__all__ = ["BatcherMapDataPipe"]


_T = TypeVar("_T")


@functional_datapipe("batch")
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    # [CLASS DOC]
    # MapDataPipe where index i yields the wrapped list of source items
    # i*batch_size .. (i+1)*batch_size-1. State: source, batch_size, drop_last and the
    # wrapper class. A short final batch is returned unless drop_last is set.
    r"""
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """

    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        datapipe: MapDataPipe[_T],
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        # [FUNCTION DOC] Stores the source, batch size, drop_last flag and wrapper class.
        # Raises AssertionError if batch_size is not larger than 0.
        if batch_size <= 0:
            raise AssertionError("Batch size is required to be larger than 0!")
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        # [FUNCTION DOC] Reads the source items for the index's batch range and returns them
        # wrapped. If the source runs out mid-batch, returns the partial batch unless
        # drop_last is set or it is empty, in which case raises IndexError.
        batch: list = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            batch.extend(self.datapipe[i] for i in indices)
            return self.wrapper_class(batch)
        except IndexError as e:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of batches: floor of source length over batch_size when
        # drop_last is set, otherwise the ceiling. Raises TypeError if the source has no
        # length.
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
