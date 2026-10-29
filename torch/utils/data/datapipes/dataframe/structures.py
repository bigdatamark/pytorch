# [MODULE DOC] Defines DataChunkDF, a DataChunk whose items are DataFrame containers that
# is iterated and measured row by row through the dataframe_wrapper backend helpers.
# No import-time side effects beyond class definition.
from collections.abc import Iterator
from typing import Any

from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DataChunk


__all__ = ["DataChunkDF"]


class DataChunkDF(DataChunk):
    # [CLASS DOC] A DataChunk (list-like batch) whose elements are DataFrames. Iteration
    # yields individual rows of every contained DataFrame in order; length is the total
    # row count across all DataFrames. The underlying frames stay in self.items.
    """DataChunkDF iterating over individual items inside of DataFrame containers, to access DataFrames user `raw_iterator`."""

    def __iter__(self) -> Iterator[Any]:
        # [FUNCTION DOC] Yields, in order, each row of each DataFrame in the chunk, using
        # the active dataframe backend's iterate helper. Takes no arguments.
        for df in self.items:
            yield from df_wrapper.iterate(df)

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the total number of rows summed over all contained
        # DataFrames, as reported by the backend's get_len helper.
        total_len = 0
        for df in self.items:
            total_len += df_wrapper.get_len(df)
        return total_len
