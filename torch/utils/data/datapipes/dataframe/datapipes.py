# mypy: allow-untyped-defs
# [MODULE DOC] Prototype iterable datapipes operating on streams of DataFrames (row split,
# concat, shuffle, filter, aggregation of plain items into DataFrames, flattening to row
# tuples). All use the dataframe_wrapper backend. Each is registered as a functional
# datapipe with a leading-underscore name at import time; most enable DataFrame API tracing.
import random
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


__all__ = [
    "ConcatDataFramesPipe",
    "DataFramesAsTuplesPipe",
    "ExampleAggregateAsDataFrames",
    "FilterDataFramesPipe",
    "PerRowDataFramesPipe",
    "ShuffleDataFramesPipe",
]


@functional_datapipe("_dataframes_as_tuples")
class DataFramesAsTuplesPipe(IterDataPipe):
    # [CLASS DOC] Flattens a datapipe of DataFrames into a stream of individual row tuples.
    # State: source_datapipe.
    def __init__(self, source_datapipe) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe of DataFrames.
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self):
        # [FUNCTION DOC] Yields every row (as a tuple) of every DataFrame from upstream,
        # in order, via the backend iterate helper.
        for df in self.source_datapipe:
            # for record in df.to_records(index=False):
            yield from df_wrapper.iterate(df)


@functional_datapipe("_dataframes_per_row", enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):
    # [CLASS DOC] Splits each incoming DataFrame into one-row DataFrames. State:
    # source_datapipe.
    def __init__(self, source_datapipe) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe; does not call the base constructor.
        self.source_datapipe = source_datapipe

    def __iter__(self):
        # [FUNCTION DOC] Yields a one-row slice for each row position of each upstream
        # DataFrame. Slices with raw pandas-style indexing, bypassing the wrapper backend.
        for df in self.source_datapipe:
            # TODO(VitalyFedyunin): Replacing with TorchArrow only API, as we are dropping pandas as followup
            for i in range(len(df)):
                yield df[i : i + 1]


@functional_datapipe("_dataframes_concat", enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):
    # [CLASS DOC] Groups consecutive upstream DataFrames and concatenates each group into
    # one DataFrame. State: source_datapipe, n_batch (group size).
    def __init__(self, source_datapipe, batch=3) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe and the group size batch (default 3).
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        # [FUNCTION DOC] Buffers upstream DataFrames and yields their concatenation each
        # time n_batch have accumulated; a final shorter group is yielded if non-empty.
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_shuffle", enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    # [CLASS DOC] Shuffles rows across all upstream DataFrames and re-emits them in
    # DataFrames of the same size as the first input. State: source_datapipe.
    def __init__(self, source_datapipe) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe; does not call the base constructor.
        self.source_datapipe = source_datapipe

    def __iter__(self):
        # [FUNCTION DOC] Reads the whole upstream into memory as one-row frames, shuffles
        # them with the global random module (not seeded by the datapipe), then yields
        # concatenated groups whose size equals the first input's row count; the last
        # group may be smaller.
        size = None
        all_buffer: list[Any] = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            all_buffer.extend(
                df_wrapper.get_item(df, i) for i in range(df_wrapper.get_len(df))
            )
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_filter", enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):
    # [CLASS DOC] Keeps only rows for which a predicate is truthy and re-emits them in
    # DataFrames of the first input's size. State: source_datapipe, filter_fn.
    def __init__(self, source_datapipe, filter_fn) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe and the row predicate filter_fn.
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        # [FUNCTION DOC] Reads the whole upstream into memory, evaluating filter_fn on
        # each row, then yields concatenated groups of passing rows sized like the first
        # input; the last group may be smaller. Uses pandas-specific row access (the
        # index and positional row lookup) directly instead of the wrapper backend.
        size = None
        all_buffer = []
        filter_res = []
        # pyrefly: ignore [bad-assignment]
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i : i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))

        buffer = []
        for df, res in zip(all_buffer, filter_res, strict=True):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_to_dataframes_pipe", enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    # [CLASS DOC] Aggregates upstream items (each treated as a row) into DataFrames of a
    # fixed number of rows. State: source_datapipe, columns, dataframe_size.
    def __init__(self, source_datapipe, dataframe_size=10, columns=None) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe, the optional column names, and the
        # rows-per-DataFrame size (default 10).
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        # [FUNCTION DOC] HELPER: Converts item to a list if it is iterable, otherwise
        # wraps it in a one-element list. Any exception from list conversion is swallowed.
        # KNOWN ISSUE: strings and dicts are iterable, so they are split into characters
        # or keys rather than treated as single values.
        try:
            return list(item)
        except (
            Exception
        ):  # TODO(VitalyFedyunin): Replace with better iterable exception
            return [item]

    def __iter__(self):
        # [FUNCTION DOC] Collects row lists from upstream items and yields a new DataFrame
        # (through the backend create_dataframe) whenever dataframe_size rows have
        # accumulated; any remainder is yielded as a final smaller DataFrame.
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
