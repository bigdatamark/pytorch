# mypy: allow-untyped-defs
# [MODULE DOC] Iterable datapipes that transform items: MapperIterDataPipe (functional
# name map) applies a function to whole items or selected columns, and CollatorIterDataPipe
# (functional name collate) is a Mapper specialized for collation, with a helper that
# collates DataFrame batches by column. Import registers the map and collate functional
# names.
import functools
from collections import namedtuple
from collections.abc import Callable, Iterator, Sized
from typing import Any, TypeVar

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    validate_input_col,
)


__all__ = [
    "CollatorIterDataPipe",
    "MapperIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe[_T_co]):
    # [CLASS DOC] Applies fn to each item of the source datapipe, either to the whole item
    # or to selected input columns with the result written to an output column. State:
    # datapipe, fn, input_col, output_col (normalized to a single value). Length equals
    # the source length when the source is sized.
    r"""
    Applies a function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source Iterable DataPipe
        fn: Function being applied over each item
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``

            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = IterableWrapper(range(10))
        >>> # Invocation via functional form is preferred
        ... map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
        >>> # Use `functools.partial` or explicitly define the function instead
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
    ) -> None:
        # [FUNCTION DOC] Validates and stores the configuration. Logs API usage once.
        # Args: datapipe, fn, input_col (None, index/key, or list/tuple of them),
        # output_col (None, index/key, or single-element list/tuple, normalized to its
        # element). Raises ValueError if output_col is given without input_col or is a
        # sequence longer than one; fn is checked for picklability and input_col is
        # validated against fn's signature by shared utilities.
        torch._C._log_api_usage_once("python.data_pipes.map")
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        # [FUNCTION DOC] HELPER: Applies fn to one item. With no columns it returns fn of
        # the item. Otherwise fn receives the selected input column(s) as arguments, and the
        # result is placed in the output column (appended when -1), or replaces the first
        # input column with the other input columns removed. Tuples are converted to lists
        # for editing and back; lists and dicts are modified in place.
        # KNOWN ISSUE: with output_col set, list/dict items are mutated in place, so the
        # source's own items change.
        if self.input_col is None and self.output_col is None:
            return self.fn(data)

        if self.input_col is None:
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            res = self.fn(*args)
        else:
            res = self.fn(data[self.input_col])

        # Copy tuple to list and run in-place modification because tuple is immutable.
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False

        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[self.input_col] = res
        else:
            if self.output_col == -1:
                data.append(res)
            else:
                data[self.output_col] = res

        # Convert list back to tuple
        return tuple(data) if t_flag else data

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC] Yields the mapped result for each source item, lazily.
        for data in self.datapipe:
            yield self._apply_fn(data)

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the source length if the source is sized; otherwise raises
        # TypeError.
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


def _collate_helper(conversion, item):
    # [FUNCTION DOC] HELPER: Collates a batch holding exactly one DataFrame into a named
    # tuple with one field per column. conversion maps column names to callables; columns
    # missing from the mapping use the TorchArrow default collation. Raises RuntimeError if
    # the batch has more than one DataFrame, a conversion key is not a column, a
    # conversion value is not callable, or the TorchArrow default cannot be imported.
    # TODO(VitalyFedyunin): Verify that item is any sort of batch
    if len(item.items) > 1:
        # TODO(VitalyFedyunin): Compact all batch dataframes into one
        raise RuntimeError("Only supports one DataFrame per batch")
    df = item[0]
    columns_name = df_wrapper.get_columns(df)
    tuple_names: list = []
    tuple_values: list = []

    for name in conversion:
        if name not in columns_name:
            raise RuntimeError("Conversion keys mismatch")

    for name in columns_name:
        if name in conversion:
            if not callable(conversion[name]):
                raise RuntimeError(
                    "Collate (DF)DataPipe requires callable as dict values"
                )
            collation_fn = conversion[name]
        else:
            # TODO(VitalyFedyunin): Add default collation into df_wrapper
            try:
                import torcharrow.pytorch as tap  # type: ignore[import]

                collation_fn = tap.rec.Default()
            except Exception as e:
                raise RuntimeError(
                    "unable to import default collation function from the TorchArrow"
                ) from e

        tuple_names.append(str(name))
        value = collation_fn(df[name])
        tuple_values.append(value)

    # TODO(VitalyFedyunin): We can dynamically extract types from the tuple_values here
    # TODO(VitalyFedyunin): Instead of ignoring mypy error, make sure tuple_names is not empty
    tpl_cls = namedtuple("CollateResult", tuple_names)  # type: ignore[misc]
    tuple = tpl_cls(*tuple_values)
    return tuple


@functional_datapipe("collate")
class CollatorIterDataPipe(MapperIterDataPipe):
    # [CLASS DOC] A Mapper whose function is a collate function: an explicit collate_fn, a
    # callable conversion (default is default_collate), or a mapping of column names to
    # collation callables for DataFrame batches.
    r"""
    Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).

    By default, it uses :func:`torch.utils.data.default_collate`.

    .. note::
        While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
        default behavior and `functools.partial` to specify any additional arguments.

    Args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
            Default function collates to Tensor(s) based on data type.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
        ...         assert end > start, "this example only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        ...     def __len__(self):
        ...         return self.end - self.start
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]
        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        conversion: Callable[..., Any]
        | dict[str | Any, Callable | Any]
        | None = default_collate,
        collate_fn: Callable | None = None,
    ) -> None:
        # [FUNCTION DOC] Chooses the mapping function: collate_fn if given; else conversion
        # if it is callable; else a partial of the DataFrame collate helper bound to the
        # conversion mapping. Then delegates to the Mapper constructor.
        # TODO(VitalyFedyunin): Replace `Callable[..., Any]` with `Callable[[IColumn], Any]`
        # TODO(VitalyFedyunin): Replace with `Dict[Union[str, IColumn], Union[Callable, Enum]]`
        if collate_fn is not None:
            super().__init__(datapipe, fn=collate_fn)
        else:
            if callable(conversion):
                super().__init__(datapipe, fn=conversion)
            else:
                # TODO(VitalyFedyunin): Validate passed dictionary
                collate_fn = functools.partial(_collate_helper, conversion)
                super().__init__(datapipe, fn=collate_fn)
