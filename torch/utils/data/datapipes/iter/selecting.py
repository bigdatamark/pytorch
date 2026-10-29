# mypy: allow-untyped-defs
# [MODULE DOC]
# Iterable DataPipe that selects elements: FilterIterDataPipe (functional name filter) keeps
# only elements for which a user predicate returns true. Registered at import time through
# the functional_datapipe decorator.
from collections.abc import Callable, Iterator
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    StreamWrapper,
    validate_input_col,
)


__all__ = ["FilterIterDataPipe"]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(IterDataPipe[_T_co]):
    # [CLASS DOC]
    # Iterable DataPipe that yields only source elements accepted by filter_fn. State: the
    # source datapipe, the predicate, and input_col which selects the part of each element
    # (index, key, or several of them) passed to the predicate. Also supports DataFrame
    # elements via a mask column. Rejected elements have their streams closed.
    r"""
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """

    datapipe: IterDataPipe[_T_co]
    filter_fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        filter_fn: Callable,
        input_col=None,
    ) -> None:
        # [FUNCTION DOC] Stores the source and predicate and the input column selector.
        # Raises if the predicate is not picklable (lambda or nested function) or if
        # validate_input_col finds the column selector inconsistent with the predicate's
        # arguments.
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(filter_fn)
        self.filter_fn = filter_fn  # type: ignore[assignment]

        self.input_col = input_col
        validate_input_col(filter_fn, input_col)

    def _apply_filter_fn(self, data) -> bool:
        # [FUNCTION DOC] HELPER: Calls the predicate on the whole element when input_col is
        # unset, on the several selected items as separate arguments when input_col is a
        # list or tuple, or on the single selected item otherwise. Returns its result.
        if self.input_col is None:
            return self.filter_fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.filter_fn(*args)
        else:
            return self.filter_fn(data[self.input_col])

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC] Yields each source element (or the filtered DataFrame rows) that
        # passes the predicate; for rejected elements closes any stream objects they hold.
        for data in self.datapipe:
            condition, filtered = self._returnIfTrue(data)
            if condition:
                yield filtered
            else:
                StreamWrapper.close_streams(data)

    def _returnIfTrue(self, data: _T) -> tuple[bool, _T]:
        # [FUNCTION DOC] HELPER: Evaluates the predicate and returns a pair of (accepted,
        # data). For a DataFrame mask column, returns the concatenation of the selected rows
        # (or rejection with no data if none are selected). Raises ValueError when a
        # non-DataFrame predicate result is not a bool.
        condition = self._apply_filter_fn(data)

        if df_wrapper.is_column(condition):
            # We are operating on DataFrames filter here
            result = []
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    result.append(df_wrapper.get_item(data, idx))
            if result:
                return True, df_wrapper.concat(result)
            else:
                return False, None  # type: ignore[return-value]

        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )

        return condition, data
