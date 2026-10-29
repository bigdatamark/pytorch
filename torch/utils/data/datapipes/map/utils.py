# [MODULE DOC]
# SequenceWrapperMapDataPipe: wraps a sequence or mapping as a MapDataPipe, optionally
# deep-copying it at construction.
import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


_T = TypeVar("_T")

__all__ = ["SequenceWrapperMapDataPipe"]


class SequenceWrapperMapDataPipe(MapDataPipe[_T]):
    # [CLASS DOC]
    # MapDataPipe backed by a sequence or mapping held in the sequence attribute; item
    # access and length delegate to it.
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> dp = SequenceWrapper({"a": 100, "b": 200, "c": 300, "d": 400})
        >>> dp["a"]
        100
    """

    sequence: Sequence[_T] | Mapping[Any, _T]

    def __init__(
        self, sequence: Sequence[_T] | Mapping[Any, _T], deepcopy: bool = True
    ) -> None:
        # [FUNCTION DOC] Stores the sequence, as a deep copy when deepcopy is set. If copying
        # raises TypeError, emits a warning and keeps the original object. Does not call the
        # parent initializer.
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)
            except TypeError:
                warnings.warn(
                    "The input sequence can not be deepcopied, "
                    "please be aware of in-place modification would affect source data",
                    stacklevel=2,
                )
                self.sequence = sequence
        else:
            self.sequence = sequence

    def __getitem__(self, index: int) -> _T:
        # [FUNCTION DOC] Returns the wrapped sequence's item for the index or key; errors
        # from the underlying object propagate.
        return self.sequence[index]

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the length of the wrapped sequence or mapping.
        return len(self.sequence)
