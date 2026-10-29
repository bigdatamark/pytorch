# mypy: allow-untyped-defs
# [MODULE DOC]
# MapDataPipe that applies a function lazily to each item of a source MapDataPipe
# (functional name map), plus the picklable identity function default_fn.
from collections.abc import Callable
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


__all__ = ["MapperMapDataPipe", "default_fn"]


_T_co = TypeVar("_T_co", covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    # [FUNCTION DOC] Identity function: returns its argument unchanged. Used as the default
    # mapper so the DataPipe stays picklable.
    return data


@functional_datapipe("map")
class MapperMapDataPipe(MapDataPipe[_T_co]):
    # [CLASS DOC]
    # MapDataPipe whose item at an index is fn applied to the source's item at that index.
    # State: source datapipe and fn. Length equals the source length; nothing is cached.
    r"""
    Apply the input function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
    ) -> None:
        # [FUNCTION DOC] Stores the source and the function (identity by default). Raises if
        # the function is not picklable (e.g. a lambda), as checked by _check_unpickable_fn.
        super().__init__()
        self.datapipe = datapipe
        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the length of the source datapipe.
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getitem__(self, index) -> _T_co:
        # [FUNCTION DOC] Returns fn applied to the source item at the given index; errors
        # from the source or fn propagate.
        return self.fn(self.datapipe[index])
