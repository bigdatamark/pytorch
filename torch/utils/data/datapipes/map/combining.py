# mypy: allow-untyped-defs
# [MODULE DOC]
# Map DataPipes that combine several Map DataPipes: ConcaterMapDataPipe (concat, indexes
# run through the sources one after another) and ZipperMapDataPipe (zip, same index from
# every source as a tuple). Registers functional names at import time.
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe


__all__ = ["ConcaterMapDataPipe", "ZipperMapDataPipe"]

_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
class ConcaterMapDataPipe(MapDataPipe):
    # [CLASS DOC]
    # MapDataPipe presenting several sized Map DataPipes as one sequence; length is the sum
    # of source lengths. State: tuple of source datapipes.
    r"""
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """

    datapipes: tuple[MapDataPipe]

    def __init__(self, *datapipes: MapDataPipe) -> None:
        # [FUNCTION DOC] Stores the source datapipes. Raises ValueError if none are given and
        # TypeError if any is not a MapDataPipe or has no length.
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # pyrefly: ignore [unsafe-overlap]
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __getitem__(self, index) -> _T_co:  # type: ignore[type-var]
        # [FUNCTION DOC] Walks the sources in order, subtracting each source's length, and
        # returns the item from the first source whose range contains the index. Raises
        # IndexError if the index is beyond all sources.
        offset = 0
        for dp in self.datapipes:
            # pyrefly: ignore [bad-argument-type]
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                # pyrefly: ignore [bad-argument-type]
                offset += len(dp)
        raise IndexError(f"Index {index} is out of range.")

    def __len__(self) -> int:
        # [FUNCTION DOC] Sum of the lengths of all source datapipes.
        # pyrefly: ignore [bad-argument-type]
        return sum(len(dp) for dp in self.datapipes)


@functional_datapipe("zip")
class ZipperMapDataPipe(MapDataPipe[tuple[_T_co, ...]]):
    # [CLASS DOC]
    # MapDataPipe whose item at an index is the tuple of the same-index items of all source
    # datapipes; length is the shortest source length. State: tuple of sources.
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MapDataPipe is out of bounds as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """

    datapipes: tuple[MapDataPipe[_T_co], ...]

    def __init__(self, *datapipes: MapDataPipe[_T_co]) -> None:
        # [FUNCTION DOC] Stores the source datapipes. Raises ValueError if none are given and
        # TypeError if any is not a MapDataPipe or has no length.
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        # pyrefly: ignore [unsafe-overlap]
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes

    def __getitem__(self, index) -> tuple[_T_co, ...]:
        # [FUNCTION DOC] Collects the item at the index from every source and returns them as
        # a tuple. Raises IndexError (chained to the original) if any source lacks the index.
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError as e:
                raise IndexError(
                    f"Index {index} is out of range for one of the input MapDataPipes {dp}."
                ) from e
        return tuple(res)

    def __len__(self) -> int:
        # [FUNCTION DOC] Smallest length among the source datapipes.
        # pyrefly: ignore [bad-argument-type]
        return min(len(dp) for dp in self.datapipes)
