# [MODULE DOC] Defines FileOpenerIterDataPipe (functional name open_files), which turns a
# datapipe of pathnames into (pathname, open file stream) tuples using the shared
# get_file_binaries_from_pathnames utility. Registration under open_files occurs at import.
from collections.abc import Iterable, Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames


__all__ = [
    "FileOpenerIterDataPipe",
]


@functional_datapipe("open_files")
class FileOpenerIterDataPipe(IterDataPipe[tuple[str, IOBase]]):
    # [CLASS DOC] Iterable datapipe that opens each incoming pathname and yields the pair
    # of pathname and file stream. State: source datapipe, open mode, encoding, nominal
    # length (-1 means unknown). Invariant: binary mode never has an encoding.
    r"""
    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import (
        ...     FileLister,
        ...     FileOpener,
        ...     StreamReader,
        ... )
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(".txt"))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "r",
        encoding: str | None = None,
        length: int = -1,
    ) -> None:
        # [FUNCTION DOC] Stores the pathname source, mode, encoding and nominal length.
        # Raises ValueError if mode is not one of b, t, rb, rt, r, or if a mode containing
        # b is combined with an encoding. Opens no files at construction.
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        self.encoding: str | None = encoding

        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken

        if "b" in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")

        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self) -> Iterator[tuple[str, IOBase]]:
        # [FUNCTION DOC] Lazily opens each pathname with the stored mode and encoding and
        # yields (pathname, stream). Side effect: opens files; closing is left to callers/GC.
        yield from get_file_binaries_from_pathnames(
            self.datapipe, self.mode, self.encoding
        )

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the nominal length given at construction; raises
        # TypeError when it is -1 (unknown).
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
