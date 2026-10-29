# [MODULE DOC]
# Iterable DataPipe (read_from_stream) that reads byte chunks from (label, stream) pairs and
# yields (label, bytes). Registers the functional name at import time.
from collections.abc import Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = ["StreamReaderIterDataPipe"]


@functional_datapipe("read_from_stream")
class StreamReaderIterDataPipe(IterDataPipe[tuple[str, bytes]]):
    # [CLASS DOC]
    # Iterable DataPipe that reads each source stream in pieces of chunk bytes (or all at
    # once when chunk is None) and yields (label, data) pairs. State: source and chunk size.
    # Each stream is closed after it is exhausted.
    r"""
    Given IO streams and their label names, yield bytes with label name as tuple.

    (functional name: ``read_from_stream``).

    Args:
        datapipe: Iterable DataPipe provides label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read until the EOF.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """

    def __init__(
        self, datapipe: IterDataPipe[tuple[str, IOBase]], chunk: int | None = None
    ) -> None:
        # [FUNCTION DOC] Stores the source datapipe and the per-read chunk size (None means
        # read to end of stream). Does not call the parent initializer.
        self.datapipe = datapipe
        self.chunk = chunk

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        # [FUNCTION DOC] For each (label, stream), repeatedly reads chunk bytes and yields
        # (label, data) until a read returns empty data, then closes the stream (side
        # effect: the source streams are consumed and closed).
        for furl, stream in self.datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    stream.close()
                    break
                yield (furl, d)
