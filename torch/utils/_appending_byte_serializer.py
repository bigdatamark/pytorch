# [MODULE DOC] Compact binary serialization of a growing list of items: a byte writer and
# reader with a leading CRC32 checksum, big-endian 64-bit ints and base64 strings, and a
# generic AppendingByteSerializer that appends items using a caller-provided function.
import base64
import zlib
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar


T = TypeVar("T")

_ENCODING_VERSION: int = 1

__all__ = ["AppendingByteSerializer"]


#######################################
# Helper classes
#######################################

CHECKSUM_DIGEST_SIZE = 4


class BytesWriter:
    # [CLASS DOC] Append-only byte buffer whose first four bytes are reserved for a CRC32
    # checksum of the rest, filled in when converting to bytes.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the buffer holding only the zeroed checksum placeholder.
        # Reserve CHECKSUM_DIGEST_SIZE bytes for checksum
        self._data = bytearray(CHECKSUM_DIGEST_SIZE)

    def write_uint64(self, i: int) -> None:
        # [FUNCTION DOC] Appends i as 8 big-endian bytes. Raises OverflowError if i is
        # negative or does not fit.
        self._data.extend(i.to_bytes(8, byteorder="big", signed=False))

    def write_str(self, s: str) -> None:
        # [FUNCTION DOC] Appends the string as length-prefixed bytes of its base64-encoded
        # UTF-8 form.
        payload = base64.b64encode(s.encode("utf-8"))
        self.write_bytes(payload)

    def write_bytes(self, b: bytes) -> None:
        # [FUNCTION DOC] Appends a 64-bit length followed by the raw bytes.
        self.write_uint64(len(b))
        self._data.extend(b)

    def to_bytes(self) -> bytes:
        # [FUNCTION DOC] Computes the CRC32 of everything after the checksum slot, stores it
        # in the slot (mutating the buffer) and returns an immutable copy of the buffer.
        # Raises AssertionError if the digest size is unexpected.
        digest = zlib.crc32(self._data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Computed checksum digest has unexpected size")
        self._data[0:CHECKSUM_DIGEST_SIZE] = digest
        return bytes(self._data)


class BytesReader:
    # [CLASS DOC] Sequential reader over bytes produced by the writer; keeps the data and a
    # read position that starts after the checksum.
    def __init__(self, data: bytes) -> None:
        # [FUNCTION DOC] Validates the leading CRC32 against the rest of the data and sets
        # the read position. Raises AssertionError if data is shorter than the checksum, and
        # RuntimeError if the checksum does not match.
        # Check for data corruption
        if len(data) < CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Input data is too short to contain checksum")
        digest = zlib.crc32(data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Computed checksum digest has unexpected size")
        if data[0:CHECKSUM_DIGEST_SIZE] != digest:
            raise RuntimeError(
                "Bytes object is corrupted, checksum does not match. "
                f"Expected: {data[0:CHECKSUM_DIGEST_SIZE]!r}, Got: {digest!r}"
            )

        self._data = data
        self._i = CHECKSUM_DIGEST_SIZE

    def is_finished(self) -> bool:
        # [FUNCTION DOC] Returns whether the read position has reached the end of the data.
        return len(self._data) == self._i

    def read_uint64(self) -> int:
        # [FUNCTION DOC] Reads 8 big-endian bytes as an unsigned int and advances the
        # position. No bounds check: a truncated tail yields a short read value.
        result = int.from_bytes(
            self._data[self._i : self._i + 8], byteorder="big", signed=False
        )
        self._i += 8
        return result

    def read_str(self) -> str:
        # [FUNCTION DOC] Reads length-prefixed bytes, base64-decodes them and returns the
        # UTF-8 text; advances the position.
        return base64.b64decode(self.read_bytes()).decode("utf-8")

    def read_bytes(self) -> bytes:
        # [FUNCTION DOC] Reads a 64-bit length then that many bytes; advances the position.
        size = self.read_uint64()
        result = self._data[self._i : self._i + size]
        self._i += size
        return result


#######################################
# AppendingByteSerializer
#######################################


class AppendingByteSerializer(Generic[T]):
    # [CLASS DOC] Accumulates items of type T into a byte buffer via a user-supplied
    # serialize function taking a writer and an item. The buffer starts with an encoding
    # version number. Methods: clear, append, extend, to_bytes, and static to_list to decode.
    """
    Provides efficient serialization and deserialization of list of bytes
    Note that this does not provide any guarantees around byte order
    """

    _serialize_fn: Callable[[BytesWriter, T], None]
    _writer: BytesWriter

    def __init__(
        self,
        *,
        serialize_fn: Callable[[BytesWriter, T], None],
    ) -> None:
        # [FUNCTION DOC] Stores the keyword-only serialize function and starts an empty buffer.
        self._serialize_fn = serialize_fn
        self.clear()

    def clear(self) -> None:
        # [FUNCTION DOC] Discards all content and starts a new buffer with the version number.
        self._writer = BytesWriter()
        # First 8-bytes are for version
        self._writer.write_uint64(_ENCODING_VERSION)

    def append(self, data: T) -> None:
        # [FUNCTION DOC] Serializes one item into the buffer using the serialize function.
        self._serialize_fn(self._writer, data)

    def extend(self, elems: Iterable[T]) -> None:
        # [FUNCTION DOC] Appends each item of the iterable in order.
        for elem in elems:
            self.append(elem)

    def to_bytes(self) -> bytes:
        # [FUNCTION DOC] Returns the checksummed bytes of everything appended so far.
        return self._writer.to_bytes()

    @staticmethod
    def to_list(data: bytes, *, deserialize_fn: Callable[[BytesReader], T]) -> list[T]:
        # [FUNCTION DOC] Decodes bytes from to_bytes into a list: verifies checksum and
        # version, then calls the deserialize function on a reader until it is exhausted.
        # Raises RuntimeError on corruption, AssertionError on version mismatch.
        # KNOWN ISSUE: the mismatch message reads another 64-bit value, so it reports the
        # next field rather than the bad version.
        reader = BytesReader(data)
        if reader.read_uint64() != _ENCODING_VERSION:
            raise AssertionError(
                f"Encoding version mismatch in AppendingByteSerializer.to_list, \
                    got {reader.read_uint64()}"
            )

        result: list[T] = []
        while not reader.is_finished():
            result.append(deserialize_fn(reader))
        return result
