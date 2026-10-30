# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the appending byte serializer utility: round trips of integers and of a
# user defined record class through the serializer, and detection of a corrupted checksum in the
# reader. Runs the inductor test case runner when executed as a program.

import dataclasses

from torch.testing._internal.common_utils import TestCase
from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)


class TestAppendingByteSerializer(TestCase):
    # [CLASS DOC] Test case for the serializer, writer and reader classes.
    def test_write_and_read_int(self) -> None:
        # [FUNCTION DOC] Serializes a list of integers with a serialize function that writes each
        # as a 64 bit unsigned value, checks decoding gives the same list, then extends with more
        # integers and checks that the decoded result is the concatenation (the serializer keeps
        # accumulating).
        def int_serializer(writer: BytesWriter, i: int) -> None:
            # [FUNCTION DOC] HELPER: Serialize function writing one integer to the writer as an
            # unsigned 64 bit value.
            writer.write_uint64(i)

        def int_deserializer(reader: BytesReader) -> int:
            # [FUNCTION DOC] HELPER: Deserialize function reading one unsigned 64 bit value from
            # the reader and returning it.
            return reader.read_uint64()

        s = AppendingByteSerializer(serialize_fn=int_serializer)

        data = [1, 2, 3, 4]
        s.extend(data)
        self.assertListEqual(
            data,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

        data2 = [8, 9, 10, 11]
        s.extend(data2)
        self.assertListEqual(
            data + data2,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

    def test_write_and_read_class(self) -> None:
        # [FUNCTION DOC] Round trips a frozen dataclass holding an int, a string and bytes using
        # its own static serialize and deserialize functions: appends one instance, decodes and
        # compares, appends a second instance and checks both come back in order.
        @dataclasses.dataclass(frozen=True, eq=True)
        class Foo:
            # [CLASS DOC] HELPER: Frozen, comparable dataclass with an int, a str and a bytes
            # field, plus static functions to write it to and read it from the byte writer and
            # reader.
            x: int
            y: str
            z: bytes

            @staticmethod
            def serialize(writer: BytesWriter, cls: "Foo") -> None:
                # [FUNCTION DOC] HELPER: Writes the int field, then the string, then the bytes
                # field, in that order.
                writer.write_uint64(cls.x)
                writer.write_str(cls.y)
                writer.write_bytes(cls.z)

            @staticmethod
            def deserialize(reader: BytesReader) -> "Foo":
                # [FUNCTION DOC] HELPER: Reads an int, a string and a bytes value in the order
                # they were written and builds a new instance from them.
                x = reader.read_uint64()
                y = reader.read_str()
                z = reader.read_bytes()
                return Foo(x, y, z)

        a = Foo(5, "ok", bytes([15]))
        b = Foo(10, "lol", bytes([25]))

        s = AppendingByteSerializer(serialize_fn=Foo.serialize)
        s.append(a)
        self.assertListEqual(
            [a],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

        s.append(b)
        self.assertListEqual(
            [a, b],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

    def test_checksum(self) -> None:
        # [FUNCTION DOC] Builds a writer containing one string, flips the first byte of the
        # produced bytes (inside the checksum), and asserts constructing a reader on the result
        # raises RuntimeError reporting a checksum mismatch.
        writer = BytesWriter()
        writer.write_str("test")
        b = writer.to_bytes()
        b = bytearray(b)
        b[0:1] = b"\x00"
        b = bytes(b)

        with self.assertRaisesRegex(
            RuntimeError, r"Bytes object is corrupted, checksum does not match.*"
        ):
            BytesReader(b)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
