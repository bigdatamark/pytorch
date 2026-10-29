# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] Experimental extension mechanism for distributed checkpointing: Extension (abstract
# base), StreamTransformExtension (byte-stream transforms such as compression), ZStandard (zstd
# compression using the zstandard or pyzstd package) and ExtensionRegistry (maps descriptor names to classes).
# Import-time side effect: tries to import pyzstd and zstandard via try_import, leaving None if missing.
# Exports: Extension, StreamTransformExtension, ZStandard, ExtensionRegistry.

import abc
import io
from collections.abc import Sequence
from typing import cast, IO

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

from torch._utils import try_import


# NOTE: everything in this file is experimental, and subject to
# change.  Feedback and bug fixes are always welcome.

pyzstd_module_name = "pyzstd"
pyzstd = try_import(pyzstd_module_name)
zstandard_module_name = "zstandard"
zstandard = try_import(zstandard_module_name)


__all__ = [
    "Extension",
    "StreamTransformExtension",
    "ZStandard",
    "ExtensionRegistry",
]


class Extension(abc.ABC):
    # [CLASS DOC] Abstract base for checkpoint format extensions. Subclasses supply a static registry name,
    # a static factory from a version string, and a descriptor string stored in metadata so the extension
    # can be rebuilt at read time.
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layout or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """

    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str:
        # [FUNCTION DOC] Abstract static: returns the name under which the extension class is registered.
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> "Extension":
        # [FUNCTION DOC] Abstract static: builds an extension instance from the version part of a descriptor.
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @abc.abstractmethod
    def get_descriptor(self) -> str:
        # [FUNCTION DOC] Abstract: returns the descriptor string for metadata, of the form name, optional local domain, optional version.
        """
        Return descriptor name to be included in metadata.  The form should be
        "extension_name[@local-domain][/version]".
        """


class StreamTransformExtension(Extension):
    # [CLASS DOC] Abstract extension that wraps byte streams (compression, encryption). Implementations
    # should process data in moderate chunks.
    """
    An extension which performs transformation on a byte stream, such as compression
    or encryption.

    Implementations should try to be memory friendly and performant.  For example, don't
    read the whole input, then transform it, and write it back.  If at all possible, do it in
    chunks.  But, don't read/transform/write one byte at a time, either.
    """

    @abc.abstractmethod
    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Abstract: given a writable stream, returns a stream that transforms data written to it and writes the result to the given stream.
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """

    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Abstract: given a readable stream, returns a stream that reads from it and returns transformed data.
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """


class ZStandard(StreamTransformExtension):
    # [CLASS DOC] Zstandard stream compression extension, registered as stream.zstd with version 1.
    # Prefers the zstandard package and falls back to pyzstd; construction fails if neither is installed.
    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Returns true if either the zstandard or pyzstd module was imported.
        return zstandard is not None or pyzstd is not None

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        # [FUNCTION DOC] Builds a ZStandard instance. Raises ValueError if the major version part is not 1 or if no zstd module is available.
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        if not ZStandard.is_available():
            raise ValueError(
                f"Stream with ZStandard compression cannot be processed because "
                f"no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )
        return ZStandard()

    @staticmethod
    def registry_name() -> str:
        # [FUNCTION DOC] Returns the registry name, the string stream.zstd.
        return "stream.zstd"

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base class. Raises ValueError if no zstd module is available.
        super().__init__()
        if not ZStandard.is_available():
            raise ValueError(
                f"ZStandard extension is unavailable because no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )

    def get_descriptor(self) -> str:
        # [FUNCTION DOC] Returns the registry name followed by a slash and version 1.
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Returns a compressing writer over output: the zstandard stream writer when available,
        # otherwise a local raw-IO class that compresses with pyzstd and writes to output.
        if zstandard is not None:
            compressor = zstandard.ZstdCompressor()  # type: ignore[union-attr]
            return compressor.stream_writer(output)

        class Writer(io.RawIOBase):
            def __init__(self, output: IO[bytes]) -> None:
            # [FUNCTION DOC] HELPER: Stores the output stream and creates a pyzstd compressor.
                self.output = output
                self.compressor = pyzstd.ZstdCompressor()  # type: ignore[union-attr]

            def writeable(self) -> bool:
            # [FUNCTION DOC] HELPER: Reports that the stream is writable (always true).
                return True

            def write(self, b: Buffer) -> int | None:
            # [FUNCTION DOC] HELPER: Compresses the buffer, writes any produced bytes to the output and returns the input length.
                outdata = self.compressor.compress(b)
                if outdata:
                    self.output.write(outdata)
                return len(memoryview(b))

            def flush(self) -> None:
            # [FUNCTION DOC] HELPER: Flushes the compressor, writes the remaining bytes and flushes the underlying output.
            # KNOWN ISSUE: flush ends the compressed frame; calling it repeatedly or mid-stream may interact badly with later writes, and the output is not closed here.
                outdata = self.compressor.flush()
                if outdata:
                    self.output.write(outdata)
                self.output.flush()

        return cast(IO[bytes], Writer(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Returns a decompressing reader over input: the zstandard stream reader when available,
        # otherwise a local raw-IO class that decompresses with pyzstd.
        if zstandard is not None:
            decompressor = zstandard.ZstdDecompressor()  # type: ignore[union-attr]
            return decompressor.stream_reader(input)

        class Reader(io.RawIOBase):
            def __init__(self, input: IO[bytes]) -> None:
            # [FUNCTION DOC] HELPER: Stores the input stream and creates a pyzstd endless decompressor.
                self.input = input
                self.decompressor = pyzstd.EndlessZstdDecompressor()  # type: ignore[union-attr]

            def readable(self) -> bool:
            # [FUNCTION DOC] HELPER: Reports that the stream is readable (always true).
                return True

            def readinto(self, b: Buffer) -> int | None:
            # [FUNCTION DOC] HELPER: Fills the given buffer with decompressed bytes. Reads about one maximal block
            # from the input when the decompressor needs data, decompresses up to the buffer length and copies
            # the result into the buffer. Returns the byte count, or None if no data is available yet.
                # This needs to read enough so it can decompress
                # something so the output doesn't look like EOF.  This
                # means reading at least one block.  The max block
                # size is 128KB, so we read that plus some
                # overhead to be sure.

                if self.decompressor.needs_input:
                    indata = self.input.read((128 + 6) * 1024)
                else:
                    indata = b""

                bview = memoryview(b)
                blen = len(bview)
                outdata = self.decompressor.decompress(indata, blen)
                if outdata is None:
                    return None

                count = len(outdata)
                bview[:count] = outdata
                return count

            def seekable(self) -> bool:
            # [FUNCTION DOC] HELPER: Reports that the stream is not seekable (always false).
                return False

        return cast(IO[bytes], Reader(input))


class ExtensionRegistry:
    # [CLASS DOC] Registry mapping extension registry names to Extension classes, pre-populated with ZStandard.
    # State: the extensions mapping. Key methods: register and from_descriptor_list.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the extensions mapping holding the built-in extension classes keyed by registry name.
        # Populate default registry contents
        self.extensions: dict[str, type[Extension]] = {
            cls.registry_name(): cls for cls in (ZStandard,)
        }

    def register(self, cls: type[Extension]) -> None:
        # [FUNCTION DOC] Adds (or replaces) an extension class in the registry under its registry name.
        self.extensions[cls.registry_name()] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        # [FUNCTION DOC] Rebuilds extension instances from descriptor strings saved at write time. Each descriptor
        # is split at the first slash into name and version; the name selects a class and the version is
        # passed to its from_descriptor factory. Raises ValueError for an unknown name; errors from the
        # factory propagate. Returns a list in the same order.
        """
        Given a sequence of descriptor strings as returned by
        Extension.get_descriptor at save time, creates a sequence of
        Extension instances.  The name[@local-domain] preceding the
        version number is used to look up an implementation class in
        the registry, and the version is passed to the class's
        from_descriptor static method.  If the registry contains no
        match, this will throw ValueError.  If the from_descriptor
        method raises an exception, that will pass through to the
        caller.
        """

        def from_descriptor(desc: str) -> Extension:
            # [FUNCTION DOC] HELPER: Splits one descriptor into name and version, looks up the class and builds the instance.
            # KNOWN ISSUE: the check for a missing version tests for None, but partition returns an empty string when there is no slash, so the default of 0 is never used and an empty version is passed instead.
            name, _, version = desc.partition("/")
            if version is None:
                version = 0
            ext = self.extensions.get(name)
            if not ext:
                raise ValueError(f"Unknown extension {name=}")
            # pyrefly: ignore [bad-argument-type]
            return ext.from_descriptor(version)

        return [from_descriptor(desc) for desc in descriptors]
