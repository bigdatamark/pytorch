# mypy: allow-untyped-defs

# [MODULE DOC]
# Test helpers for distributed checkpoint tests: a demo rot13 stream-transform extension,
# a factory for an extension registry that contains it, and two decorators that give a test
# a shared temporary directory or a temporarily changed checkpoint logger level.
# No import-time side effects beyond imports.

# Copyright (c) Meta Platforms, Inc. and affiliates

import io
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from functools import wraps
from typing import Any, cast, IO

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

import torch.distributed as dist
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)


class Rot13Example(StreamTransformExtension):
    # [CLASS DOC]
    # Example StreamTransformExtension that transforms bytes written to or read from a stream
    # with a letter-substitution cipher, applied identically in both directions. Holds only a
    # private chunk size. Registered under the name stream.rot13 with descriptor version 1.
    # KNOWN ISSUE: the transform swaps letter case (adds or subtracts the case offset) rather
    # than rotating letters by 13, and leaves digits untouched despite the docstring; it is only
    # self-inverse, which is all the tests need.
    """
    This is an example stream transform extension which just does rot13 on each
    alphanumeric character of the stream.  It is mainly intended as a demonstration
    and for testing; there isn't a production use case for this.
    """

    def __init__(self, chunk_size: int = io.DEFAULT_BUFFER_SIZE) -> None:
        # [FUNCTION DOC] Stores chunk_size (default io.DEFAULT_BUFFER_SIZE) in a private field.
        # The value is not used by the transform methods.
        super().__init__()
        self._chunk_size = chunk_size

    @staticmethod
    def from_descriptor(version: str) -> "Rot13Example":
        # [FUNCTION DOC] Rebuilds an instance from a descriptor version string. Raises ValueError
        # if the major component (text before the first dot) is not 1; otherwise returns a new
        # default-configured instance.
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        return Rot13Example()

    @staticmethod
    def registry_name() -> str:
        # [FUNCTION DOC] Returns the registry name of this extension, the string stream.rot13.
        return "stream.rot13"

    def get_descriptor(self) -> str:
        # [FUNCTION DOC] Returns the descriptor string: the registry name, a slash, and version 1.
        return f"{self.registry_name()}/1"

    @staticmethod
    def _rot13bytes(b: Buffer, count: int) -> None:
        # [FUNCTION DOC] HELPER: Transforms the first count bytes of the buffer in place through a
        # memoryview. Upper-case ASCII letters are shifted to lower-case and lower-case to
        # upper-case; other bytes are unchanged. Returns nothing.
        # KNOWN ISSUE: named rot13 but only swaps case.
        b = memoryview(b)
        for i in range(count):
            ch = b[i]
            if ch >= ord("A") and ch <= ord("Z"):
                ch += ord("a") - ord("A")
            elif ch >= ord("a") and ch <= ord("z"):
                ch += ord("A") - ord("a")
            b[i] = ch

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Wraps an output byte stream in a writer that transforms data on write.
        # Each write copies the input into a new bytearray (the caller's buffer is not mutated),
        # transforms it, and forwards it to the wrapped stream, returning that stream's result.
        # Returns the writer cast to a byte stream.
        class Writer(io.RawIOBase):
            # [CLASS DOC] HELPER: Raw writable stream that transforms each chunk and forwards it to
            # the wrapped output stream stored in the output attribute.
            def __init__(self, output: IO[bytes]) -> None:
                # [FUNCTION DOC] HELPER: Stores the wrapped output stream.
                self.output = output

            def writeable(self) -> bool:
                # [FUNCTION DOC] HELPER: Reports that the stream is writable (always True).
                return True

            def write(self, b: Buffer) -> int | None:
                # [FUNCTION DOC] HELPER: Copies the buffer, transforms the copy and writes it to
                # the wrapped stream; returns the wrapped stream's write result (byte count or
                # None.
                # Don't mutate the input
                chunk = bytearray(b)
                Rot13Example._rot13bytes(chunk, len(chunk))
                return self.output.write(chunk)

            def flush(self) -> None:
                # [FUNCTION DOC] HELPER: Flushes the wrapped output stream.
                self.output.flush()

        return cast(IO[bytes], Writer(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        # [FUNCTION DOC] Wraps an input byte stream in a reader that transforms data on read.
        # Returns the reader cast to a byte stream.
        class Reader(io.RawIOBase):
            # [CLASS DOC] HELPER: Raw readable stream that fills caller buffers from the wrapped
            # input stream and transforms the filled bytes in place. Supports seek and tell by
            # delegation.
            def __init__(self, input: IO[bytes]) -> None:
                # [FUNCTION DOC] HELPER: Stores the wrapped input stream.
                self.input = input

            def readable(self) -> bool:
                # [FUNCTION DOC] HELPER: Reports that the stream is readable (always True).
                return True

            def readinto(self, b: Buffer) -> int | None:
                # [FUNCTION DOC] HELPER: Reads into the buffer b from the wrapped stream, using
                # its readinto if present, otherwise a plain read followed by a copy. Returns the
                # byte count, or None if no data is available; a count of zero or None is returned
                # without transforming. Otherwise transforms the filled bytes in place and returns
                # the count.
                if hasattr(self.input, "readinto"):
                    count = self.input.readinto(b)
                else:
                    # It's possible self.input is an IO[bytes] with no readinto method.
                    # In that case, we emulate with a read and copy.  In practice,
                    # all of the current concrete extensions have readinto.
                    view = memoryview(b)
                    r = self.input.read(len(view))
                    if r is None:
                        count = None
                    else:
                        count = len(r)
                        view[:count] = r
                if count == 0 or count is None:
                    return count

                Rot13Example._rot13bytes(b, count)
                return count

            def seekable(self) -> bool:
                # [FUNCTION DOC] HELPER: Delegates to the wrapped stream's seekable.
                return self.input.seekable()

            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                # [FUNCTION DOC] HELPER: Delegates seek to the wrapped stream and returns the new
                # position.
                return self.input.seek(offset, whence)

            def tell(self) -> int:
                # [FUNCTION DOC] HELPER: Delegates tell to the wrapped stream and returns the
                # position.
                return self.input.tell()

        return cast(IO[bytes], Reader(input))


def get_test_extension_registry() -> ExtensionRegistry:
    # [FUNCTION DOC] Builds and returns a new ExtensionRegistry with Rot13Example registered.
    registry = ExtensionRegistry()
    registry.register(Rot13Example)
    return registry


def with_temp_dir(
    func: Callable | None = None,
) -> Callable | None:
    # [FUNCTION DOC] Decorator for test methods that gives the test a temporary directory in the
    # temp_dir attribute of the test instance, removes it afterwards (errors ignored), and returns
    # the wrapped function. With an initialized process group, rank 0 creates the directory and
    # broadcasts its path to all ranks (with filesystem syncs around it); otherwise a local one is
    # made. Prints the path. Raises AssertionError if func is None, so it cannot be used with
    # arguments.
    # KNOWN ISSUE: both branches of the cleanup do the same thing, so every rank removes the
    # directory; the rank check is redundant. The return annotation allows None but never is.
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    if func is None:
        raise AssertionError("Expected func to not be None")

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Wrapper that sets up the temp directory, runs the test function
        # with the given arguments, and removes the directory in a finally clause. Returns None.
        if dist.is_initialized():
            # Only create temp_dir when rank is 0
            if dist.get_rank() == 0:
                temp_dir = tempfile.mkdtemp()
                print(f"Using temp directory: {temp_dir}")
            else:
                temp_dir = ""
            object_list = [temp_dir]

            # Broadcast temp_dir to all the other ranks
            os.sync()
            dist.broadcast_object_list(object_list)
            self.temp_dir = object_list[0]
            os.sync()
        else:
            temp_dir = tempfile.mkdtemp()
            print(f"No process group initialized, using temp directory: {temp_dir}")
            self.temp_dir = temp_dir

        try:
            func(self, *args, **kwargs)
        finally:
            if dist.is_initialized() and dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper


def with_checkpoint_logging(
    func: Callable | None = None,
    logger_name: str = "torch.distributed.checkpoint",
    level: int = logging.INFO,
) -> Callable | None:
    # [FUNCTION DOC] Decorator for tests that sets the named logger (default the distributed
    # checkpoint logger) to the given level (default INFO) for the duration of the test and then
    # restores the original level. Raises AssertionError if func is None. Returns the wrapper.
    """
    Wrapper to configure checkpoint logging for distributed tests.

    Args:
        func: The test function to wrap
        logger_name: Name of the logger to configure (default: 'torch.distributed.checkpoint')
        level: Logging level to set (default: logging.INFO)
    """
    if func is None:
        raise AssertionError("Expected func to not be None")

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Sets the logger level, runs the test function, and restores the
        # previous level in a finally clause. Returns None.
        # Get the logger and store original level
        target_logger = logging.getLogger(logger_name)
        original_level = target_logger.level

        # Set the desired logging level
        target_logger.setLevel(level)

        try:
            func(self, *args, **kwargs)
        finally:
            # Restore original logging level
            target_logger.setLevel(original_level)

    return wrapper
