# [MODULE DOC] Experimental checkpointer API: the abstract Checkpointer plus SyncCheckpointer
# (blocking write via CheckpointWriter) and AsyncCheckpointer (stage then write in a separate
# process). Defines module constants LOG_INTERVAL and a TypeVar T; no other import-time side effects.
import abc
import logging
from concurrent.futures import Future
from typing import Any, TypeVar

from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter
from .staging import CheckpointStager
from .types import STATE_DICT
from .utils import wrap_future


logger = logging.getLogger(__name__)

LOG_INTERVAL = 60
T = TypeVar("T")


class Checkpointer(abc.ABC):
    # [CLASS DOC] Abstract interface with save, load and close; experimental and subject to change.
    """
    WARNING: This class is experimental, and is created to validate certain ideas,
    and is subjected to change or deprecation and we strong discourage any usages at
    this time.

    Abstract base class that defines the API for checkpointing.

    This class defines the interface for coordinating the writing and loading of model
    state dictionaries to and from storage. It provides abstract methods to save and load model states
    with support for both synchronous and asynchronous operations.

    Concrete implementations of this class must implement all the abstract methods.
    """

    @abc.abstractmethod
    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> tuple[Future, Future] | None:
        # [FUNCTION DOC] Abstract: saves state_dict under path. Returns None for synchronous implementations
        # or a tuple of (stage future, write future) for asynchronous ones.
        """
        Save a state dictionary to storage.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            For synchronous implementations: None
            For asynchronous implementations: tuple of (stage_future, write_future)
                                            representing the staging and writing operations.
        """

    @abc.abstractmethod
    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        # [FUNCTION DOC] Abstract: loads and returns a state dict from path, optionally restricted to keys of a
        # provided state_dict, with a default map location; strict makes missing keys an error.
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.
        """

    @abc.abstractmethod
    def close(self) -> None:
        # [FUNCTION DOC] Abstract: releases resources held by the checkpointer.
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """


class SyncCheckpointer(Checkpointer):
    # [CLASS DOC] Blocking checkpointer. State: a CheckpointWriter (_writer) and CheckpointReader (_reader).
    # Save writes immediately; load reads synchronously.
    """
    Synchronous implementation of Checkpointer.

    This class coordinates the writing and loading of model state dictionaries to and from storage
    using only synchronous operations. It provides a simple, efficient interface for checkpoint
    operations without async overhead.

    Attributes:
        _writer: CheckpointWriter for writing state dictionaries to storage.
        _reader: CheckpointReader for reading state dictionaries from storage.

    Example:
        checkpointer = SyncCheckpointer(writer=writer, reader=reader)
        checkpointer.save(state_dict, path)
        loaded_state_dict = checkpointer.load(path)
    """

    def __init__(
        self,
        writer: CheckpointWriter,
        reader: CheckpointReader,
    ):
        # [FUNCTION DOC] Stores the writer and reader; no other side effects.
        """
        Initialize a synchronous checkpointer.

        Args:
            writer: CheckpointWriter for writing checkpoints to storage.
            reader: CheckpointReader for reading checkpoints from storage.
        """
        self._writer = writer
        self._reader = reader

    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> tuple[Future, Future] | None:
        # [FUNCTION DOC] Writes state_dict to path via the writer (extra kwargs forwarded) and blocks until done.
        # Always returns None.
        # KNOWN ISSUE: docstring examples pass arguments as state dict first, path second; the real order is path first.
        """
        Save a state dictionary to storage synchronously.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            Always returns None as operations are synchronous.

        Example:
            checkpointer.save("/path/to/checkpoint", state_dict)
        """
        logger.debug("Saving checkpoint synchronously to %s", path)
        self._writer.write(path, state_dict, **kwargs)
        return None

    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        # [FUNCTION DOC] Reads the checkpoint at path through the reader, mapping tensors with default_map_location.
        # If state_dict is given only its keys are filled. Returns the loaded state dict.
        # Raises RuntimeError when strict is true and the reader reported missing keys; FileNotFoundError
        # propagates from the reader.
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        logger.info("Loading checkpoint from %s", path)

        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            state_dict=state_dict,
            map_location=default_map_location,
            **kwargs,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")
        return loaded_state_dict

    def close(self) -> None:
        # [FUNCTION DOC] Closes the underlying writer (the reader holds no resources) and logs.
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """
        self._writer.close()
        logger.info("SyncCheckpointer closed")


class AsyncCheckpointer(Checkpointer):
    # [CLASS DOC] Asynchronous checkpointer. State: reader, a CheckpointStager, a CheckpointProcess and
    # _write_future (the in-flight write, or None). Save stages the state dict then hands it to the
    # process for background writing; at most one outstanding write is allowed at a time.
    # Load is always synchronous.
    # KNOWN ISSUE: docstring example passes arguments as state dict first, path second; the real order is path first.
    """
    Asynchronous implementation of Checkpointer.

    This class coordinates the writing and loading of model state dictionaries to and from storage
    using asynchronous operations for saving. It provides efficient async checkpoint operations
    with staging and background writing capabilities.

    Attributes:
        _reader: CheckpointReader for reading state dictionaries from storage.
        _checkpoint_stager: Stager for async operations.
        _checkpoint_process: Process for async operations.
        _write_future: Future representing the ongoing async write operation.

    Example:
        checkpointer = AsyncCheckpointer(
            reader=reader,
            checkpoint_stager=stager,
            checkpoint_process=process
        )
        stage_future, write_future = checkpointer.save(state_dict, path)
        # ... do other work ...
        write_future.result()  # Wait for completion
    """

    def __init__(
        self,
        checkpoint_stager: CheckpointStager,
        checkpoint_process: CheckpointProcess,
        reader: CheckpointReader,
    ):
        # [FUNCTION DOC] Stores the stager, process and reader and initializes the pending write future to None.
        """
        Initialize an asynchronous checkpointer.

        Args:
            checkpoint_stager: Stager for async operations.
            checkpoint_process: Process for async operations.
            reader: CheckpointReader for reading checkpoints from storage.
        """
        self._reader = reader
        self._checkpoint_stager = checkpoint_stager
        self._checkpoint_process = checkpoint_process
        self._write_future: Future[Any] | None = None

    def save(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> tuple[Future, Future] | None:
        # [FUNCTION DOC] Waits for (and re-raises errors from) the previous write future, stages state_dict via the
        # stager, then submits the staged result and path to the checkpoint process, remembering the new
        # write future. Returns a tuple of (staging future wrapped to yield None, write future).
        # Raises RuntimeError if the write future is unexpectedly None, or any error from the previous write.
        """
        Save a state dictionary to storage asynchronously.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.
            **kwargs: Additional keyword arguments to pass to the stager and writer.

        Returns:
            A tuple of (stage_future, write_future) representing the staging and writing operations.

        Example:
            stage_future, write_future = checkpointer.save("/path/to/checkpoint", state_dict)
            # ... do other work ...
            write_future.result()  # Wait for completion
        """
        logger.info(
            "Initiating checkpoint save to %s. Will wait for prev checkpoints to complete.",
            path,
        )
        # Wait for previous checkpoint ops to finish and verify they are successful
        if self._write_future is not None:
            self._write_future.result()

        logger.debug("Starting state dictionary staging")
        staging_result = self._checkpoint_stager.stage(
            state_dict=state_dict,
            **kwargs,
        )

        logger.debug("Starting checkpoint write to %s", path)
        self._write_future = self._checkpoint_process.write(
            staging_result, path, **kwargs
        )
        logger.info("Checkpoint save to %s initiated", path)

        # Return futures for the staging and writing operations
        if self._write_future is not None:
            return wrap_future(staging_result), self._write_future
        else:
            # This should not happen since we just assigned _write_future above
            raise RuntimeError("Write future is unexpectedly None")

    def load(
        self,
        path: str,
        state_dict: STATE_DICT | None = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> STATE_DICT:
        # [FUNCTION DOC] Synchronous load, identical in behavior to SyncCheckpointer.load: reads through the reader
        # and returns the state dict; raises RuntimeError on missing keys when strict, FileNotFoundError if absent.
        """
        Load a state dictionary from storage.

        Loading is always performed synchronously, even in AsyncCheckpointer.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        logger.info("Loading checkpoint from %s", path)

        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            state_dict=state_dict,
            map_location=default_map_location,
            **kwargs,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")
        return loaded_state_dict

    def close(self) -> None:
        # [FUNCTION DOC] Closes the stager and then the checkpoint process, and logs.
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of async resources.
        """
        self._checkpoint_stager.close()
        self._checkpoint_process.close()
        logger.info("AsyncCheckpointer closed")
