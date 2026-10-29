# mypy: allow-untyped-defs
# [MODULE DOC] CUDA Sanitizer (CSAN): detects data races between kernels on different CUDA streams. It listens to low-level
# GPU trace callbacks (events, streams, allocations, synchronizations) and to every dispatched operator, tracks the last
# write and subsequent reads per data pointer plus per-stream happens-before state, and on an unsynchronized access
# prints a report to standard error and raises. Main parts: AccessType, Access, error classes, TensorInfo,
# _TensorsAccessed, StreamSynchronizations, EventHandler, ArgumentHandler, CUDASanitizerDispatchMode, CUDASanitizer.
# Import side effect: creates the module-level cuda_sanitizer object, whose construction creates the dispatch mode,
# activates GPU tracing and registers callbacks (the mode is only entered once enable is called).
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum
import functools
import inspect
import io
import logging
import re
import sys
import textwrap
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

DEFAULT_STREAM_ID = 0

TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")

DataPtr = int
StreamId = int
EventId = int
SeqNum = int

logger = logging.getLogger(__name__)

# Note that this is only factories that take Tensor as input as they are
# the ones we care about.
FACTORY_FUNCTION_REGEX = re.compile("(new_.*|.*_like)")


class AccessType(enum.Enum):
    # [CLASS DOC] Enumeration of access kinds READ and WRITE; text form is a phrase (reading from / writing to) used in reports.
    READ = enum.auto()
    WRITE = enum.auto()

    def __str__(self):
        # [FUNCTION DOC] Returns the phrase reading from for READ and writing to for WRITE.
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass
class Access:
    # [CLASS DOC] Dataclass describing one access to a tensor by a kernel: access type, kernel sequence number, stream id, operator
    # (schema), argument names aliasing the tensor, whether it was an output, and the captured stack trace.
    r"""Stores information about a single access to a tensor by a kernel.

    Args:
        type: either AccessType.READ or AccessType.Write.
        seq_num: the sequential number of the kernel performing the access.
        stream: the stream id of the stream executing the kernel.
        operator: the schema of the launched kernel, which lists the
            arguments and return type.
        aliases: the arguments in the schema this access corresponds to.
        is_output: Whether the tensor was an output of the kernel.
        stack_trace: the stack summary object captured during access.
    """

    type: AccessType
    seq_num: SeqNum
    stream: StreamId
    operator: str
    aliases: list[str]
    is_output: bool
    stack_trace: traceback.StackSummary


class SynchronizationError(Exception):
    # [CLASS DOC] Base class of errors detected by the sanitizer.
    """Base class for errors detected by CUDA Sanitizer."""


class UnsynchronizedAccessError(SynchronizationError):
    # [CLASS DOC] Error describing two unsynchronized accesses to the same data pointer. Attributes: data_ptr, allocation_stack_trace
    # (may be None), current_access, previous_access.
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: traceback.StackSummary | None,
        current_access: Access,
        previous_access: Access,
    ):
        # [FUNCTION DOC] Stores the data pointer, allocation stack trace and the two conflicting accesses.
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access

    def __str__(self):
        # [FUNCTION DOC] Builds the multi-line report: banner, data pointer, current access (stream, operator, argument names, output flag, stack
        # trace), previous access, and the allocation stack trace or a note that it was not found. Returns the text.
        def format_access(access: Access):
            # [FUNCTION DOC] HELPER: Appends the description of one access (operator, access type, argument names, output marker and stack trace)
            # to the enclosing message buffer.
            message.write(f"{access.operator}\n{access.type}")
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                if access.is_output:
                    message.write(", and to")
            if access.is_output:
                message.write(" the output")
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        with io.StringIO() as message:
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            format_access(self.current_access)

            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            format_access(self.previous_access)

            if self.allocation_stack_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}"
                )
            else:
                message.write("Trace for tensor allocation not found.")
            return message.getvalue()


class CUDASanitizerErrors(Exception):
    # [CLASS DOC] Exception wrapping the list of synchronization errors found for one kernel launch (attribute errors).
    """Wrapper class for errors reported by CUDA Sanitizer."""

    def __init__(self, errors: list[SynchronizationError]):
        # [FUNCTION DOC] Stores the list of errors.
        self.errors = errors

    def __str__(self):
        # [FUNCTION DOC] Returns a short message with the number of errors.
        return f"detected {len(self.errors)} errors"


@dataclass
class TensorInfo:
    # [CLASS DOC] Dataclass tracking one tensor: allocation stack trace (None if the allocation was not observed), reads since the last
    # write, and the last write access (None if none).
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
    """

    allocation_stack_trace: traceback.StackSummary | None
    reads: list[Access] = field(default_factory=list)
    write: Access | None = None


class _TensorsAccessed:
    # [CLASS DOC] HELPER: Registry mapping data pointer to TensorInfo with methods to create, delete and record reads and writes. The ensure
    # methods log at info level and backfill when the trace lacks an expected allocation or deallocation.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty pointer to TensorInfo mapping.
        self.accesses: dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        # [FUNCTION DOC] Registers the pointer with no allocation stack (and logs) if it is unknown.
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        # [FUNCTION DOC] If the pointer is already registered (a missed free), logs and deletes the stale entry.
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: traceback.StackSummary | None
    ) -> None:
        # [FUNCTION DOC] Registers a fresh TensorInfo for the pointer with the given allocation stack trace, replacing any entry.
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        # [FUNCTION DOC] Removes the pointer's entry; raises a key error if absent.
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        # [FUNCTION DOC] Returns whether any reads are recorded since the last write.
        return bool(self.accesses[data_ptr].reads)

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> traceback.StackSummary | None:
        # [FUNCTION DOC] Returns the allocation stack trace stored for the pointer (may be None).
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Access | None:
        # [FUNCTION DOC] Returns the last write access, or None.
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> list[Access]:
        # [FUNCTION DOC] Returns the live list of reads since the last write.
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        # [FUNCTION DOC] Appends a read access for the pointer.
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        # [FUNCTION DOC] Records the access as the last write and clears the pending reads.
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []


class StreamSynchronizations:
    # [CLASS DOC] Models happens-before knowledge between streams as vector clocks. Attributes: current_sync_states (per stream, mapping of
    # other stream to the latest kernel sequence number known to be ordered before it), recorded_sync_states (snapshot per
    # event), host_sync_state (what the host has synchronized with). The default stream exists from construction. Streams
    # or events missing from the trace are backfilled with an info log.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates empty state tables and registers the default stream.
        self.current_sync_states: dict[StreamId, dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: dict[EventId, dict[StreamId, SeqNum]] = {}
        self.host_sync_state: dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    def _ensure_stream_exists(self, stream: StreamId) -> None:
        # [FUNCTION DOC] HELPER: Creates the stream (with a log) if it is not known.
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream,
            )
            self.create_stream(stream)

    def _ensure_event_exists(self, event: EventId) -> None:
        # [FUNCTION DOC] HELPER: Creates the event (with a log) if it is not known.
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.create_event(event)

    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        # [FUNCTION DOC] HELPER: Deletes a stale event of the same id (with a log) if one is registered.
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.delete_event(event)

    def create_stream(self, stream: StreamId) -> None:
        # [FUNCTION DOC] Registers a stream: its host sync value starts at 0 and its state is a copy of the host state. A duplicate creation is
        # logged and ignored.
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream,
            )
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

    def create_event(self, event: EventId) -> None:
        # [FUNCTION DOC] Registers an event with an empty recorded state, replacing a stale one.
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    def delete_event(self, event: EventId) -> None:
        # [FUNCTION DOC] Removes the event's recorded state (backfilling first if unknown).
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        # [FUNCTION DOC] Sets the stream's own entry in its state to the new kernel sequence number.
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    def record_state(self, event: EventId, stream: StreamId) -> None:
        # [FUNCTION DOC] Stores a copy of the stream's current state as the event's recorded state (event record).
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    def _state_wait_for_other(
        self, state: dict[StreamId, SeqNum], other: dict[StreamId, SeqNum]
    ) -> None:
        # [FUNCTION DOC] HELPER: Merges other into state in place, taking the elementwise maximum sequence number per stream.
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        # [FUNCTION DOC] Merges the event's recorded state into the stream's state (stream waits for event).
        self._ensure_stream_exists(stream)
        self._ensure_event_exists(event)
        self._state_wait_for_other(
            self.current_sync_states[stream], self.recorded_sync_states[event]
        )

    def all_streams_wait_for_event(self, event: EventId) -> None:
        # [FUNCTION DOC] Makes every stream and the host wait for the event's recorded state (event synchronize).
        self._ensure_event_exists(event)
        for stream in self.current_sync_states:
            self.stream_wait_for_event(stream, event)

        self._state_wait_for_other(
            self.host_sync_state, self.recorded_sync_states[event]
        )

    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        # [FUNCTION DOC] Merges the given stream's state into every stream and the host (stream synchronize).
        self._ensure_stream_exists(stream)
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])

        self._state_wait_for_other(
            self.host_sync_state, self.current_sync_states[stream]
        )

    def sync_all_streams(self) -> None:
        # [FUNCTION DOC] Device synchronize: updates the host state with each stream's own latest value, then merges the host state into all streams.
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]

        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        # [FUNCTION DOC] Returns whether the kernel with seq_num on other_stream is known to be ordered before the current stream's work
        # (sequence number not above the current stream's knowledge of other_stream). Backfills unknown streams.
        self._ensure_stream_exists(current_stream)
        self._ensure_stream_exists(other_stream)
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)


class EventHandler:
    # [CLASS DOC] Receives sanitizer trace callbacks and kernel launches. State: tensors_accessed (_TensorsAccessed), syncs
    # (StreamSynchronizations), seq_num (global kernel counter). Underscore handler methods are registered as GPU trace callbacks;
    # _handle_kernel_launch checks for data races and returns the errors found.
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the tensor registry, synchronization model and zeroed kernel counter.
        self.tensors_accessed = _TensorsAccessed()
        self.syncs = StreamSynchronizations()
        self.seq_num: SeqNum = 0

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: set[DataPtr],
        read_write: set[DataPtr],
        outputs: set[DataPtr],
        operator: str,
        tensor_aliases: dict[int, list[str]],
    ) -> list[SynchronizationError]:
        # [FUNCTION DOC] HELPER: Handles one kernel launch on a stream. Increments the sequence number, records it for the stream, captures the stack
        # trace, then for each read-only pointer checks the previous write for ordering and records a read; for each written
        # pointer checks the reads since the last write (or the last write if none) and records the write. Returns the list of
        # UnsynchronizedAccessError found (possibly empty).
        def check_conflict(
            # [FUNCTION DOC] HELPER: Appends an UnsynchronizedAccessError to the enclosing error list if previous_access exists and is not ordered before
            # the current access.
            data_ptr: DataPtr, current_access: Access, previous_access: Access | None
        ) -> None:
            if previous_access is None:
                return
            if not self.syncs.is_ordered_after(
                current_access.stream, previous_access.seq_num, previous_access.stream
            ):
                error_list.append(
                    UnsynchronizedAccessError(
                        data_ptr,
                        self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                        current_access,
                        previous_access,
                    )
                )

        error_list: list[SynchronizationError] = []
        self.seq_num += 1
        self.syncs.update_seq_num(stream, self.seq_num)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()

        for data_ptr in read_only:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            check_conflict(
                data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
            )
            self.tensors_accessed.add_read(data_ptr, current_access)

        for data_ptr in read_write:
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            current_access = Access(
                AccessType.WRITE,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
                for previous_access in self.tensors_accessed.get_reads(data_ptr):
                    check_conflict(data_ptr, current_access, previous_access)
            else:
                check_conflict(
                    data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
                )
            self.tensors_accessed.set_write(data_ptr, current_access)

        return error_list

    def _handle_event_creation(self, event: EventId) -> None:
        # [FUNCTION DOC] HELPER: Callback for event creation.
        self.syncs.create_event(event)

    def _handle_event_deletion(self, event: EventId) -> None:
        # [FUNCTION DOC] HELPER: Callback for event deletion.
        self.syncs.delete_event(event)

    def _handle_event_record(self, event: EventId, stream: StreamId) -> None:
        # [FUNCTION DOC] HELPER: Callback for event record on a stream.
        self.syncs.record_state(event, stream)

    def _handle_event_wait(self, event: EventId, stream: StreamId) -> None:
        # [FUNCTION DOC] HELPER: Callback for a stream waiting on an event.
        self.syncs.stream_wait_for_event(stream, event)

    def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
        # [FUNCTION DOC] HELPER: Callback for allocation: drops any stale entry then registers the pointer with the current stack trace.
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # The stack trace generated in this way is in the inverse order, so it must be
        # reversed.
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )

    def _handle_memory_deallocation(self, data_ptr: DataPtr) -> None:
        # [FUNCTION DOC] HELPER: Callback for deallocation: ensures the pointer is known, then removes it.
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        self.tensors_accessed.delete_tensor(data_ptr)

    def _handle_stream_creation(self, stream: StreamId) -> None:
        # [FUNCTION DOC] HELPER: Callback for stream creation.
        self.syncs.create_stream(stream)

    def _handle_device_synchronization(self) -> None:
        # [FUNCTION DOC] HELPER: Callback for device-wide synchronization.
        self.syncs.sync_all_streams()

    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        # [FUNCTION DOC] HELPER: Callback for stream synchronization.
        self.syncs.all_streams_wait_for_stream(stream)

    def _handle_event_synchronization(self, event: EventId) -> None:
        # [FUNCTION DOC] HELPER: Callback for event synchronization.
        self.syncs.all_streams_wait_for_event(event)


def zip_by_key(a: dict[TK, TVa], b: dict[TK, TVb]) -> Iterator[tuple[TK, TVa, TVb]]:
    # [FUNCTION DOC] Yields (key, value from a, value from b) for each key of a that is also in b, in a's order.
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[torch.Argument, Any]]:
    # [FUNCTION DOC] Pairs schema arguments with call values: positional values with the leading schema arguments, then keyword values with
    # the schema arguments of the same name (unknown keywords are skipped). Yields (argument, value).
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    # [CLASS DOC] Collects the CUDA tensor data pointers touched by one operator call. Attributes: dataptrs_read, dataptrs_written,
    # tensor_aliases (pointer to list of schema argument names), outputs (pointers that are results).
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates empty read, write, alias and output collections.
        self.dataptrs_read: set[DataPtr] = set()
        self.dataptrs_written: set[DataPtr] = set()
        self.tensor_aliases: dict[DataPtr, list[str]] = {}
        self.outputs: set[DataPtr] = set()

    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        metadata_only: bool,
        name: str | None = None,
        is_output: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: Tree-map callback: for a CUDA tensor value, uses its data pointer (or object identity if the pointer is null) and adds
        # it to the written set, or to the read set unless metadata_only; records the argument name and output flag. Other
        # values are ignored.
        if isinstance(value, torch.Tensor) and value.is_cuda:
            # data_ptr() is preferred, but distinguish Tensors with null data_ptr()
            # otherwise two empty Tensors could incorrectly match as a conflict
            data_ptr = value.data_ptr() if value.data_ptr() else id(value)
            if is_write:
                self.dataptrs_written.add(data_ptr)
            elif not metadata_only:
                self.dataptrs_read.add(data_ptr)

            self.tensor_aliases.setdefault(data_ptr, [])
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            if is_output:
                self.outputs.add(data_ptr)

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        is_factory: bool,
    ) -> None:
        # [FUNCTION DOC] Records the CUDA tensors in the call arguments according to the schema: writes for arguments with write alias info;
        # metadata-only (not counted as read) for factory functions and non-writing aliases (views).
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            # A change is metadata only if it is a view or a factory function that
            # reads only metadata
            metadata_only = is_factory or (
                argument.alias_info is not None and not argument.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=is_write,
                    name=argument.name,
                    metadata_only=metadata_only,
                ),
                value,
            )

    def parse_outputs(
        self, schema: torch.FunctionSchema, outputs: Any, *, is_factory: bool
    ) -> None:
        # [FUNCTION DOC] Records the CUDA tensors in the operator result as outputs (written unless metadata-only).
        # KNOWN ISSUE: only the first schema return is paired with the whole result object, so multiple returns are not
        # treated per element by schema.
        for res, value in zip(schema.returns, (outputs,)):
            metadata_only = is_factory or (
                res.alias_info is not None and not res.alias_info.is_write
            )
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument,
                    is_write=not metadata_only,
                    is_output=True,
                    metadata_only=metadata_only,
                ),
                value,
            )


class CUDASanitizerDispatchMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that sanitizes every operator: wraps calls, extracts accessed tensors and forwards them to an EventHandler.
    # Construction activates GPU tracing and registers all handler callbacks. Attribute: event_handler.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the event handler, activates GPU tracing, and registers its callbacks for event, memory, stream and
        # synchronization events (global side effects).
        self.event_handler = EventHandler()
        torch._C._activate_gpu_trace()
        gpu_trace.register_callback_for_event_creation(
            self.event_handler._handle_event_creation
        )
        gpu_trace.register_callback_for_event_deletion(
            self.event_handler._handle_event_deletion
        )
        gpu_trace.register_callback_for_event_record(
            self.event_handler._handle_event_record
        )
        gpu_trace.register_callback_for_event_wait(
            self.event_handler._handle_event_wait
        )
        gpu_trace.register_callback_for_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        gpu_trace.register_callback_for_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        gpu_trace.register_callback_for_stream_creation(
            self.event_handler._handle_stream_creation
        )
        gpu_trace.register_callback_for_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        gpu_trace.register_callback_for_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        gpu_trace.register_callback_for_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Runs the operator and checks it: record_stream passes through unchecked. Otherwise parses inputs, calls func, parses
        # outputs, and submits a kernel launch on the current stream. If errors are found prints each to standard error and
        # raises CUDASanitizerErrors. Returns the operator result.
        if kwargs is None:
            kwargs = {}

        # record_stream is not a kernel dispatch, skip it
        if func is aten.record_stream.default:
            return func(*args, **kwargs)

        is_factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs, is_factory=is_factory)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(func._schema, outputs, is_factory=is_factory)
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)

        return outputs


class CUDASanitizer:
    # [CLASS DOC] Owns the dispatch mode object for the life of the program. Attributes: dispatch, enabled. Enabling enters the mode;
    # it is exited on disable or at object deletion.
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the dispatch mode (registering callbacks); not yet enabled.
        self.dispatch = CUDASanitizerDispatchMode()
        self.enabled = False

    def enable(self):
        # [FUNCTION DOC] Enters the dispatch mode and marks the sanitizer enabled (calling twice nests the mode).
        self.dispatch.__enter__()
        self.enabled = True

    def disable(self):
        # [FUNCTION DOC] Exits the dispatch mode and marks the sanitizer disabled.
        self.dispatch.__exit__(None, None, None)
        self.enabled = False

    def __del__(self):
        # [FUNCTION DOC] Disables the sanitizer on deletion if still enabled, unless the interpreter is shutting down.
        # Since this object lifetime is linked to the `torch.cuda._sanitizer` python
        # module, it often gets deleted as part of the overall `torch` module cleanup
        # At that time, depending on CPython version, the torch.* module might be in
        # different states of being already cleaned up.
        # Similarly other imports might already have been cleaned up so `sys` might
        # be already gone as well.
        # Skip exiting the mode if it outlived the runtime.
        if (sys is not None) and (not sys.is_finalizing()) and self.enabled:
            self.disable()


def enable_cuda_sanitizer():
    # [FUNCTION DOC] Enables the module-level sanitizer so subsequent CUDA operations are checked for races; errors are printed to standard error.
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    cuda_sanitizer.enable()


cuda_sanitizer = CUDASanitizer()
