# [MODULE DOC] Dynamo support for device streams and events. Defines custom ops in the
# "streams" namespace (fork, join, record_event, wait_event, synchronize_*, wait_stream,
# sync_dealloc, record_stream) that act on streams/events by index into the external object
# registry and are marked side-effecting; helpers that register graph-created streams/events;
# SymbolicStreamState tracking the stack of entered streams; and variable trackers
# (StreamContextVariable, StreamVariable and CUDA/XPU subclasses, EventVariable). Import time
# registers the custom ops and their fake implementations.
import collections
from collections.abc import Callable
from typing import Any, Optional

import torch
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import TupleVariable
from torch.fx import has_side_effect, Proxy

from .. import graph_break_hints
from ..bytecode_transformation import create_call_function
from ..exc import TYPE_CHECKING, unimplemented
from ..graph_bytecode_inputs import (
    CURRENT_STREAM_INDEX,
    get_external_object_by_index,
    register_graph_created_object,
    register_user_object,
    reset_user_object_tracking,
)
from ..source import CurrentStreamSource
from .base import GetSet, Method, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import FxTracebackAnnotateVariable
from .lazy import LazyVariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


def new_event(*args: Any, **kwargs: Any) -> int:
    # [FUNCTION DOC] Creates a real torch.Event from the arguments and registers it as a
    # graph-created object (with codegen that rebuilds it with no arguments). Returns its
    # registry index. Side effect: mutates the global object registry.
    event = torch.Event(*args, **kwargs)
    return register_graph_created_object(
        event,
        EventVariable.make_construct_in_graph_event_fn(
            TupleVariable([]), ConstDictVariable({})
        ),
    )


def new_stream(*args: tuple[Any], **kwargs: Any) -> int:
    # [FUNCTION DOC] Creates a real torch.Stream from the arguments and registers it as a
    # graph-created object (with codegen that rebuilds it with no arguments). Returns its
    # registry index. Side effect: mutates the global object registry.
    stream = torch.Stream(*args, **kwargs)  # type: ignore[no-matching-overload,call-overload]
    return register_graph_created_object(
        stream,
        StreamVariable.make_construct_in_graph_stream_fn(
            TupleVariable([]), ConstDictVariable({})
        ),
    )


def _codegen_current_stream(device: torch.device, cg: "PyCodegen") -> None:
    # [FUNCTION DOC] HELPER: Emits bytecode that fetches the device's current stream via its
    # source and passes it to the function that stashes graph-created objects.
    cg.add_push_null(
        lambda: cg.load_import_from(
            torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
            "stash_graph_created_object",
        )
    )
    cg(CurrentStreamSource(device))
    cg.extend_output(create_call_function(1, False))


def get_current_stream(device: torch.device) -> int:
    # [FUNCTION DOC] Looks up the accelerator's current stream for the device and registers it
    # as a graph-created object whose codegen re-reads the current stream. Returns the
    # registry index. Side effect: mutates the global object registry.
    stream = torch.accelerator.current_stream(device)
    return register_graph_created_object(
        stream, lambda _, cg: _codegen_current_stream(device, cg)
    )


def _get_stream_by_index(index: int) -> torch.Stream:
    # [FUNCTION DOC] HELPER: Returns the registry object at the index; raises an error if it
    # is not a torch.Stream.
    stream = get_external_object_by_index(index)
    if not isinstance(stream, torch.Stream):
        raise AssertionError(
            f"Fork/join stream expected a stream object at index {index}"
        )
    return stream


def _get_event_by_index(index: int) -> torch.Event:
    # [FUNCTION DOC] HELPER: Returns the registry object at the index; raises an error if it
    # is not a torch.Event.
    event = get_external_object_by_index(index)
    if not isinstance(event, torch.Event):
        raise AssertionError(
            f"Record/wait event expected an event object at index {index}"
        )
    return event


@custom_op("streams::fork", mutates_args=())
def fork_stream(
    from_index: int,  # kept to make stream transitions clearer
    to_index: int,
) -> None:
    # [FUNCTION DOC] Custom op streams::fork. Makes the stream at to_index the current
    # accelerator stream (from_index is unused, kept for readability). Side effect: changes
    # the current stream.
    torch.accelerator.set_stream(_get_stream_by_index(to_index))


@fork_stream.register_fake
def _(
    from_index: int,  # kept to make stream transitions clearer
    to_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::fork; does nothing.
    pass


has_side_effect(torch.ops.streams.fork.default)


@custom_op("streams::join", mutates_args=())
def join_stream(from_index: int, to_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::join. Makes the stream at to_index the current
    # accelerator stream (from_index is unused). Side effect: changes the current stream.
    torch.accelerator.set_stream(_get_stream_by_index(to_index))


@join_stream.register_fake
def _(
    from_index: int,
    to_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::join; does nothing.
    pass


has_side_effect(torch.ops.streams.join.default)


@custom_op("streams::record_event", mutates_args=())
def record_event(event_index: int, stream_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::record_event. Looks up the event and stream by registry
    # index (errors if of wrong type) and records the event on the stream.
    event = _get_event_by_index(event_index)
    stream = _get_stream_by_index(stream_index)
    event.record(stream)


@record_event.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::record_event; no-op.
    pass


has_side_effect(torch.ops.streams.record_event.default)


@custom_op("streams::wait_event", mutates_args=())
def wait_event(event_index: int, stream_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::wait_event. Looks up the event and stream by registry
    # index and makes the stream wait on the event.
    event = _get_event_by_index(event_index)
    stream = _get_stream_by_index(stream_index)
    event.wait(stream)


@wait_event.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::wait_event; no-op.
    pass


has_side_effect(torch.ops.streams.wait_event.default)


@custom_op("streams::synchronize_event", mutates_args=())
def synchronize_event(event_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::synchronize_event. Blocks the host until the event at
    # the registry index completes.
    event = _get_event_by_index(event_index)
    event.synchronize()


@synchronize_event.register_fake
def _(event_index: int) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::synchronize_event; no-op.
    pass


has_side_effect(torch.ops.streams.synchronize_event.default)


@custom_op("streams::synchronize_device", mutates_args=())
def synchronize_device(device_type: str, device_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::synchronize_device. Blocks the host until all work on
    # the device named by type and index completes.
    torch.accelerator.synchronize(torch.device(device_type, device_index))


@synchronize_device.register_fake
def _(device_type: str, device_index: int) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::synchronize_device; no-op.
    pass


has_side_effect(torch.ops.streams.synchronize_device.default)


@custom_op("streams::synchronize_stream", mutates_args=())
def synchronize_stream(stream_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::synchronize_stream. Blocks the host until the stream
    # at the registry index finishes its queued work.
    stream = _get_stream_by_index(stream_index)
    stream.synchronize()


@synchronize_stream.register_fake
def _(stream_index: int) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::synchronize_stream; no-op.
    pass


has_side_effect(torch.ops.streams.synchronize_stream.default)


@custom_op("streams::wait_stream", mutates_args=())
def wait_stream(waiting_stream_index: int, waited_on_stream_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::wait_stream. Looks up both streams by registry index
    # and makes the first wait for all work currently queued on the second.
    waiting = _get_stream_by_index(waiting_stream_index)
    waited_on = _get_stream_by_index(waited_on_stream_index)
    waiting.wait_stream(waited_on)


@wait_stream.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::wait_stream; no-op.
    # KNOWN ISSUE: parameter names (event_index, stream_index) differ from the real op's
    # (waiting_stream_index, waited_on_stream_index).
    pass


has_side_effect(torch.ops.streams.wait_stream.default)


@custom_op("streams::sync_dealloc", mutates_args=())
def sync_dealloc(
    wait_event_index: int, src_stream_index: int, to_dealloc: torch.Tensor
) -> None:
    # [FUNCTION DOC] Custom op streams::sync_dealloc. Runtime effect is only to invoke the
    # wait_event op with the given event and stream indices; the tensor argument exists so
    # the graph keeps its last use after the wait, delaying deallocation or reuse.
    """An op which waits on an event and moves the last usage of to_dealloc
    after the wait, so that after the sync occurs, the deallocation or
    subsequent reuse of the tensor's memory will be guaranteed to happen
    after a side stream is finished using it.
    See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
    for more details"""
    torch.ops.streams.wait_event.default(wait_event_index, src_stream_index)


@sync_dealloc.register_fake
def _(
    wait_event_index: int,
    src_stream_index: int,
    to_dealloc: torch.Tensor,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::sync_dealloc; no-op.
    pass


has_side_effect(torch.ops.streams.sync_dealloc.default)


@custom_op("streams::record_stream", mutates_args=())
def record_stream(tensor: torch.Tensor, stream_index: int) -> None:
    # [FUNCTION DOC] Custom op streams::record_stream. Marks the tensor as in use by the stream
    # at the registry index so the allocator delays reusing its memory.
    tensor.record_stream(_get_stream_by_index(stream_index))


@record_stream.register_fake
def _(
    tensor: torch.Tensor,
    stream_index: int,
) -> None:
    # [FUNCTION DOC] HELPER: Fake (tracing) implementation of streams::record_stream; no-op.
    pass


has_side_effect(torch.ops.streams.record_stream.default)


class SymbolicStreamState:
    # [CLASS DOC] Tracing-time state: cur_stream_stack is a deque of StreamVariables for the
    # streams entered so far. When an accelerator is available it starts holding a lazy
    # tracker of the ambient current stream (registry index 0). Methods push/pop streams and
    # query the current one.
    """Track the currently entered stream if any"""

    def __init__(self) -> None:
        # [FUNCTION DOC] If an accelerator is available: resets the user object registry,
        # registers the current stream (must land at the reserved current-stream index, else
        # raises an error) and seeds the stack with a lazy tracker for it, with the index set
        # without realizing. Otherwise the stack starts empty. Side effect: resets global
        # registry state.
        from ..source import CurrentStreamSource

        cur_stack: list[StreamVariable] = []
        if torch.accelerator.is_available():
            # Reset the registry so the current stream is guaranteed index 0.
            reset_user_object_tracking()
            stream = torch.accelerator.current_stream()
            source = CurrentStreamSource(stream.device)
            # Register the current stream so it gets index 0 (registry is
            # fresh at tracing start).  The inductor wrapper updates this
            # entry at runtime so cudagraph capture uses the capture stream
            # instead of this stale trace-time stream.
            index = register_user_object(stream, source)
            if index != CURRENT_STREAM_INDEX:
                raise AssertionError(
                    f"Current stream must be registered at index {CURRENT_STREAM_INDEX}, "
                    f"got {index}"
                )
            stream_var = LazyVariableTracker.create(stream, source=source)
            # Set user_object_index as an instance attribute so accessing it
            # does NOT trigger LazyVariableTracker realization.
            stream_var.user_object_index = index  # type: ignore[union-attr]
            cur_stack = [stream_var]  # type: ignore[list-item]

        self.cur_stream_stack: collections.deque[StreamVariable] = collections.deque(
            cur_stack
        )

    def enter_stream(self, stream: "StreamVariable") -> None:
        # [FUNCTION DOC] Pushes a stream onto the stack of entered streams.
        self.cur_stream_stack.append(stream)

    def exit_stream(self) -> None:
        # [FUNCTION DOC] Pops the most recently entered stream; fails if the stack is empty.
        self.cur_stream_stack.pop()

    def cur_stream(self, device: torch.device | None = None) -> "StreamVariable":
        # [FUNCTION DOC] Returns the innermost entered stream on the given device if one exists;
        # otherwise (or with no device) the top of the stack. Fails if the stack is empty.
        if device is not None:
            for stream in reversed(self.cur_stream_stack):
                if stream.device == device:
                    return stream

        return self.cur_stream_stack[-1]

    def in_stream_context(self) -> bool:
        # [FUNCTION DOC] Returns True if any stream is on the stack.
        return len(self.cur_stream_stack) > 0

    def cur_stream_id(self) -> int:
        # [FUNCTION DOC] Returns the Python id of the real stream object on top of the stack,
        # peeking at the underlying value of an unrealized lazy tracker instead of realizing it.
        """Get a Python object id for the current stream without realizing lazy variables."""
        stream = self.cur_stream_stack[-1]
        if isinstance(stream, LazyVariableTracker) and not stream.is_realized():
            return id(stream.peek_value())
        return id(stream.value)


class StreamContextVariable(FxTracebackAnnotateVariable):
    # [CLASS DOC] Tracker for a stream context manager (with-stream block). Wraps the stream to
    # enter in the stream attribute; entering and leaving push/pop it on the symbolic stream
    # state and emit a traceback-annotation context with the stream's registry index.
    # Supports graph breaks inside the context.
    """This represents torch.cuda.StreamContext"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        stream_to_enter: "StreamVariable",
        **kwargs: dict[str, Any],
    ) -> "StreamContextVariable":
        # [FUNCTION DOC] Factory: builds a context tracker for the given stream, forwarding the
        # extra keyword arguments.
        return StreamContextVariable(
            stream_to_enter,
            **kwargs,
        )

    def __init__(self, stream: Optional["StreamVariable"], **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the stream, then initializes the annotation context manager with
        # an annotation mapping the key stream to the stream's registry index.
        self.stream = stream
        super().__init__(
            annotation={"stream": self.get_stream().user_object_index},
            initial_values=None,
            **kwargs,
        )

    def enter(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Pushes this context's stream on the symbolic stream stack, then runs the
        # base enter. Returns the resulting tracker.
        # to stream, from stream is the order of the arguments
        # we are entering the target, and leaving the initial stream
        tx.symbolic_stream_state.enter_stream(self.get_stream())
        return super().enter(tx)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Pops the current stream from the symbolic stream stack, then runs the
        # base exit. Returns the resulting tracker.
        # to stream, from stream is the order of the arguments
        # we are leaving the target, and entering the initial stream
        tx.symbolic_stream_state.exit_stream()
        return super().exit(tx, *args)

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the CUDA stream context class.
        return torch.cuda.StreamContext

    def supports_graph_breaks(self) -> bool:
        # [FUNCTION DOC] Always True: graph breaks inside this context are supported.
        return True

    def get_stream(self) -> "StreamVariable":
        # [FUNCTION DOC] Returns the stream to enter; raises an error if none was provided.
        if not self.stream:
            raise AssertionError("Stream context should have a separate stream")
        return self.stream


class StreamVariable(StreamContextVariable):
    # [CLASS DOC] Tracker for a real torch.Stream. State: proxy (graph proxy), value (the real
    # stream), device, and user_object_index (registry slot used to pass the stream into the
    # graph; may be None for legacy global-based reconstruction). Acts as its own context
    # (get_stream returns itself). Methods wait_event, wait_stream, synchronize, query and
    # record_event emit streams custom ops or proxies. Device subclasses set the handle name.
    """Represents the device-agnostic torch.Stream class"""

    _cpython_type: type = torch.Stream  # pyrefly: ignore [bad-override]
    # Subclasses set this to the device-specific handle attribute name
    # (e.g. "cuda_stream", "sycl_queue").  None means no special handling.
    _device_handle_attr: str | None = None

    def __init__(
        self,
        proxy: Proxy,
        value: torch.Stream,
        user_object_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the proxy, real stream, its device and registry index. If the
        # proxy carries an example value that differs from the stream, raises an error. Then
        # initializes the base context with no inner stream.
        # Index into the user object table
        # used to pass arbitrary objects to the graph
        if proxy is not None and "example_value" in proxy.node.meta:
            if proxy.node.meta["example_value"] != value:
                raise AssertionError(
                    f"proxy example_value {proxy.node.meta['example_value']} != {value}"
                )

        self.proxy = proxy
        self.value = value
        self.device = value.device

        self.user_object_index = user_object_index
        super().__init__(None, **kwargs)

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the class-level Python type (device-specific for subclasses).
        return self._cpython_type

    def _stream_device_handle_get(
        self: "StreamVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the device-specific handle attribute (for example
        # cuda_stream). Raises an error if the subclass defines no attribute name. Guards the
        # stream source for equality, then returns the handle as a constant: from the named
        # attribute, else from native_handle, else None (unsupported).
        from ..guards import GuardBuilder, install_guard

        name = self._device_handle_attr
        if name is None:
            raise AssertionError(
                "StreamVariable subclass must define _device_handle_attr"
            )
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))
        if hasattr(self.value, name):
            return ConstantVariable.create(getattr(self.value, name))
        if hasattr(self.value, "native_handle"):
            return ConstantVariable.create(self.value.native_handle)
        return None

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Returns the real stream object.
        return self.value

    def wait_event(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] stream.wait_event(event): requires the first arg to be an EventVariable
        # (else raises an error). Emits a wait_event graph op for the event and this stream.
        # Returns None constant.
        event_arg = args[0]
        if not isinstance(event_arg, EventVariable):
            raise AssertionError(f"Expected EventVariable, got {type(event_arg)}")
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.wait_event,
            (event_arg.user_object_index, self.user_object_index),
            {},
        )
        return ConstantVariable.create(None)

    def wait_stream(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] stream.wait_stream(other): requires a StreamVariable (else raises an
        # error). Emits a wait_stream graph op making this stream wait on the other. Returns
        # None constant.
        other_stream = args[0]
        if not isinstance(other_stream, StreamVariable):
            raise AssertionError(f"Expected StreamVariable, got {type(other_stream)}")
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.wait_stream,
            (self.user_object_index, other_stream.user_object_index),
            {},
        )
        return ConstantVariable.create(None)

    def synchronize(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] stream.synchronize(): emits a synchronize_stream graph op for this
        # stream. Returns None constant.
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.synchronize_stream,
            (self.user_object_index,),
            {},
        )
        return ConstantVariable.create(None)

    def query(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] stream.query(): emits a method-call graph node on the stream proxy and
        # wraps the result as a constant-typed tracker (completion status).
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            target_cls=ConstantVariable,
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_method", "query", *proxy_args_kwargs([self] + args, kwargs)
            ),
        )

    def record_event(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] stream.record_event(event=None): checks no input mutation preceded the
        # record. Uses the given EventVariable, or creates and registers a new real event.
        # Emits a record_event graph op and returns a tracker for a graph node that fetches the
        # event from the registry.
        from .builder import wrap_fx_proxy

        tx.output.check_event_record_after_input_mutation(id(self.value))
        if args and isinstance(args[0], EventVariable):
            event_var = args[0]
            event = event_var.value
            event_index = event_var.user_object_index
        else:
            event = self.value.record_event()
            event_index = register_graph_created_object(
                event,
                EventVariable.make_construct_in_graph_event_fn(
                    TupleVariable([]), ConstDictVariable({})
                ),
            )
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.record_event,
            (event_index, self.user_object_index),
            {},
        )
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                get_external_object_by_index,
                (event_index,),
                {},
            ),
        )

    tp_methods = {
        "wait_event": Method(wait_event),
        "wait_stream": Method(wait_stream),
        "synchronize": Method(synchronize),
        "query": Method(query),
        "record_event": Method(record_event),
    }

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Rich comparison of streams. A non-stream operand compares unequal
        # (True only for the not-equal op). Otherwise guards both sources for equality and
        # returns a constant from comparing the real streams with the operator.
        from ..guards import GuardBuilder, install_guard
        from ..utils import cmp_name_to_op_mapping
        from .constant import ConstantVariable

        if not isinstance(other, StreamVariable):
            # Stream's tp_richcompare (THPStream_richcompare) compares
            # stream_id/device and never returns NotImplemented.
            return ConstantVariable.create(op == "__ne__")
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))
        if other.source:
            install_guard(other.source.make_guard(GuardBuilder.EQUALS_MATCH))
        op_fn = cmp_name_to_op_mapping[op]
        return VariableTracker.build(
            tx,
            op_fn(self.value, other.value),  # pyrefly: ignore[bad-argument-type]
        )

    def as_proxy(self) -> Proxy:
        # [FUNCTION DOC] Returns the graph proxy for the stream.
        return self.proxy

    def module_name(self) -> str:
        # [FUNCTION DOC] Returns the module name of the stream class.
        return "torch._C"

    def fn_name(self) -> str:
        # [FUNCTION DOC] Returns the class name used for the stream.
        return "Stream"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode producing the real stream: raises an error if a source
        # exists. With a registry index, emits a registry lookup call; otherwise installs the
        # stream as a global and loads it (legacy path).
        # If we got here, this stream is fully subsumed by the graph - this means it is
        # not an input or global
        if self.source:
            raise AssertionError("Stream should not have a source during reconstruct")
        if self.user_object_index is not None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,
                    "get_external_object_by_index",
                )
            )
            codegen.append_output(codegen.create_load_const(self.user_object_index))
            codegen.extend_output(create_call_function(1, False))
        else:
            # This will support the legacy behavior
            prefix = f"_stream_{self.device}"
            name = codegen.tx.output.install_global_by_id(prefix, self.value)
            codegen.append_output(codegen.create_load_global(name, add=True))

    def get_stream(self) -> "StreamVariable":
        # [FUNCTION DOC] Returns self: a stream is its own context stream.
        return self

    @staticmethod
    def make_construct_in_graph_stream_fn(
        args: TupleVariable, kwargs: ConstDictVariable
    ) -> Callable[[int, "PyCodegen"], None]:
        # [FUNCTION DOC] Returns a codegen callback that, given a registry index and codegen,
        # emits bytecode building a stream from the args/kwargs trackers and stashing it as a
        # graph-created object.
        def fn(index: int, codegen: "PyCodegen") -> None:
            # [FUNCTION DOC] HELPER: Emits the stash-call wrapping the stream build call.
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
                    "stash_graph_created_object",
                )
            )
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.utils.__name__, "build_stream"
                )
            )
            codegen(args)
            codegen(kwargs)
            codegen.extend_output(create_call_function(2, False))
            codegen.extend_output(create_call_function(1, False))

        return fn


class CudaStreamVariable(StreamVariable):
    # [CLASS DOC] StreamVariable for torch.cuda.Stream; exposes the cuda_stream handle attribute.
    """Represents torch.cuda.Stream, preserving device-specific type and attributes."""

    _cpython_type = torch.cuda.Stream
    _device_handle_attr = "cuda_stream"

    tp_getset = {
        "cuda_stream": GetSet(StreamVariable._stream_device_handle_get, None),
    }


class XpuStreamVariable(StreamVariable):
    # [CLASS DOC] StreamVariable for torch.xpu.Stream; exposes the sycl_queue handle attribute.
    """Represents torch.xpu.Stream, preserving device-specific type and attributes."""

    _cpython_type = torch.xpu.Stream
    _device_handle_attr = "sycl_queue"

    tp_getset = {
        "sycl_queue": GetSet(StreamVariable._stream_device_handle_get, None),
    }


_stream_fn_to_variable_cls: dict[object, type[StreamVariable]] = {
    torch.cuda.current_stream: CudaStreamVariable,
    torch.xpu.current_stream: XpuStreamVariable,
}


def _get_stream_variable_cls(stream_fn: object) -> type[StreamVariable] | None:
    # [FUNCTION DOC] HELPER: Maps a current-stream function (CUDA or XPU) to its stream tracker
    # class, or None if unknown.
    return _stream_fn_to_variable_cls.get(stream_fn)


class EventVariable(VariableTracker):
    # [CLASS DOC] Tracker for a real torch.Event. State: proxy, value (the real event) and
    # user_object_index (registry slot, or None for legacy global reconstruction). Supports
    # only wait, record, synchronize and query, which emit streams custom ops or proxies;
    # other methods graph-break.
    def __init__(
        self,
        proxy: Proxy,
        value: torch.Event,
        user_object_index: int | None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Raises an error if the proxy's example value differs from the event;
        # otherwise initializes the base tracker and stores proxy, event and registry index.
        if proxy is not None and "example_value" in proxy.node.meta:
            if proxy.node.meta["example_value"] != value:
                raise AssertionError(
                    f"proxy example_value {proxy.node.meta['example_value']} != {value}"
                )
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison; delegates to the generic object comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the concrete class of the real event.
        return type(self.value)

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Returns the real event object.
        return self.value

    def wait(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] event.wait(stream=None): resolves the stream (argument or current),
        # emits a wait_event graph op for this event and that stream. Returns None constant.
        _, stream_index = EventVariable._get_stream_arg(tx, args, kwargs)
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.wait_event,
            (
                self.user_object_index,
                stream_index,
            ),
            {},
        )
        return ConstantVariable.create(None)

    def record(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] event.record(stream=None): resolves the stream, checks no input mutation
        # preceded the record, and emits a record_event graph op. Returns None constant.
        stream_arg, stream_index = EventVariable._get_stream_arg(tx, args, kwargs)
        tx.output.check_event_record_after_input_mutation(id(stream_arg.value))
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.record_event,
            (
                self.user_object_index,
                stream_index,
            ),
            {},
        )
        return ConstantVariable.create(None)

    def synchronize(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] event.synchronize(): emits a synchronize_event graph op for this event.
        # Returns None constant.
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.synchronize_event,
            (self.user_object_index,),
            {},
        )
        return ConstantVariable.create(None)

    def query(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] event.query(): emits a method-call graph node on the event proxy and
        # wraps the result as a constant-typed tracker (completion status).
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            target_cls=ConstantVariable,
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_method", "query", *proxy_args_kwargs([self] + args, kwargs)
            ),
        )

    tp_methods = {
        "wait": Method(wait),
        "record": Method(record),
        "synchronize": Method(synchronize),
        "query": Method(query),
    }

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Dispatches supported methods (wait, record, synchronize, query) through
        # the base implementation; any other name triggers a graph break with an
        # event-specific message.
        # Event supports only the tp_methods above; every other name (including
        # dunders) graph-breaks with an event-specific message rather than the
        # generic "Unsupported method call".
        if name in self.tp_methods:
            return super().call_method(tx, name, args, kwargs)
        method_name = (
            f"{type(self.value).__module__}.{type(self.value).__qualname__}.{name}"
        )
        unimplemented(
            gb_type="Unsupported event method",
            context=str(name),
            explanation=f"Dynamo doesn't support tracing the {method_name} method. "
            f"We currently support wait, record, synchronize, and query.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def as_proxy(self) -> Proxy:
        # [FUNCTION DOC] Returns the graph proxy for the event.
        return self.proxy

    @staticmethod
    def _get_stream_arg(
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> tuple["StreamVariable", int]:
        # [FUNCTION DOC] HELPER: Resolves the stream for an event method: first positional arg,
        # else the stream keyword, else the current symbolic stream. Returns the stream tracker
        # and its registry index.
        """Returns (stream_variable, stream_index_for_op).

        The ambient current stream is registered at index 0 in the external
        object registry.  The inductor wrapper updates index 0 at runtime so
        that cudagraph capture sees the capture stream, not the stale
        trace-time default stream.
        """
        stream_arg = None
        if args:
            stream_arg = args[0]
        elif kwargs:
            stream_arg = kwargs.get("stream")

        if not stream_arg:
            stream_var = tx.symbolic_stream_state.cur_stream()
            return stream_var, stream_var.user_object_index  # type: ignore[return-value]

        return stream_arg, stream_arg.user_object_index  # type: ignore[return-value]

    @staticmethod
    def make_construct_in_graph_event_fn(
        args: TupleVariable, kwargs: ConstDictVariable
    ) -> Callable[[int, "PyCodegen"], None]:
        # [FUNCTION DOC] Returns a codegen callback that, given a registry index and codegen,
        # emits bytecode building an event from the args/kwargs trackers and stashing it as a
        # graph-created object.
        def fn(index: int, codegen: "PyCodegen") -> None:
            # [FUNCTION DOC] HELPER: Emits the stash-call wrapping the event build call.
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
                    "stash_graph_created_object",
                )
            )
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.utils.__name__, "build_event"
                )
            )
            codegen(args)
            codegen(kwargs)
            codegen.extend_output(create_call_function(2, False))
            codegen.extend_output(create_call_function(1, False))

        return fn

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode producing the real event: raises an error if a source
        # exists. With a registry index, emits a registry lookup call; otherwise installs the
        # event as a global and loads it.
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        if self.source:
            raise AssertionError("Event should not have a source during reconstruct")
        if self.user_object_index is not None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,
                    "get_external_object_by_index",
                )
            )
            codegen.append_output(codegen.create_load_const(self.user_object_index))
            codegen.extend_output(create_call_function(1, False))
        else:
            prefix = "_event"
            name = codegen.tx.output.install_global_by_id(prefix, self.value)
            codegen.append_output(codegen.create_load_global(name, add=True))
