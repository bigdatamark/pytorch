# [MODULE DOC] Infrastructure for serialized/out-of-process FX graph compilation: wire-protocol input and output containers, serializers for
# Virtualized, lowering and fake tensor mode state, log capture and replay, and the _SerializedFxCompile, _OutOfProcessFxCompile and debug compile classes.
# Import-time side effect: imports async_compile.
from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import queue
import sys
import tempfile
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypeGuard
from typing_extensions import final, override, Self

import torch._inductor.async_compile
import torch.fx
from torch._inductor.codecache import BypassFxGraphCache, FxGraphCache
from torch._inductor.metrics import CachedMetricsDeltas, CachedMetricsHelper
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    CompiledFxGraphConstantsWithGm,
    OutputCode,
)
from torch._inductor.utils import patch_subprocess_env
from torch._subclasses import FakeTensorMode
from torch.utils._ordered_set import OrderedSet

from . import config
from .compile_fx import _CompileFxKwargs, _InProcessFxCompile, FxCompile, log
from .debug import DebugContext
from .graph import GraphLowering
from .virtualized import V


if TYPE_CHECKING:
    import types
    from collections.abc import Generator, Mapping, Sequence
    from concurrent.futures import Future

    from torch._inductor.utils import InputType
    from torch.fx import GraphModule


def _graph_contains_triton_kernel_wrappers(gm: GraphModule) -> bool:
    # [FUNCTION DOC] HELPER: Return True if any (sub)GraphModule in gm has a node targeting a triton kernel wrapper higher-order op
    # (functional or mutation variant), since those depend on process-local state and cannot be sent to another process.
    """
    Check if the graph contains triton kernel wrapper nodes. These nodes contain
    references to the kernel_side_table which is process-local and can't be
    serialized across processes.
    """
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
        triton_kernel_wrapper_mutation,
    )

    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.target in (
                triton_kernel_wrapper_functional,
                triton_kernel_wrapper_mutation,
            ):
                return True
    return False


@dataclass
class _VirtualizedSerializer:
    # [CLASS DOC] HELPER: Dataclass snapshot of the serializable parts of the Virtualized global V (aot compilation, choices,
    # local buffer context, ops, kernel, current node). Field names must match attributes of V.
    """
    This handles the data for serializing Virtualized.
    """

    # The values here get serialized. We don't grab everything because some of
    # the fields can't be serialized.
    aot_compilation: Any = None
    choices: Any = None
    local_buffer_context: Any = None
    ops: Any = None
    kernel: Any = None
    current_node: Any = None

    @classmethod
    def serialize(cls) -> _VirtualizedSerializer:
        # [FUNCTION DOC] Build a snapshot by reading each field's attribute from the current V.
        """
        Turn the current state of torch._inductor.virtualized.V into a
        serializable structure.
        """
        kwargs = {}
        for f in dataclasses.fields(cls):
            kwargs[f.name] = getattr(V, f.name)
        return _VirtualizedSerializer(**kwargs)

    def patch(self) -> _VirtualizedSerializerContextManager:
        # [FUNCTION DOC] Return a context manager that installs the saved values into V; see the context manager class.
        """
        Returns a context manager which patches the saved values into the
        current environment. While patched, any value not listed above will be
        poisoned so that reads will raise an error.
        """
        return _VirtualizedSerializerContextManager(self)


class _VirtualizedSerializerContextManager(contextlib.ExitStack):
    # [CLASS DOC] HELPER: ExitStack that, on entry, sets every V handler: handlers with a saved field get that value, all others are set to
    # the poisoned placeholder so unexpected reads raise. Restores on exit.
    """
    Helper for _VirtualizedSerializer.patch()
    """

    def __init__(self, virtualized: _VirtualizedSerializer) -> None:
        # [FUNCTION DOC] HELPER: Store the snapshot.
        super().__init__()
        self.virtualized = virtualized

    @override
    def __enter__(self) -> Self:
        # [FUNCTION DOC] Enter the stack, then for every setter on V (names starting with set_, minus an optional handler suffix) enter its context
        # with the saved value or the poison placeholder. Returns self.
        super().__enter__()

        for set_name in dir(V):
            if not set_name.startswith("set_"):
                continue
            name = set_name[4:]
            name = name.removesuffix("_handler")
            set_handler = getattr(V, set_name)
            if hasattr(self.virtualized, name):
                value = getattr(self.virtualized, name)
            else:
                # poison any values that we don't serialize so that any
                # unset accesses are caught.
                value = torch._inductor.virtualized._PoisonedVirtual
            self.enter_context(set_handler(value))

        return self


def _is_fallback_handler(op: object) -> bool:
    # [FUNCTION DOC] HELPER: Return the op's fallback-handler marker, or False if it has none.
    try:
        return op._is_fallback_handler  # type: ignore[attr-defined]
    except AttributeError:
        return False


class _LoweringSerializer:
    # [CLASS DOC] HELPER: Captures which lowerings were overridden with forced fallbacks (names kept in an ordered set) so a child process can
    # reproduce them. Not a full copy of the lowering table (see comment about the limitation).
    """
    This handles the data for serializing lowering.lowering
    """

    # A full implementation would make sure that all lowerings are copied over
    # (or at least detected and raise a bypass when a non-standard lowering is
    # used). For now we just handle tests by looking for lowerings that were
    # overridden with a forced fallback.
    fallbacks: OrderedSet[str]

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Record the string names of all lowerings whose handler is a fallback handler.
        from . import lowering

        self.fallbacks = OrderedSet(
            str(k) for k, v in lowering.lowerings.items() if _is_fallback_handler(v)
        )

    def patch(self) -> _LoweringSerializerContextManager:
        # [FUNCTION DOC] Return a context manager that reapplies the recorded fallbacks.
        return _LoweringSerializerContextManager(self)


class _LoweringSerializerContextManager(contextlib.ExitStack):
    # [CLASS DOC] HELPER: ExitStack that forces fallback for each recorded lowering not already a fallback in the current process.
    """
    Helper for _LoweringSerializer.patch()
    """

    def __init__(self, lowering: _LoweringSerializer) -> None:
        # [FUNCTION DOC] HELPER: Store the serializer.
        super().__init__()
        self.lowering = lowering

    @override
    def __enter__(self) -> Self:
        # [FUNCTION DOC] Enter the stack and force fallback on every lowering named in the recorded set that is not already a fallback. Returns self.
        super().__enter__()

        from . import lowering

        for k, v in lowering.lowerings.items():
            name = str(k)
            if name in self.lowering.fallbacks:
                if not _is_fallback_handler(v):
                    self.enter_context(lowering.force_fallback(k))  # type: ignore[arg-type]

        return self


@dataclass
class _FakeTensorModeSerializer:
    # [CLASS DOC] HELPER: Carries the fake tensor mode settings to transfer: the allow_non_fake_inputs flag and the shape environment.
    allow_non_fake_inputs: bool

    def __init__(self, fake_mode: FakeTensorMode) -> None:
        # [FUNCTION DOC] HELPER: Copy the flag and shape environment from the given fake mode.
        self.allow_non_fake_inputs = fake_mode.allow_non_fake_inputs
        self.shape_env = fake_mode.shape_env

    @contextlib.contextmanager
    def patch(self, fake_mode: FakeTensorMode) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager: temporarily set the fake mode's allow_non_fake_inputs to the saved value and restore the old value afterwards.
        # KNOWN ISSUE: the restore is not in a finally block, so an exception inside the with body leaves the changed value in place.
        saved_allow_non_fake_inputs = fake_mode.allow_non_fake_inputs
        fake_mode.allow_non_fake_inputs = self.allow_non_fake_inputs

        yield

        fake_mode.allow_non_fake_inputs = saved_allow_non_fake_inputs


@dataclass
class _WireProtocolInput:
    # [CLASS DOC] HELPER: Everything sent from parent to child for an out-of-process compile: graph module, example inputs, inputs to check,
    # graph kwargs, tracing context, portable config, Virtualized snapshot, optional test determinism guard, logger state, lowering and fake mode snapshots.
    """
    For _SerializedFxCompile - encapsulates all the data being transferred
    (sent) from the parent to the child.
    """

    gm: torch.fx.GraphModule
    example_inputs: Sequence[InputType]
    inputs_to_check: Sequence[int]
    graph_kwargs: _CompileFxKwargs
    tracing_context: torch._guards.TracingContext | None
    config: dict[str, object]
    virtualized: _VirtualizedSerializer
    deterministic_guard_for_testing: (  # type: ignore[name-defined]  # mypy bug
        torch.testing._internal.common_utils.DeterministicGuard | None
    )
    logger_state: _LoggerState
    lowering: _LoweringSerializer
    fake_tensor_mode: _FakeTensorModeSerializer

    def serialize(self) -> _WireProtocolPickledInput:
        # [FUNCTION DOC] Pickle this object with the graph-aware pickler and return it wrapped as a _WireProtocolPickledInput.
        """
        Turns this object into a _WireProtocolPickledInput which can be
        directly transferred across a stream.
        """
        from torch.fx._graph_pickler import GraphPickler

        return _WireProtocolPickledInput(GraphPickler.dumps(self))


def _current_fake_mode() -> FakeTensorMode:
    # [FUNCTION DOC] HELPER: Return the fake mode of the active tracing context if one exists, otherwise a new FakeTensorMode with a fresh ShapeEnv.
    fake_mode = None
    if context := torch._guards.TracingContext.try_get():
        fake_mode = context.fake_mode
    if fake_mode is not None:
        return fake_mode

    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    return FakeTensorMode(shape_env=shape_env)


@dataclass
class _WireProtocolPickledInput:
    # [CLASS DOC] HELPER: Wrapper around the serialized bytes of a _WireProtocolInput.
    value: bytes

    def deserialize(self) -> _WireProtocolInput:
        # [FUNCTION DOC] Unpickle using the current fake mode and return the _WireProtocolInput; raises AssertionError on a wrong type.
        """
        Turn this streamable object back into a _WireProtocolInput.
        """
        from torch.fx._graph_pickler import GraphPickler

        fake_mode = _current_fake_mode()
        result = GraphPickler.loads(self.value, fake_mode)
        if not isinstance(result, _WireProtocolInput):
            raise AssertionError(f"expected _WireProtocolInput, got {type(result)}")
        return result


@dataclass
class _WireProtocolOutput:
    # [CLASS DOC] HELPER: Everything returned from child to parent: the output code, metrics deltas, captured log records,
    # recorded warnings (or None) and the shape environment.
    """
    For _SerializedFxCompile - encapsulates all the data being transferred
    (returned) back from the child to the parent.
    """

    graph: OutputCode
    metrics: CachedMetricsDeltas
    logs: list[logging.LogRecord]
    warning_replay: list[warnings.WarningMessage] | None
    shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv | None

    def serialize(self) -> _WireProtocolPickledOutput:
        # [FUNCTION DOC] Prepare a compiled graph for serialization if applicable, pickle this object and return it wrapped as a _WireProtocolPickledOutput.
        """
        Turns this object into a _WireProtocolPickledOutput which can be
        directly transferred across a stream.
        """
        from torch.fx._graph_pickler import GraphPickler

        if isinstance(self.graph, CompiledFxGraph):
            self.graph.prepare_for_serialization()
        return _WireProtocolPickledOutput(GraphPickler.dumps(self))


@dataclass
class _WireProtocolPickledOutput:
    # [CLASS DOC] HELPER: Wrapper around the serialized bytes of a _WireProtocolOutput.
    value: bytes

    def deserialize(self, constants: CompiledFxGraphConstants) -> _WireProtocolOutput:
        # [FUNCTION DOC] Unpickle using the current fake mode, checking the type (AssertionError otherwise), and for compiled graphs
        # run the post-deserialization hook with the given constants. Returns the _WireProtocolOutput.
        """
        Turn this streamable object back into a _WireProtocolOutput.
        """
        from torch.fx._graph_pickler import GraphPickler

        fake_mode = _current_fake_mode()
        result = GraphPickler.loads(self.value, fake_mode)
        if not isinstance(result, _WireProtocolOutput):
            raise AssertionError(f"expected _WireProtocolOutput, got {type(result)}")
        if isinstance(result.graph, CompiledFxGraph):
            result.graph.after_deserialization(constants)
        return result


class _LoggerState:
    # [CLASS DOC] HELPER: Context manager that records the levels of non-propagating torch._inductor loggers and, while entered, attaches queue handlers
    # to capture their records for replay in the parent. Entering returns a _CapturedLogs object.
    """
    This class is for tracking logging that happens during an out-of-process
    compile so we can "replay" those messages when the compile is done. Used as
    a context manager which returns the captured logs (object).
    """

    loggers: dict[str, int]
    # The actual log capturing mechanism - this should be None when we're not
    # actively capturing logs.
    captured_logs: _CapturedLogs | None = None

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Walk the torch._inductor logger tree (using the logger registry under a lock on older Pythons, child listing on 3.12+)
        # and store the name and level of every real logger that does not propagate to a parent.
        # Mapping from logger name to level.
        self.loggers = {}

        def filter(
            # [FUNCTION DOC] HELPER: Type guard accepting only real loggers under torch._inductor that do not propagate; placeholders and propagating loggers are rejected.
            logger: logging.Logger | logging.PlaceHolder,
        ) -> TypeGuard[logging.Logger]:
            if not isinstance(logger, logging.Logger):
                # Assume that Placeholders propagate
                return False
            # We only want to track torch._inductor logging
            if not logger.name.startswith("torch._inductor"):
                return False
            # If this logger propagates then assume we'll track its parent
            if logger.propagate:
                return False
            return True

        root = logging.getLogger("torch._inductor")
        if sys.version_info < (3, 12):
            # logging.getChildren() doesn't exist until 3.12
            logging._acquireLock()  # type: ignore[attr-defined]
            try:
                for logger in root.manager.loggerDict.values():
                    if filter(logger):
                        self.loggers[logger.name] = logger.level
            finally:
                logging._releaseLock()  # type: ignore[attr-defined]
        else:
            q = [root]
            while q:
                logger = q.pop()
                if filter(logger):
                    self.loggers[logger.name] = logger.level
                q.extend(logger.getChildren())

    def __enter__(self) -> _CapturedLogs:
        # [FUNCTION DOC] Start capturing: create and apply a _CapturedLogs, returning it. Raises AssertionError if already capturing.
        if self.captured_logs is not None:
            raise AssertionError("captured_logs already set on __enter__")
        self.captured_logs = _CapturedLogs(self)
        self.captured_logs.apply()
        return self.captured_logs

    def __exit__(
        # [FUNCTION DOC] Remove the capture handlers; raises AssertionError if not capturing.
        # KNOWN ISSUE: captured_logs is not reset to None afterwards, so entering the same instance again would raise.
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.captured_logs is None:
            raise AssertionError("captured_logs not set on __exit__")
        self.captured_logs.remove()


class _CapturedLogs:
    # [CLASS DOC] HELPER: Holds the unbounded queue of captured log records and the installed handlers by logger name (None when not applied).
    """
    Helper for _LoggerState - this class actually attaches to the logger in
    the child process and grabs the log messages themselves.
    """

    state: _LoggerState
    queue: queue.Queue[logging.LogRecord]
    handlers: dict[str, logging.Handler] | None

    def __init__(self, state: _LoggerState) -> None:
        # [FUNCTION DOC] HELPER: Store the state and create an empty queue with no handlers.
        self.state = state
        # A queue of the log entries
        # TODO: For memory purposes should we log to a file and then respond with that?
        self.queue = queue.Queue(-1)
        # Mapping from name to handler (only valid when applied)
        self.handlers = None

    def finish(self) -> list[logging.LogRecord]:
        # [FUNCTION DOC] Drain the queue and return all captured records as a list; AssertionError if handlers are still applied.
        if self.handlers is not None:
            raise AssertionError("expected handlers to be None in finish")
        logs = []
        try:
            while True:
                logs.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return logs

    def remove(self) -> None:
        # [FUNCTION DOC] Remove the queue handlers from their loggers; AssertionError if not applied.
        if self.handlers is None:
            raise AssertionError("expected handlers to be set in remove")
        handlers, self.handlers = self.handlers, None
        for name, handler in handlers.items():
            logger = logging.getLogger(name)
            logger.removeHandler(handler)

    def apply(self) -> None:
        # [FUNCTION DOC] Attach a queue handler to every tracked logger and restore its recorded level (unless unset); AssertionError if already applied.
        from logging.handlers import QueueHandler

        if self.handlers is not None:
            raise AssertionError("expected handlers to be None in apply")
        self.handlers = {}
        for name, level in self.state.loggers.items():
            logger = logging.getLogger(name)
            handler = QueueHandler(self.queue)
            self.handlers[name] = handler
            logger.addHandler(handler)
            if level != logging.NOTSET:
                logger.setLevel(level)


class _SerializedFxCompile(FxCompile):
    # [CLASS DOC] FxCompile base class for compiles that cross a serialization boundary: pickle the inputs, run the compile in a child,
    # unpickle the output. Subclasses implement _send_to_child; falls back to in-process compile when serialization is impossible.
    """
    This is used to represent an FxCompile which occurs across a serialized
    boundary.
    """

    @override
    def codegen_and_compile(
        # [FUNCTION DOC] Serialize the compile request (falling back to an in-process compile if not possible), send it to the child, deserialize the
        # result with the constants, tag compiled graphs with the region name, post-process, count the compile in stats and return the output code.
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        # If this code changes it's likely _AsyncFxCompile.codegen_and_compile()
        # will also need to match.

        serialized = self.serialize_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )
        if not serialized:
            eager_compile = _InProcessFxCompile()
            eager_compile.compile_region_name = self.compile_region_name
            return eager_compile.codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )

        inputs, constants = serialized
        output = self._send_to_child(inputs).deserialize(constants)
        if isinstance(output.graph, CompiledFxGraph):
            output.graph.compile_region_name = self.compile_region_name

        self._postprocess(output)
        self._compile_stats[type(self)].codegen_and_compile += 1

        # TODO: Do we need to figure out what changed in TracingContext in the
        # child and plumb that back up to the parent?

        return output.graph

    def serialize_compile(
        # [FUNCTION DOC] Build the pickled input and constants holder for a compile. Returns None (caller should compile in-process) if the graph has
        # uncacheable higher-order ops, triton kernel wrapper nodes, or if pickling fails (logged as a warning). Captures the tracing context,
        # portable config, Virtualized, logging, lowering and fake mode state, and the test determinism guard when available.
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> tuple[_WireProtocolPickledInput, CompiledFxGraphConstantsWithGm] | None:
        """
        Prepare a _WireProtocolInput to compile. If None is returned then it
        wasn't possible to serialize and we should fallback to in-process.
        """
        try:
            # _check_for_hop raises BypassFxGraphCache when it detects something
            # we can't cache (or serialize)
            FxGraphCache._check_for_hop(gm)
        except BypassFxGraphCache as e:
            log.debug("Skipping %s compile: %s", type(self), e)
            return None

        # Triton kernel wrapper nodes contain references to the kernel_side_table
        # which is process-local and can't be serialized across processes.
        if _graph_contains_triton_kernel_wrappers(gm):
            log.debug(
                "Skipping %s compile: graph contains triton kernel wrappers", type(self)
            )
            return None

        context = torch._guards.TracingContext.try_get()
        constants = CompiledFxGraphConstantsWithGm(gm)
        logger_state = _LoggerState()
        lowering = _LoweringSerializer()

        # If we're running tests then grab the DeterministicGuard (don't want to
        # import this if it isn't already imported because it has side-effects)
        deterministic_guard_for_testing: (  # type: ignore[name-defined]  # mypy bug
            torch.testing._internal.common_utils.DeterministicGuard | None
        ) = None
        try:
            deterministic_guard_for_testing = (
                torch.testing._internal.common_utils.DeterministicGuard._current_state()  # type: ignore[attr-defined]  # mypy bug
            )
        except AttributeError:
            pass

        fake_mode = _current_fake_mode()
        fake_tensor_mode = _FakeTensorModeSerializer(fake_mode)

        from pickle import PicklingError

        try:
            input = _WireProtocolInput(
                gm,
                example_inputs,
                inputs_to_check,
                graph_kwargs,
                context,
                config.save_config_portable(),
                _VirtualizedSerializer.serialize(),
                deterministic_guard_for_testing,
                logger_state,
                lowering,
                fake_tensor_mode,
            ).serialize()
            return (input, constants)
        except (AttributeError, BypassFxGraphCache, PicklingError):
            # For example: AttributeError: Can't pickle local object
            # 'make_opaque_unary_fn.<locals>.OpaqueUnaryFn'

            # TODO: scuba record about not being able to do this?
            log.warning("Unable to pickle input graph or example inputs", exc_info=True)

            return None

    @abstractmethod
    def _send_to_child(
        # [FUNCTION DOC] HELPER: Abstract; transfer the pickled input to the child, run the child entry point and return the pickled output.
        self, pickled_input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
        # The implementation of this should transfer `input` to the child, call
        # `_run_in_child(input)` and transfer the result back.
        ...

    def _postprocess(self, output: _WireProtocolOutput) -> None:
        # [FUNCTION DOC] HELPER: Hook run on the deserialized output in the parent; the base implementation does nothing.
        pass

    @classmethod
    def _run_in_child(
        # [FUNCTION DOC] Child-side entry point. Under an exit stack: patch subprocess environment, record warnings, deserialize the input, apply its
        # Virtualized, lowering, config, logger capture, determinism guard, tracing context, debug context and fake mode settings, then do an in-process
        # compile. Returns the pickled output with metrics deltas, captured logs, warnings and shape environment.
        cls,
        pickled_input: _WireProtocolPickledInput,
        extra_env: Mapping[str, str | None] | None = None,
    ) -> _WireProtocolPickledOutput:
        metrics = CachedMetricsHelper()

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch_subprocess_env(extra_env))

            # Save warnings to "replay" in the parent
            warning_replay = stack.enter_context(warnings.catch_warnings(record=True))

            # TODO: Should we split the input into multiple sections where each
            # section sets up state for the previous section? (i.e. a Config section
            # which we decode and apply, followed by a FakeTensorMode section which
            # we decode and apply, etc)
            input = pickled_input.deserialize()

            stack.enter_context(input.virtualized.patch())
            stack.enter_context(input.lowering.patch())
            stack.enter_context(config.patch(input.config))
            captured_logs = stack.enter_context(input.logger_state)
            if input.deterministic_guard_for_testing:
                stack.enter_context(input.deterministic_guard_for_testing)
            stack.enter_context(torch._guards.tracing(input.tracing_context))
            stack.enter_context(DebugContext())

            fake_mode = _current_fake_mode()
            stack.enter_context(input.fake_tensor_mode.patch(fake_mode))

            output_graph = _InProcessFxCompile().codegen_and_compile(
                input.gm,
                input.example_inputs,
                input.inputs_to_check,
                input.graph_kwargs,
            )

        logs = captured_logs.finish()

        return _WireProtocolOutput(
            output_graph,
            metrics.get_deltas(),
            logs,
            warning_replay,
            fake_mode.shape_env,
        ).serialize()


# This is a debugging/testing implementation of FxCompile which serializes the
# input and output but still runs the FxCompile in-process.
@final
class _DebugSerdeFxCompile(_SerializedFxCompile):
    # [CLASS DOC] HELPER: Debug/test compile that serializes input and output but runs the compile in the same process.
    @override
    def _send_to_child(
        # [FUNCTION DOC] HELPER: Run the child entry point directly in this process.
        self, pickled_input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
        # For debugging just serde the input and output but don't run in a
        # subprocess.
        return self._run_in_child(pickled_input)


class _OutOfProcessFxCompile(_SerializedFxCompile):
    # [CLASS DOC] HELPER: Base for compiles run outside the current process; adds an asynchronous send hook and metrics/log/warning replay in post-processing.
    """
    Represents an FxCompile which is run outside the current process (in
    either a subprocess or possibly even a separate machine).
    """

    @override
    @final
    def _send_to_child(
        # [FUNCTION DOC] HELPER: Send asynchronously and block on the future for its result. Final.
        self, pickled_input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
        f = self._send_to_child_async(pickled_input)

        # For debugging: If we want to print status updates...
        # last = time.time()
        # while not f.done():
        #     print("tick...")
        #     time.sleep(0.125)
        #     now = time.time()
        #     if now - last > 1:
        #         last = now

        return f.result()

    @abstractmethod
    def _send_to_child_async(
        # [FUNCTION DOC] HELPER: Abstract; send the pickled input to the child and return a future of the pickled output.
        self, pickled_input: _WireProtocolPickledInput
    ) -> Future[_WireProtocolPickledOutput]: ...

    def _postprocess(self, output: _WireProtocolOutput) -> None:
        # [FUNCTION DOC] HELPER: Apply the child's metrics deltas, forward the output source code to the test output-saving hook if set,
        # replay recorded warnings and re-emit captured log records through their loggers in the parent.
        # Since our metrics were gathered in a subprocess make sure to add them
        # here.
        CachedMetricsHelper.apply_deltas(output.metrics)

        # This is used by tests to check the output for specific details.  For
        # remote things (subproc and RE) we need to do the `save_output_code`
        # here since it didn't happen earlier in-process. In the future if this
        # doesn't have "source_code" (it's a CompiledAOTI, for example) and we
        # need it we'll have to grab it and serialize it separately from the
        # child.
        if GraphLowering.save_output_code is not None:
            GraphLowering.save_output_code(output.graph.source_code)  # type: ignore[attr-defined]

        # And forward our collected logs. The cache is cleared when the outer
        # function exits.
        @functools.cache
        def getLogger(name: str) -> logging.Logger:
            # [FUNCTION DOC] HELPER: Cached logger lookup by name.
            return logging.getLogger(name)

        if output.warning_replay:
            for w in output.warning_replay:
                warnings.warn_explicit(
                    message=w.message,
                    category=w.category,
                    filename=w.filename,
                    lineno=w.lineno,
                    source=w.source,
                )

        for record in output.logs:
            logger = getLogger(record.name)
            logger.handle(record)


# For debugging - create a _FxCompile which writes the serialized data to a file
# and then exits.
#
# TODO: make this a FxCompileMode value?
#
# The "child runner" should look something like this:
#
#     import torch
#     from torch._inductor import compile_fx
#     idx = 0
#     with open(f"/tmp/pytorch_compile_fx_tmp_input_{idx}.bin", "rb") as f:
#         input = compile_fx._WireProtocolPickledInput(f.read())
#     result = compile_fx._SubprocessFxCompile._run_in_child(input)
#     with open(f"/tmp/pytorch_compile_fx_tmp_output_{idx}.bin", "wb") as f:
#         f.write(result.value)
#
@final
class _DebugFileFxCompile(_SerializedFxCompile):
    # [CLASS DOC] HELPER: Debugging compile that writes the pickled input to a numbered temp file, prints its path and terminates the process.
    # Counter of files is a class attribute.
    file_index = 0

    @override
    def _send_to_child(
        # [FUNCTION DOC] HELPER: Write the input bytes to a temp file and print its path, then exit the whole process immediately
        # (the alternative branches that run or read back a result are disabled).
        self, pickled_input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
        idx = _DebugFileFxCompile.file_index
        _DebugFileFxCompile.file_index += 1

        name = os.path.join(
            tempfile.gettempdir(), f"pytorch_compile_fx_tmp_input_{idx}.bin"
        )
        with open(name, "wb") as f:
            f.write(pickled_input.value)
        print(f"Wrote to {name}")

        if False:
            name = os.path.join(
                tempfile.gettempdir(), f"pytorch_compile_fx_tmp_actual_{idx}.bin"
            )
            actual = self._run_in_child(pickled_input)
            with open(name, "wb") as f:
                f.write(actual.value)
            return actual
        elif False:
            name = os.path.join(
                tempfile.gettempdir(), f"pytorch_compile_fx_tmp_output_{idx}.bin"
            )
            with open(name, "rb") as f:
                result = _WireProtocolPickledOutput(f.read())
                print(f"Read from {name}")
            return result
        else:
            os._exit(-1)
