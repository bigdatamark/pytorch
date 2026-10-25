# [MODULE DOC]
# Ahead-of-time (AOT) compilation of fullgraph Dynamo captures. Provides
# CompileArtifacts (everything needed to rerun a compiled function), custom
# pickler/unpickler that handle code objects, closures, modules, bound and
# nested functions with external data references, AOTCompiledFunction (guarded,
# serializable callable), AOTCompiledModel (dispatch over several compiled
# inputs), and the drivers aot_compile_fullgraph and aot_compile_module.
# No import-time side effects beyond creating the module logger.
import dataclasses
import importlib
import inspect
import io
import logging
import os
import pickle
import tempfile
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.convert_frame import GraphRuntimeEnv
from torch._dynamo.graph_utils import _graph_device_type
from torch._dynamo.package import SystemInfo

from . import convert_frame
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SerializedCode, SourceInfo


log = logging.getLogger(__name__)


def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    # [FUNCTION DOC] Bind args and kwargs to the signature, apply default
    # values, and return the mapping of parameter names to values. Raises
    # TypeError if the arguments do not match the signature.
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


@dataclass
class CompileArtifacts:
    # [CLASS DOC]
    # Dataclass bundling everything needed to run and serialize an AOT compiled
    # function: the original signature, guard manager (or serialized guard
    # state), backend id and name, the serializable compiled callable, original
    # code object, runtime environment (bytecode and closure), source info,
    # device type, and the SystemInfo of the machine that compiled it
    # (defaults to the current system).
    signature: inspect.Signature
    guard_manager: Optional["GuardManagerWrapper"]
    guards_state: bytes
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    runtime_env: GraphRuntimeEnv
    source_info: "SourceInfo"
    device_type: str
    backend_name: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    def check_compatibility(self) -> None:
        # [FUNCTION DOC] Compare the saved system info with the current
        # system for this device type; raises if they are incompatible.
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


class AOTCompilePickler(pickle.Pickler):
    # [CLASS DOC]
    # Pickler writing to a buffer that (a) replaces objects listed in
    # external_data by persistent ids (their key), (b) records any
    # torch.nn.Module encountered in errors (by object id) because modules must
    # be passed as external data, and (c) provides custom reductions for
    # closure cells, code objects, modules, bound methods and nested
    # (local) functions. Callers must check errors after dumping.
    def __init__(self, external_data: dict[str, object], buf: io.BytesIO) -> None:
        # [FUNCTION DOC] Initialize on buf; build a map from object identity to
        # key for external_data; start with an empty errors mapping.
        super().__init__(buf)
        self.external_data = external_data
        self.id_map: dict[int, str] = {
            id(value): key for key, value in external_data.items()
        }
        self.errors = {}

    def persistent_id(self, obj: object) -> int | str | None:
        # [FUNCTION DOC] Return the external-data key if obj is one of the
        # external objects; for any nn.Module not listed, record it in errors
        # and return its id; otherwise return None (pickle normally).
        if id(obj) in self.id_map:
            return self.id_map[id(obj)]
        elif isinstance(obj, torch.nn.Module):
            self.errors[id(obj)] = obj
            return id(obj)
        else:
            return None

    @classmethod
    def _unpickle_cell(cls, val: object) -> object:
        # [FUNCTION DOC] HELPER: Rebuild a closure cell holding val by creating
        # a closure and returning its cell.
        def _() -> object:
            return val

        if _.__closure__ is None:
            raise AssertionError("closure must not be None")
        return _.__closure__[0]

    @classmethod
    # pyrefly: ignore [implicit-any]
    def _unpickle_bound_method(cls, func: Callable, base: object) -> types.MethodType:
        # [FUNCTION DOC] HELPER: Bind func to base as a method object.
        return types.MethodType(func, base)

    @classmethod
    def _unpickle_module(cls, name: str) -> types.ModuleType:
        # [FUNCTION DOC] HELPER: Import and return the module called name.
        return importlib.import_module(name)

    @classmethod
    def _unpickle_code(cls, serialized_code: "SerializedCode") -> types.CodeType:
        # [FUNCTION DOC] HELPER: Convert a SerializedCode back into a code
        # object.
        from torch._dynamo.package import SerializedCode

        return SerializedCode.to_code_object(serialized_code)

    @classmethod
    def _unpickle_nested_function(
        cls,
        code: types.CodeType,
        module: str,
        qualname: str,
        argdefs: tuple[object, ...] | None,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        # [FUNCTION DOC] HELPER: Recreate a nested function from its code,
        # using the globals of the imported named module, with the given
        # qualified name, defaults and closure.
        f_globals = importlib.import_module(module).__dict__
        return types.FunctionType(code, f_globals, qualname, argdefs, closure)

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        # [FUNCTION DOC] Custom reduction hook. Closure cells are saved by
        # contents; code objects via SerializedCode; modules by name; bound
        # methods as (underlying function, self) when the function differs from
        # the one found through the instance; functions whose qualified name
        # contains the local marker as code plus module, name, defaults and
        # closure. Returns NotImplemented to fall back to default pickling.
        if isinstance(obj, type((lambda x: lambda: x)(0).__closure__[0])):  # type: ignore[index] # noqa: PLC3002
            return type(self)._unpickle_cell, (obj.cell_contents,)
        elif inspect.iscode(obj):
            from torch._dynamo.package import SerializedCode

            return type(self)._unpickle_code, (SerializedCode.from_code_object(obj),)

        elif inspect.ismodule(obj):
            return type(self)._unpickle_module, (obj.__name__,)
        elif inspect.ismethod(obj):
            """
            By default, pickle will call getattr() directly on the self object
            for pickling bounded methods, this is not what we want, instead we
            always want to serialize the original function and the self object
            in their original form.
            """
            func = obj.__func__
            method_self = obj.__self__
            inner_func = getattr(method_self, func.__name__)
            if inspect.ismethod(inner_func):
                inner_func = inner_func.__func__
            if func is not inner_func:
                return type(self)._unpickle_bound_method, (func, method_self)
        elif inspect.isfunction(obj):
            if "<locals>" in obj.__qualname__:
                return type(self)._unpickle_nested_function, (
                    obj.__code__,
                    obj.__module__,
                    obj.__qualname__,
                    obj.__defaults__,
                    obj.__closure__,
                )

        return NotImplemented


class AOTCompileUnpickler(pickle.Unpickler):
    # [CLASS DOC]
    # Unpickler counterpart of AOTCompilePickler: resolves persistent ids by
    # looking them up in the supplied external_data mapping.
    def __init__(self, external_data: dict[str, object], file: io.BytesIO) -> object:
        # [FUNCTION DOC] Initialize on the file and store external_data.
        # KNOWN ISSUE: the return annotation says object, but an initializer
        # returns nothing (should be None).
        super().__init__(file)
        self.external_data = external_data

    def persistent_load(self, key: str) -> object:
        # [FUNCTION DOC] Return the external object for key; raises
        # RuntimeError (with a hint to pass external data) if key is missing.
        if key not in self.external_data:
            raise RuntimeError(
                f"Missing required external reference to data: {key}. "
                "Please load AOT compiled function with "
                "`external_data=<external data dictionary>`"
                f"{self.external_data}"
            )
        return self.external_data[key]


@dataclass
class AOTCompileSaveResult:
    # [CLASS DOC]
    # Dataclass holding the bytes produced by serializing an AOT compiled
    # function.
    serialized_data: bytes


def atomic_write_binary(file_path: str, data: bytes):
    # [FUNCTION DOC] Write data to file_path atomically: write and fsync a
    # temporary file in the same directory (current directory if none given),
    # then rename it over file_path. Side effects: file system writes. The
    # temporary file is not removed if writing fails.
    dir_name = os.path.dirname(file_path) or "."

    with tempfile.NamedTemporaryFile(
        dir=dir_name, delete=False, mode="wb"
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())

    os.replace(temp_path, file_path)


@dataclass
class AOTCompiledFunction:
    # [CLASS DOC]
    # A compiled, guarded, serializable function. Holds the CompileArtifacts,
    # a flag for whether guards are checked on call, and optional extra
    # globals for the forward callable. After construction, fn is the callable
    # rebuilt from the runtime environment, and the guard manager is loaded
    # from saved guard state if not already present. Calling it checks guards
    # (raising RuntimeError with the failure reason) and then runs fn.
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True
    _extra_globals: dict[str, object] | None = None

    def prepare_f_locals(self, *args: object, **kwargs: object) -> dict[str, object]:
        # [FUNCTION DOC] Build the mapping of local names used for guard
        # evaluation: closure variables (by free-variable name, requiring the
        # closure length to match; AssertionError otherwise) updated with the
        # arguments bound to the saved signature (defaults applied).
        f_locals: dict[str, object] = {}
        env = self._artifacts.runtime_env
        if env.closure:
            if not env.bytecode.co_freevars or len(env.closure) != len(
                env.bytecode.co_freevars
            ):
                raise AssertionError("closure length must match co_freevars length")
            f_locals = {
                name: cell.cell_contents
                for name, cell in zip(env.bytecode.co_freevars, env.closure)
            }
        f_locals.update(bind_locals(self._artifacts.signature, *args, **kwargs))
        return f_locals

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        # [FUNCTION DOC] Return whether the guards pass for these arguments.
        # Raises AssertionError if no guard manager is loaded.
        f_locals = self.prepare_f_locals(*args, **kwargs)
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Verify system compatibility (may raise), build the
        # forward callable fn from the runtime environment, backend id and
        # compiled callable, and if no guard manager exists, load one from the
        # saved guard state using the original code and fn's globals.
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        self.fn = self._artifacts.runtime_env.forward_callable(
            self._artifacts.backend_id,
            self._artifacts.compiled_fn,
            extra_globals=self._extra_globals,
        )

        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                self.fn.__globals__,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] If guard checking is enabled and guards fail, raise
        # RuntimeError with the verbose failure reason; otherwise call fn with
        # the arguments and return its result.
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
            f_locals = self.prepare_f_locals(*args, **kwargs)
            reason = str(self._artifacts.guard_manager.check_verbose(f_locals))
            raise RuntimeError(f"GuardManager check failed, reason: {reason}")
        return self.fn(*args, **kwargs)

    def source_info(self) -> "SourceInfo":
        # [FUNCTION DOC] Return the source info recorded at compile time.
        return self._artifacts.source_info

    def save_compiled_function(
        self, path: str, external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        # [FUNCTION DOC] Serialize this function (with optional external_data
        # for objects that cannot be pickled) and atomically write the bytes to
        # path. Returns the save result. Side effect: file write.
        result = type(self).serialize(self, external_data)
        atomic_write_binary(path, result.serialized_data)
        return result

    @classmethod
    def serialize(
        cls, fn: "AOTCompiledFunction", external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        # [FUNCTION DOC] Serialize fn's artifacts to bytes: drop the guard
        # manager (guard state is kept), convert bytecode and original code to
        # serializable form, replace the compiled callable by a pair of its
        # class's deserializer and its serialized bytes, and pickle with
        # AOTCompilePickler using external_data. Raises RuntimeError if any
        # nn.Module was reached that is not in external_data.
        from torch._dynamo.package import SerializedCode

        state = fn._artifacts.__dict__.copy()
        state["guard_manager"] = None
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.from_code_object(state["runtime_env"].bytecode),
        )
        compiled_fn = state["compiled_fn"]
        state["compiled_fn"] = (
            type(compiled_fn).deserialize_compile_artifacts,
            type(compiled_fn).serialize_compile_artifacts(compiled_fn),
        )
        state["original_code"] = SerializedCode.from_code_object(state["original_code"])
        buf = io.BytesIO()
        pickler = AOTCompilePickler(external_data or {}, buf)
        pickler.dump(state)
        if pickler.errors:
            raise RuntimeError(
                f"Failed to serialize the following objects: {list(pickler.errors.values())}\n"
                "Please mark these as external data by using `external_data={'key': ...}`"
            )
        return AOTCompileSaveResult(serialized_data=buf.getvalue())

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        f_globals: dict[str, object] | None = None,
        external_closure_data: dict[str, Any] | None = None,
    ) -> "AOTCompiledFunction":
        # [FUNCTION DOC] Inverse of serialize. Unpickle data resolving external
        # references from external_closure_data (RuntimeError if missing),
        # restore code objects, rebuild the compiled callable with the
        # inductor autograd-for-AOT option enabled temporarily, and construct
        # an instance whose forward callable uses f_globals as extra globals.
        from torch._dynamo.package import SerializedCode

        f = io.BytesIO(data)
        f.seek(0)
        unpickler = AOTCompileUnpickler(external_closure_data or {}, f)
        state = unpickler.load()
        f.close()
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.to_code_object(state["runtime_env"].bytecode),
        )
        deserializer, compiled_fn_state = state["compiled_fn"]
        with torch._inductor.config.patch(enable_autograd_for_aot=True):
            state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])

        artifacts = CompileArtifacts(**state)
        return cls(artifacts, _extra_globals=f_globals)

    def disable_guard_check(self) -> None:
        # [FUNCTION DOC] Turn off guard checking on calls (mutates this object).
        self._guard_check_enabled = False


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
    dynamic: bool | None = None,
) -> AOTCompiledFunction:
    # [FUNCTION DOC]
    # Capture model (a function or bound method) as a single full graph with
    # Dynamo for the example args/kwargs, compile the graph with backend, and
    # package the result as an AOTCompiledFunction. Optionally sets dynamic
    # shapes mode. If hooks has no guard filter, installs one dropping global
    # guards and guard types unsupported for serialization (mutates hooks).
    # Inductor/AOTAutograd-style backends have their output wrapped in
    # BundledAOTAutogradSerializableCallable. Builds guards with saving
    # enabled. Raises RuntimeError if the compiled result is not a
    # SerializableCallable, and AssertionError on internal inconsistencies.
    # Runs under metrics context and several strict autograd cache configs.
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )
    from torch._guards import TracingContext

    args, kwargs = example_inputs

    dynamic_ctx = nullcontext()
    if dynamic is not None:
        from torch._dynamo.eval_frame import set_enable_dynamic

        dynamic_ctx = set_enable_dynamic(dynamic)

    with (
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
        torch._functorch.config.patch(strict_autograd_cache=True),
        dynamic_ctx,
        torch_function_mode_stack_state_mgr,
    ):
        capture_output = convert_frame.fullgraph_capture(model, args, kwargs)
        graph_capture_output = capture_output.graph_capture_output
        if graph_capture_output.output_graph is None:
            raise AssertionError("output_graph must not be None")

        if not hooks.guard_filter_fn:
            from torch._dynamo.types import GuardFilterEntry

            def new_guard_filter_fn(
                guard_entries: Sequence[GuardFilterEntry],
            ) -> Sequence[bool]:
                # [FUNCTION DOC] HELPER: Keep (true) every guard that is not
                # global and whose type is serializable; drop the rest.
                return [
                    (
                        not (
                            g.is_global
                            or g.guard_type
                            in CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
                        )
                    )
                    for g in guard_entries
                ]

            hooks.guard_filter_fn = new_guard_filter_fn

        fn, _ = convert_frame.get_traced_fn(model)

        backend_input = capture_output.backend_input
        if backend_input is None:
            raise AssertionError("backend_input must not be None")
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        device_type = _graph_device_type(backend_input.graph_module.graph)
        if (
            backend_input.fake_mode.shape_env
            is not graph_capture_output.output_graph.shape_env
        ):
            raise AssertionError(
                "fake_mode.shape_env must be the same as output_graph.shape_env"
            )
        tracing_context = TracingContext(backend_input.fake_mode)
        tracing_context.tensor_to_context = backend_input.tensor_to_context
        with (
            torch._guards.tracing(tracing_context),
            torch._functorch.config.patch(
                {
                    "strict_autograd_cache": True,
                    "bypass_autograd_cache_key": True,
                    "bundled_autograd_cache": True,
                    "force_non_lazy_backward_lowering": True,
                    "force_autograd_cache": True,
                }
            ),
        ):
            compiled_fn = backend(
                backend_input.graph_module, backend_input.example_inputs
            )
            # If Inductor backend or AOTAutograd-based backend is used,
            # wrap the compiled_fn for serialization.
            # TODO: this should be replaced once we make the backend return the SerializableCallable directly.
            if (
                isinstance(backend, torch._TorchCompileInductorWrapper)
                or (
                    hasattr(backend, "compiler_fn")
                    and isinstance(
                        backend.compiler_fn, torch._dynamo.backends.common.AotAutograd
                    )
                )
                or (
                    hasattr(compiled_fn, "serialize")
                    and compiled_fn.serialize is not None
                )
            ):
                compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn)

        if not isinstance(compiled_fn, SerializableCallable):
            if hasattr(backend, "compiler_fn"):
                compiler_fn = backend.compiler_fn
            else:
                compiler_fn = backend
            raise RuntimeError(
                f"Compiled function type {type(compiled_fn)} (produced "
                + f"from backend {compiler_fn}) does not implement SerializableCallable."
            )

        # Temporarily restore the mode stack so guard expressions that
        # reference modes can evaluate, matching the compile_inner path.
        build_guards_ctx = ExitStack()
        if torch_function_mode_stack_state_mgr.stack:
            build_guards_ctx.enter_context(
                torch_function_mode_stack_state_mgr.temp_restore_stack()
            )
        with build_guards_ctx:
            check_fn = graph_capture_output.build_guards(
                fn.__code__, hooks=hooks, save=True, strict_error=True
            )

        if check_fn.guards_state is None:
            raise AssertionError("guards_state must not be None")

        source_info = SourceInfo(inlined_sources=set())
        for traced_code in graph_capture_output.traced_code:
            source_info.add_code(traced_code)

        artifacts = CompileArtifacts(
            signature=convert_frame._get_signature(fn),
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            runtime_env=graph_capture_output.get_runtime_env(),
            source_info=source_info,
            device_type=device_type,
            backend_name=getattr(backend, "compiler_name", "unknown"),
        )
        aot_compiled_fn = AOTCompiledFunction(
            _artifacts=artifacts, _extra_globals=fn.__globals__
        )

    return aot_compiled_fn


@dataclass
class ModelInput:
    # [CLASS DOC]
    # Dataclass describing one example input to compile a model for: positional
    # args, keyword args, and a list of context managers to enter while
    # compiling.
    """
    WIP type: represents a single model input
    Which consists of a tuple of arguments and a set of contexts in which to run the model.

    For each ModelInput, we'll compile one full graph of the model, and then use the guards generated
    to dispatch between the compiled graphs.


    """

    args: tuple[Any]
    kwargs: dict[str, Any]
    contexts: list[AbstractContextManager[Any]]


@dataclass
class AOTCompiledModel:
    # [CLASS DOC]
    # Dataclass for a module's forward with several compiled variants: the
    # model and a list of AOTCompiledFunction. Calling dispatches to the first
    # variant whose guards pass; if none pass, the first one runs and raises
    # its guard failure. Only the compiled results are serialized, so the
    # model must be supplied again on load.
    # Represents a single forward function of a model along with dispatch
    # compiled_results is serializable. We require the model to deserialize again.
    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Run the first compiled result whose guards accept the
        # model and arguments; otherwise run the first result to surface its
        # guard failure RuntimeError.
        for result in self.compiled_results:
            if result.guard_check(self.model, *args, **kwargs):
                return result(self.model, *args, **kwargs)
        # All guards failed, just run one of them and throw the guard check error.
        return self.compiled_results[0](self.model, *args, **kwargs)

    def serialize(self) -> bytes:
        # [FUNCTION DOC] Serialize every compiled result and pickle the list of
        # byte strings. No external data is supported here.
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result).serialized_data)
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, model: torch.nn.Module, data: bytes) -> "AOTCompiledModel":
        # [FUNCTION DOC] Unpickle the list of serialized results and
        # deserialize each inside a fresh compile context and metrics context;
        # returns an AOTCompiledModel bound to model.
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        results: list[bytes] = pickle.loads(data)
        compiled_results = []
        for result in results:
            with (
                compile_context(CompileContext(convert_frame.get_compile_id({}))),
                get_metrics_context(),
            ):
                compiled_results.append(AOTCompiledFunction.deserialize(result))
        return cls(model, compiled_results)


def aot_compile_module(
    model: torch.nn.Module,
    inputs: list[ModelInput],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledModel:
    # [FUNCTION DOC]
    # Compile model.forward once per ModelInput (entering each input's
    # contexts) via aot_compile_fullgraph with the given hooks and backend,
    # and return an AOTCompiledModel. Raises AssertionError if inputs is empty.
    # Logs each input being compiled.
    """
    Compiles a single nn.Module with any number of inputs, and returns a compiled forward function.
    """

    def compile_single_graph(model_input: ModelInput) -> AOTCompiledFunction:
        # [FUNCTION DOC] HELPER: Enter the input's contexts and compile the
        # model's current forward with its args and kwargs as a full graph.
        example_inputs = (model_input.args, model_input.kwargs)
        orig_forward = model.forward
        with ExitStack() as stack:
            for ctx in model_input.contexts:
                stack.enter_context(ctx)
            return aot_compile_fullgraph(
                orig_forward,
                example_inputs,
                hooks=hooks,
                backend=backend,
            )

    # pyrefly: ignore [implicit-any]
    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    if len(compiled_results) == 0:
        raise AssertionError("Expected at least one compiled result")

    return AOTCompiledModel(model, compiled_results)
