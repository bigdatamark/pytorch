# [MODULE DOC]
# Compile package infrastructure for precompile: CompilePackage records the
# code objects, guards, resume functions and backend ids produced by Dynamo,
# serializes them (SerializedCode, _DynamoCacheEntry, PrecompileCacheEntry),
# and reinstalls them (globals, precompile cache entries) in a new process.
# Also defines DynamoStore implementations (in-memory, disk) and the global
# DynamoCache disk store. Import-time side effects: registers
# PrecompileCacheArtifact with the cache artifact factory, creates the global
# code cache and the DynamoCache object (which resolves the cache directory).
"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""

import abc
import ast
import contextlib
import dataclasses
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import os
import pickle
import platform
import shutil
import sys
import types
from collections.abc import Callable, Generator, Iterator
from contextlib import nullcontext
from typing import Any, NewType, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import torch
from torch._dynamo.exc import PackageError
from torch._dynamo.graph_utils import _graph_device_type
from torch.utils.weak import WeakIdKeyDictionary

from .bytecode_transformation import (
    COMPILED_FN_PREFIX,
    get_code_keys,
    is_compiled_fn_name,
)
from .utils import CleanupHook, counters, dynamo_timed, increment_frame


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper, GuardsState


_CODE_CACHE = WeakIdKeyDictionary()


def _code_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Decorator for code conversion class methods that
    # memoizes results in the global weak-keyed code cache, keyed by the input
    # code object. Returns the wrapping function.
    def _(
        cls: type[Any], code: Union["SerializedCode", types.CodeType]
    ) -> Union["SerializedCode", types.CodeType]:
        # [FUNCTION DOC] HELPER: Return the cached conversion for code, else
        # compute via the wrapped function, store in the global cache and return it.
        if code in _CODE_CACHE:
            return _CODE_CACHE[code]
        res = fn(cls, code)
        _CODE_CACHE[code] = res
        return res

    return _


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    # [CLASS DOC] Frozen, picklable mirror of a code object: one field per code
    # attribute returned by get_code_keys (bytes, names, constants, line tables).
    # Nested code constants are themselves SerializedCode. Converted both ways
    # by from_code_object and to_code_object.
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = None
    co_qualname: str | None = None
    co_exceptiontable: bytes | None = None
    co_lnotab: str | None = None

    @classmethod
    @_code_cache
    def from_code_object(cls, code: types.CodeType) -> "SerializedCode":
        # [FUNCTION DOC] Convert a code object into a SerializedCode by copying
        # each attribute, recursively converting nested code constants. Memoized per code object.
        kwargs = {key: getattr(code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.from_code_object(c) if isinstance(c, types.CodeType) else c
            for c in kwargs["co_consts"]
        )
        return cls(**kwargs)

    @classmethod
    @_code_cache
    def to_code_object(cls, serialized_code: "SerializedCode") -> types.CodeType:
        # [FUNCTION DOC] Rebuild a real code object from a SerializedCode,
        # recursively converting nested serialized constants. Memoized per input.
        kwargs = {key: getattr(serialized_code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.to_code_object(c) if isinstance(c, SerializedCode) else c
            for c in kwargs["co_consts"]
        )
        return types.CodeType(
            *kwargs.values(),
        )


@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    # [CLASS DOC] HELPER: Serializable record of one Dynamo compilation: pickled
    # guards state bytes plus the generated Dynamo bytecode as SerializedCode.
    """
    Contains the serializable information associated with a single compilation in dynamo.
    To restore an execution of compiled code, we will need to serialize the following data:
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """

    guards_state: bytes
    dynamo_code: SerializedCode


def load_guards_state(guards_state: bytes) -> Any:
    # [FUNCTION DOC] Unpickle serialized guards state. If the FSDP fully-shard
    # module is importable, unpickling runs with its module-new-init override
    # disabled; otherwise with no special context. Returns the guards state object.
    try:
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard

        ctx = _fully_shard.disable_fsdp_module_new_init()
    except ImportError:
        ctx = nullcontext()  # type: ignore[assignment]
    with ctx:
        return pickle.loads(guards_state)


def load_guard_manager(
    guards_state: "GuardsState",
    target_code: types.CodeType,
    runtime_global_scope: Any,
) -> "GuardManagerWrapper":
    # [FUNCTION DOC] Rebuild the runtime guard manager from a deserialized
    # guards state: constructs a check function manager for the target code
    # using the saved output graph, shape code parts and local state, bound to
    # the runtime global scope. Returns its guard manager.
    from .output_graph import OutputGraphCommon

    return torch._dynamo.guards.CheckFunctionManager(
        target_code,
        OutputGraphCommon(guards_state.output_graph),
        shape_code_parts=guards_state.shape_code_parts,
        runtime_global_scope=runtime_global_scope,
        guard_build_local_state=getattr(guards_state, "local_state", None),
    ).guard_manager


_BackendId = NewType("_BackendId", str)  # __compiled_fn
_FunctionId = NewType("_FunctionId", str)  # __resume_at


def _backend_ids_from_code(code: types.CodeType) -> Iterator[_BackendId]:
    # [FUNCTION DOC] HELPER: Generator yielding every global name in the code's
    # names that looks like a compiled function name, recursing into nested code constants.
    for name in code.co_names:
        if is_compiled_fn_name(name):
            yield _BackendId(name)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _backend_ids_from_code(const)


@dataclasses.dataclass(frozen=True)
class InlinedSource:
    # [CLASS DOC] HELPER: Frozen record of source code inlined during tracing:
    # module name, line range, a checksum of that range, and the full module
    # text. Used to detect source changes when loading a package.
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str
    content: str


@functools.cache
def _get_module_content(module: types.ModuleType) -> str:
    # [FUNCTION DOC] HELPER: Return the full source text of a module (cached per module).
    return inspect.getsource(module)


@dataclasses.dataclass
class SourceInfo:
    # [CLASS DOC] Set of InlinedSource records for all user code inlined by
    # Dynamo into a package; extended by add_code.
    inlined_sources: set[InlinedSource]

    def add_code(self, code: types.CodeType) -> None:
        # [FUNCTION DOC] Record the source of a code object as an InlinedSource
        # (module, line range, checksum, module text). Silently ignores code with
        # no module. Raises AssertionError if the code's source lines differ from
        # the module text at that range. Mutates inlined_sources.
        module = inspect.getmodule(code)
        if module is None:
            return
        sourcelines, firstlineno = inspect.getsourcelines(code)
        lastlineno = firstlineno + len(sourcelines)
        source = "".join(sourcelines)
        if source != "".join(_get_sourcelines(module, firstlineno, lastlineno)):
            raise AssertionError(
                f"Source mismatch for {module.__name__} "
                f"(line {firstlineno}-{lastlineno})"
            )
        self.inlined_sources.add(
            InlinedSource(
                module=module.__name__,
                firstlineno=firstlineno,
                lastlineno=lastlineno,
                checksum=_hash_source(source),
                content=_get_module_content(module),
            )
        )


@dataclasses.dataclass
class _DynamoCodeCacheEntry:
    # [CLASS DOC] HELPER: Serializable per-code-object record (original code,
    # module, function names, guarded codes, import sources, backend ids,
    # access path, install flags, compile-id and bypass flags); see fields below.
    """
    Contains the serializable information associated with a single code object
    in dynamo. To restore an execution of compiled code, we will need the following
    ingredients:
      1. The "original" code object, which serves as the entry point for eager
         execution, i.e. the code only executed when there's no cache entry hit.
      2. The python module name this code object belongs to, for identifying the
         enclosing global scope to inject compiled and resume functions.
      3. A list of function names that pointing to this code object. There could be
         multiple function objects pointing to the same code such as recursive functions.
      4. A list of guarded code that eval frame dispatches to.
      5. A list of imported module objects unioned from all compiled branches.
      6. A list of "backends" (compiled fx graph) unioned from all compiled branches.
      7. A string path used to access the original code object users defined.
         A code object can be accessed by "{python_module}.{function_name}.{code_source}" .
      8. A boolean flag indicating whether the function is installed to global scope.
      9. A boolean flag indicating whether the function has a compile id.
      10. Whether or not this code entry was bypassed
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]
    code_source: str | None
    install_to_global: bool
    has_compile_id: bool = False
    bypassed: bool = False


def _lookup_code(entry: _DynamoCodeCacheEntry) -> types.CodeType:
    # [FUNCTION DOC] HELPER: Resolve the live code object for an entry by
    # walking attributes from the entry's module along its function name and
    # then its code source path (attribute names and bracketed literal indexes).
    # Raises AssertionError unless exactly one function name and the result is
    # a code object; raises PackageError if the source path is missing/unresolvable.
    # KNOWN ISSUE: the integer check on the result of rfind is always true.
    if len(entry.function_names) != 1:
        raise AssertionError(
            f"Expected exactly one function name, got {len(entry.function_names)}"
        )
    fn: Any = sys.modules[entry.python_module]
    parts = entry.function_names[0].split(".")
    for part in parts:
        fn = getattr(fn, part)
    if entry.code_source:
        parts = entry.code_source.split(".")
        for part in parts:
            if part.endswith("]"):
                index_begin = part.rfind("[")
                if not isinstance(index_begin, int):
                    raise AssertionError(
                        f"Expected int for index_begin, got {type(index_begin)}"
                    )
                if index_begin < 0:
                    raise AssertionError(
                        f"Expected non-negative index_begin, got {index_begin}"
                    )
                attr = getattr(fn, part[:index_begin], None)
                if attr is None:
                    raise PackageError(f"Cannot find source for code entry {entry}")
                fn = attr[ast.literal_eval(part[index_begin + 1 : -1])]
            else:
                fn = getattr(fn, part)
    else:
        raise PackageError(f"Cannot find source for code entry {entry}")
    if not isinstance(fn, types.CodeType):
        raise AssertionError(
            f"Expected CodeType, got {type(fn)} for code entry {entry}"
        )
    return fn


def _raise_resolution_error(code: types.CodeType, scope: Any) -> Never:
    # [FUNCTION DOC] HELPER: Always raises PackageError saying the code's qualified name could not be resolved within the given scope.
    raise PackageError(
        f"Cannot resolve a fully qualified name for {code}. Lookup scope: {scope}"
    )


def _get_code_source(code: types.CodeType) -> tuple[str, str]:
    # [FUNCTION DOC] Compute a serialized handle for a code object: returns the
    # qualified name of the top-level function/object found and a dotted path
    # (code constants, function code, closure cells) from it to the code. Handles
    # decorated and nested functions; on older Pythons searches module/class
    # members. Raises PackageError if the module or path cannot be found.
    """
    Given a code object, return a fully qualified name which will be used as
    a serialized handle to access the code object from the new process.
    This is normally a straightforward process, but there are some corner cases:
    1. When a function is defined with decorator, then this function will be captured
       inside a closure with the wrapper object.
    2. When a function is defined as a nested function, then the code object will be
       stored on the co_consts field of the parent code object by Python compiler.
    This function handles all of the corner cases above.
    """

    module = inspect.getmodule(code)
    if module is None:
        raise PackageError(f"Cannot find module for code {code}")

    toplevel: Any = module
    if sys.version_info >= (3, 11):
        parts = code.co_qualname.split(".")

        for part in parts:
            if not hasattr(toplevel, part):
                _raise_resolution_error(code, toplevel)
            toplevel = getattr(toplevel, part)
            if inspect.isfunction(toplevel) or inspect.ismethod(toplevel):
                break
    seen = set()

    def _find_code_source(obj: Any) -> str | None:
        # [FUNCTION DOC] HELPER: Depth-first search from obj for the target
        # code, tracking visited objects. Returns the attribute path suffix
        # (empty if obj is the code itself) or None if not reachable. Updates
        # the enclosing toplevel to the function/method through which the code was found.
        nonlocal toplevel
        nonlocal seen
        if obj in seen:
            return None

        seen.add(obj)

        if inspect.iscode(obj):
            if obj is code:
                return ""

            for i, const in enumerate(obj.co_consts):
                if (res := _find_code_source(const)) is not None:
                    return f".co_consts[{i}]{res}"

        if inspect.ismethod(obj):
            if (res := _find_code_source(obj.__func__)) is not None:
                toplevel = obj
                return f".__func__{res}"

        if inspect.isfunction(obj):
            if (res := _find_code_source(obj.__code__)) is not None:
                toplevel = obj
                return f".__code__{res}"
            if obj.__closure__ is not None:
                for i, cell in enumerate(obj.__closure__):
                    try:
                        cell_contents = cell.cell_contents
                    except ValueError:
                        continue
                    if not (
                        inspect.isfunction(cell_contents)
                        or inspect.iscode(cell_contents)
                        or inspect.ismethod(cell_contents)
                    ):
                        continue
                    if (res := _find_code_source(cell_contents)) is not None:
                        toplevel = obj
                        return f".__closure__[{i}].cell_contents{res}"

        if sys.version_info < (3, 11):
            if inspect.ismodule(obj):
                for value in obj.__dict__.values():
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        return res

            if inspect.isclass(obj):
                for name in itertools.chain(obj.__dict__.keys(), dir(obj)):
                    try:
                        value = getattr(obj, name)
                    except AttributeError:
                        continue
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        if value.__name__ != name:
                            _raise_resolution_error(code, toplevel)
                        return res
        return None

    code_source = _find_code_source(toplevel)
    if code_source is None:
        _raise_resolution_error(code, toplevel)
    return toplevel.__qualname__, code_source.strip(".")


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    # [CLASS DOC] Frozen snapshot of Python, torch, toolkit, Triton and GPU
    # identity, stored in cache entries so loads can be rejected on incompatible
    # systems. CHECK_GPUS lists device types checked (cuda, xpu).
    """
    System information including Python, PyTorch, and GPU details.
    This information is used to ensure compiled artifacts can only be loaded
    with compatible system configurations.
    """

    python_version: str
    torch_version: str
    toolkit_version: str | None
    triton_version: tuple[int, int] | None
    gpu_name: str | None
    CHECK_GPUS = ("cuda", "xpu")

    @classmethod
    def current(cls) -> "SystemInfo":
        # [FUNCTION DOC] Build a SystemInfo for this process: Python and torch
        # versions, Triton version (zero pair if absent) and, for the first
        # available GPU backend, its device name and toolkit version (errors reading them are ignored).
        """Create a SystemInfo instance with current system information."""
        # Get GPU name if CUDA or XPU is available
        gpu_name = None
        from torch.utils._triton import get_triton_version

        gpu_name, toolkit_version = None, None
        for device_type in cls.CHECK_GPUS:
            if getattr(torch, device_type).is_available():
                try:
                    gpu_name = getattr(torch, device_type).get_device_name()
                    toolkit_version = getattr(torch.version, device_type)
                    break
                except Exception:
                    pass

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            toolkit_version=toolkit_version,
            triton_version=get_triton_version((0, 0)),
            gpu_name=gpu_name,
        )

    def check_compatibility(
        self, other: "SystemInfo", device_type: str = "cpu"
    ) -> None:
        # [FUNCTION DOC] Compare this (saved) info against other (the current
        # system). Raises RuntimeError on Python or torch version mismatch; for
        # GPU device types also if the device is unavailable or toolkit, Triton
        # (when the current one is not the zero pair) or GPU name differ.
        # KNOWN ISSUE: the GPU-name error message labels cached/current with swapped attribute sources (self is cached, other is current, matches); the Triton check only skips when other is the zero pair.
        """
        Check if this SystemInfo is compatible with another SystemInfo.
        Raises RuntimeError if incompatible.
        """
        if self.python_version != other.python_version:
            raise RuntimeError(
                f"Compile package was created with a different Python version: {self.python_version}"
            )

        if self.torch_version != other.torch_version:
            raise RuntimeError(
                f"Compile package was created with a different PyTorch version: {self.torch_version}"
            )
        if device_type in self.CHECK_GPUS:
            if not getattr(torch, device_type).is_available():
                raise RuntimeError(f"{device_type} is not available")

            if self.toolkit_version != other.toolkit_version:
                raise RuntimeError(
                    f"Compile package was created with a different toolkit version: {self.toolkit_version}"
                )

            if (
                other.triton_version != (0, 0)
                and self.triton_version != other.triton_version
            ):
                raise RuntimeError(
                    f"Compile package was created with a different Triton version: {self.triton_version}"
                )

            # Check GPU name if CUDA/XPU was used
            if other.gpu_name is not None and self.gpu_name != other.gpu_name:
                raise RuntimeError(
                    f"Compile package was created with different GPU: "
                    f"cached={self.gpu_name}, current={other.gpu_name}"
                )


@dataclasses.dataclass
class _DynamoCacheEntry:
    # [CLASS DOC] HELPER: Serializable Dynamo half of a package: per-code
    # entries (first is the entry function), inlined source info, device type,
    # system info (defaults to the current system) and function name/line for debugging.
    codes: list[_DynamoCodeCacheEntry]
    source_info: SourceInfo
    device_type: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)
    fn_name: str | None = None
    fn_first_lineno: str | None = None

    @property
    def backend_ids(self) -> set[_BackendId]:
        # [FUNCTION DOC] Union of backend ids across all code entries.
        return {backend_id for code in self.codes for backend_id in code.backend_ids}

    def check_versions(self) -> None:
        # [FUNCTION DOC] Raise RuntimeError if the current system is incompatible with the saved system info.
        """Check if the current system is compatible with the system used to create this cache entry."""
        current_system_info = SystemInfo.current()
        self.system_info.check_compatibility(current_system_info, self.device_type)

    def debug_info(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a small summary mapping (code count, function
        # name and line, device type, backend ids) for logging. Raises
        # AssertionError if there are no code entries.
        if len(self.codes) == 0:
            raise AssertionError("Expected at least one code entry")
        return {
            "num_codes": str(len(self.codes)),
            "fn_name": self.fn_name,
            "fn_first_lineno": self.fn_first_lineno,
            "device_type": self.device_type,
            "backend_ids": list(self.backend_ids),
        }


from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)


@CacheArtifactFactory.register
class PrecompileCacheArtifact(CacheArtifact):
    # [CLASS DOC] Cache artifact kind for precompile packages; registered with
    # the artifact factory. Content is the pickled PrecompileCacheEntry, key the package id.
    def populate_cache(self) -> None:
        # [FUNCTION DOC] Write this artifact's pickled content to the local Dynamo disk cache under its key.
        DynamoCache._write_to_local_cache(self.content, self.key)

    @staticmethod
    def type() -> str:
        # [FUNCTION DOC] Return the artifact type name.
        return "precompile"


@dataclasses.dataclass
class PrecompileCacheEntry:
    # [CLASS DOC] Full serializable precompile record: the Dynamo cache entry
    # plus a mapping from backend id to backend artifact content.
    """
    A full cache entry for caching precompile, for a toplevel torch.compile.
    Consists of a _DynamoCacheEntry, which contains all the dynamo related contents,
    and a set of backends content. In general, the backend content here will always
    be of type precompile_context.BackendCacheArtifact
    """

    dynamo: _DynamoCacheEntry
    backends: dict[_BackendId, Any]

    @staticmethod
    def from_cache_entry(
        cache_entry: _DynamoCacheEntry, backends: dict[_BackendId, Any]
    ) -> Optional["PrecompileCacheEntry"]:
        # [FUNCTION DOC] Combine a Dynamo cache entry with available backends.
        # For each code entry, collects the backends it needs; if one is
        # missing, logs a warning, emits a structured trace artifact, marks that
        # code entry bypassed and stops scanning that entry. Returns a new
        # entry (never None despite the annotation). Mutates code entries' bypassed flag.
        backend_content: dict[_BackendId, Any] = {}

        for code in cache_entry.codes:
            for backend_id in code.backend_ids:
                if backend_id not in backends:
                    logger.warning("Backend not found")
                    debug_str = json.dumps(
                        {
                            "entry": cache_entry.debug_info(),
                            "missing_backend": backend_id,
                        }
                    )
                    torch._logging.trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "dynamo_cache_bypass",
                            "encoding": "json",
                        },
                        payload_fn=lambda: debug_str,
                        expect_trace_id=False,
                    )
                    code.bypassed = True
                    break
                else:
                    backend_content[backend_id] = backends[backend_id]

        return PrecompileCacheEntry(dynamo=cache_entry, backends=backend_content)


def _hash_source(source: str) -> str:
    # [FUNCTION DOC] HELPER: Hex SHA-256 digest of the source text.
    sha256_hash = hashlib.sha256()
    sha256_hash.update(source.encode())
    return sha256_hash.hexdigest()


def _get_sourcelines(
    m: types.ModuleType, firstlineno: int, lastlineno: int
) -> list[str]:
    # [FUNCTION DOC] HELPER: Source lines of a module from one-based first line up to (excluding) last line.
    return inspect.getsourcelines(m)[0][firstlineno - 1 : lastlineno - 1]


def _hash_sourcelines(m: types.ModuleType, firstlineno: int, lastlineno: int) -> str:
    # [FUNCTION DOC] HELPER: Checksum of a module's source line range.
    return _hash_source("".join(_get_sourcelines(m, firstlineno, lastlineno)))


def _compile_frame_context(
    code: types.CodeType,
) -> contextlib.AbstractContextManager[None]:
    # [FUNCTION DOC] HELPER: Return a context manager that, when entered,
    # simulates a compile frame for the code: bumps the frame counter, makes a
    # compile id, sets up compile context and compile timing with code
    # metadata, and logs the Dynamo start.
    from torch._dynamo.convert_frame import get_compile_id, log_dynamo_start
    from torch._guards import compile_context, CompileContext

    # Each code represents a new compile frame
    # recompiles on the same frame are all saved
    # under the same cache entry, so we don't have recompile ids
    # i.e. If cold start had 0/0, 0/1, 1/0, 1/1, these would be
    # collapsed into 0/0, 1/0 on warm.
    # pyrefly: ignore [deprecated]
    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Generator context body; mutates global frame counter state and logs.
        increment_frame()
        compile_id = get_compile_id(frame_state={})
        with (
            compile_context(CompileContext(compile_id)),
            dynamo_timed(
                "_compile.compile_inner",
                phase_name="entire_frame_compile",
                dynamo_compile_column_us="dynamo_cumulative_compile_time_us",
                # TODO: save all relevant compilation metrics
                metadata={
                    "frame_key": str(torch._dynamo.utils.curr_frame),
                    "co_name": code.co_name,
                    "co_filename": code.co_filename,
                    "co_firstlineno": code.co_firstlineno,
                },
            ),
        ):
            log_dynamo_start(code)
            yield

    return _ctx()


class CompilePackage:
    # [CLASS DOC] Mutable recorder/installer for one top-level compiled
    # function. State: the innermost user function, mapping code object to
    # _DynamoCodeCacheEntry (first is the entry function), current entry
    # during compilation, globals installed per module, device type, cached
    # backends, source info, resume codes, initialized flag. Key methods:
    # initialize, code_context, add_* recorders, validate, install/uninstall, cache_entry.
    """
    CompilePackage is considered a low level component and should not be directly exposed to
    end users. It has the following interface:

    1. `CompilePackage.__init__()` which optionally takes previously serialized dynamo states.
        a. when `dynamo` argument is None, it will construct a brand new CompilePackage object.
        b. when `dynamo` argument is not None, it will load a pre-compiled dynamo state.
    2. `package.save()` which dumps the dynamo and backend states to a DynamoCacheEntry object.
    3. `package.install(backends) which will handle all the side-effectful global scope
        updates with compiled functions and resume functions.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        # [FUNCTION DOC] Create a package. With a function, immediately
        # initializes (fresh, or from a saved Dynamo entry), uninstalls any
        # installed state and validates. With no function, leaves it
        # uninitialized for a later initialize call.
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}

        self._current_entry: _DynamoCodeCacheEntry | None = None
        self._installed_globals: dict[types.ModuleType, list[str]] = {}
        # device_type that model compiled with.
        self._device_type = "cpu"

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}
        self._source_info: SourceInfo = SourceInfo(inlined_sources=set())
        self._resume_codes: set[types.CodeType] = set()
        self._initialized = False
        if fn is not None:
            self.initialize(fn, dynamo, ignore_inlined_sources)
            self.uninstall()
            self.validate()

    def is_initialized(self) -> bool:
        # [FUNCTION DOC] Whether initialize has completed.
        return self._initialized

    def initialize(
        self,
        fn: Any,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        # [FUNCTION DOC] Bind the package to fn (unwrapped to its innermost
        # function). With a saved entry: checks system compatibility, and
        # unless ignoring sources verifies inlined-source checksums (raises
        # RuntimeError on change) and adopts its source info; then maps code
        # objects to saved entries (first entry is the function itself).
        # Without one, registers the function's code. Raises AssertionError if
        # already initialized or arguments are wrong type.
        from .eval_frame import innermost_fn

        if self._initialized:
            raise AssertionError("CompilePackage is already initialized")
        self._source_info = SourceInfo(inlined_sources=set())
        self._innermost_fn = innermost_fn(fn)  # type: ignore[assignment]
        if self._innermost_fn is None:
            raise AssertionError("innermost_fn returned None")
        if dynamo is not None:
            if not isinstance(dynamo, _DynamoCacheEntry):
                raise AssertionError(f"Expected _DynamoCacheEntry, got {type(dynamo)}")
            dynamo.check_versions()
            if not ignore_inlined_sources:
                for code in dynamo.source_info.inlined_sources:
                    m = importlib.import_module(code.module)
                    checksum = _hash_sourcelines(m, code.firstlineno, code.lastlineno)
                    if checksum != code.checksum:
                        raise RuntimeError(
                            f"Source code changes detected for {code.module} (line {code.firstlineno} - line {code.lastlineno})"
                        )

                self._source_info = dynamo.source_info

            main, *codes = dynamo.codes
            self._codes = {self._innermost_fn.__code__: main}
            for code in codes:
                self._codes[SerializedCode.to_code_object(code.python_code)] = code
        else:
            self._add_function(
                self._innermost_fn.__code__, self._innermost_fn.__module__
            )
        self._initialized = True

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: _FunctionId | None = None,
        code_source: str | None = None,
        install_to_global: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: Register a code object, creating an empty
        # entry if new, or checking an existing entry agrees on module,
        # install-to-global flag and code source (AssertionError otherwise).
        # Appends function_name to the entry's names when given.
        if python_code not in self._codes:
            code = _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(python_code),
                python_module=python_module,
                function_names=[],
                guarded_codes=[],
                import_sources={},
                backend_ids=[],
                code_source=code_source,
                install_to_global=install_to_global,
            )
            self._codes[python_code] = code
        else:
            code = self._codes[python_code]
            if code.python_module != python_module:
                raise AssertionError(
                    f"python_module mismatch: {code.python_module} != {python_module}"
                )
            if code.install_to_global != install_to_global:
                raise AssertionError(
                    f"install_to_global mismatch: {code.install_to_global} != {install_to_global}"
                )
            if code.code_source != code_source:
                raise AssertionError(
                    f"code_source mismatch: {code.code_source} != {code_source}"
                )

        if function_name is not None:
            code.function_names.append(function_name)

    @property
    def cached_backends(self) -> dict[_BackendId, Any]:
        # [FUNCTION DOC] Mapping of backend id to backend object recorded in this process (debug/testing).
        return self._cached_backends

    @functools.cached_property
    def source_id(self) -> str:
        # [FUNCTION DOC] Stable id (hash of qualified name and first line) of the package's function; cached. Raises AssertionError if uninitialized.
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set")
        return CompilePackage.source_id_from_fn(self._innermost_fn)

    def _add_user_function(self, code: types.CodeType) -> None:
        # [FUNCTION DOC] HELPER: Register user code encountered during compile,
        # computing its qualified name and code source path. Raises PackageError
        # if the module or path cannot be found.
        function_name, code_source = _get_code_source(code)
        module = inspect.getmodule(code)
        if module is None:
            raise PackageError(f"Cannot find module for code {code}")
        self._add_function(
            code,
            module.__name__,
            function_name=_FunctionId(function_name),
            code_source=code_source,
        )

    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager marking code as the entry currently
        # being compiled (registering it first if unknown). Not reentrant
        # (AssertionError). On exit marks the entry as having a compile id and
        # clears the current entry.
        if self._current_entry is not None:
            raise AssertionError("_current_entry is already set in code_context")

        # Sometimes user code cannot be inlined in dynamo resulting in extra user code
        # being compiled. We should record these as when they are actually invoked.
        if code not in self._codes:
            self._add_user_function(code)

        entry = self._codes[code]
        self._current_entry = entry
        try:
            yield
        finally:
            entry.has_compile_id = True
            self._current_entry = None

    def add_guarded_code(
        self,
        guards_state: bytes,
        dynamo_code: types.CodeType,
    ) -> None:
        # [FUNCTION DOC] Record a compiled variant (pickled guards and Dynamo
        # bytecode) on the current entry, plus backend ids referenced by the
        # bytecode. No-op if the entry is bypassed; AssertionError outside code_context.
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_guarded_code")
        if self._current_entry.bypassed:
            return
        guarded_code_entry = _GuardedCodeCacheEntry(
            guards_state=guards_state,
            dynamo_code=SerializedCode.from_code_object(dynamo_code),
        )
        self._current_entry.guarded_codes.append(guarded_code_entry)
        for backend_id in _backend_ids_from_code(dynamo_code):
            self._add_backend_id(backend_id)

    def add_inlined_source(self, sources: list[types.CodeType]) -> None:
        # [FUNCTION DOC] Record source info for each inlined code object,
        # skipping resume functions. No-op if bypassed; AssertionError outside
        # code_context. Mutates source info.
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_inlined_source")
        if self._current_entry.bypassed:
            return
        for code in sources:
            if code in self._resume_codes:
                continue
            self._source_info.add_code(code)

    def update_device_type(self, graph: torch.fx.Graph | None) -> None:
        # [FUNCTION DOC] Set the package device type from the graph's devices (overwrites the previous value).
        self._device_type = _graph_device_type(graph)

    def bypass_current_entry(self) -> None:
        # [FUNCTION DOC] Mark the current entry bypassed so nothing further is recorded or installed for it. AssertionError outside code_context.
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in bypass_current_entry")
        self._current_entry.bypassed = True

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: str,
    ) -> None:
        # [FUNCTION DOC] Register a generated resume function's code to be
        # installed into its module globals under function_name, and remember it
        # as a resume code (excluded from inlined sources).
        self._add_function(
            python_code,
            python_module,
            function_name=_FunctionId(function_name),
            install_to_global=True,
        )
        self._resume_codes.add(python_code)

    def add_import_source(self, alias: str, module_name: str) -> None:
        # [FUNCTION DOC] Record that global alias should be bound to the imported module module_name on install. AssertionError outside code_context.
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_import_source")
        self._current_entry.import_sources[alias] = module_name

    def _add_backend_id(
        self, backend_id: _BackendId, backend: Any | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Add a backend id to the current entry if not present; optionally cache the backend object. AssertionError outside code_context.
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_backend_id")
        if backend_id not in self._current_entry.backend_ids:
            self._current_entry.backend_ids.append(backend_id)
        if backend is not None:
            self._cached_backends[backend_id] = backend

    def add_backend_id(self, backend_id: str, backend: Any | None = None) -> None:
        # [FUNCTION DOC] Public wrapper to record a backend id, which must start with the compiled-function prefix and underscore (AssertionError otherwise).
        if not backend_id.startswith(f"{COMPILED_FN_PREFIX}_"):
            raise AssertionError(
                f"backend_id must start with '{COMPILED_FN_PREFIX}_', got '{backend_id}'"
            )
        self._add_backend_id(_BackendId(backend_id), backend)

    def validate(self) -> None:
        # [FUNCTION DOC] Check invariants: no current entry, initialized, and the
        # first code entry is the innermost function's code. Raises AssertionError.
        if self._current_entry is not None:
            raise AssertionError("_current_entry should be None during validate")
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set during validate")
        if not self._initialized:
            raise AssertionError("CompilePackage is not initialized during validate")
        if next(iter(self._codes)) is not self._innermost_fn.__code__:
            raise AssertionError(
                "First code entry does not match _innermost_fn.__code__"
            )

    def _install_global(self, module: types.ModuleType, name: str, value: Any) -> None:
        # [FUNCTION DOC] HELPER: Bind name to value in the module globals,
        # first disowning any pending cleanup hook for that name, and record it
        # for later uninstall. Mutates module globals.
        # A pre-reset compile in this process may still own `name` via a
        # CleanupHook that hasn't fired yet. We're taking over the binding now,
        # so that hook must not delete it once its code object is collected.
        CleanupHook.disown(module.__dict__, name)
        module.__dict__[name] = value
        self._installed_globals.setdefault(module, []).append(name)

    def uninstall(self) -> None:
        # [FUNCTION DOC] Remove every global this package installed and reset
        # precompile cache entries on the function's code. Mutates module
        # globals and eval-frame state. AssertionError if uninitialized.
        from torch._C._dynamo.eval_frame import _reset_precompile_entries

        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in uninstall")
        for module, names in self._installed_globals.items():
            for name in names:
                module.__dict__.pop(name, None)

        self._installed_globals = {}

        _reset_precompile_entries(self._innermost_fn.__code__)

    def install(self, backends: dict[_BackendId, Any]) -> None:
        # [FUNCTION DOC] Activate the package: uninstall old state, then for
        # each code entry (inside a simulated compile frame when it had a compile
        # id) install import aliases, resume functions (factories when they have
        # free variables), deserialized disabled backends under their ids,
        # populate the builtins dict if needed, rebuild guard managers from saved
        # guards and load precompile entries into eval-frame. Bypassed entries
        # only get imports and resume functions; entries without guarded code
        # make the code skipped. Raises RuntimeError for a missing backend and
        # AssertionError on inconsistent guards state. Mutates module globals.
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        from .convert_frame import input_codes
        from .output_graph import get_builtins_dict

        self.uninstall()
        for code, entry in self._codes.items():
            context = (
                _compile_frame_context(code)
                if entry.has_compile_id
                else contextlib.nullcontext()
            )
            with context:
                module = sys.modules[entry.python_module]
                for alias, module_name in entry.import_sources.items():
                    self._install_global(
                        module, alias, importlib.import_module(module_name)
                    )
                target_code = code
                if entry.install_to_global:
                    for function_name in entry.function_names:
                        if code.co_freevars:
                            # Resume functions with freevars need a factory
                            # that takes a closure tuple, matching
                            # install_resume_function_global in output_graph.py.
                            f_globals = module.__dict__
                            fn_name = function_name

                            def _make_fn(
                                closure: tuple[types.CellType, ...],
                                _code: types.CodeType = code,
                                _globals: dict[str, Any] = f_globals,
                                _name: str = fn_name,
                            ) -> types.FunctionType:
                                # [FUNCTION DOC] HELPER: Factory building the resume function from the captured code and globals with the given closure cells.
                                return types.FunctionType(
                                    _code, _globals, _name, None, closure
                                )

                            self._install_global(module, function_name, _make_fn)
                        else:
                            fn = types.FunctionType(
                                code, module.__dict__, function_name
                            )
                            self._install_global(module, function_name, fn)
                if entry.code_source:
                    target_code = _lookup_code(entry)

                if entry.bypassed:
                    # If the entry is bypassed, do not install backends
                    # or guarded codes.
                    continue

                input_codes.add(target_code)
                for backend_id in entry.backend_ids:
                    if backend_id not in backends:
                        raise RuntimeError(
                            f"Backend {backend_id} is not found in the given backends"
                        )
                    with dynamo_timed(
                        "after_deserialization", phase_name="backend_compile"
                    ):
                        backend = backends[backend_id].after_deserialization()
                        self._install_global(
                            module,
                            backend_id,
                            torch._dynamo.disable(backend),
                        )

                if len(entry.guarded_codes) == 0:
                    # Dynamo generates empty graph for trivial functions, should just skip them
                    # in these cases.
                    torch._dynamo.eval_frame.skip_code(target_code)

                for guarded_code in entry.guarded_codes:
                    with dynamo_timed("precompile_load_guards"):
                        guards_state = load_guards_state(guarded_code.guards_state)
                    runtime_global_scope = sys.modules[entry.python_module].__dict__
                    # The installed builtins dict might be absent from the runtime
                    # while loading guards. Populate it if it's missing.
                    if (
                        builtin_dict_name
                        := guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                    ):
                        # A pre-reset compile's CleanupHook may still own this
                        # name even when we're about to leave its value alone
                        # below (same dict object every compile in this
                        # module), so it must not delete it once collected.
                        CleanupHook.disown(runtime_global_scope, builtin_dict_name)
                        builtins_dict = get_builtins_dict(runtime_global_scope)
                        if builtin_dict_name in runtime_global_scope:
                            if (
                                runtime_global_scope[builtin_dict_name]
                                is not builtins_dict
                            ):
                                raise AssertionError(
                                    f"Builtins dict mismatch for key '{builtin_dict_name}'"
                                )
                        else:
                            runtime_global_scope[builtin_dict_name] = builtins_dict
                    if not isinstance(guards_state, torch._dynamo.guards.GuardsState):
                        raise AssertionError(
                            f"Expected GuardsState, got {type(guards_state)}"
                        )
                    with dynamo_timed("precompile_build_guards"):
                        guard_manager = load_guard_manager(
                            guards_state, target_code, runtime_global_scope
                        )
                    _load_precompile_entry(
                        target_code,
                        guard_manager,
                        SerializedCode.to_code_object(guarded_code.dynamo_code),
                    )

    def cache_entry(self) -> _DynamoCacheEntry:
        # [FUNCTION DOC] Validate, then snapshot the package as a serializable
        # Dynamo cache entry (codes, source info, device type, function name and line).
        self.validate()
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in cache_entry")
        return _DynamoCacheEntry(
            codes=list(self._codes.values()),
            source_info=self._source_info,
            device_type=self._device_type,
            fn_name=self._innermost_fn.__qualname__,
            fn_first_lineno=self._innermost_fn.__code__.co_firstlineno,
        )

    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str:
        # [FUNCTION DOC] Hex SHA-256 over the qualified name and first line number of fn's innermost function; used as the package key.
        from .eval_frame import innermost_fn

        innermost_fn_ = innermost_fn(fn)

        sha256_hash = hashlib.sha256()
        sha256_hash.update(innermost_fn_.__qualname__.encode())
        sha256_hash.update(str(innermost_fn_.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()


_Backends = dict[_BackendId, Any]


class DynamoStore(abc.ABC):
    # [CLASS DOC] Abstract storage for packages. Subclasses implement clear,
    # write and read; concrete methods record packages to PrecompileContext and
    # save/load entries by key, attaching backend artifacts.
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.

    This is an abstract base class for different storage implementations.
    """

    def record_package(self, package: CompilePackage) -> None:
        # [FUNCTION DOC] Register the package's cache entry with PrecompileContext under its source id.
        """
        Records a package to PrecompileContext, so that it can be serialized later.
        """
        from torch._dynamo.precompile_context import PrecompileContext

        cache_entry = package.cache_entry()
        PrecompileContext.record_dynamo_cache_entry(
            cache_entry=cache_entry, key=package.source_id
        )

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        # [FUNCTION DOC] Wrap an eager backend object in an artifact and record it in PrecompileContext (testing aid).
        """
        Records eager fx graphs to PrecompileContext for testing purposes.
        """
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        result = EagerCacheArtifact(key=backend_id, content=backend)
        PrecompileContext.record_artifact(result)

    @abc.abstractmethod
    def clear(self) -> None:
        # [FUNCTION DOC] Abstract: remove all stored packages.
        ...

    @abc.abstractmethod
    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        # [FUNCTION DOC] Abstract: persist a precompile entry under the given path or key.
        """
        Abstract method to write dynamo cache entry and backends to storage.

        Args:
            dynamo: The dynamo cache entry to write
            backends: Dictionary of backend content to write
            path: Path or key to identify where to write the data
        """
        ...

    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None:
        # [FUNCTION DOC] Gather serialized backend artifacts for each backend id
        # from PrecompileContext (RuntimeError if missing, AssertionError if wrong
        # type), build a precompile entry and write it under key.
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        backend_content: _Backends = {}
        for backend_id in cache_entry.backend_ids:
            serialized_backend = PrecompileContext.serialize_artifact_by_key(backend_id)
            if serialized_backend is None:
                raise RuntimeError(
                    f"Backend {backend_id} is not found in the given backends"
                )
            if not isinstance(serialized_backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(serialized_backend)}"
                )
            backend_content[backend_id] = serialized_backend

        entry = PrecompileCacheEntry(cache_entry, backend_content)

        self.write(entry, key)

    def save_package(self, package: CompilePackage, key: str) -> None:
        # [FUNCTION DOC] Record the package then save its cache entry under key.
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        self.record_package(package)
        cache_entry = package.cache_entry()
        self.save_cache_entry(cache_entry, key)

    @abc.abstractmethod
    def read(self, path: str) -> PrecompileCacheEntry:
        # [FUNCTION DOC] Abstract: load the precompile entry stored under path.
        """
        Abstract method to read dynamo cache entry and backends from storage.

        Args:
            path: Path or key to identify where to read the data from

        Returns:
            A tuple containing (dynamo_cache_entry, backend_content)
        """
        ...

    def load_cache_entry(self, key: str) -> PrecompileCacheEntry:
        # [FUNCTION DOC] Read the entry for key and register each backend
        # artifact with PrecompileContext (AssertionError if a backend has the wrong type). Returns the entry.
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        precompile_entry = self.read(key)
        for backend in precompile_entry.backends.values():
            if not isinstance(backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(backend)}"
                )
            PrecompileContext.record_artifact(backend)

        return precompile_entry

    def load_package(
        self, fn: Any, key: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        # [FUNCTION DOC] Load the entry for key and build a CompilePackage for fn from it; returns the package and the backends mapping (not yet installed).
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        entry = self.load_cache_entry(key)
        package = CompilePackage(fn, entry.dynamo)
        return package, entry.backends


class InMemoryDynamoStore(DynamoStore):
    # [CLASS DOC] DynamoStore keeping entries in a dict (packages) keyed by path.
    """
    A DynamoStore implementation that keeps state about CompilePackages in memory.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Start with an empty mapping of stored packages.
        self.packages: dict[str, PrecompileCacheEntry] = {}

    def clear(self) -> None:
        # [FUNCTION DOC] Remove all stored packages.
        self.packages.clear()

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        # [FUNCTION DOC] Store the entry in the dict under path (overwrites).
        """
        Store the dynamo cache entry and backends in memory instead of writing to disk.
        """
        self.packages[path] = cache_entry

    def read(self, path: str) -> PrecompileCacheEntry:
        # [FUNCTION DOC] Return the stored entry for path; RuntimeError if absent.
        """
        Read dynamo cache entry and backends from memory.
        """
        if path not in self.packages:
            raise RuntimeError(f"No package found with key {path}")

        return self.packages[path]


class DiskDynamoStore(DynamoStore):
    # [CLASS DOC] DynamoStore persisting each entry as a pickled file named
    # "entry" inside a directory per key, under an optional path prefix.
    """
    A DynamoStore implementation that keeps state about CompilePackages on disk.
    """

    def __init__(self, path_prefix: str = "") -> None:
        # [FUNCTION DOC] Remember the directory prefix under which packages are stored.
        """
        Initialize a DiskDynamoStore with a path prefix.

        Args:
            path_prefix: Prefix directory for where to put CompilePackages on disk
        """
        self._path_prefix = path_prefix

    def path_prefix(self) -> str:
        # [FUNCTION DOC] Return the configured directory prefix.
        return self._path_prefix

    def clear(self) -> None:
        # [FUNCTION DOC] Recursively delete the prefix directory (errors ignored); does nothing if the prefix is empty.
        """
        Clear all CompilePackages from disk.
        """
        if self.path_prefix():
            shutil.rmtree(self.path_prefix(), ignore_errors=True)

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        # [FUNCTION DOC] Pickle the entry, record it as a cache artifact, and
        # write it to the local cache directory for path. Any failure is wrapped in RuntimeError.
        """
        Write dynamo cache entry and backends to disk.
        """
        try:
            pickled_content: bytes = pickle.dumps(cache_entry)
            CacheArtifactRecorder(PrecompileCacheArtifact.type(), path).record(
                pickled_content
            )
            self._write_to_local_cache(pickled_content, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def _write_to_local_cache(self, pickled_content: bytes, path: str) -> None:
        # [FUNCTION DOC] HELPER: Create the directory (under the prefix) and
        # atomically write the pickled bytes to a file named entry inside it.
        # Failures are wrapped in RuntimeError. File I/O side effect.
        from torch._inductor.codecache import write_atomic

        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            os.makedirs(path, exist_ok=True)
            write_atomic(os.path.join(path, "entry"), pickled_content)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def read(self, path: str) -> PrecompileCacheEntry:
        # [FUNCTION DOC] Read and unpickle the entry file for path (under the
        # prefix). Failures are wrapped in RuntimeError.
        """
        Read dynamo cache entry and backends from disk.
        """
        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            with open(os.path.join(path, "entry"), "rb") as f:
                pickled_content = f.read()
                entry = pickle.loads(pickled_content)
                return entry
        except Exception as e:
            raise RuntimeError(f"Failed to load package from path {path}: {e}") from e


class DiskDynamoCache(DiskDynamoStore):
    # [CLASS DOC] Disk store at the Inductor cache directory's dynamo subfolder
    # with convenience save/load keyed automatically by function source id, and
    # hit/miss/error counters.
    """
    Special DiskDynamoStore which adds some helper functions for automatically
    tracking paths of packages
    """

    def save(self, package: CompilePackage) -> None:
        # [FUNCTION DOC] Save the package under its source id and log it.
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        key = package.source_id
        logger.info("Saving CompilePackage for %s", package.source_id)
        super().save_package(package, key)

    def load(self, fn: Callable[..., Any]) -> PrecompileCacheEntry | None:
        # [FUNCTION DOC] Look up the entry for fn by source id. Returns it and
        # bumps the hit counter; returns None with an error counter bump if
        # loading fails, or a miss counter bump if the directory does not exist.
        # KNOWN ISSUE: the failure warning has a format placeholder but no argument is supplied.
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        key = CompilePackage.source_id_from_fn(fn)
        logger.info("Loading CompilePackage for %s", key)
        path = os.path.join(self.path_prefix(), key)
        if os.path.exists(path):
            try:
                result = super().load_cache_entry(key)
                counters["dynamo_cache"]["dynamo_cache_hit"] += 1
                return result
            except Exception:
                counters["dynamo_cache"]["dynamo_cache_error"] += 1
                logger.warning("Failed to load package from path %s", exc_info=True)
                return None
        logger.info("No package found for %s", key)
        counters["dynamo_cache"]["dynamo_cache_miss"] += 1
        return None

    def load_and_install_package(self, fn: Callable[..., Any]) -> CompilePackage | None:
        # [FUNCTION DOC] Load fn's entry; if none, return None; else build a
        # package, install its backends (mutating globals) and return it.
        """
        Load directly into a package and install backends
        """
        results = self.load(fn)
        if results is None:
            return None
        else:
            package = CompilePackage(fn, results.dynamo)
            package.install(results.backends)
            return package

    def path_prefix(self) -> str:
        # [FUNCTION DOC] Directory prefix computed freshly from the current cache directory (the constructor argument is ignored).
        return os.path.join(cache_dir(), "dynamo")


def cache_dir() -> str:
    # [FUNCTION DOC] HELPER: Return the Inductor cache directory (lazy import).
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return cache_dir()


DynamoCache = DiskDynamoCache(os.path.join(cache_dir(), "dynamo"))
