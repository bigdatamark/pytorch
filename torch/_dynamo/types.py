# [MODULE DOC]
# Core type definitions and callback/hook protocols shared across Dynamo.
# Contents: FrameState alias, GuardFail, GuardFilterEntry, GuardFn protocol,
# GuardedCode, ConvertFrameReturn, wrap_guarded_code, the DynamoCallbackFn
# protocol and DynamoCallback alias, compiler-config types, and hook protocols
# (guard hook, guard-complete hook, profiler start/end, bytecode hook).
# Import-time side effects: imports C-extension types from
# torch._C._dynamo.eval_frame (re-exported under public aliases CacheEntry,
# ExtraState, FrameAction, FrameExecStrategy, DynamoFrameType); no other state.
"""This module contains the core type definitions and protocols used throughout Dynamo.

The types defined here fall into several categories:
- Guard related types (GuardFn, GuardFail, GuardedCode): Used for tracking and managing guards that protect compiled code
- Frame and cache types (FrameState, CacheEntry): Used for managing interpreter frame state and caching
- Callback protocols (DynamoCallbackFn): Define the interface for frame evaluation callbacks
- Hook protocols (DynamoGuardHook, ProfilerStartHook, ProfilerEndHook, BytecodeHook): Define various hook points for
  instrumentation and customization

These types provide the foundational interfaces that enable Dynamo's dynamic compilation and optimization system,
ensuring type safety and clear contracts between different components of the system.
"""

import dataclasses
import types
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

# CacheEntry has a `guard_manager` field for the guard, and a `code` field for the code object.
from torch._C._dynamo.eval_frame import (
    _CacheEntry as CacheEntry,
    _ExtraState as ExtraState,
    _FrameAction as FrameAction,
    _FrameExecStrategy as FrameExecStrategy,
    _PyInterpreterFrame as DynamoFrameType,
)
from torch._guards import CompileId, Guard


# We use a dict to store additional data per frame.
FrameState = dict[Any, Any]


class GuardFail(NamedTuple):
    # [CLASS DOC] Immutable record describing a guard failure: reason is the
    # text of the failed guard code part, orig_code is the code object whose
    # guard failed. Passed to guard_fail_fn callbacks.
    # A string repr of the piece of failed guard code we eval-ed
    reason: str
    # A code object where we failed a guard
    orig_code: types.CodeType


@dataclasses.dataclass(frozen=True)
class GuardFilterEntry:
    # [CLASS DOC] Frozen record describing one guard, handed to guard filter
    # functions: guard name, whether it has a value and the value, the guard
    # type and any derived guard types, whether it is a global guard, and the
    # original Guard object it was built from.
    name: str
    has_value: bool
    value: object
    guard_type: str
    derived_guard_types: tuple[str, ...]
    is_global: bool
    orig_guard: Guard


class GuardFn(Protocol):
    # [CLASS DOC] Protocol for a compiled guard function/manager. Attributes
    # hold the guard closure variables, argument names, code parts (plain and
    # verbose), global scope, optional failure callback, and the owning
    # cache entry and extra state. Calling it with the frame locals mapping
    # returns True if all guards pass (cached code is reusable).
    closure_vars: dict[str, object]
    args: list[str]
    code_parts: list[str]
    verbose_code_parts: list[str]
    global_scope: dict[str, object]
    guard_fail_fn: Callable[[GuardFail], None] | None
    cache_entry: CacheEntry | None
    extra_state: ExtraState | None

    # maps locals of user function to bool
    def __call__(self, f_locals: dict[str, object]) -> bool: ...


@dataclasses.dataclass
class GuardedCode:
    # [CLASS DOC] Compiled output for a frame: the new code object, the guard
    # manager that must pass for it to be used, the CompileId that produced
    # it, and a trace annotation string (defaults to "Unknown").
    code: types.CodeType
    guard_manager: GuardFn
    compile_id: CompileId
    trace_annotation: str = "Unknown"


@dataclasses.dataclass
class ConvertFrameReturn:
    # [CLASS DOC] Result returned by a frame-conversion callback. Fields:
    # frame_exec_strategy (default: skip this frame, default-run recursive
    # frames), apply_to_code (apply the strategy to future frames with the same
    # code, default True), guarded_code (compiled result or None), and
    # skip_reason (optional text). Default instance means no compilation.
    # default return is no compiled code (i.e. `return None`):
    # strategy is to skip non-recursively, for all future intercepted frames too

    # eval frame execution strategy for this frame
    frame_exec_strategy: FrameExecStrategy = dataclasses.field(
        default_factory=lambda: FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
    )
    # also apply frame_exec strategy to future frames with same code
    apply_to_code: bool = True
    guarded_code: GuardedCode | None = None
    skip_reason: str | None = None


def wrap_guarded_code(guarded_code: GuardedCode) -> ConvertFrameReturn:
    # [FUNCTION DOC] Wrap a GuardedCode in a ConvertFrameReturn that runs the
    # compiled code with the default strategy for both the frame and its
    # recursive frames. Takes the guarded code, returns the new
    # ConvertFrameReturn; no side effects, no exceptions raised.
    return ConvertFrameReturn(
        frame_exec_strategy=FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT),
        guarded_code=guarded_code,
    )


class DynamoCallbackFn(Protocol):
    # [CLASS DOC] Protocol for the frame-evaluation callback: called with the
    # intercepted frame, the matching cache entry (or None) and the per-frame
    # state mapping; returns a ConvertFrameReturn describing what to run.
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        frame_state: FrameState,
    ) -> ConvertFrameReturn: ...


DynamoCallback = DynamoCallbackFn | None | bool


CompilerConfig = Mapping[str, Any]


@runtime_checkable
class CompilerConfigProvider(Protocol):
    # [CLASS DOC] Runtime-checkable protocol for objects (e.g. backends) that
    # expose their compiler configuration via get_compiler_config, which
    # returns a read-only mapping of options or None if there is none.
    def get_compiler_config(self) -> CompilerConfig | None: ...


class DynamoGuardHook(Protocol):
    # [CLASS DOC] Protocol for a hook invoked during guard evaluation with the
    # guard manager, the code object, the frame locals, the index of the cache
    # entry being checked, and whether it is the last entry. Returns nothing.
    def __call__(
        self,
        guard_manager: GuardFn,
        code: types.CodeType,
        f_locals: dict[str, object],
        index: int,
        last: bool,
    ) -> None: ...


class DynamoGuardCompleteHook(Protocol):
    # [CLASS DOC] Protocol for a hook invoked after guard evaluation finishes,
    # given whether a cache hit occurred; returns a bool result.
    def __call__(
        self,
        cache_hit: bool,
    ) -> bool: ...


class ProfilerStartHook(Protocol):
    # [CLASS DOC] Protocol for a hook that starts a profiler range given a
    # name and returns an opaque record handle (to be passed to the end hook).
    def __call__(
        self,
        name: str,
        # TODO(whc) how do I annotate a _RecordFunction here?
    ) -> Any: ...


class ProfilerEndHook(Protocol):
    # [CLASS DOC] Protocol for a hook that ends a profiler range, given the
    # record handle returned by the start hook; returns nothing.
    def __call__(self, record: Any) -> None: ...


class BytecodeHook(Protocol):
    # [CLASS DOC] Protocol for a hook called with the original and newly
    # generated code objects; may return a replacement code object, or None
    # to keep the new code unchanged.
    def __call__(
        self, code: types.CodeType, new_code: types.CodeType
    ) -> types.CodeType | None: ...
