# mypy: disable-error-code="method-assign"

# [MODULE DOC] Runtime entry point of TorchDynamo. Defines the context managers/decorators
# (OptimizeContext, RunOnlyContext, DisableContext) that install the PEP 523 eval-frame callback,
# the OptimizedModule wrapper returned for compiled nn.Modules, the optimize/optimize_assert/explain
# /export public entry points, the "stance" mechanism (set_stance) that overrides the callback,
# thread-local trace bookkeeping (DynamoTLS), and TorchPatcher which disables Dynamo on some torch
# internals. Import-time side effects: creates module globals (default stance, cached_backends,
# thread-local state, always_optimize_code_objects), and registers an atexit hook that logs traced frames.

"""
This module implements the core frame evaluation handler for TorchDynamo's compilation system.
The eval frame handler intercepts Python bytecode execution at runtime to enable dynamic
compilation and optimization of PyTorch code.

Key components defined here:
- Frame evaluation handlers that intercept and analyze Python execution frames
- Guards management for tracking dependencies and invalidating compiled code
- Optimization contexts and decorators (optimize, run_once, disable, etc.)
- Export functionality for saving optimized graphs
- Backend compiler integrations and callback management

Functions in this file are responsible for modifying the eval frame handler at RUNTIME.
Therefore, all functions in this file are hot and performance-critical. Functions that
only execute at compile time should be placed in torch._dynamo.convert_frame.

The eval frame handler is the core mechanism that enables TorchDynamo to dynamically
intercept, analyze and optimize PyTorch code during execution. It works by registering
a custom frame evaluation function that gets called for every Python frame, allowing
us to detect PyTorch operations and trigger compilation as needed.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import inspect
import itertools
import logging
import os
import sys
import sysconfig
import textwrap
import threading
import traceback
import types
import unittest
import warnings
import weakref
from collections.abc import Generator, Sized
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import Any, cast, Literal, NamedTuple, TYPE_CHECKING
from unittest.mock import patch

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards

# see discussion at https://github.com/pytorch/pytorch/issues/120699
from torch._C._dynamo.eval_frame import (  # noqa: F401
    get_eval_frame_isolate_recompiles_id,
    reset_code,
    set_code_exec_strategy,
    set_eval_frame,
    set_eval_frame_isolate_recompiles_id,
    set_fullgraph_compiled_frame_count,
    set_fullgraph_error_on_nested_compile,
    set_guard_complete_hook,
    set_guard_error_hook,
    set_skip_guard_eval_unsafe,
    unsupported,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.types import (
    CompilerConfig,
    CompilerConfigProvider,
    ConvertFrameReturn,
    FrameAction,
    FrameExecStrategy,
)
from torch._export.utils import _compiling_state_context
from torch._library.opaque_object import is_custom_class
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._utils_internal import DISABLE_JUSTKNOBS, justknobs_check, log_export_usage
from torch.export.dynamic_shapes import (
    _combine_args,
    _DimHint,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
    _RelaxedConstraint,
    Constraint,
)
from torch.fx import GraphModule, traceback as fx_traceback
from torch.fx.experimental._dynamism import (
    clone_and_convert_to_meta,
    track_dynamism_across_examples,
)
from torch.fx.experimental.dynamic_spec import (
    _coerce_to_shapes_spec,
    ParamsSpec,
    ShapesSpec,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from . import config, convert_frame, distributed, external_utils, trace_rules, utils
from .backends.registry import CompilerFn, lookup_backend
from .code_context import code_context
from .exc import (
    CondOpArgsMismatchError,
    ShortenTraceback,
    UncapturedHigherOrderOpError,
    Unsupported,
    UserError,
    UserErrorType,
)
from .hooks import Hooks
from .mutation_guard import install_generation_tagging_init
from .utils import (
    _get_error_on_graph_break,
    _set_error_on_graph_break,
    common_constant_types,
    compile_times,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.package import CompilePackage
    from torch._dynamo.repro.after_dynamo import WrapBackendDebug
    from torch._subclasses import fake_tensor
    from torch.export._trace import _DynamicShapesInput
    from torch.fx.node import Argument, Node, Target

    from .types import (
        CacheEntry,
        DynamoCallback,
        DynamoFrameType,
        GuardFail,
        GuardFilterEntry,
    )


log = logging.getLogger(__name__)

_next_isolate_recompiles_id = itertools.count()


always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


# See https://github.com/python/typing/pull/240
class Unset(Enum):
    # [CLASS DOC] HELPER: Single-member enum used as a sentinel meaning "no value set yet"
    # (distinct from None/False, which are valid callbacks).
    token = 0


cached_backends: dict[int, CompilerFn] = {}

unset = Unset.token

_in_optimized_module = False


if DISABLE_JUSTKNOBS:
    _maybe_set_eval_frame = set_eval_frame
else:

    def _maybe_set_eval_frame(callback: DynamoCallback) -> DynamoCallback:
        # [FUNCTION DOC] HELPER: Defined only when justknobs are enabled. Installs the given eval-frame
        # callback and returns the previous one, unless the enable_compiler_set_eval_frame justknob is
        # off, in which case it warns once and returns the given callback without installing anything.
        # A wrapper on set_eval_frame that is guarded by a Justknob.
        # Users can disable torchDynamo by setting the JK to False.
        if not justknobs_check("pytorch/compiler:enable_compiler_set_eval_frame"):
            torch._dynamo.utils.warn_once(
                "Dynamo disabled by Justknob: enable_compiler_set_eval_frame, skipping set_eval_frame"
            )
            return callback
        else:
            return set_eval_frame(callback)


# The set of valid stance strings accepted by torch.compiler.set_stance. Kept as
# a shared alias so producers (set_stance) and the exhaustive consumer
# (_callback_from_stance) stay in sync at type-check time.
StanceStr = Literal[
    "default",
    "eager_then_compile",
    "aot_eager_then_compile",
    "force_eager",
    "eager_on_recompile",
    "fail_on_recompile",
]


@dataclass
class DynamoStance:
    # [CLASS DOC] Dataclass describing the process-wide compile stance set via torch.compiler.set_stance.
    # Fields: stance (one of the StanceStr names), skip_guard_eval_unsafe (skip guard evaluation when
    # the stance is trusted), backend (optional backend name/callable forced onto compiled regions
    # in the default stance). A single global instance (_stance) holds the active stance.
    stance: StanceStr = "default"
    skip_guard_eval_unsafe: bool = False
    backend: str | Callable[..., Any] | None = None


_stance = DynamoStance()
_force_eager_nested_compile = threading.local()


@contextlib.contextmanager
def _use_eager_on_nested_compile() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that increments a thread-local depth counter so nested
    # torch.compile wrappers run their functions eagerly (used inside compiler-internal tracing).
    # Restores the previous depth on exit, even on error.
    """Run torch.compile wrappers eagerly inside compiler-internal tracing."""
    prior = getattr(_force_eager_nested_compile, "depth", 0)
    _force_eager_nested_compile.depth = prior + 1
    try:
        yield
    finally:
        _force_eager_nested_compile.depth = prior


def _is_eager_on_nested_compile() -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the current thread is inside an eager-on-nested-compile region.
    return getattr(_force_eager_nested_compile, "depth", 0) > 0


def _set_stance(stance: DynamoStance) -> DynamoStance:
    # [FUNCTION DOC] HELPER: Replaces the global stance with the given one and returns the prior stance.
    # Raises RuntimeError if called while a Dynamo eval-frame callback is active (inside a compile region).
    # Side effect: mutates the module-level _stance.
    global _stance

    from torch._C._dynamo.eval_frame import get_eval_frame_callback

    callback = get_eval_frame_callback()

    if callback is not False and callback is not None:
        raise RuntimeError("attempted to set_stance in a torch.compile region")

    prior = _stance
    _stance = stance
    return prior


_set_stance._dynamo_forbidden = True  # type: ignore[attr-defined]

_EXAMPLE_INPUTS: dict[str, list[Any]] | None = None


def get_example_inputs(key: str) -> list[Any]:
    # [FUNCTION DOC] Returns the (mutable) list of recorded example inputs for a key, lazily creating the
    # global cache mapping and an empty list for new keys. Used by the eager_then_compile stances.
    global _EXAMPLE_INPUTS
    if _EXAMPLE_INPUTS is None:
        _EXAMPLE_INPUTS = {}

    if key not in _EXAMPLE_INPUTS:
        _EXAMPLE_INPUTS[key] = []

    return _EXAMPLE_INPUTS[key]


@contextlib.contextmanager
def _set_in_optimized_module() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the global in-optimized-module flag to true for the
    # duration of the block (so graph-breaking forward hooks are not registered/traced) and restores the
    # previous value on exit.
    # Set in dynamo's OptimizedModule forward, to have better coverage than is_compiling().
    # Prevents graph-breaking forward hooks from being registered & traced.
    # TODO(pianpwk): subsume this flag with better is_compiling() coverage
    global _in_optimized_module
    _old_in_optimized_module = (
        _in_optimized_module  # do we need this? can we just set it to False after
    )
    _in_optimized_module = True
    try:
        yield
    finally:
        _in_optimized_module = _old_in_optimized_module


def _is_in_optimized_module() -> bool:
    # [FUNCTION DOC] HELPER: Returns the global flag saying whether execution is inside an OptimizedModule call.
    global _in_optimized_module
    return _in_optimized_module


def _callback_from_stance(callback: DynamoCallback) -> DynamoCallback:
    # [FUNCTION DOC] HELPER: Maps the intended callback to the one actually installed, based on the global
    # stance. default: returns the callback (or a wrapped callback using the forced backend if one is set and
    # the callback is active). eager_then_compile / aot_eager_then_compile: wraps an active callback in a
    # delayed-compile callback. force_eager: returns None (disabled). eager_on_recompile: returns False
    # (run-only). fail_on_recompile: wraps an active callback so any compile attempt on a frame with tensors
    # raises RuntimeError describing the recompile. Raises RuntimeError for an unknown stance.
    if _stance.stance == "default":
        # force_backend
        if _stance.backend is not None and callback not in (False, None):
            callback = _create_wrapped_callback(get_compiler_fn(_stance.backend))

        return callback
    elif _stance.stance == "eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "aot_eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "force_eager":
        # disable
        return None
    elif _stance.stance == "eager_on_recompile":
        # run mode
        return False
    elif _stance.stance == "fail_on_recompile":
        if callback in (False, None):
            return callback

        def fail_callback(
            frame: DynamoFrameType, *args: Any, **kwargs: Any
        ) -> ConvertFrameReturn:
            # [FUNCTION DOC] HELPER: Eval-frame callback for the fail_on_recompile stance. Returns an empty
            # ConvertFrameReturn (skip) for frames that trace rules skip or that hold no tensors. Otherwise
            # builds a message with the frame location, any guard failure reasons from existing cache
            # entries and precompiled guards, and raises RuntimeError with it.
            if trace_rules.check(frame.f_code):
                return ConvertFrameReturn()
            if not convert_frame.has_tensor_in_frame(frame):
                return ConvertFrameReturn()

            from torch._C._dynamo.eval_frame import (
                _debug_get_cache_entry_list,
                _debug_get_precompile_entries,
            )
            from torch._dynamo.guards import get_and_maybe_log_recompilation_reasons

            message = (
                "Detected recompile when torch.compile stance is 'fail_on_recompile'. "
                + f"filename: '{frame.f_code.co_filename}', "
                + f"function name: '{frame.f_code.co_name}', "
                + f"line number: {frame.f_lineno}"
            )
            cache_entries = _debug_get_cache_entry_list(frame.f_code)
            if cache_entries:
                reasons = get_and_maybe_log_recompilation_reasons(
                    cache_entries,
                    frame,
                    innermost_backend(callback),  # pyrefly: ignore [bad-argument-type]
                    skip_logging=True,
                )
                if reasons:
                    failures = textwrap.indent("\n".join(reasons), "- ")
                    guard_failure_details = (
                        f"triggered by the following guard failure(s):\n{failures}"
                    )
                    message += f"\n{textwrap.indent(guard_failure_details, '    ')}"
            precompile_entries = _debug_get_precompile_entries(frame.f_code)
            if len(precompile_entries) > 0:
                message += "\nFailed on the following precompiled guards: "
                for entry in precompile_entries:
                    message += f"\n{entry.guard_manager}{entry.guard_manager.check_verbose(frame.f_locals)}"  # type: ignore[attr-defined]
            raise RuntimeError(message)

        # to prevent cache miss due to different backend
        fail_callback._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

        return fail_callback
    else:
        raise RuntimeError(f"invalid torch.compile stance '{_stance}'")


def _create_wrapped_callback(
    compiler_fn: CompilerFn,
) -> convert_frame.CatchErrorsWrapper:
    # [FUNCTION DOC] HELPER: Builds a standard Dynamo callback for a backend: converts frames with a fresh
    # Hooks object and wraps the converter in the error-catching wrapper. Returns the wrapped callback.
    hooks = Hooks()
    return convert_frame.catch_errors_wrapper(
        convert_frame.convert_frame(  # type: ignore[arg-type]
            compiler_fn,
            hooks,
        ),
        hooks,
    )


def _get_or_add_example_inputs(frame: DynamoFrameType) -> list[Any]:
    # [FUNCTION DOC] HELPER: Looks up the example-input list keyed by the frame's file name and first line;
    # if fewer than two examples are recorded, appends a meta-device clone of the frame's locals.
    # Returns the list (mutated in place).
    key = frame.f_code.co_filename + str(frame.f_code.co_firstlineno)
    example_inputs = get_example_inputs(key)

    if len(example_inputs) < 2:
        example_inputs.append(clone_and_convert_to_meta(frame.f_locals))

    return example_inputs


def _create_delayed_compile_callback(
    callback: DynamoCallback, stance: StanceStr
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Wraps a callback for the eager_then_compile and aot_eager_then_compile stances.
    # The returned callback runs the first call eagerly (or with aot_eager compilation), and on the second
    # call computes which input dimensions changed between examples, stores that in the code context, and
    # compiles with the original backend. The result carries the original callback for cache keying.
    def callback_fn(*args: Any, **kwargs: Any) -> convert_frame.ConvertFrameReturn:
        # [FUNCTION DOC] HELPER: Callback body. Records example inputs for the frame; on the first sight
        # returns a default-strategy result (eager) or compiles with aot_eager; later computes dynamism across
        # examples, saves it in the code context and compiles with the underlying backend.
        frame = args[0]
        example_inputs = _get_or_add_example_inputs(frame)

        if len(example_inputs) == 1:
            if stance == "eager_then_compile":
                return ConvertFrameReturn(
                    frame_exec_strategy=FrameExecStrategy(
                        FrameAction.DEFAULT, FrameAction.DEFAULT
                    )
                )
            elif stance == "aot_eager_then_compile":
                aot_eager_fn = get_compiler_fn("aot_eager")
                return _create_wrapped_callback(aot_eager_fn)(*args, **kwargs)

        dynamism = track_dynamism_across_examples(example_inputs)
        code_context.get_context(frame.f_code)["dynamism"] = dynamism
        compiler_fn = callback._torchdynamo_orig_backend._torchdynamo_orig_backend  # type: ignore[union-attr]
        return _create_wrapped_callback(compiler_fn)(*args, **kwargs)

    # to prevent cache miss due to different backend
    callback_fn._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

    return callback_fn


def _is_skip_guard_eval_unsafe_stance() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether the active stance asked to skip guard evaluation unsafely.
    return _stance.skip_guard_eval_unsafe


def _reset_guarded_backend_cache() -> None:
    # [FUNCTION DOC] HELPER: Calls reset on every cached backend that supports it, then clears the global
    # cached_backends mapping. Called from torch._dynamo.reset.
    global cached_backends
    for backend in cached_backends.values():
        if hasattr(backend, "reset"):
            backend.reset()
    cached_backends.clear()


DONT_WRAP_FILES = {
    # For tracing into fx modules
    inspect.getsourcefile(GraphModule),
    join(dirname(dirname(__file__)), "onnx/_internal/fx/dynamo_graph_extractor.py"),
}


def _debug_get_cache_entry_list(
    code: types.CodeType | Callable[..., Any],
) -> list[CacheEntry]:
    # [FUNCTION DOC] HELPER: Debug accessor returning the Dynamo cache entries attached to a code object
    # (or to a callable's code), delegating to the C extension.
    """
    Given a code object or a callable object, retrieve the cache entries
     stored in this code.
    """
    if callable(code):
        code = code.__code__
    return torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)


def _get_cache_entries_for_region(
    code: types.CodeType | Callable[..., Any],
    isolate_recompiles_id: int,
) -> list[CacheEntry]:
    # [FUNCTION DOC] HELPER: Returns the cache entries owned by one isolate_recompiles bucket of a code
    # object (or callable's code), most recently used first; -1 selects the default bucket. Does not
    # include the read-only fallback entries from the default bucket.
    """
    Return the cache entries for a specific isolate_recompiles region on
    ``code``, in LRU order (most-recently-used first).

    Pass ``isolate_recompiles_id=-1`` to get the default (non-isolated)
    bucket; pass a region's id (as exposed via
    ``opt._isolate_recompiles_id``) to get that region's bucket.

    Returns only entries owned by the requested bucket. During an actual
    lookup, isolated regions also fall back read-only to the default
    bucket for BC-friendly reuse — this helper does not include those
    fallback entries.
    """
    if callable(code):
        code = code.__code__
    return torch._C._dynamo.eval_frame._get_cache_entries_for_region(
        code, isolate_recompiles_id
    )


def _get_total_cache_entry_count(
    code: types.CodeType | Callable[..., Any],
) -> int:
    # [FUNCTION DOC] HELPER: Returns the number of cache entries across all isolate_recompiles buckets.
    """Total cache entries across all isolate_recompiles regions for a code object."""
    if callable(code):
        code = code.__code__
    return torch._C._dynamo.eval_frame._get_total_cache_entry_count(code)


class OptimizedModule(torch.nn.Module):
    # [CLASS DOC] Module wrapper returned by torch.compile(module). Holds the original module in _orig_mod
    # and the creating context in dynamo_ctx; its forward is replaced by a compiled/wrapped callable.
    # Attribute reads/writes/deletes not in _opt_mod_attributes are forwarded to the original module, and
    # training mirrors the original module's flag. Picklable via __reduce__ (forward is rebuilt on load).
    # Key methods: _initialize (installs forward), __call__, AOT compile/save/load helpers.
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], CompilerConfig | None]

    _opt_mod_attributes = {
        "_orig_mod",
        "dynamo_ctx",
        "_torchdynamo_orig_callable",
        "_torchdynamo_wrapper_id",
        "get_compiler_config",
        "forward",
        "_forward",
        "__dict__",
        "named_children_walk",
        "_super_module_initialized",
    }

    @staticmethod
    def _forward_has_skip_rule(mod: torch.nn.Module) -> bool:
        # [FUNCTION DOC] HELPER: True if the module's forward is a bound method that trace rules say Dynamo
        # must skip (e.g. a built-in torch.nn module), so an extra wrapper frame is needed to capture it.
        return isinstance(mod.forward, types.MethodType) and trace_rules.check(
            mod.forward
        )

    @staticmethod
    def _should_wrap_module_call_impl(fn: Any) -> bool:
        # [FUNCTION DOC] HELPER: True if fn is a bound _call_impl of an nn.Module whose forward has a skip
        # rule; in that case the call implementation must still be wrapped for inlining.
        if getattr(fn, "__name__", "") != "_call_impl":
            return False

        mod = getattr(fn, "__self__", None)
        if not isinstance(mod, torch.nn.Module):
            return False
        return OptimizedModule._forward_has_skip_rule(mod)

    def __init__(self, mod: torch.nn.Module, dynamo_ctx: _TorchDynamoContext) -> None:
        # [FUNCTION DOC] Stores the original module (first, via object-level set, because attribute access
        # depends on it), initializes nn.Module state, registers the module as a child, saves the context,
        # installs the optimized forward through _initialize, and copies the training flag.
        # NOTE: this must go first, because attribute reads/writes of `self`
        # uses `_orig_mod`, and sometimes users override `Module.__init__` to
        # do attribute reads/writes on `self`.
        #
        # We also can't use regular setattr because `super().__setattr__` will
        # complain for module value before `super().__init__()`
        object.__setattr__(self, "_orig_mod", mod)
        self._super_module_initialized = False
        super().__init__()
        self._super_module_initialized = True

        # Installs the params/buffer
        self._orig_mod = mod  # `super().__setattr__` will register this module
        self.dynamo_ctx = dynamo_ctx
        self._initialize()
        self.training = self._orig_mod.training

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the length of the original module if it is sized; otherwise raises TypeError.
        # Proxy the len call to the original module
        # pyrefly: ignore [unsafe-overlap]
        if isinstance(self._orig_mod, Sized):
            return len(self._orig_mod)
        # Mimic python's default behavior for objects without a length
        raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

    def _initialize(self) -> None:
        # [FUNCTION DOC] HELPER: Sets self.forward by applying dynamo_ctx. For a disable context uses the
        # module's call; if wrap_top_frame is set or forward has a skip rule, wraps the module in an
        # inline wrapper first; otherwise compiles the module call so hooks run outside Dynamo. If the
        # module is lazy (has an initialize hook), saves forward as _forward and routes through
        # _call_lazy_check.
        # Do this stuff in constructor to lower overhead slightly
        if isinstance(self.dynamo_ctx, DisableContext):
            # No need to check trace rules
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)
        elif config.wrap_top_frame or self._forward_has_skip_rule(self._orig_mod):
            # This may be a torch.nn.* instance in trace_rules.py which
            # won't trigger a frame evaluation workaround to add an extra
            # frame we can capture
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            # Invoke hooks outside of dynamo then pickup the inner frame
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)

        if inspect.getattr_static(self._orig_mod, "_initialize_hook", None) is not None:
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Calls the module through nn.Module call machinery with the in-optimized-module flag
        # set. Emits a warning (stacklevel 2) if global module hooks exist, since they fire an extra time.
        if torch.nn.modules.module._has_any_global_hook():
            warnings.warn(
                "Using `torch.compile(module)` when there are global hooks on "
                "modules (e.g., from `register_module_forward_hook`); this will"
                " cause the hooks to fire an extra time for the "
                "`OptimizedModule` created by `torch.compile(module)`. If this "
                "causes undesired behavior, please try using `module.compile()`"
                ", or use the per-module hooks instead",
                stacklevel=2,
            )
        with _set_in_optimized_module():
            return super().__call__(*args, **kwargs)

    def _aot_compile(self, inputs: list[torch._dynamo.aot_compile.ModelInput]) -> None:
        # [FUNCTION DOC] HELPER: Experimental. Ahead-of-time compiles the original module for the given model
        # inputs and replaces forward with the result. Raises AssertionError without hooks, and RuntimeError
        # if aot compile is disabled in config, fullgraph is off, or the callback is not callable.
        """
        Experimental: AOT Compile a set of inputs and use that as the forward function
        """
        model = self._orig_mod
        hooks = self.dynamo_ctx._hooks
        if hooks is None:
            raise AssertionError("hooks must not be None for AOT compilation")
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_compile=True"
            )
        if not self.dynamo_ctx.fullgraph:
            raise RuntimeError(
                "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
            )

        if not callable(self.dynamo_ctx.callback):
            raise RuntimeError("aot compile requires a callable dynamo callback.")

        backend = innermost_backend(self.dynamo_ctx.callback)
        from torch._dynamo.aot_compile import aot_compile_module

        self.forward = aot_compile_module(model, inputs, hooks, backend)

    def _save_aot_compiled_module(self, path: str | None = None) -> bytes:
        # [FUNCTION DOC] HELPER: Serializes the AOT-compiled forward to bytes and returns them; if a path is
        # given also writes the bytes to that file. Raises RuntimeError if aot compile is disabled and
        # AssertionError if forward is not an AOT compiled model.
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_compile=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        if not isinstance(self.forward, AOTCompiledModel):
            raise AssertionError(
                f"Expected self.forward to be AOTCompiledModel, got {type(self.forward)}"
            )
        result: bytes = self.forward.serialize()
        if path is not None:
            with open(path, "wb") as f:
                f.write(result)
        return result

    def _load_aot_compiled_module(self, data: bytes) -> None:
        # [FUNCTION DOC] HELPER: Deserializes bytes into an AOT compiled model bound to the original module
        # and installs it as forward. Raises RuntimeError if aot compile is disabled.
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_compile=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        compiled_forward = AOTCompiledModel.deserialize(self._orig_mod, data)
        if not isinstance(compiled_forward, AOTCompiledModel):
            raise AssertionError(
                f"Expected compiled_forward to be AOTCompiledModel, got {type(compiled_forward)}"
            )
        self.forward = compiled_forward

    def __reduce__(
        self,
    ) -> tuple[type[OptimizedModule], tuple[torch.nn.Module, _TorchDynamoContext]]:
        # [FUNCTION DOC] Pickle support: reconstruct from the class, original module and dynamo context.
        return (self.__class__, (self._orig_mod, self.dynamo_ctx))

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a copy of the instance dictionary without the forward and __call__ entries
        # (they are rebuilt on load).
        state = dict(self.__dict__)
        state.pop("forward", None)
        state.pop("__call__", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restores the instance dictionary from state and re-runs _initialize to rebuild forward.
        self.__dict__ = state
        self._initialize()

    @property
    # pyrefly: ignore [bad-override]
    def training(self) -> bool:
        # [FUNCTION DOC] Property getter: returns the original module's training flag.
        return self._orig_mod.training

    @training.setter
    def training(self, value: bool) -> None:
        # [FUNCTION DOC] Property setter: forwards the value to the original module's training flag, but only
        # after nn.Module initialization finished (earlier writes are the nn.Module default and ignored).
        # Ignore the `training` mutation in `super().__init__()`, since that's
        # setting the default on `nn.Module`, but we are mirroring the
        # `training` attr in `self._orig_mod`.
        if self._super_module_initialized:
            self._orig_mod.training = value

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Fallback attribute lookup: returns the registered _orig_mod child for that name,
        # otherwise reads the attribute from the original module (raising whatever it raises).
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # [FUNCTION DOC] Sets the attribute on the wrapper if it is a class attribute or one of the wrapper's
        # own attributes; otherwise sets it on the original module.
        # Allow patching over class attributes
        if hasattr(type(self), name):
            return super().__setattr__(name, value)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__setattr__(name, value)
        return setattr(self._orig_mod, name, value)

    def __delattr__(self, name: str) -> None:
        # [FUNCTION DOC] Mirror of __setattr__: deletes from the wrapper for class/own attributes, otherwise
        # from the original module.
        # This mirrors `__setattr__`
        if hasattr(type(self), name):
            return super().__delattr__(name)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__delattr__(name)
        return delattr(self._orig_mod, name)

    def _call_lazy_check(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forward for lazy modules. If the original module is still lazy (has an
        # initialize hook and a callable parameter-inference method), runs the inference so parameters
        # materialize outside Dynamo, then calls the saved compiled forward with the arguments.
        if (
            inspect.getattr_static(self._orig_mod, "_initialize_hook", None) is not None
            and inspect.getattr_static(self._orig_mod, "_infer_parameters", None)
            is not None
            and callable(self._orig_mod._infer_parameters)
        ):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            self._orig_mod._infer_parameters(self._orig_mod, args, kwargs)
        return self._forward(*args, **kwargs)

    def __dir__(self) -> list[str]:
        # [FUNCTION DOC] Returns the original module's attribute names followed by wrapper-only names.
        orig_mod_attrs = self._orig_mod.__dir__()
        return orig_mod_attrs + [
            attr for attr in super().__dir__() if attr not in orig_mod_attrs
        ]


def remove_from_cache(f: Any) -> None:
    # [FUNCTION DOC] Invalidates the Dynamo cache for f: resets its code object (a code object, a function,
    # or an object whose forward has code). If none can be found, resets all of Dynamo and logs a warning.
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from . import reset  # type: ignore[attr-defined]

        reset()
        log.warning("could not determine __code__ for %s", f)


def nothing() -> None:
    # [FUNCTION DOC] HELPER: No-op callable; used as the default (identity-compared) on_enter/patch hook.
    pass


def always_false() -> bool:
    # [FUNCTION DOC] HELPER: Returns false; a constant predicate.
    return False


def _static_getattr(obj: Any, name: str, default: Any = None) -> Any:
    # [FUNCTION DOC] HELPER: Reads an attribute via the base object lookup (bypassing user-defined attribute
    # hooks and avoiding extra Python frames); returns default if the attribute is missing.
    # Not inspect.getattr_static: this runs while the eval frame callback is
    # installed, so its Python frames land in Dynamo's frame counters.
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return default


def innermost_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Unwraps nested Dynamo wrappers by following the original-callable attribute, but only
    # while the wrapper-id attribute equals the identity of the current object (so bound methods and
    # functools-copied attributes are not followed). Returns the innermost callable; raises AssertionError
    # if an unwrapped value is not callable.
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    # Probe statically so these lookups never run a user-defined __getattr__,
    # which may not tolerate unknown names.
    unaltered_fn = fn
    while (
        _static_getattr(unaltered_fn, "_torchdynamo_orig_callable") is not None
        # Only follow the chain if _torchdynamo_wrapper_id matches id(fn).
        # This prevents following chains in two cases:
        # 1. Bound methods: id(bound_method) != id(wrapper_function), so we
        #    won't unwrap through __func__ and lose the self binding.
        # 2. functools.wraps copies: When functools.wraps copies
        #    _torchdynamo_orig_callable from a wrapped function, the copied
        #    _torchdynamo_wrapper_id won't match the outer wrapper's id.
        and _static_getattr(unaltered_fn, "_torchdynamo_wrapper_id") == id(unaltered_fn)
    ):
        # pyrefly: ignore [missing-attribute]
        unaltered_fn = unaltered_fn._torchdynamo_orig_callable
        if not callable(unaltered_fn):
            raise AssertionError(
                f"A callable function is expected, but {type(unaltered_fn)} is provided."
            )
    return unaltered_fn


def innermost_backend(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Follows the original-backend attribute chain to the innermost backend callable.
    # Raises AssertionError if a link is not callable.
    """
    Unwrap backend wrapper chain via _torchdynamo_orig_backend to find the
    innermost backend function.
    """
    while hasattr(fn, "_torchdynamo_orig_backend"):
        fn = fn._torchdynamo_orig_backend
        if not callable(fn):
            raise AssertionError(
                f"A callable function is expected, but {type(fn)} is provided."
            )
    return fn


def make_set_enable_dynamic(enable: bool) -> Any:
    # [FUNCTION DOC] Returns a config patcher (callable that applies and returns a cleanup) for dynamic
    # shape mode: enable true makes everything dynamic by default; false disables automatic dynamic shapes
    # and assumes static. Raises AssertionError if enable is not a bool.
    if not isinstance(enable, bool):
        raise AssertionError(f"Expected enable to be a bool, got {type(enable)}")
    if enable:
        # Assume everything is dynamic by default
        return config._make_closure_patcher(assume_static_by_default=False)
    else:
        return config._make_closure_patcher(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        )


@contextlib.contextmanager
def set_enable_dynamic(enable: bool) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that temporarily applies the dynamic-shape config for enable and
    # restores the config on exit.
    cleanup = make_set_enable_dynamic(enable)()
    try:
        yield
    finally:
        cleanup()


# A thread local storage that serves to store information as Dynamo traces
# through a user provided function.
class DynamoTLS(threading.local):
    # [CLASS DOC] Thread-local storage for trace bookkeeping: traced_frame_infos (summaries of frames
    # Dynamo tried to trace, in order) and skip_reasons (collected skip messages during a fullgraph call,
    # or None when not collecting). A single instance, dynamo_tls, is used.
    # Each string is a summary of a frame Dynamo attempted to trace, stored in
    # temporal order.
    traced_frame_infos: list[str] = []

    # Accumulated skip reasons during a fullgraph compile_wrapper call.
    # Each entry is a formatted string like "fn (file.py:42): reason".
    # None when not collecting (fullgraph not active).
    skip_reasons: list[str] | None = None


dynamo_tls = DynamoTLS()


def reset_skip_reasons() -> None:
    # [FUNCTION DOC] Starts collecting skip reasons for this thread by setting the list to empty.
    dynamo_tls.skip_reasons = []


def add_skip_reason(reason: str, frame: DynamoFrameType) -> None:
    # [FUNCTION DOC] If collection is active, appends a message with the frame's function name, file,
    # first line and the reason; otherwise does nothing.
    if dynamo_tls.skip_reasons is not None:
        code = frame.f_code
        dynamo_tls.skip_reasons.append(
            f"{code.co_name} ({code.co_filename}:{code.co_firstlineno}): {reason}"
        )


def get_skip_reasons() -> list[str]:
    # [FUNCTION DOC] Returns the collected skip reasons for this thread (empty list if none/not collecting).
    return dynamo_tls.skip_reasons or []


def clear_dynamo_tls() -> None:
    # [FUNCTION DOC] Clears the traced frame summaries and stops skip-reason collection for this thread.
    dynamo_tls.traced_frame_infos.clear()
    dynamo_tls.skip_reasons = None


@atexit.register
def _log_traced_frames() -> None:
    # [FUNCTION DOC] HELPER: Registered at exit; logs at info level the list of traced frame summaries.
    # Reads the calling thread's thread-local storage only.
    """
    At program exit, log all of the frames Dynamo has attempted to trace from,
    excluding the continuation frames generated by Dynamo.
    """
    msg = "\n".join(dynamo_tls.traced_frame_infos)
    msg = textwrap.indent(msg, "  * ")
    msg = f"TorchDynamo attempted to trace the following frames: [\n{msg}\n]"
    log.info(msg)


def guard_collectives_hook(guard_eval_result: bool) -> bool:
    # [FUNCTION DOC] Guard-complete hook for distributed jobs. If a guard process group is configured,
    # gathers every rank's guard result (a collective, timed as guard_collective) and returns true only if
    # all ranks hit the cache; so one miss forces recompilation everywhere. Without a group returns the
    # input unchanged.
    import torch.distributed as dist
    from torch._dynamo.utils import dynamo_timed

    # guard_eval_result == True  ==>  cache hit
    if pg := distributed.get_guard_pg():
        with dynamo_timed(
            "guard_collective", log_pt2_compile_event=False, log_waitcounter=True
        ):
            log.debug("guard_collective %s", guard_eval_result)
            # TODO: a bit awkward to time, this isn't inside of the dynamo compile region
            all_results = [None] * pg.size()
            dist.all_gather_object(all_results, guard_eval_result, group=pg)
            # True = everyone hit, OK to run
            # False = someone missed, force recompile everywhere
            res = all(all_results)
            log.debug("guard_collective %s -> %s", guard_eval_result, res)
            return res
    return guard_eval_result


_not_set = object()


class _TorchDynamoContext:
    # [CLASS DOC] Base context manager/decorator that enables a Dynamo eval-frame callback. Holds the
    # callback, compile options (fullgraph, error_on_graph_break, export, dynamic, dynamic_shapes,
    # compiler_config, package, hooks), a unique isolate-recompiles id (or -1), and enter/exit hooks that
    # patch config on entry. As a context manager it swaps the eval-frame callback in and out; as a
    # decorator (__call__) it wraps functions, classes or modules so each call runs with the callback
    # installed and global state restored afterward.
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter: Callable[[], Any] = nothing,
        backend_ctx_ctor: Callable[
            [], contextlib.AbstractContextManager[Any]
        ] = null_context,
        patch_fn: Callable[[], Any] = nothing,
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: bool | None = None,
        export: bool = False,
        dynamic: bool | None = None,
        compiler_config: CompilerConfig | None = None,
        package: CompilePackage | None = None,
        hooks: Hooks | None = None,
        isolate_recompiles: bool = False,
        dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Validates and stores options. Raises AssertionError if callback is not callable,
        # False or None; raises ValueError if both dynamic and dynamic_shapes are set. Normalizes
        # dynamic_shapes into a shapes spec, allocates an isolate-recompiles id when requested, runs
        # patch_fn, registers the innermost backend in cached_backends, and builds the list of enter/exit
        # hooks (dynamic mode, dynamic shapes spec, on_enter, backend context).
        super().__init__()
        if not (callable(callback) or callback is False or callback is None):
            raise AssertionError(
                f"callback must be callable, False, or None, got {type(callback)}"
            )
        # Normalize the shorthand forms: dict / ParamsSpec / ShapesSpec all
        # land here as a ShapesSpec (or None).
        dynamic_shapes = _coerce_to_shapes_spec(dynamic_shapes)
        self.callback: DynamoCallback = callback
        self._backend_ctx_ctor = backend_ctx_ctor
        self.prior: Unset | DynamoCallback = unset
        self.first_ctx = first_ctx
        self.fullgraph = fullgraph
        self.error_on_graph_break = error_on_graph_break
        self.export = export
        self._dynamic = dynamic
        self._dynamic_shapes = dynamic_shapes
        self.compiler_config = compiler_config
        self.cleanup_fns: list[Callable[[], Any]] = []
        self.enter_exit_hooks = []
        self._package = package
        self._hooks = hooks
        self._isolate_recompiles_id = (
            next(_next_isolate_recompiles_id) if isolate_recompiles else -1
        )
        patch_fn()

        # Save the backends so that we can reset them during torch._dynamo.reset
        backend = innermost_backend(callback)  # type: ignore[arg-type]
        cached_backends.setdefault(id(backend), backend)  # type: ignore[arg-type]

        if dynamic is not None and dynamic_shapes is not None:
            raise ValueError(
                "`dynamic` and `dynamic_shapes` cannot both be set. "
                "`dynamic_shapes` controls dynamic behavior."
            )

        if dynamic is not None:
            self.enter_exit_hooks.append(make_set_enable_dynamic(dynamic))

        if dynamic_shapes is not None:
            self.enter_exit_hooks.append(
                config._make_closure_patcher(_dynamic_shapes_spec=dynamic_shapes)
            )

        if on_enter is not nothing:
            # this case is not common
            def call_on_enter() -> Callable[[], None]:
                # [FUNCTION DOC] HELPER: Enter hook that runs on_enter and returns a no-op cleanup.
                on_enter()
                return nothing

            self.enter_exit_hooks.append(call_on_enter)

        if backend_ctx_ctor is not contextlib.nullcontext:
            # this case is not common
            def call_backend_ctx() -> functools.partial[bool | None]:
                # [FUNCTION DOC] HELPER: Enter hook that creates and enters the backend context manager and
                # returns a cleanup that exits it (with no exception info).
                ctx = backend_ctx_ctor()
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_backend_ctx)

    def __enter__(self) -> None:
        # [FUNCTION DOC] Context-manager entry. Raises RuntimeError if config forbids context-manager use.
        # Otherwise disables the eval-frame callback while saving the prior one, runs the enter hooks (saving
        # cleanups), sets skip-guard-eval from the stance, and installs the stance-adjusted callback.
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        self.prior = set_eval_frame(None)
        self.cleanup_fns = [enter() for enter in self.enter_exit_hooks]
        self.prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
            _is_skip_guard_eval_unsafe_stance()
        )
        _maybe_set_eval_frame(_callback_from_stance(self.callback))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None:
        # [FUNCTION DOC] Context-manager exit. Raises AssertionError if entry did not happen. Disables the
        # callback, restores the skip-guard-eval flag, runs and clears the cleanups, reinstalls the prior
        # callback (via the stance) and resets prior. Returns None (never suppresses exceptions).
        if self.prior is unset:
            raise AssertionError("__exit__ called without matching __enter__")
        set_eval_frame(None)
        set_skip_guard_eval_unsafe(self.prior_skip_guard_eval_unsafe)
        for cleanup in self.cleanup_fns:
            cleanup()
        self.cleanup_fns.clear()
        _maybe_set_eval_frame(_callback_from_stance(self.prior))
        self.prior = unset
        return None

    def __call__(self, fn: Any) -> Any:
        # [FUNCTION DOC] Decorator entry point. Handles staticmethod by wrapping the underlying function.
        # With caching precompile, loads or initializes the compile package for the function. Unwraps nested
        # Dynamo wrappers; tags GraphModule code with its module. For an nn.Module returns an
        # OptimizedModule (raising RuntimeError for scripted modules); for a class patches its call (and
        # nn.Module call implementation) with the decorator; for other callables decides whether to wrap
        # for inlining (skipped/built-in/polyfilled targets) and returns compile_wrapper, which runs the
        # function with the callback installed. Raises AssertionError for non-callables and RuntimeError
        # when an active callback is applied to an object without code. Side effects: updates
        # always_optimize_code_objects and may use the Dynamo cache.
        if isinstance(fn, staticmethod):
            return staticmethod(self(fn.__func__))

        # public api for compiler config/options
        def get_compiler_config() -> CompilerConfig | None:
            # [FUNCTION DOC] HELPER: Returns the compiler config stored on the context.
            return self.compiler_config

        from .package import DynamoCache

        # If self._package is lazily initialized, we should check the dynamo cache now
        if config.caching_precompile:
            if self._package is not None and not self._package.is_initialized():
                fn_key = fn.forward if isinstance(fn, torch.nn.Module) else fn
                result = DynamoCache.load(fn_key)
                if result is None:
                    # Create a fresh CompilePackage
                    self._package.initialize(fn_key, None, ignore_inlined_sources=False)
                else:
                    try:
                        self._package.initialize(
                            fn_key, result.dynamo, ignore_inlined_sources=False
                        )
                        self._package.install(result.backends)
                    except RuntimeError:
                        log.warning(
                            "Failed to load entry from dynamo cache", exc_info=True
                        )
                        self._package.initialize(
                            fn_key, None, ignore_inlined_sources=False
                        )

        fn = innermost_fn(fn)

        def aot_compile(example_inputs: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
            # [FUNCTION DOC] HELPER: Ahead-of-time compiles the function for example inputs (args and
            # kwargs) with the context's backend and hooks. Raises RuntimeError if inductor caches are
            # force-disabled, fullgraph is off, or the callback is not callable; AssertionError without
            # hooks. Returns the compiled function.
            from torch._dynamo.aot_compile import aot_compile_fullgraph

            if torch._inductor.config.force_disable_caches:
                raise RuntimeError(
                    "Cannot precompile with torch._inductor.config.force_disable_caches=True; caching is required."
                )

            if not self.fullgraph:
                raise RuntimeError(
                    "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
                )

            if not callable(self.callback):
                raise RuntimeError("aot compile requires a callable dynamo callback.")

            if self._hooks is None:
                raise AssertionError("hooks must not be None for AOT compilation")

            return aot_compile_fullgraph(
                fn,
                example_inputs,
                hooks=self._hooks,
                backend=innermost_backend(self.callback),
                dynamic=self._dynamic,
            )

        # add context containing GraphModule to any GraphModule forward functions
        if isinstance(fn, GraphModule):
            # add context containing GraphModule to any GraphModule forward functions
            code_context.get_context(fn.forward.__code__)["orig_graphmodule"] = (
                weakref.ref(fn)
            )

        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            if type(fn) is torch.jit._script.RecursiveScriptModule:
                raise RuntimeError(
                    "torch.compile does not support compiling torch.jit.script or "
                    "torch.jit.freeze models directly.\n\n"
                    "Workaround: compile the original eager module instead:\n"
                    "  model = torch.nn.Linear(3, 3)\n"
                    "  compiled_model = torch.compile(model)  # compile the eager module\n\n"
                    "torch.jit.script and torch.jit.freeze are deprecated in favor of "
                    "torch.compile. See https://pytorch.org/docs/main/jit.html for details."
                )
            mod = fn
            new_mod = OptimizedModule(mod, self)
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod.forward
            new_mod._torchdynamo_wrapper_id = id(new_mod)

            # when compiling torch.nn.Module,
            # provide public api OptimizedModule.get_compiler_config()
            # check mod, not new_mod: OptimizedModule.__getattr__ delegates to mod
            if inspect.getattr_static(mod, "get_compiler_config", None) is not None:
                raise AssertionError(
                    "new_mod already has a get_compiler_config attribute"
                )
            new_mod.get_compiler_config = get_compiler_config

            return new_mod

        if inspect.isclass(fn):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl.
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        if not callable(fn):
            raise AssertionError(
                f"A callable function is expected, but {type(fn)} is provided."
            )

        # NOTE [Top-level TorchInGraph and polyfilled functions]
        # Some callables (e.g. torch.exp) are represented as TorchInGraphFunctionVariable
        # when traced inside a frame. When such a function is passed directly to
        # torch.compile, we detect it here so we can force it through wrap_inline.
        # Similarly, functions registered via substitute_in_graph have a polyfill
        # that Dynamo can trace, so they also need wrap_inline.
        from .variables import TorchInGraphFunctionVariable

        rule = trace_rules.lookup(fn)
        top_level_in_graph = isinstance(rule, type) and issubclass(
            rule, TorchInGraphFunctionVariable
        )
        has_polyfill = trace_rules.is_polyfilled_callable(fn)

        try:
            filename = inspect.getsourcefile(fn)
        except TypeError:
            filename = None
        should_wrap_module_call_impl = OptimizedModule._should_wrap_module_call_impl(fn)
        if config.debug_force_nested_calls and filename not in DONT_WRAP_FILES:
            fn = external_utils.wrap_inline(fn)
            # Create a new code object for `fn` so that functions have different
            # recompilation caches.
            # Copy hack since deepcopy doesn't actually give a new code object
            fn.__code__ = fn.__code__.replace(co_varnames=fn.__code__.co_varnames)  # type: ignore[attr-defined]
        elif config.wrap_top_frame or (
            (
                # exec/eval'd Python functions also report no source file
                # but can be traced directly; wrapping collapses them onto
                # wrap_inline's shared `inner` code (#124269).
                (filename is None and not inspect.isfunction(fn))
                or trace_rules.check(fn)
                or top_level_in_graph
                or has_polyfill
            )
            and (
                getattr(fn, "__name__", "")
                not in ["_call_impl", "_wrapped_call_impl", "_lazy_forward"]
                or should_wrap_module_call_impl
            )
            and filename not in DONT_WRAP_FILES
        ):
            # call to a builtin without a frame for us to capture
            fn = external_utils.wrap_inline(fn)

        def do_nothing(*arg: Any, **kwargs: Any) -> None:
            # [FUNCTION DOC] HELPER: Default no-op callback.
            pass

        callback: Callable[..., Any] = do_nothing
        if hasattr(self, "callback"):
            callback = self.callback  # type: ignore[assignment]

        is_jit_tracing = torch._C._is_tracing
        is_fx_symbolic_tracing = torch.fx._symbolic_trace.is_fx_symbolic_tracing

        @functools.wraps(fn)
        def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] The compiled entry point for a decorated function. Disables the eval-frame
            # callback, then either runs the function eagerly (nested compile inside compiler tracing,
            # fake-kernel invocation, exporting, FX symbolic tracing) or runs it with the stance-adjusted
            # callback installed, after applying enter hooks, isolate-recompile id, skip-guard-eval and
            # graph-break settings and snapshotting dispatch keys and functorch layer depth. Raises
            # RuntimeError for FX/jit tracing of compiled code, and for fullgraph runs with no compiled
            # frames (listing skip reasons). Converts Dynamo user errors to clean tracebacks unless verbose.
            # In all cases restores global state (callback, flags, layer stack, dispatch keys) afterward.
            # NB: function calls here could change global state (e.g. random state)
            # and that can result in different behavior between eager and compiled!
            # In particular, we don't have control over internal functions like justknobs_check
            # called in _maybe_set_eval_frame.
            # Unlike in eval_frame_cpp.cpp/convert_frame.py, we don't attempt to restore global state
            # due to additional overhead costs.
            prior = set_eval_frame(None)
            prior_error_on_nested_compile: bool | None = None
            fullgraph_count_enabled = False
            # Fake propagation and AOT metadata collection execute user
            # subclass code under active fake/functional modes. Nested compile
            # must run the original function in those regions.  Still honor
            # explicit/internal requests to compile during FX tracing: HOP export
            # depends on its internal compile to preserve HOPs, and some tests
            # intentionally enable force_compile_during_fx_trace to get nested
            # subgraphs.
            if (
                _is_eager_on_nested_compile()
                and not config.force_compile_during_fx_trace
            ):
                from torch._higher_order_ops.utils import _in_hop_compile

                if not _in_hop_compile():
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        _maybe_set_eval_frame(prior)

            if self.fullgraph:
                prior_error_on_nested_compile = set_fullgraph_error_on_nested_compile(
                    torch._dynamo.config.error_on_dynamo_callback_in_fullgraph_compiled_code
                )
                if not self.export:
                    fullgraph_count_enabled = set_fullgraph_compiled_frame_count(0) < 0
                    if fullgraph_count_enabled:
                        reset_skip_reasons()
            try:
                # We shouldn't compile inside kernel invocation.
                if tracing_context := torch._guards.TracingContext.try_get():
                    if (
                        tracing_context.fake_mode is not None
                        and tracing_context.fake_mode.in_kernel_invocation
                    ):
                        return fn(*args, **kwargs)
                # Skip nested compile during export (but not HOP internal compile)
                # Only skip if there's an active TracingContext (nested), not for top-level export
                if (
                    torch.compiler.is_exporting()
                    and not config.force_compile_during_fx_trace
                ):
                    from torch._higher_order_ops.utils import _in_hop_compile

                    if not _in_hop_compile():
                        if torch._guards.TracingContext.try_get() is not None:
                            return fn(*args, **kwargs)
                # Skip nested compile - just inline the function
                if (
                    is_fx_symbolic_tracing()
                    and not config.force_compile_during_fx_trace
                ):
                    if config.error_on_nested_fx_trace:
                        raise RuntimeError(
                            "Detected that you are using FX to symbolically trace "
                            "a dynamo-optimized function. This is not supported at the moment."
                        )
                    else:
                        return fn(*args, **kwargs)

                if is_jit_tracing():
                    raise RuntimeError(
                        "Detected that you are using FX to torch.jit.trace "
                        "a dynamo-optimized function. This is not supported at the moment."
                    )

                cleanups = [enter() for enter in self.enter_exit_hooks]
                prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
                    _is_skip_guard_eval_unsafe_stance()
                )
                prior_isolate_recompiles_id = set_eval_frame_isolate_recompiles_id(
                    self._isolate_recompiles_id
                )
                prior_error_on_graph_break = None
                if not self.fullgraph and self.error_on_graph_break is not None:
                    current_error_on_graph_break = _get_error_on_graph_break()
                    if current_error_on_graph_break != self.error_on_graph_break:
                        prior_error_on_graph_break = current_error_on_graph_break
                        _set_error_on_graph_break(self.error_on_graph_break)

                # Ensure that if an assertion occurs after graph pushes
                # something onto the DynamicLayerStack then we pop it off (the
                # constructed graph code isn't guarded with try/finally).
                #
                # This used to be a context but putting a `with` here is a noticeable
                # perf regression (#126293)
                saved_dynamic_layer_stack_depth = (
                    torch._C._functorch.get_dynamic_layer_stack_depth()
                )

                _maybe_set_eval_frame(_callback_from_stance(callback))

                # Snapshot the local dispatch key set onto a C++ thread-local
                # stack so it can be restored after the compiled call, matching
                # the old _ForceDispatchKeyGuard behavior. Keeping the snapshot
                # in C++ avoids constructing pybind11 DispatchKeySet /
                # context-manager instances on every compiled call. The restore
                # runs in the outer finally below (after the inner finally, i.e.
                # after pop_dynamic_layer_stack) so the save/restore stack stays
                # balanced even if the inner finally itself raises (e.g. the
                # fullgraph "found no compiled frames" error).
                torch._C._dynamo_save_local_dispatch_key_set()
                try:
                    call_succeeded = False
                    try:
                        result = fn(*args, **kwargs)
                        call_succeeded = True
                    except (Unsupported, UncapturedHigherOrderOpError, UserError) as e:
                        if config.verbose:
                            raise
                        # strip internal tracebacks from causes
                        cur_exn: BaseException = e
                        while cur_exn.__cause__ is not None:
                            cur_exn.__cause__.with_traceback(None)
                            cur_exn = cur_exn.__cause__

                        raise e.with_traceback(
                            None
                        ) from e.__cause__  # User compiler error
                    except ShortenTraceback as e:
                        # Failures in the backend likely don't have useful
                        # data in the TorchDynamo frames, so we strip them out.
                        raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
                    finally:
                        # Restore the dynamic layer stack depth if necessary.
                        set_eval_frame(None)
                        if fullgraph_count_enabled and call_succeeded:
                            count = set_fullgraph_compiled_frame_count(-1)
                            if count == 0 and _stance.stance == "default":
                                skip_reasons = get_skip_reasons()
                                msg = "torch.compile with fullgraph=True found no compiled frames."
                                if skip_reasons:
                                    reasons_str = "\n".join(
                                        f"  - {r}" for r in skip_reasons
                                    )
                                    msg += f" Skipped frames:\n{reasons_str}"
                                else:
                                    msg += (
                                        " Compilation was not attempted and no cached compiled"
                                        " code was found."
                                    )
                                raise RuntimeError(msg)
                        if prior_error_on_graph_break is not None:
                            _set_error_on_graph_break(prior_error_on_graph_break)
                        if prior_error_on_nested_compile is not None:
                            set_fullgraph_error_on_nested_compile(
                                prior_error_on_nested_compile
                            )
                        torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
                            saved_dynamic_layer_stack_depth
                        )

                        set_skip_guard_eval_unsafe(prior_skip_guard_eval_unsafe)
                        set_eval_frame_isolate_recompiles_id(
                            prior_isolate_recompiles_id
                        )
                        for cleanup in cleanups:
                            cleanup()
                finally:
                    # Restore the local dispatch key set snapshotted above. In an
                    # outer finally so it runs even if the inner finally raised,
                    # keeping the save/restore stack balanced.
                    torch._C._dynamo_restore_local_dispatch_key_set()
                return result
            finally:
                if fullgraph_count_enabled:
                    set_fullgraph_compiled_frame_count(-1)
                    dynamo_tls.skip_reasons = None
                _maybe_set_eval_frame(prior)

        # hooks to properly handle inlining
        if self.error_on_graph_break is not None:
            compile_wrapper._torchdynamo_inline = (  # type: ignore[attr-defined]
                external_utils.wrap_inline_with_error_on_graph_break(
                    fn, self.error_on_graph_break
                )
            )
        else:
            compile_wrapper._torchdynamo_inline = fn  # type: ignore[attr-defined]
        # Save the function pointer to find the original callable while nesting
        # of decorators.
        compile_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
        compile_wrapper._torchdynamo_wrapper_id = id(compile_wrapper)  # type: ignore[attr-defined]
        compile_wrapper._isolate_recompiles_id = self._isolate_recompiles_id  # type: ignore[attr-defined]

        # when compiling user function instead of nn.Module
        # provide public api _fn.get_compiler_config()
        if hasattr(compile_wrapper, "get_compiler_config"):
            raise AssertionError(
                "compile_wrapper already has a get_compiler_config attribute"
            )
        compile_wrapper.get_compiler_config = get_compiler_config  # type: ignore[attr-defined]
        if torch._dynamo.config.enable_aot_compile:
            compile_wrapper.aot_compile = aot_compile  # type: ignore[attr-defined]

        # If the function is called using torch._dynamo.optimize decorator, we
        # should prevent any type of skipping.
        if callback not in (None, False):
            if not hasattr(fn, "__code__"):
                raise RuntimeError(
                    textwrap.dedent(
                        """

                        torch._dynamo.optimize is called on a non function object.
                        If this is a callable class, please wrap the relevant code into a function and optimize the
                        wrapper function.

                        >> class CallableClass:
                        >>     def __init__(self) -> None:
                        >>         super().__init__()
                        >>         self.relu = torch.nn.ReLU()
                        >>
                        >>     def __call__(self, x):
                        >>         return self.relu(torch.sin(x))
                        >>
                        >>     def print_hello(self):
                        >>         print("Hello world")
                        >>
                        >> mod = CallableClass()

                        If you want to optimize the __call__ function and other code, wrap that up in a function

                        >> def wrapper_fn(x):
                        >>     y = mod(x)
                        >>     return y.sum()

                        and then optimize the wrapper_fn

                        >> opt_wrapper_fn = torch._dynamo.optimize(wrapper_fn)
                        """
                    )
                )
            always_optimize_code_objects[fn.__code__] = True

        return compile_wrapper


class OptimizeContext(_TorchDynamoContext):
    # [CLASS DOC] Context for full compilation (torch.compile / optimize). Adds generation-tagging on enter,
    # patches torch internals via TorchPatcher, and optionally enables compiled autograd on enter. Marks
    # decorated results as torch-compiled.
    def __init__(
        self,
        callback: DynamoCallback,
        backend_ctx_ctor: Callable[[], contextlib.AbstractContextManager[Any]],
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: bool | None = None,
        export: bool = False,
        dynamic: bool | None = None,
        compiler_config: CompilerConfig | None = None,
        rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator] | None = None,
        package: CompilePackage | None = None,
        hooks: Hooks | None = None,
        isolate_recompiles: bool = False,
        dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the base context with an on_enter hook installing generation tagging and
        # TorchPatcher.patch as patch_fn. If compiled autograd is configured, adds an enter hook that
        # enables compiled autograd with a compiler rebuilt via rebuild_ctx (raises AssertionError if
        # rebuild_ctx is missing at that time).
        def on_enter() -> None:
            # [FUNCTION DOC] HELPER: Installs generation tagging on module init.
            install_generation_tagging_init()

        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
            first_ctx=first_ctx,
            fullgraph=fullgraph,
            error_on_graph_break=error_on_graph_break,
            export=export,
            dynamic=dynamic,
            compiler_config=compiler_config,
            package=package,
            hooks=hooks,
            isolate_recompiles=isolate_recompiles,
            dynamic_shapes=dynamic_shapes,
        )

        if config.compiled_autograd:
            _dynamic = self._dynamic
            if _dynamic is None:
                _dynamic = not torch._dynamo.config.assume_static_by_default

            def call_compiled_autograd() -> functools.partial[bool | None]:
                # [FUNCTION DOC] HELPER: Enter hook that enables compiled autograd using a rebuilt
                # compile context and returns a cleanup that exits it.
                if rebuild_ctx is None:
                    raise AssertionError(
                        "rebuild_ctx must not be None for compiled autograd"
                    )
                compiler_fn = rebuild_ctx()
                ctx = torch._dynamo.compiled_autograd._enable(
                    compiler_fn,
                    # pyrefly: ignore [bad-argument-type]
                    dynamic=_dynamic,
                    ignore_active_disable_ctx=False,
                )
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_compiled_autograd)

    def __call__(self, fn: Any) -> Any:
        # [FUNCTION DOC] Decorates fn via the base class and tags the result as created by torch.compile.
        result = super().__call__(fn)
        result._is_torch_compile = True  # type: ignore[attr-defined]
        return result

    def __reduce__(
        self,
    ) -> tuple[type[OptimizeContext], tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC] Pickle support: returns class, constructor args (callback, backend context
        # constructor, first_ctx) and a state mapping (export, dynamic, compiler_config).
        # KNOWN ISSUE: the state mapping is returned as the third element, which pickle applies as object
        # state, not as constructor keyword arguments; fullgraph/hooks etc. are not preserved.
        return (
            self.__class__,
            (self.callback, self._backend_ctx_ctor, self.first_ctx),
            {
                "export": self.export,
                "dynamic": self._dynamic,
                "compiler_config": self.compiler_config,
            },
        )


class RunOnlyContext(_TorchDynamoContext):
    # [CLASS DOC] Context that runs only already-compiled code (callback False): cache hits execute but no
    # new compilation happens. Bumps the module generation counter on enter.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base context with callback False and an on_enter hook.
        # cudagraph trees relies on generation increment
        def on_enter() -> None:
            # [FUNCTION DOC] HELPER: Increments the global generation counter (used by cudagraph trees).
            torch._dynamo.mutation_guard.GenerationTracker.generation += 1

        super().__init__(callback=False, on_enter=on_enter)

    def __reduce__(self) -> tuple[type[RunOnlyContext], tuple[Any, ...]]:
        # [FUNCTION DOC] Pickle support: reconstruct with no arguments.
        return (self.__class__, ())


class DisableContext(_TorchDynamoContext):
    # [CLASS DOC] Context/decorator that disables Dynamo (callback None) for the wrapped code. State: msg
    # (reason string attached to wrapped functions) and wrapping (whether to copy metadata from the
    # wrapped function). Wrapped functions are tagged so Dynamo skips them.
    def __init__(self, msg: str | None = None, wrapping: bool = True) -> None:
        # [FUNCTION DOC] Initializes the base context with callback None and stores msg and wrapping.
        super().__init__(callback=None)
        self.msg = msg
        self.wrapping = wrapping

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] Decorates fn so Dynamo does not trace it. An nn.Module becomes an OptimizedModule;
        # a class gets its init, call (and nn.Module call implementation) disabled; any other callable is
        # wrapped in a function that turns off the eval-frame callback while running (re-enabling it only if
        # the stance requires, and annotating the FX graph when exporting), then restores the prior
        # callback. The wrapper carries disable markers, the message and original-callable links.
        # Raises AssertionError if fn is not callable.
        # Earlier this code was in the base class _TorchDynamoContext. But we
        # moved it here to have better code organization. For disable, we just
        # want the callback to be None. We don't have to check trace_rules or
        # create any wrapper.
        fn = innermost_fn(fn)

        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            new_mod._torchdynamo_orig_callable = mod.forward
            new_mod._torchdynamo_wrapper_id = id(new_mod)
            return new_mod

        if isinstance(fn, type):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            # Disable on init is useful for reconstruction of bytecodes where we
            # want to prevent Dynamo from tracing into the init function. Check
            # test_reconstruction in test_model_output.py.
            cls_obj.__init__ = self(cls_obj.__init__)  # type: ignore[misc]
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl. Disable it.
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        if not callable(fn):
            raise AssertionError(
                f"A callable function is expected, but {type(fn)} is provided."
            )

        def _fn(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Disabled-call wrapper. Turns off the eval-frame callback, optionally
            # reinstalls the stance-derived callback, calls fn (inside an FX annotation when exporting,
            # except for the dispatch hook), and restores the previous callback in a finally block.
            # Hot path (e.g. around custom operators); keep it minimal.
            prior = set_eval_frame(None)
            try:
                # disable => self.callback is None; _callback_from_stance(None) is
                # None ("off") for most stances but False (run-only) for
                # eager_on_recompile. Install only when non-None (skips the
                # justknob-guarded _maybe_set_eval_frame on the common path).
                callback = _callback_from_stance(self.callback)
                if callback is not None:
                    _maybe_set_eval_frame(callback)
                try:
                    # Only export needs the annotation work.
                    if torch.compiler.is_exporting():
                        fn_name = getattr(fn, "__name__", type(fn).__name__)
                        # Skip annotation for __torch_dispatch__ (internal detail).
                        if fn_name != "__torch_dispatch__":
                            with fx_traceback.annotate(
                                {
                                    "_torchdynamo_disable": True,
                                    "_torchdynamo_disable_recursive": True,
                                    "_torchdynamo_disable_method": fn_name,
                                }
                            ):
                                return fn(*args, **kwargs)
                    return fn(*args, **kwargs)
                finally:
                    if callback is not None:
                        set_eval_frame(None)
            finally:
                # Restore via _maybe_set_eval_frame so re-installing honors the
                # enable_compiler_set_eval_frame killswitch; skip when prior None.
                if prior is not None:
                    _maybe_set_eval_frame(prior)

        # Under some circumstances (e.g. precompile) we can end up calling @disable
        # decorator in generated bytecode and trigger recompile. This is due to the
        # fact that the old callback from torch.compile() is still active and under
        # this circumstance we will trigger a failure with set_stance("fail_on_recompile").
        # Therefore we want to skip calling into any frame in this case.
        if self.wrapping:
            _fn = functools.wraps(fn)(_fn)

        _fn._torchdynamo_disable = True  # type: ignore[attr-defined]
        _fn._torchdynamo_disable_msg = self.msg  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
        _fn._torchdynamo_wrapper_id = id(_fn)  # type: ignore[attr-defined]

        _fn._torchdynamo_disable_recursive = True  # type: ignore[attr-defined]

        return _fn

    def __reduce__(self) -> tuple[type[DisableContext], tuple[Any, ...]]:
        # [FUNCTION DOC] Pickle support: reconstruct with no arguments.
        # KNOWN ISSUE: msg and wrapping are not preserved across pickling.
        return (self.__class__, ())


def _optimize_catch_errors(
    compile_fn: convert_frame.ConvertFrameProtocol,
    hooks: Hooks,
    backend_ctx_ctor: Callable[
        [], contextlib.AbstractContextManager[Any]
    ] = null_context,
    fullgraph: bool = False,
    error_on_graph_break: bool | None = None,
    export: bool = False,
    dynamic: bool | None = None,
    compiler_config: CompilerConfig | None = None,
    rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator] | None = None,
    package: CompilePackage | None = None,
    isolate_recompiles: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> OptimizeContext:
    # [FUNCTION DOC] HELPER: Builds a top-level OptimizeContext whose callback is the given frame converter
    # wrapped by the error-catching wrapper, forwarding all options (fullgraph, export, dynamic, config,
    # package, hooks, isolate_recompiles, dynamic_shapes) and marking it as the first context.
    return OptimizeContext(
        convert_frame.catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        fullgraph=fullgraph,
        error_on_graph_break=error_on_graph_break,
        export=export,
        dynamic=dynamic,
        compiler_config=compiler_config,
        rebuild_ctx=rebuild_ctx,
        package=package,
        hooks=hooks,
        isolate_recompiles=isolate_recompiles,
        dynamic_shapes=dynamic_shapes,
    )


def get_compiler_fn(
    compiler_fn: str | Callable[..., Any] | None,
) -> WrapBackendDebug:
    # [FUNCTION DOC] Resolves a backend (None, name string, or callable) to a compiler function via the
    # backend registry and wraps it with the debug wrapper, using the backend's compiler_name or the
    # string name. Raises AssertionError if compiler_name is not a string.
    from .repro.after_dynamo import wrap_backend_debug

    if compiler_fn is None:
        # Special case None to avoid crashing in hasattr
        compiler_str = None
    elif hasattr(compiler_fn, "compiler_name"):
        compiler_str = compiler_fn.compiler_name  # type: ignore[union-attr]
        if not isinstance(compiler_str, str):
            raise AssertionError(
                f"Expected compiler_name to be a str, got {type(compiler_str)}"
            )
    elif isinstance(compiler_fn, str):
        compiler_str = compiler_fn
    else:
        compiler_str = None
    compiler_fn = lookup_backend(compiler_fn)  # type: ignore[arg-type]
    return wrap_backend_debug(compiler_fn, compiler_str)


class _NullDecorator(contextlib.nullcontext):  # type: ignore[type-arg]
    # [CLASS DOC] HELPER: No-op context manager and decorator returned when Dynamo is disabled.
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Returns fn unchanged; raises AssertionError if fn is not callable.
        if not callable(fn):
            raise AssertionError(
                f"A callable function is expected, but {type(fn)} is provided."
            )
        return fn


# Make dynamo graph to have same input/output spec as user code
def argument_names(
    f_sig: inspect.Signature,
    args: list[Any] | tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[str]:
    # [FUNCTION DOC] Computes the names of graph inputs matching a call: positional args take the declared
    # parameter names, extra positionals take the varargs name with a numeric suffix, then keyword
    # argument names are appended. Raises AssertionError if there are too many positionals with no varargs,
    # or a required positional or keyword-only parameter is not provided. Returns the list of names.
    def signature_to_fullargspec(sig: inspect.Signature) -> inspect.FullArgSpec:
        # [FUNCTION DOC] HELPER: Converts a Signature into a FullArgSpec (positional names, varargs,
        # varkw, defaults, keyword-only names and defaults, annotations including return).
        # Get a list of Parameter objects from the Signature object
        params = list(sig.parameters.values())
        # Separate positional arguments, keyword-only arguments and varargs/varkw
        args = [
            p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        kwonlyargs = [
            p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        varargs = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL),
            None,
        )
        varkw = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )
        # Get default values for positional arguments and keyword-only arguments
        defaults = tuple(
            p.default
            for p in params
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty
        )
        kwonlydefaults = {
            p.name: p.default
            for p in params
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is not inspect.Parameter.empty
        }
        # Get annotations for parameters and return value
        # pyrefly: ignore [implicit-any]
        annotations = {}
        if sig.return_annotation:
            annotations = {"return": sig.return_annotation}
        for parameter in params:
            annotations[parameter.name] = parameter.annotation
        # Return a FullArgSpec object with the extracted attributes
        return inspect.FullArgSpec(
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations
        )

    fullargspec = signature_to_fullargspec(f_sig)

    # 1. Map `args` 1-to-1 to positional arguments in original signature.
    input_strs = fullargspec.args[: len(args)]

    if len(args) > len(fullargspec.args):
        # 2. If there are more arguments left in `args`, they map to varargs in original
        # signature. Assign names as {varargs}_0, {varargs}_1, ...
        if fullargspec.varargs is None:
            raise AssertionError("More arguments than expected")
        input_strs += [
            f"{fullargspec.varargs}_{i}" for i in range(len(args) - len(input_strs))
        ]
    elif len(args) < len(fullargspec.args):
        # 3. If there are fewer arguments in `args` than `fullargspec.args`,
        # it implies these are arguments either with default values, or provided in
        # `kwargs`. The former can be safely ignored. Because Dynamo.export does not
        # export them as part of the function signature. The latter will be handled
        # in the next step.
        for unprovided_arg in fullargspec.args[
            len(args) : -len(fullargspec.defaults or [])
        ]:
            if unprovided_arg not in kwargs:
                raise AssertionError(f"Missing argument {unprovided_arg}")

    # 4. Keyword arguments provided in `kwargs`.
    input_strs += list(kwargs.keys())

    # 5. Keyword-only arguments with default values if not provided are not exported
    # as part of the function signature.
    for kwonly_arg in fullargspec.kwonlyargs:
        kwonlydefaults = fullargspec.kwonlydefaults or {}
        if kwonly_arg not in kwargs and kwonly_arg not in kwonlydefaults:
            raise AssertionError(f"Missing keyword only argument {kwonly_arg}")

    return input_strs


def check_if_dynamo_supported() -> None:
    # [FUNCTION DOC] Raises RuntimeError on Python 3.15 or newer, or on free-threaded builds older than
    # 3.13.3; otherwise returns.
    if sys.version_info >= (3, 15):
        raise RuntimeError("Python 3.15+ not yet supported for torch.compile")
    elif sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and sys.version_info < (
        3,
        13,
        3,
    ):
        raise RuntimeError(
            "torch.compile is not supported on Python < 3.13.3 built with GIL disabled. "
            "Please use Python 3.13.3+."
        )


def is_dynamo_supported() -> bool:
    # [FUNCTION DOC] Returns true if the Dynamo support check passes, false if it raises.
    try:
        check_if_dynamo_supported()
        return True
    except Exception:
        return False


def check_if_inductor_supported() -> None:
    # [FUNCTION DOC] Raises RuntimeError if Dynamo is unsupported (Inductor has no extra requirements here).
    check_if_dynamo_supported()


def is_inductor_supported() -> bool:
    # [FUNCTION DOC] Returns true if the Inductor support check passes, false if it raises.
    try:
        check_if_inductor_supported()
        return True
    except Exception:
        return False


def check_for_incompatible_configs() -> None:
    # [FUNCTION DOC] Raises AssertionError if suppress_errors and fail_on_recompile_limit_hit are both on.
    # Some of the configs should be mutually exclusive
    if config.suppress_errors and config.fail_on_recompile_limit_hit:
        raise AssertionError(
            "Dynamo configs suppress_error and fail_on_recompile_limit_hit can not both be active at the same time."
        )


def optimize(*args: Any, **kwargs: Any) -> OptimizeContext | _NullDecorator:
    # [FUNCTION DOC] Public entry point: forwards all arguments to _optimize together with a rebuild_ctx
    # callable that re-invokes optimize (used by compiled autograd to build a matching compiler).
    def rebuild_ctx() -> OptimizeContext | _NullDecorator:
        # [FUNCTION DOC] HELPER: Re-calls optimize with the same arguments; if a compiled-autograd kwargs
        # override is configured, maps its fullgraph value onto the nopython keyword (raises AssertionError
        # if the override contains other keys).
        ca_kwargs_override = config.compiled_autograd_kwargs_override
        if ca_kwargs_override:
            # NOTE: The process of translating other `torch.compile` kwargs to `torch._dynamo.optimize` kwargs
            # is more complicated, we will add it in the future when needed.
            if set(ca_kwargs_override.keys()) != {"fullgraph"}:
                raise AssertionError(
                    f"Only `fullgraph` kwarg override is supported for now, but got {ca_kwargs_override.keys()}"
                )
            kwargs["nopython"] = ca_kwargs_override["fullgraph"]
        return optimize(*args, **kwargs)

    return _optimize(rebuild_ctx, *args, **kwargs)


def _optimize(
    rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator],
    backend: str | Callable[..., Any] = "inductor",
    *,
    nopython: bool = False,
    error_on_graph_break: bool | None = None,
    guard_export_fn: Callable[[_guards.GuardsSet], None] | None = None,
    guard_fail_fn: Callable[[GuardFail], None] | None = None,
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    disable: bool = False,
    dynamic: bool | None = None,
    package: CompilePackage | None = None,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> OptimizeContext | _NullDecorator:
    # [FUNCTION DOC] Main Dynamo entry point. Checks Python support and config compatibility, builds Hooks
    # from the guard callbacks and logs API usage. Returns a no-op decorator when disabled (disable flag,
    # TORCHDYNAMO_DISABLE=1 env var, or the enable_dynamo justknob off). With nopython (and no forced leaf
    # graph break) delegates to optimize_assert. Otherwise resolves the backend, creates a precompile
    # package if caching precompile is on, and returns an OptimizeContext around the frame converter.
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
            - Or, a string backend name in `torch._dynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.
        error_on_graph_break: If not None, the current `error_on_graph_break` setting is set to the given value.
            See `torch._dynamo.error_on_graph_break()` for more details on what `error_on_graph_break` means.

            Unlike `nopython=True` (i.e. `fullgraph=True`), there is no guarantee of a single whole-program graph.
            If `nopython` is True, `error_on_graph_break` does nothing.
        disable: If True, turn this decorator into a no-op
        dynamic: If True, upfront compile as dynamic a kernel as possible.  If False,
            disable all dynamic shapes support (always specialize).  If None, automatically
            detect when sizes vary and generate dynamic kernels upon recompile.
        recompile_limit: Maximum number of recompilations for this region.
            If None, uses ``torch._dynamo.config.recompile_limit``.
        isolate_recompiles: If True, this compile call gets its own isolated
            cache so recompilations are tracked independently from other
            compile calls on the same function.

    Example Usage::

        @torch._dynamo.optimize()
        def toy_example(a, b): ...
    """
    check_if_dynamo_supported()
    check_for_incompatible_configs()
    # Note: The hooks object could be global instead of passed around, *however* that would make
    # for a confusing API usage and plumbing story wherein we nest multiple .optimize calls.
    # There is some prior art around this, w/r/t nesting backend calls are enforced to be the same
    # compiler, however, this feels onerous for callback and hooks, and it feels better to give our users an
    # easier to understand UX at the cost of a little more plumbing on our end.
    hooks = Hooks(
        guard_export_fn=guard_export_fn,
        guard_fail_fn=guard_fail_fn,
        guard_filter_fn=guard_filter_fn,
    )
    torch._C._log_api_usage_once("torch._dynamo.optimize")
    if (
        disable
        or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1"
        or (not justknobs_check("pytorch/compiler:enable_dynamo"))
    ):
        return _NullDecorator()

    if nopython and not config.debug_force_graph_break_on_leaf_return:
        return optimize_assert(
            backend,
            dynamic=dynamic,
            hooks=hooks,
            rebuild_ctx=rebuild_ctx,
            package=package,
            recompile_limit=recompile_limit,
            isolate_recompiles=isolate_recompiles,
            dynamic_shapes=dynamic_shapes,
        )

    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    # The backend function is stashed in the callable returned by
    # _optimize_catch_errors in the field _torchdynamo_orig_backend. This can
    # be used by eval_frame.c to insert a guard on the backend.

    # With CachingPrecompile, instantiate an uninitialized CompilePackage
    # which gets initialized by _optimize_catch_errors.__call__ once we have a function
    if config.caching_precompile and package is None:
        from .package import CompilePackage

        package = CompilePackage(fn=None, dynamo=None, ignore_inlined_sources=False)

    return _optimize_catch_errors(
        convert_frame.convert_frame(
            backend,
            hooks,
            package=package,
            recompile_limit=recompile_limit,
        ),
        hooks,
        backend_ctx_ctor,
        fullgraph=False,
        error_on_graph_break=error_on_graph_break
        and not config.debug_force_graph_break_on_leaf_return,
        dynamic=dynamic,
        compiler_config=(
            cast(CompilerConfigProvider, backend).get_compiler_config()
            if hasattr(backend, "get_compiler_config")
            else None
        ),
        rebuild_ctx=rebuild_ctx,
        package=package,
        isolate_recompiles=isolate_recompiles,
        dynamic_shapes=dynamic_shapes,
    )


# TODO(voz): Consider making "explain" output alongside a run / part of a run
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f: Callable[..., Any], *extra_args: Any, **extra_kwargs: Any) -> Any:
    # [FUNCTION DOC] Returns a function that, when called with arguments, resets Dynamo, runs f compiled
    # with a graph-accumulating backend, and returns an ExplainOutput (graphs, counts, break reasons, ops
    # per graph, guards, compile time). If extra args or kwargs are given (deprecated), warns with
    # FutureWarning and runs immediately. Patches the symbolic_convert explain flag during the call.
    from .backends.debugging import ExplainOutput

    def inner(*args: Any, **kwargs: Any) -> ExplainOutput:
        # [FUNCTION DOC] HELPER: Resets Dynamo, runs the compiled f on the arguments collecting graphs,
        # break reasons, op counts and guards, resets Dynamo again and returns the ExplainOutput summary.
        # TODO(voz): Do we want a decorator for this?
        from . import reset  # type: ignore[attr-defined]

        reset()

        graphs: list[torch.fx.GraphModule] = []
        break_reasons: list[Any] = []
        op_count: int = 0
        ops_per_graph: list[list[Target]] = []
        out_guards: list[_guards.Guard] = []

        def dynamo_graph_accumulating_compiler(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> Callable[..., Any]:
            # [FUNCTION DOC] HELPER: Backend that records each graph's details into the enclosing lists
            # and counters and returns the graph's forward unchanged.
            from .backends.debugging import _explain_graph_detail

            nonlocal graphs
            nonlocal op_count
            nonlocal ops_per_graph
            nonlocal break_reasons

            gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(
                gm, graphs, op_count, ops_per_graph, break_reasons
            )

            return gm.forward

        def guard_export_print(guards: Iterable[_guards.Guard]) -> None:
            # [FUNCTION DOC] HELPER: Guard export hook that appends the guards to the enclosing list.
            nonlocal out_guards
            out_guards.extend(guards)

        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideeffects and reject.
        opt_f(*args, **kwargs)

        graph_count = len(graphs)
        graph_break_count = graph_count - 1
        compile_time = compile_times(repr="str")

        # TODO(voz): Do we want a decorator for this?
        reset()

        return ExplainOutput(
            graphs,
            graph_count,
            graph_break_count,
            break_reasons,
            op_count,
            ops_per_graph,
            out_guards,
            compile_time,
        )

    if extra_args or extra_kwargs:
        warnings.warn(
            "explain(f, *args, **kwargs) is deprecated, use explain(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your explain call in the future if your user defined kwargs "
            "conflict with future kwargs added to explain(f).",
            FutureWarning,
            stacklevel=2,
        )
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner


class FlattenInputOutputSignature(torch.fx.Transformer):
    # [CLASS DOC] FX transformer that rewrites a Dynamo graph to take all flattened user inputs (one
    # placeholder per flat arg) and return all flattened user outputs, re-adding inputs Dynamo pruned and
    # constants it inlined. State: new_args (new placeholders), old_args_gen (iterator mapping old
    # placeholders to matched new ones), matched output positions and flat results. Copies value metadata.
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: list[Any],
        matched_input_elements_positions: list[int],
        flat_results: Sequence[Any],
        matched_output_elements_positions: list[int],
        example_fake_inputs: list[torch.Tensor],
        flat_args_dynamic_dims: list[set[int]],
        fake_mode: fake_tensor.FakeTensorMode | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates one placeholder per flat arg. Matched args get the Dynamo example fake
        # value; unmatched tensors get a fake tensor built with static or dynamic dims per
        # flat_args_dynamic_dims (when a fake mode is given); wrapped ints and other values keep their value.
        # Raises AssertionError if flat_args_dynamic_dims length differs from flat_args length.
        super().__init__(m)

        if len(flat_args_dynamic_dims) != len(flat_args):
            raise AssertionError(
                f"flat_args_dynamic_dims length ({len(flat_args_dynamic_dims)}) "
                f"must match flat_args length ({len(flat_args)})"
            )
        matched_input_elements_to_fake = {
            val: example_fake_inputs[ix]
            for ix, val in enumerate(matched_input_elements_positions)
        }

        self.new_args = []
        for i in range(len(flat_args)):
            arg = super().placeholder(f"arg{i}", (), {})
            if i in matched_input_elements_to_fake:
                arg.node.meta["val"] = matched_input_elements_to_fake[i]
            else:
                # Fill node.meta["val"] with faketensor from the input,
                # if it's not found in matched_input_elements_positions
                if fake_mode is not None and isinstance(flat_args[i], torch.Tensor):
                    # TODO(zhxchen17) Also preserve all the user constraints here.
                    arg.node.meta["val"] = fake_mode.from_tensor(
                        flat_args[i],
                        symbolic_context=StatelessSymbolicContext(
                            dynamic_sizes=[
                                (
                                    DimDynamic.DYNAMIC
                                    if d in flat_args_dynamic_dims[i]
                                    else DimDynamic.STATIC
                                )
                                for d in range(len(flat_args[i].shape))
                            ],
                            constraint_sizes=[None] * len(flat_args[i].shape),
                        ),
                    )
                elif isinstance(flat_args[i], _IntWrapper):
                    arg.node.meta["val"] = flat_args[i].val
                else:
                    arg.node.meta["val"] = flat_args[i]

            self.new_args.append(arg)
        self.old_args_gen = (self.new_args[i] for i in matched_input_elements_positions)
        self.matched_output_elements_positions = matched_output_elements_positions
        self.flat_results = flat_results

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Replaces each old placeholder with the next matched new placeholder, copying
        # value, tensor dict, example value and unbacked bindings metadata from the old node.
        arg = next(self.old_args_gen)
        if "val" in self.current_node.meta:
            arg.node.meta["val"] = self.current_node.meta["val"]
        if "tensor_dict" in self.current_node.meta:
            arg.node.meta["tensor_dict"] = self.current_node.meta["tensor_dict"]
        if "example_value" in self.current_node.meta:
            # NB: intentionally do not use set_example_value
            arg.node.meta["example_value"] = self.current_node.meta["example_value"]
        if "unbacked_bindings" in self.current_node.meta:
            arg.node.meta["unbacked_bindings"] = self.current_node.meta[
                "unbacked_bindings"
            ]
        return arg

    def output(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Builds the output from flat results: matched entries are taken from the Dynamo
        # outputs followed by the new placeholders; unmatched entries must be common constants and are
        # emitted directly (AssertionError otherwise).
        dynamo_result_flat = args[0]
        lookup = [*dynamo_result_flat, *self.new_args]  # type: ignore[misc]
        # pyrefly: ignore [implicit-any]
        new_results_flat = []
        for i in range(len(self.flat_results)):
            if self.matched_output_elements_positions[i] is not None:
                new_results_flat.append(
                    lookup[self.matched_output_elements_positions[i]]
                )
            else:
                const_val = self.flat_results[i]
                if not isinstance(const_val, tuple(common_constant_types)):
                    raise AssertionError(
                        f"Expected const_val to be a common constant type, got {type(const_val)}"
                    )
                new_results_flat.append(const_val)
        return super().output(target, (new_results_flat,), {})

    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Records the current node, runs the base transform, copies value, example value
        # and unbacked bindings metadata to the new node, and renames it to match the old node's name
        # (except for the output node). Returns the resulting proxy.
        self.current_node = n
        result_proxy = super().run_node(n)
        if "val" in self.current_node.meta:
            result_proxy.node.meta["val"] = self.current_node.meta["val"]
        if "example_value" in self.current_node.meta:
            # NB: intentionally do not use set_example_value
            result_proxy.node.meta["example_value"] = self.current_node.meta[
                "example_value"
            ]
        if "unbacked_bindings" in self.current_node.meta:
            result_proxy.node.meta["unbacked_bindings"] = self.current_node.meta[
                "unbacked_bindings"
            ]
        if self.current_node.op != "output":
            result_proxy.node._rename(
                getattr(self.current_node, "name", result_proxy.node.name)
            )
        return result_proxy

    def transform(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Runs the base transform and copies the flat-name-to-FQN mapping and compile id
        # metadata from the source module to the result. Returns the new GraphModule.
        result_gm = super().transform()
        if "dynamo_flat_name_to_original_fqn" in self.module.meta:  # type: ignore[operator]
            result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[  # type: ignore[index]
                "dynamo_flat_name_to_original_fqn"  # type: ignore[index]
            ]
        if "dynamo_compile_id" in self.module.meta:  # type: ignore[operator]
            result_gm.meta["dynamo_compile_id"] = self.module.meta["dynamo_compile_id"]  # type: ignore[index]
        return result_gm


class ExportResult(NamedTuple):
    # [CLASS DOC] Result of export: the captured graph module and the set of guards. Callers destructure
    # it as a pair, so adding fields requires overriding iteration.
    graph_module: torch.fx.GraphModule
    guards: _guards.GuardsSet
    # NB: Do not add new fields without overriding __iter__; people are
    # destructuring so it is BC-breaking


# NOTE: this function only supports graphs created by Dynamo's OutputGraph module
def check_signature_rewritable(graph: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Verifies every placeholder of a Dynamo-built graph is a direct input. Placeholders
    # with recorded user stacks (free variables not passed as inputs) are collected with their access
    # stack; if any exist raises UserError (invalid input) listing them. Raises AssertionError if the graph
    # lacks Dynamo source metadata (not created by OutputGraph).
    # pyrefly: ignore [implicit-any]
    input_errors = []
    for node in graph.graph.find_nodes(op="placeholder"):
        # set in OutputGraph._call_user_compiler
        if not hasattr(node, "_dynamo_source"):
            raise AssertionError("node is missing _dynamo_source attribute")
        if not hasattr(graph, "_source_to_user_stacks"):
            raise AssertionError("graph is missing _source_to_user_stacks attribute")

        # NOTE: We can safely ignore these type warnings if and only if
        # the function is made from OutputGraph (checked in the assertions)
        source = node._dynamo_source  # type: ignore[attr-defined]
        user_stacks = graph._source_to_user_stacks.get(source)  # type: ignore[operator, union-attr]
        if user_stacks is None:
            continue
        if len(user_stacks) == 0:
            raise AssertionError("user_stacks must not be empty")
        # In some cases we may not have a useful stack.  Look for a
        # useful stack
        stack = None
        for s in user_stacks:
            if len(s) == 0:
                continue
            stack = s
            break
        if stack is None:
            msg = f"{source.name}, a closed over free variable"
        else:
            tb = "".join(traceback.format_list(stack))
            extra = ""
            if len(user_stacks) > 1:
                extra = f"(elided {len(user_stacks) - 1} more accesses)"
            msg = f"{source.name}, accessed at:\n{tb}{extra}"
        # TODO: option to print ALL of the stack traces at once
        input_errors.append(msg)

    if input_errors:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Cannot export model which references tensors that are neither "
            "buffers/parameters/constants nor are direct inputs.  For each tensor, if you'd "
            "like this tensor to be an explicit input, add it as a dummy argument "
            "to the top-level model definition you are exporting; if you would "
            "like its value to be embedded as an exported constant, wrap its access "
            "in a function marked with @assume_constant_result.\n\n"
            + "\n\n".join(input_errors),
        )


def check_user_input_output(flat_values: list[Any], error_type: UserErrorType) -> None:
    # [FUNCTION DOC] Checks every flattened value is a supported type (tensor, symbolic scalar, script
    # object, int wrapper, common constant, or registered custom class). None is allowed for inputs only.
    # Raises UserError of the given type naming the offending value type otherwise.
    supported_types = [
        torch.Tensor,
        torch.SymInt,
        torch.SymFloat,
        torch.SymBool,
        torch._C.ScriptObject,
        _IntWrapper,
    ] + list(common_constant_types)

    def is_supported_type(val: Any) -> bool:
        # [FUNCTION DOC] HELPER: True if val is an instance of a supported type or a registered custom class.
        return isinstance(val, tuple(supported_types)) or is_custom_class(type(val))

    value_type = "input" if error_type == UserErrorType.INVALID_INPUT else "output"
    # We only check that the outputs are not None. Inputs can be None.
    for v in flat_values:
        if not is_supported_type(v):
            if error_type == UserErrorType.INVALID_INPUT and v is None:
                continue

            raise UserError(
                error_type,
                f"It looks like one of the {value_type}s with type `{type(v)}` "
                "is not supported or pytree-flattenable. \n"
                f"Exported graphs {value_type}s can only contain the "
                f"following supported types: {supported_types}. \n"
                "If you are using a custom class object, "
                "please register a pytree_flatten/unflatten function "
                "using `torch.utils._pytree.register_pytree_node` or "
                "`torch.export.register_dataclass`.",
            )


def rewrite_signature(
    f_sig: inspect.Signature,
    graph: torch.fx.GraphModule,
    fake_mode: fake_tensor.FakeTensorMode | None,
    flat_args: list[Any],
    in_spec: pytree.TreeSpec,
    example_fake_inputs: list[Any],
    graph_captured_input: Iterable[Any],
    graph_captured_output: Iterable[Any] | None,
    dynamo_traced_result: Any,
    flat_args_dynamic_dims: list[set[int]],
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Rewrites an exported graph so its signature matches the user function: validates flat
    # inputs and traced outputs, matches graph inputs/outputs to flat args by object identity (constants
    # map to no position), runs FlattenInputOutputSignature, installs a pytree code generator with the
    # original argument names and specs, recompiles, and returns the new graph module. Raises
    # AssertionError for unmatched values or missing captured outputs, and UserError for unsupported types.
    orig_args, orig_kwargs = pytree.tree_unflatten(flat_args, in_spec)

    check_user_input_output(flat_args, UserErrorType.INVALID_INPUT)
    flat_results_traced, out_spec_traced = pytree.tree_flatten(dynamo_traced_result)
    check_user_input_output(flat_results_traced, UserErrorType.INVALID_OUTPUT)

    def check_optional_input_and_error(f_sig: inspect.Signature) -> None:
        # [FUNCTION DOC] HELPER: If any parameter has a default value, logs an error and raises a graph
        # break (unsupported: optional function input); otherwise returns normally.
        # Check if function has optional input.
        for name, param in f_sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                import torch._dynamo.graph_break_hints as graph_break_hints
                from torch._dynamo.exc import unimplemented

                log.error(
                    "Parameter %s is optional with a default value of %s",
                    name,
                    param.default,
                )
                unimplemented(
                    gb_type="rewrite_signature: cannot trace optional function input",
                    context="",
                    explanation=f"Parameter {name} is optional with a default value of {param.default}. This is not supported yet.",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

    def produce_matching(
        debug_type: str, sources: Iterable[Any], candidates: Iterable[Any]
    ) -> list[int | None]:
        # [FUNCTION DOC] HELPER: For each candidate returns the index of the same object (by identity) in
        # sources, or no index for common constants. If a non-constant candidate is not found, checks for
        # optional inputs (when matching inputs) and raises AssertionError asking for a bug report.
        matched_elements_positions: list[int | None] = []
        dict_of_source_vals = {}
        for i, val in enumerate(sources):
            dict_of_source_vals[id(val)] = i

        for val in candidates:
            if isinstance(val, tuple(common_constant_types)):
                matched_elements_positions.append(None)
            elif id(val) not in dict_of_source_vals:
                if debug_type == "inputs":
                    check_optional_input_and_error(f_sig)
                raise AssertionError(
                    f"Unexpectedly found a {type(val)} in the {debug_type}.\n"
                    'Please file an issue along with a paste of the logs from TORCH_LOGS="+export"',
                )
            else:
                matched_elements_positions.append(dict_of_source_vals[id(val)])

        return matched_elements_positions

    matched_input_elements_positions = produce_matching(
        "inputs", flat_args, graph_captured_input
    )

    if graph_captured_output is None:
        raise AssertionError("graph_captured_output must not be None")
    matched_output_elements_positions = produce_matching(
        "outputs", list(graph_captured_output) + flat_args, flat_results_traced
    )

    new_graph = FlattenInputOutputSignature(
        graph,
        flat_args,
        matched_input_elements_positions,  # type: ignore[arg-type]
        flat_results_traced,
        matched_output_elements_positions,  # type: ignore[arg-type]
        example_fake_inputs,
        flat_args_dynamic_dims,
        fake_mode,
    ).transform()

    new_graph.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            argument_names(f_sig, orig_args, orig_kwargs),
            in_spec,
            out_spec_traced,
        )
    )
    new_graph.recompile()
    return new_graph


def export(
    f: Callable[..., Any],
    *extra_args: Any,
    aten_graph: bool = False,
    pre_dispatch: bool = False,
    decomposition_table: dict[torch._ops.OpOverload, Callable[..., Any]] | None = None,
    tracing_mode: str = "symbolic",
    dynamic_shapes: _DynamicShapesInput = None,
    specialize_float: bool = True,
    assume_static_by_default: bool = False,
    same_signature: bool = True,
    disable_constraint_solver: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    _log_export_usage: bool = True,
    constraints: list[Constraint] | None = None,
    **extra_kwargs: Any,
) -> Callable[..., ExportResult]:
    # [FUNCTION DOC] Legacy Dynamo export. Returns a function that traces f with example inputs into a
    # single FX graph and guards, returning an ExportResult. Raises SkipTest when leaf graph breaks are
    # forced; ValueError if a ShapesSpec is combined with constraints or deferred runtime asserts. If extra
    # args/kwargs are given (deprecated), warns with FutureWarning and runs immediately. Logs export usage.
    # Options select ATen-level graph (optionally pre-dispatch with decompositions), symbolic tracing,
    # float specialization, signature rewriting, and constraint solving.
    """
    Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

    Args:
        f (callable): A PyTorch function to be exported.

        aten_graph (bool): If True, exports a graph with ATen operators.
        If False, exports a graph with Python operators. Default is False.

        pre_dispatch (bool): If True, exports a graph with ATen operators,
        but before any logic in the PyTorch dispatcher has run.
        This can be useful if you want to apply further transformations on a graph before running it
        through autograd, autocast, or any other functionalities that are integrated into the dispatcher.
        This flag is only valid if aten_graph=True is set.
        Default is False.

        decomposition_table (dict): A dictionary that maps operators to their decomposition functions.
        Required if aten_graph or tracing_mode is specified. Default is None.

        tracing_mode (str): If "symbolic", turn on dynamic shapes support. Default is "symbolic".

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

         **ShapesSpec API.** ``dynamic_shapes`` may also be a
         :class:`torch.fx.experimental.dynamic_spec.ShapesSpec` (or its
         shorthand :class:`torch.fx.experimental.dynamic_spec.ParamsSpec`) --
         the same spec API exposed via ``dynamic_shapes=`` in
         :func:`torch.compile`. See :func:`torch.export.export` for usage
         and semantics, and :mod:`torch.fx.experimental.dynamic_spec` for
         full details.

        same_signature (bool): If True, rewrite the returned graph's signature to be the same as f.

        disable_constraint_solver (bool): Whether the dim constraint solver must be disabled.

    Returns:
        A function that given args and kwargs, returns a tuple of (graph, guards)
        Graph: An FX graph representing the execution of the input PyTorch function with the provided arguments and options.
        Guards: The guards we accumulated during tracing f above

    Raises:
        AssertionError: If decomposition_table is specified without setting aten_graph=True,
        or if graph breaks during tracing in export.

        AssertionError: If Dynamo input and output is not consistent with traced input/output.

    Note - this headerdoc was authored by ChatGPT, with slight modifications by the author.
    """
    if config.debug_force_graph_break_on_leaf_return:
        raise unittest.SkipTest("Cannot force graph break on export")

    # `dynamic_shapes` is overloaded: it accepts the Dim-based dict/tuple/list
    # spec, OR the new ShapesSpec/ParamsSpec API. If the latter is passed, we
    # route it through dynamo's `dynamic_shapes` mechanism and skip the legacy
    # constraint processing.
    from torch.fx.experimental.dynamic_spec import (
        _SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG,
    )

    dyn_spec: ShapesSpec | ParamsSpec | None = None
    if isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec)):
        if constraints:
            raise ValueError(
                "`dynamic_shapes=ShapesSpec(...)` cannot be combined with "
                "`constraints`. ShapesSpec controls dynamic behavior on its own."
            )
        if prefer_deferred_runtime_asserts_over_guards:
            raise ValueError(_SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG)
        # ParamsSpec is normalized to ShapesSpec downstream in OptimizeContext.
        dyn_spec = dynamic_shapes
        dynamic_shapes = None

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_dynamo"})

    # Deal with "local variable referenced before assignment"
    _f = f
    _specialize_float = specialize_float
    _assume_static_by_default = assume_static_by_default
    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> ExportResult:
        # [FUNCTION DOC] HELPER: Performs the export on concrete arguments. Builds dimension constraints,
        # flattens inputs, clears the code cache, traces under patched config with a capturing backend
        # that must emit exactly one graph, then solves dimension constraints and raises constraint
        # violations (including constraints on static values). If no graph was produced, builds a trivial
        # one from tensor inputs (requires same_signature). Optionally re-traces to ATen with make_fx
        # (wrapping cond argument mismatch as a UserError), and rewrites the signature to match the
        # function. Raises AssertionError for illegal option combinations or multiple graphs. Returns
        # ExportResult.
        if not _constraints:
            combined_args = _combine_args(_f, args, kwargs)
            constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)
        else:
            constraints = _constraints

        f = _f
        specialize_float = _specialize_float
        assume_static_by_default = _assume_static_by_default
        check_if_dynamo_supported()
        torch._C._log_api_usage_once("torch._dynamo.export")
        if decomposition_table is not None:
            if not aten_graph:
                raise AssertionError(
                    "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
                )
        if pre_dispatch:
            if not aten_graph:
                raise AssertionError(
                    "pre_dispatch=True can only be used when aten_graph=True"
                )
        f = innermost_fn(f)
        call_to_inspect = f.forward if isinstance(f, torch.nn.Module) else f
        original_signature = inspect.signature(call_to_inspect)  # type: ignore[arg-type]
        graph = None
        out_guards = None
        graph_captured_input = None
        graph_captured_result: tuple[torch.Tensor, ...] | None = None
        fake_mode = None
        result_traced = None

        def guard_export_print(guards: _guards.GuardsSet) -> None:
            # [FUNCTION DOC] HELPER: Stores the exported guards; raises AssertionError if called twice.
            nonlocal out_guards
            if out_guards is not None:
                raise AssertionError(
                    "whole graph export entails exactly one guard export"
                )
            out_guards = guards

        example_inputs: list[Any] = []

        def dynamo_normalization_capturing_compiler(
            gm: torch.fx.GraphModule, inner_example_inputs: list[Any]
        ) -> Callable[..., Any]:
            # [FUNCTION DOC] HELPER: Backend used for export. Saves the single graph (AssertionError if a
            # second one is emitted), the Dynamo fake mode and the example inputs, and returns a wrapper
            # that records the real graph inputs and fake-evaluates the graph to capture its outputs.
            nonlocal graph
            if graph is not None:
                raise AssertionError(
                    "Tried to emit a second graph during export. Tracing through 'f' must produce a single graph."
                )
            graph = gm

            nonlocal fake_mode, example_inputs
            # NB: do NOT pass inner_example_inputs here, we are detecting the
            # Dynamo allocated fake mode, which should be DISTINCT from a
            # potential outer ambient fake mode which the user provided.
            # example_inputs is always the user specified inputs, so they
            # would have the wrong fake mode attached to them
            fake_mode = _guards.detect_fake_mode()
            example_inputs = inner_example_inputs

            def result_capturing_wrapper(*graph_inputs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Records the graph inputs, fakifies parameters, buffers and
                # inputs with the ambient (or Dynamo) fake mode, runs the graph functionally under the
                # python dispatcher, stores and returns the fake result. Raises AssertionError if the
                # graph or fake mode is missing.
                nonlocal graph_captured_result
                nonlocal graph_captured_input

                graph_captured_input = graph_inputs
                if graph is None:
                    raise AssertionError(
                        "graph must not be None in result_capturing_wrapper"
                    )

                named_parameters = dict(graph.named_parameters(remove_duplicate=False))
                named_buffers = dict(graph.named_buffers(remove_duplicate=False))

                ambient_fake_mode = (
                    _guards.detect_fake_mode(graph_inputs)
                    if _guards.detect_fake_mode(graph_inputs) is not None
                    else fake_mode
                )

                # We reran fake tensor propagation, but we didn't do
                # anything with the resulting unbacked SymInts.  Drop them
                # from the pending list.
                # NB: this is wrong if graph_captured_result has
                # data-dependent output size!
                ignore_fresh_unbacked = null_context()
                if ambient_fake_mode is None:
                    raise AssertionError("ambient_fake_mode must not be None")
                if shape_env := ambient_fake_mode.shape_env:
                    ignore_fresh_unbacked = shape_env.ignore_fresh_unbacked_symbols()  # type: ignore[assignment]

                with (
                    ambient_fake_mode,
                    enable_python_dispatcher(),
                    ignore_fresh_unbacked,
                ):
                    params_and_buffers = {
                        **named_parameters,
                        **named_buffers,
                    }
                    fake_params_buffers = {}

                    for name, value in params_and_buffers.items():
                        fake_params_buffers[name] = ambient_fake_mode.from_tensor(
                            value, static_shapes=True
                        )

                    from torch._export.non_strict_utils import (
                        key_path_to_source,
                        KeyPath,
                    )

                    def fakify_with_ambient(
                        path: KeyPath, t: torch.Tensor | _IntWrapper | Any
                    ) -> Any:
                        # [FUNCTION DOC] HELPER: Converts a graph input to a fake value: tensors become
                        # static-shape fake tensors; int wrappers with dynamic/auto hints become fresh
                        # symbolic ints, other int wrappers their value; everything else is unchanged.
                        if isinstance(t, torch.Tensor):
                            # pyrefly: ignore [missing-attribute]
                            return ambient_fake_mode.from_tensor(t, static_shapes=True)
                        elif isinstance(t, _IntWrapper):
                            if (
                                t.dynamism is not None
                                and isinstance(t.dynamism, _DimHint)
                                and t.dynamism.type
                                in (
                                    _DimHintType.DYNAMIC,
                                    _DimHintType.AUTO,
                                )
                            ):  # type: ignore[union-attr]
                                source = key_path_to_source(path)
                                symint = ambient_fake_mode.shape_env.create_unspecified_symint_and_symbol(  # type: ignore[union-attr]
                                    t.val, source, DimDynamic.DYNAMIC
                                )
                                return symint
                            else:
                                return t.val
                        else:
                            return t

                    fake_graph_inputs = pytree.tree_map_with_path(
                        fakify_with_ambient, graph_inputs
                    )
                    graph_captured_result = torch.func.functional_call(
                        graph,
                        fake_params_buffers,  # type: ignore[arg-type]
                        fake_graph_inputs,  # type: ignore[arg-type]
                    )

                return graph_captured_result

            return result_capturing_wrapper

        # Note: This is needed by rewrite_signature. We need to put it before
        # optimize_assert since user program may mutate the inputs.
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))

        remove_from_cache(f)
        constraint_violation_error = None
        if tracing_mode != "symbolic":
            assume_static_by_default = True
        with (
            config.patch(
                specialize_int=True,
                specialize_float=specialize_float,
                assume_static_by_default=assume_static_by_default,
                automatic_dynamic_shapes=False,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
                constant_fold_autograd_profiler_enabled=True,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                # install_free_tensors ensures that params and buffers are still
                # added as graph attributes, and makes Dynamo emits graphs that
                # follow export pytree-able input requirements
                install_free_tensors=config.install_free_tensors_for_export,
            ),
            _compiling_state_context(),
        ):
            opt_f = optimize_assert(
                dynamo_normalization_capturing_compiler,
                hooks=Hooks(
                    guard_export_fn=guard_export_print,
                    guard_fail_fn=None,
                ),
                export=True,
                export_constraints=constraints,
                dynamic_shapes=dyn_spec,
            )(f)
            # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideeffects and reject.
            try:
                result_traced = opt_f(*args, **kwargs)
            except ConstraintViolationError as e:
                constraint_violation_error = e
        remove_from_cache(f)

        if (
            not disable_constraint_solver
            and (shape_env := getattr(fake_mode, "shape_env", None)) is not None
            and (dim_constraints := shape_env.dim_constraints) is not None
            and not isinstance(
                call_to_inspect, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
            )
            and not trace_rules.check(call_to_inspect)
        ):
            dim_constraints.solve()

            forced_specializations = dim_constraints.forced_specializations()

            msg = dim_constraints.prettify_results(
                original_signature,
                dynamic_shapes,
                constraint_violation_error,
                forced_specializations,
            )
            if constraint_violation_error:
                constraint_violation_error.args = (
                    constraint_violation_error.args[0] + msg,
                )
            else:
                if forced_specializations:
                    constraint_violation_error = ConstraintViolationError(msg)
                else:
                    log.info(
                        "Summary of dimension constraints:%s",
                        msg,
                    )

            # Error if we have any constraints on static values

            for k in shape_env.var_to_range:
                if isinstance(k, sympy.Integer):
                    constraint_violation_error = ConstraintViolationError(
                        f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                        "It appears that you're trying to set a constraint on a "
                        f"value which we evaluated to have a static value of {k}. "
                        'Set TORCH_LOGS="+export" for more information.'
                    )
        if constraint_violation_error:
            raise constraint_violation_error

        if graph is None:
            if not same_signature:
                raise AssertionError(
                    "Failed to produce a graph during tracing as no tensor operations were found and same_signature is False."
                )
            # If the module does not contain any tensor computation, we would create a graph with inputs and outputs.
            # To be consistent with the graph traced by dynamo, `graph` will have only tensor inputs as placeholders
            # and tensor outputs as output nodes. non-tensor inputs and outputs will be added when rewriting signature.
            # We will also construct the `example_inputs`, `graph_captured_input`, and `graph_captured_result` corresponding
            # to `graph`.
            example_inputs = []
            graph_captured_input = ()
            graph_captured_result = ()
            fake_mode = torch._subclasses.FakeTensorMode(
                shape_env=ShapeEnv(), export=True
            )
            if out_guards is None:
                out_guards = _guards.GuardsSet()
            if out_guards is None:  # suppress mypy error
                raise AssertionError("out_guards must not be None")
            parameter_names = list(original_signature.parameters.keys())
            fx_graph = torch.fx.Graph()
            for i, name in enumerate(parameter_names):
                if torch.is_tensor(flat_args[i]):
                    node = fx_graph.placeholder(name)
                    node.meta["val"] = fake_mode.from_tensor(
                        flat_args[i], static_shapes=True
                    )
                    graph_captured_input = graph_captured_input + (flat_args[i],)
                    example_inputs.append(flat_args[i])
            fx_graph.output(graph_captured_result)
            module = torch.nn.Module()
            graph = torch.fx.GraphModule(module, fx_graph)
            log.info(
                "Failed to capture a graph during tracing as no tensor operations were found.:\n\n%s",
                graph.print_readable(print_output=False, colored=True),
            )
        else:
            if out_guards is None:
                raise AssertionError("Failed to produce guards during tracing")
            if fake_mode is None:
                raise AssertionError("fake_mode must not be None after tracing")

            log.info(
                "Dynamo captured graph:\n\n%s",
                graph.print_readable(print_output=False, colored=True),
            )

            # This check need to happened before aten_graph
            # because placeholder's _source_node attribute is not preserved by make_fx
            if same_signature:
                check_signature_rewritable(graph)

        # NB: This is mostly hitting the cache; Dynamo already converted these
        example_fake_inputs = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        if aten_graph:
            # Running graph with interpreter is needed for propagating the stack_trace
            def graph_with_interpreter(*args: Any) -> Any:
                # [FUNCTION DOC] HELPER: Runs the graph with an FX interpreter while preserving node
                # metadata (e.g. stack traces) so make_fx output keeps it.
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph).run(*args)  # type: ignore[arg-type]

            with unset_fake_temporarily(), enable_python_dispatcher(), fake_mode:
                try:
                    graph = make_fx(
                        graph_with_interpreter,
                        decomposition_table=decomposition_table,
                        tracing_mode="real",
                        _allow_non_fake_inputs=True,
                        pre_dispatch=pre_dispatch,
                        _allow_fake_constant=False,
                    )(*example_fake_inputs)
                except CondOpArgsMismatchError as e:
                    # Wrap the internal error to the user-facing error
                    raise UserError(  # noqa: B904
                        UserErrorType.DYNAMIC_CONTROL_FLOW,
                        str(e),
                        case_name="cond_operands",
                    )

            if graph is None:
                raise AssertionError("graph must not be None after tracing")
            for node in graph.graph.find_nodes(op="get_attr"):
                if isinstance(getattr(graph, node.target), torch.Tensor):  # type: ignore[arg-type]
                    node.meta["val"] = fake_mode.from_tensor(
                        getattr(graph, node.target),  # type: ignore[arg-type]
                        static_shapes=True,
                    )

        if same_signature:
            flat_args_dynamic_dims = [
                {
                    c.dim
                    for c in (constraints or ())
                    if (
                        c.t_id == id(x)
                        and not isinstance(c, _RelaxedConstraint)
                        and c.constraint_range.vr.lower != c.constraint_range.vr.upper
                    )
                }
                for x in flat_args
            ]
            graph = rewrite_signature(
                original_signature,
                graph,
                fake_mode,
                flat_args,
                in_spec,
                example_fake_inputs,
                graph_captured_input,  # type: ignore[arg-type]
                graph_captured_result,
                result_traced,  # type: ignore[possibly-undefined]
                flat_args_dynamic_dims,
            )
        return ExportResult(graph, out_guards)

    if extra_args or extra_kwargs:
        warnings.warn(
            "export(f, *args, **kwargs) is deprecated, use export(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your export call in the future if your user defined kwargs "
            "conflict with future kwargs added to export(f).",
            FutureWarning,
            stacklevel=2,
        )
        return inner(*extra_args, **extra_kwargs)  # type: ignore[return-value]
    else:
        return inner


def optimize_assert(*args: Any, **kwargs: Any) -> OptimizeContext:
    # [FUNCTION DOC] Public single-graph entry point: uses a rebuild_ctx passed in kwargs (removing it) or
    # defines one that re-calls itself, then delegates to _optimize_assert.
    if "rebuild_ctx" in kwargs and kwargs["rebuild_ctx"] is not None:
        # called from optimize
        rebuild_ctx = kwargs["rebuild_ctx"]
        del kwargs["rebuild_ctx"]
    else:

        def rebuild_ctx() -> OptimizeContext:
            # [FUNCTION DOC] HELPER: Re-calls optimize_assert with the original arguments.
            return optimize_assert(*args, **kwargs)

    return _optimize_assert(rebuild_ctx, *args, **kwargs)


def _optimize_assert(
    rebuild_ctx: Callable[[], OptimizeContext],
    backend: str | Callable[..., Any] | None,
    *,
    hooks: Hooks = Hooks(None, None, None),
    export: bool = False,
    export_constraints: Any | None = None,
    dynamic: bool | None = None,
    package: CompilePackage | None = None,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> OptimizeContext:
    # [FUNCTION DOC] HELPER: Builds an OptimizeContext that asserts single-graph capture (fullgraph) using
    # the assert-style frame converter with export options; resolves the backend, creates a precompile
    # package if caching precompile is on and none was given.
    """
    Guarantees single-graph capture.
    The same as `torch._dynamo.optimize(backend)` but ignores
    symbolic_convert.error_on_graph_break setting.

    Used for fullgraph=True and export, since we must always error on graph breaks and ignore
    symbolic_convert.error_on_graph_break. Can also be used for testing.
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    if config.caching_precompile and package is None:
        # Create an uninitialized package that will be set/filled by
        # _OptimizeContext.__call__
        # We need to instantiate the object here because the same CompilePackage
        # needs to be shared between convert_frame_assert
        # and OptimizeContext.
        from .package import CompilePackage

        package = CompilePackage(fn=None, dynamo=None, ignore_inlined_sources=False)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(
            backend,
            export=export,
            export_constraints=export_constraints,
            package=package,
            recompile_limit=recompile_limit,
        ),
        hooks,
        backend_ctx_ctor,
        fullgraph=True,
        export=export,
        dynamic=dynamic,
        rebuild_ctx=rebuild_ctx,
        package=package,
        isolate_recompiles=isolate_recompiles,
        dynamic_shapes=dynamic_shapes,
    )


class TorchPatcher:
    # [CLASS DOC] Namespace (static methods only) for one-time patches that stop Dynamo from tracing into
    # unsupported torch internals.
    @staticmethod
    @functools.cache
    def patch() -> None:
        # [FUNCTION DOC] Runs once (cached). Replaces TorchScript tracing functions and the FX tracer's
        # trace method with disabled versions, turns off distribution arg validation by default, disables
        # fused optimizer functions, the step of sparse/LBFGS optimizers and every optimizer's group
        # initialization. Side effects: mutates torch and torch.optim module/class attributes globally.
        # A better way to disable the following would be decorate the source
        # functions with @torch._disable_dynamo. However, this causes issues
        # with torch.deploy internally.
        from .decorators import disable

        torch.jit.trace = disable(
            torch.jit.trace, reason="tracing into TorchScript not fully supported"
        )
        torch.jit.trace_module = disable(
            torch.jit.trace_module,
            reason="tracing into TorchScript not fully supported",
        )
        torch.jit._get_trace_graph = disable(
            torch.jit._get_trace_graph,
            reason="tracing into TorchScript not fully supported",
        )
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace,
            reason="tracing into FX not fully supported",
        )
        torch.distributions.Distribution.set_default_validate_args(False)

        from torch.optim import (
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        )

        optimizer_modules = {
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        }

        for opt_mod in optimizer_modules:
            opt_name = opt_mod.__name__.split(".")[-1]
            fused_fn_name = f"_fused_{opt_name}"

            if hasattr(opt_mod, fused_fn_name):
                setattr(
                    opt_mod,
                    fused_fn_name,
                    disable(
                        getattr(opt_mod, fused_fn_name),
                        reason="don't trace into fused optimizer",
                    ),
                )

        optimizer_classes = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # Note: we don't support sparsity or tracing through backwards
        excluded_optimizer_classes = {
            torch.optim.SparseAdam,
            torch.optim.LBFGS,
        }

        for opt in optimizer_classes:
            if opt in excluded_optimizer_classes:
                opt.step = disable(
                    opt.step, reason=f"optimizer {opt} step not supported"
                )

            if hasattr(opt, "_init_group"):
                opt._init_group = disable(
                    opt._init_group, reason=f"optimizer {opt} _init_group not supported"
                )

    @staticmethod
    def suppress_torch_distributed_warnings(
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns a wrapper around fn that runs it with torch distributed user warnings hidden.
        def inner_fn(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Calls fn inside the warning-hiding context and returns its result.
            with torch._logging.hide_warnings(
                torch._logging._internal.user_warning_filter
            ):
                return fn(*args, **kwargs)

        return inner_fn


def skip_code(code: types.CodeType) -> None:
    # [FUNCTION DOC] Marks a code object so Dynamo skips it (skip on first execution, default afterward)
    # by setting its execution strategy in the C extension.
    set_code_exec_strategy(
        code, FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
    )
