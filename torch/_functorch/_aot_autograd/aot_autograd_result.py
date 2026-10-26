# [MODULE DOC] Result classes for AOT Autograd compilation, analogous to inductor's
# OutputCode. Defines per-graph "loadable" wrappers (InductorOutput subclasses) that
# either reference FxGraphCache entries or bundle compiled code directly, forward and
# backward holders, a serialized graph module helper, and GenericAOTAutogradResult with
# its concrete subclasses, whose wrap_post_compile rebuilds the original runtime callable
# from a cached/precompiled result. Also has the precompile deserialization entry point
# and a cache artifact class. Import-time side effects: creates loggers and type variables.
"""
This module provides result classes for AOT Autograd compilation.

Similar to how torch._inductor.output_code provides OutputCode classes for inductor
compilation results, this module provides AOTAutogradResult classes that represent
the compiled artifacts produced by AOT Autograd.

These results are:
- Serializable: can be saved/loaded from disk without recompilation
- Addressable: can be stored in caches with keys for later retrieval
- Reusable: can be used for both caching and ahead-of-time compilation (precompile)

The main result types are:
- GenericAOTAutogradResult: Abstract base for all AOT Autograd results
- AOTAutogradResult: Regular result that references FxGraphCache entries
- BundledAOTAutogradResult: Result that bundles the entire compiled code directly
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.precompile_context import BackendCacheArtifact
from torch._inductor.codecache import FxGraphCache
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    OutputCode,
)
from torch._inductor.utils import should_use_remote_fx_graph_cache
from torch._logging import getArtifactLogger

from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchAutogradCompileSpec,
    AOTDispatchSubclassWrapper,
    CachedAutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import AOTAutogradCacheInfo, AOTConfig  # noqa: F401
from .utils import simple_wraps


if TYPE_CHECKING:
    from torch._inductor.compile_fx import _CompileFxKwargs

    from .schemas import CacheableAOTConfig, ViewAndMutationMeta

log = logging.getLogger(__name__)
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


TOut = TypeVar("TOut", bound=OutputCode)


class InductorOutput(ABC, Generic[TOut]):
    # [CLASS DOC] Abstract interface for one compiled inductor artifact: pre_save prepares it for serialization,
    # load turns it into an OutputCode given example inputs, and post_compile finishes it using the fx config.
    # The abstract methods are one-line stubs with no behavior.
    """
    Class representing a single inductor output
    """

    @abstractmethod
    def pre_save(self) -> None: ...

    @abstractmethod
    def load(self, example_inputs: Sequence[Any]) -> TOut: ...

    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...


TOutputCode = TypeVar("TOutputCode", bound=OutputCode)


@dataclass
class BundledOutputCodeLoadable(InductorOutput[TOutputCode], Generic[TOutputCode]):
    # [CLASS DOC] Dataclass holding an OutputCode object directly (field result) so it is serialized with the cache
    # entry instead of being looked up in FxGraphCache. load stores example inputs on the instance (not a declared
    # field) and returns the bundled result; post_compile must be called after load.
    """
    A generic wrapper for OutputCode objects that are bundled directly in the cache
    (rather than looked up via FxGraphCache).

    This works for any OutputCode subclass (CompiledFxGraph, RegionalOutputCode, etc.)
    """

    result: TOutputCode

    def pre_save(self) -> None:
        # [FUNCTION DOC] Replaces the held result with a shallow copy that has had prepare_for_serialization applied,
        # so the saved form is serializable without altering the original object.
        disk_result = copy(self.result)
        disk_result.prepare_for_serialization()
        self.result = disk_result
        return

    def load(self, example_inputs: Sequence[Any]) -> TOutputCode:
        # [FUNCTION DOC] Remembers the example inputs on the instance and returns the bundled result unchanged.
        # KNOWN ISSUE: example_inputs is set dynamically and is not a declared dataclass field.
        self.example_inputs = example_inputs
        return self.result

    def post_compile(
        self, result: TOutputCode, fx_config: _CompileFxKwargs
    ) -> TOutputCode:
        # [FUNCTION DOC] Finishes a bundled result after load. For a CompiledFxGraph, reruns the FxGraphCache cache-hit
        # post-compile step (RuntimeError if it yields no graph), logs a bundled-hit artifact, and records the
        # compile region name from fx_config. Then calls the result's own post_compile with the stored example inputs
        # and constants, and finally lets the inductor cudagraph policy, if any, wrap the output. Returns the result.
        constants = CompiledFxGraphConstants()

        # Special handling for CompiledFxGraph - needs FxGraphCache.cache_hit_post_compile
        if isinstance(result, CompiledFxGraph):
            graph, cache_info = FxGraphCache.cache_hit_post_compile(
                result, {}, constants
            )
            if graph is None:
                raise RuntimeError("Failed to reload cache entry from disk")
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_bundled_cache_hit",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
            result = graph  # type: ignore[assignment]
            result.compile_region_name = (  # pyrefly: ignore[missing-attribute]
                fx_config.get("compile_region_name")
            )

        # Run normal post compile
        result.post_compile(self.example_inputs, constants, fx_config)

        # Let the CUDAGraph policy do outer-level wrapping (e.g. wrapping
        # an entire RegionalOutputCode as a single CUDA graph instead of
        # per-inner-region).
        import torch._inductor.config as _inductor_config

        policy = _inductor_config.cudagraph_policy
        if policy is not None:
            result = policy.wrap_output(result)

        return result


# Backwards compatibility alias
CompiledFxGraphLoadable: type[BundledOutputCodeLoadable[CompiledFxGraph]] = (
    BundledOutputCodeLoadable[CompiledFxGraph]
)


@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    # [CLASS DOC] Dataclass for a compiled graph stored in FxGraphCache: holds the cache key with debug lines and the
    # exact guard expression saved with it. load looks the entry up by key (local plus optional remote cache),
    # requiring an identical guard expression; post_compile finalizes the graph. Subclasses define _is_backward.
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: str | None

    def pre_save(self) -> None:
        # [FUNCTION DOC] No-op: only a cache key is stored, so nothing needs preparing.
        return

    def _is_backward(self) -> bool:
        # [FUNCTION DOC] HELPER: Whether this entry is a backward graph; the base class says no.
        return False

    def load(self, example_inputs: Sequence[Any]) -> CompiledFxGraph:
        # [FUNCTION DOC] Loads the compiled graph from FxGraphCache using the stored key, requiring the cached entry's guard
        # expression to equal the stored one. Clears the compiled Triton kernel cache first, uses the remote cache if
        # enabled, logs a structured hit or miss artifact, and remembers example inputs and constants on the instance.
        # Raises FXGraphCacheMiss if no entry is found. Returns the CompiledFxGraph.
        # KNOWN ISSUE: the long comment block mid-function ends in a truncated sentence, and the miss artifact is
        # labeled with a comment saying always a hit.
        from .autograd_cache import FXGraphCacheMiss

        # [Note: AOTAutogradCache and FXGraphCache Guard interactions]
        # As mentioned, AOTAutograd takes in the symint inputs from dynamo's list of arguments.
        # FXGraphCache serializes guards that are needed in the shape_env based on these symint inputs to the graph.
        # The invariant that AOTAutograd uses here is that the sources for symints given to it by dynamo are exactly
        # the same as the ones it passes to inductor, for both the forward and backward passes.
        # (This does not mean that the tensor values passed in are the same: only that their symints are).
        # That is, AOTAutograd and Inductor never create new guards based on symints with different sources
        # than those passed to it by inductor.
        # We pass the post compile function, which sets various fx_config boxed values,
        # so we can call it only after we're sure both forward and backward have
        # Clear CompiledTritonKernels before loading from FXGraphCache
        torch._inductor.async_compile.CompiledTritonKernels.cache_clear()
        remote_cache = None
        constants = CompiledFxGraphConstants()
        if should_use_remote_fx_graph_cache():
            remote_cache = FxGraphCache.get_remote_cache()
        (cache_key, debug_lines) = self.fx_graph_cache_info

        def check_exact_guard_match(guard_expr: str, _hints: Any) -> bool:
            # [FUNCTION DOC] HELPER: Guard evaluator passed to the cache loader; returns True only if the candidate guard
            # expression equals the stored one. The hints argument is ignored.
            """
            AOTAutogradCache tracks its own guards, so we just need to treat these guard expressions as a second
            cache key of sorts: we just check for equality, i.e. the FXGraphCache entry with
            the exact same guards as we originally saved into the cache.
            """
            return guard_expr == self.fx_graph_guard_expr

        result, cache_info = FxGraphCache.load_with_key(
            cache_key,
            debug_lines,
            example_inputs,
            local=True,
            remote_cache=remote_cache,
            is_backward=self._is_backward(),
            constants=constants,
            evaluate_guards=check_exact_guard_match,
        )
        if result is None:
            log.info("FXGraphCache cache miss for key %s", self.fx_graph_cache_info)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_cache_miss",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )

            raise FXGraphCacheMiss

        # No need to log chromium event because AOTAutograd will log that immediately for us
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hit",  # always a hit
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )
        self.example_inputs = example_inputs
        self.constants = constants
        return result

    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph:
        # [FUNCTION DOC] Sets the compile region name from fx_config on the loaded graph, runs its post_compile with the
        # saved example inputs and constants (which mutates fx_config-related boxed values), and applies the inductor
        # cudagraph policy wrapper if one is set. Returns the possibly wrapped result.
        """
        Called after FXGraphCacheLoadable.load, mutates fx_config
        """
        result.compile_region_name = fx_config.get(  # pyrefly: ignore[bad-assignment]
            "compile_region_name"
        )
        result.post_compile(self.example_inputs, self.constants, fx_config)

        import torch._inductor.config as _inductor_config

        policy = _inductor_config.cudagraph_policy
        if policy is not None:
            result = policy.wrap_output(result)

        return result


@dataclass
class CompiledForward(FxGraphCacheLoadable):
    # [CLASS DOC] FxGraphCache-backed entry for a forward graph; only reports that it is not a backward graph.
    """
    Cacheable entry for a forward function
    """

    def _is_backward(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns False (forward graph).
        return False


@dataclass
class GenericCompiledBackward(InductorOutput[TOut]):
    # [CLASS DOC] Mixin-style dataclass for backward outputs, adding backward_state_indices and the count of symints
    # saved for backward (used by AOTDispatchAutograd). Its post_compile delegates to the loadable class next in
    # the MRO and then wraps the result with dynamo disable so the backward is not traced.
    # Used by AOTDispatchAutograd.post_compile
    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int

    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut:
        # [FUNCTION DOC] Calls the next class's post_compile, then returns the result wrapped by torch dynamo disable
        # (reason: do not trace generated backwards pass), reapplying what the bw compiler wrapper does on a cache miss.
        # The concrete post_compile comes from the loadable mixin in each subclass MRO.
        compiled_bw = super().post_compile(  # pyrefly: ignore[missing-attribute]
            result, fx_config
        )
        # See note [Wrapping bw_compiler in disable]
        # This is done by _wrapped_bw_compiler in torch/_dynamo/backends/common.py
        # But since on cache hit we do not call the bw_compiler, we need to reapply the disable
        return torch._dynamo.disable(  # type: ignore[return-value]
            compiled_bw, reason="do not trace generated backwards pass"
        )


@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    # [CLASS DOC] FxGraphCache-backed backward entry: combines backward metadata with FxGraphCache loading.
    """
    Cacheable entry for a backward function
    """

    def _is_backward(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns True (backward graph).
        return True


# Generic bundled forward/backward classes that work with any OutputCode type
@dataclass
class BundledCompiledForward(
    BundledOutputCodeLoadable[TOutputCode], Generic[TOutputCode]
):
    # [CLASS DOC] Bundled forward entry: a BundledOutputCodeLoadable for any OutputCode type. No extra behavior.
    """
    Generic forward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """


@dataclass
class BundledCompiledBackward(
    GenericCompiledBackward[TOutputCode],
    BundledOutputCodeLoadable[TOutputCode],
    Generic[TOutputCode],
):
    # [CLASS DOC] Bundled backward entry: backward metadata plus bundled OutputCode loading. No extra behavior.
    """
    Generic backward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """


@dataclass
class SerializedGraphModule:
    # [CLASS DOC] Picklable snapshot of a GraphModule, stored as the reconstruction function and arguments from the
    # module's reduce protocol. deserialize rebuilds the GraphModule on demand.
    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] Captures the reconstruction function and args from the module's reduce method.
        self.fn, self.args = gm.__reduce__()

    def deserialize(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Calls the stored function with the stored args and returns the GraphModule.
        # Raises AssertionError if the result is not a GraphModule.
        gm = self.fn(*self.args)
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"expected fx.GraphModule, got {type(gm)}")
        return gm


def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule:
    # [FUNCTION DOC] Clears the metadata of the graph module and of all its nodes (mutating them), then returns a
    # SerializedGraphModule of it.
    # NOTE: mutates the graph module
    gm.meta = {}
    for node in gm.graph.nodes:
        # pyrefly: ignore [implicit-any]
        node.meta = {}
    return SerializedGraphModule(gm)


TForward = TypeVar("TForward", bound="InductorOutput[Any]")
TBackward = TypeVar("TBackward", bound="GenericCompiledBackward[Any]")


@dataclass
class GenericAOTAutogradResult(Generic[TForward, TBackward]):
    # [CLASS DOC] Dataclass capturing everything needed to rebuild AOT Autograd's compiled callable from a cache or
    # precompile artifact: the forward and optional backward InductorOutputs, logged graph strings, runtime
    # metadata, dispatch wrappers, subclass metadata, detach indices, timings, a sanitized AOT config, guard
    # expression and an optional serialized backward module. Key methods: pre_save and wrap_post_compile.
    """A single result from AOT Autograd compilation, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradResult is essentially parameterized by 1. the method of loading
    from the cache (either Bundled or UnBundled), and 2. The type of the output. For now,
    the only type of output we support is Python Wrapper output, i.e. OutputCode.CompiledFxGraph,
    but the same technique works for C++ wrapper code; we'd just add an extra InductorOutput type.
    """

    # Forward and Backward info
    compiled_fw: TForward
    compiled_bw: TBackward | None

    # Code of the joint graph using print_readable()
    # Used for logging purposes
    aot_joint_graph_str: str | None
    aot_forward_graph_str: str | None
    aot_backward_graph_str: str | None
    min_cut_info_str: str | None

    # Runtime_metadata saved right before compilation
    runtime_metadata: ViewAndMutationMeta

    # Wrappers that run after each aot_dispatch_* function
    dispatch_wrappers: list[CompilerWrapper]

    # Used by AOTSubclassWrapper
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None

    # Used by RuntimeWrapper
    indices_of_inps_to_detach: list[int]

    # Time taken to trace/compile the forward
    # forward_time_taken includes AOTAutograd tracing time + inductor compilation time
    # backward_time_taken is essentially just the time inductor took to compile
    forward_time_taken_ns: int
    backward_time_taken_ns: int

    # Used by standalone_compile
    sanitized_aot_config: CacheableAOTConfig

    guards_expr: str | None

    # Used by Compiled Autograd
    serialized_bw_module: SerializedGraphModule | None

    def pre_save(self) -> None:
        # [FUNCTION DOC] Calls pre_save on the forward output and on the backward output if present.
        """
        Perform any preparations to make the result ready for serialization.
        """
        self.compiled_fw.pre_save()
        if self.compiled_bw is not None:
            self.compiled_bw.pre_save()

    def _log_cached_graphs(self, aot_config: AOTConfig | CacheableAOTConfig) -> None:
        # [FUNCTION DOC] HELPER: When logging is enabled in the config, replays structured-trace and logger output for the
        # stored joint, forward (plus metadata artifacts), backward graph strings and min-cut info, noting they came
        # from cache. The forward graph is labeled an inference graph when there is no backward string.
        if not aot_config.enable_log:
            return

        if self.aot_joint_graph_str is not None:
            torch._logging.trace_structured(
                "aot_joint_graph", payload_fn=lambda: self.aot_joint_graph_str
            )
            aot_graphs_log.info(
                "Joint graph (from cache)\n\n%s", self.aot_joint_graph_str
            )

        if self.aot_forward_graph_str is not None:
            from torchgen.utils import dataclass_repr

            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(self.runtime_metadata),
            )
            if self.maybe_subclass_meta is not None:
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "aot_forward_graph_fw_subclass_metadata",
                        "encoding": "string",
                    },
                    payload_fn=lambda: dataclass_repr(self.maybe_subclass_meta),
                )

            # It's called an inference graph if not running with autograd
            has_backward = self.aot_backward_graph_str is not None
            torch._logging.trace_structured(
                "aot_forward_graph" if has_backward else "aot_inference_graph",
                payload_fn=lambda: self.aot_forward_graph_str,
            )
            aot_graphs_log.info(
                "Forward graph (from cache)\n\n%s",
                self.aot_forward_graph_str,
            )

        if self.aot_backward_graph_str is not None:
            torch._logging.trace_structured(
                "aot_backward_graph", payload_fn=lambda: self.aot_backward_graph_str
            )
            aot_graphs_log.info(
                "Backward graph (from cache)\n\n%s",
                self.aot_backward_graph_str,
            )

        if self.min_cut_info_str is not None:
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "min_cut_information",
                    "encoding": "json",
                },
                payload_fn=lambda: self.min_cut_info_str,
            )

    def _load_and_post_compile(
        self,
        args: list[torch.Tensor],
        fx_config: _CompileFxKwargs,
    ) -> tuple[Callable[..., Any], Callable[..., Any] | None, bool]:
        # [FUNCTION DOC] HELPER: Loads forward (and backward if present) from their cache entries, records the dispatch mode
        # (autograd or inference) in the compile event log, then runs post_compile on each with fx configs marked
        # is_backward accordingly, only after both loads succeeded. Returns (forward callable, backward callable or
        # None, needs_autograd flag). Cache misses propagate as exceptions from load.
        from torch._dynamo.utils import CompileEventLogger

        compiled_fw_func = self.compiled_fw.load(args)
        if self.compiled_bw is not None:
            compiled_bw_func = self.compiled_bw.load(args)
            needs_autograd = True
            CompileEventLogger.try_add_pt2_compile(
                "backend_compile", dispatch_mode="autograd"
            )
            # Now that we've loaded forward and backward, call post compile on both
            # This avoids setting things like BoxedBools in fx_config until
            # after both forward and backward cache hit
            fw_fx_config: _CompileFxKwargs = {
                **fx_config,
                "is_backward": False,
            }
            bw_fx_config: _CompileFxKwargs = {
                **fx_config,
                "is_backward": True,
            }
            compiled_fw_func = self.compiled_fw.post_compile(
                compiled_fw_func, fw_fx_config
            )
            compiled_bw_func = self.compiled_bw.post_compile(
                compiled_bw_func, bw_fx_config
            )
            return compiled_fw_func, compiled_bw_func, needs_autograd

        inference_fx_config: _CompileFxKwargs = {
            **fx_config,
            "is_backward": False,
        }

        needs_autograd = False
        CompileEventLogger.try_add_pt2_compile(
            "backend_compile", dispatch_mode="inference"
        )
        compiled_fw_func = self.compiled_fw.post_compile(
            compiled_fw_func, inference_fx_config
        )
        return compiled_fw_func, None, needs_autograd

    def _apply_runtime_wrappers(
        self,
        compiled_fw_func: Callable[..., Any],
        compiled_bw_func: Callable[..., Any] | None,
        needs_autograd: bool,
        aot_config: AOTConfig,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Reapplies the runtime wrapper stack to the loaded callables: subclass wrapper, functionalized
        # RNG wrapper, then either AOTDispatchAutograd post_compile (building a compile spec with no bw compiler and no
        # cache-save callback, and a lazy backward from the serialized module if present) or the plain RuntimeWrapper,
        # and finally the saved dispatch wrappers. Records subclass dispatch in the event log. Returns the final callable.
        # Raises AssertionError if autograd is needed but no backward entry exists.
        from torch._dynamo.utils import CompileEventLogger

        compiled_fw_func = AOTDispatchSubclassWrapper(
            trace_joint=needs_autograd,
            fw_only=None,
            maybe_subclass_meta=self.maybe_subclass_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )

        req_subclass_dispatch = self.maybe_subclass_meta is not None
        CompileEventLogger.try_add_pt2_compile(
            "backend_compile", requires_subclass_dispatch=req_subclass_dispatch
        )

        # In autograd case, functionalizedRngWrapper should not modify outs
        return_new_outs = not needs_autograd
        compiled_fw_func = FunctionalizedRngRuntimeWrapper(
            return_new_outs=return_new_outs
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )
        # pyrefly: ignore [missing-attribute]
        compiled_fw_func._boxed_call = True
        disable_amp = torch._C._is_any_autocast_enabled()

        if needs_autograd:
            if self.compiled_bw is None:
                raise AssertionError("compiled_bw must not be None when needs_autograd")

            cached_lazy_backward = None
            if self.serialized_bw_module is not None:
                cached_lazy_backward = CachedAutogradLazyBackwardCompileInfo(
                    self.serialized_bw_module.deserialize
                )
            # This function is run on both cache miss and cache hit, either here
            # or in aot_dispatch_autograd. On a cache hit,
            # 1. the bw is already compiled
            # 2. we don't need to save to the cache again
            # so those corresponding arguments are set to None.
            compile_spec = AOTDispatchAutogradCompileSpec(
                compiled_fw_func=compiled_fw_func,
                compiled_bw_func=compiled_bw_func,
                maybe_subclass_meta=self.maybe_subclass_meta,
                num_symints_saved_for_bw=self.compiled_bw.num_symints_saved_for_bw_,
                backward_state_indices=self.compiled_bw.backward_state_indices,
                disable_amp=disable_amp,
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                lazy_backward_info=cached_lazy_backward,
                bw_compiler=None,
                aot_config=aot_config,
                fw_metadata=self.runtime_metadata,
                try_save_cache_entry=None,
            )
            compiled_function = AOTDispatchAutograd.post_compile(compile_spec)

        else:
            compiled_function = RuntimeWrapper(
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                trace_joint=False,
                disable_amp=disable_amp,
            ).post_compile(
                compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
            )

        # Add serialization function back onto object
        compiled_function, _ = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )
        return compiled_function

    def _check_guards(self, args: list[torch.Tensor]) -> None:
        # [FUNCTION DOC] HELPER: If a guard expression is stored, evaluates it against the backed symints found in args via
        # AOTAutogradCache and raises AssertionError when it is not exactly true. Does nothing otherwise.
        if self.guards_expr:
            from .autograd_cache import AOTAutogradCache

            symints = AOTAutogradCache._filter_backed_symints(args)
            check = bool(AOTAutogradCache.evaluate_guards(self.guards_expr, symints))
            if check is not True:
                raise AssertionError(f"guards check failed: {check}")

    # Turn result into the original callable
    def wrap_post_compile(
        self,
        args: list[torch.Tensor],
        aot_config: AOTConfig | CacheableAOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Reconstructs the callable AOT Autograd originally returned. Replays cached graph logging, loads and
        # post-compiles forward/backward under a dynamo timer, applies the runtime wrappers using the (narrowed) AOT
        # config, then checks guards against args. Returns the callable. May raise the cache-miss exception from load or
        # AssertionError from the guard check.
        """
        This function takes a result and carefully reconstructs the original callable
        that AOTAutograd returned the first time it was run. It does this by running the various
        post compile steps that AOTAutograd runs on its compiled artifact after running the fw/bw compilers.

        In the inference path, this consists of the Subclass, FunctionalzedRngRuntime, and RuntimeWrappers.
        In the autograd path, this consists of AOTAutogradDispatch.post_compile.

        The steps here should match exactly the steps that are run in aot_dispatch_base and aot_dispatch_autograd.

        Notably absent from the cached path are:
        - DebugAssertWrapper
        - FakifiedOutWrapper

        Which we'll handle separately later on, if necessary.
        """
        from torch._dynamo.utils import dynamo_timed

        self._log_cached_graphs(aot_config)
        # Cache hits only retain the cacheable subset of AOTConfig. Narrow once
        # here so the existing post-compile wrapper stack can keep its compile-time
        # AOTConfig annotations.
        runtime_aot_config = cast(AOTConfig, aot_config)
        with dynamo_timed("AOTAutogradCache.inductor_load"):
            compiled_fw_func, compiled_bw_func, needs_autograd = (
                self._load_and_post_compile(args, fx_config)
            )

        compiled_function = self._apply_runtime_wrappers(
            compiled_fw_func,
            compiled_bw_func,
            needs_autograd,
            runtime_aot_config,
        )
        # Now that we're pretty sure it's a successful load, add guards
        # to the existing shape environment from the cache.
        self._check_guards(args)
        return compiled_function


class AOTAutogradResult(GenericAOTAutogradResult[CompiledForward, CompiledBackward]):
    # [CLASS DOC] Result variant whose forward/backward are FxGraphCache references. No added behavior.
    """
    Regular AOTAutogradResult: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """


class BundledAOTAutogradResult(
    GenericAOTAutogradResult[
        BundledCompiledForward[TOutputCode], BundledCompiledBackward[TOutputCode]
    ],
    Generic[TOutputCode],
):
    # [CLASS DOC] Result variant that bundles the full OutputCode for forward and backward in the entry itself,
    # for any OutputCode subclass, so no FxGraphCache lookup is required. No added behavior.
    """
    Generic AOTAutogradResult where we bundle the entire OutputCode directly
    (rather than looking it up via FxGraphCache).

    This works with any OutputCode type:
    - CompiledFxGraph: Traditional inductor compilation
    - RegionalOutputCode: Regional inductor compilation with GraphPickler serialization
    - Any future OutputCode subclasses

    Type parameter:
        TOutputCode: The OutputCode subclass (e.g., CompiledFxGraph, RegionalOutputCode)

    Usage with CompiledFxGraph:
        entry = BundledAOTAutogradResult[CompiledFxGraph](
            compiled_fw=BundledCompiledForward(result=CompiledFxGraph(...)),
            compiled_bw=BundledCompiledBackward(
                result=CompiledFxGraph(...),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )

    Usage with RegionalOutputCode:
        entry = BundledAOTAutogradResult[RegionalOutputCode](
            compiled_fw=BundledCompiledForward(result=RegionalOutputCode(gm)),
            compiled_bw=BundledCompiledBackward(
                result=RegionalOutputCode(gm),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )
    """


def deserialize_bundled_cache_entry(
    entry: BundledAOTAutogradResult[Any],
) -> Callable[..., Any]:
    # [FUNCTION DOC] Turns a bundled result into a runnable, still-serializable function for precompile. Clears the guard
    # expression (dynamo already serialized guards), deep-copies the entry for later re-serialization, and under the
    # current tracing context (or a fresh fake-tensor context if none) calls wrap_post_compile with empty example
    # args and a minimal fx config (cudagraphs flag from inductor config, empty device index). Wraps the result in
    # SerializableCompiledFunction and returns a forward function taking positional args that calls it with a list,
    # carrying the serialize attribute. Raises AssertionError if serialize is missing.
    # KNOWN ISSUE: per its TODO, cudagraphs config is not shared and flat_params are ignored.
    from copy import deepcopy

    from torch._inductor.cudagraph_utils import BoxedDeviceIndex
    from torch._inductor.utils import BoxedBool

    # In the precompile use case, guards are already serialized
    # by dynamo, so we don't need to add them to the environment
    entry.guards_expr = None
    # TODO: this isn't exactly right, because cudagraphs needs to be a shared config
    # which is set by compile_fx. But in precompile, we never actually call compile_fx
    # so we don't have a place to track cudagraphs here.
    cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
    boxed_forward_device_index = BoxedDeviceIndex(None)
    # We need to make a clean copy of the cache entry
    # in case it needs to be serialized again
    serializable_copy = deepcopy(entry)

    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    context = torch._guards.TracingContext.try_get()
    if context is None:
        # Create a clean environment when running fx graph post compile
        # if one is not available
        context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
    with torch._guards.tracing(context):
        compiled_fn = entry.wrap_post_compile(
            [],
            entry.sanitized_aot_config,
            {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            },
        )
    # Ensure the deserialized cache entry is still serializable

    compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: serializable_copy)

    # TODO: this ignores flat_params, which can exist
    # if inline_builtin_nn_modules=False
    @simple_wraps(compiled_fn)
    def forward(*runtime_args: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the compiled function with the runtime args packed into a list (boxed calling convention).
        return compiled_fn(list(runtime_args))

    if not hasattr(compiled_fn, "serialize"):
        raise AssertionError("compiled_fn must have serialize attribute")
    forward.serialize = compiled_fn.serialize  # type: ignore[attr-defined]

    return forward


@dataclass
# pyrefly: ignore [implicit-any]
class BundledAOTAutogradCacheArtifact(BackendCacheArtifact[Callable]):
    # [CLASS DOC] Backend cache artifact for a bundled AOT Autograd result; deserialization yields a runnable function.
    # pyrefly: ignore [implicit-any]
    def after_deserialization(self) -> Callable:
        # [FUNCTION DOC] Builds the runnable callable from the stored content using deserialize_bundled_cache_entry.
        return deserialize_bundled_cache_entry(self.content)
