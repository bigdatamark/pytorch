# [MODULE DOC] Local/remote caching of AOTAutograd compile results. Provides the
# policy predicates (should_use_*_autograd_cache), cacheability checks over FX
# graphs (check_node_safe, check_cacheable), the cache-key machinery
# (AOTAutogradCacheDetails, AOTAutogradCachePickler, autograd_cache_key,
# normalize_placeholder_names) and the AOTAutogradCache class that loads, saves
# and bypasses entries on disk or in a remote cache. Cache misses or unsafe graphs
# are signalled with BypassAOTAutogradCache/FXGraphCacheMiss. Import-time side
# effects: creates a module logger only.
"""
Utils for caching the outputs of AOTAutograd
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import time
import traceback
from copy import copy
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.trace_rules import torch_non_c_binding_in_graph_functions
from torch._dynamo.utils import (
    chromium_event_log_active,
    CompileEventLogger,
    counters,
    is_safe_numpy_wrapper,
    numpy_wrapper_cache_key,
    warn_once,
)
from torch._functorch import config
from torch._inductor.codecache import (
    _ident,
    add_ephemeral_timeout_increase_for_distributed,
    AOTAUTOGRAD_CACHE_PREFIX,
    BypassFxGraphCache,
    create_cache,
    extract_tensor_metadata_for_cache_key,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    GuardedCache,
    sha256_hash,
    write_atomic,
)
from torch._inductor.custom_graph_pass import (
    CustomKnapsackSolver,
    CustomRuntimeEstimator,
)
from torch._inductor.output_code import OutputCode
from torch._inductor.remote_cache import cache_stats
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import BoxedBool, should_use_remote_fx_graph_cache
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import LazyString
from torch._utils_internal import log_cache_bypass
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from torch.fx.experimental.symbolic_shapes import guarding_hint_or_throw
from torch.fx.node import Node
from torch.fx.traceback import _get_memory_budget_annotation
from torch.utils._triton import has_triton_package

from .aot_autograd_result import (
    AOTAutogradResult,
    BundledAOTAutogradCacheArtifact,
    BundledAOTAutogradResult,
    BundledCompiledBackward,
    BundledCompiledForward,
    CompiledBackward,
    CompiledForward,
    GenericAOTAutogradResult,
    SerializedGraphModule,
)
from .runtime_wrappers import (
    CompilerWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import (
    ActInputPaths,
    AOTAutogradCacheInfo,
    AOTConfig,
    CacheableAOTConfig,
    ViewAndMutationMeta,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._inductor.compile_fx import _CompileFxKwargs, CompilerConfigExtra
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache


log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    # [CLASS DOC] Exception signalling that AOTAutograd caching must be skipped for
    # the current graph (unsupported content, disabled features, cache failure).
    # Carries only a message; callers catch it and fall back to a normal compile.
    pass


# Used to signify when FXGraphCache missed when AOTAutogradCache uses it
class FXGraphCacheMiss(BypassAOTAutogradCache):
    # [CLASS DOC] Subclass of BypassAOTAutogradCache raised when the inner FX graph
    # cache lookup missed, so the whole AOTAutograd entry is treated as a miss.
    pass


def should_use_remote_autograd_cache() -> bool:
    # [FUNCTION DOC] Decides whether the remote AOTAutograd cache is enabled.
    # Returns False if caches are force-disabled; otherwise the explicit config
    # value if set; otherwise False outside fbcode and in FB unit tests; otherwise
    # compares the remote cache version against a justknob value (False if the
    # internal remote cache module is missing).
    if torch.compiler.config.force_disable_caches:
        return False
    if config.enable_remote_autograd_cache is not None:
        return config.enable_remote_autograd_cache
    if not config.is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    jk_name = "pytorch/remote_cache:aot_autograd_cache_version"

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(jk_name)


def should_use_local_autograd_cache() -> bool:
    # [FUNCTION DOC] Returns False if caches are force-disabled globally, otherwise
    # the enable_autograd_cache config value.
    if torch.compiler.config.force_disable_caches:
        return False
    return config.enable_autograd_cache


def should_bundle_autograd_cache() -> bool:
    # [FUNCTION DOC] Returns True if the bundled autograd cache config is on or
    # dynamo's caching precompile mode is on (entries embed compiled artifacts).
    return config.bundled_autograd_cache or torch._dynamo.config.caching_precompile


def check_node_safe(node: Node) -> None:
    # [FUNCTION DOC] Validates that one FX node can be cached. call_function nodes
    # need an allowlisted target (op overloads, safe numpy wrappers, public torch
    # functional APIs, builtins, allowlisted functions, cacheable higher-order ops,
    # or fx.wrap functions with a user cache hash); call_method nodes must act on a
    # tensor-valued node with a string or method-descriptor name; placeholder,
    # get_attr, call_module and output are accepted. Returns None when safe, raises
    # BypassAOTAutogradCache otherwise (also for unknown ops). Raises AssertionError
    # on malformed nodes (string call_function target, non-Node method receiver).
    """
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.

    [Note: AOTAutograd Cacheability checks]
    - Our cache key is computed from the FX graph produced by Dynamo and the input example values
    - A node is "safe" if the same cache key results in a compiled artifact that has the same behavior
        (i.e, the set of inputs that go into our cache key is sufficient to distinguish its behavior)

    To accomplish this safety check, we consider the following functions to be safe:
        - Public functions under modules torch, torch.functional, and torch.nn.functional: these are
        allowed in the graph by dynamo, so we can assume they are safe to cache.
        - method calls on base tensor types
        - Any call_module that dynamo deemed safe to allow AOTAutograd to trace
        - Non callable nodes, such as placeholder, output, get_attr

    The test suite test_aot_autograd_cache.py::AOTAutogradCachePicklerTests tries its best to fully cover/specify this behavior.
    """
    SAFE_TORCH_MODULES = ("torch.functional", "torch.nn.functional")
    SAFE_TORCH_FUNCTIONS = (
        "torch.Size",
        "torch.Tensor",
        "torch.sym_int",
        "torch._sym_sqrt",
        "torch.sym_float",
        "torch.sym_sum",
        "torch.autograd.grad",
        "torch.distributed.tensor._api.from_local",
        # An autocast context manager *inside* a compiled region is traced into
        # these calls. What they do is fully determined by their arguments
        # (device type, dtype, enabled, cache_enabled), which are constants in
        # the graph and therefore part of the cache key, so a graph compiled
        # under one autocast setting can never be reused for another.
        "torch.amp.autocast_mode._enter_autocast",
        "torch.amp.autocast_mode._exit_autocast",
    )
    SAFE_NON_TORCH_FUNCTIONS = (
        "einops.einops.rearrange",
        "einops.einops.repeat",
    )

    def is_public_torch_api(target: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: True if target is defined in torch.functional or
        # torch.nn.functional and its name does not start with an underscore.
        # Don't blindly allow private functions in the torch namespace
        is_private = target.__name__.startswith("_")

        return (
            getattr(target, "__module__", None) in SAFE_TORCH_MODULES and not is_private
        )

    def is_safe_torch_function(target: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: True if the qualified name of target is on an
        # allowlist: the non-C-binding in-graph function set, the local safe list,
        # or the inductor config's unsafe-marked-cacheable list. The custom autograd
        # FunctionCtx is allowed only if the corresponding functorch config is on.
        """Allowlisted torch functions"""
        function_name = f"{target.__module__}.{target.__name__}"
        # Allow torch.autograd.function.FunctionCtx if custom autograd functions are allowed
        if function_name == "torch.autograd.function.FunctionCtx":
            return (
                torch._functorch.config.autograd_cache_allow_custom_autograd_functions
            )

        # Functions in torch_non_c_binding_in_graph_functions
        # are guaranteed to be cache safe.
        # See NOTE: [Cacheability of in-graph torch functions]
        return (
            function_name in torch_non_c_binding_in_graph_functions
            or function_name in SAFE_TORCH_FUNCTIONS
            or function_name in torch._inductor.config.unsafe_marked_cacheable_functions
        )

    def is_cacheable_function(target: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: True if target is an op overload/packet, safe numpy
        # wrapper, public torch API, builtin, allowlisted torch function or
        # allowlisted non-torch function; for a higher-order operator, returns its
        # own cacheable() answer.
        if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            return True
        if is_safe_numpy_wrapper(target):
            return True
        if is_public_torch_api(target):
            return True
        # Technically, FXGraphCache._check_for_hop already checks this,
        # but better to error earlier anyway
        if isinstance(target, torch._ops.HigherOrderOperator):
            return target.cacheable()
        is_builtin_fun_or_type = type(target).__name__ == "builtin_function_or_method"
        if is_builtin_fun_or_type:
            return True
        if is_safe_torch_function(target):
            return True
        function_name = f"{target.__module__}.{target.__name__}"
        if function_name in SAFE_NON_TORCH_FUNCTIONS:
            return True
        return False

    def is_tensor(target: Node) -> bool:
        # [FUNCTION DOC] HELPER: True if the node carries an example value in its
        # metadata, which is used as the signal that it represents a tensor.
        # Tensors always have example values in meta field
        return "example_value" in target.meta

    # I'd love to use a match statement here, but it wasn't introduced until py3.10
    if node.op == "call_function":
        if node.meta and node.meta.get("is_wrapped", False):
            # This is fx.wrap function
            # By default we BypassAOTAutogradCache for unknown functions,
            # But if user explicitly specified cache hash - allow to cache it.
            if node.meta.get("user_cache_hash", None):
                return
        if isinstance(node.target, str):
            raise AssertionError(
                f"expected node.target to not be a string, got {node.target}"
            )
        if not is_cacheable_function(node.target):
            module = getattr(node.target, "__module__", None)
            name = getattr(node.target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_function target {node.target}. \n Function module: {module}, \nFunction name: {name}"
            )
    elif node.op == "call_method":
        method_name = node.target
        method_target = node.args[0]
        # Only support method calls on base tensors
        if not isinstance(method_target, Node):
            raise AssertionError(
                f"expected method_target to be Node, got {type(method_target)}"
            )
        if not is_tensor(method_target):
            module = getattr(method_target, "__module__", None)
            name = getattr(method_target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_method target {method_target}. \nMethod module: {module}, \nMethod name: {name}"
            )
        if (
            type(method_name) is not str
            and type(method_name).__name__ != "method_descriptor"
        ):
            raise BypassAOTAutogradCache(
                f"Unsupported call_method method {node.target}: {method_name}"
            )
    # Cache safe
    elif node.op in ("placeholder", "get_attr", "call_module", "output"):
        # Assumption today for call_module being a safe op:
        # (1) today the only call_module ops that can show up in a graph come from "built-in-nn-modules"
        # that dynamo assumes are safe to trace. If dynamo assumes they are safely to blindly trace, then
        # they should be safe to cache as well.
        # (2) in the steady-state (some time in H2?) we shouldn't see these anymore, once inline builtin nn modules by default
        # (3) We do not allow user made nn modules in the graph today, only function calls.
        pass
    else:
        raise BypassAOTAutogradCache(f"Unsupported node op {node.op}")


def check_cacheable(gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Raises BypassAOTAutogradCache if inductor freezing is on, if
    # neither the local nor remote FX graph cache is enabled, if the current tracing
    # context fakifies on first call, or if any node is unsafe (see check_node_safe).
    # Also recursively checks the saved-tensors-hook pack and unpack subgraphs when
    # present. Returns None if cacheable.
    """
    Checks that the graph module only uses supported operators
    """
    nodes = gm.graph.nodes
    if torch._inductor.config.freezing:
        raise BypassAOTAutogradCache("Cannot cache a graph with freezing enabled")

    if not (
        torch._inductor.config.fx_graph_cache or should_use_remote_fx_graph_cache()
    ):
        raise BypassAOTAutogradCache("FX graph cache is not enabled")

    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context and tracing_context.fakify_first_call:
        raise BypassAOTAutogradCache(
            "Won't cache a graph with fakify_first_call enabled"
        )
    for node in nodes:
        check_node_safe(node)

    # Saved tensors hooks are globally set subgraphs,
    # that are not used explicitly in the main graph.
    # They are inlined in aot_autograd graphs.
    # Subgraphs are only used for caching logic.
    if hasattr(gm, "saved_tensors_hooks_pack_0"):
        check_cacheable(gm.saved_tensors_hooks_pack_0)  # type: ignore[arg-type]
        # We have guarantee of unpack subgraph existence if pack subgraph exists
        check_cacheable(gm.saved_tensors_hooks_unpack_0)  # type: ignore[arg-type]


def _get_context_fn_cache_hash(context_fn: Callable[..., Any]) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the user-supplied cache_hash attribute of a
    # selective-activation-checkpointing context function, or None if absent.
    """
    Extract a cache hash from a context_fn used for selective activation checkpointing (SAC).

    The context_fn determines which ops are saved vs recomputed in the SAC region.
    Since context_fn can be an arbitrary Python function, we cannot reliably pickle
    it for cache key generation (pickle only captures the function name, not the code).

    Users must provide a stable hash by setting a `cache_hash` attribute on the context_fn.
    For functools.partial objects, set the cache_hash on the partial object itself, not on
    the underlying function.

    Returns:
        The cache hash if found
        None: If no hash is provided (caller should bypass caching)
    """
    if hasattr(context_fn, "cache_hash"):
        return context_fn.cache_hash

    return None


def _iter_graph_modules(
    gm: torch.fx.GraphModule,
) -> Generator[torch.fx.GraphModule, None, None]:
    # [FUNCTION DOC] HELPER: Generator yielding gm and every nested submodule that
    # is itself a GraphModule, in module traversal order.
    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule):
            yield module


def _collect_context_fn_hashes(gm: torch.fx.GraphModule) -> list[str]:
    # [FUNCTION DOC] HELPER: For each graph module (including nested ones) holding a
    # checkpoint context function in its metadata, collects that function's cache
    # hash. Raises BypassAOTAutogradCache if any such function has no hash.
    """
    Collect cache hashes from all context_fn used in SAC HOPs within the graph module.

    Returns a list of hashes. Raises BypassAOTAutogradCache if any context_fn
    lacks a cache_hash attribute.
    """
    hashes = []
    for module in _iter_graph_modules(gm):
        context_fn = module.meta.get("_checkpoint_context_fn")
        if context_fn is not None:
            cache_hash = _get_context_fn_cache_hash(context_fn)
            if cache_hash is None:
                raise BypassAOTAutogradCache(
                    "SAC context_fn does not have a cache_hash attribute. "
                    "To enable caching with selective activation checkpointing, "
                    "add a 'cache_hash' attribute to your context_fn. This can be "
                    "a string or any hashable value that uniquely identifies the checkpointing "
                    "behavior (e.g., based on source code hash and closed-over globals). "
                    "For functools.partial objects, set cache_hash on the partial itself."
                )
            hashes.append(cache_hash)
    return hashes


def _collect_wrapped_user_cache_hashes(gm: torch.fx.GraphModule) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the user_cache_hash metadata of every node in
    # gm marked as an fx.wrap function, in graph order.
    # KNOWN ISSUE: indexes the metadata key directly, so a wrapped node lacking
    # user_cache_hash raises a lookup error instead of a cache bypass.
    wrapped_user_cache_hashes = []
    for node in gm.graph.nodes:
        if node.meta and node.meta.get("is_wrapped", False):
            wrapped_user_cache_hashes.append(node.meta["user_cache_hash"])
    return wrapped_user_cache_hashes


def _collect_saved_tensors_hooks_fx_wrap_cache_hashes(
    gm: torch.fx.GraphModule,
) -> tuple[list[str], list[str]]:
    # [FUNCTION DOC] HELPER: If gm has saved-tensors-hook subgraphs, returns the
    # wrapped-function user cache hashes of the pack subgraph and of the unpack
    # subgraph as two lists; otherwise two empty lists.
    if not hasattr(gm, "saved_tensors_hooks_pack_0"):
        return ([], [])

    return (
        _collect_wrapped_user_cache_hashes(
            # pyrefly: ignore[bad-argument-type]
            gm.saved_tensors_hooks_pack_0
        ),
        _collect_wrapped_user_cache_hashes(
            # pyrefly: ignore[bad-argument-type]
            gm.saved_tensors_hooks_unpack_0
        ),
    )


def _get_custom_estimator_solver_uuids(
    autograd_config: Any,
) -> tuple[object | None, object | None]:
    # [FUNCTION DOC] HELPER: Reads the memory-budget runtime estimator and knapsack
    # solver from the given config object and returns their uuids as a pair (None
    # for unset or string-named built-ins). Raises BypassAOTAutogradCache if a custom
    # object's uuid is None or if the value is a raw callable without uuid support.
    """
    Extract uuid values from custom runtime estimator and solver configs if they have uuid() methods.

    Returns a tuple of (runtime_estimator_uuid, solver_uuid).

    Returns None for each component if:
    - The config field value is None
    - The config field value is a string (built-in option like "flops", "greedy")

    Raises BypassAOTAutogradCache if:
    - The config field value is a raw callable without uuid() method (caching not supported)
    - The CustomRuntimeEstimator/CustomKnapsackSolver's uuid() method returns None
    (caching explicitly disabled by implementation)
    """

    runtime_estimator = getattr(
        autograd_config, "activation_memory_budget_runtime_estimator", None
    )
    solver = getattr(autograd_config, "activation_memory_budget_solver", None)

    runtime_estimator_uuid = None
    solver_uuid = None

    if isinstance(runtime_estimator, CustomRuntimeEstimator):
        runtime_estimator_uuid = runtime_estimator.uuid()
        if runtime_estimator_uuid is None:
            raise BypassAOTAutogradCache(
                "CustomRuntimeEstimator.uuid() returned None, bypassing cache"
            )
    elif callable(runtime_estimator) and not isinstance(runtime_estimator, str):
        raise BypassAOTAutogradCache(
            "activation_memory_budget_runtime_estimator is a raw callable without uuid() method, "
            "bypassing cache. Use CustomRuntimeEstimator for cache support."
        )

    if isinstance(solver, CustomKnapsackSolver):
        solver_uuid = solver.uuid()
        if solver_uuid is None:
            raise BypassAOTAutogradCache(
                "CustomKnapsackSolver.uuid() returned None, bypassing cache"
            )
    elif callable(solver) and not isinstance(solver, str):
        raise BypassAOTAutogradCache(
            "activation_memory_budget_solver is a raw callable without uuid() method, "
            "bypassing cache. Use CustomKnapsackSolver for cache support."
        )

    return runtime_estimator_uuid, solver_uuid


class AOTAutogradCacheDetails(FxGraphHashDetails):
    # [CLASS DOC] Bag of everything that goes into an AOTAutograd cache key, built on
    # the inductor FX graph hash details. Attributes set at construction: aot_config,
    # act_input_paths, saved-tensors-hook wrap hashes, SAC context-fn hashes, region
    # activation memory budget, custom estimator/solver uuids, grad mode, per-device
    # autocast dtypes, deterministic-algorithms flag, saved autograd config and (if
    # triton is installed) user triton kernel sources. It is then pickled by
    # AOTAutogradCachePickler to produce the key.
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def _iter_triton_kernels_from_node(self, node: Node) -> Generator[Any, None, None]:
        # [FUNCTION DOC] HELPER: Yields registered user-defined triton kernels
        # associated with the node's target: for an op packet, every overload
        # attribute found; for a single overload, its own op name. Other targets
        # yield nothing.
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            for attr in node.target._dir:
                custom_op = getattr(node.target, attr, None)
                if custom_op is not None:
                    yield from torch._library.triton.get_triton_kernels_for_op(
                        custom_op._name
                    )
            return
        if isinstance(node.target, torch._ops.OpOverload):
            yield from torch._library.triton.get_triton_kernels_for_op(
                node.target._name
            )

    def get_triton_source_codes_from_gm(
        self,
        gm: torch.fx.GraphModule,
    ) -> list[str]:
        # [FUNCTION DOC] Collects user-defined triton kernels referenced by any node
        # in gm or its nested graph modules and returns, per kernel, the transitive
        # source code string (autotuned kernels are unwrapped to the inner function).
        # Raises AssertionError if triton is unavailable.
        if not has_triton_package():
            raise AssertionError("Triton is not available")

        triton_kernels = []
        for module in _iter_graph_modules(gm):
            for node in module.graph.nodes:
                triton_kernels.extend(self._iter_triton_kernels_from_node(node))

        triton_kernel_source_codes = []
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        for kernel in triton_kernels:
            from triton.runtime.autotuner import Autotuner

            if isinstance(kernel, Autotuner):
                # Grab the Inner JITFunction
                kernel = kernel.fn
            source_codes = user_defined_triton_kernel_transitive_closure_source_code(
                kernel
            )
            triton_kernel_source_codes.append(source_codes)

        return triton_kernel_source_codes

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
        act_input_paths: ActInputPaths = (),
    ) -> None:
        # [FUNCTION DOC] Captures runtime state (grad mode, autocast, determinism,
        # autograd config, triton sources), saved-tensors-hook and SAC hashes, the
        # graph-wide activation memory budget taken from the first node, and custom
        # estimator/solver uuids (read from the live config module), then initializes
        # the inductor hash details. Raises BypassAOTAutogradCache when anything is
        # not hashable or inductor says the graph cannot be cached.
        # FxGraphHashDetails contains all the keys related to inductor. Also
        # includes some system info.
        self.aot_config = aot_config
        self.act_input_paths = tuple(act_input_paths)
        self._record_runtime_state(gm)
        self.saved_tensors_hooks_fx_wrap_cache_hashes = (
            _collect_saved_tensors_hooks_fx_wrap_cache_hashes(gm)
        )
        self.sac_context_fn_hashes = _collect_context_fn_hashes(gm)

        # region_activation_memory_budget is graph-wide (the partitioner enforces
        # a single value across the graph) and propagates to every node, so the
        # cache key only needs the value off the first node. node.meta is stripped
        # by GraphModule.__reduce__, so without recording it here a budget change
        # would not invalidate the cache.
        first_node = next(iter(gm.graph.nodes), None)
        self.region_activation_memory_budget: float | None = (
            _get_memory_budget_annotation(first_node)
            if first_node is not None
            else None
        )

        # Note: We use the live config module, not self.autograd_config (the
        # saved config), because activation_memory_budget_runtime_estimator and
        # activation_memory_budget_solver are excluded from save_config (in
        # _save_config_ignore) since they're not serializable. We must access the
        # config module directly to get the patched runtime values.
        self.custom_estimator_solver_uuids = _get_custom_estimator_solver_uuids(config)
        self._init_fx_graph_hash_details(gm, example_inputs, fx_config)

    def _record_runtime_state(self, gm: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] HELPER: Snapshots grad-enabled state, the autocast dtype of
        # each enabled autocast device, the deterministic-algorithms flag and the
        # saved functorch config onto self; if triton is installed also records the
        # triton kernel sources used by gm.
        self.grad_enabled = torch.is_grad_enabled()
        # Include per-device autocast dtype in cache key to avoid reusing
        # a graph compiled for one autocast dtype (e.g. bfloat16) when
        # running under a different autocast dtype (e.g. float16).
        self.autocast_state: dict[str, torch.dtype] = {}
        for device_type in torch._C._autocast_supported_devices():
            if torch.is_autocast_enabled(device_type):
                self.autocast_state[device_type] = torch.get_autocast_dtype(device_type)
        self.deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.autograd_config = config.save_config()
        if has_triton_package():
            self.triton_kernel_source_codes = self.get_triton_source_codes_from_gm(gm)

    def _init_fx_graph_hash_details(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        fx_config: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] HELPER: Checks that inductor's FX graph cache can handle the
        # graph and config, then runs the parent hash-details initializer (with no
        # extra inputs list). Converts inductor's BypassFxGraphCache into
        # BypassAOTAutogradCache.
        try:
            # FXGraphCache has constraints on what can be pickled in its inductor
            # config. Check that the gm is cacheable by inductor first,
            # and if it raises an exception, also bypass on our end.
            FxGraphCache._check_can_cache(gm, example_inputs, fx_config)
            super().__init__(gm, example_inputs, fx_config, [])
        except BypassFxGraphCache as e:
            # Sometimes inductor configs are unpickleable and can fail
            raise BypassAOTAutogradCache(str(e)) from e


class AOTAutogradCachePickler(FxGraphCachePickler):
    # [CLASS DOC] Pickler producing deterministic bytes for cache keys. Extends the
    # inductor FX graph pickler with stable reducers for AOTConfig, tensors (metadata
    # only), tensor subclasses (stable hash), fake script objects, graph modules
    # (generated code plus non-graph state) and numpy wrappers. State: a mapping from
    # raw symint names to normalized names and the set of symint names appearing in
    # the graph.
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        symint_name_map: dict[str, str] | None = None,
        graph_symint_names: set[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base pickler for gm, stores the optional
        # symint name normalization map and graph symint name set (default empty),
        # and registers the AOTConfig, tensor and fake script object reducers in
        # the dispatch table.
        super().__init__(gm)
        self._symint_name_map: dict[str, str] = symint_name_map or {}
        self._graph_symint_names: set[str] = graph_symint_names or set()
        # pyrefly: ignore[missing-attribute]
        self.dispatch_table.update(
            {
                AOTConfig: functools.partial(self._reduce_aot_config),
                torch.Tensor: functools.partial(self._reduce_tensor),
                FakeScriptObject: functools.partial(self._reduce_fake_script_object),
            }
        )

    def _reduce_symint(
        self, s: torch.SymInt
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # [FUNCTION DOC] HELPER: Reducer for symbolic ints. Looks up the normalized
        # name; for graph symints returns just the normalized name, for other
        # normalized symints returns the normalized name with the symbolic hint (when
        # available), and for unknown names returns the raw name. Result is an
        # identity-reconstruction pair used by the pickler.
        # KNOWN ISSUE: not registered in the dispatch table in this class.
        name = str(s)
        normalized = self._symint_name_map.get(name)
        if normalized is not None:
            if name in self._graph_symint_names:
                return (_ident, (normalized,))
            # Non-graph SymInt with non-deterministic name (from MetaConverter).
            # Include the hint to distinguish different specializations.
            hint = s.node.hint
            if hint is not None:
                return (_ident, (normalized, hint))
        return (_ident, (name,))

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        # [FUNCTION DOC] Pickler hook called for every object. Graph modules go to the
        # graph module reducer; numpy wrappers reduce to their cache key; tensor
        # subclasses (not exact tensors) go to the subclass reducer; everything else
        # falls back to the parent implementation. Returns a reduce tuple or the
        # parent's result.
        """
        Override to handle tensor subclasses (like DTensor) that aren't caught
        by the dispatch_table's exact type matching.

        The dispatch_table only matches exact types, so subclasses like DTensor
        fall through to the default __reduce_ex__ which includes non-deterministic
        storage addresses. This method catches those cases using isinstance checks.
        """
        if isinstance(obj, torch.fx.GraphModule):
            return self._reduce_graph_module_for_cache_key(obj)
        if (numpy_key := numpy_wrapper_cache_key(obj)) is not None:
            return (_ident, (numpy_key,))
        # Handle tensor subclasses that aren't exactly torch.Tensor
        # dispatch_table already handles torch.Tensor exactly
        if isinstance(obj, torch.Tensor) and type(obj) is not torch.Tensor:
            return self._reduce_tensor_subclass(obj)
        # Fall back to parent class reducer which handles unpicklable types
        return super().reducer_override(obj)

    @staticmethod
    def _format_global_for_cache_key(name: str, obj: Any) -> str:
        # [FUNCTION DOC] HELPER: Static method rendering one generated-code global as
        # a line of text for the key: a comment line with the numpy wrapper cache key
        # for numpy wrappers, otherwise the import statement for the object computed
        # with the package system importer.
        if (numpy_key := numpy_wrapper_cache_key(obj)) is not None:
            return f"# cache_key_numpy_wrapper {name}: {numpy_key!r}"

        from torch.fx.graph_module import _format_import_statement
        from torch.package import sys_importer

        return _format_import_statement(name, obj, sys_importer)

    def _reduce_graph_module_for_cache_key(
        self, gm: torch.fx.GraphModule
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # [FUNCTION DOC] HELPER: Reducer for GraphModules. Regenerates the Python code
        # (using the real recompile if the module overrides it), copies the module
        # state dict minus the graph object, and builds a sorted block of import lines
        # for the code's globals. The pair (state, import block) is the stable payload.
        # Side effect: triggers a recompile of gm.
        recompile = getattr(gm, "_real_recompile", gm.recompile)
        python_code = recompile()
        # FxGraphCache strips user-defined Triton side-table indices from
        # Inductor wrapper code. This reducer handles arbitrary AOT GraphModule
        # code, so keep the generated code exact and let AOTAutogradCacheDetails
        # carry Triton source/constant-arg metadata separately.
        dict_without_graph = gm.__dict__.copy()
        dict_without_graph.pop("_graph", None)

        import_block = "\n".join(
            sorted(
                self._format_global_for_cache_key(name, obj)
                for name, obj in python_code.globals.items()
            )
        )
        return (_ident, ((dict_without_graph, import_block),))

    # [NOTE] Tensor subclass stable hashing for AOT autograd cache
    # Python's hash() varies with PYTHONHASHSEED, making cache keys unstable
    # across processes. We use blake2b for cross-process determinism.
    #
    # EXTENSION POINT: Traceable wrapper subclasses can override cache key
    # generation by implementing _stable_hash_for_caching(self) -> str.
    # This method should return a deterministic string that uniquely identifies
    # the tensor's metadata for caching purposes. See DTensor for an example.
    #
    # We can't define a default method on subclasses because there is no abstract
    # base subclass, and we don't want to pollute torch.Tensor. Instead, we provide
    # a default implementation here that uses __tensor_flatten__ to recursively
    # hash inner tensors and metadata.

    def _hash_bytes_for_cache(self, data: bytes) -> str:
        # [FUNCTION DOC] HELPER: Returns a 16-byte blake2b hex digest of data, stable
        # across processes (unlike the built-in hash).
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _hash_pickled_value_for_cache(self, value: Any) -> str:
        # [FUNCTION DOC] HELPER: Serializes value with the standard pickler and
        # returns the stable digest of the bytes. Raises whatever pickling raises.
        return self._hash_bytes_for_cache(pickle.dumps(value))

    def _stable_hash_for_cache_value(self, obj: Any) -> str:
        # [FUNCTION DOC] HELPER: Stable hash of an object found in subclass metadata.
        # Order: object-provided stable hash method; traceable wrapper subclass via
        # the default recursive hash; opaque custom class instance via its type name
        # only; plain tensor via cache-key metadata; anything else via pickled bytes.
        """Get a stable hash for an object used inside tensor subclass metadata."""
        from torch._custom_class_base import CustomClassBase
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if hasattr(obj, "_stable_hash_for_caching"):
            return obj._stable_hash_for_caching()
        if isinstance(obj, torch.Tensor) and is_traceable_wrapper_subclass(obj):
            return self._default_stable_hash_for_caching(obj)
        if isinstance(obj, CustomClassBase):
            # Opaque objects are runtime pass-throughs; only the type matters
            # for cache key purposes, not the instance identity or value.
            return self._hash_bytes_for_cache(type(obj).__qualname__.encode())
        if isinstance(obj, torch.Tensor):
            return self._hash_pickled_value_for_cache(
                extract_tensor_metadata_for_cache_key(obj)
            )
        return self._hash_pickled_value_for_cache(obj)

    def _reduce_tensor_subclass(
        self, tensor: torch.Tensor
    ) -> tuple[Callable[..., Any], tuple[Any]]:
        # [FUNCTION DOC] HELPER: Reducer for tensor subclass instances. Uses the
        # subclass's own stable-hash method if present; else for traceable wrapper
        # subclasses warns once and uses the default recursive hash; else falls back
        # to the plain tensor metadata reducer. Returns an identity-reconstruction pair.
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if hasattr(tensor, "_stable_hash_for_caching"):
            return (_ident, (tensor._stable_hash_for_caching(),))
        if is_traceable_wrapper_subclass(tensor):
            warn_once(
                f"{type(tensor).__name__} does not implement _stable_hash_for_caching. "
                "For PT2-compatible tensor subclasses, it is recommended to implement "
                "_stable_hash_for_caching(self) -> str for stable AOT autograd caching."
            )
            return (_ident, (self._default_stable_hash_for_caching(tensor),))
        return self._reduce_tensor(tensor)

    def _collect_inner_tensor_hashes(
        self, tensor: torch.Tensor, inner_tensor_names: Sequence[str]
    ) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Maps each named inner tensor attribute of a subclass
        # to its stable hash string.
        inner_hashes = {}
        for name in inner_tensor_names:
            inner_hashes[name] = self._stable_hash_for_cache_value(
                getattr(tensor, name)
            )
        return inner_hashes

    def _stabilize_tensor_subclass_metadata(self, obj: Any) -> Any:
        # [FUNCTION DOC] HELPER: Recursively rebuilds tuples, lists and mappings of
        # subclass metadata, replacing opaque custom class instances by their type
        # qualified name so the result is deterministic. Other values are unchanged.
        from torch._custom_class_base import CustomClassBase

        if isinstance(obj, CustomClassBase):
            return type(obj).__qualname__
        if isinstance(obj, tuple):
            return tuple(self._stabilize_tensor_subclass_metadata(x) for x in obj)
        if isinstance(obj, list):
            return [self._stabilize_tensor_subclass_metadata(x) for x in obj]
        if isinstance(obj, dict):
            return {
                k: self._stabilize_tensor_subclass_metadata(v) for k, v in obj.items()
            }
        return obj

    def _default_stable_hash_for_caching(self, tensor: torch.Tensor) -> str:
        # [FUNCTION DOC] HELPER: Stable hash for a traceable wrapper subclass built
        # from its shape, requires_grad flag, stabilized flatten metadata and the
        # stable hashes of its inner tensors.
        """
        Default stable hash implementation for traceable wrapper subclasses.
        """
        inner_tensor_names, subclass_metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
        cache_payload = (
            tensor.shape,
            tensor.requires_grad,
            self._stabilize_tensor_subclass_metadata(subclass_metadata),
            self._collect_inner_tensor_hashes(tensor, inner_tensor_names),
        )
        return self._hash_pickled_value_for_cache(cache_payload)

    def _reduce_aot_config(
        self, aot_config: AOTConfig
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # [FUNCTION DOC] HELPER: Reducer for AOTConfig. Converts to the cacheable
        # config and keeps only key-relevant fields (parameter/buffer count, input
        # mutation flag, export, no-tangents, dynamic shapes, arg position sources,
        # static input indices, logging flag, pre-dispatch, precompile backend id),
        # deliberately omitting the debug aot_id.
        """
        Reduce the config to a stable key for caching.
        """
        cacheable = aot_config.to_cacheable()
        return (
            # Cache entries persist the full CacheableAOTConfig, but cache keys
            # intentionally ignore the debug-only aot_id so equivalent graphs can hit.
            _ident,
            (
                cacheable.num_params_buffers,
                cacheable.keep_inference_input_mutations,
                cacheable.is_export,
                cacheable.no_tangents,
                cacheable.dynamic_shapes,
                cacheable.aot_autograd_arg_pos_to_source,
                cacheable.static_input_indices,
                cacheable.enable_log,
                cacheable.pre_dispatch,
                cacheable.precompile_backend_id,
            ),
        )

    def _reduce_tensor(self, t: torch.Tensor) -> tuple[Callable[..., Any], tuple[Any]]:
        # [FUNCTION DOC] HELPER: Reducer for plain tensors; only the cache-key
        # metadata of the tensor is used, never its data or addresses.
        """
        Reduce the tensor to a stable key for caching.
        """
        metadata = extract_tensor_metadata_for_cache_key(t)
        return (_ident, (metadata,))


@contextlib.contextmanager
def normalize_placeholder_names(
    gm: torch.fx.GraphModule,
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that temporarily renames graph placeholders to
    # canonical names (plain inputs p_0, p_1, ... and symbolic-int inputs s_0, ...,
    # in sorted placeholder order), recompiling the module, so isomorphic graphs hash
    # alike. Does nothing when normalization is disabled in config or gm has no graph.
    # On exit (even on error) restores the original node names, targets and the
    # graph's used-name set, recompiles again, and raises AssertionError if the
    # restored counts mismatch. Mutates gm in place for the duration.
    """
    Context manager that normalizes the placeholder names in the graph module.
    This is used while generating a cache key for AOTAutogradCache, so that two graphs
    that are isomorphic when normalizing names can hit the same cache entry.
    This is safe because nothing underneath AOTAutograd uses the node names on the
    original dynamo graph: AOTAutograd re-traces with its own nodes, and guards are
    in terms of original sources rather than placeholder names.
    """
    # Standalone inductor: we're bypassing AOTAutogradCache anyway, so return the graph
    # as-is
    if not config.autograd_cache_normalize_inputs or not hasattr(gm, "graph"):
        yield
        return

    # Track all the old state of placeholders
    old_placeholder_names = []
    old_symint_names = []
    old_used_names = copy(gm.graph._graph_namespace._used_names)
    i = 0
    j = 0
    for n in gm.graph.find_nodes(op="placeholder", sort=True):
        if n.type == torch.SymInt:
            new_name = f"s_{j}"
            old_symint_names.append((n.name, n.target))
            n.target = new_name
            n._rename(new_name)
            j += 1
        else:
            # _rename renames the node in the body of the function,
            # but it doesn't change the raw name from node.target
            # So we also set the raw_name of node.target to a new placeholder name
            new_placeholder_name = f"p_{i}"
            old_placeholder_names.append((n.name, n.target))
            n.target = new_placeholder_name
            n._rename(new_placeholder_name)
            i += 1
    gm.recompile()
    try:
        yield
    finally:
        # Used_names contains all our old placeholder names,
        # so we clear it temporarily when we put them back
        gm.graph._graph_namespace._used_names = set()
        # Restore the placeholder names
        i = 0
        j = 0
        for n in gm.graph.find_nodes(op="placeholder", sort=True):
            if n.type == torch.SymInt:
                (name, target) = old_symint_names[j]
                n.target = target
                n._rename(name)
                j += 1
            else:
                (name, target) = old_placeholder_names[i]
                n.target = target
                n._rename(name)
                i += 1
        if i != len(old_placeholder_names):
            raise AssertionError(
                f"i={i} != len(old_placeholder_names)={len(old_placeholder_names)}"
            )
        if j != len(old_symint_names):
            raise AssertionError(
                f"j={j} != len(old_symint_names)={len(old_symint_names)}"
            )
        # Now restore the old namespace's used names
        gm.graph._graph_namespace._used_names = old_used_names
        gm.recompile()


def create_fx_config(
    compiler_config_extra: CompilerConfigExtra | None = None,
    compile_region_name: str | None = None,
) -> _CompileFxKwargs:
    # [FUNCTION DOC] Builds the inductor compile keyword mapping used for hashing and
    # loading: cudagraphs flag, forward device index and compile region name. Without
    # extra config, cudagraphs comes from a boxed copy of the inductor triton
    # cudagraphs setting and the device index is None.
    if compiler_config_extra is None:
        cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
        boxed_forward_device_index = None
    else:
        cudagraphs = compiler_config_extra.cudagraphs
        boxed_forward_device_index = compiler_config_extra.forward_device
    return {
        "cudagraphs": cudagraphs,
        "boxed_forward_device_index": boxed_forward_device_index,
        "compile_region_name": compile_region_name,  # pyrefly: ignore[bad-typed-dict-key]
    }


def _check_triton_cache_version() -> None:
    # [FUNCTION DOC] HELPER: Raises BypassAOTAutogradCache if triton is installed
    # with a version string below 3.2.0 (known CUDA-context init problem when
    # loading cache entries); returns None otherwise or when triton is absent.
    # KNOWN ISSUE: compares version strings lexicographically, which misorders
    # versions such as 3.10.
    if not has_triton_package():
        return

    # Due to https://github.com/triton-lang/triton/issues/3729, if triton is <
    # 3.2.0, AOTAutogradCache may cause us to attempt to load a cache entry
    # without initializing the CUDA context on the autograd thread.
    #
    # Without caching, we naturally do this initialization when tracing through
    # the graph with the autograd engine.
    import triton

    if triton.__version__ < "3.2.0":
        raise BypassAOTAutogradCache("AOTAutogradCache requires triton 3.2.0")


def _get_debug_lines_for_cache_key(
    pickler: AOTAutogradCachePickler,
    details: AOTAutogradCacheDetails,
    key: str,
) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns per-attribute hash description lines of the
    # details object from the pickler, and logs them at debug level; returns an empty
    # list (skipping the costly work) unless debug logging is enabled.
    # debug_lines re-hashes every attribute individually and is expensive. Only
    # compute when debug logging is enabled.
    if not log.isEnabledFor(logging.DEBUG):
        return []

    debug_lines = pickler.debug_lines(details)
    log.debug(
        "Autograd graph cache hash details for key %s:\n%s",
        key,
        LazyString(lambda: "\n".join(debug_lines)),
    )
    return debug_lines


def autograd_cache_key(
    mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
    example_inputs: Sequence[Any],
    config: AOTConfig,
    compiler_config_extra: CompilerConfigExtra | None = None,
    act_input_paths: ActInputPaths = (),
    # TODO: add args and parameters
) -> tuple[str, list[str]]:
    # [FUNCTION DOC] Computes the AOTAutograd cache key for a graph module (or its
    # wrapper) and example inputs. Records graph and input symint names for
    # normalization, then, with the module sanitized, checks cacheability and the
    # triton version, builds the details object and pickler, and returns the key
    # (cache prefix plus hash) with the debug description lines. If key generation
    # fails, returns a random nonce key and no lines when the bypass-key config is on,
    # otherwise re-raises (typically BypassAOTAutogradCache).
    """
    Generate a unique hash of the FX graph for caching.
    """

    gm = mod.gm if isinstance(mod, torch._dynamo.utils.GmWrapper) else mod
    try:
        # Capture graph SymInt placeholder names before normalization so the
        # pickler can distinguish graph-native SymInts (deterministic names)
        # from MetaConverter-generated SymInts (non-deterministic names).
        graph_symint_names: set[str] = set()
        symint_name_map: dict[str, str] = {}
        if torch._functorch.config.autograd_cache_normalize_inputs:
            for n in gm.graph.find_nodes(op="placeholder", sort=True):
                if n.type == torch.SymInt:
                    graph_symint_names.add(n.target)
            j = 0
            for inp in example_inputs:
                if isinstance(inp, torch.SymInt):
                    symint_name_map[str(inp)] = f"s_{j}"
                    j += 1
        with sanitize_gm_for_cache(gm):
            check_cacheable(gm)
            _check_triton_cache_version()
            details = AOTAutogradCacheDetails(
                gm,
                example_inputs,
                config,
                create_fx_config(compiler_config_extra),
                act_input_paths,
            )
            pickler = AOTAutogradCachePickler(gm, symint_name_map, graph_symint_names)
            # The prefix distinguishes among the other kinds of objects we cache
            key = AOTAUTOGRAD_CACHE_PREFIX + pickler.get_hash(details)
            debug_lines = _get_debug_lines_for_cache_key(pickler, details, key)
            return key, debug_lines
    except Exception:
        # If enable_aot_compile is set, we're in AOT precompile mode where we always
        # want to use fallback nonce keys. Unlike caching, it's fine if we can't generate
        # a proper key because we are guaranteed in an AOT precompile world users are in
        # complete control of distributing and loading artifacts.
        if torch._functorch.config.bypass_autograd_cache_key:
            log.info(
                "Failed to generate AOTAutograd cache key; falling back to nonce due to enable_aot_compile",
                exc_info=True,
            )
            return str(random.random()), []
        else:
            raise


@contextlib.contextmanager
def sanitize_gm_for_cache(
    gm: torch.fx.GraphModule,
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that resets unstable Dynamo-supplied attributes
    # of gm (meta, compile reason, parameter-name sources, backend id) to defaults
    # and normalizes placeholder names while the body runs, so the cache key is
    # canonical. Always restores the original attribute values on exit.
    """
    Clears a few fields in a dynamo supplied Graph Module that are not stable between graph inputs, but don't
    affect inductor or aotdispatch correctness.

    These fields **can** be used by code calling into aotdispatch (namely, dynamo), so we can't null them out completely.

    To ensure that these fields are not accessed by inductor or aotdispatch, we clear them during AOTAutogradCache.load,
    and then put them back before returning. This way, we generate a cache key based off of a canonical graph
    without these fields, and also guarantee they aren't used to affect the cache's output.
    """
    # Mapping from each field to a default value
    IGNORED_FIELDS: dict[str, Any] = {
        # pyrefly: ignore [implicit-any]
        "meta": {},  # metadata used by export
        "compile_subgraph_reason": None,  # Used by dynamo only for logging, no change in inductor/autograd behavior
        "_param_name_to_source": None,  # Encapsulated by aot_config.aot_autograd_arg_pos_to_source
        "_backend_id": None,
    }
    saved_fields = {}
    for field, default_value in IGNORED_FIELDS.items():
        saved_fields[field] = getattr(gm, field, None)
        # Clear the field
        setattr(gm, field, default_value)
    try:
        with normalize_placeholder_names(gm):
            yield
    finally:
        for field, value in saved_fields.items():
            setattr(gm, field, value)


@CacheArtifactFactory.register
class AOTAutogradCacheArtifact(CacheArtifact):
    # [CLASS DOC] Registered cache artifact type for serialized AOTAutograd entries
    # (key plus pickled bytes), used to export and re-import compiler caches.
    @override
    def populate_cache(self) -> None:
        # [FUNCTION DOC] Writes this artifact's content into the local AOTAutograd
        # cache directory under its key. Side effect: file I/O.
        AOTAutogradCache._write_to_local_cache(self.key, self.content)

    @override
    @staticmethod
    def type() -> str:
        # [FUNCTION DOC] Returns the artifact type tag "aot_autograd".
        return "aot_autograd"


class AOTAutogradCache(GuardedCache[GenericAOTAutogradResult[Any, Any]]):
    # [CLASS DOC] Stateless (all static/class methods) front end for saving and loading
    # AOTAutograd compile results in a local directory and/or remote cache, keyed by
    # autograd_cache_key and guarded by dynamic-shape guard expressions. Entries are
    # pickled GenericAOTAutogradResult objects. Key methods: try_load, save, make_entry,
    # clear, generate_guards_expression, evaluate_guards.
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradResult handles the wrapping/unwrapping logic.

    Cache Inputs (AOTAutogradCacheDetails)
    - AOTAutogradCache takes in the following inputs, which are analogous to inputs given
        to AOTAutograd by dynamo:
        - A fx graph module generated by dynamo
        - A list of args, which consists of:
            - Symint inputs to the graph, generated by dynamo
            - The **real tensor** inputs, which inductor uses for cudagraphs
            - Notably, the real tensor inputs don't have symints in their metadata.
        AOTAutograd then retraces those real tensor arguments into FakeTensors later during execution.
        - A set of global configurations that affect AOTAutograd or Inductor behavior.

    It then generates a cache key given these values. Notably, this means AOTAutogradCache currently
    specializes on the sizes and strides of the real tensor inputs when dynamic shapes are turned on.
    In a later PR, we'll likely generate the cache key based on the FakeTensors AOTAutograd generates
    based on the real tensor inputs, which can contain symints.

    # Cache Outputs (AOTAutogradResult)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradResult for more info

    [Note: Caching guards generated by AOTAutograd and Inductor]
    AOTAutograd and inductor both can introduce new guards to the shape environment. FXGraphCache saves guards with each
    compiled graph inductor generates. On a cache hit, AOTAutograd reloads the compiled forward and backward functions
    from FXGraphCache, giving it new symint arguments from the input args.
    FXGraphCache uses those symints and its saved guards to repopulate the ShapeEnv with guards.
    **No new guards are generated into the shape env after inductor finishes compiling**, so the guards
    saved by inductor are sufficient for correctness for both AOTAutograd and Inductor's caches.
    """

    @staticmethod
    def clear() -> None:
        # [FUNCTION DOC] Deletes the whole local AOTAutograd cache directory tree,
        # ignoring errors. Side effect: file deletion.
        """Clear the cache"""
        shutil.rmtree(AOTAutogradCache._get_tmp_dir(), ignore_errors=True)

    @staticmethod
    def try_load(
        mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
        args: list[Any],
        aot_config: AOTConfig,
        compiler_config_extra: CompilerConfigExtra | None,
        local: bool,
        remote: bool,
        act_input_paths: ActInputPaths = (),
        compile_region_name: str | None = None,
    ) -> tuple[Callable[..., Any] | None, AOTConfig]:
        # [FUNCTION DOC] Main load entry. Computes the key, looks up local and/or
        # remote caches, and on a hit rebuilds the runtime-wrapped compiled function
        # (made serializable by re-unpickling the stored bytes), reporting time saved
        # and any distributed timeout increase. On a miss or bypass, returns None and,
        # if a key exists, an AOT config copy carrying cache info (key, timestamp,
        # backed forward symints) for the later save. Never raises on cache problems
        # (counts hit/miss/bypass, records reasons), except re-raising in strict
        # autograd-cache or strict-precompile mode. Side effects: counters, logging,
        # structured-trace and chromium event logging.
        # KNOWN ISSUE: time saved values use float division by one million with
        # floor, so they are floats, not ints.
        """
        Load a result from the cache, and reconstruct a runtime wrapper around
        the object. On a cache miss, return an updated config carrying the
        cache metadata needed to save the eventual compile result.
        """
        compiled_fn = None
        cache_info: dict[str, Any] = {}
        cache_key = None
        debug_lines: list[str] = []
        cache_event_time = time.time_ns()
        cache_state = None
        try:
            cache_key, debug_lines = autograd_cache_key(
                mod, args, aot_config, compiler_config_extra, act_input_paths
            )
            result: tuple[GenericAOTAutogradResult[Any, Any], bytes] | None = (
                AOTAutogradCache._lookup(
                    cache_key, local, remote, args, cache_info, aot_config
                )
            )
            if result is not None:
                (entry, pickled_content) = result
                fx_config = create_fx_config(compiler_config_extra, compile_region_name)
                compiled_fn = entry.wrap_post_compile(args, aot_config, fx_config)
                # Make the compiled_fn serializable, where the serialize function just
                # makes a copy of the original entry before post compile via the pickled content
                compiled_fn = SerializableCompiledFunction(
                    compiled_fn, lambda: pickle.loads(pickled_content)
                )
                log.info("AOTAutograd cache hit for key %s", cache_key)

                counters["aot_autograd"]["autograd_cache_hit"] += 1
                cache_state = "hit"
                cache_event_time = time.time_ns()
                forward_time_saved = entry.forward_time_taken_ns // 1e6
                backward_time_saved = entry.backward_time_taken_ns // 1e6
                cache_info.update(
                    {
                        "forward_time_saved_ms": forward_time_saved,
                        "backward_time_saved_ms": backward_time_saved,
                        "time_saved_ms": forward_time_saved + backward_time_saved,
                    }
                )
                time_saved_ns = (
                    entry.forward_time_taken_ns + entry.backward_time_taken_ns
                )
                # TODO: should we use the same field for remote cache time saved for both
                # FXGraphCache and AOTAutogradCache?
                # get_metrics_context().increment(...)
                if (
                    ephemeral_increase
                    := add_ephemeral_timeout_increase_for_distributed(time_saved_ns)
                ) != 0:
                    cache_info["ephemeral_timeout_increase"] = ephemeral_increase

            if compiled_fn is None:
                log.info("AOTAutograd cache miss for key %s", cache_key)
                counters["aot_autograd"]["autograd_cache_miss"] += 1
                cache_state = "miss"
                cache_event_time = time.time_ns()
        # Count missing the FXGraphCache as a miss not a bypass
        except FXGraphCacheMiss as e:
            counters["aot_autograd"]["autograd_cache_miss"] += 1
            cache_state = "miss"
            if config.strict_autograd_cache or torch._dynamo.config.strict_precompile:
                raise e
        # Most often this is BypassAOTAutogradCache, but
        # if there's ever different reason we can't cache,
        # we still never want to hard throw an exception, since
        # we can always fallback to a cache bypass.
        # As an example, if the user calls autograd via
        # standalone inductor, we will sometimes get a GraphModule
        # that doesn't actually have a `.graph` on it. Instead
        # of checking every single case, we safely catch the exception
        # in those cases.
        except Exception as e:
            cache_key = None
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)
            cache_state = "bypass"
            cache_event_time = time.time_ns()
            cache_info["cache_bypass_reason"] = str(e)
            cache_info["cache_bypass_exception_type"] = type(e).__name__
            cache_info["cache_bypass_traceback"] = traceback.format_exc().split("\n")
            # TODO: this gets logged implicitly by cache_bypass_reason,
            # and here we explicitly log it into tlparse.
            # We may want to log this as an extra column in Scuba, though.
            cache_info["cache_bypass_hard_exception"] = not isinstance(
                e, BypassAOTAutogradCache
            )
            if remote:
                log_cache_bypass("bypass_aot_autograd", str(e))
            if config.strict_autograd_cache or torch._dynamo.config.strict_precompile:
                raise e
        if compiled_fn is None:
            # Set the cache key so we can save a cache result later
            symints = AOTAutogradCache._filter_backed_symints(args)
            if cache_key is not None:
                aot_config = dataclasses.replace(
                    aot_config,
                    cache_info=AOTAutogradCacheInfo(
                        cache_key,
                        time.time_ns(),
                        forward_symints=symints,
                    ),
                )

        cache_info.update(
            {
                "key": cache_key,
                "cache_state": cache_state,
                "components": debug_lines,
            }
        )
        if chromium_event_log_active():
            CompileEventLogger.instant(
                f"autograd_cache_{cache_state}",
                metadata=cache_info,
                time_ns=cache_event_time,
            )
            CompileEventLogger.try_add_pt2_compile(
                "backend_compile",
                cache_state=cache_state,
                cache_event_time=cache_event_time,
                key=cache_info.get("key"),
                components=cache_info.get("components"),
                cache_bypass_reason=cache_info.get("cache_bypass_reason"),
                remote_cache_enabled=remote,
                local_cache_enabled=local,
            )

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"aotautograd_cache_{cache_state}",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )

        return compiled_fn, aot_config

    @classmethod
    def generate_guards_expression(
        cls: type[AOTAutogradCache], cache_info: AOTAutogradCacheInfo
    ) -> str | None:
        # [FUNCTION DOC] Builds the dynamic-shape guard expression string for the
        # forward symint inputs from the active shape env: prunes guards to those
        # relevant to the symints, then renders them. Returns None if there is no
        # shape env.
        shape_env = cls._get_shape_env()

        if shape_env is None:
            return None

        symints = cache_info.forward_symints
        guards = shape_env.get_pruned_guards(symints)
        return shape_env.produce_guards_expression(placeholders=symints, guards=guards)

    @classmethod
    def _get_tmp_dir(cls: type[AOTAutogradCache]) -> str:
        # [FUNCTION DOC] HELPER: Returns the root directory of the local cache (an
        # aotautograd folder inside the inductor cache dir).
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "aotautograd")

    @classmethod
    def _get_tmp_dir_for_key(cls: type[AOTAutogradCache], _key: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the per-key subdirectory (root joined with
        # the key) holding entries for that key.
        # KNOWN ISSUE: the docstring repeats the toplevel-directory description,
        # but the code returns the per-key subdirectory.
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cls._get_tmp_dir(), _key)

    @classmethod
    def _record_result(
        cls: type[AOTAutogradCache],
        key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        # [FUNCTION DOC] Callback from the guarded cache base class. Updates cache
        # statistics for local hits and misses and increments top-level compile event
        # counters for local/remote hits and misses. The key argument is unused.
        """
        Called by GuardedCache to record hit/miss statistics.
        """
        if local_hit:
            cache_stats.hit("LocalAOTAutogradCache")
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_hit_count",
            )
        if remote_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_hit_count",
            )
        if local_miss:
            cache_stats.miss("LocalAOTAutogradCache")
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_miss_count",
            )
        if remote_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_miss_count",
            )

    @staticmethod
    def evaluate_guards(guard_expr: str, hints: list[int] | list[torch.SymInt]) -> bool:
        # [FUNCTION DOC] Evaluates a stored guard expression against the given symint
        # hint values using the active shape env; returns True (accept) immediately
        # when the inductor config skips dynamic-shape cache guards. Raises
        # AssertionError if no shape env exists.
        if torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards:
            return True
        shape_env = AOTAutogradCache._get_shape_env()
        if shape_env is None:
            raise AssertionError("shape_env must not be None")
        result = shape_env.evaluate_guards_expression(guard_expr, hints)
        return result

    @staticmethod
    def _lookup(
        key: str,
        local: bool,
        remote: bool,
        args: list[Any],
        cache_info: dict[str, Any],
        aot_config: AOTConfig | None,
    ) -> tuple[GenericAOTAutogradResult[Any, Any], bytes] | None:
        # [FUNCTION DOC] HELPER: Finds a cached entry for key in local and/or remote
        # cache whose guards pass for the backed symint hints of args. Fills cache_info
        # with guard lookup details, bumps a guard-miss counter, records the artifact
        # for cache export, and (in bundling mode with a backend id) records a bundled
        # precompile artifact. Returns (entry, pickled bytes) or None. Errors while
        # loading are logged and swallowed unless strict mode is on.
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        remote_cache: RemoteCache[JsonDataTy] | None = None
        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()

        symints = AOTAutogradCache._filter_backed_symints(args)
        hints = [guarding_hint_or_throw(s) for s in symints]
        entry = None
        pickled_content = None
        try:
            (
                entry,
                pickled_content,
                guard_info,
            ) = AOTAutogradCache.find_guarded_entry(
                key, local, remote_cache, AOTAutogradCache.evaluate_guards, hints
            )

            if entry is None and guard_info["cache_status_detailed"] == "guard_miss":
                counters["aot_autograd"]["autograd_cache_guard_miss"] += 1
            cache_info.update(guard_info)
            CacheArtifactRecorder(
                AOTAutogradCacheArtifact.type(), key
            ).record_if_present(pickled_content)
            if pickled_content is not None:
                if (
                    should_bundle_autograd_cache()
                    and aot_config is not None
                    and aot_config.precompile_backend_id is not None
                ):
                    # NB: We don't want to use the cached aot_config.precompile_backend_id
                    # 1. because we set it to None on save 2. even if we didn't, this new run
                    # that cache hit has a *new* backend id associated with it.
                    PrecompileContext.record_artifact(
                        BundledAOTAutogradCacheArtifact(
                            aot_config.precompile_backend_id, entry
                        ),
                    )
        except Exception as e:
            log.info("AOTAutograd cache unable to load compiled graph: %s", e)
            if config.strict_autograd_cache:
                raise e
        if entry is not None:
            if pickled_content is None:
                raise AssertionError(
                    "pickled_content must not be None when entry is not None"
                )
            return (entry, pickled_content)
        else:
            return None

    @staticmethod
    def _write_to_local_cache(key: str, content: bytes) -> None:
        # [FUNCTION DOC] HELPER: Atomically writes content to a file named by its
        # sha256 inside the per-key directory, creating the directory if needed.
        # Side effect: file I/O and info logging.
        """Write an entry to the local cache."""
        subdir = AOTAutogradCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized entry to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, sha256_hash(content))
        log.info("Writing AOTAutograd cache entry to %s", path)
        write_atomic(path, content)

    @staticmethod
    def _find_unpicklable_field(
        entry: GenericAOTAutogradResult[Any, Any],
    ) -> str | None:
        # [FUNCTION DOC] HELPER: Diagnostic. Tries to pickle each dataclass field (or
        # instance attribute) of entry separately and returns the name of the first
        # one that fails, or None if all succeed.
        """Find which field of entry is causing pickle to fail."""
        # pyrefly: ignore [implicit-any]
        fields = []
        if hasattr(entry, "__dataclass_fields__"):
            fields = list(entry.__dataclass_fields__.keys())
        elif hasattr(entry, "__dict__"):
            fields = list(entry.__dict__.keys())

        for name in fields:
            try:
                pickle.dumps(getattr(entry, name))
            except Exception:
                return name
        return None

    @staticmethod
    def _pickle_entry(
        entry: GenericAOTAutogradResult[Any, Any], remote: bool
    ) -> bytes | None:
        # [FUNCTION DOC] HELPER: Pickles entry with dispatch modes disabled and returns
        # the bytes. On pickling, type or attribute errors, logs a warning, emits a
        # structured-trace artifact naming the unpicklable field, reports a remote
        # bypass if remote, and returns None (re-raises in strict mode).
        """Pickle entry, returning None on failure."""
        try:
            with torch.utils._python_dispatch._disable_current_modes():
                return pickle.dumps(entry)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            bad_field = AOTAutogradCache._find_unpicklable_field(entry)
            error_str = str(e)
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aotautograd_cache_pickle_failure",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps({"error": error_str, "field": bad_field}),
            )
            if remote:
                log_cache_bypass(
                    "bypass_aot_autograd", "Unable to serialize: " + str(e)
                )
            if config.strict_autograd_cache:
                raise
            return None

    @staticmethod
    def _handle_save_error(e: Exception, remote: bool, is_bypass: bool) -> None:
        # [FUNCTION DOC] HELPER: Logs a save failure. A bypass increments the bypass
        # counter; any other error warns about serialization. Reports to the remote
        # bypass log when remote, and re-raises e in strict mode.
        """Handle exceptions during save, re-raising if strict mode is enabled."""
        if is_bypass:
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)
            bypass_reason = str(e)
        else:
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            bypass_reason = "Unable to serialize: " + str(e)
        if remote:
            log_cache_bypass("bypass_aot_autograd", bypass_reason)
        if config.strict_autograd_cache:
            raise e

    @staticmethod
    def save(key: str, entry: GenericAOTAutogradResult[Any, Any], remote: bool) -> None:
        # [FUNCTION DOC] Saves one entry under key: runs the entry's pre-save hook,
        # pickles it, records the cache artifact, in bundling mode hands a bundled
        # precompile artifact to the precompile context (and clears the backend id on
        # the sanitized config), writes to the local cache and, if remote, uploads
        # base64 content plus time taken to the remote cache. Failures are routed to
        # the save error handler and nothing is raised outside strict mode. Returns None.
        """Save a single entry into the cache."""
        content: bytes | None = None
        try:
            entry.pre_save()
            content = AOTAutogradCache._pickle_entry(entry, remote)
            if content is None:
                return None
            CacheArtifactRecorder(AOTAutogradCacheArtifact.type(), key).record(content)
            if (
                should_bundle_autograd_cache()
                and entry.sanitized_aot_config.precompile_backend_id is not None
            ):
                precompile_key = entry.sanitized_aot_config.precompile_backend_id
                artifact = BundledAOTAutogradCacheArtifact(precompile_key, entry)
                entry.sanitized_aot_config.precompile_backend_id = None
                PrecompileContext.record_artifact(artifact)
            AOTAutogradCache._write_to_local_cache(key, content)
            counters["aot_autograd"]["autograd_cache_saved"] += 1
            cache_stats.put("LocalAOTAutogradCache")
        except BypassAOTAutogradCache as e:
            AOTAutogradCache._handle_save_error(e, remote, is_bypass=True)
            return None
        except Exception as e:
            AOTAutogradCache._handle_save_error(e, remote, is_bypass=False)
            return None

        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()
            if remote_cache is not None:
                time_taken_ms = int(
                    (entry.forward_time_taken_ns + entry.backward_time_taken_ns) // 1e6
                )
                cache_data: JsonDataTy = {
                    "data": base64.b64encode(content).decode("ascii"),
                    "time_taken_ms": time_taken_ms,
                }
                remote_cache.put(key, cache_data)

    @staticmethod
    @functools.cache
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        # [FUNCTION DOC] Creates (memoized for the process) the remote cache client
        # with the fixed id "autograd-experimental"; returns None when no remote
        # backend is available.
        """
        Attempts to load the remote cache, returns None on error.
        """
        cache_id = "autograd-experimental"
        return create_cache(
            cache_id,
            config.is_fbcode(),
            "FbRemoteAOTAutogradCache",
            "RemoteAOTAutogradCache",
        )

    @staticmethod
    def make_entry(
        compiled_fw_func: OutputCode,
        compiled_bw_func: OutputCode | None,
        aot_joint_graph_str: str | None,
        aot_forward_graph_str: str | None,
        aot_backward_graph_str: str | None,
        runtime_metadata: ViewAndMutationMeta,
        dispatch_wrappers: list[CompilerWrapper],
        maybe_subclass_meta: SubclassMeta | None,
        num_fw_outs_saved_for_bw: int | None,
        indices_of_inps_to_detach: list[int],
        forward_time_taken_ns: int,
        backward_time_taken_ns: int,
        sanitized_aot_config: CacheableAOTConfig,
        guards_expr: str | None,
        backward_state_indices: list[int] | None,
        num_symints_saved_for_bw: int | None,
        serialized_bw_module: SerializedGraphModule | None,
        min_cut_info_str: str | None,
    ) -> GenericAOTAutogradResult[Any, Any]:
        # [FUNCTION DOC] Packages compile results into a cache entry. In bundling mode,
        # unwraps the compiled forward/backward back to their OutputCode objects and
        # returns a BundledAOTAutogradResult embedding them. Otherwise requires the
        # forward (and backward) functions to carry an FX graph cache key and returns
        # an AOTAutogradResult referencing them by key and guard expression. Other
        # arguments (metadata, wrappers, graph strings, timings, sanitized config,
        # guards, serialized backward module) are stored as-is. Raises AssertionError
        # for missing keys, or missing backward state indices or saved symint count
        # when a backward exists.
        if should_bundle_autograd_cache():
            # Helper function to unwrap all the wrappers we added during aotdispatch
            # They get reapplied on cache load
            def unwrap_output_code(obj: object) -> OutputCode:
                # [FUNCTION DOC] HELPER: Follows the chain of wrapped-function links
                # to the innermost object and returns it; raises AssertionError if it
                # is not an OutputCode.
                while hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                if not isinstance(obj, OutputCode):
                    raise AssertionError(f"expected OutputCode, got {type(obj)}")
                return obj

            compiled_fw_graph = unwrap_output_code(compiled_fw_func)
            bundled_compiled_forward = BundledCompiledForward(compiled_fw_graph)
            bundled_compiled_backward = None
            if compiled_bw_func is not None:
                if backward_state_indices is None:
                    raise AssertionError("backward_state_indices must not be None")
                if num_symints_saved_for_bw is None:
                    raise AssertionError("num_symints_saved_for_bw must not be None")
                compiled_bw_graph = unwrap_output_code(compiled_bw_func)
                bundled_compiled_backward = BundledCompiledBackward(
                    compiled_bw_graph, backward_state_indices, num_symints_saved_for_bw
                )

            return BundledAOTAutogradResult(
                compiled_fw=bundled_compiled_forward,
                compiled_bw=bundled_compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                min_cut_info_str=min_cut_info_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )

        else:
            fw_key = getattr(compiled_fw_func, "_fx_graph_cache_key", None)
            fw_debug_lines = getattr(
                compiled_fw_func, "_fx_graph_cache_debug_lines", []
            )

            if fw_key is None:
                raise AssertionError("fw_key must not be None")
            compiled_forward = CompiledForward(
                fx_graph_cache_info=(fw_key, fw_debug_lines),
                fx_graph_guard_expr=getattr(compiled_fw_func, "guards_expr", None),
            )
            compiled_backward = None
            if compiled_bw_func is not None:
                bw_key = getattr(compiled_bw_func, "_fx_graph_cache_key", None)
                bw_debug_lines = getattr(
                    compiled_bw_func, "_fx_graph_cache_debug_lines", []
                )
                if bw_key is None:
                    raise AssertionError("bw_key must not be None")
                if backward_state_indices is None:
                    raise AssertionError("backward_state_indices must not be None")
                if num_symints_saved_for_bw is None:
                    raise AssertionError("num_symints_saved_for_bw must not be None")
                compiled_backward = CompiledBackward(
                    fx_graph_cache_info=(bw_key, bw_debug_lines),
                    fx_graph_guard_expr=getattr(compiled_bw_func, "guards_expr", None),
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw_=num_symints_saved_for_bw,
                )

            return AOTAutogradResult(
                compiled_fw=compiled_forward,
                compiled_bw=compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                min_cut_info_str=min_cut_info_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )
