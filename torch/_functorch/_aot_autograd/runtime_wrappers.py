# [MODULE DOC] Runtime wrappers for AOTAutograd: the code that runs around a compiled graph at call time.
# Contains the epilogue/prologue logic (input mutation replay, output alias regeneration, metadata
# propagation), code-generated runtime wrapper functions built through PySourceBuilder, and
# CompilerWrapper/InductorWrapper subclasses for functionalized RNG, fakified outputs, tensor
# subclasses, effect tokens, input deduplication/synthetic bases, and the autograd.Function
# (forward/backward) machinery for compiled training graphs.
# Import-time side effects: creates the aot_graphs artifact logger and rebinds the module-level name
# zip to the strict_zip helper (raises on length mismatch).
"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""

import collections
import contextlib
import copy
import functools
import itertools
import pprint
import typing
import warnings
import weakref
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch
import torch.fx as fx
from torch import Tensor
from torch._custom_class_base import CustomClassBase
from torch._dynamo import config as dynamo_config
from torch._dynamo.callback import callback_handler, CallbackTrigger
from torch._dynamo.graph_bytecode_inputs import index_to_external_object_weakref
from torch._dynamo.utils import CompileEventLogger, dynamo_timed, get_metrics_context
from torch._guards import (
    compile_context,
    CompileContext,
    detect_fake_mode,
    DuplicateInputs,
    tracing,
    TracingContext,
)
from torch._library.opaque_object import is_custom_class
from torch._library.utils import is_builtin
from torch._logging import getArtifactLogger
from torch._ops import OpOverload
from torch._prims_common import CUDARngStateHelper
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import HANDLED_TYPES
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten

from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .descriptors import (
    AOTInput,
    AOTOutput,
    DummyAOTInput,
    MetadataMutationAOTOutput,
    SyntheticBaseAOTInput,
    ViewBaseAOTInput,
)
from .functional_utils import gen_alias_from_base
from .graph_capture_wrappers import aot_dispatch_subclass
from .input_output_analysis import (
    compute_overlapping_inputs,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .schemas import (
    AOTConfig,
    CompilerWrapper,
    FxValue,
    InductorWrapper,
    InputAliasInfo,
    MemoryFormatMeta,
    MutationType,
    OpaqueMeta,
    OutputAliasInfo,
    OutputType,
    PlainTensorMeta,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    TraceFn,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    requires_subclass_dispatch,
    runtime_unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
)
from .utils import (
    call_and_expect_output_descs,
    call_func_at_runtime_with_args,
    make_boxed_func,
    partial_flatten_asdict,
    simple_wraps,
    strict_zip,
    without_output_descs,
)


if typing.TYPE_CHECKING:
    from .codegen import PySourceBuilder


def _snapshot_external_objects(ctx: Any) -> None:
    # [FUNCTION DOC] HELPER: Records on the given context object an attribute holding a mapping from
    # external-object index to the live object, for every entry of the global external object weakref
    # registry whose referent is still alive. Mutates ctx; used so backward can restore these objects.
    """Snapshot the external object registry onto ctx for backward restore."""
    ctx._external_objects = {
        k: ref()
        for k, ref in index_to_external_object_weakref.items()
        if ref() is not None
    }


def _unwrap_tensor_subclasses_no_symints(
    args: list[Any],
) -> list[Any]:
    # [FUNCTION DOC] HELPER: Unwraps tensor subclasses in the argument list into their inner dense tensors
    # without appending symbolic ints. Returns the flattened list.
    return runtime_unwrap_tensor_subclasses(args, append_symints=False)  # type: ignore[arg-type]


zip = strict_zip

aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _unwrap_no_symints(args: list[Any]) -> list[Any]:
    # [FUNCTION DOC] HELPER: Same as the other no-symints unwrap helper: flattens tensor subclass arguments
    # into dense inner tensors, without appending symints. Returns the new list.
    return runtime_unwrap_tensor_subclasses(args, append_symints=False)


def _describe_arg_for_logging(arg: object) -> str:
    # [FUNCTION DOC] HELPER: Builds a short human readable description string of one runtime argument for
    # debug logs. DTensors report shape, dtype, device, mesh shape and placements; plain tensors report
    # shape, dtype, device; opaque custom-class objects report their type name; anything else reports type
    # name and value. Tolerates builds where the distributed tensor class is unavailable.
    from torch._library import opaque_object

    try:
        is_dtensor = isinstance(arg, torch.distributed.tensor.DTensor)
    except AttributeError:
        is_dtensor = False

    if is_dtensor:
        arg = typing.cast(torch.distributed.tensor.DTensor, arg)
        mesh = arg.device_mesh
        return (
            f"DTensor(shape={arg.shape}, dtype={arg.dtype}, "
            f"device={arg.device}, mesh_shape={mesh.shape}, "
            f"placements={arg.placements})"
        )
    elif isinstance(arg, torch.Tensor):
        return f"Tensor(shape={arg.shape}, dtype={arg.dtype}, device={arg.device})"
    elif opaque_object.is_custom_class(type(arg)):
        return f"Opaque: {type(arg).__name__}"
    else:
        return f"{type(arg).__name__}: {arg}"


def _log_input_metadata(runtime_metadata: ViewAndMutationMeta) -> None:
    # [FUNCTION DOC] HELPER: Debug-logs the count and each entry of the expected subclass input metadata
    # from the runtime metadata, to the aot_graphs logger. No return value.
    aot_graphs_log.debug(
        "Expected input metadata (count=%s):", len(runtime_metadata.subclass_inp_meta)
    )
    for i, meta in enumerate(runtime_metadata.subclass_inp_meta):
        aot_graphs_log.debug("  [%s] %s", i, meta)


def _log_args_list(args: Sequence[object], label: str) -> None:
    # [FUNCTION DOC] HELPER: Debug-logs a labeled, indexed listing of the given arguments (count, then one
    # described line per argument) to the aot_graphs logger.
    aot_graphs_log.debug("%s (count=%s):", label, len(args))
    for i, arg in enumerate(args):
        aot_graphs_log.debug("  [%s] %s", i, _describe_arg_for_logging(arg))


def _log_args_maybe_list(arg: object, label: str) -> None:
    # [FUNCTION DOC] HELPER: Debug-logs a labeled argument. Lists and tuples are logged element by element;
    # any other value is logged as a single described item.
    if isinstance(arg, (list, tuple)):
        _log_args_list(arg, label)
    else:
        aot_graphs_log.debug("%s: %s", label, _describe_arg_for_logging(arg))


# The wrapper created by this function handles all of the runtime aliasing and mutation "epilogue" logic
# that needs to run after the compiled function.
#
# This function accepts a trace_joint flag, indicating whether or not we're generating the runtime
# epilogue for a forward-only inference graph, or for an autograd.Function.apply function.
# This is because there are some minor differences in how we treat these cases at runtime:
# - resize_() is currently handled in the inference case, but not fully handled in the autograd case.
# - the autograd cases inserts TensorAlias wrapper objects for outputs that alias inputs
@dataclass
class RuntimeWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper that adds the runtime aliasing/mutation epilogue around a compiled
    # function. State: the indices of inputs to detach, whether the graph is a joint (autograd) trace, and
    # whether to disable autocast. The only behavior is post_compile, which delegates to
    # _create_runtime_wrapper. See the comment above the class for inference vs autograd differences.
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Wraps the compiled function with the runtime wrapper built by _create_runtime_wrapper,
        # using this wrapper's detach indices, joint flag and amp flag plus the config setting for keeping
        # inference input mutations. Args: compiled_fn, aot_config, runtime_metadata. Returns the new callable.
        return _create_runtime_wrapper(
            compiled_fn,
            runtime_metadata=runtime_metadata,
            indices_of_inps_to_detach=self.indices_of_inps_to_detach,
            trace_joint=self.trace_joint,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=self.disable_amp,
        )


class NoopAliasHandler:
    # [CLASS DOC] Output handler for outputs that need no alias regeneration (non-alias, unsafe view alias
    # and custom function view outputs). Stateless; calling it returns the output unchanged.
    def __init__(
        self,
        info: OutputAliasInfo,
        runtime_metadata: ViewAndMutationMeta,
        trace_joint: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Accepts and ignores the output info, runtime metadata and joint flag so that
        # all handler types share one constructor signature.
        pass

    def __call__(
        self, orig_inputs: dict[int, Tensor], fw_outs: list[Any], out: Any
    ) -> Any:
        # [FUNCTION DOC] Returns the compiled output unchanged. The original inputs and forward outputs are
        # ignored.
        return out


def _unwrap_tensoralias(x: TensorAlias) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Extracts the wrapped tensor from a TensorAlias wrapper object. Raises an
    # assertion error if the argument is not a TensorAlias.
    if not isinstance(x, TensorAlias):
        raise AssertionError(f"expected TensorAlias, got {type(x)}")
    return x.alias


def _identity(x: Any) -> Any:
    # [FUNCTION DOC] HELPER: Returns its argument unchanged (used as a no-op unwrap function).
    return x


class AliasOfInputHandler:
    # [CLASS DOC] Output handler for outputs that are views of a user input. Holds the index of the base
    # input, an unwrap function (TensorAlias unwrap for joint graphs, identity otherwise), the
    # requires-grad flag, the recorded view replay sequence, and the config flag for replaying views.
    # Calling it regenerates the alias from the original input.
    def __init__(
        self,
        info: OutputAliasInfo,
        runtime_metadata: ViewAndMutationMeta,
        trace_joint: bool,
    ) -> None:
        # [FUNCTION DOC] Reads the base input index, requires-grad flag and view metadata sequence from the
        # output alias info and picks the unwrap function based on trace_joint. Raises an assertion error if
        # the info has no base index. Also captures the view-replay config setting at construction time.
        if info.base_idx is None:
            raise AssertionError("expected info.base_idx to be set")
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.view_meta_sequence = info.view_meta_sequence
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(
        self, orig_inputs: dict[int, Tensor], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
        # [FUNCTION DOC] Looks up the original input at the base index and builds a fresh alias of it shaped
        # like the compiled output (after unwrapping any TensorAlias), replaying recorded views when enabled.
        # Returns the new tensor.
        aliased_base_tensor = orig_inputs[self.base_idx]
        return gen_alias_from_base(
            aliased_base_tensor,
            self.unwrap_out(out),
            self.requires_grad,
            self.view_meta_sequence,
            replay_views=self.replay_views,
        )


class IsInputHandler:
    # [CLASS DOC] Output handler for outputs that are exactly a user input object. Holds the base input
    # index; calling it returns that original input.
    def __init__(
        self,
        info: OutputAliasInfo,
        runtime_metadata: ViewAndMutationMeta,
        trace_joint: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the base input index from the alias info. Raises an assertion error if the info
        # has no base index. Other arguments are unused.
        if info.base_idx is None:
            raise AssertionError("expected info.base_idx to be set")
        self.base_idx = info.base_idx

    def __call__(
        self, orig_inputs: dict[int, Tensor], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
        # [FUNCTION DOC] Returns the original input tensor at the stored base index, ignoring the compiled
        # output.
        aliased_base_tensor = orig_inputs[self.base_idx]
        return aliased_base_tensor


class AliasOfIntermediateHandler:
    # [CLASS DOC] Output handler for outputs that are views of an intermediate tensor produced by the
    # graph. Holds the index (into the forward outputs) of the base, an unwrap function for the base (only
    # TensorAlias unwrap when the base is itself a user output aliased to an input), an unwrap function for
    # the output, requires-grad flag, view metadata sequence and the view-replay config flag. Calling it
    # regenerates the alias from the base found among the forward outputs.
    def __init__(
        self,
        info: OutputAliasInfo,
        runtime_metadata: ViewAndMutationMeta,
        trace_joint: bool,
    ) -> None:
        # [FUNCTION DOC] Computes the base index. For intermediate-base output types the base lives after the
        # user outputs, so the index is offset by the number of user outputs; otherwise it is used directly and
        # the base is unwrapped if that index is in the aliased output indices. Raises an assertion error if
        # the info has no base index. Captures the other alias fields and the view-replay config.
        if info.base_idx is None:
            raise AssertionError("expected info.base_idx to be set")
        self._unwrap_aliased_base_tensor = _identity
        if info.output_type in (
            OutputType.alias_of_intermediate,
            OutputType.alias_of_intermediate_save_as_output,
        ):
            num_user_outputs = len(runtime_metadata.output_info)
            self.base_idx = info.base_idx + num_user_outputs
        else:
            self.base_idx = info.base_idx
            if self.base_idx in runtime_metadata.aliased_out_indices:
                self._unwrap_aliased_base_tensor = _unwrap_tensoralias

        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.view_meta_sequence = info.view_meta_sequence
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(
        self, orig_inputs: dict[int, Tensor], fw_outs: list[Any], out: Any
    ) -> torch.Tensor:
        # [FUNCTION DOC] Takes the base tensor from the forward outputs at the stored index (unwrapped if
        # needed) and builds an alias of it matching the compiled output, optionally replaying views. Returns
        # the new tensor.
        aliased_base_tensor = fw_outs[self.base_idx]
        return gen_alias_from_base(
            self._unwrap_aliased_base_tensor(aliased_base_tensor),
            self.unwrap_out(out),
            self.requires_grad,
            self.view_meta_sequence,
            replay_views=self.replay_views,
        )


_HANDLER_MAP: dict[
    OutputType,
    type[
        NoopAliasHandler
        | AliasOfInputHandler
        | IsInputHandler
        | AliasOfIntermediateHandler
    ],
] = {
    OutputType.non_alias: NoopAliasHandler,
    OutputType.unsafe_view_alias: NoopAliasHandler,
    OutputType.custom_function_view: NoopAliasHandler,
    OutputType.alias_of_input: AliasOfInputHandler,
    OutputType.is_input: IsInputHandler,
    OutputType.alias_of_intermediate: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_save_as_output: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_base_is_user_output: AliasOfIntermediateHandler,
}


def make_output_handler(
    info: OutputAliasInfo, runtime_metadata: ViewAndMutationMeta, trace_joint: bool
) -> (
    NoopAliasHandler | AliasOfInputHandler | IsInputHandler | AliasOfIntermediateHandler
):
    # [FUNCTION DOC] Factory that selects the handler class for the output type of the given alias info via
    # a fixed table and constructs it with the info, runtime metadata and joint flag. Raises a key error
    # for an output type missing from the table.
    handler_type = _HANDLER_MAP[info.output_type]
    return handler_type(info, runtime_metadata, trace_joint)


# _dynamo_propagated_dynamic_indices: A guardless attribute for cross-graph-break
# dynamism propagation. When AOTAutograd traces a graph and discovers that output
# dimensions are symbolic (dynamic), it stamps this attribute on the output tensors
# at runtime. This tells Dynamo to treat those dims as weakly dynamic when the tensor
# appears as input to a subsequent graph (e.g., after a graph break), avoiding
# unnecessary specialization and recompilation.
#
# Unlike _dynamo_weak_dynamic_indices (which is user-facing, set via maybe_mark_dynamic(),
# and guarded on), this attribute is compiler-internal and NOT guarded on. This avoids
# the problem where the compiler mutating an input tensor's attributes (through an
# input-aliased output) would cause a spurious guard failure and recompilation.
#
# Note: this is best-effort. Eager code does not propagate
# _dynamo_propagated_dynamic_indices, so there is no guarantee that the next graph
# will have proper dynamism information. It only helps when the output of one compiled
# graph feeds directly into the next.
def mark_dynamo_propagated_dynamic_indices(t: torch.Tensor, dims: set[int]) -> None:
    # [FUNCTION DOC] Records on a tensor which dimensions are dynamic, for Dynamo to treat as weakly dynamic
    # when the tensor feeds a later graph. If the tensor already has the guardless attribute, the dims are
    # merged into it by set union; otherwise a copy of the set is stored. Mutates the tensor attribute.
    if hasattr(t, "_dynamo_propagated_dynamic_indices"):
        # pyrefly: ignore [missing-attribute]
        t._dynamo_propagated_dynamic_indices |= dims
    else:
        t._dynamo_propagated_dynamic_indices = dims.copy()  # type: ignore[attr-defined]


def _should_disable_saved_tensors_hooks() -> bool:
    # [FUNCTION DOC] HELPER: Returns true when saved-tensors hooks should be disabled inside the compiled
    # region: false when inside a compiled autograd region; otherwise true only if the current top saved
    # tensors hooks are inlineable.
    # Compiled autograd is not supported yet, to be added in future.
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        return False

    get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
    are_inline_hooks = (
        torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
    )

    hooks = get_hooks()
    if are_inline_hooks(hooks):
        return True

    return False


def _schema_allows_aliasing(func: Any) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the operator schema permits aliasing, i.e. it is a view op or
    # any return value carries alias annotation info. Used to skip custom op alias checks.
    schema = func._schema
    # View ops have non-write aliases declared in arguments
    if schema._is_view_op():
        return True
    # Handles cases like mkldnn::_convolution_pointwise_.binary
    # where the schema is Tensor(a!) other -> Tensor(a!) Y
    for ret in schema.returns:
        if ret.alias_info is not None:
            return True
    return False


def _check_custom_op_aliasing(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
) -> None:
    # [FUNCTION DOC] HELPER: Runs the native aliasing constraint check on a custom op call (name, args,
    # kwargs, result). If the check raises a runtime error, re-raises it when the config flag demands errors,
    # otherwise emits a user warning noting the behavior is deprecated.
    """
    Check if custom op outputs alias inputs or other outputs.
    If config.error_on_custom_op_aliasing is True, raises RuntimeError.
    Otherwise, emits a warning.
    """
    try:
        torch._library.utils._c_check_aliasing_constraint(
            name,
            args,
            kwargs,
            result,
        )
    except RuntimeError as e:
        if config.error_on_custom_op_aliasing:
            raise
        else:
            msg = f"{e} This is deprecated and will become an error in a future version of PyTorch."
            warnings.warn(msg, UserWarning, stacklevel=3)


@functools.lru_cache(None)
def _is_fsdp_all_gather_copy_in(func: Any) -> bool:
    # [FUNCTION DOC] HELPER: Cached predicate: true if the op is the fsdp all_gather_copy_in default overload,
    # detected by comparing namespace and name strings so that the fsdp ops need not be registered.
    """
    Check if func is torch.ops.fsdp.all_gather_copy_in.default by comparing
    namespace and name strings. This avoids accessing torch.ops.fsdp directly,
    which would fail on platforms where FSDP ops aren't registered (e.g., macOS
    builds with USE_DISTRIBUTED=0).
    """
    return (
        hasattr(func, "namespace")
        and func.namespace == "fsdp"
        and hasattr(func, "__name__")
        and func.__name__ == "all_gather_copy_in.default"
    )


class _AnalyzeCustomOpInputOutputMode(TorchDispatchMode):
    # [CLASS DOC] HELPER: Torch dispatch mode that executes each op and, for non-builtin custom ops that
    # claim to be functional, checks that outputs do not alias inputs or each other (error or warning per
    # config). Supports higher order operators. Skips some namespaces and an fsdp op with a bad schema.
    """
    Checks if inp/out of custom ops alias each other.
    If config.error_on_custom_op_aliasing is True, violations raise errors.
    Otherwise, violations emit warnings.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the dispatch mode and enables support for higher order operators.
        super().__init__()
        self.supports_higher_order_operators = True

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # [FUNCTION DOC] Dispatch handler. Defers (returns NotImplemented) if any tensor argument, after taking
        # the data of Parameters, has a type not in the handled proxy types. Otherwise runs the op, and if it is
        # a non-builtin custom op not in skipped namespaces and whose schema forbids aliasing, checks aliasing
        # of args, kwargs and result. Returns the op result.
        if not kwargs:
            kwargs = {}

        flat_tensor_args = filter(
            lambda x: isinstance(x, torch.Tensor), tree_flatten((args, kwargs))[0]
        )

        # Defer this to subclass torchdispatch modes (probably shouldn't have fake tensor here tho)
        # For Parameters, we need to check the underlying tensor type, not the Parameter itself
        for tensor in flat_tensor_args:
            underlying_tensor = tensor
            if isinstance(tensor, torch.nn.Parameter):
                underlying_tensor = tensor.data
            if type(underlying_tensor) not in HANDLED_TYPES:
                return NotImplemented

        res = func(*args, **kwargs)
        # Only check aliasing for custom ops (non-aten/prim/prims/_c10d_functional/c10d)
        # that claim to be functional
        # Skip ops whose schema declares aliasing is allowed
        if (
            not isinstance(func, torch._ops.HigherOrderOperator)
            and not is_builtin(func)
            # TODO (https://github.com/pytorch/pytorch/issues/170986)
            and func.namespace not in ("_c10d_functional", "c10d", "onednn")
            # This op is quite important but has wrong schema, so lets skip for now
            and not _is_fsdp_all_gather_copy_in(func)
            and not _schema_allows_aliasing(func)
        ):
            _check_custom_op_aliasing(
                func.name(),
                args,
                kwargs,
                res,
            )
        return res

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # [FUNCTION DOC] HELPER: Class-level hook that tells the compiler internals to ignore this mode; always
        # returns true.
        return True


class _FirstInvocationContext:
    # [CLASS DOC] HELPER: Callable context-manager factory with a single boolean state recording whether the
    # first invocation is still pending. Used so the custom op aliasing analysis only runs on the first
    # call (cold start).
    """
    Context manager that tracks first invocation and conditionally enables _AnalyzeCustomOpInputOutputMode.
    This is useful when we have a custom op where we want to analyze its input
    and output during cold start.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the first invocation as not yet happened.
        self._is_first = True

    def __call__(self) -> AbstractContextManager[Any]:
        # [FUNCTION DOC] Returns the aliasing analysis dispatch mode on the first call if the config enables
        # custom op alias checking and Dynamo is not forcing compile during FX tracing (and flips the state so
        # later calls differ); otherwise returns a null context manager.
        """
        Returns a context manager: _AnalyzeCustomOpInputOutputMode on first invocation, nullcontext thereafter.
        Automatically updates state after first use.
        """
        # NB: Don't run the analyzer when you're forcing compile during FX
        # tracing, as the analyzer doesn't play nicely when it's being
        # make_fx'ed through
        if (
            self._is_first
            and config.check_custom_op_aliasing
            and not torch._dynamo.config.force_compile_during_fx_trace
        ):
            self._is_first = False
            return _AnalyzeCustomOpInputOutputMode()
        return nullcontext()


# Note [RuntimeWrapper codegen specification methods]
# The run() method on _RuntimeCompiledFnInvoker and the capture_orig_inputs(),
# increment_mutation_versions(), and finalize() methods on _RuntimeForwardEpilogue
# are the readable reference implementations for the codegen'd _runtime_wrapper
# generated in _create_runtime_wrapper(). They are not called on the hot path;
# the codegen inlines their logic with all branches resolved at compile time.
#
# WARNING: Any semantic change to the runtime wrapper must be reflected in both
# the reference methods here and the codegen in _create_runtime_wrapper().
@dataclass
class _RuntimeCompiledFnInvoker:
    # [CLASS DOC] Reference (non-hot-path) implementation of invoking the compiled function. State: the
    # compiled function (made boxed if not already), detach indices, joint flag, amp flag and a first
    # invocation context. The hot path uses code generated to match run; see the Note above on keeping them
    # in sync.
    compiled_fn: Callable[..., Any]
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool
    first_invocation_ctx: _FirstInvocationContext = field(
        default_factory=_FirstInvocationContext
    )

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Wraps the compiled function into a boxed-calling-convention function unless it
        # already declares itself boxed.
        if not getattr(self.compiled_fn, "_boxed_call", False):
            self.compiled_fn = make_boxed_func(self.compiled_fn)

    # WARNING: this is a reference implementation; the hot path uses codegen'd
    # code from _create_runtime_wrapper(). Keep both in sync.
    # See Note [RuntimeWrapper codegen specification methods]
    def run(self, args: list[Any], *, on_before_call: Callable[[], None]) -> list[Any]:
        # [FUNCTION DOC] Reference implementation of calling the compiled function. Inside the first-invocation
        # context: for joint graphs, copies the args, detaches the listed tensor inputs, forces view replay on
        # (restored afterward), enables grad, calls the hook, then calls the compiled function with arguments
        # stolen. For inference graphs, disables grad if enabled (restored afterward), calls the hook and then
        # the function. Returns the raw output list. Side effects: toggles grad mode and view replay thread state.
        with self.first_invocation_ctx():
            if self.trace_joint:
                args_ = list(args)
                # See Note [Detaching inputs that never need gradients]
                for idx in self.indices_of_inps_to_detach:
                    if isinstance(args_[idx], torch.Tensor):
                        args_[idx] = args_[idx].detach()

                # It's possible to have trace_joint inside user specified with no_grad() region,
                # if there is a nested with enable_grad(), that forces some outputs to require gradients.
                # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
                prev_view_replay_enabled = torch._C._is_view_replay_enabled()
                try:
                    if not prev_view_replay_enabled:
                        torch._C._set_view_replay_enabled(True)
                    with torch.enable_grad():
                        on_before_call()
                        return call_func_at_runtime_with_args(
                            self.compiled_fn,
                            args_,
                            disable_amp=self.disable_amp,
                            steal_args=True,
                        )
                finally:
                    if torch._C._is_view_replay_enabled() != prev_view_replay_enabled:
                        torch._C._set_view_replay_enabled(prev_view_replay_enabled)

            # When we have an inference graph, we run with grad disabled.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
            grad_enabled = torch.is_grad_enabled()
            try:
                if grad_enabled:
                    torch._C._set_grad_enabled(False)
                on_before_call()
                return call_func_at_runtime_with_args(
                    self.compiled_fn,
                    args,
                    disable_amp=self.disable_amp,
                    steal_args=True,
                )
            finally:
                if grad_enabled:
                    torch._C._set_grad_enabled(True)


@dataclass
class _RuntimeForwardEpilogue:
    # [CLASS DOC] Reference implementation of the post-call epilogue. State: runtime metadata, joint flag,
    # keep-input-mutations flag, the tuple of argument indices whose originals must be captured (mutated
    # inputs plus bases of outputs that are or alias inputs), and per-output alias handlers (empty when no
    # output is aliased). Methods capture originals, bump versions, then finalize outputs. Not on the hot path.
    runtime_metadata: ViewAndMutationMeta
    trace_joint: bool
    keep_input_mutations: bool
    epilogue_args_idx: tuple[int, ...] = field(init=False)
    output_handlers: tuple[
        NoopAliasHandler
        | AliasOfInputHandler
        | IsInputHandler
        | AliasOfIntermediateHandler,
        ...,
    ] = field(init=False)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Computes the indices of inputs to capture (mutated runtime input indices plus base
        # indices of outputs of alias-of-input or is-input type) and builds the output handlers when any output
        # is aliased. Raises an assertion error if a base index is not an integer or if effect tokens exist while
        # unlifting of effect tokens is configured.
        epilogue_args_idx = list(self.runtime_metadata.mutated_inp_runtime_indices)
        for info in self.runtime_metadata.output_info:
            if (
                info.output_type == OutputType.alias_of_input
                or info.output_type == OutputType.is_input
            ):
                if not isinstance(info.base_idx, int):
                    raise AssertionError(
                        f"expected info.base_idx to be int, got {type(info.base_idx)}"
                    )
                epilogue_args_idx.append(info.base_idx)
        self.epilogue_args_idx = tuple(epilogue_args_idx)

        if config.unlift_effect_tokens:
            if len(self.runtime_metadata.tokens) != 0:
                raise AssertionError(
                    "expected no tokens when unlift_effect_tokens is True, "
                    f"got {len(self.runtime_metadata.tokens)}"
                )

        if self.runtime_metadata.num_outputs_aliased > 0:
            self.output_handlers = tuple(
                make_output_handler(info, self.runtime_metadata, self.trace_joint)
                for info in self.runtime_metadata.output_info
            )
        else:
            self.output_handlers = ()

    # WARNING: this is a reference implementation; the hot path uses codegen'd
    # code from _create_runtime_wrapper(). Keep both in sync.
    # See Note [RuntimeWrapper codegen specification methods]
    def capture_orig_inputs(self, args: list[Any]) -> dict[int, Tensor]:
        # [FUNCTION DOC] Returns a mapping from each captured argument index to the original argument tensor
        # taken before running the compiled function.
        return {i: typing.cast(Tensor, args[i]) for i in self.epilogue_args_idx}

    # WARNING: this is a reference implementation; the hot path uses codegen'd
    # code from _create_runtime_wrapper(). Keep both in sync.
    # See Note [RuntimeWrapper codegen specification methods]
    def increment_mutation_versions(self, args: list[Any]) -> None:
        # [FUNCTION DOC] When input mutations are kept in the graph, bumps the autograd version counter of the
        # mutated inputs that autograd should see. No-op otherwise. Mutates the version counters.
        if self.keep_input_mutations:
            mutated_args = (
                args[i]
                for i in self.runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
            )
            torch.autograd.graph.increment_version(mutated_args)

    # WARNING: this is a reference implementation; the hot path uses codegen'd
    # code from _create_runtime_wrapper(). Keep both in sync.
    # See Note [RuntimeWrapper codegen specification methods]
    def finalize(self, orig_inputs: dict[int, Tensor], all_outs: list[Any]) -> Any:
        # [FUNCTION DOC] Post-processing of compiled outputs: validates the output count, splits off updated
        # inputs, applies input mutations to the originals, regenerates aliased outputs, stamps dynamic dim info
        # on outputs that have it, and sets the grad-enabled state if the graph changed it. Returns the list of
        # user-visible outputs.
        self._validate_compiled_output_arity(all_outs)
        updated_inputs, fw_outs = self._split_mutated_inputs(all_outs)
        if updated_inputs is not None:
            self._apply_input_mutations(orig_inputs, updated_inputs)

        ret_outs = self._replay_output_aliases(orig_inputs, fw_outs)
        if self.runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, self.runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                mark_dynamo_propagated_dynamic_indices(t, o.dynamic_dims)
        if self.runtime_metadata.grad_enabled_mutation is not None:
            torch._C._set_grad_enabled(self.runtime_metadata.grad_enabled_mutation)
        return ret_outs

    def _validate_compiled_output_arity(self, all_outs: list[Any]) -> None:
        # [FUNCTION DOC] HELPER: Raises an assertion error unless the number of compiled outputs equals the
        # mutated-input count plus user output count plus intermediate base count.
        expected_outs = (
            self.runtime_metadata.num_mutated_inp_runtime_indices
            + self.runtime_metadata.num_outputs
            + self.runtime_metadata.num_intermediate_bases
        )
        if len(all_outs) != expected_outs:
            raise AssertionError(
                f"expected {expected_outs} outputs, got {len(all_outs)}"
            )

    def _split_mutated_inputs(
        self, all_outs: list[Any]
    ) -> tuple[list[Any] | None, list[Any]]:
        # [FUNCTION DOC] HELPER: Splits the compiled outputs into the leading updated mutated inputs and the
        # remaining forward outputs. Returns none and the full list when there are no mutated runtime inputs.
        num_mutated_runtime_inps = self.runtime_metadata.num_mutated_inp_runtime_indices
        if num_mutated_runtime_inps == 0:
            return None, all_outs
        return (
            all_outs[:num_mutated_runtime_inps],
            all_outs[num_mutated_runtime_inps:],
        )

    def _apply_input_mutations(
        self, orig_inputs: dict[int, Tensor], updated_inputs: list[Any]
    ) -> None:
        # [FUNCTION DOC] HELPER: Replays input mutations from the compiled graph onto the original inputs. Per
        # mutated input: skips ones without data/metadata mutation; for storage-metadata mutations performs a
        # no-grad set or shallow data copy; for metadata-only mutations re-strides the input; for data
        # mutations (plus optional re-stride) copies the updated value in, detaching if the input is a leaf
        # requiring grad. Unwraps TensorAlias for joint graphs. Raises a runtime error if the mutated input has
        # a user stream index recorded, and an assertion error for inconsistent flags.
        for i, inpt_idx in enumerate(self.runtime_metadata.mutated_inp_runtime_indices):
            meta = self.runtime_metadata.input_info[inpt_idx]
            if not meta.mutates_data and not meta.mutates_metadata:
                continue
            original_inpt = orig_inputs[inpt_idx]
            updated_inpt = updated_inputs[i]
            if meta.mutates_storage_metadata:
                # See Note [set_() Input Mutations in AOTAutograd]
                # mutates_storage_metadata means our input saw a x.set_(y) call.
                # What if x **also** saw a data and/or a metadata mutation?
                # (1) If the [meta]data mutation occurred after the set_(),
                #     then there is no need to copy_() the data.
                #     When we perform x.set_(x_updated), we are guaranteed that
                #     x_updated already has the final version of the data/metadata
                # (2) If a data mutation occurred before the set_().
                #     This case seems very difficult to support.
                #     TODO: discuss on the PR and decide if we want to tr to
                #     either support it, or detect and ban it.
                if self.trace_joint:
                    if not isinstance(updated_inpt, TensorAlias):
                        raise AssertionError(
                            f"expected TensorAlias for updated_inpt, got {type(updated_inpt)}"
                        )
                    updated_inpt = updated_inpt.alias
                with torch.no_grad():
                    if meta.mutation_is_shallow_copy_data:
                        torch.ops.aten.shallow_copy_data_(original_inpt, updated_inpt)
                    else:
                        original_inpt.set_(updated_inpt)
                continue
            if meta.mutates_metadata and not meta.mutates_data:
                if self.trace_joint:
                    if not isinstance(updated_inpt, TensorAlias):
                        raise AssertionError(
                            f"expected TensorAlias for updated_inpt, got {type(updated_inpt)}"
                        )
                    updated_inpt = updated_inpt.alias
                # We need to grab the size/stride/storage_offset from the compiled forward,
                # and use that to mutate the metadata of the input
                original_inpt.as_strided_(
                    updated_inpt.size(),
                    updated_inpt.stride(),
                    updated_inpt.storage_offset(),
                )
            else:
                if meta.mutates_data and meta.mutates_metadata:
                    original_inpt.as_strided_(
                        updated_inpt.size(),
                        updated_inpt.stride(),
                        updated_inpt.storage_offset(),
                    )
                else:
                    if not meta.mutates_data:
                        raise AssertionError("expected meta.mutates_data to be True")
                if meta.is_leaf and original_inpt.requires_grad:
                    # We can hit this situation in this case:
                    #   def f(x):
                    #       x.detach().mul_(2)
                    #       return x + 1
                    # AOTAutograd will see a mutation in the above case, and try to
                    # apply a copy_() here, in the epilogue.
                    # But if x required gradients, and is a leaf, then autograd
                    # will yell at us for trying to mutate it.
                    # However, it's only possible to end up in this scenario (like the above)
                    # if all of the mutations to the leaf input were non-autograd-tracking mutations
                    # (aka mutations under no_grad(), or on detached views).
                    # In that case, we fully want to hide the mutation from autograd, so detaching is ok.
                    original_inpt.detach().copy_(updated_inpt)
                else:
                    # Check if we have stream index information for this mutated input
                    if (
                        self.runtime_metadata.mutated_inp_stream_indices is not None
                        and i < len(self.runtime_metadata.mutated_inp_stream_indices)
                        and self.runtime_metadata.mutated_inp_stream_indices[i]
                        is not None
                    ):
                        raise RuntimeError(
                            "Mutations on inputs with user-specified streams are not yet supported. "
                            "See: https://github.com/pytorch/pytorch/issues/172522"
                        )
                    original_inpt.copy_(updated_inpt)

    def _replay_output_aliases(
        self, orig_inputs: dict[int, Tensor], fw_outs: list[Any]
    ) -> Any:
        # [FUNCTION DOC] HELPER: Regenerates the aliased outputs. Returns the forward outputs unchanged when no
        # output is aliased. Otherwise checks the output count (user outputs plus intermediate bases) and applies
        # each output's handler, returning a list with one entry per user output (intermediate bases dropped).
        # Raises an assertion error on count mismatch.
        if self.runtime_metadata.num_outputs_aliased == 0:
            return fw_outs

        # The compiled forward also returned intermediate bases. We don't want to return them to the user.
        expect_num_outputs = (
            len(self.output_handlers) + self.runtime_metadata.num_intermediate_bases
        )
        if len(fw_outs) != expect_num_outputs:
            raise AssertionError(
                f"expected {expect_num_outputs} fw_outs, got {len(fw_outs)}"
            )
        return [
            handler(orig_inputs, fw_outs, out)
            for out, handler in zip(fw_outs, self.output_handlers)
        ]


def _codegen_capture_orig_inputs(
    buf: "PySourceBuilder",
    epilogue_args_idx: tuple[int, ...],
) -> None:
    # [FUNCTION DOC] HELPER: Emits generated source that builds a mapping from captured argument index to
    # the original argument (or an empty mapping if there are none), into the builder at indent level one.
    if epilogue_args_idx:
        idx_str = ", ".join(f"{i}: args[{i}]" for i in epilogue_args_idx)
        buf.emit(f"orig_inputs = {{{idx_str}}}", indent=1)
    else:
        buf.emit("orig_inputs = {}", indent=1)


def _codegen_increment_mutation_versions(
    buf: "PySourceBuilder",
    keep_input_mutations: bool,
    runtime_metadata: ViewAndMutationMeta,
) -> None:
    # [FUNCTION DOC] HELPER: When input mutations are kept and there are mutated inputs seen by autograd,
    # registers the version-increment function as a global of the generated code and emits a call bumping
    # the version of those arguments. Otherwise emits nothing.
    if (
        keep_input_mutations
        and runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
    ):
        buf.add_global("_increment_version_", torch.autograd.graph.increment_version)
        mut_idx = tuple(runtime_metadata.mutated_graph_handled_indices_seen_by_autograd)
        gen_expr = ", ".join(f"args[{i}]" for i in mut_idx)
        buf.emit(f"_increment_version_(({gen_expr},))", indent=1)


def _codegen_normalize_as_list(
    buf: "PySourceBuilder", var_name: str, *, indent_level: int
) -> None:
    # [FUNCTION DOC] HELPER: Emits generated code that normalizes the named variable to a list: tuples are
    # converted to lists and non-list values are wrapped in a one element list. Uses the given indent level.
    buf.emit(f"if isinstance({var_name}, tuple):", indent=indent_level)
    buf.emit(f"{var_name} = list({var_name})", indent=indent_level + 1)
    buf.emit(f"elif not isinstance({var_name}, list):", indent=indent_level)
    buf.emit(f"{var_name} = [{var_name}]", indent=indent_level + 1)


def _codegen_compiled_fn_invocation(
    buf: "PySourceBuilder",
    trace_joint: bool,
    indices_of_inps_to_detach: list[int],
    disable_amp: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Emits generated code for invoking the compiled function inside the first
    # invocation context. Joint graphs: copy args, detach listed inputs, force view replay on (restored in a
    # finally block), enable grad, call the pre-call hook, call compiled function (under autocast disabled
    # if requested) and normalize outputs to a list. Inference graphs: disable grad (restored in finally),
    # same call. Finally deletes the args variable. Branches are resolved at codegen time.
    buf.emit("with _first_ctx_():", indent=1)
    # trace_joint is known at codegen time. Only the joint/training path needs
    # forced view replay; inference wrappers should not touch this TLS state.
    if trace_joint:
        buf.emit("args_ = list(args)", indent=2)
        for idx in indices_of_inps_to_detach:
            buf.emit(
                f"if isinstance(args_[{idx}], torch.Tensor): "
                f"args_[{idx}] = args_[{idx}].detach()",
                indent=2,
            )
        buf.emit(
            "prev_view_replay_enabled = torch._C._is_view_replay_enabled()", indent=2
        )
        buf.emit("try:", indent=2)
        buf.emit("if not prev_view_replay_enabled:", indent=3)
        buf.emit("torch._C._set_view_replay_enabled(True)", indent=4)
        buf.emit("with torch.enable_grad():", indent=3)
        buf.emit("_on_before_call_()", indent=4)
        if disable_amp:
            buf.add_global("_DisableAutocast_", torch._C._DisableAutocast)
            buf.emit("with _DisableAutocast_():", indent=4)
            buf.emit("all_outs = _compiled_fn_(args_)", indent=5)
            _codegen_normalize_as_list(buf, "all_outs", indent_level=5)
        else:
            buf.emit("all_outs = _compiled_fn_(args_)", indent=4)
            _codegen_normalize_as_list(buf, "all_outs", indent_level=4)
        buf.emit("finally:", indent=2)
        buf.emit(
            "if torch._C._is_view_replay_enabled() != prev_view_replay_enabled:",
            indent=3,
        )
        buf.emit(
            "torch._C._set_view_replay_enabled(prev_view_replay_enabled)", indent=4
        )
    else:
        buf.emit("grad_enabled = torch.is_grad_enabled()", indent=2)
        buf.emit("try:", indent=2)
        buf.emit("if grad_enabled: torch._C._set_grad_enabled(False)", indent=3)
        buf.emit("_on_before_call_()", indent=3)
        if disable_amp:
            buf.add_global("_DisableAutocast_", torch._C._DisableAutocast)
            buf.emit("with _DisableAutocast_():", indent=3)
            buf.emit("all_outs = _compiled_fn_(args)", indent=4)
            _codegen_normalize_as_list(buf, "all_outs", indent_level=4)
        else:
            buf.emit("all_outs = _compiled_fn_(args)", indent=3)
            _codegen_normalize_as_list(buf, "all_outs", indent_level=3)
        buf.emit("finally:", indent=2)
        buf.emit("if grad_enabled: torch._C._set_grad_enabled(True)", indent=3)
    buf.emit("del args", indent=1)


# signatures mirror the _RuntimeForwardEpilogue reference methods
# (_apply_input_mutations / _replay_output_aliases). The codegen'd _alias_fn
# always builds and returns a list, so list[Any] is tighter than the reference's
# -> Any (the early `return fw_outs` path does not exist in the codegen'd fn).
_EpilogueApplyMutationsFn = Callable[[dict[int, Tensor], list[Any]], None]
_EpilogueReplayAliasesFn = Callable[[dict[int, Tensor], list[Any]], list[Any]]


def _codegen_epilogue(
    buf: "PySourceBuilder",
    runtime_metadata: ViewAndMutationMeta,
    apply_mutations_fn: _EpilogueApplyMutationsFn | None,
    replay_aliases_fn: _EpilogueReplayAliasesFn | None,
    num_mutated_runtime_inps: int,
    expected_outs: int,
) -> None:
    # [FUNCTION DOC] HELPER: Emits the generated epilogue: an output-count check, splitting updated inputs
    # from forward outputs and applying the mutation function, applying the alias replay function if any
    # output is aliased, stamping dynamic dims on outputs that have them, setting the grad mode if the graph
    # changed it, and returning the outputs. Binds the supplied functions and dim sets as globals of the code.
    buf.emit(f"if len(all_outs) != {expected_outs}:", indent=1)
    buf.emit(
        f'raise AssertionError(f"expected {expected_outs} outputs, '
        f'got {{len(all_outs)}}")',
        indent=2,
    )

    if num_mutated_runtime_inps > 0:
        buf.emit(f"updated_inputs = all_outs[:{num_mutated_runtime_inps}]", indent=1)
        buf.emit(f"fw_outs = all_outs[{num_mutated_runtime_inps}:]", indent=1)
        buf.emit("_apply_mutations_(orig_inputs, updated_inputs)", indent=1)
        buf.add_global("_apply_mutations_", apply_mutations_fn)
    else:
        buf.emit("fw_outs = all_outs", indent=1)

    if runtime_metadata.num_outputs_aliased > 0:
        buf.add_global("_replay_aliases_", replay_aliases_fn)
        buf.emit("ret_outs = _replay_aliases_(orig_inputs, fw_outs)", indent=1)
    else:
        buf.emit("ret_outs = fw_outs", indent=1)

    if runtime_metadata.dynamic_outputs:
        buf.add_global("_mark_dynamic_", mark_dynamo_propagated_dynamic_indices)
        for i, o in enumerate(runtime_metadata.output_info):
            if o.dynamic_dims is not None:
                dims_name = f"_dyn_dims_{i}"
                buf.add_global(dims_name, o.dynamic_dims)
                buf.emit(f"_mark_dynamic_(ret_outs[{i}], {dims_name})", indent=1)

    if runtime_metadata.grad_enabled_mutation is not None:
        buf.emit(
            f"torch._C._set_grad_enabled({runtime_metadata.grad_enabled_mutation!r})",
            indent=1,
        )

    buf.emit("return ret_outs", indent=1)


def _create_runtime_wrapper(
    compiled_fn: Callable[..., Any],
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: list[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
) -> Callable[..., Any]:
    # [FUNCTION DOC] Builds the runtime wrapper around a compiled function, using generated code. Creates the
    # reference invoker and epilogue objects, then generates (when needed) an alias regeneration function
    # and a mutation function with per-output and per-input branches resolved at compile time, and an
    # orchestration function that captures originals, bumps versions, calls the compiled function and runs
    # the epilogue. Returns a Python wrapper (boxed args list in, outputs out) that optionally records a
    # profiler region for the prologue; for joint graphs with inlineable saved tensors hooks it is further
    # wrapped to run with those hooks disabled. Args: compiled function, runtime metadata, detach indices,
    # joint flag, keep-input-mutations flag, amp flag.
    compiled_invoker = _RuntimeCompiledFnInvoker(
        compiled_fn=compiled_fn,
        indices_of_inps_to_detach=indices_of_inps_to_detach,
        trace_joint=trace_joint,
        disable_amp=disable_amp,
    )
    runtime_epilogue = _RuntimeForwardEpilogue(
        runtime_metadata=runtime_metadata,
        trace_joint=trace_joint,
        keep_input_mutations=keep_input_mutations,
    )

    # The orchestration closes over these two codegen'd epilogue functions directly
    # (bound into rw_globals as _apply_mutations_ / _replay_aliases_ in _codegen_epilogue).
    # Each stays None when its epilogue step is absent for this graph.
    codegen_apply_mutations: _EpilogueApplyMutationsFn | None = None
    codegen_alias_fn: _EpilogueReplayAliasesFn | None = None

    # Codegen output alias regeneration: emit straight-line code per output
    # with all handler branches resolved at compile time.
    if runtime_metadata.num_outputs_aliased > 0:
        output_handlers = runtime_epilogue.output_handlers
        from .codegen import PySourceBuilder

        buf = PySourceBuilder(
            "_alias_fn",
            args="orig_inputs, fw_outs",
            artifact_name="output_alias_wrapper",
        )
        buf.bind(
            gen_alias_from_base=gen_alias_from_base,
            _unwrap_tensoralias=_unwrap_tensoralias,
        )
        with buf.indent():
            buf.writeline("ret_outs = []")
            for i, handler in enumerate(output_handlers):
                if isinstance(handler, NoopAliasHandler):
                    buf.writeline(f"ret_outs.append(fw_outs[{i}])")
                elif isinstance(handler, IsInputHandler):
                    buf.writeline(f"ret_outs.append(orig_inputs[{handler.base_idx}])")
                elif isinstance(handler, AliasOfInputHandler):
                    vms_name = buf.bind_value("_vms", handler.view_meta_sequence)
                    out_expr = (
                        f"_unwrap_tensoralias(fw_outs[{i}])"
                        if trace_joint
                        else f"fw_outs[{i}]"
                    )
                    buf.writeline(
                        f"ret_outs.append(gen_alias_from_base("
                        f"orig_inputs[{handler.base_idx}], {out_expr}, "
                        f"{handler.requires_grad!r}, {vms_name}, "
                        f"replay_views={handler.replay_views!r}))"
                    )
                elif isinstance(handler, AliasOfIntermediateHandler):
                    vms_name = buf.bind_value("_vms", handler.view_meta_sequence)
                    out_expr = (
                        f"_unwrap_tensoralias(fw_outs[{i}])"
                        if trace_joint
                        else f"fw_outs[{i}]"
                    )
                    base_unwrap = (
                        handler._unwrap_aliased_base_tensor is _unwrap_tensoralias
                    )
                    base_expr = (
                        f"_unwrap_tensoralias(fw_outs[{handler.base_idx}])"
                        if base_unwrap
                        else f"fw_outs[{handler.base_idx}]"
                    )
                    buf.writeline(
                        f"ret_outs.append(gen_alias_from_base("
                        f"{base_expr}, {out_expr}, "
                        f"{handler.requires_grad!r}, {vms_name}, "
                        f"replay_views={handler.replay_views!r}))"
                    )
                else:
                    raise AssertionError(
                        f"unhandled output handler type: {type(handler).__name__}"
                    )
            buf.writeline("return ret_outs")

        codegen_alias_fn = typing.cast(_EpilogueReplayAliasesFn, buf.build())

    def record_runtime_wrapper_prologue_enter() -> AbstractContextManager[None] | None:
        # [FUNCTION DOC] HELPER: If the profiler is enabled and runtime overhead recording is configured, starts
        # and enters a record-function scope named for the runtime wrapper prologue and returns it; else returns
        # none.
        if (
            torch.autograd.profiler._is_profiler_enabled
            and dynamo_config.record_runtime_overhead
        ):
            cm = torch._C._profiler._RecordFunctionFast(
                "AOTDispatcher Runtime Wrapper Prologue"
            )
            cm.__enter__()
            return cm
        return None

    def record_runtime_wrapper_prologue_exit(
        cm: AbstractContextManager[None] | None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Exits the profiler scope returned by the matching enter function, if there is one.
        if cm is not None:
            cm.__exit__(None, None, None)

    # Codegen mutation epilogue: emit straight-line code per mutated input
    # with all branches resolved at compile time.
    if runtime_metadata.num_mutated_inp_runtime_indices > 0:
        from .codegen import PySourceBuilder

        buf = PySourceBuilder(
            "_apply_mutations",
            args="orig_inputs, updated_inputs",
            artifact_name="mutation_epilogue",
        )
        buf.bind(torch=torch, _unwrap_tensoralias=_unwrap_tensoralias)
        wrote_body = False
        with buf.indent():
            for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
                meta = runtime_metadata.input_info[inpt_idx]
                if not meta.mutates_data and not meta.mutates_metadata:
                    continue
                wrote_body = True
                oi = f"orig_inputs[{inpt_idx}]"
                ui = f"updated_inputs[{i}]"
                if meta.mutates_storage_metadata:
                    u = buf.fresh_name("_u")
                    if trace_joint:
                        buf.writeline(f"{u} = _unwrap_tensoralias({ui})")
                    else:
                        buf.writeline(f"{u} = {ui}")
                    buf.writeline(f"with torch.no_grad(): {oi}.set_({u})")
                elif meta.mutates_metadata and not meta.mutates_data:
                    u = buf.fresh_name("_u")
                    if trace_joint:
                        buf.writeline(f"{u} = _unwrap_tensoralias({ui})")
                    else:
                        buf.writeline(f"{u} = {ui}")
                    buf.writeline(
                        f"{oi}.as_strided_({u}.size(), {u}.stride(), {u}.storage_offset())"
                    )
                else:
                    if meta.mutates_data and meta.mutates_metadata:
                        buf.writeline(
                            f"{oi}.as_strided_({ui}.size(), {ui}.stride(), {ui}.storage_offset())"
                        )
                    else:
                        if not meta.mutates_data:
                            raise AssertionError(
                                f"expected mutates_data for input {inpt_idx}"
                            )
                    if meta.is_leaf:
                        buf.writeline(
                            f"if {oi}.requires_grad: {oi}.detach().copy_({ui})"
                        )
                        buf.writeline(f"else: {oi}.copy_({ui})")
                    else:
                        has_stream = (
                            runtime_metadata.mutated_inp_stream_indices is not None
                            and i < len(runtime_metadata.mutated_inp_stream_indices)
                            and runtime_metadata.mutated_inp_stream_indices[i]
                            is not None
                        )
                        if has_stream:
                            msg_name = buf.bind_value(
                                "_stream_err",
                                "Mutations on inputs with user-specified streams are not yet supported. "
                                "See: https://github.com/pytorch/pytorch/issues/172522",
                            )
                            buf.writeline(f"raise RuntimeError({msg_name})")
                        else:
                            buf.writeline(f"{oi}.copy_({ui})")
            if not wrote_body:
                buf.writeline("pass")

        codegen_apply_mutations = typing.cast(_EpilogueApplyMutationsFn, buf.build())

    from .codegen import PySourceBuilder

    epilogue_args_idx = runtime_epilogue.epilogue_args_idx
    num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
    expected_outs = (
        num_mutated_runtime_inps
        + runtime_metadata.num_outputs
        + runtime_metadata.num_intermediate_bases
    )

    buf = PySourceBuilder(
        "_runtime_wrapper",
        args="_compiled_fn_, _first_ctx_, _on_before_call_, args",
        artifact_name="runtime_wrapper_orchestration",
    )
    buf.bind(torch=torch)

    _codegen_capture_orig_inputs(buf, epilogue_args_idx)
    _codegen_increment_mutation_versions(buf, keep_input_mutations, runtime_metadata)
    _codegen_compiled_fn_invocation(
        buf, trace_joint, indices_of_inps_to_detach, disable_amp
    )
    _codegen_epilogue(
        buf,
        runtime_metadata,
        codegen_apply_mutations,
        codegen_alias_fn,
        num_mutated_runtime_inps,
        expected_outs,
    )

    _codegen_runtime_wrapper = buf.build()

    _inner_compiled_fn = compiled_invoker.compiled_fn
    _first_invocation_ctx = compiled_invoker.first_invocation_ctx

    @simple_wraps(_inner_compiled_fn)
    def runtime_wrapper(args: list[Any]) -> Any:
        # [FUNCTION DOC] The runtime wrapper entry point. Takes the boxed argument list, enters the profiler
        # prologue scope, calls the generated orchestration function with the inner compiled function, the
        # first-invocation context, a callback that ends the prologue scope just before the compiled call, and
        # the args. Ensures the prologue scope is exited, then returns the result.
        cm = record_runtime_wrapper_prologue_enter()
        prologue_exited = False

        def exit_prologue() -> None:
            # [FUNCTION DOC] HELPER: Idempotent callback that exits the profiler prologue scope exactly once.
            nonlocal prologue_exited
            if not prologue_exited:
                record_runtime_wrapper_prologue_exit(cm)
                prologue_exited = True

        try:
            result = _codegen_runtime_wrapper(
                _inner_compiled_fn,
                _first_invocation_ctx,
                exit_prologue,
                args,
            )
        finally:
            exit_prologue()
        del args
        return result

    if not (trace_joint and _should_disable_saved_tensors_hooks()):
        return runtime_wrapper

    @simple_wraps(runtime_wrapper)
    def _runtime_wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Variadic wrapper that calls the runtime wrapper with saved tensors hooks
        # disabled.
        with _disable_saved_tensors_hooks():
            return runtime_wrapper(*args, **kwargs)

    return _runtime_wrapper


# WARNING: this does NOT operate on TraceFn
@dataclass
class FunctionalizedRngRuntimeWrapper(InductorWrapper):
    # [CLASS DOC] InductorWrapper for functionalized RNG ops. Attribute return_new_outs controls whether
    # the rng offset output is stripped from the outputs returned to the caller (default true). Does not
    # operate on TraceFn. pre_compile appends seed and offset example inputs; post_compile wraps the
    # function to pass and update RNG state.
    # TODO: I would love to get rid of this argument, but it's
    # Wrapped pretty tightly around our aot_dispatch_autograd logic.
    # Specifically, tensors_saved_for_backwards_slice's value is both used for calculating indices
    # for setting placeholder strides(which is done before runtime, before this wrapper runs)
    # and for saving tensors for backward (which is done during runtime, after this wrapper runs)
    # So in aot_dispatch_autograd, this wrapper can't edit the set of outs without making one
    # of those two indices incorrect.
    return_new_outs: bool = True

    def pre_compile(
        self,
        fw_module: torch.fx.GraphModule,
        flat_args: list[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None:
        # [FUNCTION DOC] When functionalized rng ops are configured, appends the current seed and offset (as
        # fake tensors, from the detected fake mode) to the example flat args in place. Raises an assertion
        # error if no fake mode is active. No-op otherwise.
        if config.functionalize_rng_ops:
            # Update example inputs for the fw_compiler
            fake_mode = detect_fake_mode()
            if fake_mode is None:
                raise AssertionError(
                    "fake_mode must not be None when functionalize_rng_ops is True"
                )
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            flat_args.extend([seed, offset])
            # We are not clearing flat_args here because
            # 1) There is a check in the debug compiler at the end
            # 2) It does not matter as these are fake tensors

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns the compiled function unchanged unless rng ops were functionalized. Otherwise
        # (requires exactly one rng offset output, else assertion error) generates a boxed wrapper that appends
        # the current torch seed and offset to the runtime args, calls the compiled function, stores the new
        # offset from the output at the offset position, and returns outputs with that position removed (or all
        # outputs when return_new_outs is false).
        if not runtime_metadata.is_rng_op_functionalized:
            return compiled_fn

        if runtime_metadata.num_outputs_rng_offset != 1:
            raise AssertionError(
                f"expected num_outputs_rng_offset == 1, got {runtime_metadata.num_outputs_rng_offset}"
            )

        from .codegen import PySourceBuilder

        offset_index = runtime_metadata.num_forward_returns
        buf = PySourceBuilder(
            "_functionalized_rng_wrapper",
            args="runtime_args",
            artifact_name="functionalized_rng_wrapper",
        )
        buf.bind(
            _compiled_fn_=compiled_fn,
            _get_rng_state_=CUDARngStateHelper.get_torch_state_as_tuple,
            _set_offset_=CUDARngStateHelper.set_new_offset,
        )
        with buf.indent():
            buf.writeline("seed, offset = _get_rng_state_()")
            buf.writeline("runtime_args.extend([seed, offset])")
            buf.writeline("outs = _compiled_fn_(runtime_args)")
            buf.writeline(f"_set_offset_(outs[{offset_index}])")
            if self.return_new_outs:
                buf.writeline(
                    f"return outs[:{offset_index}] + outs[{offset_index + 1}:]"
                )
            else:
                buf.writeline("return outs")

        inner_fn = buf.build(wrapped_fn=compiled_fn)
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn

    # Calling convention: If we are running functionalized RNG, then outs consists
    # of (user_outs, rng_offset)
    def _functionalized_rng_runtime_epilogue(
        self,
        metadata: ViewAndMutationMeta,
        outs: Any,
        offset_index: int,
    ) -> Any:
        # [FUNCTION DOC] Eager reference for the post-call RNG handling: if rng ops are functionalized
        # (requires exactly one offset output, else assertion error), stores the new offset taken from the output
        # list and returns outputs minus the offset entry (or unchanged when return_new_outs is false). Otherwise
        # returns the outputs unchanged.
        if metadata.is_rng_op_functionalized:
            if metadata.num_outputs_rng_offset != 1:
                raise AssertionError(
                    f"expected num_outputs_rng_offset == 1, got {metadata.num_outputs_rng_offset}"
                )
            new_rng_offset = outs[offset_index]
            CUDARngStateHelper.set_new_offset(new_rng_offset)
            if self.return_new_outs:
                user_outs = outs[:offset_index] + outs[offset_index + 1 :]
                return user_outs
            else:
                return outs

        return outs


# WARNING: this does NOT operate on TraceFn
@dataclass
class FakifiedOutWrapper(InductorWrapper):
    # [CLASS DOC] InductorWrapper that makes the first call of a compiled forward return precomputed fake
    # outputs instead of running it (used when the tracing context requests fakify on first call). State:
    # output fake tensors taken from the forward graph, strides computed by Inductor (set after compile,
    # entries None for non tensors), and a flag for whether post_compile work is needed. Does not operate
    # on TraceFn.
    out_metas: list[torch.Tensor] = field(default_factory=list)
    # TracingContext.fwd_output_strides
    # Generated from actually doing compile
    # NB: an entry is None if it's not a Tensor
    fwd_output_strides: list[list[int] | None] | None = None
    needs_post_compile: bool = True

    def pre_compile(
        self,
        fw_module: fx.GraphModule,  # Must be fw_module from aot_dispatch_*_graph
        flat_args: list[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None:
        # [FUNCTION DOC] If the tracing context asks to fakify the first call, records the fake output values
        # found in the last (output) node of the forward graph; otherwise marks that post_compile is not needed.
        tracing_context = torch._guards.TracingContext.try_get()
        if tracing_context and tracing_context.fakify_first_call:
            self.out_metas = [
                n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
            ]
        else:
            self.needs_post_compile = False

    def _compute_output_meta_with_inductor_strides(self) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Returns the recorded fake outputs, restriding tensor outputs whose stride
        # statically differs from the Inductor-computed output strides (entries with unknown strides are skipped).
        # Mutates and returns the stored list; returns it unchanged when no strides are known.
        out = self.out_metas
        fwd_output_strides = self.fwd_output_strides
        if not fwd_output_strides:
            return out

        from torch.fx.experimental.symbolic_shapes import statically_known_true

        for i in range(len(out)):
            if not isinstance(out[i], Tensor):
                continue
            strides = fwd_output_strides[i]
            # fwd_output_strides is best effort by Inductor.  When an output
            # Tensor has unbacked SymInts, Inductor may sometimes be unable
            # to compute what the output stride would be.  If Inductor doesn't
            # have any clear direction on the layout, we don't have to run
            # as_strided.  To repro without this, run:
            #
            # python test/distributed/test_dynamo_distributed.py
            # TestFakeDistributedSingleProc.test_unbacked_symbol_splitting_no_binding
            if strides is None:
                continue
            if all(
                statically_known_true(s1 == s2)
                for s1, s2 in zip(out[i].stride(), strides)
            ):
                continue
            out[i] = out[i].as_strided(out[i].shape, strides)
        return out

    # To be called post compile
    def set_fwd_output_strides(
        self, fwd_output_strides: list[list[int] | None]
    ) -> None:
        # [FUNCTION DOC] Stores the Inductor-computed output strides; to be called after compilation.
        self.fwd_output_strides = fwd_output_strides

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] When needed, returns a wrapper that on its first call returns the precomputed fake
        # outputs (and drops them) and on later calls runs the compiled function; otherwise returns the compiled
        # function unchanged. Raises an assertion error if strides were never set but are needed.
        if self.needs_post_compile:
            if self.fwd_output_strides is None:
                raise AssertionError(
                    "fwd_output_strides must not be None when needs_post_compile is True"
                )
            fakified_out = self._compute_output_meta_with_inductor_strides()

            @wraps(compiled_fn)
            def wrapper(runtime_args: list[Any]) -> Any:
                # [FUNCTION DOC] HELPER: Returns the precomputed fake outputs the first time and clears them; afterward
                # delegates to the compiled function with the runtime args.
                nonlocal fakified_out
                if fakified_out is not None:
                    out = fakified_out
                    fakified_out = None
                    return out
                return compiled_fn(runtime_args)

            return wrapper
        # If we don't need to fakify, we can just return the original compiled function
        return compiled_fn


# This wrapper handles the AOTDispatch runtime logic for tensor subclasses.
# At runtime, we have a compiled function that knows how to operate on the domain of DenseTensor -> DenseTensor,
# But the user might have passed us some tensor subclass inputs (or expect some subclass tensor outputs).
# This function handles the wrapping and unwrapping of tensor subclasses at runtime.
@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper handling tensor subclasses at runtime: the compiled graph works on dense
    # tensors, so this wraps/unwraps subclass inputs and outputs. State: joint-trace flag, a forward only
    # function (used only in pre_compile, not cached), the subclass metadata (set by pre_compile) and the
    # count of forward outputs saved for backward.
    trace_joint: bool
    fw_only: Callable[..., Any] | None  # Not cached, only used in pre_compile
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        # [FUNCTION DOC] Traces the function so that tensor subclasses are desugared (via the subclass dispatch
        # tracing helper). Stores the resulting subclass metadata on the wrapper and returns the new flat
        # function, args, arg descriptors and the unchanged forward metadata.
        (new_flat_fn, new_flat_args, new_flat_args_descs, subclass_meta) = (
            aot_dispatch_subclass(
                flat_fn,
                flat_args,
                flat_args_descs,
                is_joint_structure=self.trace_joint,
                meta=fw_metadata,
                fw_only=self.fw_only,  # type: ignore[arg-type]
            )
        )
        self.maybe_subclass_meta = subclass_meta
        return new_flat_fn, new_flat_args, new_flat_args_descs, fw_metadata

    @staticmethod
    def _get_frozen_inp_indices() -> frozenset[int]:
        # [FUNCTION DOC] HELPER: Returns the set of parameter indices that were frozen (replaced by none in the
        # current tracing context's flat params), or an empty set if there is no tracing context or params.
        # fw_compiler_freezing (compile_fx.py) bakes frozen params into the
        # graph and sets their TracingContext.params_flat entries to None
        # before post_compile runs.  We pass these indices to codegen so it
        # can emit straight-line code instead of a runtime None check.
        tc = TracingContext.try_get()
        if tc is None or tc.params_flat is None:
            return frozenset()
        return frozenset(i for i, p in enumerate(tc.params_flat) if p is None)

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns the compiled function unchanged if there is no subclass metadata and no
        # activation input paths. Otherwise generates a wrapper (via the subclass codegen helper) that unwraps
        # subclass inputs, calls the compiled function and rewraps outputs, using the input and output subclass
        # metadata, saved-for-backward count and frozen indices. Marks it as boxed-call and returns it.
        if self.maybe_subclass_meta is None and not runtime_metadata.act_input_paths:
            return compiled_fn

        from .subclass_codegen import codegen_subclass_wrapper

        inner_fn = codegen_subclass_wrapper(
            compiled_fn=compiled_fn,
            inp_metas=runtime_metadata.subclass_inp_meta,
            out_metas=runtime_metadata.subclass_fw_graph_out_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
            frozen_inp_indices=self._get_frozen_inp_indices(),
            act_input_paths=runtime_metadata.act_input_paths,
        )
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


@dataclass
class EffectTokensWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper that handles effect tokens at runtime by prepending dummy token arguments
    # and stripping the token outputs. Stateless; see post_compile.
    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns the compiled function unchanged if there are no effect tokens. Otherwise
        # generates a wrapper that prepends one none per token to the args, clears the original arg list,
        # calls the compiled function, returns none if it returned none, and otherwise drops the leading
        # token outputs.
        num_tokens = len(runtime_metadata.tokens)
        if num_tokens == 0:
            return compiled_fn

        from .codegen import PySourceBuilder

        buf = PySourceBuilder(
            "_effect_tokens_wrapper",
            args="args",
            artifact_name="effect_tokens_wrapper",
        )
        buf.bind(_compiled_fn_=compiled_fn)
        with buf.indent():
            buf.writeline(f"new_args = [{', '.join(['None'] * num_tokens)}, *args]")
            buf.writeline("args.clear()")
            buf.writeline("outs = _compiled_fn_(new_args)")
            buf.writeline("if outs is None:")
            with buf.indent():
                buf.writeline("return None")
            buf.writeline(f"return outs[{num_tokens}:]")

        inner_fn = buf.build(wrapped_fn=compiled_fn)
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


# MOTIVATION:
#
# When tracing functions for future execution, one must be careful not to pass
# in the same input tensor multiple times (e.g., f(x, x), as this can result
# in graphs that are ONLY valid if you later pass a new tensor in exactly the
# same way (e.g., f(y, y)).  (NB: we really mean duplicate; two distinct
# tensors that alias each other is a different situation that is covered by
# aot_dispatch_deduplicated_autograd). Here are two examples:
#
# (1) Suppose you have a function:
#
#   def f(x, y):
#       return x + y
#
# If you make_fx(f)(x, x), you will trace out:
#
#   def f(x, y):
#       return y + y
#
# Oops!
#
# (2) For most tensors x and y, you can compute f's gradient with respect to
# these to inputs by saying torch.autograd.grad(f(x, y), (x, y)).  However,
# if x is y, you will trace out a program that gets incorrect gradients:
#
#   >>> x = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + x, (x, x))
#   (tensor([2.]), tensor([2.]))
#
# In other words, the gradient is double-counted.  Deduplicating the arguments
# gives you an appropriate gradient:
#
#   >>> y = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + y, (x, y))
#   (tensor([1.]), tensor([1.]))
#
# HOW TO DEDUPLICATE:
#
# There are a few strategies, in order of preference:
#
# 1. For every duplicate argument to the function, detach it into
#    a separate leaf tensor, so that it is no longer duplicated.
#
#       PRO: The resulting compiled graph works for any configuration
#       of duplicated arguments.
#
#       CON: It does not (naively) work if you mutate the metadata of inputs:
#
#           def f(x, y):
#               x.transpose_(0, 1)
#               y.transpose_(0, 2)
#
#           x = torch.randn(2, 3, 4)
#           f(x, x)
#
#       The ordering of the transposes inside f dictates whether or not
#       you get [4, 2, 3] or [3, 4, 2].  This means that you cannot precompute
#       what metadata mutations should get applied to each input; you need to
#       assume they aren't duplicates (what we do today) or preserve
#       the original metadata mutations exactly in order, so that they work
#       for any duplicate configuration.
#
#       CON: It does not (naively) work if you mutate the data of inputs.
#       In particular, leaf tensors that require grad cannot be mutated,
#       this makes it impossible to differentiate with respect to the original
#       base.
#
# 2. For every duplicate argument to the function, remove it, so it is
#    no longer part of the "true" signature:
#
#       PRO: Implemented naively, it still works for metadata/data mutation.
#
#       CON: The resulting compiled graph is duplicate-specialized: it only
#       works if future calls duplicate arguments in exactly the same way.
#       Horribly, Dynamo doesn't guard on this at the moment.  But even if
#       it did, you could still end up recompiling a bunch of each duplicate.
#
# Our strategy is to do (1) if we can, and do (2) otherwise, erroring if
# Dynamo's guards are not enough.  In practice, this seems to cover
# everything.
#
@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper that removes duplicate tensor inputs (the same tensor object passed more
    # than once) before compilation. Strategy 1: detach duplicates into separate leaves when the duplicate
    # input is not mutated. Strategy 2 (when a duplicate is mutated): drop the duplicates from the signature
    # and specialize the graph to that duplication pattern. State: keep_arg_mask (which original positions
    # are kept), add_dupe_map (for each original position, the position in the deduplicated list),
    # old_input_metadata (unused here), and needs_post_compile (false when strategy 1 sufficed).
    keep_arg_mask: list[bool] = field(default_factory=list)
    add_dupe_map: list[int] = field(default_factory=list)
    old_input_metadata: list[InputAliasInfo] = field(default_factory=list)
    needs_post_compile: bool = True

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(self, args: list[Any]) -> list[Any]:
        # [FUNCTION DOC] Hot-path helper: returns the arguments whose keep mask entry is true, in order, i.e.
        # the deduplicated argument list. Raises on length mismatch between args and mask (strict zip).
        return [t for t, keep in zip(args, self.keep_arg_mask) if keep]

    def add_dupe_args(self, args: list[Any]) -> list[Any]:
        # [FUNCTION DOC] Hot-path helper: rebuilds the original argument list from a deduplicated list by
        # reading each position of the add_dupe_map.
        return [args[i] for i in self.add_dupe_map]

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        # [FUNCTION DOC] Prepares function and inputs for tracing without duplicate inputs. First tries strategy
        # 1: walks the flat args, keeping non-tensors and first occurrences, and replacing repeated unmutated
        # tensors by detached leaf copies with the same requires-grad; if that works, no post_compile is needed
        # and the inputs are returned with the function and metadata unchanged. If some duplicate is mutated:
        # raises a runtime error when subclasses are involved or when exporting; otherwise builds the
        # keep mask and duplicate map, removes duplicates from args and descriptors, rewrites the metadata, and
        # (when a tracing context and arg sources exist) records duplicate-input guards. Returns a wrapped
        # function taking deduplicated args, the deduplicated args and descriptors, and the updated metadata.
        # Under the debug-assert config the updated metadata is checked against a fresh trace.
        # KNOWN ISSUE: the walrus assignment of tracing_context binds the whole conjunction expression, so it
        # holds the arg-position-to-source mapping rather than the tracing context; the later guards_context
        # access would then fail with an attribute error if any duplicate was removed with sources available.
        # Use information about whether or not flat_fn mutates its arguments
        # or not to handle dupe args

        # Strategy 1: For any input that is not mutated, we can leafify it if we
        # need to remove a duplicate.
        leaf_flat_args: list[FxValue] = []
        leaf_flat_args_descs: list[AOTInput] = []
        args_set = set()
        ok = True

        for i, (a, a_desc) in enumerate(zip(flat_args, flat_args_descs)):
            if not isinstance(a, torch.Tensor):
                leaf_flat_args.append(a)
                leaf_flat_args_descs.append(a_desc)
            elif a not in args_set:
                args_set.add(a)
                leaf_flat_args.append(a)
                leaf_flat_args_descs.append(a_desc)
            elif (
                not fw_metadata.input_info[i].mutates_data
                and not fw_metadata.input_info[i].mutates_metadata
            ):
                leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
                leaf_flat_args_descs.append(a_desc)
            else:
                ok = False
                break

        if ok:
            self.needs_post_compile = False
            return flat_fn, leaf_flat_args, leaf_flat_args_descs, fw_metadata

        if requires_subclass_dispatch(leaf_flat_args, fw_metadata):  # type: ignore[arg-type]
            raise RuntimeError(
                """\
        Encountered duplicate inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        # export path: ban duplicate inputs for now, add later if requested.
        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered duplicated inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        fw_metadata={str(fw_metadata)}
            """
            )

        # Strategy 2: Duplicate specialization
        #
        # When we have duplicate arguments in a function call, we need to handle them specially.
        # For example, if we have a function call f(a, b, a, c), we need to:
        #
        # 1. Remove duplicates to get a deduplicated list [a, b, c]
        # 2. Compile our function to work with this deduplicated list
        # 3. At runtime, convert incoming arguments with duplicates to the deduplicated form
        # 4. Pass the deduplicated arguments to our compiled function
        #
        # To do this, we need two helper functions:
        #
        # - remove_dupe_args: Converts [a, b, a, c] -> [a, b, c]
        # - add_dupe_args: Converts [a, b, c] -> [a, b, a, c]
        #
        # For our example [a, b, a, c], we track:
        #
        # - seen_args = {a: 0, b: 1, c: 2} (maps each unique arg to its first position)
        # - add_dupe_map = [0, 1, 0, 2] (tells us how to reconstruct the original list)
        # - keep_arg_mask = [True, True, False, True] (tells us which args to keep when deduplicating)

        seen_args: dict[Tensor, int] = {}
        # Implicitly map duped arg position (list index) to de-duped arg position
        keep_arg_mask: list[bool] = []
        add_dupe_map: list[int] = []
        duped_arg_len = len(flat_args)

        j = 0  # index into deduped_flat_args
        for t in flat_args:
            if isinstance(t, torch.Tensor):
                if t in seen_args:
                    keep_arg_mask.append(False)
                    add_dupe_map.append(seen_args[t])
                    continue
                seen_args[t] = j

            keep_arg_mask.append(True)
            add_dupe_map.append(j)
            j += 1
        if len(add_dupe_map) != duped_arg_len:
            raise AssertionError(
                f"Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}"
            )

        self.keep_arg_mask = keep_arg_mask
        self.add_dupe_map = add_dupe_map

        deduped_flat_args = self.remove_dupe_args(flat_args)
        # TODO: instead of arbitrarily removing args, it might be useful to
        # have a record that these were duped, perhaps as a mutable attribute
        # on the kept arg?  Do this if someone needs it
        deduped_flat_args_descs = self.remove_dupe_args(flat_args_descs)

        # Update our input metadata to remove duped input metadata.
        updated_fw_metadata = remove_dupe_metadata(
            fw_metadata, keep_arg_mask, add_dupe_map
        )

        if (
            tracing_context := TracingContext.try_get()
            and aot_config.aot_autograd_arg_pos_to_source
        ):
            # TODO(voz): This structure is 1:1, we could consider an alternate structure like
            # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
            # which feels like needless complexity for a tiny bit of efficiency at this point.
            for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(
                zip(add_dupe_map, keep_arg_mask)
            ):
                if not keep_arg:
                    dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        dupe_arg_pos
                    ]
                    kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        kept_pos
                    ]
                    tracing_context.guards_context.aotautograd_guards.append(  # type: ignore[attr-defined]
                        DuplicateInputs(kept_arg_source, dupe_arg_source)
                    )

        @simple_wraps(flat_fn)
        def wrapped_flat_fn(
            *args: FxValue,
        ) -> tuple[list[FxValue], list[AOTOutput]]:
            # [FUNCTION DOC] HELPER: Traced function that takes deduplicated args, re-expands duplicates with the
            # add_dupe_args step, calls the original flat function and returns its outputs and output descriptors.
            outs, out_descs = call_and_expect_output_descs(
                flat_fn,
                self.add_dupe_args(args),  # type: ignore[arg-type]
            )
            return outs, out_descs

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                without_output_descs(wrapped_flat_fn),
                flat_args_descs=deduped_flat_args_descs,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
            )(*deduped_flat_args)
            if ref_fw_metadata != updated_fw_metadata:
                raise AssertionError(
                    f"ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}"
                )

        return (
            wrapped_flat_fn,
            deduped_flat_args,
            deduped_flat_args_descs,
            updated_fw_metadata,
        )

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] If no deduplication was done, returns the compiled function. Otherwise generates a
        # boxed wrapper that selects the kept arguments by index, clears the original arg list and calls the
        # compiled function with the deduplicated list; then wraps it in a debug function that verifies the
        # dedupe/redupe round trip reproduces the original args (raising an assertion error with a guard bug
        # message if a position would be a duplicate of another) before calling the generated wrapper.
        # Returns the debug wrapper marked as boxed-call.
        # KNOWN ISSUE: the debug wrapper is always installed (the conditional to skip it is commented out),
        # and its tracking dict of seen args is built but never used for a check.
        if not self.needs_post_compile:
            return compiled_fn

        keep_indices = [i for i, keep in enumerate(self.keep_arg_mask) if keep]
        idx_list = ", ".join(f"args[{i}]" for i in keep_indices)
        from .codegen import PySourceBuilder

        buf = PySourceBuilder("inner_fn", args="args", artifact_name="dedup_wrapper")
        buf.bind(compiled_fn=compiled_fn)
        with buf.indent():
            buf.writeline(f"deduped_args = [{idx_list}]")
            buf.writeline("args.clear()")
            buf.writeline("return compiled_fn(deduped_args)")

        wrapped_compiled_fn: Callable[..., Any] = buf.build(  # type: ignore[assignment]
            wrapped_fn=compiled_fn
        )

        wrapped_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        # This can be uncommented when we properly guard for duplicates,
        # but right now we must not do it.
        # if not config.debug_assert:
        #     return wrapped_compiled_fn

        @wraps(wrapped_compiled_fn)
        def debugged_compiled_fn(args: list[Any]) -> Any:
            # [FUNCTION DOC] HELPER: Debug check that removing then re-adding duplicate args reproduces the incoming
            # args identically; raises an assertion error with a guard-failure message naming the offending inputs
            # otherwise. Then calls the generated wrapper with the args and returns its result.
            # Test that the computed remove/add arg functions are an inverse
            new_args = self.add_dupe_args(self.remove_dupe_args(args))
            seen: dict[Any, None] = {}
            for i, (x, y) in enumerate(zip(new_args, args)):
                seen[y] = None
                if x is not y:
                    raise AssertionError(
                        format_guard_bug_msg(
                            aot_config,
                            f"{describe_input(i, aot_config)} would be a duplicate of "
                            f"{describe_input(self.add_dupe_map[i], aot_config)}",
                        )
                    )
            # This is only an error if there is metadata mutation on both of
            # the duped arguments; in this case, we need to know what order
            # the metadata mutation applies in.  You'll get the correct result
            # otherwise, because a graph that assumes distinct inputs works if
            # you dupe the inputs (the gradient contributions from each input
            # will get summed up appropriately.)
            #
            # TODO: work out how to setup this assert correctly
            """
            assert len(seen) == unique_args, format_guard_bug_msg(aot_config,
                f"there would be {unique_args} distinct arguments"
            )
            """
            return wrapped_compiled_fn(args)

        debugged_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        return debugged_compiled_fn


# This layer handles the situation where you have two inputs that alias each other,
# and one of the inputs is mutated.
# We need to take special care to ensure that the mutation is applied to the other aliases in the graph.
#
# pre-condition: AOTDedupWrapper has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper that merges mutated inputs that alias each other (share storage) into a
    # single synthetic base argument. Assumes deduplication already ran. State: trace_joint (autograd case
    # has more restrictions), needs_post_compile (false when no synthetic base was needed),
    # aliased_arg_idx_with_metadata_mutations (inputs that saw metadata mutation and are returned as extra
    # outputs to replay the mutation), base_groups (mapping from synthetic base index to original arg
    # indices) and other_arg_indices (original indices of non-merged args in new calling order).
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    trace_joint: bool  # TODO: refactor trace_joint
    needs_post_compile: bool = True
    aliased_arg_idx_with_metadata_mutations: list[int] = field(default_factory=list)
    base_groups: dict[int, list[int]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )
    other_arg_indices: list[int] = field(default_factory=list)

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Callable[..., Any], list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        # [FUNCTION DOC] Merges aliased mutated inputs into synthetic bases using merge_view_inputs. If none are
        # needed, marks post_compile unnecessary and returns inputs unchanged. Raises a runtime error for
        # subclasses or export, and an assertion error on metadata length mismatch. Otherwise updates the
        # metadata for synthetic bases, precomputes base_groups and other_arg_indices (ordered by position in
        # the bases-then-others calling convention) for the later codegen, and returns a wrapped function (which
        # regenerates the original views from the bases and returns the metadata mutated aliased args as extra
        # outputs), the merged args, their descriptors, and the new metadata. Debug-assert config verifies the
        # metadata against a fresh trace.
        is_inference = not self.trace_joint
        (
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
            synthetic_base_info,
        ) = merge_view_inputs(
            aot_config,
            flat_args,
            flat_args_descs,
            fw_metadata.input_info,
            is_inference=is_inference,
        )

        # Happy path: we don't need synthetic bases
        if synthetic_base_info is None:
            self.needs_post_compile = False
            return flat_fn, flat_args, flat_args_descs, fw_metadata

        # export path: ban synthetic bases for now, add later if requested.
        if requires_subclass_dispatch(flat_args, fw_metadata):  # type: ignore[arg-type]
            raise RuntimeError(
                """\
        Encountered aliased inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered aliased inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        synthetic_base_info={str(synthetic_base_info)}

        fw_metadata={str(fw_metadata)}
                """
            )

        if len(fw_metadata.input_info) != len(synthetic_base_info):
            raise AssertionError(
                f"expected len(fw_metadata.input_info) == len(synthetic_base_info), "
                f"got {len(fw_metadata.input_info)} != {len(synthetic_base_info)}"
            )

        # Update our forward metadata to take synthetic bases into account
        (
            fw_metadata_updated,
            aliased_arg_idx_with_metadata_mutations,
        ) = create_synthetic_base_metadata(
            fw_metadata,
            synthetic_base_info,
            flat_args,
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
        )
        # Pre-compute base_groups and other_arg_map from
        # synthetic_base_info so the post_compile codegen can emit a
        # wrapper without calling merge_view_inputs at runtime.
        #
        # other_arg_entries collects (new_idx, orig_idx) pairs so we can
        # sort by new_idx -- the position in the post-merge calling
        # convention (bases ++ others).  merge_view_inputs puts
        # non-tensor args before non-aliased tensor args, so iterating
        # by orig_idx alone would produce a wrong order when non-tensors
        # are interleaved with tensors.
        other_arg_entries: list[tuple[int, int]] = []
        for orig_idx, entry in enumerate(synthetic_base_info):
            if isinstance(entry, int):
                other_arg_entries.append((entry, orig_idx))
            else:
                base_idx, _ = entry
                self.base_groups[base_idx].append(orig_idx)
        other_arg_entries.sort()
        self.other_arg_indices = [orig_idx for _, orig_idx in other_arg_entries]

        self.aliased_arg_idx_with_metadata_mutations = (
            aliased_arg_idx_with_metadata_mutations
        )
        replay_views = config.view_replay_for_aliased_outputs

        def _unpack_synthetic_bases(primals: tuple[Any, ...]) -> list[Any]:
            # [FUNCTION DOC] HELPER: Rebuilds the original argument list from the merged primals: integer entries
            # are looked up directly; tuple entries regenerate a view from the synthetic base at that index (using
            # the recorded view tensor, its requires-grad and the view-replay config). Returns the list.
            f_args_inner = []
            # pyrefly: ignore [not-iterable]
            for inner_idx_or_tuple in synthetic_base_info:
                if isinstance(inner_idx_or_tuple, int):
                    f_args_inner.append(primals[inner_idx_or_tuple])
                else:
                    inner_base_idx, view_tensor = inner_idx_or_tuple
                    base = primals[inner_base_idx]
                    view_arg = gen_alias_from_base(
                        base,
                        view_tensor,
                        view_tensor.requires_grad,
                        replay_views=replay_views,
                    )
                    f_args_inner.append(view_arg)
            return f_args_inner

        @simple_wraps(flat_fn)
        def wrapped_flat_fn(*args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Traced function taking the merged args: unpacks synthetic bases into original
            # inputs, remembers the inputs that had metadata mutations, calls the original function, and returns
            # its outputs followed by those inputs (with matching descriptors) so downstream logic treats them as
            # outputs aliasing inputs. Without such inputs returns outputs and descriptors unchanged.
            unpacked_args = _unpack_synthetic_bases(args)
            # This is a bit subtle. The goal of this entire function (aot_dispatch_synthetic_bases)
            # is to relieve the downstream logic from having to reason about mutations on inputs that alias
            # each other, by replacing aliased inputs with a synthetic base.
            # One area where this breaks down a bit however is if one of those aliased inputs
            # experienced a metadata mutation.
            # We are now obligated to reapply the metadata mutation directly to the user's input;
            # it isn't enough to apply mutations back to the synthetic base in the downstream logic.
            #
            # The way we handle this is by pretending that those aliased inputs that experience metadata mutations
            # are additional outputs in the user's forward function.
            # The downstream logic will just treat these as "user outputs that alias inputs".
            # However, we will manually grab them at runtime here, use them to reapply the metadata mutation
            # to the user inputs, and not return them to the user.
            aliased_args_with_metadata_mutations = [
                x
                for i, x in enumerate(unpacked_args)
                if i in self.aliased_arg_idx_with_metadata_mutations
            ]
            out, out_descs = call_and_expect_output_descs(flat_fn, unpacked_args)  # type: ignore[arg-type]
            if len(aliased_args_with_metadata_mutations) > 0:
                # TODO: record more detailed desc information here
                return (*out, *aliased_args_with_metadata_mutations), (
                    *out_descs,
                    *(
                        [
                            MetadataMutationAOTOutput(i)
                            for i in range(
                                len(self.aliased_arg_idx_with_metadata_mutations)
                            )
                        ]
                    ),
                )
            else:
                return out, out_descs

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                without_output_descs(wrapped_flat_fn),
                flat_args_descs=flat_args_descs_with_synthetic_bases,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
            )(*flat_args_with_synthetic_bases)
            if ref_fw_metadata != fw_metadata_updated:
                raise AssertionError(
                    f"ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, "
                    f"\nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}"
                )
        return (
            wrapped_flat_fn,
            flat_args_with_synthetic_bases,
            flat_args_descs_with_synthetic_bases,
            fw_metadata_updated,
        )

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns the compiled function unchanged when no synthetic base was used. Otherwise
        # generates a boxed wrapper that: for each base group takes the shared _base of the inputs (or builds a
        # fresh empty tensor sharing the first input's storage), builds the argument list of bases followed by
        # the other args, clears the incoming list, calls the compiled function, and if there were metadata
        # mutated aliased inputs strips their trailing outputs and re-applies the size/stride/offset of those
        # outputs to the original inputs. Returns the wrapper marked as boxed-call.
        if not self.needs_post_compile:
            return compiled_fn

        from .codegen import PySourceBuilder

        base_groups = self.base_groups
        other_indices = self.other_arg_indices

        num_bases = len(base_groups)
        aliased_meta_mut_indices = self.aliased_arg_idx_with_metadata_mutations
        num_meta_mut = len(aliased_meta_mut_indices)

        buf = PySourceBuilder(
            "_synthetic_base_wrapper",
            args="args",
            artifact_name="synthetic_base_wrapper",
        )
        buf.bind(torch=torch, _compiled_fn_=compiled_fn)

        with buf.indent():
            buf.writeline(f"_bases = [None] * {num_bases}")
            for base_idx in sorted(base_groups):
                group = base_groups[base_idx]
                first_orig = group[0]
                if len(group) == 1:
                    indices_check = f"args[{first_orig}]._base is not None"
                    base_expr = f"args[{first_orig}]._base"
                else:
                    indices_str = ", ".join(str(i) for i in group)
                    indices_check = (
                        f"any(args[_i]._base is not None for _i in [{indices_str}])"
                    )
                    base_expr = f"next(args[_i]._base for _i in [{indices_str}] if args[_i]._base is not None)"
                buf.writeline(f"if {indices_check}:")
                with buf.indent():
                    buf.writeline(f"_bases[{base_idx}] = {base_expr}")
                buf.writeline("else:")
                with buf.indent():
                    buf.writeline(
                        f"_b = torch.empty((0,), dtype=args[{first_orig}].dtype, device=args[{first_orig}].device)"
                    )
                    buf.writeline(f"_b.set_(args[{first_orig}].untyped_storage())")
                    buf.writeline(f"_bases[{base_idx}] = _b")

            other_items = ", ".join(f"args[{orig}]" for orig in other_indices)
            buf.writeline(f"_new_args = _bases + [{other_items}]")

            if num_meta_mut > 0:
                meta_items = ", ".join(f"args[{i}]" for i in aliased_meta_mut_indices)
                buf.writeline(f"_meta_mut = [{meta_items}]")

            buf.writeline("args.clear()")
            buf.writeline("_outs = _compiled_fn_(_new_args)")

            if num_meta_mut > 0:
                # Handles metadata mutations on inputs that were merged into
                # synthetic bases (create_synthetic_base_metadata hides these
                # from the normal tracking by forcing mutates_metadata=False
                # on the base).  Metadata mutations on non-aliased inputs are
                # handled separately by _apply_input_mutations in the runtime
                # epilogue.
                buf.writeline(f"_mut_inps = _outs[-{num_meta_mut}:]")
                buf.writeline(f"_user_outs = _outs[:-{num_meta_mut}]")
                buf.writeline("for _inp, _mi in zip(_meta_mut, _mut_inps):")
                with buf.indent():
                    buf.writeline(
                        "_inp.as_strided_(_mi.size(), _mi.stride(), _mi.storage_offset())"
                    )
                buf.writeline("return _user_outs")
            else:
                buf.writeline("return _outs")

        inner_fn = buf.build(wrapped_fn=compiled_fn)
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


# Note [Handling mutations on an input that aliases other inputs]
# The easiest example to show-case this edge case is here:
#
# def f(a, b):
#     a.mul_(2)
#     out = a + b
#     return out
# b = torch.ones(...)
# a = b.view(-1)
# f(a, b)
#
# In this situation, if a and b happened to be aliased, we need to trace something different!
# Suppose we had b = a.view(-1)
# (In this case, that means that `a._base is b`)
#
# We need to ensure that the aliasing relationship between a and b is preserved.
# We do that detecting the specific situation above (mutate an input that aliases another input),
# and when we do that, we create a synthetic base argument. Then inside of the traced forward,
# we regenerate a and b off of that base.
# The complete example of the transformed function looks like this:
#
# // The traced forward takes in a synthetic base, and regenerates the aliased inputs as views
# // We could consider getting view-replay support here to minimize as_strided_scatter ops in the graph
# def traced_forward(base):
#     a = base.as_strided(...)
#     b = base.as_strided(...)
#     a_updated = a.mul(2)
#     base_updated = torch.as_strided_scatter(base, a_updated, ...)
#     b_updated = base_updated.as_strided(...)
#     out = a_updated + b_updated
#     return a_updated, out
#
# def compiled_fn(a, b):
#     // we detect that a is the "differentiable base" here
#     base = a
#     // In other situations, we might do either:
#     // (1) a and b are both views off of some larger differentiable base
#     //     assert a._base is b._base and a._base is not None
#     //     base = a._base
#     // (2) a and b both don't require gradients. Create a base from the storage
#     //     assert a._base is None and b._base is None
#     //     base = torch.Tensor(a.storage())
#     a_updated, out = traced_forward(base)
#     a.copy_(a_updated)
#     return out
#
# This function:
# (1) Merges input views into a synthetic base argument, when any of those input views are mutated
# (2) Returns metadata telling the autograd.Function how to modify their arguments properly,
#     to respect the new calling convention.
#
# The calling convention is as follows.
# Any inputs that were originally views of one another get yanked, and replaced with a synthetic base.
# The argument list ordering goes [base1, ..., baseN], [arg1, ..., argN],
# Where the ordering of the bases is determined from the ordering of the original view args.
# baseA will come before baseB if the earliest original argument coming from baseA
# showed up earlier in the argument list than the earliest original argument coming from baseB.
#
# Example, given some tensors a, b, c, d
# call site:
#   f(a, c.view(-1), b.view(-1), b, c, d)
# Modified argument list:
#   c_base comes first because the first c view came earlier in arg list than the first b view
#   a and d still show up in the modified arg list, but b and c don't- they're regenerated from their bases
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    aot_config: AOTConfig,
    fwd_inputs: list[Any],
    # This is None when called at runtime from post_compile closure
    fwd_inputs_descs: list[AOTInput] | None,
    mutated_input_info: list[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> tuple[list[Any], list[AOTInput], list[int | tuple[int, torch.Tensor]] | None]:
    # [FUNCTION DOC] Decides whether mutated inputs that alias each other must be replaced by synthetic base
    # arguments, and builds the new calling convention. Args: AOT config, forward inputs, optional input
    # descriptors (none at runtime; dummy ones are made), per-input mutation info, and an inference flag.
    # Returns a triple: new args (synthetic bases followed by the other args), their descriptors, and a list
    # whose i-th entry tells how to produce the i-th original argument (either an index into the new args or
    # a pair of base index and view tensor to regenerate from). Returns the inputs unchanged with none as
    # the third value when there are no data mutations, no aliasing group with a mutation, or aliasing is
    # only false sharing (non overlapping). Groups inputs by shared storage; for each mutated group, checks
    # (raising an assertion error) that in the autograd case the views are differentiable views of each
    # other, that dtypes agree, and that autograd bases are consistent. The synthetic base is the common
    # autograd base, or a new empty tensor sharing the storage when no input has a base. Side effects: none
    # beyond constructing new base tensors.
    if fwd_inputs_descs is None:
        fwd_inputs_descs = [DummyAOTInput(i) for i in range(len(fwd_inputs))]

    def _are_differentiable_views(view1: torch.Tensor, view2: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True if the two tensors are the same object or autograd knows they are views of
        # a common base (one is the other's base or they share a base); False if neither has a base or they
        # differ.
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1: torch.Tensor, view2: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True only if both views have the same dtype and neither differs in dtype from
        # its own autograd base.
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True

    def _format_input(idx: int) -> str:
        # [FUNCTION DOC] HELPER: Builds a display string for an input position, using the source local name from
        # the config's arg-position-to-source table when available, otherwise the argument descriptor's
        # expression. Used in error messages.
        if (
            aot_config.aot_autograd_arg_pos_to_source is not None
            and idx < len(aot_config.aot_autograd_arg_pos_to_source)
            and aot_config.aot_autograd_arg_pos_to_source[idx] is not None
        ):
            source = aot_config.aot_autograd_arg_pos_to_source[idx]
            name = getattr(source, "local_name", source.name)
        else:
            name = fwd_inputs_descs[idx].expr()
        return f"input {idx} ({name})"

    if len(fwd_inputs) != len(mutated_input_info):
        raise AssertionError(
            f"expected len(fwd_inputs) == len(mutated_input_info), "
            f"got {len(fwd_inputs)} != {len(mutated_input_info)}"
        )
    if not [info for info in mutated_input_info if info.mutates_data]:
        # Return early when there are no mutations.
        return fwd_inputs, fwd_inputs_descs, None

    storage_ref_to_idx: dict[StorageWeakRef, list[int]] = collections.defaultdict(list)
    # pyrefly: ignore [implicit-any]
    base_args = []
    # pyrefly: ignore [implicit-any]
    other_args = []
    base_args_descs = []
    other_args_descs = []
    for i, (inpt, source) in enumerate(zip(fwd_inputs, fwd_inputs_descs)):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
            other_args_descs.append(source)
    # Note [Synthetic Base Info Metadata]
    # This list contains metadata that tells you what the i'th argument in the inner calling convention should be.
    # It's either:
    # - another int (corresponding to the index in the argument list of the element from the outer calling convention)
    # - idx, view_tensor, where we can generate the new output with view_tensor._view_func(old_args[idx])
    #   idx corresponds to which synthetic base from the outer calling context to view
    inner_calling_convention_meta: dict[int, int | tuple[int, torch.Tensor]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) <= 1 or not any(
            # We only care about mutations that affect all aliases,
            # so metadata mutations on an input doesn't require us to do synthetic base handling.
            mutated_input_info[inpt_idx].mutates_data
            for inpt_idx in aliased_input_indices
        ):
            other_args.extend(
                fwd_inputs[curr_idx] for curr_idx in aliased_input_indices
            )
            other_args_descs.extend(
                fwd_inputs_descs[curr_idx] for curr_idx in aliased_input_indices
            )
            continue

        # Here, we attempt to do a more complicated check to detect false aliasing
        # (e.g. if all the tensors have the same storage, but don't actually overlap)
        # In theory, we could have a large group of tensors that all share storages, where only *some* of them
        # have overlapping memory.
        # I don't bother with that case for now: here, we only bail out earlier if we detect that **every** pair
        # of tensors in the current group that shares a storage is non-overlapping.
        aliased_input_indices_no_false_sharing = compute_overlapping_inputs(
            aot_config, fwd_inputs, aliased_input_indices
        )
        if len(aliased_input_indices_no_false_sharing) <= 1:
            other_args.extend(
                fwd_inputs[curr_idx] for curr_idx in aliased_input_indices
            )
            other_args_descs.extend(
                fwd_inputs_descs[curr_idx] for curr_idx in aliased_input_indices
            )
            continue

        # We detected an input that was mutated, AND aliases with another input.
        # we need to replace this set of aliased inputs with a single synthetic base.
        # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
        # and error out. We can fix them later.
        # These checks are transitive, so we don't need to check every pair.
        for idx1, idx2 in zip(
            aliased_input_indices, aliased_input_indices[1:], strict=False
        ):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            # The "inputs that are aliased but have different differentiable bases" case
            # is more complicated and hopefully pretty rare. Not currently handled.
            if not is_inference:
                if not _are_differentiable_views(view1, view2):
                    raise AssertionError(
                        f"aot_autograd() does not yet handle non-differentiable view input mutations. "
                        f"{_format_input(idx1)} and {_format_input(idx2)} share storage but are "
                        f"not differentiable views of each other."
                    )
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            if not _same_dtype_views(view1, view2):
                raise AssertionError(
                    "aot_autograd() does not yet handle input mutations on views with different dtypes."
                )
        non_none_bases = [
            (i, fwd_inputs[i]._base)
            for i in aliased_input_indices
            if fwd_inputs[i]._base is not None
        ]
        aliases_with_none_bases = [
            fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None
        ]
        synthetic_base_desc: AOTInput
        if len(non_none_bases) == 0:
            # Case where none of the aliases have a ._base
            # we generate a synthetic base without gradients, and generate views off of it
            # We hit this case when we have input tensors to the graph that share a storage,
            # but do not have a ._base field.
            # Wondering when we hit this case?
            # The _base field simply says that autograd knows about the aliasing relationship,
            # but sometimes we create tensors which are aliased out of the same storage but guaranteed
            # to be disjoint. In these cases, we will skip setting up the _base relationship
            # for performance reasons (because the fact that the tensors share the same storage
            # is unobservable unless you (1) do naughty things with resize_/as_strided
            # or (2) look at the storage--as we are doing here.)
            # One particular example of this is optimizer steps on the LSTM module:
            # LSTM parameters are packed into a contiguous storage for efficiency reasons when
            # calling cuDNN kernels, so when these parameters get passed to the optimizer we will
            # find they share the same storage, but do not have _base set since they are all disjoint.
            #
            # NOTE: There is one case where this is unsafe:
            # torch.Tensor(storage) will ALWAYS create a 1D tensor, which is not necessarily
            # the same shape as the "actual" base that the tensor came from.
            # For the most part this is fine, because we always use as_strided()
            # to generate the original aliased inputs again.
            # If we were to use view-replay though, this could cause the aliased views
            # to have incorrect sizes.
            example_idx = aliased_input_indices[0]
            example_alias = fwd_inputs[example_idx]
            # Note that this function is reused at both trace time and runtime.
            # At trace time, we're under a FakeMode so synthetic_base becomes a FakeTensor.
            synthetic_base = torch.empty(
                (0,), dtype=example_alias.dtype, device=example_alias.device
            )
            # We don't actually have a convenient way of going from storage -> tensor,
            # So using set_() here (we suffer some minor overhead, but this case is rare).
            synthetic_base.set_(example_alias.untyped_storage())
            synthetic_base_desc = SyntheticBaseAOTInput(fwd_inputs_descs[example_idx])
        else:
            # Case where all of the aliases require gradients, and have the same _base.
            i, synthetic_base = non_none_bases[0]
            synthetic_base_desc = ViewBaseAOTInput(fwd_inputs_descs[i])
            for j, other_base in non_none_bases[1:]:
                if other_base is not synthetic_base:
                    raise AssertionError(
                        f"aot_autograd() does not yet handle non-differentiable view input mutations. "
                        f"Aliased inputs share storage but have different autograd ._base tensors: "
                        f"{_format_input(i)} and {_format_input(j)} have ._base fields that point to different tensors."
                    )
            for alias in aliases_with_none_bases:
                if alias is not synthetic_base:
                    none_base_indices = [
                        _format_input(k)
                        for k in aliased_input_indices
                        if fwd_inputs[k]._base is None
                    ]
                    has_base_indices = [
                        _format_input(k)
                        for k in aliased_input_indices
                        if fwd_inputs[k]._base is not None
                    ]
                    raise AssertionError(
                        f"aot_autograd() does not yet handle non-differentiable view input mutations. "
                        f"Aliased inputs share storage but have mixed autograd ._base states: "
                        f"{has_base_indices} have ._base set, while "
                        f"{none_base_indices} have ._base=None (and are not the synthetic base)."
                    )
        base_args.append(synthetic_base)
        base_args_descs.append(synthetic_base_desc)
        for curr_view_idx in aliased_input_indices:
            curr_view = fwd_inputs[curr_view_idx]
            base_idx = len(base_args) - 1
            # We store just enough info here so that we can regenerate the view later.
            # Regeneration: curr_view._view_func(args[base_idx])
            inner_calling_convention_meta[curr_view_idx] = (base_idx, curr_view)
    if len(base_args) == 0:
        if len(other_args) != len(fwd_inputs):
            raise AssertionError(
                f"expected len(other_args) == len(fwd_inputs), "
                f"got {len(other_args)} != {len(fwd_inputs)}"
            )
        # If no synthetic bases are necessary, just return the original inputs.
        return fwd_inputs, fwd_inputs_descs, None
    else:
        from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

        def make_hashable(arg: Any) -> Any:
            # [FUNCTION DOC] HELPER: Wraps symbolic ints in a hashable equality-by-expression wrapper so they can be
            # used as dictionary keys; returns other values unchanged.
            if isinstance(arg, torch.SymInt):
                # Since only nested SymInt objects can be hashed, we wrap them with
                # SymIntEqByExpr, which is a hashable wrapper of SymInts.
                return SymIntEqByExpr(arg)
            return arg

        # Otherwise, return:
        # (1) The new args according to the updated calling convention: (synthetic_bases, other_args)
        # (2) Metadata telling functionalization how to generate the inner argument list given the outer calling convention.
        #     We post-process it into a list, where meta[i] tells you info about the i'th argument in the inner calling convention.
        args_to_functionalization = base_args + other_args
        args_to_functionalization_descs = base_args_descs + other_args_descs

        # Map each argument into its old index.
        # There may be some repeated arguments, so we collect their indices in a list.
        arg_to_old_idx_map = collections.defaultdict(list)
        for i, arg in enumerate(fwd_inputs):
            arg_to_old_idx_map[make_hashable(arg)].append(i)
        # Reverse the list of each argument, so that we can easily pop them one-after-the-other in order.
        for hashable_arg in arg_to_old_idx_map:
            arg_to_old_idx_map[hashable_arg] = list(
                reversed(arg_to_old_idx_map[hashable_arg])
            )

        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[make_hashable(other_arg)].pop()
            inner_calling_convention_meta[old_idx] = new_idx

        # post process into a list
        post_processed_calling_convention_meta: list[int | tuple[int, torch.Tensor]] = [
            -1 for _ in range(len(inner_calling_convention_meta))
        ]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            if x == -1:
                raise AssertionError(
                    "every argument in the inner calling convention should be accounted for"
                )
        # pyrefly: ignore [bad-return]
        return (
            args_to_functionalization,
            args_to_functionalization_descs,
            post_processed_calling_convention_meta,
        )


# Note: [Backward graph lazy lowering]
# After AOTDispatch traces the backward for graphs requiring autograd, we will lower the graph lazily,
# unless we suspect that inductor might specialize and insert additional guards. When we do lazy
# lowering, we stash the AOT backward graph (bw_module) in this class.
#
# Lowering passes are performed on a deepcopy of this bw_module due to compatibility
# with compiled autograd. See: https://github.com/pytorch/pytorch/pull/149229#discussion_r2002122645.
@dataclass
class AutogradLazyBackwardCompileInfo:
    # [CLASS DOC] Holds what is needed to lazily compile the backward graph at first backward run: the
    # backward graph module, the placeholder example inputs list, and the tracing and compile contexts saved
    # from forward compile time. Plain data container. See the note above about lazy lowering.
    bw_module: Callable[..., Any]
    placeholder_list: list[Any]
    saved_context: TracingContext | None
    saved_compile_context: CompileContext | None


# On an AOT Autograd cache hit, we already have a lowered backward, so there is usually
# no need to keep information around for a new lazy compilation. Except for compiled autograd,
# which wants to retrace this backward into a larger graph, and it needs the graph module to do so.
@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    # [CLASS DOC] Data container stored on a cache hit instead of the full lazy info: just the backward module
    # function, kept only so compiled autograd can retrace the backward. Plain data.
    bw_module_fn: Callable[..., Any]


def initialize_rng_states(
    num_rng: int,
    graphsafe_idx: int,
    fwd_rng_states: list[torch.Generator],
    bwd_rng_states: list[torch.Generator],
    *,
    device: torch.device,
) -> None:
    # [FUNCTION DOC] Fills the forward and backward lists of graph-safe RNG generator states at runtime
    # (deferred from compile time since compile preserves rng state). Draws one random 64-bit seed per rng on
    # CPU (advancing the global rng), clones the default generator for the device state once per rng for each
    # of the two lists and seeds each with the corresponding seed, so the forward and backward generators
    # start identical. Runs with dispatch modes disabled. Mutates the two lists in place. The graphsafe index
    # argument is unused.
    """
    Initialize the graphsafe rng states.

    Initialization of rng states should have a few properties:
    - the initialization for each rng state should be independent
    - the initialization should be deterministic
    - the initialization should be based off current rng state, so that independent graphs do not
    have equal rng behavior

    We defer initialization of rng states until runtime because compilation is wrapped
    with preserve_rng_states. Seed initialization should advance the rng states so consecutive compilations
    do not give equal randomness.
    """
    from .utils import get_default_generator

    with torch.utils._python_dispatch._disable_current_modes():
        seeds = torch.randint(0, torch.iinfo(torch.int64).max, (num_rng,), device="cpu")
        generator = get_default_generator(device)
        fwd_rng_states.extend(
            [generator.clone_state().manual_seed(int(seeds[i])) for i in range(num_rng)]
        )
        bwd_rng_states.extend(
            [generator.clone_state().manual_seed(int(seeds[i])) for i in range(num_rng)]
        )


def coerce_to_expected_memory_format(
    x: torch.Tensor, memory_format: MemoryFormatMeta
) -> torch.Tensor:
    # [FUNCTION DOC] Returns the tangent tensor in the layout the compiled backward expects. If the metadata
    # carries a memory format, makes the tensor contiguous in that format if it is not already. Otherwise
    # expects static size and stride in the metadata (assertion error if missing) and returns the tensor if
    # it already matches, else copies it into a new tensor with the expected strides, same dtype, device,
    # layout and requires-grad. Raises assertion error on missing size or stride.
    if memory_format.memory_format is not None:
        # Coerce to torch.memory_format
        if not x.is_contiguous(memory_format=memory_format.memory_format):
            x = x.contiguous(memory_format=memory_format.memory_format)
        return x

    expected_size = memory_format.size
    if expected_size is None:
        raise AssertionError("memory_format.size must not be None")
    expected_stride = memory_format.stride
    if expected_stride is None:
        raise AssertionError("memory_format.stride must not be None")
    # Expected size and stride are static ints
    # ok to use == to compare runtime tensor strides and shapes

    if x.shape == expected_size and x.stride() == expected_stride:
        # Runtime tangent size and stride are the same as expected, no need to coerce
        return x

    # Empty_strided creates a raw Tensor.
    # We are guaranteed that only raw Tensors has expected size and stride.
    # Subclasses have only expected memory_format.
    restrided = torch.empty_strided(
        size=expected_size,
        stride=expected_stride,
        dtype=x.dtype,
        device=x.device,
        layout=x.layout,
        requires_grad=x.requires_grad,
    )
    restrided.copy_(x)
    return restrided


@contextlib.contextmanager
def _disable_saved_tensors_hooks() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that disables saved tensors hooks with an explanatory error
    # message (not failing if hooks stack is already non-empty) for the duration of the block, and on exit
    # restores the previous disabled state: re-disables with the earlier message if there was one, otherwise
    # re-enables hooks. Used because inlined hooks must not be pushed or popped inside the compiled region.
    error_message = (
        "Saved tensors hooks were specialized as GraphModules."
        "In this case aot_autograd inlines them in forward and backward graph "
        "and disables them during runtime of aot_autograd compiled region."
        "If you see this error, that means that there is some unexpected push or pop manipulation "
        "during aot_autograd compiled region runtime."
        "Compilation with different hooks must result in recompilation."
    )
    fail_if_non_empty = False
    maybe_prev_message = None
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        torch._C._autograd._saved_tensors_hooks_disable(
            error_message, fail_if_non_empty
        )
        yield
    finally:
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(
                maybe_prev_message, fail_if_non_empty
            )


@dataclass
class SerializableCompiledFunction:
    # [CLASS DOC] Pairs a compiled function with a callable that serializes it; behaves like the compiled
    # function when called and copies its documentation/annotation attributes. Attributes: compiled_fn,
    # serialize_fn.
    """
    Represents a result of AOTDispatch after calling the inner compiler
    that can be serialized
    """

    compiled_fn: Callable[..., Any]
    serialize_fn: Callable[[], Any]

    def __init__(
        self, compiled_fn: Callable[..., Any], serialize_fn: Callable[[], Any]
    ) -> None:
        # [FUNCTION DOC] Stores the compiled function and serializer and copies the doc, annotations and type
        # params from the compiled function onto this object (wraps-like).
        self.compiled_fn = compiled_fn
        self.serialize_fn = serialize_fn
        # Equivalent to functools.wraps
        functools.update_wrapper(
            self,
            compiled_fn,
            assigned=("__doc__", "__annotations__", "__type_params__"),
        )

    def serialize(self) -> Any:
        # [FUNCTION DOC] Calls the stored serialization function and returns its result.
        return self.serialize_fn()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Forwards all positional and keyword arguments to the compiled function and returns its
        # result.
        return self.compiled_fn(*args, **kwargs)


@dataclass
class AOTDispatchAutogradCompileSpec:
    # [CLASS DOC] Plain data bundle of everything needed to construct the autograd function for a compiled
    # training graph: compiled forward and (maybe lazy) backward functions, subclass metadata, count of
    # saved symints, backward state indices, amp flag, input indices to detach, lazy backward info,
    # backward compiler, AOT config, forward metadata and an optional cache-entry saving callback.
    compiled_fw_func: Callable[..., Any]
    compiled_bw_func: Callable[..., Any] | None
    maybe_subclass_meta: SubclassMeta | None
    num_symints_saved_for_bw: int
    backward_state_indices: list[int]
    disable_amp: bool
    indices_of_inps_to_detach: list[int]
    lazy_backward_info: (
        AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None
    )
    bw_compiler: Callable[..., Any] | None
    aot_config: AOTConfig
    fw_metadata: ViewAndMutationMeta
    try_save_cache_entry: Callable[..., Any] | None


@dataclass
class _AutogradSavedState:
    # [CLASS DOC] HELPER: Knows how to store forward results needed by backward onto the autograd context.
    # State: the forward/mutation metadata describing slices of the forward outputs.
    metadata: ViewAndMutationMeta

    def save_from_forward(self, ctx: Any, fw_outs: Sequence[Any]) -> None:
        # [FUNCTION DOC] Splits forward outputs into saved tensors (with and without version counter check),
        # symints and opaque objects using the metadata slices, validating each group's types (assertion
        # errors on mismatch). Saved tensors that are views and not graph inputs are detached. Marks dynamic dims
        # on saved tensors per the metadata. Calls the context's save-for-backward on the version-checked
        # tensors; stores the others, the symints and the opaque objects as attributes on the context. Mutates
        # ctx.
        tensors_saved_with_vc_check = fw_outs[
            self.metadata.tensors_saved_for_backwards_with_vc_check_slice
        ]
        tensors_saved_no_vc_check = fw_outs[
            self.metadata.tensors_saved_for_backwards_no_vc_check_slice
        ]
        if not all(isinstance(x, torch.Tensor) for x in tensors_saved_with_vc_check):
            raise AssertionError(
                "expected all tensors_saved_with_vc_check to be Tensors, "
                f"got types: {[type(x) for x in tensors_saved_with_vc_check]}"
            )
        if not all(isinstance(x, torch.Tensor) for x in tensors_saved_no_vc_check):
            raise AssertionError(
                "expected all tensors_saved_no_vc_check to be Tensors, "
                f"got types: {[type(x) for x in tensors_saved_no_vc_check]}"
            )

        # See Note [Detaching saved tensors in AOTAutograd]
        num_vc_check = len(tensors_saved_with_vc_check)
        is_graph_input = self.metadata.saved_tensor_is_graph_input
        tensors_to_save = [
            x if is_graph_input[i] or not x._is_view() else x.detach()
            for i, x in enumerate(tensors_saved_with_vc_check)
        ]
        tensors_no_vc_check = [
            x if is_graph_input[num_vc_check + i] or not x._is_view() else x.detach()
            for i, x in enumerate(tensors_saved_no_vc_check)
        ]

        # dynamic_saved_tensors_idxs has indices relative to all saved tensors
        # (vc_check + no_vc_check combined). Mark dynamics on the detached tensors.
        for idx, dims in self.metadata.dynamic_saved_tensors_idxs.items():
            if idx < num_vc_check:
                mark_dynamo_propagated_dynamic_indices(tensors_to_save[idx], dims)
            else:
                mark_dynamo_propagated_dynamic_indices(
                    tensors_no_vc_check[idx - num_vc_check], dims
                )

        ctx.save_for_backward(*tensors_to_save)
        ctx._tensors_no_vc_check = tensors_no_vc_check

        symint_outs = fw_outs[self.metadata.symints_saved_for_backwards_slice]
        if not all(
            isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
            for x in symint_outs
        ):
            raise AssertionError(
                "expected all symint_outs to be int/float/SymInt/SymFloat, "
                f"got types: {[type(x) for x in symint_outs]}"
            )
        ctx.symints = symint_outs

        opaque_object_outs = fw_outs[
            self.metadata.opaque_objects_saved_for_backwards_slice
        ]
        if not all(
            is_custom_class(type(obj)) or isinstance(obj, CustomClassBase)
            for obj in opaque_object_outs
        ):
            raise AssertionError(
                "expected all opaque_object_outs to be opaque types, "
                f"got types: {[type(obj) for obj in opaque_object_outs]}"
            )
        ctx.opaque_objects = opaque_object_outs


@dataclass
class _AutogradForwardEpilogue:
    # [CLASS DOC] HELPER: Post-processes forward outputs of the autograd function. State: the metadata.
    metadata: ViewAndMutationMeta

    def finalize(self, ctx: Any, fw_outs: Sequence[Any]) -> tuple[Any, ...]:
        # [FUNCTION DOC] Takes the compiled forward outputs and returns the tuple the autograd function must
        # return: keeps only the first num_forward_returns values; wraps metadata-only mutated inputs and
        # outputs that alias inputs/others in alias wrapper objects so autograd does not treat them as tensors;
        # replaces unsafe-view outputs by unsafe views; marks outputs that do not require grad as non
        # differentiable on the context, disables materializing non diff grads, and snapshots external objects
        # on the context. Debug-assert config adds sanity checks (assertion errors). Mutates ctx.
        num_outputs = self.metadata.num_outputs
        num_outputs_aliased = self.metadata.num_outputs_aliased
        num_mutated_runtime_inps = self.metadata.num_mutated_inp_runtime_indices
        num_forward_returns = self.metadata.num_forward_returns

        raw_returns = list(fw_outs[:num_forward_returns])

        # Wrap all autograd.Function.forward() outputs that are aliases
        # so that autograd.Function doesn't treat them as tensors
        if num_mutated_runtime_inps > 0:
            for i, idx in enumerate(self.metadata.mutated_inp_runtime_indices):
                # We could make this faster by only looping over inputs with metadata-only mutations
                # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                info = self.metadata.input_info[idx]
                if info.mutates_metadata and not info.mutates_data:
                    raw_returns[i] = TensorAlias(raw_returns[i])

            if config.debug_assert:
                user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                mut_inp_infos = [
                    x
                    for x in self.metadata.input_info
                    if x.mutates_data or x.mutates_metadata
                ]
                if len(user_mutated_inputs_raw) != len(mut_inp_infos):
                    raise AssertionError(
                        "expected len(user_mutated_inputs_raw) == len(mut_inp_infos), "
                        f"got {len(user_mutated_inputs_raw)} != {len(mut_inp_infos)}"
                    )

        if self.metadata.num_unsafe_view_outputs > 0:
            for idx in self.metadata.unsafe_view_out_indices:
                raw_return_idx = num_mutated_runtime_inps + idx
                o = raw_returns[raw_return_idx]
                raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)

        if num_outputs_aliased > 0:
            for idx in self.metadata.aliased_out_indices:
                raw_return_idx = num_mutated_runtime_inps + idx
                raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

            if config.debug_assert:
                intermediates_raw = raw_returns[
                    num_mutated_runtime_inps + num_outputs :
                ]
                if any(isinstance(x, TensorAlias) for x in intermediates_raw):
                    raise AssertionError("expected no TensorAlias in intermediates_raw")

        # invariant: intermediate bases always require gradients, so we don't have to
        # consider marking them as non-differentiable.
        raw_returns_not_including_intermediate_bases = raw_returns[
            : num_mutated_runtime_inps + num_outputs
        ]
        raw_returns_meta = [
            x
            for x in self.metadata.input_info
            if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
        ] + self.metadata.output_info

        fw_outs_not_requiring_grad = [
            x
            for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
            if isinstance(x, torch.Tensor) and not raw_returns_meta[i].requires_grad
        ]
        ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
        ctx._materialize_non_diff_grads = False
        _snapshot_external_objects(ctx)

        return tuple(raw_returns)


@dataclass
class _AutogradRngStateTracker:
    # [CLASS DOC] HELPER: Tracks graph-safe RNG generator state across compiled forward and backward calls
    # so that each backward replays the rng state that its forward used, even when backward order differs.
    # State: number of rng args, graph-safe index, device, lists of forward and backward generators (filled
    # lazily), a counter of forward iterations, the position of the next expected backward, the set of
    # forwards pending a backward, and saved backward states keyed by iteration.
    num_rng: int
    graphsafe_idx: int | None
    device: torch.device | None = None
    fwd_rng_states: list[torch.Generator] = field(default_factory=list)
    bwd_rng_states: list[torch.Generator] = field(default_factory=list)
    curr_fwd_iter: Any = field(default_factory=lambda: itertools.count(0))
    backward_state_position: int = 0
    pending_forwards: set[int] = field(default_factory=set)
    saved_backward_tensor_states: dict[int, list[torch.Tensor]] = field(
        default_factory=dict
    )

    def add_forward_args(self, ctx: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        # [FUNCTION DOC] Forward side: returns args unchanged if no rng is used. Lazily initializes the generator
        # lists on first use (assertion error if graphsafe index or device is missing). Assigns a forward
        # iteration id on the context; if it is not the next backward position, saves the current forward
        # generator states for later. Marks the forward pending and returns the args followed by the forward
        # generators.
        if self.num_rng == 0:
            return args

        if len(self.fwd_rng_states) == 0:
            if self.graphsafe_idx is None:
                raise AssertionError("graphsafe_idx must not be None when num_rng > 0")
            if self.device is None:
                raise AssertionError("device must not be None when num_rng > 0")
            initialize_rng_states(
                self.num_rng,
                self.graphsafe_idx,
                self.fwd_rng_states,
                self.bwd_rng_states,
                device=self.device,
            )

        curr_iter = next(self.curr_fwd_iter)
        ctx._curr_iter = curr_iter

        # if this state is not contained in the backward,
        # we need to save it for when its backward pass happens
        if curr_iter != self.backward_state_position:
            self.saved_backward_tensor_states[curr_iter] = [
                rng_state.get_state() for rng_state in self.fwd_rng_states
            ]

        self.pending_forwards.add(curr_iter)
        return (*args, *self.fwd_rng_states)

    def add_backward_args(self, ctx: Any, all_args: list[Any]) -> None:
        # [FUNCTION DOC] Backward side: no-op if no rng is used. Using the iteration id on the context and the
        # current graph task retain-graph flag, saves current backward generator states if another pending
        # forward will need them, restores the saved states for this iteration (deleting them unless retaining
        # the graph), or asserts that this backward is the next in order. Advances the backward position,
        # removes the forward from the pending set unless retaining, and appends the backward generators to
        # the argument list in place.
        if self.num_rng == 0:
            return

        curr_backward_iter = ctx._curr_iter
        retain_graph = torch._C._autograd._get_current_graph_task_keep_graph()

        # Save current state if we have a pending forward that needs this state
        # or this state may be needed again because of retain graph
        if (
            self.backward_state_position in self.pending_forwards
            and self.backward_state_position not in self.saved_backward_tensor_states
            and (self.backward_state_position != curr_backward_iter or retain_graph)
        ):
            self.saved_backward_tensor_states[self.backward_state_position] = [
                rng_state.get_state() for rng_state in self.bwd_rng_states
            ]

        # Restore saved states if needed
        if curr_backward_iter in self.saved_backward_tensor_states:
            if self.backward_state_position != curr_backward_iter:
                for bwd_state, saved_state in zip(
                    self.bwd_rng_states,
                    self.saved_backward_tensor_states[curr_backward_iter],
                ):
                    bwd_state.set_state(saved_state)
            if not retain_graph:
                del self.saved_backward_tensor_states[curr_backward_iter]
        else:
            if self.backward_state_position != curr_backward_iter:
                raise AssertionError(
                    "expected backward_state_position == curr_backward_iter, "
                    f"got {self.backward_state_position} != {curr_backward_iter}"
                )

        self.backward_state_position = curr_backward_iter + 1
        if not retain_graph:
            self.pending_forwards.remove(curr_backward_iter)
        all_args.extend(self.bwd_rng_states)


@dataclass
class _AutogradBackwardCompiler:
    # [CLASS DOC] HELPER: Holds the possibly lazy compiled backward. State: the compiled backward (none until
    # compiled), lazy compile info, amp flag, backward compiler, AOT config, forward metadata and an
    # optional cache saving callback.
    compiled_bw: Callable[..., Any] | None
    lazy_backward_info: (
        AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None
    )
    disable_amp: bool
    bw_compiler: Callable[..., Any] | None
    aot_config: AOTConfig
    fw_metadata: ViewAndMutationMeta
    try_save_cache_entry: Callable[..., Any] | None

    def get_or_compile(self, *, saved_tensors_use_once: bool) -> Callable[..., Any]:
        # [FUNCTION DOC] Returns the compiled backward, compiling it first if needed. On first call (asserts a
        # non-cache lazy info and backward compiler exist), prepares the DDP/donation context, then compiles a
        # deep copy of the backward module with the backward compiler under the saved tracing/compile contexts,
        # autocast disabled if requested, graph compile tracking, metrics, timing and lazy-backward callbacks,
        # and optionally saves a cache entry. Stores and returns the result. Arg: whether saved tensors are used
        # only once.
        if self.compiled_bw is not None:
            return self.compiled_bw

        if self.lazy_backward_info is None:
            raise AssertionError("lazy_backward_info must not be None")
        if not isinstance(self.lazy_backward_info, AutogradLazyBackwardCompileInfo):
            raise AssertionError(
                "expected AutogradLazyBackwardCompileInfo, "
                f"got {type(self.lazy_backward_info)}"
            )

        self._prepare_lazy_backward_context(saved_tensors_use_once)

        bw_module = self.lazy_backward_info.bw_module
        placeholder_list = self.lazy_backward_info.placeholder_list
        saved_context = self.lazy_backward_info.saved_context
        saved_compile_context = self.lazy_backward_info.saved_compile_context

        context = torch._C._DisableAutocast if self.disable_amp else nullcontext
        metrics_context = get_metrics_context()
        with (
            tracing(saved_context),
            compile_context(saved_compile_context),
            context(),
            track_graph_compiling(self.aot_config, "backward"),
            metrics_context,
            dynamo_timed(
                "backward._backward_impl",
                phase_name="entire_backward_compile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="backward_cumulative_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="entire_backward_compile",
            ),
            callback_handler.install_callbacks(
                CallbackTrigger.LAZY_BACKWARD,
                str(CompileContext.current_compile_id()),
            ),
        ):
            CompileEventLogger.compilation_metric(is_forward=False)
            # See Note: [Backward graph lazy lowering]
            if self.bw_compiler is None:
                raise AssertionError("bw_compiler must not be None")
            self.compiled_bw = self.bw_compiler(
                copy.deepcopy(bw_module), placeholder_list
            )
            # Maybe save cache entry
            if self.try_save_cache_entry is not None:
                self.try_save_cache_entry(
                    self.compiled_bw,
                    bw_module,
                    self.fw_metadata,
                    self.aot_config,
                )

        return self.compiled_bw

    def _prepare_lazy_backward_context(self, saved_tensors_use_once: bool) -> None:
        # [FUNCTION DOC] HELPER: Before lazy backward compile, adjusts shared metadata. Asserts that lazy info is
        # a real lazy info and that a saved tracing context, if any, is valid. With a DDP optimizer context it
        # steps the bucket counter down and points the saved context's forward metadata at that bucket's
        # metadata (asserting the counter is valid). If saved tensors may be used more than once, clears the
        # donated buffer indices in both the forward metadata and the saved context metadata.
        if self.lazy_backward_info is None:
            raise AssertionError("lazy_backward_info must not be None")
        if not isinstance(self.lazy_backward_info, AutogradLazyBackwardCompileInfo):
            raise AssertionError(
                "expected AutogradLazyBackwardCompileInfo, "
                f"got {type(self.lazy_backward_info)}"
            )

        if (
            hasattr(self.lazy_backward_info, "saved_context")
            and self.lazy_backward_info.saved_context is not None
        ):
            if not isinstance(self.lazy_backward_info.saved_context, TracingContext):
                raise AssertionError(
                    f"expected TracingContext, got {type(self.lazy_backward_info.saved_context)}"
                )
            ddp_ctx = self.lazy_backward_info.saved_context.ddp_optimizer_ctx
            if ddp_ctx is not None:
                if ddp_ctx.curr_bucket < 0:
                    raise AssertionError(
                        "expected same # of fw and bw compiles, "
                        f"but found bucket {ddp_ctx.curr_bucket}"
                    )
                curr_fw_meta = ddp_ctx.metadata_per_bucket[ddp_ctx.curr_bucket]
                # Note [DDPOptimizer and fw_metadata]
                # When using the DDPOptimizer, we have a single dynamo graph (and TracingContext),
                # but multiple AOTDispatcher graph.
                #
                # One consequence is that there will be **multiple** fw_metadata objects, one per AOT graph,
                # which we stash the fw_metadata on the TracingContext.
                #
                # Normally what happens is that as we compile AOT graphs 1...N, we clobber the fw_metadata
                # for graph i-1 when we start running AOT for graph i.
                # Ordinarily this is fine, because inductor no longer needs the metadata from graph i-1.
                #
                # However, this is a problem for lazy compilation of the backward. During backward compilation,
                # we compile the backward lazily at backward runtime, meaning that we will first compile
                # backward graph N, N-1, ..., 1.
                # We need to ensure that at the time inductor compiles bw graph N-1, it can access
                # the corresponding fw_metadata for graph N-1.
                #
                # We do this by stashing a DDPOptimizerContext, which tracks:
                # - the metadata of all N graphs
                # - the graph we are currently compiling in our DDPOptimizer region.
                ddp_ctx.curr_bucket -= 1
                self.lazy_backward_info.saved_context.fw_metadata = curr_fw_meta

        if not saved_tensors_use_once:
            self.fw_metadata.bw_donated_idxs = []
            # Update bw_donated_idxs if using lazy_backward_info from `aot_dispatch_autograd`
            if (
                hasattr(self.lazy_backward_info, "saved_context")
                and hasattr(self.lazy_backward_info.saved_context, "fw_metadata")
                and hasattr(
                    self.lazy_backward_info.saved_context.fw_metadata,  # type: ignore[union-attr]
                    "bw_donated_idxs",
                )
            ):
                self.lazy_backward_info.saved_context.fw_metadata.bw_donated_idxs = (  # type: ignore[union-attr]
                    # pyrefly: ignore [implicit-any]
                    []
                )


def _codegen_backward_prologue(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    codegen_unwrap_fn: Callable[..., Any] | None,
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Generates, from forward metadata, the function that prepares backward inputs.
    # The generated function takes saved tensors, saved symints, saved opaque objects and the incoming
    # gradients list. It raises (via a check) if called under a functorch transform, and if the forward
    # ran non-deterministic but deterministic algorithms are now on; asserts the gradient count equals
    # mutated inputs plus outputs plus intermediate bases; selects only gradients that survive (mutated
    # inputs with grad, outputs that need grad for backward, and all intermediate bases); clears the
    # incoming list; builds the full argument list (symints, saved tensors, opaque objects, gradients,
    # optional backward token placeholders, optional rng state); then coerces tangents to the expected
    # metadata/layout (flattening and unwrapping subclasses if subclass metadata exists), and raises a
    # runtime error if input mutations requiring grad exist in backward while grad mode is on
    # (create_graph). Returns the built function. Arg codegen_unwrap_fn optionally supplies the
    # subclass unwrap function.
    from .codegen import PySourceBuilder

    num_mutated_runtime_inps = fw_metadata.num_mutated_inp_runtime_indices
    num_outputs = fw_metadata.num_outputs
    num_intermediate_bases = fw_metadata.num_intermediate_bases
    expected_grad_outs = num_outputs + num_mutated_runtime_inps + num_intermediate_bases
    deterministic = fw_metadata.deterministic
    num_backward_tokens = fw_metadata.num_backward_tokens
    is_rng_op_functionalized = fw_metadata.is_rng_op_functionalized
    has_subclass = maybe_subclass_meta is not None
    has_mutations_in_bw = bool(
        fw_metadata.indices_of_inputs_that_requires_grad_with_mutations_in_bw
    )
    input_info = fw_metadata.input_info

    inp_surviving: list[int] = []
    for i, info_idx in enumerate(fw_metadata.mutated_inp_runtime_indices):
        if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad:
            inp_surviving.append(i)

    out_surviving: list[int] = []
    for i, info in enumerate(fw_metadata.output_info):
        if (
            info.output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            and issubclass(info.raw_type, torch.Tensor)
            and info.requires_grad_for_backward
        ):
            out_surviving.append(num_mutated_runtime_inps + i)

    intermediate_start = num_mutated_runtime_inps + num_outputs
    all_surviving = (
        inp_surviving
        + out_surviving
        + list(range(intermediate_start, expected_grad_outs))
    )
    num_flat_bw_args_with_grads = len(all_surviving)

    buf = PySourceBuilder(
        "_backward_prologue",
        args="ctx_saved_tensors, ctx_symints, ctx_opaque_objects, flat_args",
        artifact_name="backward_prologue",
    )
    buf.bind(torch=torch)

    with buf.indent():
        buf.writeline("stack = torch._C._functorch.peek_interpreter_stack()")
        buf.writeline(
            "torch._check(stack is None, lambda: ("
            "\"It looks like you're trying to call a compiled backward "
            'function within vmap/grad/vjp, "'
            " \"which isn't supported. Try wrapping vmap inside "
            'torch.compile, or skip compiling the "'
            ' "backward function."),)'
        )

        if deterministic is not None and not deterministic:
            buf.writeline("_gd = torch.are_deterministic_algorithms_enabled()")
            buf.writeline(
                "torch._check(not _gd, lambda: ("
                "'This compiled backward function is being run with '"
                " 'torch.use_deterministic_algorithms(True), '"
                " 'but it was previously generated during the forward function"
                " while '"
                " 'torch.use_deterministic_algorithms(False) was set.'))"
            )

        buf.writeline(f"if len(flat_args) != {expected_grad_outs}:")
        with buf.indent():
            buf.writeline(
                f"raise AssertionError("
                f"f'expected {expected_grad_outs} grad_outs, "
                f"got {{len(flat_args)}}')"
            )

        if all_surviving:
            items = ", ".join(f"flat_args[{i}]" for i in all_surviving)
            buf.writeline(f"_bw_grads = [{items}]")
        else:
            buf.writeline("_bw_grads = []")

        buf.writeline("if isinstance(flat_args, list):")
        with buf.indent():
            buf.writeline("flat_args.clear()")

        buf.writeline("_n_saved = len(ctx_saved_tensors)")

        parts = [
            "*ctx_symints",
            "*ctx_saved_tensors",
            "*ctx_opaque_objects",
            "*_bw_grads",
        ]
        if num_backward_tokens > 0:
            parts.append(f"*([None] * {num_backward_tokens})")
        if is_rng_op_functionalized:
            buf.bind(_get_rng_state_=CUDARngStateHelper.get_torch_state_as_tuple)
            parts.append("*_get_rng_state_()")
        buf.writeline(f"all_args = [{', '.join(parts)}]")
        buf.writeline("del ctx_saved_tensors")

        buf.writeline("_ts = len(ctx_symints) + _n_saved + len(ctx_opaque_objects)")
        buf.writeline(f"_te = _ts + {num_flat_bw_args_with_grads}")

        buf.bind(
            _process_tangent_=AOTDispatchAutograd.process_runtime_tangent,
            _tangent_metas_=fw_metadata.subclass_tangent_meta,
            _tangent_descs_=fw_metadata.traced_tangents_descs,
            _compile_id_=fw_metadata.compile_id_str,
            _stack_traces_=fw_metadata.tangent_source_stack_traces or (),
        )

        if has_subclass:
            buf.bind(_chain_=itertools.chain.from_iterable)

            buf.writeline("_tangents = all_args[_ts:_te]")
            buf.writeline(
                f"if len(_tangents) != {len(fw_metadata.subclass_tangent_meta)}:"
            )
            with buf.indent():
                buf.writeline(
                    "raise RuntimeError("
                    "'The grad inputs should be same number as forward output tangents')"
                )
            buf.writeline(
                "_fpt = list(_chain_("
                "_process_tangent_(t, m, idx, desc, _compile_id_, "
                "_stack_traces_[idx] if _stack_traces_ else None)[1] "
                "for idx, (t, m, desc) in enumerate("
                "zip(_tangents, _tangent_metas_, _tangent_descs_))))"
            )

            if codegen_unwrap_fn is not None:
                buf.bind(_unwrap_=codegen_unwrap_fn)
            else:
                buf.bind(_unwrap_=_unwrap_no_symints)
            buf.writeline(
                "all_args = _unwrap_(all_args[:_ts]) + _fpt + _unwrap_(all_args[_te:])"
            )
        elif num_flat_bw_args_with_grads > 0:
            buf.writeline(f"for j in range({num_flat_bw_args_with_grads}):")
            with buf.indent():
                buf.writeline("_i = _ts + j")
                buf.writeline(
                    "all_args[_i] = _process_tangent_(all_args[_i], "
                    "_tangent_metas_[j], j, _tangent_descs_[j], _compile_id_, "
                    "_stack_traces_[j] if _stack_traces_ else None)[0]"
                )

        if has_mutations_in_bw:
            buf.writeline("if torch.is_grad_enabled():")
            with buf.indent():
                buf.writeline(
                    "raise RuntimeError("
                    "'aot_autograd does not support input mutations with "
                    "requires_grad in backward for create_graph=True')"
                )

        buf.writeline("return all_args")

    return buf.build()


def _codegen_backward_epilogue(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    codegen_wrap_fn: Callable[..., Any] | None,
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Generates the function that post-processes backward outputs. The generated
    # function drops trailing backward-token outputs, if rng ops are functionalized pops the last output
    # as the new rng offset and stores it (asserting exactly one rng offset output), converts to a tuple,
    # and with subclass metadata rewraps gradients into tensor subclasses (using a generated wrap function
    # when present and no override is passed, else the generic wrapper). Returns the built function.
    # Raises an assertion error if rng offset count is not one or grad input metadata is missing.
    from .codegen import PySourceBuilder

    num_bw_tokens = fw_metadata.num_backward_tokens
    is_rng = fw_metadata.is_rng_op_functionalized
    has_subclass = maybe_subclass_meta is not None

    if has_subclass:
        args = "out, make_subclass_override=None"
    else:
        args = "out"
    buf = PySourceBuilder(
        "_backward_epilogue", args=args, artifact_name="backward_epilogue"
    )

    with buf.indent():
        if num_bw_tokens > 0:
            buf.writeline(f"out = out[:-{num_bw_tokens}]")

        if is_rng:
            if fw_metadata.num_outputs_rng_offset != 1:
                raise AssertionError(
                    f"expected num_outputs_rng_offset == 1, got {fw_metadata.num_outputs_rng_offset}"
                )
            buf.bind(_set_offset_=CUDARngStateHelper.set_new_offset)
            buf.writeline("_oi = len(out) - 1")
            buf.writeline("_set_offset_(out[_oi])")
            buf.writeline("out = out[:_oi] + out[_oi + 1:]")

        buf.writeline("out = tuple(out)")

        if has_subclass:
            buf.bind(_wrap_subclasses_=wrap_tensor_subclasses)
            if (
                maybe_subclass_meta.grad_input_metas is None
            ):  # pyrefly: ignore [missing-attribute]
                raise AssertionError("grad_input_metas must not be None")
            buf.bind(
                _grad_input_metas_=maybe_subclass_meta.grad_input_metas
            )  # pyrefly: ignore [missing-attribute]
            if codegen_wrap_fn is not None:
                buf.bind(_wrap_=codegen_wrap_fn)
                buf.writeline("if make_subclass_override is None:")
                with buf.indent():
                    buf.writeline("return _wrap_(out)")
            buf.writeline(
                "return _wrap_subclasses_(out, "
                "subclass_metas=_grad_input_metas_, "
                "included_subclass_symints=True, is_runtime=True, "
                "make_subclass_override=make_subclass_override)"
            )
        else:
            buf.writeline("return out")

    return buf.build()


def _codegen_compiled_forward(
    fw_metadata: ViewAndMutationMeta,
    backward_state_indices: list[int],
    disable_amp: bool,
    num_rng: int,
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Generates the autograd forward function body. The generated function takes
    # the context, args, rng-add, save, finalize and compiled forward callables. It optionally records the
    # backward state argument on the context (asserting its type), adds rng generator args when used, calls
    # the compiled forward on a copy of the args (with autocast disabled if requested), normalizes the
    # result to a list, saves tensors for backward and returns the finalized outputs. Returns the built
    # function.
    from .codegen import PySourceBuilder

    buf = PySourceBuilder(
        "_compiled_forward",
        args="ctx, args, _rng_add_, _save_, _finalize_, _compiled_fw_",
        artifact_name="compiled_function_forward",
    )
    buf.bind(torch=torch, BackwardState=BackwardState)

    with buf.indent():
        if backward_state_indices:
            idx = backward_state_indices[0]
            buf.writeline(f"_bw_state = args[{idx}]")
            buf.writeline("if not isinstance(_bw_state, BackwardState):")
            with buf.indent():
                buf.writeline(
                    "raise AssertionError("
                    "f'expected BackwardState, got {type(_bw_state)}')"
                )
            buf.writeline("ctx._compiled_autograd_backward_state = _bw_state")

        if num_rng > 0:
            buf.writeline("args = _rng_add_(ctx, args)")

        if disable_amp:
            buf.bind(_DisableAutocast_=torch._C._DisableAutocast)
            buf.writeline("with _DisableAutocast_():")
            with buf.indent():
                buf.writeline("fw_outs = _compiled_fw_(list(args))")
                _codegen_normalize_as_list(buf, "fw_outs", indent_level=2)
        else:
            buf.writeline("fw_outs = _compiled_fw_(list(args))")
            _codegen_normalize_as_list(buf, "fw_outs", indent_level=1)

        buf.writeline("_save_(ctx, fw_outs)")
        buf.writeline("return _finalize_(ctx, fw_outs)")

    return buf.build()


def _codegen_compiled_backward(
    num_rng: int,
    num_tensors_no_vc_check: int | None,
    inputs_require_grad: bool,
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Generates the autograd backward function body. The generated function takes
    # the gradient arguments, context, prologue, rng-add, implementation, epilogue and double-backward
    # callables. It requires gradients passed as a single mutable list (assertion error otherwise), gathers
    # saved tensors (including those without version counter check when any), runs the prologue, adds
    # rng args if used, defines an implementation closure running the implementation then the epilogue
    # (wrapped to run in the saved compiled-autograd context if one is in thread local state), and if
    # grad mode is on and some tensor could require grad (always, when any input requires grad) routes through
    # the double-backward path, else runs the implementation. Returns the built function.
    from .codegen import PySourceBuilder

    buf = PySourceBuilder(
        "_compiled_backward",
        args="_flat_args_, _ctx_, _prologue_,"
        " _rng_add_, _impl_, _epilogue_, _double_bw_",
        artifact_name="compiled_function_backward",
    )
    buf.bind(torch=torch, functools=functools)

    with buf.indent():
        buf.writeline(
            "if len(_flat_args_) != 1 or not isinstance(_flat_args_[0], list):"
        )
        with buf.indent():
            buf.writeline(
                "raise AssertionError("
                "'Compiled backward expects grads as a single mutable list '"
                " f'argument, but got {len(_flat_args_)} args. '"
                " 'Grads must be passed as [grad0, grad1, ...] to allow '"
                " 'freeing individual grads mid-backward.')"
            )
        buf.writeline("grad_args = _flat_args_[0]")
        buf.writeline("del _flat_args_")

        if num_tensors_no_vc_check is not None and num_tensors_no_vc_check > 0:
            buf.writeline(
                "_saved = (*_ctx_.saved_tensors, *_ctx_._tensors_no_vc_check)"
            )
        else:
            buf.writeline("_saved = _ctx_.saved_tensors")

        buf.writeline(
            "all_args = _prologue_(_saved, _ctx_.symints,"
            " _ctx_.opaque_objects, grad_args)"
        )
        buf.writeline("del _saved")

        if num_rng > 0:
            buf.writeline("_rng_add_(_ctx_, all_args)")

        buf.writeline("def impl_fn(double_ctx=None):")
        with buf.indent():
            buf.writeline("out = _impl_(_ctx_, all_args)")
            buf.writeline("return _epilogue_(out)")

        buf.writeline(
            "if (torch._C._is_key_in_tls('context')"
            " and (_cc := torch._C._get_obj_in_tls('context')) is not None):"
        )
        with buf.indent():
            buf.writeline("impl_fn = functools.partial(_cc.copy().run, impl_fn)")

        if inputs_require_grad:
            buf.writeline("if torch.is_grad_enabled():")
        else:
            buf.writeline(
                "_ng = torch.is_grad_enabled() and any("
                "t.requires_grad for t in all_args if isinstance(t, torch.Tensor))"
            )
            buf.writeline("if _ng:")
        with buf.indent():
            buf.writeline("return _double_bw_(_ctx_, impl_fn, all_args)")
        buf.writeline("return impl_fn()")

    return buf.build()


@dataclass
class _AOTDispatchAutogradFunctionFactory:
    # [CLASS DOC] Factory that turns an AOTDispatchAutogradCompileSpec into a torch autograd Function
    # subclass representing the compiled forward/backward pair. Holds the spec. Method build does the work.
    spec: AOTDispatchAutogradCompileSpec

    def build(self) -> type[torch.autograd.Function]:
        # [FUNCTION DOC] Builds and returns a CompiledFunction autograd Function class. Records the current
        # compile id string on the forward metadata; creates state helper objects (saved state, forward epilogue,
        # rng tracker, backward compiler); generates code for backward prologue and epilogue, forward, backward,
        # and the raw-return transform; replaces the forward epilogue finalize method on its instance with a
        # generated version (monkey patch); then defines the autograd Function subclass whose class attributes
        # expose the compiled functions, metadata and generated functions. Side effects: mutates the metadata
        # compile id; code generation registers generated sources.
        compile_id = CompileContext.current_compile_id()
        compile_id_str = str(compile_id) if compile_id is not None else None
        self.spec.fw_metadata.compile_id_str = compile_id_str

        saved_state = _AutogradSavedState(self.spec.fw_metadata)
        forward_epilogue = _AutogradForwardEpilogue(self.spec.fw_metadata)
        rng_state = _AutogradRngStateTracker(
            num_rng=self.spec.fw_metadata.num_graphsafe_rng_states,
            graphsafe_idx=self.spec.fw_metadata.graphsafe_rng_state_index,
            device=self.spec.fw_metadata.graphsafe_rng_device,
        )
        backward_compiler = _AutogradBackwardCompiler(
            compiled_bw=self.spec.compiled_bw_func,
            lazy_backward_info=self.spec.lazy_backward_info,
            disable_amp=self.spec.disable_amp,
            bw_compiler=self.spec.bw_compiler,
            aot_config=self.spec.aot_config,
            fw_metadata=self.spec.fw_metadata,
            try_save_cache_entry=self.spec.try_save_cache_entry,
        )

        compiled_fw_func = self.spec.compiled_fw_func
        compiled_bw_func = self.spec.compiled_bw_func
        maybe_subclass_meta = self.spec.maybe_subclass_meta
        num_symints_saved_for_bw_ = self.spec.num_symints_saved_for_bw
        backward_state_indices = self.spec.backward_state_indices
        disable_amp = self.spec.disable_amp
        lazy_backward_info = self.spec.lazy_backward_info
        aot_config = self.spec.aot_config
        fw_metadata = self.spec.fw_metadata

        _codegen_bw_unwrap_fn = None
        _codegen_bw_wrap_fn = None
        if maybe_subclass_meta is not None:
            from .subclass_codegen import codegen_backward_subclass_fns

            _codegen_bw_unwrap_fn, _codegen_bw_wrap_fn = codegen_backward_subclass_fns(
                grad_input_metas=maybe_subclass_meta.grad_input_metas,
            )

        _codegen_bw_prologue = _codegen_backward_prologue(
            fw_metadata,
            maybe_subclass_meta,
            _codegen_bw_unwrap_fn,
        )

        _codegen_bw_epilogue = _codegen_backward_epilogue(
            fw_metadata,
            maybe_subclass_meta,
            _codegen_bw_wrap_fn,
        )

        _codegen_fwd = _codegen_compiled_forward(
            fw_metadata,
            backward_state_indices,
            disable_amp,
            rng_state.num_rng,
        )

        _codegen_bwd = _codegen_compiled_backward(
            rng_state.num_rng,
            fw_metadata.num_tensors_saved_with_no_vc_check,
            any(inp.requires_grad for inp in fw_metadata.input_info),
        )

        # Codegen for CompiledFunction.forward: emit straight-line TensorAlias
        # wrapping, _unsafe_view, and non-differentiable output collection with
        # all indices resolved at compile time.
        num_mutated_runtime_inps = fw_metadata.num_mutated_inp_runtime_indices
        num_outputs = fw_metadata.num_outputs
        num_outputs_aliased = fw_metadata.num_outputs_aliased

        from .codegen import PySourceBuilder

        buf = PySourceBuilder(
            "_transform_raw_returns",
            args="raw_returns",
            artifact_name="compiled_fn_wrapper",
        )
        buf.bind(TensorAlias=TensorAlias, torch=torch, Tensor=Tensor)

        with buf.indent():
            for i, idx in enumerate(fw_metadata.mutated_inp_runtime_indices):
                info = fw_metadata.input_info[idx]
                if info.mutates_metadata and not info.mutates_data:
                    buf.writeline(f"raw_returns[{i}] = TensorAlias(raw_returns[{i}])")

            if fw_metadata.num_unsafe_view_outputs > 0:
                for idx in fw_metadata.unsafe_view_out_indices:
                    ri = num_mutated_runtime_inps + idx
                    buf.writeline(f"_o = raw_returns[{ri}]")
                    buf.writeline(
                        f"raw_returns[{ri}] = torch.ops.aten._unsafe_view(_o, _o.shape)"
                    )

            if num_outputs_aliased > 0:
                for idx in fw_metadata.aliased_out_indices:
                    ri = num_mutated_runtime_inps + idx
                    buf.writeline(f"raw_returns[{ri}] = TensorAlias(raw_returns[{ri}])")

            # Non-differentiable output collection: build a list of specific indices
            # at compile time rather than iterating at runtime.
            _non_diff_indices: list[int] = []
            _returns_meta = [
                x
                for x in fw_metadata.input_info
                if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
            ] + list(fw_metadata.output_info)
            for i, meta in enumerate(_returns_meta):
                if (
                    i < num_mutated_runtime_inps + num_outputs
                    and not meta.requires_grad
                ):
                    _non_diff_indices.append(i)
            if _non_diff_indices:
                checks = " + ".join(
                    f"([raw_returns[{i}]] if isinstance(raw_returns[{i}], Tensor) else [])"
                    for i in _non_diff_indices
                )
                buf.writeline(f"non_diff = {checks}")
            else:
                buf.writeline("non_diff = []")
            buf.writeline("return non_diff")

        _codegen_transform_raw_returns: Callable[..., list[Any]] = buf.build()  # type: ignore[assignment]

        # Monkey-patch forward_epilogue.finalize to use codegen'd transform
        def _codegen_finalize(ctx: Any, fw_outs: Any) -> tuple[Any, ...]:
            # [FUNCTION DOC] HELPER: Replacement for the epilogue finalize method. Takes the forward outputs, keeps
            # the first num_forward_returns, applies the generated transform (wrapping aliases, unsafe views) which
            # yields the tensors that need no gradient, optionally runs debug assertions (assertion errors), marks
            # the non-requiring-grad outputs as non differentiable on ctx, disables materializing non diff grads,
            # snapshots external objects on ctx, and returns the outputs as a tuple. Mutates ctx and, in place, the
            # raw returns list.
            num_forward_returns = fw_metadata.num_forward_returns
            raw_returns = list(fw_outs[:num_forward_returns])
            fw_outs_not_requiring_grad = _codegen_transform_raw_returns(raw_returns)
            if config.debug_assert:
                if num_mutated_runtime_inps > 0:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                    mut_inp_infos = [
                        x
                        for x in fw_metadata.input_info
                        if x.mutates_data or x.mutates_metadata
                    ]
                    if len(user_mutated_inputs_raw) != len(mut_inp_infos):
                        raise AssertionError(
                            f"expected len(user_mutated_inputs_raw) == len(mut_inp_infos), "
                            f"got {len(user_mutated_inputs_raw)} != {len(mut_inp_infos)}"
                        )
                if num_outputs_aliased > 0:
                    intermediates_raw = raw_returns[
                        num_mutated_runtime_inps + num_outputs :
                    ]
                    if any(isinstance(x, TensorAlias) for x in intermediates_raw):
                        raise AssertionError(
                            "expected no TensorAlias in intermediates_raw"
                        )
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
            ctx._materialize_non_diff_grads = False
            _snapshot_external_objects(ctx)

            return tuple(raw_returns)

        forward_epilogue.finalize = _codegen_finalize  # type: ignore[method-assign]

        class CompiledFunction(torch.autograd.Function):
            # [CLASS DOC] Autograd Function for a compiled graph produced by AOTDispatch. Class attributes (bound at
            # build time): compiled forward and backward callables, forward metadata, subclass metadata, count of
            # symints saved for backward, aot id, lazy backward info, generated prologue/epilogue/forward/backward
            # functions, and a flag that backward receives grads as a boxed list. forward and backward delegate to
            # generated functions; backward compiles lazily on first use.
            compiled_fw = compiled_fw_func
            compiled_bw = compiled_bw_func
            metadata: ViewAndMutationMeta = fw_metadata  # type: ignore[assignment]
            maybe_subclass_metadata: SubclassMeta | None = maybe_subclass_meta
            num_symints_saved_for_bw = num_symints_saved_for_bw_
            _aot_id = aot_config.aot_id
            _lazy_backward_info = lazy_backward_info
            _bw_prologue_fn = _codegen_bw_prologue
            _bw_epilogue_fn = _codegen_bw_epilogue
            _fwd_fn = _codegen_fwd
            _bwd_fn = _codegen_bwd
            boxed_grads_call = True

            @staticmethod
            def _compiled_autograd_key(ctx: Any) -> tuple[Any, ...]:
                # [FUNCTION DOC] HELPER: Returns the key used by compiled autograd to identify this function call: the
                # autograd function id from the context followed by the saved symints.
                return (ctx._autograd_function_id, *ctx.symints)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx: Any, *deduped_flat_tensor_args: Any) -> Any:
                # [FUNCTION DOC] Autograd forward: delegates to the generated forward function with the context, the
                # inputs, the rng argument adder, the saved-state saver, the forward epilogue finalizer and the compiled
                # forward. Returns the forward outputs for autograd.
                return CompiledFunction._fwd_fn(
                    ctx,
                    deduped_flat_tensor_args,
                    rng_state.add_forward_args,
                    saved_state.save_from_forward,
                    forward_epilogue.finalize,
                    CompiledFunction.compiled_fw,
                )

            @staticmethod
            def backward(ctx: Any, *flat_args: Any) -> tuple[Any, ...]:
                # [FUNCTION DOC] Autograd backward: delegates to the generated backward function with the incoming
                # gradients, the context, prologue, rng adder, backward implementation, epilogue and double backward
                # handler. Returns the tuple of input gradients.
                return CompiledFunction._bwd_fn(
                    flat_args,
                    ctx,
                    CompiledFunction._bw_prologue_fn,
                    rng_state.add_backward_args,
                    CompiledFunction._backward_impl,
                    CompiledFunction._bw_epilogue_fn,
                    CompiledFunction._double_backward,
                )

            @staticmethod
            def _double_backward(
                ctx: Any, impl_fn: Callable[..., Any], all_args: list[Any]
            ) -> Any:
                # [FUNCTION DOC] HELPER: Used when the backward is run with grad mode enabled (create_graph). Defines an
                # inner autograd Function whose forward runs the backward implementation and whose backward raises a
                # runtime error saying double backward is unsupported. If no tensor argument requires grad, prepends a
                # dummy empty tensor requiring grad so outputs get a grad_fn. Copies the compiled autograd key onto the
                # inner class and returns the result of applying it to the args.
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    # [CLASS DOC] HELPER: Inner autograd Function that only keeps the graph connected for create_graph
                    # backward and rejects actual double backward. Carries the aot id for Dynamo.
                    # CompiledFunctionBackward is not yet supported in dynamo skipfiles
                    _aot_id = aot_config.aot_id

                    @staticmethod
                    # pyrefly: ignore [bad-override]
                    def forward(double_ctx: Any, *unused_args: Any) -> Any:
                        # [FUNCTION DOC] HELPER: Runs the wrapped backward implementation with the given context and returns
                        # its result; the extra tensor arguments are ignored.
                        return impl_fn(double_ctx)

                    @staticmethod
                    def backward(ctx: Any, *args: Any) -> None:
                        # [FUNCTION DOC] HELPER: Always raises a runtime error stating that double backward is not supported
                        # with aot_autograd.
                        raise RuntimeError(
                            "torch.compile with aot_autograd does not currently support double backward"
                        )

                CompiledFunctionBackward._compiled_autograd_key = (  # type: ignore[method-assign]
                    CompiledFunction._compiled_autograd_key
                )

                # Saved tensors are detached (forward runs under no-grad), so no
                # real arg requires grad.  Prepend a dummy requires_grad=True
                # input so CompiledFunctionBackward.apply attaches a grad_fn and
                # gradient outputs report requires_grad=True for create_graph=True.
                if not any(
                    t.requires_grad for t in all_args if isinstance(t, torch.Tensor)
                ):
                    all_args = [torch.empty(0, requires_grad=True)] + all_args
                return CompiledFunctionBackward.apply(*all_args)

            @staticmethod
            def _backward_impl(ctx: Any, all_args: list[Any]) -> Any:
                # [FUNCTION DOC] Runs the compiled backward. Raises an assertion error if backward state is needed (that
                # requires compiled autograd). Clears saved tensors on the context if allowed, determines whether saved
                # tensors are used only once from the graph task keep-graph flag, gets or lazily compiles the backward
                # through the backward compiler and stores it on the class, raises (via a check) if donated buffers are
                # enabled and non-empty but the graph may run more than once, re-registers external objects from the
                # context as weak references in the global registry, then calls the compiled backward with the
                # arguments stolen and autocast disabled per config. Returns its outputs. Compiled autograd replaces
                # this method.
                # compiled autograd reimplements this function at proxy_call_aot_backward
                if backward_state_indices:
                    raise AssertionError("BackwardState requires CompiledAutograd")
                ctx.maybe_clear_saved_tensors()

                saved_tensors_use_once = (
                    not torch._C._autograd._get_current_graph_task_keep_graph()
                )
                compiled_bw = backward_compiler.get_or_compile(
                    saved_tensors_use_once=saved_tensors_use_once
                )
                CompiledFunction.compiled_bw = compiled_bw

                if (
                    torch._functorch.config.donated_buffer
                    and not saved_tensors_use_once
                    and fw_metadata.bw_donated_idxs != []
                ):
                    torch._check(
                        False,
                        lambda: (
                            "This backward function was compiled with non-empty donated "
                            "buffers which requires create_graph=False and retain_graph=False. "
                            "Please keep backward(create_graph=False, retain_graph=False) "
                            "across all backward() function calls, or set "
                            "torch._functorch.config.donated_buffer=False to disable "
                            "donated buffer."
                        ),
                    )

                for idx, obj in getattr(ctx, "_external_objects", {}).items():
                    index_to_external_object_weakref[idx] = weakref.ref(obj)

                return call_func_at_runtime_with_args(
                    compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

        return CompiledFunction


# This is wrapped in a class just for namespacing purposes
# No need to make it into an actual CompilerWrapper because it doesn't fit the abstract as cleanly
class AOTDispatchAutograd:
    # [CLASS DOC] Namespace class (only static methods, no state) for AOTDispatch autograd runtime pieces:
    # tangent validation/coercion for backward and creation of the final compiled function from a spec.
    @staticmethod
    def _raise_tangent_metadata_error(
        expected_type: type | None,
        expected_meta: Any,
        runtime_type: type,
        runtime_meta: Any,
        orig_x: torch.Tensor,
        tangent_idx: int | None,
        tangent_desc: Any | None,
        compile_id_str: str | None,
        tangent_stack_trace: str | None,
    ) -> RuntimeError:
        # [FUNCTION DOC] HELPER: Builds (does not raise) a runtime error describing a tangent whose metadata or
        # type did not match what compile time guessed. If a subclass was expected but a plain tensor was
        # received, the message explains the likely cause (backward called on only some outputs) and includes the
        # tangent index, the originating output, the compile graph id and the forward stack trace when
        # available. Otherwise the message shows expected and actual metadata and types and the tangent shape,
        # and requires the subclass to implement the coercion method. Caller raises the returned error.
        expected_subclass_got_plain_tensor = (
            expected_type is not None
            and expected_type is not torch.Tensor
            and runtime_type is torch.Tensor
        )
        if expected_subclass_got_plain_tensor:
            tangent_msg = ""
            if tangent_idx is not None:
                tangent_msg = f" (tangent index: {tangent_idx})"

            output_hint = ""
            if tangent_desc is not None:
                from .descriptors import PlainAOTOutput, TangentAOTInput

                if isinstance(tangent_desc, TangentAOTInput) and isinstance(
                    tangent_desc.output, PlainAOTOutput
                ):
                    idx = tangent_desc.output.idx
                    output_hint = f"\n\nThe problematic output is: forward output at index {idx} (0-indexed)"
                else:
                    output_hint = (
                        f"\n\nThe problematic output is: {tangent_desc.expr()}"
                    )

            graph_hint = ""
            if compile_id_str is not None:
                graph_hint = (
                    f"\n\nThis error occurred in compiled graph [{compile_id_str}]."
                )

            stack_trace_hint = ""
            if tangent_stack_trace is not None:
                stack_trace_hint = (
                    f"\n\nThe forward output was created here:\n{tangent_stack_trace}"
                )

            return RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
Expected a {expected_type.__name__} tangent but got a plain Tensor{tangent_msg}.
This happens when a compiled function returns multiple outputs that
require gradients, but .backward() is only called on some of them.
To fix: call .detach() on forward outputs you don't need gradients for.{output_hint}{graph_hint}{stack_trace_hint}

This error is also more likely to occur if your compiled model is suffering
from a large number of graph breaks. For more advice on finding and fixing
graph breaks, see:
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html

For more info about this error, see:
https://github.com/pytorch/pytorch/issues/172556"""
            )
        else:
            return RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
Expected: {expected_meta} (type {expected_type}),
got: {runtime_meta} (type {runtime_type}), shape: {orig_x.shape}.
Your tensor subclass must implement __coerce_same_metadata_as_tangent__."""
            )

    @staticmethod
    def process_runtime_tangent(
        x: Any,
        meta: PlainTensorMeta | SubclassCreationMeta,
        tangent_idx: int | None = None,
        tangent_desc: Any | None = None,
        compile_id_str: str | None = None,
        tangent_stack_trace: str | None = None,
    ) -> tuple[Any, list[Any]]:
        # [FUNCTION DOC] Coerces one runtime tangent (gradient) to the form the compiled backward expects and
        # returns it with the list of its flattened leaf tensors. Non-tensors pass through. Fake tensors only
        # get their memory format coerced. Otherwise: determines expected and runtime subclass type and
        # metadata, coerces the type/metadata via the subclass coercion hook (raising the error built by the
        # helper if it is not possible), coerces memory format, and for traceable wrapper subclasses recurses
        # over inner attributes (opaque attributes are passed through as leaves), replacing inner tensors on
        # the object in place when they changed. Raises assertion errors on inconsistent metadata (missing
        # memory format, wrong meta kind, attribute count mismatch).
        if not isinstance(x, torch.Tensor):
            return x, [x]

        if is_fake_tensor(x):
            if not meta.memory_format:
                raise AssertionError(
                    "meta.memory_format must not be None for FakeTensor"
                )
            x = coerce_to_expected_memory_format(x, meta.memory_format)
            return x, [x]

        expected_type: type | None = torch.Tensor
        expected_meta = None
        if isinstance(meta, SubclassCreationMeta):
            expected_type = meta.original_subclass_type
            expected_meta = meta.meta

        runtime_type = type(x)
        # When we're inside compiled autograd's AOTDispatcher step,
        # regular Tensors look like FunctionalTensors.
        # Tensor subclasses still look like Tensor subclasses though.
        if isinstance(x, torch._subclasses.functional_tensor.FunctionalTensor):
            runtime_type = torch.Tensor

        runtime_meta = None
        runtime_subclass_keys: Sequence[str] = []

        if is_traceable_wrapper_subclass(x):
            runtime_subclass_keys, runtime_meta = x.__tensor_flatten__()

        def maybe_coerce(x: torch.Tensor) -> torch.Tensor | None:
            # [FUNCTION DOC] HELPER: Returns the tensor itself when its type and metadata equal the expected ones;
            # returns none if it differs and lacks the coercion method; otherwise returns the result of the
            # tensor's coercion method (passing the expected type too, unless the types already match).
            same_type: bool = expected_type == runtime_type
            same_meta: bool = expected_meta == runtime_meta

            if same_type and same_meta:
                return x

            if not hasattr(x, "__coerce_same_metadata_as_tangent__"):
                return None

            if same_type:
                # Backward Compatibility, as some Subclass impls can have original 1-arg function.
                return x.__coerce_same_metadata_as_tangent__(expected_meta)

            return x.__coerce_same_metadata_as_tangent__(expected_meta, expected_type)

        # Coerce to expected type and metadata
        orig_x = x
        x = maybe_coerce(x)
        if x is None:
            raise AOTDispatchAutograd._raise_tangent_metadata_error(
                expected_type,
                expected_meta,
                runtime_type,
                runtime_meta,
                orig_x,
                tangent_idx,
                tangent_desc,
                compile_id_str,
                tangent_stack_trace,
            )

        # Coerce to expected memory format
        if not meta.memory_format:
            raise AssertionError("meta.memory_format must not be None")
        x = coerce_to_expected_memory_format(x, meta.memory_format)

        if not is_traceable_wrapper_subclass(x):
            return x, [x]

        if not isinstance(meta, SubclassCreationMeta):
            raise AssertionError(f"expected SubclassCreationMeta, got {type(meta)}")
        if orig_x is not x:
            runtime_subclass_keys = x.__tensor_flatten__()[0]

        if len(meta.attrs) != len(runtime_subclass_keys):
            raise AssertionError(
                f"expected len(meta.attrs) == len(runtime_subclass_keys), "
                f"got {len(meta.attrs)} != {len(runtime_subclass_keys)}"
            )
        leaves = []
        for attr, attr_meta in meta.attrs.items():
            if isinstance(attr_meta, OpaqueMeta):
                # Opaques aren't differentiable but occupy a flat arg slot.
                leaves.append(getattr(x, attr))
                continue
            elem = getattr(x, attr)
            new_elem, elem_leaves = AOTDispatchAutograd.process_runtime_tangent(
                elem, attr_meta
            )
            if new_elem is not elem:
                setattr(x, attr, new_elem)
            leaves.extend(elem_leaves)

        return x, leaves

    @staticmethod
    def post_compile(spec: AOTDispatchAutogradCompileSpec) -> Callable[..., Any]:
        # [FUNCTION DOC] Builds the final callable for a compiled training graph: creates the autograd Function
        # class from the spec through the factory, then wraps its apply method in a RuntimeWrapper configured for
        # joint tracing with the spec's detach indices and amp flag. Returns the wrapped callable.
        compiled_function_cls = _AOTDispatchAutogradFunctionFactory(spec).build()
        return RuntimeWrapper(
            indices_of_inps_to_detach=spec.indices_of_inps_to_detach,
            trace_joint=True,
            disable_amp=spec.disable_amp,
        ).post_compile(
            compiled_function_cls.apply,
            spec.aot_config,
            runtime_metadata=spec.fw_metadata,
        )


@dataclass
class DebugAssertWrapper(CompilerWrapper):
    # [CLASS DOC] CompilerWrapper that generates a runtime check on inputs for debugging. State:
    # flat_requires_grad, one entry per flat input: none means the input must not be a tensor, false means the
    # input must not require grad, true means no check.
    flat_requires_grad: list[bool | None] = field(default_factory=list)

    def post_compile(
        self,
        compiled_fn: Callable[..., Any],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Generates a wrapper that checks each argument against the expected requires-grad
        # status: raises an assertion error if an argument expected to be a non-tensor is a tensor, or if an
        # argument expected not to require grad does. Then calls the compiled function with the args.
        # Returns the wrapper (preserving the compiled function's attributes).
        from .codegen import PySourceBuilder

        buf = PySourceBuilder(
            "inner_fn", args="args", artifact_name="debug_assert_wrapper"
        )
        buf.bind(compiled_fn=compiled_fn)
        with buf.indent():
            for i, can_require_grad in enumerate(self.flat_requires_grad):
                if can_require_grad is None:
                    buf.writeline(
                        f"if isinstance(args[{i}], Tensor):"
                        f" raise AssertionError("
                        f"'expected non-Tensor for arg {i}, got Tensor')"
                    )
                elif not can_require_grad:
                    msg_name = buf.bind_value(
                        "_msg",
                        format_guard_bug_msg(
                            aot_config,
                            f"{describe_input(i, aot_config)} would not require grad",
                        ),
                    )
                    buf.writeline(
                        f"if args[{i}].requires_grad: raise AssertionError({msg_name})"
                    )
            buf.writeline("return compiled_fn(args)")

        buf.bind(Tensor=Tensor)

        return buf.build(wrapped_fn=compiled_fn)


def pre_compile(
    wrappers: list[CompilerWrapper],
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
    # [FUNCTION DOC] Runs each wrapper's pre_compile in order, threading the function, args, descriptors and
    # metadata through. Wrappers may mutate themselves. Returns the final function, args, descriptors and
    # metadata.
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """
    for wrapper in wrappers:
        flat_fn, flat_args, flat_args_descs, fw_metadata = wrapper.pre_compile(
            flat_fn, flat_args, flat_args_descs, aot_config, fw_metadata=fw_metadata
        )
    return flat_fn, flat_args, flat_args_descs, fw_metadata


def post_compile(
    wrappers: list[CompilerWrapper],
    compiled_fn: Callable[..., Any],
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> tuple[Callable[..., Any], ViewAndMutationMeta]:
    # [FUNCTION DOC] Applies each wrapper's post_compile to the compiled function in reverse order of the
    # wrapper list. Returns the final compiled function and the unchanged runtime metadata.
    """
    Runs a sequence of wrappers on the given function. Should be called after pre_compile()
    """
    for wrapper in reversed(wrappers):
        compiled_fn = wrapper.post_compile(
            compiled_fn, aot_config, runtime_metadata=runtime_metadata
        )
    return compiled_fn, runtime_metadata


def make_runtime_safe(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
) -> None:
    # [FUNCTION DOC] Makes the forward metadata (and, if present, the subclass metadata's inner forward
    # metadata and subclass grad input metas) safe for caching by calling their runtime-safe conversion.
    # Mutates both arguments.
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
    fw_metadata.make_runtime_safe()
    if maybe_subclass_meta is not None:
        maybe_subclass_meta.fw_metadata.make_runtime_safe()
        if maybe_subclass_meta.grad_input_metas:
            for meta in maybe_subclass_meta.grad_input_metas:
                if isinstance(meta, SubclassCreationMeta):
                    meta.make_runtime_safe()
