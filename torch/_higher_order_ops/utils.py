# mypy: allow-untyped-defs
# [MODULE DOC]
# Shared utilities for higher-order ops (cond, scan, while_loop, map, etc.): autograd-not-
# implemented wrappers, subgraph tracing helpers (make_fx re-entry, fake tracing), alias and
# mutation analysis of subgraphs, metadata consistency checks, forward/backward graph
# construction, compilation-environment setup for HOP eager execution, pytree/mask helpers,
# and HopInstance/schema helpers. Import-time side effects: creates a thread-local flag
# object used to detect HOP-internal compilation; otherwise only definitions.
import contextlib
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, overload, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dispatch.python import suspend_functionalization
from torch._guards import detect_fake_mode
from torch._higher_order_ops.schema import HopSchema
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._subclasses.fake_tensor import (
    FakeTensor,
    is_fake_tensor,
    maybe_get_fake_mode,
)
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    make_fx,
)
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    # [CLASS DOC] Error (a RuntimeError dataclass) raised when a subgraph aliases or mutates
    # inputs in an unsupported way; carries a human readable reason string.
    reason: str


def autograd_not_implemented_inner(
    operator: OperatorBase, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    # [FUNCTION DOC] Calls the operator below autograd and returns its result. If grad mode is
    # on and any tensor argument requires grad, either raises RuntimeError (default) or,
    # when delayed_error is true, wraps each tensor output in a node that raises only when
    # backward reaches it (floating or complex outputs are detached and given requires_grad).
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The Operator to call with the *args and **kwargs with
        op_name: The name of the Operator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the Operator
        kwargs: The keyword arguments to the Operator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the Operator
    """
    with torch._C._AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands = pytree.arg_tree_leaves(*args)
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                def fake_requires_grad(tensor):
                    # [FUNCTION DOC] HELPER: Detaches floating/complex tensors and marks them as requiring grad; others unchanged.
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )
            else:
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")
        return result


def autograd_not_implemented(op: OperatorBase, deferred_error: bool) -> Callable:
    # [FUNCTION DOC] Returns a callable that applies the autograd-not-implemented behavior to op,
    # with errors raised immediately or deferred according to deferred_error.
    def inner(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the inner autograd-not-implemented routine for op.
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    # [FUNCTION DOC] HELPER: If fn is a graph module and node metadata preservation is active,
    # returns a wrapper that runs it with an fx Interpreter (so stack traces propagate);
    # otherwise returns fn unchanged.
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            # [FUNCTION DOC] HELPER: Runs the graph module with an Interpreter under preserved node metadata.
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


def _hop_compile_and_call(fn, args, kwargs=None):
    # [FUNCTION DOC] HELPER: Compiles fn with a full-graph requirement using the HOP compilation
    # environment backend and calls it with args and optional kwargs, returning the result.
    # Temporarily resets the global fullgraph frame counter to zero and restores it afterwards.
    """Compile and call fn with fullgraph=True for HOP eager execution.

    Pre-activates the fullgraph counter so that compile_wrapper treats this as
    a nested compile.  This avoids erroring when a non-infra dispatch mode
    causes the frame to be skipped — the function still executes eagerly within
    compile_wrapper and returns normally.
    """
    from torch._dynamo.eval_frame import set_fullgraph_compiled_frame_count

    with setup_compilation_env() as backend:
        old_count = set_fullgraph_compiled_frame_count(0)
        try:
            return torch.compile(fn, backend=backend, fullgraph=True)(
                *args, **(kwargs or {})
            )
        finally:
            set_fullgraph_compiled_frame_count(old_count)


def _maybe_compile_and_run_fn(fn, *args):
    # [FUNCTION DOC] HELPER: Runs fn on args via compile-and-call when not already inside dynamo
    # compilation; otherwise calls fn directly. Returns the result.
    if not torch.compiler.is_dynamo_compiling():
        return _hop_compile_and_call(fn, args)
    else:
        return fn(*args)


def reenter_make_fx(fn, subgraph_decomp_table=None):
    # [FUNCTION DOC] Returns a function that traces fn as a subgraph inside the currently
    # active make_fx session (optionally with a custom decomposition table) and returns the
    # resulting graph module.
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    @functools.wraps(fn)
    def wrapped(*args):
        # [FUNCTION DOC] HELPER: Traces fn on args as a subgraph of the current tracer and returns
        # the graph module. Raises an assertion error when no make_fx session is active.
        # KNOWN ISSUE: the tracer global is imported once when reenter_make_fx runs, so a
        # session started afterwards is not seen by an already created wrapper.
        if _CURRENT_MAKE_FX_TRACER is None:
            raise AssertionError(
                "Cannot reenter make_fx when we're not under a make_fx tracing session"
            )
        if subgraph_decomp_table is None:
            gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph(
                _maybe_run_with_interpreter(fn), *args
            )
        else:
            gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph_custom_decomp(
                _maybe_run_with_interpreter(fn), subgraph_decomp_table, *args
            )

        return gm

    return wrapped


def _maybe_reenter_make_fx(fn, subgraph_decomp_table=None):
    # [FUNCTION DOC] HELPER: If a make_fx session is active returns the re-entering tracer for fn;
    # otherwise returns a function that runs a fresh make_fx on fn, in fake tracing mode when
    # the inputs carry no fake mode and in real mode (under their fake mode) otherwise.
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    if _CURRENT_MAKE_FX_TRACER is not None:
        return reenter_make_fx(fn, subgraph_decomp_table=subgraph_decomp_table)
    else:

        def _maybe_make_fx_with_fake_mode(fn):
            # [FUNCTION DOC] HELPER: Wraps fn in a function that traces it with make_fx and returns the graph module.
            @functools.wraps(fn)
            def wrapped(*args):
                # [FUNCTION DOC] HELPER: Detects a fake mode from args and traces fn with make_fx accordingly.
                from torch._guards import detect_fake_mode

                fake_mode = detect_fake_mode(args)
                if fake_mode is None:
                    # we creaeta a fake_mode here to make sure we could
                    # trace the graph with data-dependent calls e.g. .item()
                    return make_fx(
                        fn,
                        tracing_mode="fake",
                        decomposition_table=subgraph_decomp_table,
                    )(*args)
                # Tracing with real if all inputs have been fakfied
                return make_fx(fn, decomposition_table=subgraph_decomp_table)(*args)

            return wrapped

        return _maybe_make_fx_with_fake_mode(fn)


def check_meta_consistency(
    lhs_list: list[torch.Tensor | torch.SymInt | int],
    rhs_list: list[torch.Tensor | torch.SymInt | int],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    # [FUNCTION DOC] Verifies that two lists of values (tensors, symints, ints) match pairwise
    # in tensor metadata (optionally including contiguity), device and basic type. Raises
    # UncapturedHigherOrderOpError if lengths differ or any pair differs, with a message
    # listing every differing pair. Returns nothing.
    def diff_meta_pairs(
        lhs_list: list[torch.Tensor | torch.SymInt | int],
        rhs_list: list[torch.Tensor | torch.SymInt | int],
    ) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns descriptions of differing pairs; raises if list lengths differ.
        def diff_meta(
            lhs: torch.Tensor | torch.SymInt | int,
            rhs: torch.Tensor | torch.SymInt | int,
        ) -> str:
            # [FUNCTION DOC] HELPER: Returns a text of metadata differences for tensor pairs, a type
            # mismatch text if kinds differ (int-like vs tensor), or an empty string if equal.
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                return ", ".join(
                    diff_tensor_meta(
                        _extract_tensor_metadata(
                            lhs, include_contiguity=include_contiguity
                        ),
                        _extract_tensor_metadata(
                            rhs, include_contiguity=include_contiguity
                        ),
                        check_grad=False,
                    )
                )
            else:

                def _both_int_types(lhs, rhs):
                    # [FUNCTION DOC] HELPER: True if both values are ints or symints.
                    return isinstance(lhs, (int, torch.SymInt)) and isinstance(
                        rhs, (int, torch.SymInt)
                    )

                def _both_tensor(lhs, rhs):
                    # [FUNCTION DOC] HELPER: True if both values are tensors.
                    return isinstance(lhs, torch.Tensor) and isinstance(
                        rhs, torch.Tensor
                    )

                if not _both_int_types(lhs, rhs) and not _both_tensor(lhs, rhs):
                    return f"type: {lhs} vs {rhs}"

            return ""

        # Manually check the device of lhs and rhs as this field is currently not part of TensorMetadata
        def diff_device(
            lhs: torch.Tensor | torch.SymInt | int,
            rhs: torch.Tensor | torch.SymInt | int,
        ) -> str:
            # [FUNCTION DOC] HELPER: For tensor pairs returns "device" if device type or index differ, else an empty string.
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                if (
                    rhs.device.type == lhs.device.type
                    and rhs.device.index == lhs.device.index
                ):
                    return ""
                else:
                    return "device"
            return ""

        if len(lhs_list) != len(rhs_list):
            raise torch._dynamo.exc.UncapturedHigherOrderOpError(
                f"Expected {lhs_name} and {rhs_name} to have same number of outputs but got lhs:{lhs_list} and rhs:{rhs_list}"
            )
        all_diffs = []
        for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
            if diff := diff_meta(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
            if diff := diff_device(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
        return all_diffs

    if all_diffs := diff_meta_pairs(lhs_list, rhs_list):
        diff_str = "\n".join(all_diffs)
        raise torch._dynamo.exc.UncapturedHigherOrderOpError(
            f"Expected {lhs_name} and {rhs_name} to have same metadata but found:\n{diff_str}"
        )


# Thread-local flag to indicate we're inside HOP internal compilation
import threading


_hop_compile_tls = threading.local()


def _in_hop_compile() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether the current thread is inside HOP internal compilation (false by default).
    return getattr(_hop_compile_tls, "in_hop_compile", False)


@contextmanager
def setup_compilation_env():
    # [FUNCTION DOC] Context manager yielding a dynamo backend for compiling inside HOP or
    # export regions. Sets the thread-local in-HOP-compile flag, applies the compilation
    # config overrides, disables the cache limit, and temporarily removes the pre-dispatch and
    # metadata torch function modes; yields an eager backend that re-enters those modes if
    # any were active, otherwise the plain eager backend. Restores the flag on exit.
    """
    Context manager that sets up proper environment and backend when invoking torch.compile
    inside torch.export region or inside HOP.
    """
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_modes,
    )
    from torch._dynamo.backends.registry import lookup_backend
    from torch.fx.experimental.proxy_tensor import (
        _temp_remove_pre_dispatch_torch_function_mode,
    )

    old_in_hop_compile = getattr(_hop_compile_tls, "in_hop_compile", False)
    _hop_compile_tls.in_hop_compile = True
    try:
        with (
            _set_compilation_env(),
            torch._dynamo.utils.disable_cache_limit(),
            _temp_remove_pre_dispatch_torch_function_mode() as pre_dispatch_mode,
            _temp_remove_metadata_torch_function_mode() as metadata_mode,
        ):
            modes = [
                mode for mode in (pre_dispatch_mode, metadata_mode) if mode is not None
            ]
            if modes:
                yield make_eager_backend_with_torch_function_modes(modes)
            else:
                yield lookup_backend("eager")
    finally:
        _hop_compile_tls.in_hop_compile = old_in_hop_compile


@contextmanager
def _set_compilation_env():
    # [FUNCTION DOC] HELPER: Context manager that turns off the fx-tracing flag and sets dynamo
    # config to allow empty graphs and capture scalar outputs, restoring all three on exit.
    _old_is_tracing = torch.fx._symbolic_trace._get_is_fx_tracing()
    _old_allow_empty_graphs = torch._dynamo.config.allow_empty_graphs
    _old_capture_scalar_outputs = torch._dynamo.config.capture_scalar_outputs
    # The issue is tracked in https://github.com/pytorch/pytorch/issues/144360: when dynamo finds
    # the top-level frame produces no graph, the default behavior is to fallback to eager.
    # Then when it encounters an inner function, it will try to trace that function again, which is unnecessary.
    # For while_loop, during inspecting the inner call, we trace into the python dispatcher
    # logic, which is not tracable as of today. So the proper fix can be either 1. allow dispatch
    # logic to be dynamo tracable or 2. fixing https://github.com/pytorch/pytorch/issues/144360.
    # but it exposes some bugs in existing tests so we have to have a temporary flag to control
    # the behavior, which allows dynamo to store an empty graph for a frame without falling back to eager
    try:
        # We need to turn off the is_fx_tracing_flag. Remove this flag check from dynamo
        # once we are confident fx tracing works with dynamo.
        torch.fx._symbolic_trace._set_is_fx_tracing(False)
        # pyrefly: ignore [bad-assignment]
        torch._dynamo.config.allow_empty_graphs = True
        torch._dynamo.config.capture_scalar_outputs = True
        yield
    finally:
        torch.fx._symbolic_trace._set_is_fx_tracing(_old_is_tracing)
        torch._dynamo.config.allow_empty_graphs = _old_allow_empty_graphs
        torch._dynamo.config.capture_scalar_outputs = _old_capture_scalar_outputs


# The invariant here is that we always trace the branch with fake tensor
def _maybe_fake_tracing(fn, inputs: list[Any], pre_dispatch):
    # [FUNCTION DOC] HELPER: Traces fn on inputs with make_fx (data-dependent op errors off),
    # always using fake tensors: reuses the fake mode of the first fake input in real mode,
    # else fake tracing mode. Proxy tracing is disabled around it. When a shape environment
    # exists, inserts deferred runtime asserts into the graph. Returns the graph module.
    fake_mode_det = None
    for inp in pytree.tree_leaves(inputs):
        if is_fake_tensor(inp):
            fake_mode_det = maybe_get_fake_mode(inp)
            break

    fake_mode: AbstractContextManager = nullcontext()
    tracing_mode = "fake"
    if fake_mode_det is not None:
        fake_mode = fake_mode_det
        tracing_mode = "real"

    # Note: we need to turn off proxy tensor mode to avoid tracing infra
    # code that happens in make_fx e.g. we now call as_strided when wrapping tensor
    # as fake tensor.
    with fake_mode, disable_proxy_modes_tracing():
        gm = make_fx(
            fn,
            tracing_mode=tracing_mode,
            pre_dispatch=pre_dispatch,
            _error_on_data_dependent_ops=False,
        )(*inputs)
        if not isinstance(fake_mode, nullcontext) and fake_mode.shape_env is not None:  # type: ignore[attr-defined]
            insert_deferred_runtime_asserts(
                gm,
                fake_mode.shape_env,  # type: ignore[attr-defined]
                "hoo_maybe_fake_tracing",
                export=True,  # type: ignore[attr-defined]
            )
        return gm


def potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    # [FUNCTION DOC] Traces gm on inputs and analyzes aliasing and mutation. Returns the bare
    # boolean True when tracing raises UnsupportedAliasMutationException (nested functionalized
    # cond); otherwise a pair of (input-input, input-output, output-output alias maps) and the
    # input mutation collection. Other exceptions propagate.
    # KNOWN ISSUE: the exceptional return is a bool while callers unpack a nested pair, so
    # callers fail with a type error in that case.
    try:
        gm = _maybe_fake_tracing(gm, inputs, pre_dispatch)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond_op is
        # functionalized
        return True
    except Exception as e:
        raise e

    example_inputs = [
        ph.meta.get("val", None) for ph in gm.graph.find_nodes(op="placeholder")
    ]
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        inp_mutation,
    ) = check_input_alias_and_mutation(gm, example_inputs)
    return (inp_inp_alias_map, inp_out_alias_map, out_out_alias_map), inp_mutation


def analyze_potential_input_alias_or_mutation(name, aliases, input_mutations):
    # [FUNCTION DOC] Raises RuntimeError naming the offending inputs if any alias map is
    # non-empty, or naming the mutating nodes if input_mutations is non-empty. name prefixes
    # the message. Returns nothing otherwise.
    if any(len(a) > 0 for a in aliases):
        # TODO: Investigate here further which node is exactly aliasing
        raise RuntimeError(
            f"{name} where aliases appear. "
            + f"In particular, these inputs \
            {set(el for el_map in aliases if len(el_map.keys()) > 0 for el in el_map)} "  # noqa: C401
            + "get aliased. Please ensure that this doesn't happen."
        )
    if len(input_mutations):
        # TODO: Investigate here further which node is exactly mutating the inputs
        raise RuntimeError(
            f"{name} where the inputs are mutated. "
            + f"In particular, these nodes are mutating the inputs \
            {set(el for el in input_mutations)}."  # noqa: C401
            + "Please ensure that this doesn't happen."
        )


def _has_potential_branch_input_mutation(gm, inputs, pre_dispatch=False):
    # [FUNCTION DOC] HELPER: Returns true if tracing gm on inputs shows any input mutation.
    (
        (_, _, _),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)

    return len(inp_mutation) > 0


def has_potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    # [FUNCTION DOC] Returns a pair of booleans for gm traced on inputs: whether any aliasing
    # exists (input-input, input-output or output-output) and whether any input is mutated.
    (
        (
            inp_inp_alias_map,
            inp_out_alias_map,
            out_out_alias_map,
        ),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)
    return (
        any(
            (
                len(inp_inp_alias_map) > 0,
                len(inp_out_alias_map) > 0,
                len(out_out_alias_map) > 0,
            )
        ),
        len(inp_mutation) > 0,
    )


def _collect_fake_inputs(inputs):
    # [FUNCTION DOC] HELPER: Maps proxies/nodes (using their stored example values) and plain ints
    # to a flat list of example inputs. Batched or functional tensors are unwrapped to their
    # fake tensor; tensor subclasses contribute their fake inner tensors; non-tensor example
    # values pass through; ints pass through. Raises an assertion error on non-fake tensors or
    # unexpected types.
    # KNOWN ISSUE: a proxy or node without a meta attribute is silently skipped.
    from torch._subclasses.fake_tensor import FakeTensor

    # Get the example values of the inputs.
    inputs_fake: list[FakeTensor | torch.Tensor | int] = []
    for inp in inputs:
        if isinstance(inp, (torch.fx.proxy.Proxy, torch.fx.node.Node)):
            inp = inp.node if isinstance(inp, torch.fx.proxy.Proxy) else inp
            if hasattr(inp, "meta"):
                val = inp.meta["example_value"]
                if isinstance(val, torch.Tensor):
                    if torch._C._functorch.is_batchedtensor(
                        val
                    ) or torch._C._functorch.is_functionaltensor(val):
                        # This case is for batched or functional tensors
                        # Unwrap the tensors
                        while torch._C._functorch.is_batchedtensor(
                            val
                        ) or torch._C._functorch.is_functionaltensor(val):
                            val = torch._C._functorch.get_unwrapped(val)
                        if not is_fake_tensor(val):
                            raise AssertionError(
                                f"Expected FakeTensor after unwrapping, got {type(val)}"
                            )
                        inputs_fake.append(val)
                    elif is_traceable_wrapper_subclass(val):
                        for attr_name in val.__tensor_flatten__()[0]:
                            unwrapped_input = getattr(val, attr_name)
                            if not isinstance(unwrapped_input, torch.Tensor):
                                continue
                            if not is_fake_tensor(unwrapped_input):
                                raise AssertionError(
                                    f"Expected FakeTensor after unwrapping, got {type(unwrapped_input)}"
                                )
                            inputs_fake.append(unwrapped_input)
                    else:
                        # This is the standard case of a TensorVariable
                        if not is_fake_tensor(val):
                            raise AssertionError(
                                f"Expected FakeTensor, got {type(val)}"
                            )
                        inputs_fake.append(val)
                else:
                    # This case is for SymInts and other non-Tensor elements
                    if isinstance(val, torch.Tensor):
                        raise AssertionError(f"Expected non-Tensor, got {type(val)}")
                    inputs_fake.append(val)
        else:
            # This case is for ints
            if not isinstance(inp, int):
                raise AssertionError(f"Expected int, got {type(inp)}")
            inputs_fake.append(inp)

    return inputs_fake


def _check_alias_and_mutation(graph_module, inputs_fake, name, pre_dispatch):
    # [FUNCTION DOC] HELPER: Raises RuntimeError saying name may alias input/output or modify
    # inputs, when analysis of graph_module on inputs_fake finds either. Returns nothing.
    aliases, inp_mutation = has_potential_input_alias_or_mutation(
        graph_module, inputs_fake, pre_dispatch=pre_dispatch
    )
    if aliases:
        raise RuntimeError(f"{name} might be aliasing the input or the output!")
    if inp_mutation:
        raise RuntimeError(f"{name} might be modifying the input!")


def unique_graph_id(proxy_mode, prefix):
    # [FUNCTION DOC] Returns (index, name) for a graph name not yet used on the tracer root of proxy_mode.
    """Returns a unique name and id for a graph to be added to a proxy_mode tracer"""
    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    return unique_graph_name_with_root(proxy_mode.tracer.root, prefix)


def unique_graph_name_with_root(
    root: torch.fx.GraphModule, prefix: str
) -> tuple[int, str]:
    # [FUNCTION DOC] Finds the smallest non-negative integer i such that the root module has no
    # attribute named prefix_i; returns (i, that name). Does not register anything.
    next_name = None
    i = 0
    # pyrefly: ignore [bad-assignment]
    while not next_name:
        candidate = f"{prefix}_{i}"
        if hasattr(root, candidate):
            i += 1
        else:
            next_name = candidate
    return i, next_name


def _from_fun(t):
    # [FUNCTION DOC] HELPER: Converts a tensor to a fresh uninitialized strided tensor with the same
    # metadata (non-bool dtypes); for bool tensors syncs and unwraps functional wrappers and
    # returns a clone of the plain tensor. Non-tensors are returned unchanged.
    from torch._functorch.aot_autograd import from_fun

    if isinstance(t, torch.Tensor):
        if t.dtype != torch.bool:
            return torch.empty_strided(
                t.size(),
                t.stride(),
                dtype=t.dtype,
                requires_grad=t.requires_grad,
                device=t.device,
            )
        else:
            # clone of a functional tensor produces a functional tensor
            # but we want to avoid it so we clone a non-functional version
            maybe_unfunc_t = t
            if isinstance(t, FunctionalTensor):
                torch._sync(t)
                maybe_unfunc_t = from_fun(t)
            elif torch._is_functional_tensor(t):
                # need to handle both types of functionalization here:
                # these are the tensors that came from the user,
                # which could be either FunctionalTensorWrapper or FunctionalTensor
                torch._sync(t)
                maybe_unfunc_t = torch._from_functional_tensor(t)
            # pyrefly: ignore[missing-attribute]
            return maybe_unfunc_t.clone()
    return t


def clone_outputs_aliasing_inputs(args):
    # [FUNCTION DOC] Returns a function that clones a tensor if it shares storage with any tensor
    # in args and returns other values unchanged. Used to keep backward graphs functional.
    input_storage = {
        StorageWeakRef(arg._typed_storage())
        for arg in args
        if isinstance(arg, torch.Tensor)
    }

    def maybe_clone(t):
        # [FUNCTION DOC] HELPER: Clones t if its storage matches an input's storage, else returns t.
        if (
            isinstance(t, torch.Tensor)
            and StorageWeakRef(t._typed_storage()) in input_storage
        ):
            return t.clone()
        return t

    return maybe_clone


def query_requires_grad(t: torch.Tensor) -> bool:
    # [FUNCTION DOC] Returns the requires_grad flag of t, first unwrapping a functional tensor wrapper.
    """requires_grad of ``t``, looking through a functional wrapper.

    A FunctionalTensor does not carry the autograd metadata of the tensor it
    wraps, so asking it directly reports False for something that does require
    grad. This matters when the joint is traced with functionalization active.
    """
    if torch._is_functional_tensor(t):
        t = torch._from_functional_tensor(t)
    return t.requires_grad


def prepare_fw_with_masks(fn):
    # [FUNCTION DOC] Returns a function that calls fn and returns its outputs together with a list
    # of booleans saying which outputs are tensors that require grad.
    def fw_with_masks(*args):
        # [FUNCTION DOC] HELPER: Runs fn on args; returns (outputs, per-output requires-grad mask).
        fw_out = fn(*args)
        return fw_out, [
            bool(isinstance(ret, torch.Tensor) and query_requires_grad(ret))
            for ret in fw_out
        ]

    return fw_with_masks


def prepare_fw_with_masks_all_requires_grad(fn):
    # [FUNCTION DOC] Like prepare_fw_with_masks, but if any input tensor requires grad, marks all
    # floating point output tensors as requiring grad (in place), matching autograd.Function
    # semantics. Returns the wrapper producing (outputs, nested mask of requires-grad per tensor).
    def fw_with_masks(*args):
        # [FUNCTION DOC] HELPER: Runs fn on args, forces floating outputs to require grad when any input does, returns (outputs, masks).
        fw_out = fn(*args)
        # Note [force all outputs to be require grad]
        # Instead of using the original fn, we set the output of original
        # fn to all require grad. This is consistent with the behavior
        # of autograd.Function, where if any one of the inputs requires grad
        # all output will be require grad. This also makes the downstream
        # require_gradness reasoning much easier.
        if pytree.tree_any_only(torch.Tensor, lambda t: t.requires_grad, args):
            fw_out = pytree.tree_map_only(
                torch.Tensor,
                lambda x: x.requires_grad_(True) if x.dtype.is_floating_point else x,
                fw_out,
            )

        return fw_out, pytree.tree_map_only(torch.Tensor, query_requires_grad, fw_out)

    return fw_with_masks


# This function replaces None gradients with all-zero gradients.
# `None` gradients are problematic for CUDA graphs. Those gradients are
# replaced with an all-zero tensor for better optimization
def unmask_none_gradients(grads, operands):
    # [FUNCTION DOC] Pairs each gradient with its operand; keeps non-None gradients, replaces
    # None with zeros shaped like the operand if it is a tensor, or None for int/symint
    # operands. Raises an assertion error if an operand is not a tensor, int or symint.
    # Returns a list.
    allowed_types = (torch.Tensor, int, torch.SymInt)
    if not all(isinstance(o, allowed_types) for o in operands):
        raise AssertionError(
            f"operands can only be of {allowed_types} but got {[type(o) for o in operands]}"
        )

    unmasked_grads = []
    for g, o in zip(grads, operands):
        if g is not None:
            unmasked_grads.append(g)
        else:
            # In case the operand is an int or a torch.SymInt, return None
            # This can happen for lifted_arguments. E.g., the shapes of a dynamic tensor are lifted and passed
            # as additional arguments
            unmasked_grads.append(
                torch.zeros_like(o) if isinstance(o, torch.Tensor) else None
            )

    return unmasked_grads


def _maybe_fake_prop_ignore_unbacked(fn, args):
    # [FUNCTION DOC] HELPER: Calls fn on plain (non-functional) versions of args with
    # functionalization, functional mode, proxy tracing and other current modes disabled. If the
    # args carry a fake mode, enters it and ignores fresh unbacked symbols. Returns fn's result.
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            unfunc_args = [_from_fun(arg) for arg in args]
        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(
                torch.utils._python_dispatch._disable_current_modes()
            )
            if (fake_mode := detect_fake_mode(unfunc_args)) is not None:
                ctx_stack.enter_context(fake_mode)
                if fake_mode.shape_env is not None:
                    ctx_stack.enter_context(
                        fake_mode.shape_env.ignore_fresh_unbacked_symbols()
                    )
            return fn(*unfunc_args)


def redirect_to_mode(hop: OperatorBase, mode):
    # [FUNCTION DOC] Registers on hop an implementation for the given mode type that forwards the
    # call to the mode's torch dispatch handler; returns the registered function.
    """Utility for redispatching HOP to underlying mode

    Args:
        hop: The HOP to redispatch
        mode: The mode to redispatch to

    Returns:
        A decorated function that implements the HOP for the given mode
    """

    @hop.py_impl(mode)
    def impl(mode, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the mode's torch dispatch handler with the HOP, no types, and the args.
        return mode.__torch_dispatch__(hop, [], args, kwargs)

    return impl


# TODO: The parameter use_output_and_grad_bw is required because some operations
# that utilize this function, such as the while_loop, may require (grad, fwd_outputs)
def create_fw_bw_graph(fn, use_output_and_grad_bw, fw_inputs, fw_outputs):
    # [FUNCTION DOC] Builds a forward graph and a joint (backward) graph for fn. The forward is
    # traced on fw_inputs. The joint function takes gradients and inputs (or, with
    # use_output_and_grad_bw, a nested pair of grads and inputs+outputs, using only the last
    # element of the second as inputs), runs an autograd joint of the masked forward, fills
    # None gradients with zeros, and clones outputs aliasing inputs. Example gradients derive
    # from fw_outputs. Returns (fw_graph, joint_graph). Uses dummy AOT config.
    # KNOWN ISSUE: in the use_output_and_grad_bw path only the last of the supplied
    # inputs-and-outputs is treated as the inputs, which looks specific to while_loop.
    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # Note:[HOP create fw_bw graph] We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    example_grad = [_from_fun(out) for out in fw_outputs]
    num_grads = len(example_grad)
    fw_graph = _maybe_reenter_make_fx(fn)(*fw_inputs)

    def joint_fn(*joint_operands_grads):
        # [FUNCTION DOC] HELPER: Splits the arguments into grads and inputs, runs the joint
        # forward-backward, and returns gradient list with None filled and aliases cloned.
        if use_output_and_grad_bw:
            grads = joint_operands_grads[0]
            inputs = joint_operands_grads[1][-1:]
        else:
            grads = joint_operands_grads[:num_grads]
            inputs = joint_operands_grads[num_grads:]

        joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
        _, grads = joint(
            list(inputs),
            [grad for grad in grads if grad is not None and grad.requires_grad],
        )

        # Unmask None gradients to all-zero gradients
        unmasked_grads = unmask_none_gradients(grads, inputs)

        # In order to keep map functional for backward graph,
        # we clone outputs that are aliasing inputs
        maybe_clone = clone_outputs_aliasing_inputs(joint_operands_grads)

        return pytree.tree_map(maybe_clone, unmasked_grads)

    if use_output_and_grad_bw:
        example_xs_out = list(fw_inputs) + list(fw_outputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            (list(example_grad), list(example_xs_out))
        )
    else:
        example_xs_out = list(fw_inputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            *(list(example_grad) + list(example_xs_out))
        )

    return fw_graph, joint_graph


def _unstack_pytree(xs):
    # [FUNCTION DOC] HELPER: Splits a pytree of tensors along the leading dimension into a list of
    # pytrees, one per index. Raises RuntimeError if any leaf is not a tensor or if leading
    # sizes differ.
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    a = zip(*flat_xs)

    pytrees = [pytree.tree_unflatten(tuple, inspec) for tuple in a]
    return pytrees


def _stack_pytree(pytrees):
    # [FUNCTION DOC] HELPER: Inverse of unstacking: stacks the corresponding leaves of a list of
    # same-structured pytrees into tensors along a new leading dimension. Leaf positions that
    # are None in all trees stay None. Raises RuntimeError for mixed or non-tensor leaves and
    # an assertion error for an empty list. Returns a pytree with the shared structure.
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    if out_spec is None:
        raise AssertionError("out_spec cannot be None")
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


# We cannot call save_for_backward for symints. This helper function
# can be used to save symints as direct attributes of ctx in autograd.Function.
#
# For example, if args = (x, y, s0, z, s1),
# save_values_for_backward will partition the args into two lists, and a bookkeeping list pos:
#   partitioned_args[0] = (x, y, z)
#   partitioned_args[1] = (s0, s1)
#   pos = (0, 0, 1, 0, 1)
# pos list keeps track of which partition the args
# is partitioned into in order to recover it in saved_values.
#
# In saved_values, we can recover the original args by:
# iterating over the pos list and pop one item from the front of partitioned_args[pos[i]].
# We use t_idx and s_idx to keep track of the next index of the item we are going to pop for the two lists.
def save_values_for_backward(ctx, args):
    # [FUNCTION DOC] Saves args on an autograd context: tensors through save_for_backward, all
    # other values (symints, ints, None, script objects, opaque classes) as a plain list
    # attribute, plus a position list recording which group each arg came from. Raises an
    # assertion error for disallowed types or if ctx already has these attributes.
    if not all(
        isinstance(arg, (torch.Tensor, torch.SymInt, int, type(None), FakeScriptObject))
        or is_custom_class(type(arg))
        for arg in args
    ):
        raise AssertionError(f"Invalid arg types in {args}")
    partitioned_args: list[Any] = [[], []]
    pos = []
    for arg in args:
        idx = 0 if isinstance(arg, torch.Tensor) else 1
        partitioned_args[idx].append(arg)
        pos.append(idx)

    if hasattr(ctx, "non_tensor_args"):
        raise AssertionError("ctx already has non_tensor_args attribute.")
    if hasattr(ctx, "pos"):
        raise AssertionError("ctx already has pos attribute.")
    ctx.save_for_backward(*partitioned_args[0])
    ctx.non_tensor_args = partitioned_args[1]
    ctx.pos = pos


def saved_values(ctx):
    # [FUNCTION DOC] Reconstructs the original ordered args saved by save_values_for_backward by
    # interleaving saved tensors and stored non-tensor values per the position list. Returns a
    # tuple; raises an assertion error if counts are inconsistent.
    args = []
    t_idx = 0
    s_idx = 0
    saved_tensors = ctx.saved_tensors
    for p in ctx.pos:
        if p == 0:
            args.append(saved_tensors[t_idx])
            t_idx += 1
        else:
            args.append(ctx.non_tensor_args[s_idx])
            s_idx += 1
    if t_idx + s_idx != len(ctx.pos):
        raise AssertionError(
            f"t_idx ({t_idx}) + s_idx ({s_idx}) != len(ctx.pos) ({len(ctx.pos)})"
        )
    return tuple(args)


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]:
    # [FUNCTION DOC] Cuts a sequence into consecutive slices with the given sizes and returns the
    # list of slices. Raises an assertion error if the sizes do not sum to the sequence length.
    if sum(chunk_sizes) != len(iterable):
        raise AssertionError(
            f"the sum of all chunks ({sum(chunk_sizes)}) needs to match the length of the iterable ({len(iterable)})."
        )
    elements = []
    idx = 0
    for size in chunk_sizes:
        elements.append(iterable[idx : idx + size])
        idx += size
    return elements


def _clone_aliasing_output(inputs: Sequence[Any], outputs: Sequence[Any]):
    # [FUNCTION DOC] HELPER: Returns a list of outputs in which any tensor sharing storage with an
    # input or with an earlier output is replaced by a clone; other values are unchanged.
    # For tensors whose grad is None, create zero tensors as gradients
    # This invariant is useful for cudagraph.

    # Elimitate input-output, output-output aliasing
    seen_input_storages = {
        StorageWeakRef(t._typed_storage())
        for t in inputs
        if isinstance(t, torch.Tensor)
    }
    seen_output_storages = set()
    final_outputs = []
    for out in outputs:
        if isinstance(out, torch.Tensor):
            out_storage = StorageWeakRef(out._typed_storage())
            if (
                out_storage in seen_input_storages
                or out_storage in seen_output_storages
            ):
                out = out.clone()
            seen_output_storages.add(StorageWeakRef(out._typed_storage()))
        final_outputs.append(out)
    return final_outputs


def create_bw_fn(
    fn: Callable, args: tuple[Any, ...], return_fw_outputs: bool = False
) -> Callable:
    # [FUNCTION DOC] Builds a backward function for fn at the given example args. The result takes
    # the primals followed by output gradients, runs a joint forward-backward (forcing outputs
    # to require grad when any input does), replaces None gradients of tensor inputs by zeros,
    # clones gradients aliasing inputs or each other, and returns the gradient list (preceded
    # by the forward outputs when return_fw_outputs). Raises an assertion error if the
    # gradient count differs from the arg count.
    """
    For a fn that accepts flat inputs and returns flat outputs:
        fw_out = fn(*args),
    this function returns:
        grad_args = bw_fn(*args_and_grad_output)
    with the following invariants:
      1. args + fw_out has an 1-1 correspondence to args_and_grad_output
      2. grad_args has an 1-1 corresponsence to args
      3. for tensor arg whose requires_grad is False, its corresponding grad in
         grad_args will be a zero tensor with the same shape.
    """

    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # pyrefly: ignore [missing-module-attribute]
    from torch._higher_order_ops.utils import prepare_fw_with_masks_all_requires_grad

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )
    n_primals = len(args)

    bw_fn = create_joint(
        prepare_fw_with_masks_all_requires_grad(fn), aot_config=dummy_aot_config
    )

    def flat_fn(*args_and_grad_outs):
        # [FUNCTION DOC] HELPER: Splits inputs into primals and tangents, runs the joint, and returns
        # the (optionally forward-output-prefixed) cleaned gradients.
        primals = args_and_grad_outs[:n_primals]
        tangents = args_and_grad_outs[n_primals:]
        fw_outs, grad_args = bw_fn(primals, tangents)
        if len(args) != len(grad_args):
            raise AssertionError(
                f"Expected {len(args)} grad_args, got {len(grad_args)}"
            )

        # For tensors whose grad is None, create zero tensors as gradients
        # This invariant is useful for cudagraph.
        grad_args = [
            torch.zeros_like(arg)
            if isinstance(arg, torch.Tensor) and grad is None
            else grad
            for grad, arg in zip(grad_args, primals)
        ]

        final_grads = _clone_aliasing_output(args_and_grad_outs, grad_args)
        if return_fw_outputs:
            return *fw_outs, *final_grads
        return final_grads

    return flat_fn


def get_dummy_aot_autograd_config():
    # [FUNCTION DOC] Returns a placeholder AOT autograd configuration with no compilers or
    # partitioner, empty decompositions and no parameters, for tracing joint graphs.
    from torch._functorch.aot_autograd import AOTConfig

    return AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )


# Slices off the first element of a given dimension. Used to build a
# representative single slice for subgraph tracing of control-flow ops.
# When `dim` has size 0 (e.g. a zero-length scan) there is no real slice to
# take, so return an uninitialized tensor with `dim` removed.
def first_slice_copy(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # [FUNCTION DOC] Returns a copy of the first slice of t along dim. If dim has size zero,
    # returns a zero-filled tensor shaped like t with that dim removed.
    if t.shape[dim] == 0:
        shape = list(t.shape)
        del shape[dim]
        return t.new_zeros(shape)
    return torch.select_copy(t, dim, 0)


# Returns a mask whether a list element is a tensor or not
def get_tensor_mask(tensor_list: Iterable[Any]) -> list[bool]:
    # [FUNCTION DOC] Returns a list of booleans, true where the element is a tensor.
    return [bool(isinstance(v, torch.Tensor)) for v in tensor_list]


def mask_list(
    mask: list[bool], inp: list[Any], other: list[Any] | None = None
) -> list[Any]:
    # [FUNCTION DOC] Applies a boolean mask to inp. Without other, returns only elements where the
    # mask is true; with other, returns inp elements where true and other elements where
    # false. Raises an assertion error if the lengths of mask, inp (and other) differ.
    # Masks elements on an `inp` list.
    # If other is None, then the elements of the `inp` list where the mask is False are removed
    # If other is not None, then the elements of the `inp` list where the mask is False are
    # replaced with the elements of the `other` list
    if len(mask) != len(inp):
        raise AssertionError(
            f"The length of the mask ({len(mask)}) needs to be identical to the length of the input ({len(inp)})"
        )
    if other is not None:
        if len(inp) != len(other):
            raise AssertionError(
                f"If an input and an other list is provided, they need to have the same length ({len(inp)} != {len(other)})"
            )
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    else:
        return [i for m, i in zip(mask, inp) if m]


def first_slice_copy_with_grad(li: Iterable[Any]) -> list[Any]:
    # [FUNCTION DOC] Returns the first-slice copy of each tensor in li, re-applying each source
    # tensor's requires_grad flag to the copy. Every element must be a tensor.
    # First_slice_copy does not keep the original requires_grad flag,
    # but we need it for materialize_as_graph
    # in order to compute the correct gradients
    # The reason why first_slice_copy doesn't keep requires_grad flag is
    # because it's called in torch.autograd.Function.backward/forward.
    slc = [first_slice_copy(x).requires_grad_(x.requires_grad) for x in li]
    return slc


# Reports the difference between meta of two tensors in a string
def diff_tensor_meta(
    meta1: TensorMetadata, meta2: TensorMetadata, check_grad=True
) -> list[str]:
    # [FUNCTION DOC] Compares two tensor metadata records field by field (skipping requires_grad
    # unless check_grad) and returns a list of description strings for differing fields.
    # A comparison that would need a data-dependent guard is counted as a difference.
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    pair_diffs = []
    for meta_name in TensorMetadata._fields:
        if not check_grad and meta_name == "requires_grad":
            continue
        val1 = getattr(meta1, meta_name)
        val2 = getattr(meta2, meta_name)
        try:
            if val1 != val2:
                pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
        except GuardOnDataDependentSymNode:
            pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
            continue
    return pair_diffs


# Note [lifted arg types in hop]
# For dynamoed hops, we automatically lift the free symbols in tensors as arguments.
# This has implications for the types of lifted args for different dispatch keys:
#   1. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd need to support torch.Symint
#      lifted args because it's on the path of torch.compile(dynamic=True).
#   2. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd, CompositeExplicitAutograd need
#      to support int arguments. In the eager run case, we re-trace the subgraph in AutogradKey, so inner
#      hops may receive int inputs from the shape of outer tensor inputs.
#      However, CompositeExplicitAutograd won't receive SymInt inputs because it only accepts real tensor inputs.
def validate_subgraph_args_types(lifted_args: tuple[Any, ...] | list[Any]):
    # [FUNCTION DOC] Raises an assertion error unless every arg is a tensor, int or symint.
    allowed_types = (torch.Tensor, int, torch.SymInt)
    if not all(
        isinstance(arg, (torch.Tensor, int, torch.SymInt)) for arg in lifted_args
    ):
        raise AssertionError(
            f"{lifted_args} can only be of {allowed_types} but got {tuple(type(arg) for arg in lifted_args)}"
        )


# TODO: Return a more detailed information as to which node
# causes a mutation or an alias. This may requires a per operator tensor version checking
def check_input_alias_and_mutation(
    gm: torch.fx.GraphModule,
    fake_args: list[FakeTensor],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
    # [FUNCTION DOC] Analyzes graph module gm and returns four items: input-input alias map,
    # input-output alias map, output-output alias map (index to index mappings by shared
    # storage) and the list of mutated input indices. Delegates to the variant that also
    # returns outputs.
    # KNOWN ISSUE: the fake_args parameter is unused; example inputs come from the graph's
    # placeholder metadata.
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        mutated_inputs,
    ) = check_input_alias_and_mutation_return_outputs(gm)[:-1]
    # pyrefly: ignore [bad-return]
    return inp_inp_alias_map, inp_out_alias_map, out_out_alias_map, mutated_inputs


def _tensor_storage(t) -> StorageWeakRef:
    # [FUNCTION DOC] HELPER: Returns a weak reference to the tensor's underlying storage.
    return StorageWeakRef(t._typed_storage())


def _get_example_value(n):
    # [FUNCTION DOC] HELPER: For a graph node returns its stored value metadata, else its example
    # value metadata, else None; non-node inputs are returned unchanged.
    if not isinstance(n, torch.fx.Node):
        return n
    if "val" in n.meta:
        return n.meta["val"]
    if "example_value" in n.meta:
        return n.meta["example_value"]
    return None


def get_graph_output_example_values(gm: torch.fx.GraphModule) -> list[Any]:
    # [FUNCTION DOC] Returns the example values of the flattened outputs of the graph module's
    # output node (None where a node has no value metadata).
    output_node = gm.graph.find_nodes(op="output")[0]
    return [_get_example_value(n) for n in pytree.tree_flatten(output_node.args[0])[0]]


def check_input_alias_and_mutation_return_outputs(
    gm: torch.fx.GraphModule,
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    list[int],
    tuple[Any, ...] | list[Any],
]:
    # [FUNCTION DOC] Inspects the graph module using value metadata: builds storage maps for
    # placeholder inputs and outputs, derives input-input, input-output and output-output alias
    # maps (index to index), and scans call nodes of operator overloads for write arguments
    # whose storage belongs to an input to collect mutated input indices (an assertion error
    # if a written argument is not a tensor). Returns the three maps, the mutated indices,
    # and the output example values.
    # KNOWN ISSUE: placeholders without value metadata are dropped, which can shift the input
    # indices relative to the true placeholder positions.
    fake_args = [
        _get_example_value(n)
        for n in gm.graph.find_nodes(op="placeholder")
        if isinstance(n, torch.fx.Node) and "val" in n.meta
    ]
    outputs = get_graph_output_example_values(gm)

    # We need to analyze the original fake_args to detect
    # inp-inp alias.
    inp_storage_map = {
        _tensor_storage(inp): i
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor)
    }
    out_storage_map = {
        _tensor_storage(out): i
        for i, out in enumerate(outputs)
        if isinstance(out, torch.Tensor)
    }
    inp_inp_alias_map = {
        i: inp_storage_map[_tensor_storage(inp)]
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor) and inp_storage_map[_tensor_storage(inp)] != i
    }
    out_out_alias_map = {
        i: out_storage_map[_tensor_storage(out)]
        for i, out in enumerate(outputs)
        if isinstance(out, torch.Tensor) and out_storage_map[_tensor_storage(out)] != i
    }
    inp_out_alias_map = {
        i: out_storage_map[_tensor_storage(inp)]
        for i, inp in enumerate(fake_args)
        if isinstance(inp, torch.Tensor) and _tensor_storage(inp) in out_storage_map
    }
    mutated_inputs = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            for arg_node, arg_schema in zip(node.args, node.target._schema.arguments):
                if arg_schema.is_write:
                    arg_val = _get_example_value(arg_node)
                    if not isinstance(arg_val, torch.Tensor):
                        raise AssertionError(
                            f"Expected arg_val to be a Tensor, got {type(arg_val)}"
                        )
                    if _tensor_storage(arg_val) in inp_storage_map:
                        mutated_inputs.append(inp_storage_map[_tensor_storage(arg_val)])

    return (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        mutated_inputs,
        outputs,
    )


registered_hop_fake_fns: dict[torch._ops.OpOverload, Callable] = {}

# Set skip_cache=True for HOPs that should not be cached in FakeTensorMode
# (previously these ops were registered with .py_impl(FakeTensorMode)) and
# were never cached so we are just preserving original behaviour
hops_that_skip_faketensor_cache: set[torch._ops.OpOverload] = set()


F = TypeVar("F", bound=Callable)
_P = ParamSpec("_P")


@overload
def register_fake(
    hop, fn: None = None, *, skip_cache: bool = ...
) -> Callable[[F], F]: ...


@overload
def register_fake(hop, fn: F, *, skip_cache: bool = ...) -> F: ...


def register_fake(hop, fn=None, *, skip_cache=False):
    # [FUNCTION DOC] Registers a fake-tensor implementation for hop in a global registry (assertion
    # error if already registered) and routes the FakeTensorMode key to the mode handler; if
    # skip_cache the hop is also recorded as not cacheable. Usable as a decorator (fn omitted,
    # returns the registering function) or directly (returns fn).
    """
    Register a fake function for a HOP. This is conceptually equivalent of the
    register_fake utility for the custom ops. The registered function is called
    inside the fake_tensor _dispatch_impl.

    Set skip_cache=True to keep a HOP's fake output out of the FakeTensorMode
    cache -- either to preserve the historical uncached behavior of
    py_impl(FakeTensorMode), or because its outputs alias in ways the cache
    cannot reconstruct (e.g. auto_functionalized).
    """
    if hop in registered_hop_fake_fns:
        raise AssertionError(f"hop {hop} already registered in registered_hop_fake_fns")

    def register(func: F) -> F:
        # [FUNCTION DOC] HELPER: Performs the registration of func for hop and returns func.
        from torch._subclasses.fake_tensor import FakeTensorMode

        redirect_to_mode(hop, FakeTensorMode)

        registered_hop_fake_fns[hop] = func
        if skip_cache:
            hops_that_skip_faketensor_cache.add(hop)
        return func

    if fn is None:
        return register
    return register(fn)


class SubgraphCallableWrapper:
    # [CLASS DOC] Callable wrapper holding a function and the subgraph it came from; hashes by
    # the subgraph's identity so fake tensor dispatch caching keys on the subgraph, and
    # copies the subgraph's boxed-call flag.
    """
    Wraps a callable while preserving the original subgraph identity for
    fake tensor caching. Without this, passing a plain closure to
    invoke_subgraph would bypass the dispatch cache entirely.
    """

    def __init__(self, fn, subgraph):
        # [FUNCTION DOC] Stores fn and subgraph and copies the subgraph's boxed-call flag (default false).
        self.fn = fn
        self.subgraph = subgraph
        self._boxed_call = getattr(subgraph, "_boxed_call", False)

    def __hash__(self):
        # [FUNCTION DOC] Returns the object identity number of the subgraph.
        # KNOWN ISSUE: equality is not overridden, so equal hashes do not imply equal wrappers.
        return id(self.subgraph)

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Calls the wrapped function with the given arguments and returns its result.
        return self.fn(*args, **kwargs)


class FunctionalizeCtxWrapper(SubgraphCallableWrapper):
    # [CLASS DOC] Subgraph wrapper that runs a subgraph through a functionalization context.
    # State: ctx and the indices of inputs whose updated values are appended to the outputs.
    # Boxed subgraphs take a single list of inputs.
    """Functionalizes a subgraph and optionally returns selected input updates.

    Each update is a new tensor; the original input is not mutated.
    """

    # Prevents PYTORCH_TEST_WITH_DYNAMO=1 test failures
    @torch._disable_dynamo
    def __init__(self, ctx, subgraph, mutated_input_indices=()):
        # [FUNCTION DOC] Wraps subgraph as both function and identity, and stores the functionalization
        # context and the tuple of mutated input indices. Dynamo is disabled for this method.
        super().__init__(subgraph, subgraph)
        self.ctx = ctx
        self.mutated_input_indices = mutated_input_indices

    def __repr__(self):
        # [FUNCTION DOC] Returns a description string naming the wrapped subgraph.
        # KNOWN ISSUE: the string has an unbalanced closing parenthesis.
        return f"FunctionalizeCtxWrapper on subgraph {self.subgraph})"

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the functionalized subgraph. Graph modules run via the
        # functionalization context (boxed ones take a list, building one from positional args
        # if needed; unboxed ones run through an Interpreter with node metadata preserved);
        # other callables are functionalized directly. If mutated input indices are set, the
        # outputs are extended with the values of those inputs, read before the call.
        def append_mutated_inputs(fn):
            # [FUNCTION DOC] HELPER: Returns fn unchanged when no mutated indices are set; otherwise a
            # wrapper returning fn's outputs followed by the selected inputs.
            if not self.mutated_input_indices:
                return fn

            def wrapped(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Snapshots the selected inputs, calls fn, and returns outputs plus the snapshots.
                inputs = args[0] if self._boxed_call else args
                # Snapshot before a boxed GraphModule clears its input list.
                mutated_inputs = tuple(inputs[i] for i in self.mutated_input_indices)
                # Dynamo normalizes invoke_subgraph outputs to tuples.
                return (*fn(*args, **kwargs), *mutated_inputs)

            return wrapped

        if isinstance(self.subgraph, torch.fx.GraphModule):
            if self._boxed_call:
                # Not all callers respect _boxed_call (e.g. reenter_make_fx).
                functionalized = self.ctx.functionalize(
                    append_mutated_inputs(self.subgraph)
                )
                if len(args) == 1 and isinstance(args[0], list):
                    return functionalized(args[0])
                return functionalized(list(args))
            else:
                # Running graph with interpreter is needed for propagating the stack_trace
                with fx_traceback.preserve_node_meta():
                    return self.ctx.functionalize(
                        append_mutated_inputs(torch.fx.Interpreter(self.subgraph).run)
                    )(*args, **kwargs)
        functionalized = self.ctx.functionalize(append_mutated_inputs(self.subgraph))
        if self._boxed_call:
            if len(args) == 1 and isinstance(args[0], list):
                return functionalized(args[0])
            return functionalized(list(args))
        return functionalized(*args, **kwargs)


# A wrapper over HigherOrderOperator that also carries its schema
class HopInstance:
    # [CLASS DOC] Pairs a higher-order operator with the schema generated for a particular call;
    # calling it calls the operator. Attributes: _op and _schema.
    def __init__(self, op: HigherOrderOperator, schema: HopSchema):
        # [FUNCTION DOC] Stores op and schema; raises an assertion error if op is not a HigherOrderOperator.
        if not isinstance(op, HigherOrderOperator):
            raise AssertionError(f"Expected HigherOrderOperator, got {type(op)}")
        self._op = op
        # Using "_" to be consistent with how we access _schema of OpOverload
        self._schema = schema

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Calls the wrapped operator with the arguments and returns its result.
        return self._op(*args, **kwargs)

    @staticmethod
    def create(hop: HigherOrderOperator, *args, **kwargs):
        # [FUNCTION DOC] Generates the schema of hop for the given example arguments and returns a HopInstance for it.
        return HopInstance(hop, hop.gen_schema(*args, **kwargs))


# This call_op can be used to call a HopInstance with
# flat args and kwargs. We need to make use of the hop's schema's tree_spec
# to unflatten the args and kwargs before calling the hop.
def call_op(op: OpOverload | HopInstance, args, kwargs):
    # [FUNCTION DOC] Calls an operator overload directly with args and kwargs. For a HopInstance,
    # fills the remaining schema arguments from kwargs by name (positional unless keyword-only),
    # then, if the schema has a tree spec, unflattens the bound values into the original args
    # and kwargs before calling; otherwise calls with bound args and kwargs. Raises an
    # assertion error for missing arguments, wrong counts or a bad op type.
    if isinstance(op, OpOverload):
        return op(*args, **kwargs)

    if not isinstance(op, HopInstance):
        raise AssertionError(f"Expected HopInstance, got {type(op)}")
    schema = op._schema
    bound_args = list(args)
    bound_kwargs = {}
    for arg in schema.arguments[len(bound_args) :]:
        if arg.name not in kwargs:
            raise AssertionError(f"arg {arg.name} not in kwargs: {kwargs}")
        val = kwargs[arg.name]
        if not arg.kwarg_only:
            bound_args.append(val)
        else:
            bound_kwargs[arg.name] = val

    if schema.tree_spec is not None:
        if len(bound_args) != len(schema.arguments) or len(bound_kwargs) != 0:
            raise AssertionError(
                f"Expected {len(schema.arguments)} bound_args and 0 bound_kwargs, "
                f"got {len(bound_args)} and {len(bound_kwargs)}"
            )
        args, kwargs = pytree.tree_unflatten(bound_args, schema.tree_spec)
        return op(*args, **kwargs)
    else:
        if len(bound_args) + len(bound_kwargs) != len(schema.arguments):
            raise AssertionError(
                f"Expected {len(schema.arguments)} total args, "
                f"got {len(bound_args)} + {len(bound_kwargs)}"
            )
        return op(*bound_args, **bound_kwargs)


def materialize_as_graph(
    fn: Callable,
    args: tuple[Any, ...],
    include_key_set: torch._C.DispatchKeySet | None = None,
    exclude_key_set: torch._C.DispatchKeySet | None = None,
    force_enable_grad=False,
    subgraph_decomp_table: Mapping[OpOverload, Callable] | None = None,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Traces fn on args into a graph module in a clean environment: functionalization
    # suspended, current modes disabled, a fake mode entered if one is available (from the
    # active make_fx tracer or globally), args converted to plain tensors, dispatch key sets
    # forced to the given (default: current) include and exclude sets, optionally with grad
    # enabled. Returns the graph module; raises an assertion error if tracing yields none.
    if include_key_set is None:
        include_key_set = torch._C._dispatch_tls_local_include_set()
    if exclude_key_set is None:
        exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

    @torch._dynamo.disable(recursive=True, reason=None)
    def _materialize_as_graph_inner():
        # [FUNCTION DOC] HELPER: Does the traced materialization described above, excluded from dynamo, and returns the graph module.
        from torch._guards import active_fake_mode
        from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

        with suspend_functionalization(), disable_functional_mode():
            fake_mode = None
            if _CURRENT_MAKE_FX_TRACER is not None:
                fake_mode = _CURRENT_MAKE_FX_TRACER.fake_tensor_mode
            if fake_mode is None:
                fake_mode = active_fake_mode()

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch.utils._python_dispatch._disable_current_modes()
                )
                if fake_mode is not None:
                    stack.enter_context(fake_mode)

                with disable_proxy_modes_tracing():
                    unfunc_t = [_from_fun(arg) for arg in args]

                stack.enter_context(
                    torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
                )
                if force_enable_grad:
                    stack.enter_context(torch.enable_grad())
                return _maybe_reenter_make_fx(
                    fn, subgraph_decomp_table=subgraph_decomp_table
                )(*unfunc_t)

    gm = _materialize_as_graph_inner()
    if gm is None:
        raise AssertionError("materialize_as_graph returned None")
    return gm


def materialize_callable_in_args(op: HopInstance, args, kwargs):
    # [FUNCTION DOC] Traces the HOP call to turn callable arguments into graph modules. Flattens
    # args and kwargs, traces the call, finds the HOP node, and for every argument that became
    # a graph module attribute replaces the original callable with that graph module (the
    # original must be callable, else assertion error); other args are kept. Returns the
    # (args, kwargs) structure rebuilt from the materialized values. Raises an assertion
    # error if the schema is not a function schema matching the traced argument count.
    schema = op._schema
    hop = op._op
    flat_args, flat_spec = pytree.tree_flatten((args, kwargs))

    def wrapped_fn(*flat_args):
        # [FUNCTION DOC] HELPER: Ignores its flat arguments and performs the HOP call with the captured args and kwargs.
        return call_op(op, args, kwargs)

    # We need to trace the higher order op in order to materilaize the callable inputs that
    # are a callable (e.g. after functionalization key)
    gm = reenter_make_fx(wrapped_fn)(*flat_args)
    hop_node = gm.graph.find_nodes(op="call_function", target=hop)[0]
    arg_proxies = pytree.tree_leaves((hop_node.args, hop_node.kwargs))
    if not isinstance(schema, torch._C.FunctionSchema) or len(arg_proxies) != len(
        schema.arguments
    ):
        raise AssertionError(
            f"Expected FunctionSchema with {len(arg_proxies)} arguments"
        )

    # call_op preserves ordering of proxies via schema
    materialized_args = []
    for i, proxy in enumerate(arg_proxies):
        if (
            isinstance(proxy, torch.fx.Node)
            and proxy.op == "get_attr"
            and isinstance(getattr(gm, proxy.target), torch.fx.GraphModule)  # type: ignore[arg-type]
        ):
            if not callable(flat_args[i]):
                raise AssertionError(
                    f"Expected flat_args[{i}] to be callable for {schema}"
                )
            materialized_args.append(getattr(gm, proxy.target))  # type: ignore[arg-type]
        else:
            materialized_args.append(flat_args[i])

    return pytree.tree_unflatten(materialized_args, flat_spec)


def has_user_subclass(args, allowed_subclasses):
    # [FUNCTION DOC] Flattens args and returns true if any is a tensor of a subclass type that is not
    # in allowed_subclasses; false otherwise.
    """Check if any tensor arguments are user subclasses.

    This is used to determine if tensor subclasses should get a chance to run
    their own implementation first before falling back to the default implementation.

    Args:
        args: Arguments to check (will be flattened with pytree)
        allowed_subclasses: Tuple of allowed subclass types

    Returns:
        True if user tensor subclasses are found, False otherwise
    """
    flat_args, _ = pytree.tree_flatten(args)

    val = any(
        isinstance(a, torch.Tensor)
        and type(a) is not torch.Tensor
        and not isinstance(a, allowed_subclasses)
        for a in flat_args
    )
    return val


def _has_gen_schema(op: HigherOrderOperator):
    # [FUNCTION DOC] HELPER: True if the operator's class overrides schema generation, except for
    # invoke_quant_packed, which cannot generate one.
    # There is an InvokeQuant argument we cannot gen_schema.
    if op is torch.ops.higher_order.invoke_quant_packed:
        return False
    method = "gen_schema"
    return hasattr(type(op), method) and getattr(type(op), method) is not getattr(
        HigherOrderOperator, method
    )


def filter_with_masks(data: list[torch.Tensor | None], masks: list[bool]):
    # [FUNCTION DOC] Returns the elements of data where the mask is true; raises an assertion error on length mismatch.
    if len(data) != len(masks):
        raise AssertionError(
            f"data length ({len(data)}) != masks length ({len(masks)})"
        )
    return [item for item, keep in zip(data, masks) if keep]


def fill_none_with_masks(data: list[torch.Tensor | None], masks: list[bool]):
    # [FUNCTION DOC] Inverse of filtering: builds a list with the next data element at each true
    # mask position and None at false positions. data must have at least as many elements as
    # true masks (otherwise iteration ends with an error).
    data_iter = iter(data)
    return [next(data_iter) if kept else None for kept in masks]


def create_fn_remove_none(
    fn: Callable[_P, Any],
) -> tuple[Callable[_P, list[torch.Tensor]], list[bool]]:
    # [FUNCTION DOC] Returns (wrapped, mask). wrapped calls fn, flattens the result, records in the
    # shared mask list (mutated on each call) which leaves are tensors, and returns only the
    # tensor leaves. Read the mask only after wrapped has run.
    """Wrap ``fn`` so its non-Tensor output leaves are dropped, and expose the
    mask of which leaves survived.

    Returns ``(wrapped, mask)``:
      - ``wrapped(*args)`` calls ``fn(*args)``, flattens the pytree result and
        returns only its Tensor leaves as a list.
      - ``mask`` is a list populated whenever ``wrapped`` runs -- one bool per
        leaf, ``True`` where the leaf is a Tensor. Callers should read it
        AFTER invoking ``wrapped`` (typically indirectly via
        ``materialize_as_graph``), then pass it to ``fill_none_with_masks``
        to reconstruct the full output with ``None`` at the dropped slots.

    Used in HOP autograd impls in two places:
      - Around a forward branch that may return non-Tensor leaves (None,
        int/SymInt), so ``create_bw_fn``'s joint sees a tensor-only signature.
      - Around the resulting backward joint whose grad list contains ``None``
        at non-differentiable input slots, so the materialized ``GraphModule``
        returns a uniform Tensor list.
    """
    mask: list[bool] = []

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls fn, updates the shared mask, and returns the tensor leaves of the result.
        leaves = pytree.tree_leaves(fn(*args, **kwargs))
        mask.clear()
        mask.extend(isinstance(o, torch.Tensor) for o in leaves)
        return filter_with_masks(leaves, mask)

    return wrapped, mask
