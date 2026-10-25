# [MODULE DOC]
# Shared helpers for Dynamo backends: the AotAutograd backend wrapper (and its
# aot_autograd factory and keyword TypedDict) that routes a graph through
# AOTAutograd with given forward/backward/inference compilers, a decorator that
# swaps fake tensors for real zero tensors, kwargs for memory efficient fusion,
# and device/dtype detection from example inputs. Updates the global counters
# for aot_autograd outcomes. No import-time side effects beyond logger setup.
"""
This module provides common utilities and base classes for TorchDynamo backends.

Key components:
- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends
- Backend utilities for handling:
  - Fake tensor conversion
  - Device/dtype detection from inputs
  - Memory efficient fusion
  - Graph flattening
  - Common compiler configurations

The utilities here are used by various backend implementations to handle
common operations and provide consistent behavior across different backends.
AOT autograd functionality is particularly important as it enables ahead-of-time
optimization of both forward and backward passes.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, Required, TypedDict, TypeVar, Unpack
from unittest.mock import patch

import torch
from torch._dynamo import disable
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
from torch._dynamo.utils import counters, defake, flatten_graph_inputs
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    SerializableAOTDispatchCompiler,
)
from torch.utils._python_dispatch import _disable_current_modes


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._inductor.compile_fx import CompilerConfigExtra
    from torch._ops import OpOverload


log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class AotAutogradKwargs(TypedDict, total=False):
    # [CLASS DOC] HELPER: Typed mapping of the optional keyword arguments that
    # AotAutograd stores and forwards to aot_module_simplified; only
    # fw_compiler is required.
    """Keyword arguments accepted by AotAutograd and forwarded verbatim to
    aot_module_simplified. decompositions may also be a zero-arg thunk returning
    the decomposition table (a workaround for circular imports); AotAutograd
    resolves it to the table before forwarding."""

    fw_compiler: Required[Callable[..., Any]]
    bw_compiler: Callable[..., Any] | None
    inference_compiler: Callable[..., Any] | None
    partition_fn: Callable[..., Any]
    decompositions: (
        dict[OpOverload, Callable[..., Any]]
        | Callable[[], dict[OpOverload, Callable[..., Any]]]
    )
    keep_inference_input_mutations: bool
    compiler_config_extra: CompilerConfigExtra | None
    ignore_shape_env: bool
    disable_functionalization: bool
    pre_grad_passes: Callable[..., Any] | None
    compile_region_name: str | None


class AotAutograd:
    # [CLASS DOC]
    # Callable Dynamo backend object configured with AOTAutograd options
    # (stored in kwargs). Calling it compiles a graph module with
    # aot_module_simplified, filling in default backward and inference
    # compilers (the forward compiler) and wrapping the backward compiler so
    # Dynamo does not trace it. State: __name__ (fixed to compiler_fn) and
    # kwargs, which are mutated on call (decompositions thunk resolved,
    # bw_compiler and inference_compiler set).
    def __init__(self, **kwargs: Unpack[AotAutogradKwargs]) -> None:
        # [FUNCTION DOC] Store the options and set the name used by Dynamo.
        self.__name__ = "compiler_fn"
        self.kwargs: AotAutogradKwargs = kwargs

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
    ) -> Callable[..., Any]:
        # [FUNCTION DOC]
        # Compile gm with AOTAutograd. Extra kwargs are ignored with a warning.
        # If example inputs contain containers, flattens graph inputs first and
        # recurses. Otherwise resolves a decompositions thunk, counts totals,
        # wraps the backward compiler (unless already wrapped), defaults the
        # inference compiler, optionally enables debug asserts for the no-op
        # compiler, and calls aot_module_simplified inside AOT logging. Returns
        # the compiled callable with Dynamo disabled on it. Updates counters
        # (ok or not_ok) and re-raises any error, including restart requests.
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        # Hack to get around circular import problems with aot_eager_decomp_partition
        decompositions = self.kwargs.get("decompositions")
        if callable(decompositions):
            self.kwargs["decompositions"] = decompositions()

        # NB: don't delete counter increment
        counters["aot_autograd"]["total"] += 1

        def wrap_bw_compiler(bw_compiler_fn: Callable[P, R]) -> Callable[..., R]:
            # [FUNCTION DOC] HELPER: Return a wrapper of the backward compiler
            # that runs it with Dynamo disabled and disables Dynamo on the
            # produced backward callable; the wrapper is marked so it is not
            # wrapped twice.
            def _wrapped_bw_compiler(*args: P.args, **kwargs: P.kwargs) -> R:
                # [FUNCTION DOC] HELPER: Call the backward compiler untraced
                # and return its result with Dynamo disabled on it.
                # Note [Wrapping bw_compiler in disable]
                # The two disables here:
                # - stop TorchDynamo from trying to compile the bw_compiler function itself
                # - stop TorchDynamo from trying to compile our the generated backwards pass bw_compiler produces

                return disable(
                    disable(
                        bw_compiler_fn, reason="do not trace backward compiler function"
                    )(*args, **kwargs),  # type: ignore[misc]
                    reason="do not trace generated backwards pass",
                )

            _wrapped_bw_compiler._is_wrapped_bw_compiler = (  # pyrefly: ignore [missing-attribute]
                True
            )
            return _wrapped_bw_compiler

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]

        if isinstance(bw_compiler, SerializableAOTDispatchCompiler):
            if not getattr(bw_compiler.compiler_fn, "_is_wrapped_bw_compiler", False):
                bw_compiler.compiler_fn = wrap_bw_compiler(bw_compiler.compiler_fn)
        elif not getattr(bw_compiler, "_is_wrapped_bw_compiler", False):
            bw_compiler = wrap_bw_compiler(bw_compiler)

        self.kwargs["bw_compiler"] = bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if self.kwargs.get("fw_compiler", None) is nop:
            patch_config: contextlib.AbstractContextManager[Any] = patch(
                "functorch.compile.config.debug_assert", True
            )
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            with enable_aot_logging(), patch_config:
                # The decompositions thunk is resolved to a concrete table by the
                # callable check above, but pyrefly cannot narrow that through the
                # TypedDict subscript assignment, so the spread looks ill-typed.
                cg = aot_module_simplified(
                    gm,
                    example_inputs,
                    **self.kwargs,  # pyrefly: ignore [bad-argument-type]
                )
                counters["aot_autograd"]["ok"] += 1
                return disable(cg, reason="do not trace AOT-compiled graph")
        except TensorifyScalarRestartAnalysis:
            raise
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise


def aot_autograd(**kwargs: Unpack[AotAutogradKwargs]) -> AotAutograd:
    # [FUNCTION DOC] Factory: build an AotAutograd backend from the keyword
    # options.
    return AotAutograd(**kwargs)


def mem_efficient_fusion_kwargs(use_decomps: bool) -> dict[str, Any]:
    # [FUNCTION DOC] Return AOTAutograd options for memory efficient fusion:
    # TorchScript compile for forward and backward, and the min-cut
    # rematerialization partitioner; adds the default decompositions when
    # use_decomps is true.
    from functorch.compile import (
        default_decompositions,
        min_cut_rematerialization_partition,
        ts_compile,
    )

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
    }

    if use_decomps:
        # pyrefly: ignore [bad-typed-dict-key]
        kwargs["decompositions"] = default_decompositions

    return kwargs


def fake_tensor_unsupported(fn: Callable[[Any, list[Any], Any], R]) -> Any:
    # [FUNCTION DOC] Decorator for backends needing real inputs: returns a
    # wrapper that, with active dispatch modes disabled, converts fake input
    # tensors to real zero tensors and calls fn with the model, inputs and
    # kwargs.
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model: Any, inputs: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Defake the inputs under disabled modes and
        # delegate to the wrapped backend function.
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)  # type: ignore[call-arg]

    return wrapper


def device_from_inputs(example_inputs: Iterable[Any]) -> torch.device:
    # [FUNCTION DOC] Return the device of the first tensor in the inputs, or
    # cpu if there is none.
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")  # Default fallback


def dtype_from_inputs(example_inputs: Iterable[Any]) -> torch.dtype:
    # [FUNCTION DOC] Return the dtype of the first tensor in the inputs, or
    # float32 if there is none.
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            return x.dtype
    return torch.float32  # Default fallback
