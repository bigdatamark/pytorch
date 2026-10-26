# mypy: allow-untyped-defs

# [MODULE DOC] Defines the aoti_call_delegate higher-order op, which calls an
# AOTInductor-compiled submodule from an ExportedProgram graph. Provides the HOP object
# and its implementations for real execution, proxy tracing, fake tensor mode and
# functionalization. Import time: creates the HOP and registers fallthroughs for the
# PythonDispatcher, PythonTLSSnapshot, ADInplaceOrView and AutocastCPU dispatch keys.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


AOTI_LOWERED_MODULE = "AOTInductorEPModule/AOTInductorRunnerWrapper"


class AOTICallDelegate(HigherOrderOperator):
    """aoti_call_delegate is a HOP for calling AOTInductor lowered submodule in ExportedProgram.

    It has the following signature:
    aoti_call_delegate(
        lowered_module: Union[AOTInductorEPModule, AOTInductorRunnerWrapper]
        original_gm:fx.GraphModule,
        weight_args: List[Tensor],
        input_args: List[Tensor],
    ) -> outputs: List[Tensor]

    where,
    - lowered_module is the AOTInductor lowered submodule, backed by compiled .so file, supporting real tensor inputs
    - original_gm is the stateless version of the original GraphModule before lowering, allowing FakeTensor propagation
    - weight_args is the list of weights in original GraphModule, including parameters and buffers
    - input_args is the list of flatten inputs
    """
    # [CLASS DOC] HigherOrderOperator named "aoti_call_delegate"; stateless. Takes the
    # lowered module, the original graph module, weights and inputs; returns output tensors.

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the HOP under the name aoti_call_delegate.
        super().__init__("aoti_call_delegate")

    def __call__(
        self,
        lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
        original_gm: torch.fx.GraphModule,
        weight_args: list[torch.Tensor],
        input_args: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC] Dispatches the four arguments through the base HOP call and
        # returns the resulting list of output tensors.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(lowered_module, original_gm, weight_args, input_args)


aoti_call_delegate = AOTICallDelegate()
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)


@aoti_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def call_delegate_cpu(
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Eager implementation (CompositeExplicitAutograd). Converts FX immutable
    # dicts/lists inside weights+inputs to plain dicts/lists. If any argument is a fake
    # tensor, runs the stateless original_gm on all of them and returns its output.
    # Otherwise drops the weights and runs the AOTI module on the inputs: an
    # AOTInductorRunnerWrapper is called with unpacked inputs, an AOTInductorEPModule with
    # the inputs as one sequence. Raises RuntimeError for any other module type.
    # FX creates this immutable_dict/list concept. Get rid of this.
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        weight_args + input_args,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    has_fake_args = any(is_fake_tensor(arg) for arg in new_args)
    if has_fake_args:
        # use stateless original_gm for tracing with fake tensors
        fake_out = original_gm(*new_args)
        return fake_out
    else:
        # use AOTI Runner for real tensors
        new_input_args = new_args[len(weight_args) :]
        if type(lowered_module).__name__ == "AOTInductorRunnerWrapper":
            return lowered_module(*new_input_args)  # type: ignore[misc]
        elif type(lowered_module).__name__ == "AOTInductorEPModule":
            return lowered_module(new_input_args)  # type: ignore[misc]
        else:
            raise RuntimeError(
                f"Unexpected lowered_module type: {type(lowered_module)}."
            )


def trace_aoti_call_delegate(
    proxy_mode, func_overload, lowered_module, original_gm, weight_args, input_args
):
    # [FUNCTION DOC] HELPER: Traces the HOP into the FX graph. Registers lowered_module and
    # original_gm as submodules of the tracer root (mutating it), emits a call_function node
    # named aoti_call_delegate with proxied args, computes real outputs with proxy tracing
    # disabled via call_delegate_cpu, and returns them tracked against the output proxy.
    proxy_mode.tracer.root.register_module("lowered_module", lowered_module)
    proxy_mode.tracer.root.register_module("original_gm", original_gm)

    node_args = (lowered_module, original_gm, weight_args, input_args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="aoti_call_delegate"
    )
    with disable_proxy_modes_tracing():
        out = call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@aoti_call_delegate.py_impl(ProxyTorchDispatchMode)
def call_delegate_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
):
    # [FUNCTION DOC] Proxy-mode dispatch implementation: delegates to
    # trace_aoti_call_delegate and returns the traced outputs.
    res = trace_aoti_call_delegate(
        mode, aoti_call_delegate, lowered_module, original_gm, weight_args, input_args
    )
    return res


@register_fake(aoti_call_delegate, skip_cache=True)
def call_delegate_fake_tensor_mode(
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Fake tensor mode implementation (cache skipped): reuses
    # call_delegate_cpu, which runs the original graph module on the fake inputs.
    return call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)


@aoti_call_delegate.py_functionalize_impl
def call_delegate_functionalize(
    ctx,
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
):
    # [FUNCTION DOC] Functionalization implementation: unwraps functional tensors in weights
    # and inputs, redispatches the HOP to the next level, and re-wraps the result. The op
    # is assumed not to mutate its inputs. Returns the wrapped outputs.
    unwrapped_weight_args = tuple(
        ctx.unwrap_tensors(weight_arg) for weight_arg in weight_args
    )
    unwrapped_input_args = tuple(
        ctx.unwrap_tensors(input_arg) for input_arg in input_args
    )
    with ctx.redispatch_to_next():
        res = aoti_call_delegate(
            lowered_module,
            original_gm,
            unwrapped_weight_args,  # type: ignore[arg-type]
            unwrapped_input_args,  # type: ignore[arg-type]
        )
        return ctx.wrap_tensors(res)
