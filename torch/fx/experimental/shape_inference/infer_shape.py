# [MODULE DOC]
# Entry point for experimental symbolic shape inference over an FX GraphModule.
# infer_shape repeatedly traces the graph with fake tensors whose dimensions are
# symbolic, using error messages from failed attempts to solve for symbol values
# (see infer_symbol_values). mksym creates a symbolic integer from a ShapeEnv.
# No import-time side effects beyond imports.
import copy
from collections import defaultdict

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.types import IntLikeType
from torch.utils import _pytree


"""
This is the function that runs shape inference. It will modify the input graph module so that shapes are annotated.
"""


def infer_shape(
    gm: torch.fx.GraphModule, input_tensors: list[torch.Tensor]
) -> (
    tuple[torch.fx.GraphModule, list[torch.Tensor], FakeTensorMode, IntLikeType] | None
):
    # [FUNCTION DOC]
    # Tries to make gm traceable with symbolic fake inputs shaped like input_tensors. It
    # creates a fresh ShapeEnv and fake mode, allocates one symbol for the shared leading
    # (batch) dimension plus one per remaining dimension of the flattened inputs, and
    # loops up to twice that count: build symbolic inputs, trace gm with make_fx in
    # symbolic pre-dispatch mode and on a RuntimeError or ValueError feed the message to
    # infer_symbol_values to refine the symbol values. Returns (gm, input_tensors, fake_mode,
    # the batch symbol) on success, or None when attempts run out.
    # KNOWN ISSUE: the docstring-like string says the input graph is modified, but gm is
    # not changed here; the failure paths in each handler are guarded by a pointless truthiness check
    # on the exception, and the first symbol is shared by all inputs, assuming equal batch size.
    # Prepare environments
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    flatten_inputs, spec = _pytree.tree_flatten(input_tensors)
    dim_count = 1
    for input_tensor in flatten_inputs:
        dim_count += input_tensor.dim() - 1

    sample = {f"s{i}": 2 for i in range(dim_count)}
    init_symints: list[IntLikeType] = [
        mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
        for k, v in sample.items()
    ]
    symints: list[IntLikeType] = copy.deepcopy(init_symints)
    symbol_to_idx_dict = {f"s{i}": i for i in range(dim_count)}
    padding_constraints = defaultdict(list)  # type: ignore[var-annotated]

    complete_flag = False
    allowed_try_times = dim_count * 2

    while not complete_flag and allowed_try_times > 0:
        # Create symbolic input tensors
        with fake_mode:
            sym_tensors = []
            i = 1
            for input_tensor in flatten_inputs:
                curr_dim = input_tensor.dim()
                desired_size = [symints[0]] + [
                    symints[ii] for ii in range(i, i + curr_dim - 1)
                ]
                sym_tensor = torch.randn(desired_size)
                sym_tensors.append(sym_tensor)
                i += curr_dim - 1
            sym_tensors = _pytree.tree_unflatten(sym_tensors, spec)
        try:
            with fake_mode:
                make_fx(
                    gm,
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=True,
                    pre_dispatch=True,
                    _allow_fake_constant=True,
                )(*sym_tensors)
            complete_flag = True
            return (gm, input_tensors, fake_mode, symints[0])
        except RuntimeError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1
        except ValueError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1


def mksym(
    shape_env: ShapeEnv, value: int, source: LocalSource, dynamic_dim: DimDynamic
) -> IntLikeType:
    # [FUNCTION DOC] Creates a symbol in shape_env with the given hint value, source and
    # dynamism setting, and wraps it into a symbolic integer node with the same hint and source.
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )
