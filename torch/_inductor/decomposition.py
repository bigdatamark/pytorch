# [MODULE DOC] Inductor decomposition table. Builds the mapping from ATen/prim operators to
# decomposition functions used when tracing for Inductor: core ATen decompositions plus a curated
# list, minus an exclusion list. Defines Inductor-specific decompositions (matmul variants, cat, lerp,
# random ops, pooling, quantized ops, ...) registered into that table via register_decomposition,
# plus select_decomp_table to pick the table by config. A decomposition returning NotImplemented
# means: keep the original op. Import-time side effects: builds and mutates the module-global
# decompositions table, registers decompositions, and conditionally registers a distributed one.
# mypy: allow-untyped-decorators
import functools
import logging
import math
import operator
import sys
from collections.abc import Callable
from typing import Any, cast, TypeAlias, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
    core_aten_decompositions,
    get_decompositions,
    remove_decompositions,
)
from torch._decomp.decompositions import (
    _grid_sampler_2d as decomp_grid_sampler_2d,
    _index_add,
    embedding_dense_backward as decomp_embedding_dense_backward,
    pw_cast_for_opmath,
    pw_cast_for_opmath_non_tensor_args,
)
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._dynamo.utils import counters
from torch._environment import is_fbcode
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.utils import pad_listlike
from torch._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    suggest_memory_format,
    type_to_dtype,
)
from torch._refs import native_layer_norm as decomp_native_layer_norm
from torch.fx.experimental.symbolic_shapes import (
    guard_or_false,
    statically_known_true,
    sym_eq,
)

from . import config, inductor_prims
from .utils import (
    is_gpu,
    needs_fallback_due_to_atomic_add_limitations,
    use_scatter_fallback,
)


_T = TypeVar("_T")
_P = ParamSpec("_P")

_GenericOperator: TypeAlias = torch._ops.OperatorBase | torch._ops.OpOverloadPacket

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized
_quantized = torch.ops._quantized
quantized_decomposed = torch.ops.quantized_decomposed

inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.adaptive_max_pool2d,
        aten.adaptive_max_pool3d,
        aten.index_select,
        aten.addmv,
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.dist,
        aten.elu,
        aten.empty_like,
        aten.flip,
        aten.gelu,
        aten.hardtanh,
        aten.lcm,
        aten.leaky_relu,
        aten.linalg_vector_norm,
        aten._log_softmax,
        aten.max_pool2d_with_indices_backward,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten._batch_norm_with_update,
        aten._batch_norm_with_update_functional,
        aten._batch_norm_no_update,
        aten.batch_norm_backward,
        aten.native_batch_norm,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten.nll_loss2d_backward,
        aten.permute_copy,
        aten.rrelu_with_noise_backward,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.unbind_copy.int,
        aten.upsample_bilinear2d.vec,
        aten.hann_window,
        quantized.linear_dynamic_fp16_unpacked_weight,
        _quantized.wrapped_quantized_linear,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude: list[torch._ops.OpOverload | torch._ops.OpOverloadPacket] = [
    aten._unsafe_index,
    aten._unsafe_masked_index,
    aten._unsafe_masked_index_put_accumulate,
    aten._scaled_dot_product_flash_attention_for_cpu.default,  # See comments in torch/_decomp/decompositions.py
    aten._softmax_backward_data,
    aten.clamp_max,
    aten.clamp_min,
    aten.embedding_dense_backward,  # we fall back on xpu
    aten.native_layer_norm,  # we fall back on mtia
    aten.index_add,  # we conditionally call this decomp
    aten.glu,  # inductor lowers this directly
    aten.select_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.slice_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.silu,  # inductor uses exact eager decomposition
    aten.split.Tensor,  # inductor lowers this directly
    aten.squeeze,  # inductor lowers this directly
    aten.sum,  # inductor lowers this directly
    aten.unbind,  # inductor lowers this directly
    aten.baddbmm,  # upcasts to fp32, perf issue
    # FMA ops - we have lowerings that use FMA to match eager CUDA behavior
    aten.addcmul,
    aten.addcmul_,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_,
    aten.addcdiv,
    aten.addcdiv_,
    aten._foreach_addcdiv.Scalar,
    aten._foreach_addcdiv_,
    aten.lerp,
    aten.lerp_,
]

remove_decompositions(decompositions, decomps_to_exclude)


def register_decomposition(
    ops: _GenericOperator | list[_GenericOperator],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory registering a decomposition for one op or a list of ops into the Inductor
    # table. Logs a warning for ops already present (duplicates). Returns the registering decorator.
    for op in ops if isinstance(ops, list) else [ops]:
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


if torch.distributed.is_available():

    @register_decomposition([torch.ops._dtensor.shard_dim_alltoall.default])
    def shard_dim_alltoall_decomp(
        inp: torch.Tensor,
        gather_dim: int,
        shard_dim: int,
        group_name: str,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Decompose the DTensor shard_dim_alltoall op into view, movedim, clone, all_to_all_single, wait
        # and view steps. Returns NotImplemented when disabled by config, input is complex, dims are out
        # of range or equal, the group size lookup fails, or the shard dim is not provably divisible by
        # the group size. Side effect: increments an inductor counter. Dims may be negative.
        if not config.decompose_shard_dim_alltoall:
            return NotImplemented
        if inp.dtype.is_complex:
            return NotImplemented

        ndim = inp.dim()
        gather_dim = gather_dim + ndim if gather_dim < 0 else gather_dim
        shard_dim = shard_dim + ndim if shard_dim < 0 else shard_dim
        if not (0 <= gather_dim < ndim and 0 <= shard_dim < ndim):
            return NotImplemented
        if gather_dim == shard_dim:
            return NotImplemented

        try:
            from torch.distributed.distributed_c10d import (
                _get_group_size_by_name,
                GroupName,
            )

            group_size = _get_group_size_by_name(GroupName(group_name))
        except (RuntimeError, ValueError):
            return NotImplemented

        input_shape = list(inp.shape)
        shard_dim_size = input_shape[shard_dim]
        # Match eager shard_dim_alltoall semantics. DTensor pads uneven logical
        # shards before reaching this op, so the local shard dim should split
        # evenly across the process group here. If the guard fails at runtime,
        # Dynamo recompiles and may keep the original shard_dim_alltoall fallback.
        if not guard_or_false(sym_eq(shard_dim_size % group_size, 0)):
            return NotImplemented
        local_shard_dim_size = shard_dim_size // group_size

        pre_view_shape = list(input_shape)
        pre_view_shape[shard_dim] = local_shard_dim_size
        pre_view_shape.insert(shard_dim, group_size)

        post_view_shape = list(input_shape)
        post_view_shape[shard_dim] = local_shard_dim_size
        post_view_shape[gather_dim] = post_view_shape[gather_dim] * group_size

        out = aten.view.default(inp, pre_view_shape)
        out = aten.movedim.int(out, shard_dim, 0)
        out = aten.clone.default(out, memory_format=torch.contiguous_format)
        out = torch.ops._c10d_functional.all_to_all_single.default(
            out,
            [1] * group_size,
            [1] * group_size,
            group_name,
        )
        out = torch.ops._c10d_functional.wait_tensor.default(out)
        out = aten.movedim.int(out, 0, gather_dim)
        out = aten.clone.default(out, memory_format=torch.contiguous_format)
        counters["inductor"]["decompose_shard_dim_alltoall"] += 1
        return aten.view.default(out, post_view_shape)


@register_decomposition([aten.lerp.Scalar])
def _lerp_scalar(start: torch.Tensor, end: torch.Tensor, weight: float) -> torch.Tensor:
    # [FUNCTION DOC] Decompose lerp with scalar weight into a subtraction plus add with alpha, so FMA is used.
    # Converts end to start's suggested memory format; uses the end-based formula when weight
    # magnitude is at least one half, else the start-based formula.
    # Decompose into sub + add(alpha=weight) so that the add lowering emits FMA,
    # matching eager CUDA's dual-formula (see aten/src/ATen/native/Lerp.h).
    # Convert end to start's memory format so the output preserves start's layout,
    # matching eager TensorIterator behavior.
    fmt = suggest_memory_format(start)
    if fmt != torch.contiguous_format:
        end = end.contiguous(memory_format=fmt)
    diff = end - start
    if weight >= 0.5 or weight <= -0.5:
        return torch.add(end, diff, alpha=-(1.0 - weight))
    return torch.add(start, diff, alpha=weight)


@register_decomposition([aten.lerp.Tensor])
def _lerp_tensor(
    start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] Decompose lerp with tensor weight into subtraction plus addcmul, choosing per element the
    # end-based or start-based formula depending on weight magnitude (half or more).
    # Same dual-formula as foreach_lerp polyfill: decompose into sub + addcmul.
    # Convert end to start's memory format so the output preserves start's layout.
    fmt = suggest_memory_format(start)
    if fmt != torch.contiguous_format:
        end = end.contiguous(memory_format=fmt)
    diff = end - start
    mask = weight.abs() >= 0.5
    neg_omw = -(1.0 - weight)
    w = torch.where(mask, neg_omw, weight)
    base = torch.where(mask, end, start)
    return torch.addcmul(base, w, diff, value=1)


@register_decomposition([aten.embedding_dense_backward])
def _embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    # [FUNCTION DOC] Embedding dense backward: returns NotImplemented on XPU (fallback), else defers to the
    # generic decomposition.
    # TODO: check if XE4 still need this fallback
    # check torch.xpu.get_device_properties(grad_output.device).architecture
    if grad_output.is_xpu:
        return NotImplemented
    # We can write a util function to update decomp table if we have more ops to fallback.
    return decomp_embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_decomposition(aten.native_layer_norm)
def _native_layer_norm(
    input: torch.Tensor,
    normalized_shape: utils.ShapeType,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Layer norm: returns NotImplemented on MTIA (fallback), else defers to the reference decomposition.
    if input.is_mtia:
        return NotImplemented
    # We can write a util function to update decomp table if we have more ops to fallback.
    return decomp_native_layer_norm(input, normalized_shape, weight, bias, eps)


@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(
    symbol: torch.SymInt,
    *,
    min: torch.types.Number | None = None,
    max: torch.types.Number | None = None,
) -> None:
    # [FUNCTION DOC] No-op decomposition for sym_constrain_range_for_size; returns None.
    return


@register_decomposition([aten.clamp])
@pw_cast_for_opmath_non_tensor_args
def clamp(
    x: torch.Tensor,
    min: torch.types.Number | None = None,
    max: torch.types.Number | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Clamp as clamp_min followed by clamp_max, applying each bound only if given.
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x


# Inductor-specific SiLU decomposition for exact eager matching.
# The core decomposition uses x * sigmoid(x), but this form
# x / (1 + exp(-x)) matches eager execution more precisely.
@register_decomposition([aten.silu])
@pw_cast_for_opmath
def silu(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] SiLU computed as x divided by one plus exp of negative x, to match eager numerics (computed in opmath type).
    return x / (1 + x.neg().exp())


@register_decomposition([aten.full])
def full(
    size: list[int | torch.SymInt],
    fill_value: torch.types.Number,
    **kwargs: Any,
) -> torch.Tensor:
    # [FUNCTION DOC] Full: if no dtype is given, infer it from the Python type of fill_value and call full again;
    # otherwise returns NotImplemented.
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return torch.full(size, fill_value, **kwargs)
    return NotImplemented


@register_decomposition([aten.index_add])
def index_add(
    x: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    tensor: torch.Tensor,
    *,
    alpha: torch.types.Number = 1,
) -> torch.Tensor:
    # [FUNCTION DOC] Index_add via the generic decomposition (out of place, with alpha); returns NotImplemented for
    # bfloat16 outside fbcode so the kernel is used.
    # If we are not in fbcode and dtype is bfloat16
    # fallback to index_add kernel
    # see https://github.com/pytorch/pytorch/issues/137425 for details
    if not is_fbcode() and x.dtype == torch.bfloat16:
        return NotImplemented
    else:
        return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


# Not really sure how to put this into the main library.  PrimTorch wants
# empty_permuted to go to the prim, and typically users don't really want
# to decompose to empty_strided (but inductor is OK with it, because we are
# cool with strides and everything goes to empty_strided)
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(
    size: list[int | torch.SymInt],
    physical_layout: list[int],
    **kwargs: Any,
) -> torch.Tensor:
    # [FUNCTION DOC] Empty_permuted: empty tensor directly for identity layout, otherwise allocate in physical
    # order and permute back, so the strides follow the physical layout.
    is_identity = list(physical_layout) == list(range(len(physical_layout)))

    if is_identity:
        return torch.empty(size, **kwargs)
    else:
        perm = [0] * len(size)
        for p, l in enumerate(physical_layout):
            perm[l] = p
        return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)


@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias_sizes: list[int],
    stride: int | list[int],
    padding: int | list[int],
    dilation: int | list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Convolution backward: only handles the case where the bias gradient is requested on a GPU
    # device (else NotImplemented). Computes bias grad by summing (zeros if input channels are 0),
    # and calls the op again without the bias grad. Returns (grad_input, grad_weight, grad_bias).
    if not output_mask[2] or not is_gpu(grad_output.device.type):
        return NotImplemented
    if statically_known_true(input.size(1) == 0) and statically_known_true(
        bias_sizes[0] != 0
    ):
        # deal with the `input_channel=0` case
        grad_bias = grad_output.new_zeros(bias_sizes)
    else:
        grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)


@register_decomposition([aten.round.decimals])
def round_dec(x: torch.Tensor, decimals: int = 0) -> torch.Tensor:
    # [FUNCTION DOC] Round with decimals: scale by ten to the decimals, round, scale back.
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(
    self: torch.Tensor,
    batch2: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Batched matmul. Handles outer-product case via broadcast multiply; with coordinate descent
    # tuning on non CPU/MPS uses multiply and sum for vector cases; on CPU uses it for a single
    # row times single column (bumps a counter). Otherwise NotImplemented. out_dtype is unused.
    # Outer-product specialization: [B, M, 1] x [B, 1, N] -> [B, M, N].
    # This avoids introducing a reduction and maps directly to broadcasted mul.
    if statically_known_true(self.shape[2] == 1) and statically_known_true(
        batch2.shape[1] == 1
    ):
        return (self * batch2).contiguous()

    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/pytorch/pytorch/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[1] == 1) or statically_known_true(
            batch2.shape[2] == 1
        ):
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == "cpu":
        if statically_known_true(self.size(1) == 1) and statically_known_true(
            batch2.size(-1) == 1
        ):
            counters["inductor"]["decompose_bmm"] += 1
            return torch.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    return NotImplemented


@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    beta: torch.types.Number = 1,
    alpha: torch.types.Number = 1,
) -> torch.Tensor:
    # [FUNCTION DOC] Addmm decomposition for degenerate shapes: on non CPU/MPS device with inner dim 1 uses a
    # broadcast multiply (not for beta zero on CUDA); on CPU uses multiply and sum for 1xK by Kx1 or
    # tiny matrices. Applies alpha and beta. Otherwise NotImplemented. Bumps a counter. out_dtype unused.
    def add_input(out: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Apply alpha to the product and add the input scaled by beta.
        if alpha != 1:
            out = alpha * out
        if beta != 1:
            return out + beta * self
        return out + self

    if mat1.device.type not in ["cpu", "mps"]:
        if beta == 0 and mat1.device.type == "cuda":
            return NotImplemented
        if (
            statically_known_true(mat1.size(-1) == 1)
            and statically_known_true(mat1.size(0) != 1)
            and statically_known_true(mat2.size(1) != 1)
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = mat1 * mat2
            return add_input(out)

    if self.device.type == "cpu":
        if statically_known_true(mat1.size(0) == 1) and statically_known_true(
            mat2.size(-1) == 1
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = torch.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            return alpha * out + beta * self
        if (
            statically_known_true(mat1.size(0) == 1)
            and guard_or_false(mat2.size(0) <= 16)
            and guard_or_false(mat2.size(1) <= 16)
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
    return NotImplemented


@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(
    self: torch.Tensor,
    input2: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Mm decomposition: with coordinate descent tuning uses multiply and sum for vector cases; for
    # inner dim 1 uses a broadcast multiply (always off CPU/MPS, tiny equal-dtype inputs on CPU);
    # on CPU single row by single column uses multiply and sum. Otherwise NotImplemented. out_dtype unused.
    # Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning.
    # todo: Look into why and fix it (hopefully)

    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/pytorch/pytorch/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[0] == 1) or statically_known_true(
            input2.shape[1] == 1
        ):
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    # Non-CPU/MPS: always decompose. CPU: only for small tensors.
    if (
        statically_known_true(self.size(-1) == 1)
        and statically_known_true(self.size(0) != 1)
        and statically_known_true(input2.size(1) != 1)
    ):
        if self.device.type not in ["cpu", "mps"] or (
            self.device.type == "cpu"
            and statically_known_true(self.size(0) > 0)
            and statically_known_true(input2.size(0) == 1)
            and (self.dtype == input2.dtype)
            and guard_or_false((torch.numel(self) + torch.numel(input2)) <= 32)
        ):
            counters["inductor"]["decompose_mm"] += 1
            return self * input2
    if self.device.type == "cpu":
        if statically_known_true(self.size(0) == 1) and statically_known_true(
            input2.size(-1) == 1
        ):
            counters["inductor"]["decompose_mm"] += 1
            return torch.sum(
                self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
    return NotImplemented


# This pass does two things:
# - Eliminate cat when there is only one tensor input
# - Normalize cat calls, so that legacy empty 1-D tensors are removed (NB: we
#   don't remove ALL empty tensors, only the naughty ones)
@register_decomposition([aten.cat.default])
def cat(
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    # [FUNCTION DOC] Cat normalization: drops empty inputs (guarded, legacy empty 1-D tensors); with one tensor left
    # returns a clone or dtype-promoted copy; if some were dropped, re-calls cat; if one tensor is
    # repeated, uses expand and flatten; else NotImplemented (no more decomposition needed).
    def non_empty_tensor(x: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True unless the tensor is an empty 1-D tensor or empty along the cat dimension (guard-free).
        # For better or worse, this is a valid cat:
        #
        #   torch.cat([torch.randn(2, 2, 4), torch.randn(0), torch.randn(3, 2, 4)])
        #
        # We'd like to eliminate naughtiness like this for downstream passes
        # like split_cat.  The easiest way is to just drop such inputs
        # (guarding that they are non-zero).
        #
        # Is it permissible for this filtering to be size-oblivious?  A case
        # where this could matter is cat([(2, 2), (u0,)], dim=0); if u0
        # happened to be zero, we would have liked to have filtered it out.
        # But actually, the ONLY way this could have passed is if u0 == 0,
        # so by the time we get here we have already installed a deferred
        # runtime assert forcing u0 to be zero.  So if this hasn't happened,
        # we know that the unbacked SymInt has appropriate size and there are
        # no problems.
        if len(x.shape) == 1 and guard_or_false(x.shape[0] == 0):
            return False

        if dim < len(x.shape) and guard_or_false(x.shape[dim] == 0):
            return False

        return True

    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        # check dtype promotion
        promoted_dtype = elementwise_dtypes(
            *tensors,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )[1]
        filtered_t = filtered_tensors[0]
        return (
            filtered_t.clone()
            if promoted_dtype == filtered_t.dtype
            else filtered_t.to(dtype=promoted_dtype)
        )
    elif 1 < len(filtered_tensors) < len(tensors):
        # on the first call, when we remove empty tensors, we redispatch recursively
        return aten.cat.default(filtered_tensors, dim)

    # optimization, avoid concat for single, repeated input
    if len(filtered_tensors) > 1 and all(
        t is filtered_tensors[0] for t in filtered_tensors
    ):
        inp = filtered_tensors[0]
        shape = list(inp.shape)
        dim = dim + len(inp.shape) if dim < 0 else dim
        shape.insert(dim, len(filtered_tensors))
        return inp.unsqueeze(dim).expand(*shape).flatten(dim, dim + 1).clone()

    # when no 'filtering' has occurred, we raise to prevent infinite recursion (no more decomposition needed)
    return NotImplemented


@register_decomposition([aten.angle])
def angle(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Angle: for complex input, atan2 of imag and real (nan where real is nan); for real input,
    # pi for negatives, 0 otherwise, nan preserved (float result dtype via int-to-float promotion).
    if x.is_complex():
        return torch.where(
            torch.isnan(x.real), float("nan"), torch.atan2(x.imag, x.real)
        )

    # when x is real number
    #   if x >= 0, return 0
    #   if x < 0, return pi
    #   if x is nan, return nan
    _, dtype = elementwise_dtypes(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    pi = torch.scalar_tensor(math.pi, dtype=dtype, device=x.device)
    ret = torch.where(x < 0, pi, 0.0)
    return torch.where(torch.isnan(x), float("nan"), ret)


@register_decomposition([aten.add])
def add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: torch.types.Number | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Add for complex tensors only (NotImplemented otherwise or if last stride is not 1): views as
    # real pairs, adds in real form (alpha applied to y), and views back as complex. Handles 0-dim
    # inputs by reshaping and indexing result. Adds zero first to resolve conjugation.
    # Require both x and y to be complex tensors.
    x_is_complex_tensor = torch.is_tensor(x) and x.is_complex()
    y_is_complex_tensor = torch.is_tensor(y) and y.is_complex()
    if not x_is_complex_tensor or not y_is_complex_tensor:
        return NotImplemented

    def _requires_fallback(tensor: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True for non-scalar tensors whose last stride is not 1 (cannot be viewed as real).
        if tensor.ndim == 0:
            return False
        # Viewing complex tensors as their real dtype requires the last stride to be 1.
        return tensor.stride()[-1] != 1

    output_size_zero = False
    if x.ndim == 0 and y.ndim == 0:
        output_size_zero = True

    if x.ndim == 0:
        x = x.reshape(1)
    if y.ndim == 0:
        y = y.reshape(1)

    z = y
    if alpha is not None:
        z = alpha * y
    complex_type = torch.promote_types(x.dtype, y.dtype)

    if _requires_fallback(x) or _requires_fallback(z):
        return NotImplemented

    # For complex typed `x`, `x.view(x.real.dtype)` doubles the last dimension and can cause problem
    # when broadcasting the add.
    def reshape_tensor_complex(tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: View a real-typed tensor with an even last dim as shape (..., last/2, 2). Raises an
        # error if the last dimension is odd.
        """Reshape tensor from [*initial_dims, last_dim] to *initial_dims, last_dim/2, 2]"""
        # Get the current shape of the tensor
        *initial_dims, last_dim = tensor.shape

        # Check if the last dimension is even. We should never reach here since `x.view(x.real.dtype)`
        # doubles the last dimension for complex numbers.
        if last_dim % 2 != 0:
            raise AssertionError(
                "The size of the last dimension must be even to reshape it to [..., last_dim/2, 2]"
            )

        # Reshape the tensor
        new_shape = (*initial_dims, last_dim // 2, 2)
        reshaped_tensor = tensor.view(new_shape)
        return reshaped_tensor

    # Manually resolve complex tensors, as .is_conj() is unreliable after cloning during compilation.
    x = x + 0
    z = z + 0

    x_reshaped = reshape_tensor_complex(x.view(x.real.dtype))
    z_reshaped = reshape_tensor_complex(z.view(y.real.dtype))
    result = torch.flatten(x_reshaped + z_reshaped, start_dim=-2).view(complex_type)

    if output_size_zero:
        return result[0]
    return result


@register_decomposition([aten.conj_physical])
def conj_physical(self: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Conj_physical: NotImplemented for complex input; identity for real input.
    if self.is_complex():
        return NotImplemented
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Lift and detach_: return the input unchanged.
    return self


@register_decomposition([aten.fmin, prims.fmin])
def fmin(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fmin: choose self when other is NaN or greater than self, else other (NaN-ignoring minimum).
    return torch.where(torch.isnan(other) | (other > self), self, other)


@register_decomposition([aten.fmax, prims.fmax])
def fmax(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fmax: choose self when other is NaN or less than self, else other (NaN-ignoring maximum).
    return torch.where(torch.isnan(other) | (other < self), self, other)


@register_decomposition(prims.normal)
def normal(
    shape: list[int | torch.SymInt],
    *,
    mean: torch.types.Number,
    std: torch.types.Number,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Prim normal: sample standard normal values of the given shape (optionally with a generator),
    # scale by std and shift by mean.
    if generator is None:
        normal_samples = torch.randn(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    else:
        normal_samples = torch.randn(
            shape,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    return std * normal_samples + mean


@register_decomposition(aten.amax)
def amax(
    self: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Amax: for bool input uses any over the dims; otherwise NotImplemented.
    if self.dtype == torch.bool:
        return torch.any(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition(aten.amin)
def amin(
    self: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Amin: for bool input uses all over the dims; otherwise NotImplemented.
    if self.dtype == torch.bool:
        return torch.all(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition([aten.narrow_copy])
def narrow_copy(
    self: torch.Tensor,
    dim: int,
    start: int,
    length: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Narrow_copy: narrow then clone with contiguous format so strides are correct.
    # Use memory_format=torch.contiguous_format to ensure correct strides.
    # For empty tensors, a plain clone() preserves the input view's strides.
    return torch.narrow(self, dim, start, length).clone(
        memory_format=torch.contiguous_format
    )


@register_decomposition([aten.view_copy.default])
def view_copy_default(
    self: torch.Tensor,
    size: list[int | torch.SymInt],
) -> torch.Tensor:
    # [FUNCTION DOC] View_copy (size): view then clone.
    return aten.view(self, size).clone()


@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(
    self: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] View_copy (dtype): clone then reinterpret as the dtype.
    return self.clone().view(dtype)


def _get_shape_permutation_like(
    self: torch.Tensor,
) -> tuple[utils.ShapeType, utils.StrideType]:
    # [FUNCTION DOC] HELPER: For a tensor return (shape in physical dimension order, permutation back to logical order),
    # using the elementwise output physical layout.
    physical_layout, _ = utils.compute_elementwise_output_logical_to_physical_perm(self)
    shape = [self.shape[l] for l in physical_layout]

    permutation = [0] * len(shape)
    for p, l in enumerate(physical_layout):
        permutation[l] = p

    return (shape, permutation)


@register_decomposition(aten.full_like)
def full_like(
    self: torch.Tensor,
    fill_value: int | float,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    # [FUNCTION DOC] Full_like: with a non-preserve memory format, make a full tensor and convert it; otherwise
    # make a full tensor in physical shape and permute back to match input strides (clone). Defaults
    # dtype, layout and device from self. Raises an error if layout is not strided.
    dtype = self.dtype if dtype is None else dtype
    layout = self.layout if layout is None else layout
    device = self.device if device is None else device

    if memory_format != torch.preserve_format:
        result = torch.full(
            self.shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return result.to(memory_format=memory_format)

    else:
        if layout != torch.strided:
            raise AssertionError(f"expected torch.strided layout, got {layout}")
        shape, permutation = _get_shape_permutation_like(self)
        result = torch.full(
            shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        if permutation == list(range(len(permutation))):
            return result
        return result.permute(permutation).clone()


def _rand_like(
    rand_fn: Callable[..., torch.Tensor],
    self: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
    **kwargs: Any,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Shared implementation for random-like ops: build the random tensor in physical shape
    # with rand_fn and permute/clone to match self's layout, or convert to a requested memory format.
    # Defaults dtype and device from self.
    dtype = self.dtype if dtype is None else dtype
    device = self.device if device is None else device

    if memory_format != torch.preserve_format:
        return rand_fn(
            self.shape,
            dtype=dtype,
            device=device,
            **kwargs,
        ).to(memory_format=memory_format)

    shape, permutation = _get_shape_permutation_like(self)
    result = rand_fn(
        shape,
        dtype=dtype,
        device=device,
        **kwargs,
    )
    if permutation == list(range(len(permutation))):
        return result
    return result.permute(permutation).clone()


@decomp.register_decomposition([aten.rand_like], extra_random_decomps)
def rand_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    # [FUNCTION DOC] Rand_like via the shared helper with uniform random values.
    return _rand_like(torch.rand, self, **kwargs)


@decomp.register_decomposition([aten.randn_like], extra_random_decomps)
def randn_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    # [FUNCTION DOC] Randn_like via the shared helper with normal random values.
    return _rand_like(torch.randn, self, **kwargs)


@decomp.register_decomposition([aten.randint_like.default], extra_random_decomps)
def randint_like(self: torch.Tensor, high: int, **kwargs: Any) -> torch.Tensor:
    # [FUNCTION DOC] Randint_like with low zero and given high via the shared helper.
    return _rand_like(functools.partial(aten.randint.low, 0, high), self, **kwargs)


@decomp.register_decomposition([aten.randint_like.low_dtype], extra_random_decomps)
def randint_like_low(
    self: torch.Tensor, low: int, high: int, **kwargs: Any
) -> torch.Tensor:
    # [FUNCTION DOC] Randint_like with explicit low and high via the shared helper.
    return _rand_like(functools.partial(aten.randint.low, low, high), self, **kwargs)


@register_decomposition(aten.randint.default)
def randint(
    high: int,
    size: list[int | torch.SymInt],
    **kwargs: Any,
) -> torch.Tensor:
    # [FUNCTION DOC] Randint with default low of zero: calls the low/high variant.
    return aten.randint.low(0, high, size, **kwargs)


@register_decomposition(prims.uniform)
def uniform(
    shape: list[int | torch.SymInt],
    *,
    low: torch.types.Number,
    high: torch.types.Number,
    dtype: torch.dtype,
    device: torch.device,
    stride: list[int | torch.SymInt],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Prim uniform: sample random values for the required storage length of the given strides,
    # scale to [low, high) (optionally with a generator), and view as strided with shape and stride.
    storage_len = utils.compute_required_storage_length(
        cast(list[int], shape), cast(list[int], stride), 0
    )
    sample_shape: list[int | torch.SymInt] = [storage_len]
    if generator is None:
        rand_samples = torch.rand(sample_shape, dtype=dtype, device=device)
    else:
        rand_samples = torch.rand(
            sample_shape, generator=generator, dtype=dtype, device=device
        )
    res = (high - low) * rand_samples + low
    return res.as_strided(shape, stride)


@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Dynamic fp16 linear with unpacked weight: pack the weight with the fbgemm fp16 packer, then
    # run the fp16 weight linear with the output channel count taken from the weight's first dim.
    packed_weight = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(weight)
    return torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight(
        input, packed_weight, bias, weight.size()[0]
    )


@register_decomposition(_quantized.wrapped_quantized_linear.default)
def wrapped_quantized_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    input_zero_point: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    bias: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    out_channel: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Wrapped quantized linear: prepack the weight with its scale, zero point and bias, then
    # call the prepacked quantized linear with input and output quantization parameters.
    packed_weight = torch.ops._quantized._wrapped_linear_prepack(
        weight, weight_scale, weight_zero_point, bias
    )
    return torch.ops._quantized._wrapped_quantized_linear_prepacked(
        input,
        input_scale,
        input_zero_point,
        packed_weight,
        out_scale,
        out_zero_point,
        out_channel,
    )


@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Decompose embedding_bag byte unpack: split the last 8 bytes of each row into float32 scale
    # and offset (bitcast from bytes, honoring byte order) and dequantize the remaining bytes.
    def bitcast_u8_to_f32(u8: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Combine four uint8 slices into int32 and reinterpret as float32 with a trailing axis,
        # respecting system byte order.
        x, y, z, w = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
        if sys.byteorder == "little":
            return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]
        else:
            return ((x << 24) + (y << 16) + (z << 8) + w).view(torch.float32)[..., None]

    scales = bitcast_u8_to_f32(packed[..., -8:-4])
    offsets = bitcast_u8_to_f32(packed[..., -4:])
    return packed[..., :-8].to(torch.float32) * scales + offsets


@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(
    a: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Grid_sampler_2d via the generic decomposition. Chooses to skip expanding the grid for CPU
    # bilinear contiguous inputs (performance hack), otherwise expands it.
    # KNOWN ISSUE: a trailing comment about foreach addcmul/addcdiv sits after the return and is unreachable.
    # We do not expand the grid (_expand_grid=False) on cpu for performance reasons
    # Experimenting locally it was found that compiled CUDA code is accelerated by ~5x
    # and CPU code by ~2x on bicubic mode, if we expand the grid from (N, H, W, 2) into (N, C, H, W, 2)
    # However, this leads to a slowdown around ~0.8x on CPU bilinear mode, channels first.
    # Thus we apply this hack to not expand the grid for this case.
    _expand_grid = not (
        a.device == torch.device("cpu")
        and interpolation_mode == 0
        and a.is_contiguous(memory_format=torch.contiguous_format)
    )

    output = decomp_grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        _expand_grid=_expand_grid,
    )
    return output

    # _foreach_addcmul and _foreach_addcdiv decompositions removed —
    # the inductor lowering in lowering.py handles these with FMA.


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    weight: torch.types.Number,
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Foreach lerp with a scalar weight. Uses foreach sub and addcmul when all tensors share a
    # dtype and device and are not half precision; otherwise computes per tensor in opmath type
    # with the end-based formula for weights of large magnitude. Handles complex weights by
    # casting to the tensor dtype. Returns a list of results.
    def opmath_dtype(t: torch.Tensor) -> torch.dtype:
        # [FUNCTION DOC] HELPER: Float32 for half/bfloat16 tensors, else the tensor's dtype.
        if t.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return t.dtype

    def cast_weight(dtype: torch.dtype) -> float | complex:
        # [FUNCTION DOC] HELPER: Weight as a Python number cast to dtype; drops the imaginary part for non-complex
        # dtypes (using it when nonzero).
        if isinstance(weight, complex) and not dtype.is_complex:
            if weight.imag != 0:
                return torch.tensor(weight, dtype=dtype).item()
            return torch.tensor(weight.real, dtype=dtype).item()
        return torch.tensor(weight, dtype=dtype).item()

    def use_high_formula(dtype: torch.dtype) -> bool:
        # [FUNCTION DOC] HELPER: Whether to use the end-based formula: magnitude at least one half (squared magnitude
        # at least a quarter for complex weights) after casting to dtype.
        casted_weight = cast_weight(dtype)
        if isinstance(casted_weight, complex):
            return (
                casted_weight.real * casted_weight.real
                + casted_weight.imag * casted_weight.imag
                >= 0.25
            )
        return casted_weight >= 0.5 or casted_weight <= -0.5

    def scalar_tensor(
        dtype: torch.dtype, device: torch.device, high_weight: bool
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Weight as a scalar tensor in dtype/device, replaced by minus (one minus weight) for
        # the high-weight formula.
        if isinstance(weight, complex) and not dtype.is_complex:
            if weight.imag != 0:
                weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
                if high_weight:
                    return -(1.0 - weight_tensor)
                return weight_tensor
            weight_tensor = torch.tensor(weight.real, dtype=dtype, device=device)
            if high_weight:
                return -(1.0 - weight_tensor)
            return weight_tensor
        weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
        if high_weight:
            return -(1.0 - weight_tensor)
        return weight_tensor

    dtype = start_tensors[0].dtype
    device = start_tensors[0].device
    compute_dtype = opmath_dtype(start_tensors[0])
    use_foreach = dtype not in (torch.float16, torch.bfloat16) and all(
        t.dtype == dtype and t.device == device for t in start_tensors
    )
    if use_foreach:
        high_weight = use_high_formula(compute_dtype)
        bases = end_tensors if high_weight else start_tensors
        diff = aten._foreach_sub.List(end_tensors, start_tensors)
        scalar = scalar_tensor(dtype, device, high_weight)
        return aten._foreach_addcmul.Scalar(bases, [scalar] * len(diff), diff)

    return [
        torch.addcmul(
            (end if high_weight else t).to(compute_dtype),
            scalar_tensor(compute_dtype, t.device, high_weight),
            end.to(compute_dtype) - t.to(compute_dtype),
        ).to(t.dtype)
        for t, end, compute_dtype, high_weight in (
            (
                t,
                end,
                opmath_dtype(t),
                use_high_formula(opmath_dtype(t)),
            )
            for t, end in zip(start_tensors, end_tensors)
        )
    ]


@register_decomposition(aten._foreach_lerp.ScalarList)
def _foreach_lerp_scalarlist(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    scalars: list[torch.types.Number],
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Foreach lerp with per-tensor scalars: start plus (end minus start) times scalar via foreach ops.
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.ScalarList(
            aten._foreach_sub.List(end_tensors, start_tensors), scalars
        ),
    )


@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] MIOpen batch norm via native batch norm. Returns the three outputs in training mode; in
    # eval mode the saved mean and var outputs are replaced with empty tensors.
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )

    if training:
        return (a, b, c)
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
    )


@functools.cache
def fast_random_decomps() -> dict[Any, Callable[..., Any]]:
    # [FUNCTION DOC] Cached table: the Inductor decompositions merged with the extra random decompositions.
    return {**decompositions, **extra_random_decomps}


# TODO(aakhundov): replace this (and the above) Any by more
# specific type and fix all the cascading mypy errors
def select_decomp_table() -> dict[Any, Callable[..., Any]]:
    # [FUNCTION DOC] Return the decomposition table by config: plain table if fallback_random is set; if
    # fallback_embedding_bag_byte_unpack is set, removes that op's decomposition from the shared
    # table (in place) and returns it; otherwise the cached table including random decompositions.
    # KNOWN ISSUE: the removal permanently mutates the global table for later callers.
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    if config.fallback_embedding_bag_byte_unpack:
        # remove q_embedding_bag_byte_unpack_decomp from decompositions
        decompositions.pop(torch.ops.quantized.embedding_bag_byte_unpack.default, None)
        return decompositions
    result = fast_random_decomps()
    return result


@register_decomposition(aten.masked_scatter)
def masked_scatter(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    # [FUNCTION DOC] Masked_scatter when the backend supports masked scatter with index: broadcast, compute source
    # positions with a cumulative sum of the mask, gather with an unsafe masked index, and select.
    # Otherwise NotImplemented.
    from .codegen.common import BackendFeature, has_backend_feature

    if has_backend_feature(self.device, BackendFeature.MASKED_SCATTER_WITH_INDEX):
        # This two-step algorithm is the same as eager CUDA, for eager CPU we
        # use a 1-shot serial iteration.
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        self_flat, mask_flat, source_flat = (x.flatten() for x in (self, mask, source))
        result = aten._unsafe_masked_index(source_flat, mask_flat, [source_idx], 0)
        return torch.where(mask_flat, result, self_flat).view(self.shape)
    return NotImplemented


@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Quantized choose_qparams: compute scale from min/max with eps floor, zero point clamped to the
    # quantization range. Returns (scale as float64, zero point as int64).
    min_val, max_val = torch.aminmax(input)
    scale = (max_val - min_val) / float(quant_max - quant_min)
    scale = torch.max(scale, torch.Tensor([eps]))
    zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale.to(torch.float64), zero_point.to(torch.int64)


@register_decomposition(aten.put)
def put(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Put: flatten self, index_put at flat indices with the reshaped source, reshape back.
    flattened = self.flatten()
    flattened = torch.index_put(
        flattened, [index], source.reshape(index.shape), accumulate
    )
    return flattened.reshape(self.shape)


@register_decomposition(aten.put_)
def put_(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] In-place put: compute out of place put and copy into self; returns self.
    out = aten.put(self, index, source, accumulate=accumulate)
    return self.copy_(out)


@register_decomposition(aten._softmax_backward_data.default)
@pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Softmax backward: grad times output minus output times its sum along dim (using FMA), cast
    # to input_dtype if different, and made contiguous.
    # KNOWN ISSUE: commented-out CPU special case remains in the body.
    new_grad_output = grad_output * output
    sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
    # grad_input = new_grad_output - output * sum_new_grad
    grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input.contiguous()


@register_decomposition(aten.index_reduce)
def index_reduce(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduction_type: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    # [FUNCTION DOC] Index_reduce: checks all tensors are on one device (raises an error otherwise). For mean
    # (when atomic add is usable) uses index_add and counts; if scatter fallback applies returns
    # NotImplemented; otherwise expands the index and uses scatter_reduce.
    torch._check(
        self.device == index.device and self.device == src.device,
        lambda: (
            f"index_reduce(): self, index and source expected to be in the same device, "
            f"but got (self) {self.device}, (index) {index.device}, "
            f"and (source) {src.device}"
        ),
    )
    if reduction_type == "mean" and not needs_fallback_due_to_atomic_add_limitations(
        self.dtype
    ):
        true_division = self.dtype.is_floating_point or self.dtype.is_complex
        ones = torch.ones_like(src)
        if include_self:
            out = self
            counts = torch.ones_like(self).index_add(dim, index, ones)
        else:
            out = self.index_fill(dim, index, 0)
            counts = torch.zeros_like(self).index_add(dim, index, ones)
            counts = counts.masked_fill(counts < 1, 1)
        out = out.index_add(dim, index, src)
        return out / counts if true_division else out // counts

    if use_scatter_fallback(
        aten.scatter_reduce_.two,
        reduction_type,
        self.dtype,
        src.dtype,
        src.device.type,
        True,
    ):
        return NotImplemented

    # pyrefly: ignore [missing-attribute]
    repeats = self.shape[dim + 1 :].numel() * self.shape[:dim].numel()
    index_shape = (index.numel(), *self.shape[dim + 1 :], *self.shape[:dim])
    perm = (*range(self.ndim - dim, self.ndim), 0, *range(1, self.ndim - dim))
    scatter_index = (
        index.to(torch.int64)
        .repeat_interleave(repeats)
        .reshape(index_shape)
        .permute(perm)
    )
    return self.scatter_reduce(
        dim,
        scatter_index,
        src,
        reduction_type,
        include_self=include_self,
    )


def _max_pool_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None,
    padding: int | list[int],
    dilation: int | list[int],
    ceil_mode: bool,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Shared max pool with indices via low memory prims (values with offsets, offsets
    # converted to indices). Normalizes dilation, padding, stride and kernel to dim-length lists.
    # Returns NotImplemented if lowering requires fallback or the window exceeds the int8 maximum.
    if dilation == 1:
        dilation = [1] * dim

    if padding == 0:
        padding = [0] * dim

    if not stride:
        stride = kernel_size

    # pyrefly: ignore [bad-assignment]
    kernel_size = pad_listlike(kernel_size, dim)
    # pyrefly: ignore [bad-assignment]
    dilation = pad_listlike(dilation, dim)
    # pyrefly: ignore [bad-assignment]
    padding = pad_listlike(padding, dim)
    # pyrefly: ignore [bad-assignment]
    stride = pad_listlike(stride, dim)

    window_size = functools.reduce(operator.mul, kernel_size)
    # We fallback when using non-default dilation or when the window size is too large
    if (
        torch._inductor.lowering.should_fallback_max_pool_with_indices(
            kernel_size, n_dim=dim
        )
        or window_size > torch.iinfo(torch.int8).max
    ):
        return NotImplemented

    vals, offsets = prims._low_memory_max_pool_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    indices = prims._low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-dim:],
        stride,
        padding,
        dilation,
    )
    return vals, indices


@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Max_pool2d_with_indices using the shared helper with two spatial dims.
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=2
    )


@register_decomposition(aten.max_pool3d_with_indices)
def max_pool3d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = None,
    padding: int | list[int] = 0,
    dilation: int | list[int] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Max_pool3d_with_indices using the shared helper with three spatial dims.
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=3
    )


@register_decomposition(aten.searchsorted.Scalar)
def searchsorted_scalar(
    sorted_sequence: torch.Tensor,
    self: torch.types.Number,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Searchsorted for a scalar value: search a one-element tensor and take the first result.
    return aten.searchsorted(
        sorted_sequence,
        torch.tensor([self], device=sorted_sequence.device),
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )[0]


@register_decomposition(aten.bucketize.Scalar)
def bucketize_scalar(
    self: torch.types.Number,
    boundaries: torch.Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Bucketize for a scalar value: bucketize a one-element tensor and squeeze the result.
    return aten.bucketize(
        torch.tensor([self], device=boundaries.device),
        boundaries,
        out_int32=out_int32,
        right=right,
    ).squeeze(0)


@decomp.register_decomposition(aten.rrelu_with_noise_functional, extra_random_decomps)
def rrelu_with_noise_functional(
    self: torch.Tensor,
    noise: torch.Tensor,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Functional rrelu with noise: in training, samples uniform slopes for non-positive inputs and
    # returns (output, noise); otherwise leaky relu with the mean slope and an empty noise tensor.
    if training:
        not_positive = self <= 0
        r = aten.uniform(self, lower, upper, generator=generator)
        output = torch.where(not_positive, self * r, self)
        noise_out = torch.where(not_positive, r, 1)
        return output, noise_out
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu(self, negative_slope), torch.Tensor()


@register_decomposition(aten.repeat_interleave.Tensor)
def repeat_interleave_Tensor(
    repeat: torch.Tensor,
    output_size: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Repeat_interleave with repeats tensor: only with a known int output_size, not under compile-time
    # autotune, not on MPS; uses cumulative sum and searchsorted over positions, clamped to the last
    # index. Otherwise NotImplemented. Raises an error if repeat is not 1-D int32/int64.
    if config.triton.autotune_at_compile_time:
        # We can't compile-time auto-tune this because
        # it expects specific data in `repeat`
        return NotImplemented
    if output_size is None or type(output_size) is not int:
        return NotImplemented
    if repeat.device.type == "mps":
        return NotImplemented
    if repeat.dtype not in [torch.int32, torch.int64]:
        raise AssertionError(
            f"expected repeat dtype int32 or int64, got {repeat.dtype}"
        )
    if repeat.ndim != 1:
        raise AssertionError(f"expected repeat.ndim == 1, got {repeat.ndim}")
    cumsum = repeat.cumsum(0)
    pos = torch.arange(output_size, device=repeat.device)
    indices = torch.searchsorted(
        cumsum, pos, out_int32=(repeat.dtype == torch.int32), right=True
    )
    return torch.clamp(indices, max=repeat.size(0) - 1)


# intentionally not registered
def conv1d_to_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int] = (1,),
    padding: tuple[int] = (0,),
    dilation: tuple[int] = (1,),
    groups: int = 1,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Not registered. Rewrite a 1-D convolution as 2-D by adding a trailing unit dim to
    # input and weight and using scalar stride, padding and dilation from the 1-tuples, then
    # squeeze it. Raises an error unless input and weight are 3-D.
    # Shapes:
    # input:  (N, C_in, L_in)
    # weight: (C_out, C_in // groups, K)
    # bias:   (C_out,)
    if not (input.dim() == 3 and weight.dim() == 3):
        raise AssertionError("Expect (N,C_in,L) and (C_out,C_in//groups,K)")

    # pyrefly: ignore [bad-assignment]
    stride = stride[0]
    # pyrefly: ignore [bad-assignment]
    padding = padding[0]
    # pyrefly: ignore [bad-assignment]
    dilation = dilation[0]

    # Unsqueeze to make input 2D: (N,C,L) -> (N,C,L,1)
    input_2d = input.unsqueeze(-1)
    # Unsqueeze kernel: (C_out,C_in/groups,K) -> (C_out,C_in/groups,K,1)
    weight_2d = weight.unsqueeze(-1)

    # Call conv2d with adjusted args
    out_2d = aten.conv2d.default(
        input_2d,
        weight_2d,
        bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
    )

    # Squeeze dummy dimension back out: (N,C_out,L_out,1) -> (N,C_out,L_out)
    return out_2d.squeeze(-1)
