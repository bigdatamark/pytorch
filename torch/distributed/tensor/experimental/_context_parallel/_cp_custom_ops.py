# [MODULE DOC] Custom ops for flex-attention context parallelism. Defines the
# cplib::flex_cp_allgather op (all-gather of key and value along the sequence
# dim) and its backward op (reduce-scatter with sum), plus fake implementations
# for tracing and autograd registration wiring them together. Import-time side
# effect: registers the two custom ops and the autograd formula on the library.
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


@torch.library.custom_op("cplib::flex_cp_allgather", mutates_args=())
def flex_cp_allgather(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: c10d.GroupName
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Custom op: all-gathers key and value tensors along seq_dim over
    # the process group named pg_name. Inputs are made contiguous first; async
    # collective results are waited on so returned tensors are ready. Returns the
    # gathered (k, v) pair, each seq_dim times the group size larger. Communication only.
    k = k.contiguous()
    v = v.contiguous()
    k = funcol.all_gather_single(k, seq_dim, pg_name)
    v = funcol.all_gather_single(v, seq_dim, pg_name)
    if isinstance(k, funcol.AsyncCollectiveTensor):
        k = k.wait()
    if isinstance(v, funcol.AsyncCollectiveTensor):
        v = v.wait()
    return k, v


@flex_cp_allgather.register_fake
def _(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: c10d.GroupName
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of the all-gather op. Returns
    # empty tensors whose seq_dim size is multiplied by the process group size.
    shape_k = list(k.shape)
    shape_v = list(v.shape)
    shape_k[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    new_k = torch.empty(shape_k, dtype=k.dtype, device=k.device)
    new_v = torch.empty(shape_v, dtype=v.dtype, device=v.device)
    return new_k, new_v


@torch.library.custom_op("cplib::flex_cp_allgather_backward", mutates_args=())
def flex_cp_allgather_backward(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: c10d.GroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Custom op: backward of the K/V all-gather. Reduce-scatters
    # (sum) each full-length gradient along seq_dim over the named process group,
    # waiting on async results. Returns the local-shard (grad_k, grad_v).
    grad_k = funcol.reduce_scatter_single(grad_full_k, "sum", seq_dim, pg_name)
    if isinstance(grad_k, funcol.AsyncCollectiveTensor):
        grad_k = grad_k.wait()
    grad_v = funcol.reduce_scatter_single(grad_full_v, "sum", seq_dim, pg_name)
    if isinstance(grad_v, funcol.AsyncCollectiveTensor):
        grad_v = grad_v.wait()

    return grad_k, grad_v


@flex_cp_allgather_backward.register_fake
def _(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: c10d.GroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of the backward op. Returns
    # empty tensors whose seq_dim size is divided by the process group size.
    shape_k = list(grad_full_k.shape)
    shape_v = list(grad_full_v.shape)
    shape_k[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    new_grad_k = torch.empty(
        shape_k, dtype=grad_full_k.dtype, device=grad_full_k.device
    )
    new_grad_v = torch.empty(
        shape_v, dtype=grad_full_v.dtype, device=grad_full_v.device
    )
    return new_grad_k, new_grad_v


def _flex_cp_allgather_backward(
    ctx: Any, grad_full_k: torch.Tensor, grad_full_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    # [FUNCTION DOC] HELPER: Autograd backward formula. Calls the backward custom op
    # with seq_dim and pg_name saved on ctx; returns the two gradients plus
    # no-gradient markers for the seq_dim and group-name inputs.
    grad_k, grad_v = flex_cp_allgather_backward(
        grad_full_k, grad_full_v, ctx.seq_dim, ctx.pg_name
    )
    return grad_k, grad_v, None, None


def _flex_cp_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    # [FUNCTION DOC] HELPER: Autograd setup hook; stores the seq_dim and pg_name
    # inputs on ctx for use in backward. Mutates ctx only.
    _, _, ctx.seq_dim, ctx.pg_name = inputs


flex_cp_allgather.register_autograd(
    _flex_cp_allgather_backward, setup_context=_flex_cp_setup_context
)
