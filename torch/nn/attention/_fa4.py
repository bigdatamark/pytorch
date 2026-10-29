"""UBER PROTOTYPE!!!"""
# mypy: allow-untyped-defs
# [MODULE DOC]
# Prototype Flash Attention 4 backend (compute capability 9.0 or 10.0). Wraps the forward and
# backward functions of an external FA4 python module (default flash_attn.cute.interface) and
# overrides the CUDA implementations of the aten flash attention ops (varlen/dense flash forward,
# in-place no-dropout forward, backward, and the SDPA flash forward/backward) via a dispatcher
# library. Contains support checks, layout helpers and the registration entry point. Import-time
# side effect: registers "FA4" in the flash attention registry (not activated). Keeps a
# module-level record of the module path chosen at registration.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from . import _registry


if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library


__all__ = [
    "register_flash_attention_fa4",
]


_FA4_MODULE_PATH: str | None = None


@dataclass
class _FA4Handle:
    # [CLASS DOC] HELPER: dataclass holding the dispatcher Library that keeps the FA4 kernel
    # overrides registered. Dropping the reference unregisters them.
    library: Library | None

    def remove(self) -> None:
        # [FUNCTION DOC] HELPER: release the library, unregistering the kernel overrides.
        # The registered module path stays recorded.
        self.library = None


@cache
def _get_device_major(device: torch.device) -> int:
    # [FUNCTION DOC] HELPER: return the CUDA compute capability major version of device
    # (memoized per device).
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa4(
    module_path: str = "flash_attn.cute.interface",
) -> _FA4Handle:
    # [FUNCTION DOC]
    # Activate FA4: import and validate the named module (must expose the forward and backward
    # functions), record its path in module state, register the aten kernel overrides, and
    # return a handle wrapping the registration library. Raises RuntimeError if the module
    # lacks the kernels, or ImportError if it cannot be imported.
    """
    Register FA4 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA4 implementation.
    """
    global _FA4_MODULE_PATH
    _ = _fa4_import_module(module_path)
    _FA4_MODULE_PATH = module_path
    return _FA4Handle(_fa4_register_kernels())


@cache
def _fa4_import_module(module_path: str) -> ModuleType:
    # [FUNCTION DOC] HELPER: import (memoized per path) and return the module; raises
    # RuntimeError if it lacks the forward or backward kernel functions.
    module = importlib.import_module(module_path)
    if not hasattr(module, "_flash_attn_fwd") or not hasattr(module, "_flash_attn_bwd"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA4 kernels")
    return module


def _fa4_register_kernels() -> Library:
    # [FUNCTION DOC] HELPER: create an aten implementation library for CUDA and register FA4
    # implementations for flash attention forward, in-place no-dropout forward, backward, and
    # the SDPA flash forward and backward ops. Returns the library.
    lib = Library("aten", "IMPL", "CUDA")
    lib.impl("_flash_attention_forward", _fa4_flash_attention_forward_impl, "CUDA")
    lib.impl(
        "_flash_attention_forward_no_dropout_inplace",
        _fa4_flash_attention_forward_no_dropout_inplace_impl,
        "CUDA",
    )
    lib.impl("_flash_attention_backward", _fa4_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa4_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa4_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa4_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    cum_seq_q: torch.Tensor | None,
    require_fp32: tuple[tuple[str, torch.Tensor], ...] = (),
) -> str | None:
    # [FUNCTION DOC] HELPER: checks shared by forward and backward. Returns an error string or
    # None. Rejects non-CUDA tensors, tensors on different devices, query dtype other than
    # fp16/bf16, any named tensor in require_fp32 that is not float32, dense query not 4D,
    # ragged query (cum_seq_q given) not 3D, CUDA unavailable, or compute capability major
    # outside 9 and 10.
    # KNOWN ISSUE: the CUDA availability check comes after tensors were already tested for CUDA.
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query dtype must be float16 or bfloat16"
    for name, tensor in require_fp32:
        if tensor.dtype != torch.float32:
            return f"{name} dtype must be float32"
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) not in (9, 10):
        return "FA4 requires compute capability 9.0 or 10.0"
    return None


def _fa4_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> str | None:
    # [FUNCTION DOC] HELPER: forward support check; returns the first unsupported condition as
    # a string, else None. Rejects nonzero dropout, return_debug_mask, alibi slopes, seqused_k
    # not int32 CUDA, paged KV (block_table) or split-KV (num_splits above 1) unless the
    # device major is 10, then applies the common checks (rewording the shared-device message).
    # KNOWN ISSUE: the device capability is queried before verifying the tensors are on CUDA,
    # so a CPU query may fail with a different error instead of a clean message.
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    major = _get_device_major(query.device)
    if block_table is not None and major != 10:
        return f"paged KV (block_table) not supported on SM {major}0"
    if num_splits is not None and num_splits > 1 and major != 10:
        return f"SplitKV (num_splits > 1) not supported on SM {major}0"
    error = _fa4_common_support_error(
        query,
        (query, key, value),
        cum_seq_q,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa4_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
) -> str | None:
    # [FUNCTION DOC] HELPER: backward support check; returns an error string or None. Rejects
    # nonzero dropout, then applies the common checks across all tensors, additionally
    # requiring logsumexp to be float32.
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    error = _fa4_common_support_error(
        query,
        (grad_out, query, key, value, out, logsumexp),
        cum_seq_q,
        require_fp32=(("logsumexp", logsumexp),),
    )
    if error is not None:
        return error
    return None


def _aten_to_fa4_window_size(val: int | None) -> int | None:
    # [FUNCTION DOC] HELPER: map the aten sentinel -1 (no window) to None; other values unchanged.
    """need to convert -1 to None for FA4"""
    return None if val == -1 else val


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: Unpack[Ts]) -> tuple[Unpack[Ts]]:
    # [FUNCTION DOC] HELPER: return a tuple with dimensions 1 and 2 swapped (views) for each
    # input, converting between batch-head-seq and batch-seq-head layouts.
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _fa4_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_q: int | None,
    max_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: call the registered FA4 module's forward function with softmax
    # scale, causal flag, window sizes (-1 mapped to None), cumulative lengths, max lengths,
    # contiguous seqused_k, page table, num_splits (default 1) and optional preallocated out,
    # requesting logsumexp. Returns (output, contiguous logsumexp). Raises RuntimeError if
    # FA4 has not been registered.
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)

    kwargs: dict[str, Any] = {
        "softmax_scale": scale,
        "causal": is_causal,
        "window_size_left": _aten_to_fa4_window_size(window_size_left),
        "window_size_right": _aten_to_fa4_window_size(window_size_right),
        "return_lse": True,
        "cu_seqlens_q": cu_seq_q,
        "cu_seqlens_k": cu_seq_k,
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
        "seqused_k": seqused_k.contiguous() if seqused_k is not None else None,
        "page_table": block_table,
        "num_splits": num_splits or 1,
        "out": out,
    }
    out, lse, *_ = module._flash_attn_fwd(query, key, value, **kwargs)
    return out, lse.contiguous()


def _fa4_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: call the registered FA4 module's backward function with the
    # forward tensors, grad_out, contiguous logsumexp, scale, causal flag, window sizes
    # (-1 mapped to None), cumulative lengths and the deterministic flag. Returns (dq, dk, dv).
    # Raises RuntimeError if FA4 has not been registered.
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    dq, dk, dv = module._flash_attn_bwd(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp.contiguous(),
        softmax_scale=scale,
        causal=is_causal,
        window_size_left=_aten_to_fa4_window_size(window_size_left),
        window_size_right=_aten_to_fa4_window_size(window_size_right),
        cu_seqlens_q=cu_seq_q,
        cu_seqlens_k=cu_seq_k,
        deterministic=deterministic,
    )
    return dq, dk, dv


def _fa4_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    compute_auxiliary: bool = True,
    num_splits: int | None = None,
):
    # [FUNCTION DOC] HELPER: FA4 override of the flash attention forward op. Raises RuntimeError
    # with the reason if the forward support check fails; otherwise runs the kernel. Returns
    # (output, logsumexp, rng_state, philox_offset, debug_mask); the last three are placeholder
    # zero/empty tensors when compute_auxiliary is true, else None (dropout unsupported).
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
        block_table,
        num_splits,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention forward unsupported: {error}")
    out, lse = _fa4_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
        out,
        block_table,
        num_splits,
    )
    if compute_auxiliary:
        rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
        philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    else:
        rng_state = None
        philox_offset = None
        debug_mask = None
    return out, lse, rng_state, philox_offset, debug_mask


def _fa4_flash_attention_forward_no_dropout_inplace_impl(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
):
    # [FUNCTION DOC] HELPER: override for the in-place no-dropout flash forward op. Writes the
    # result into the supplied out tensor via the general forward (no auxiliary outputs) and
    # returns only the logsumexp.
    _, lse, _, _, _ = _fa4_flash_attention_forward_impl(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        out=out,
        block_table=block_table,
        compute_auxiliary=False,
        num_splits=num_splits,
    )
    return lse


def _fa4_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    rng_state: torch.Tensor,
    unused: torch.Tensor,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
):
    # [FUNCTION DOC] HELPER: FA4 override of the flash attention backward op. Raises RuntimeError
    # with the reason if the backward support check fails; otherwise runs the backward kernel,
    # taking the deterministic flag from the global deterministic algorithms setting. Returns
    # (dq, dk, dv). rng_state, unused, max_q and max_k are accepted and ignored.
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention backward unsupported: {error}")
    deterministic = torch.are_deterministic_algorithms_enabled()
    dq, dk, dv = _fa4_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
    )
    return dq, dk, dv


def _fa4_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    # [FUNCTION DOC] HELPER: FA4 override of the SDPA flash forward op on dense batch-head-seq
    # tensors. Raises RuntimeError if unsupported; transposes inputs to batch-seq-head,
    # preallocates an output with the query's layout, runs the general forward on a transposed
    # view of it, and returns the 9-tuple (output, logsumexp, None, None, max query length,
    # max key length, rng_state, philox_offset, debug_mask).
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    # Pre-allocate output with query's strides (BHSD layout), then create
    # a BSHD view for the kernel. This ensures the returned output has
    # the same memory layout as the input query.
    out_bhsd = torch.empty_like(query)
    out_bshd = out_bhsd.transpose(1, 2)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    _, lse, rng_state, philox_offset, debug_mask = _fa4_flash_attention_forward_impl(
        q,
        k,
        v,
        None,
        None,
        max_q_flash,
        max_k_flash,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        out=out_bshd,
    )
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out_bhsd,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa4_scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    # [FUNCTION DOC] HELPER: FA4 override of the SDPA flash backward op on dense batch-head-seq
    # tensors. Raises RuntimeError if unsupported; transposes inputs to batch-seq-head, calls the
    # flash backward override, transposes the gradients back and returns (dq, dk, dv). The
    # max lengths come from the tensors, not the max_q/max_k arguments.
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA backward unsupported: {error}")
    q, k, v, o, go = _transpose_dense(query, key, value, out, grad_out)
    max_q = query.size(2)
    max_k = key.size(2)
    dq, dk, dv = _fa4_flash_attention_backward_impl(
        go,
        q,
        k,
        v,
        o,
        logsumexp,
        None,
        None,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )
    dq, dk, dv = _transpose_dense(dq, dk, dv)
    return dq, dk, dv


_registry.register_flash_attention_impl("FA4", register_fn=register_flash_attention_fa4)
