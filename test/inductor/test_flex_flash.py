# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the FlexAttention "FLASH" backend (CuTeDSL flash attention) in Inductor.
# Contains small score_mod / mask_mod factories, helpers that compare the FLASH backend against
# the TRITON backend and an fp32 eager reference, case tables of dataclasses, and device-generic
# test classes covering forward, backward, GQA/MQA, block masks, captured buffers, scalar aux
# values, vec-size selection and error paths. No import-time side effects beyond defining helpers;
# most tests need CUDA (SM80+/SM90+/SM100) and the flash-attention CuTe package.

import functools
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

import torch
import torch._inductor.kernel.flex.flex_flash_attention as flex_flash_attention_module
from torch._dynamo.testing import CompileCounterWithBackend, EagerAndRecordGraphs
from torch._inductor.kernel.flex.flex_flash_attention import (
    _flash_attention_unavailable_message,
    _hierarchical_indexer_cute,
    ensure_flash_available,
    flash_supports_aux_scalars,
    HierarchicalIndex,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    AuxRequest,
    create_block_mask,
    flex_attention,
)
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_cuda import (
    IS_SM90,
    PLATFORM_SUPPORTS_FP8,
    SM120OrLater,
    SM80OrLater,
    SM90OrLater,
    xfailIfSM120OrLater,
    xfailIfSM12X,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    e4m3_type,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    decorateIf,
    DeterministicGuard,
    parametrize,
)


IS_SM8X = SM80OrLater and not SM90OrLater


def _times_two(score, _b, _h, _m, _n):
    # [FUNCTION DOC] HELPER: score_mod that doubles the score; the index arguments are ignored.
    return score * 2


def _causal(score, _b, _h, token_q, token_kv):
    # [FUNCTION DOC] HELPER: score_mod applying causal masking: keeps the score where the query
    # index is at least the key/value index, otherwise yields negative infinity.
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _causal_mask(_b, _h, token_q, token_kv):
    # [FUNCTION DOC] HELPER: mask_mod returning a boolean tensor that is true where the query
    # token index is greater than or equal to the key/value token index (causal mask).
    return token_q >= token_kv


def _rel_bias(score, _b, _h, token_q, token_kv):
    # [FUNCTION DOC] HELPER: score_mod adding a relative-position bias (query index minus
    # key/value index) to the score.
    return score + (token_q - token_kv)


def _score_squared(score, _b, _h, _q, _k):
    # [FUNCTION DOC] HELPER: score_mod returning the score multiplied by itself (nonlinear in
    # score, exercises backward through the score_mod).
    return score * score


def _softcap(score, _b, _h, _q, _k):
    # [FUNCTION DOC] HELPER: score_mod implementing tanh soft-capping with a fixed cap of 50: the
    # score is divided by the cap, passed through tanh, and rescaled by the cap.
    cap = 50.0
    return torch.tanh(score / cap) * cap


def _head_offset(score, _b, h, _q, _k):
    # [FUNCTION DOC] HELPER: score_mod adding a per-head offset equal to the head index times 0.1.
    return score + h * 0.1


def _distance_decay(score, _b, _h, q_idx, kv_idx):
    # [FUNCTION DOC] HELPER: score_mod scaling the score by a decay factor that shrinks with the
    # squared distance between query and key/value index (decay = 1 / (1 + 0.0001 * distance^2)).
    distance_sq = (q_idx - kv_idx) * (q_idx - kv_idx)
    decay = 1.0 / (1.0 + distance_sq * 0.0001)
    return score * decay


@contextmanager
def force_flex_flash_score_mod_vec_size(vec_size: int):
    # [FUNCTION DOC] HELPER: context manager that forces the FLASH forward config list so that any
    # kernel with both a score_mod and aux tensors uses the given score_mod vector size; all other
    # kernels get the default FlexFlashConfig. It temporarily patches get_flex_flash_fwd_configs
    # in the flex_flash_attention module and restores it on exit. Takes vec_size (int).
    def configs(
        has_score_mod,
        has_aux_tensors,
        device=None,
        score_mod_graph_module=None,
        score_mod_other_buffers=(),
        has_mask_mod=False,
        has_mask_aux_tensors=False,
        mask_mod_graph_module=None,
        mask_mod_other_buffers=(),
        aux_scalar_symbols=(),
    ):
        # [FUNCTION DOC] HELPER: replacement for the config getter accepting the same keyword
        # arguments as the real one (most are ignored). Returns a one-element list holding a
        # FlexFlashConfig with the captured score_mod_vec_size if a score_mod and aux tensors are
        # present, else a default one.
        if has_score_mod and has_aux_tensors:
            return [
                flex_flash_attention_module.FlexFlashConfig(score_mod_vec_size=vec_size)
            ]
        return [flex_flash_attention_module.FlexFlashConfig()]

    with mock.patch.object(
        flex_flash_attention_module, "get_flex_flash_fwd_configs", configs
    ):
        yield


@contextmanager
def use_explicit_flex_flash_mask_mod_vec_size(vec_size: int | None):
    # [FUNCTION DOC] HELPER: context manager that patches get_flex_flash_fwd_configs so kernels
    # having a mask_mod get a FlexFlashConfig with the explicit mask_mod_vec_size (int or None for
    # unset) and all others get the default config. Patch is removed on exit.
    def configs(
        has_score_mod,
        has_aux_tensors,
        device=None,
        score_mod_graph_module=None,
        score_mod_other_buffers=(),
        has_mask_mod=False,
        has_mask_aux_tensors=False,
        mask_mod_graph_module=None,
        mask_mod_other_buffers=(),
        aux_scalar_symbols=(),
    ):
        # [FUNCTION DOC] HELPER: replacement config getter ignoring most arguments; returns a
        # single FlexFlashConfig with the captured mask_mod_vec_size when has_mask_mod is true,
        # else a default FlexFlashConfig.
        if has_mask_mod:
            return [
                flex_flash_attention_module.FlexFlashConfig(mask_mod_vec_size=vec_size)
            ]
        return [flex_flash_attention_module.FlexFlashConfig()]

    with mock.patch.object(
        flex_flash_attention_module, "get_flex_flash_fwd_configs", configs
    ):
        yield


def _sm100_bias(*shape):
    # [FUNCTION DOC] HELPER: creates a random float16 CUDA tensor of the given shape for use as an
    # aux bias on SM100 tests.
    return torch.randn(*shape, device="cuda", dtype=torch.float16)


SM100_AUX_SCORE_MOD_CASES = (
    (128, "kv", lambda seq_len: (_sm100_bias(seq_len),), 8, True),
    (128, "kv_mod_4", lambda seq_len: (_sm100_bias(4),), 4, True),
    # Inductor copies/normalizes sliced inputs, so this case verifies that a
    # normalized zero-offset aux layout can still autovec. A surviving nonzero
    # offset is covered by
    # test_direct_aux_load_vec_size_requires_contiguous_aligned_vector_dim.
    (
        128,
        "kv_sliced_storage_offset",
        lambda seq_len: (_sm100_bias(seq_len + 1)[1:],),
        8,
        True,
    ),
    (128, "kv_stride_mix", lambda seq_len: (_sm100_bias(2 * seq_len),), 1, False),
    (128, "q", lambda seq_len: (_sm100_bias(seq_len),), 8, False),
    (128, "q_kv", lambda seq_len: (_sm100_bias(seq_len, seq_len),), 8, True),
    (
        128,
        "multi",
        lambda seq_len: (
            _sm100_bias(seq_len),
            _sm100_bias(seq_len),
            _sm100_bias(seq_len, seq_len),
        ),
        8,
        True,
    ),
)


def sm100_aux_score_mod_case_name(case):
    # [FUNCTION DOC] HELPER: builds a test id string for an SM100 aux score_mod case tuple, formed
    # from the index-dimension label and the sequence length (label_seqN). Other tuple items
    # unused.
    seq_len, index_dim, _bias_factory, _expected_vec_size, _expect_autovec = case
    return f"{index_dim}_seq{seq_len}"


def create_alibi_learned(num_heads=4, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning an ALiBi-like score_mod with learned per-head slopes
    # (base-2 exponentials of negative values, created on CUDA in dtype). The returned function
    # adds (kv_idx - q_idx) times the slope of the head to the score, loading a captured tensor by
    # head.
    """ALiBi with learned per-head slopes (tests tensor loading)."""
    slopes = torch.exp2(-torch.linspace(1, 8, num_heads, device="cuda", dtype=dtype))

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the ALiBi bias using the captured
        # slopes.
        bias = (kv_idx - q_idx) * slopes[h]
        return score + bias

    return alibi_score_mod


def create_pos_bias_table(seq_len=512, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning a score_mod that adds a relative-position bias looked up
    # from a captured random table of size 2*seq_len-1 (scaled by 0.1) at the shifted relative
    # position (computed indexing into a tensor).
    """Relative position bias table (tests computed indexing)."""
    max_len = seq_len
    table = torch.randn(2 * max_len - 1, device="cuda", dtype=dtype) * 0.1

    def pos_bias_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure; computes the shifted relative position and
        # adds the captured table entry to the score.
        rel_pos = kv_idx - q_idx + max_len - 1
        bias = table[rel_pos]
        return score + bias

    return pos_bias_mod


def create_head_scale(num_heads=4, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning a score_mod that multiplies the score by a captured per-
    # head scale (random values in 0.5 to 1.5).
    """Per-head scaling factors (tests multiplication with tensor loading)."""
    scales = torch.rand(num_heads, device="cuda", dtype=dtype) + 0.5

    def head_scale_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure multiplying the score by the captured scale of
        # head h.
        return score * scales[h]

    return head_scale_mod


def create_batch_bias(batch_size=2, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning a score_mod that adds a captured per-batch bias (random,
    # scaled by 0.1) indexed by the batch index.
    """Per-batch bias (tests batch indexing)."""
    bias = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1

    def batch_bias_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the captured bias of batch b to the
        # score.
        return score + bias[b]

    return batch_bias_mod


def create_batch_head_bias(batch_size=2, num_heads=4, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning a score_mod that adds a captured 2D bias matrix (batch x
    # heads, random scaled by 0.5) indexed by both batch and head.
    """Per-batch-head bias matrix (tests 2D indexing with batch + head)."""
    bias_matrix = torch.randn(batch_size, num_heads, device="cuda", dtype=dtype) * 0.5

    def batch_head_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the captured bias at batch b and head h.
        bias = bias_matrix[b, h]
        return score + bias

    return batch_head_mod


def create_dual_buffer_bias(num_heads=4, seq_len=512, dtype=torch.float16):
    # [FUNCTION DOC] Factory returning a score_mod that loads from two captured tensors: a per-
    # head bias (scaled by 0.2) and a position scale (arange over seq_len, multiplied by 0.01 and
    # indexed by the query index); both are added to the score.
    """Dual buffer loading (tests loading from 2 separate tensors)."""
    head_bias = torch.randn(num_heads, device="cuda", dtype=dtype) * 0.2
    pos_scale = torch.arange(seq_len, device="cuda", dtype=dtype)

    def dual_buffer_mod(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the head bias and scaled position term.
        head_component = head_bias[h]
        pos_component = pos_scale[q_idx] * 0.01
        return score + head_component + pos_component

    return dual_buffer_mod


def create_score_view_mod(num_heads=4, dtype=torch.float16, device="cuda"):
    # [FUNCTION DOC] Factory returning a score_mod that adds a per-head scale taken from a non-
    # contiguous view (column 0 of a heads x 2 tensor). Raises AssertionError if the view happens
    # to be contiguous.
    base_scales = torch.rand(num_heads, 2, device=device, dtype=dtype) + 0.5
    scales_view = base_scales[:, 0]
    if scales_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def score_view_mod(score, _b, h, _q_idx, _kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the viewed per-head scale to the score.
        return score + scales_view[h]

    return score_view_mod


def create_complex_score_mod(
    batch_size=2, num_heads=4, seq_len=512, dtype=torch.float16, device="cuda"
):
    # [FUNCTION DOC] Factory returning a score_mod using several captured buffers: per-head bias,
    # per-query scale, per-kv scale and per-batch bias. The score gets the head term, the
    # difference of query and kv terms, and the batch term added.
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.15
    query_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    kv_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.1

    def complex_score(score, b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure combining the four captured buffers described
        # above.
        head_term = head_bias[h]
        query_term = query_scale[q_idx] - kv_scale[kv_idx]
        batch_term = batch_bias[b]
        return score + head_term + query_term + batch_term

    return complex_score


def create_score_mod_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    # [FUNCTION DOC] Factory returning a score_mod that adds a captured per-head bias (random
    # scaled by 0.2) to the score; used together with mask buffers.
    score_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2

    def score_with_buffer(score, _b, h, _q_idx, _kv_idx):
        # [FUNCTION DOC] HELPER: score_mod closure adding the captured per-head bias.
        return score + score_bias[h]

    return score_with_buffer


def create_mask_mod_scalar_tensor(device="cuda"):
    # [FUNCTION DOC] Factory returning a mask_mod that captures a 0-dim int32 tensor (value 5) on
    # the device; the mask keeps positions where q_idx plus the offset is >= kv_idx. Regression
    # helper for scalar-tensor loads in CuTeDSL codegen (see the docstring issue reference).
    """mask_mod that captures a 0-dim (scalar) tensor.

    Regression test: loading a 0-dim tensor in CuTeDSL codegen produces a
    constant index 0, which was incorrectly passed as a bare Python int to
    ssa_to_indexable (expects TensorSSA).  See #177813.
    """
    offset = torch.tensor(5, dtype=torch.int32, device=device)

    def mask_with_scalar_tensor(_b, _h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: mask_mod closure comparing the query index shifted by the
        # captured scalar tensor against the kv index.
        return (q_idx + offset) >= kv_idx

    return mask_with_scalar_tensor


def create_mask_mod_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    # [FUNCTION DOC] Factory returning a mask_mod that captures a per-head random bias; a position
    # is kept if it is causal (q_idx >= kv_idx) or the head's bias is positive.
    mask_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.1

    def custom_mask(_b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: mask_mod closure; returns causal-or-positive-head-bias boolean
        # tensor.
        bias_value = mask_bias[h]
        return (q_idx >= kv_idx) | (bias_value > 0)

    return custom_mask


def create_document_mask(lengths_per_batch, device):
    # [FUNCTION DOC] Builds a document mask. lengths_per_batch is a list (per batch) of document
    # lengths; they are expanded into per-token document ids and stored in a long tensor on the
    # device. Returns a mask_mod that is true when the document of the query token (index halved
    # by integer division) equals the document of the kv token for that batch.
    document_ids = []
    for lengths in lengths_per_batch:
        doc_tokens = []
        for doc_id, length in enumerate(lengths):
            doc_tokens.extend([doc_id] * length)
        document_ids.append(doc_tokens)
    document_ids = torch.tensor(document_ids, device=device, dtype=torch.long)

    def document_mask(b, _h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: mask_mod closure comparing document ids of query (q_idx // 2) and
        # kv.
        doc_id_q = document_ids[b, q_idx // 2]
        doc_id_kv = document_ids[b, kv_idx]
        return doc_id_q == doc_id_kv

    return document_mask


def create_mask_mod_view_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    # [FUNCTION DOC] Factory returning a mask_mod that captures a non-contiguous per-head view
    # (column 1 of a heads x 3 tensor). Raises AssertionError if the view is contiguous. The mask
    # is causal-or-(twice the head bias positive).
    base_bias = torch.randn(num_heads, 3, device=device, dtype=dtype)
    mask_bias_view = base_bias[:, 1]
    if mask_bias_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def mask_with_view_buffer(_b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: mask_mod closure using the viewed bias, doubled, for the
        # positivity test.
        bias_value = mask_bias_view[h]
        double_bias = bias_value * 2
        return (q_idx >= kv_idx) | (double_bias > 0)

    return mask_with_view_buffer


def create_mask_mod_dual_buffers(
    batch_size=2, num_heads=4, dtype=torch.float16, device="cuda"
):
    # [FUNCTION DOC] Factory returning a mask_mod that captures a per-head and a per-batch bias
    # (each scaled by 0.2); keeps positions that are causal or where the summed bias (cast to
    # float32) is positive.
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.2

    def dual_buffer_mask(b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: mask_mod closure OR-ing the causal condition with the bias
        # condition.
        head_term = head_bias[h]
        batch_term = batch_bias[b]
        causal = q_idx >= kv_idx
        bias_cond = (head_term + batch_term).to(torch.float32) > 0
        return causal | bias_cond

    return dual_buffer_mask


def create_test_tensors(
    batch_size=2,
    num_heads=4,
    seq_len=512,
    dim=64,
    dtype=torch.float16,
    device="cuda",
    requires_grad=False,
    num_heads_kv=None,
):
    # [FUNCTION DOC] Creates random q, k, v tensors for attention tests. q has shape (batch,
    # heads, seq, dim); k and v use num_heads_kv heads (defaults to num_heads). Returns the tuple
    # (q, k, v) with the given dtype, device and requires_grad flag.
    num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
    q_shape = (batch_size, num_heads, seq_len, dim)
    kv_shape = (batch_size, num_heads_kv, seq_len, dim)
    q = torch.randn(q_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def _create_block_mask_for_device(
    mask_mod, batch_size, num_heads, q_len, kv_len, *, device
):
    # [FUNCTION DOC] HELPER: builds a block mask for a mask_mod using sparse block sizes adapted
    # to the device compute capability: on CUDA, major 8 uses a kv block of 64 and major 10
    # doubles the query block; otherwise the default sparse block size. Returns the BlockMask.
    """Match FlexAttention's block-height expectations per compute capability."""
    q_block = _DEFAULT_SPARSE_BLOCK_SIZE
    kv_block = _DEFAULT_SPARSE_BLOCK_SIZE
    dev = torch.device(device)
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        if major == 8:
            kv_block = 64
        elif major == 10:
            q_block *= 2
    return create_block_mask(
        mask_mod,
        batch_size,
        num_heads,
        q_len,
        kv_len,
        device=device,
        BLOCK_SIZE=(q_block, kv_block),
    )


@contextmanager
def cuda_kernel_profiler(kernel_pattern="flash_attncute"):
    # [FUNCTION DOC] HELPER: context manager profiling CUDA activity. Yields a result dict with
    # found=False and an empty kernel_names list; after the with-body exits it fills kernel_names
    # with the names of CUDA events and sets found to whether any name contains kernel_pattern.
    """Context manager for profiling CUDA kernels."""
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


def flash_vs_triton(q, k, v, score_mod=None, block_mask=None, rtol=2, *, dynamic=False):
    # [FUNCTION DOC] Core comparison helper. Computes an fp32 eager reference, then runs compiled
    # flex_attention with the FLASH backend and with the TRITON backend (GQA enabled automatically
    # if q and k head counts differ). Raises AssertionError on shape mismatch, NaN or non-finite
    # outputs, or if the flash error versus reference exceeds rtol times the triton error plus a
    # noise-floor tolerance. If any input requires grad, also compares dq/dk/dv the same way using
    # a random upstream gradient. Returns (out_flash, out_triton, out_ref_fp32).
    compiled_fn = torch.compile(flex_attention, dynamic=dynamic)
    enable_gqa = q.shape[1] != k.shape[1]

    out_ref_fp32 = flex_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    ).to(q.dtype)

    out_flash = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options={"BACKEND": "FLASH"},
    )
    out_triton = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options={"BACKEND": "TRITON"},
    )

    if not (out_flash.shape == out_ref_fp32.shape == out_triton.shape):
        raise AssertionError(
            f"Shape mismatch: flash={out_flash.shape}, ref={out_ref_fp32.shape}, triton={out_triton.shape}"
        )
    if torch.isnan(out_flash).any():
        raise AssertionError("out_flash contains NaN")
    if torch.isnan(out_triton).any():
        raise AssertionError("out_triton contains NaN")
    if torch.isnan(out_ref_fp32).any():
        raise AssertionError("out_ref_fp32 contains NaN")
    if not torch.isfinite(out_flash).all():
        raise AssertionError("out_flash contains non-finite values")
    if not torch.isfinite(out_triton).all():
        raise AssertionError("out_triton contains non-finite values")
    if not torch.isfinite(out_ref_fp32).all():
        raise AssertionError("out_ref_fp32 contains non-finite values")

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()

    triton_error = (out_triton - out_ref_fp32).abs().max().item()
    flash_error = (out_flash - out_ref_fp32).abs().max().item()

    if flash_error > rtol * triton_error + fwd_atol:
        raise AssertionError(
            f"Flash error {flash_error:.2e} exceeds {rtol}x Triton error {triton_error:.2e} + {fwd_atol:.2e}"
        )

    needs_backward = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v)
    )
    if needs_backward:
        grad = torch.randn_like(out_flash)
        inputs = (q, k, v)
        grads_ref = torch.autograd.grad(out_ref_fp32, inputs, grad)
        grads_triton = torch.autograd.grad(out_triton, inputs, grad)
        grads_flash = torch.autograd.grad(out_flash, inputs, grad)

        dq_atol = 2 * (grads_ref[0] + 0.3 - 0.3 - grads_ref[0]).abs().max().item()
        dk_atol = 2 * (grads_ref[1] + 0.3 - 0.3 - grads_ref[1]).abs().max().item()
        dv_atol = 2 * (grads_ref[2] + 0.3 - 0.3 - grads_ref[2]).abs().max().item()

        atol_pack = (dq_atol, dk_atol, dv_atol)
        for grad_flash, grad_triton, grad_ref, atol in zip(
            grads_flash, grads_triton, grads_ref, atol_pack
        ):
            if not torch.isfinite(grad_flash).all():
                raise AssertionError("grad_flash contains non-finite values")
            if not torch.isfinite(grad_triton).all():
                raise AssertionError("grad_triton contains non-finite values")
            if not torch.isfinite(grad_ref).all():
                raise AssertionError("grad_ref contains non-finite values")

            triton_error = (grad_triton - grad_ref).abs().max().item()
            flash_error = (
                (grad_flash - grad_ref.to(grad_flash.dtype)).abs().max().item()
            )
            if flash_error > rtol * triton_error + atol:
                raise AssertionError(
                    f"Flash error {flash_error:.2e} exceeds {rtol}x Triton error {triton_error:.2e} + {atol:.2e}"
                )

    return out_flash, out_triton, out_ref_fp32


@dataclass
class ScoreModCase:
    # [CLASS DOC] Dataclass describing one score_mod test case: name, a factory taking (dtype,
    # device) and returning a score_mod or None, and shape parameters (batch, heads, kv heads,
    # seq_len, dim) plus requires_grad for backward testing.
    name: str
    score_mod_factory: Callable[[torch.dtype, str], Callable | None]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    seq_len: int = 512
    dim: int = 64
    requires_grad: bool = False


@dataclass
class MaskModCase:
    # [CLASS DOC] Dataclass describing one mask_mod test case: name, a mask_mod factory taking
    # (dtype, device), shape parameters, optional block_mask_num_heads (heads dimension of the
    # block mask), an optional score_mod factory, and requires_grad for backward testing.
    name: str
    mask_mod_factory: Callable[[torch.dtype, str], Callable]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    block_mask_num_heads: int | None = None
    seq_len: int = 512
    dim: int = 64
    score_mod_factory: Callable[[torch.dtype, str], Callable | None] | None = None
    requires_grad: bool = False


def score_case_name(case: ScoreModCase):
    # [FUNCTION DOC] HELPER: returns the name of a ScoreModCase, used as a parametrize id.
    return case.name


def mask_case_name(case: MaskModCase):
    # [FUNCTION DOC] HELPER: returns the name of a MaskModCase, used as a parametrize id.
    return case.name


SCORE_MOD_CASES = [
    ScoreModCase("basic", lambda _dtype, _device: None),
    ScoreModCase("times_two", lambda _dtype, _device: _times_two),
    ScoreModCase("causal", lambda _dtype, _device: _causal),
    ScoreModCase("rel_bias", lambda _dtype, _device: _rel_bias),
    ScoreModCase("causal_unfriendly_127", lambda _dtype, _device: _causal, seq_len=127),
    ScoreModCase("causal_unfriendly_255", lambda _dtype, _device: _causal, seq_len=255),
    ScoreModCase("causal_unfriendly_383", lambda _dtype, _device: _causal, seq_len=383),
    ScoreModCase("causal_unfriendly_511", lambda _dtype, _device: _causal, seq_len=511),
    ScoreModCase(
        "alibi_learned",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "pos_bias_table",
        lambda dtype, device: create_pos_bias_table(seq_len=512, dtype=dtype),
    ),
    ScoreModCase(
        "head_scale",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "batch_bias",
        lambda dtype, device: create_batch_bias(batch_size=2, dtype=dtype),
    ),
    ScoreModCase(
        "batch_head_bias",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
    ),
    ScoreModCase(
        "dual_buffer_bias",
        lambda dtype, device: create_dual_buffer_bias(
            num_heads=4, seq_len=512, dtype=dtype
        ),
    ),
    ScoreModCase(
        "score_view_buffer",
        lambda dtype, device: create_score_view_mod(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "score_many_buffers",
        lambda dtype, device: create_complex_score_mod(
            batch_size=2, num_heads=4, seq_len=512, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "backward_times_two",
        lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_score_squared",
        lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_rel_bias",
        lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_softcap",
        lambda _dtype, _device: _softcap,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_offset",
        lambda _dtype, _device: _head_offset,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_distance_decay",
        lambda _dtype, _device: _distance_decay,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_alibi_captured",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_scale_captured",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_batch_head_bias_captured",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_pos_bias_table_captured",
        lambda dtype, device: create_pos_bias_table(seq_len=257, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "gqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "mqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "backward_gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_gqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
]

DETERMINISTIC_SCORE_MOD_CASES = [case for case in SCORE_MOD_CASES if case.requires_grad]


MASK_MOD_CASES = [
    MaskModCase(
        "mask_mod_scalar_tensor",
        lambda _dtype, device: create_mask_mod_scalar_tensor(device=device),
    ),
    MaskModCase("block_mask_causal", lambda _dtype, _device: _causal_mask),
    MaskModCase(
        "block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
    ),
    MaskModCase(
        "mask_mod_buffer",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "doc_mask",
        lambda _dtype, device: create_document_mask(
            ((16, 31, 25, 56), (40, 9, 23, 56)), device=device
        ),
        block_mask_num_heads=1,
        seq_len=128,
    ),
    MaskModCase(
        "mask_mod_view_buffer",
        lambda dtype, device: create_mask_mod_view_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        requires_grad=True,
    ),
    MaskModCase(
        "mask_mod_dual_buffers",
        lambda dtype, device: create_mask_mod_dual_buffers(
            batch_size=2, num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "score_and_mask_buffers",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        score_mod_factory=lambda dtype, device: create_score_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "backward_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_squared",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_rel_bias",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
]

DETERMINISTIC_MASK_MOD_CASES = [case for case in MASK_MOD_CASES if case.requires_grad]

GQA_MQA_BLOCK_MASK_CASES = [
    MaskModCase(
        "gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
]


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
@xfailIfSM12X
class TestFlexFlash(InductorTestCase):
    # [CLASS DOC] Main test class for the FlexAttention FLASH backend, skipped when the CuTe flash
    # attention library is unavailable and marked expected-failure on SM12x. Tests compare the
    # FLASH backend against TRITON and an fp32 reference through flash_vs_triton, and also inspect
    # generated code for vectorization choices and check error messages. Device-generic test
    # methods take device and dtype arguments; others require CUDA directly.
    # `FlashAttentionForwardSm120` does not have `apply_score_mod`.
    @xfailIfSM120OrLater
    def test_vectorized_group_per_lane_gather(self):
        # [FUNCTION DOC] Regression test (see docstring issue): a score_mod mixing a per-lane
        # gather (relative-position table) with a lane-uniform per-head scale load must keep the
        # gather per-lane under vectorization. Compares FLASH against TRITON via flash_vs_triton.
        # Expected to fail on SM120+.
        """Per-lane gather + lane-uniform load in one vec group (#188871).

        The uniform scale load enables score_mod vectorization; the rel-pos
        gather's wrapped index must still classify per-lane, not lane-uniform
        (which silently broadcast lane 0's bias across the group).
        """
        seq_len = 512
        rel_bias = torch.randn(2 * seq_len, device="cuda")
        head_scale = torch.randn(4, device="cuda")
        offset = seq_len - 1

        def score_mod(score, _b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding a relative-position table value (index
            # shifted by a constant offset) multiplied by the per-head scale to the score.
            return score + rel_bias[q_idx - kv_idx + offset] * head_scale[h]

        q, k, v = create_test_tensors(seq_len=seq_len, device="cuda")
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @xfailIfSM120OrLater
    def test_vectorized_group_untraceable_index_op(self):
        # [FUNCTION DOC] Regression test: nonlinear index ops (absolute value, clamp, minimum,
        # maximum) used to index a captured table must keep per-lane semantics under
        # vectorization. Loops over four index functions in subtests, builds a score_mod for each,
        # and compares FLASH versus TRITON. Expected to fail on SM120+.
        """Nonlinear index ops (abs/clamp/min/max) must keep per-lane index
        semantics under vectorization — whether traced to sympy or treated as
        opaque, never classified lane-uniform (#188878)."""
        seq_len = 512
        tbl = torch.randn(2 * seq_len, device="cuda")
        head_scale = torch.randn(4, device="cuda")

        index_fns = {
            "abs": lambda q_idx, kv_idx: (q_idx - kv_idx).abs(),
            "clamp": lambda q_idx, kv_idx: (q_idx - kv_idx).clamp(0, seq_len - 1),
            "minimum": lambda q_idx, kv_idx: torch.minimum(q_idx, kv_idx),
            "maximum": lambda q_idx, kv_idx: torch.maximum(
                kv_idx - q_idx, torch.zeros_like(kv_idx)
            ),
        }
        for name, index_fn in index_fns.items():
            with self.subTest(index_fn=name):

                def score_mod(score, _b, h, q_idx, kv_idx):
                    # [FUNCTION DOC] HELPER: score_mod adding the table value at the index
                    # produced by the current subtest's index function, multiplied by the per-head
                    # scale.
                    return score + tbl[index_fn(q_idx, kv_idx)] * head_scale[h]

                q, k, v = create_test_tensors(seq_len=seq_len, device="cuda")
                flash_vs_triton(q, k, v, score_mod=score_mod)

    @xfailIfSM120OrLater
    def test_chained_kv_gather(self):
        # [FUNCTION DOC] Regression test: a chained gather (table indexed by a document-id tensor
        # loaded at the kv index) must stay a per-lane gather even though the inner load is
        # contiguous. Compares FLASH against TRITON. Expected to fail on SM120+.
        """Document-style chained gather: the contiguous inner load promotes
        vectorization and the outer loaded-value index is opaque; it must stay
        a per-lane gather (#188878)."""
        seq_len = 512
        doc_bias = torch.randn(8, device="cuda")
        doc_id = torch.arange(seq_len, device="cuda") % 8

        def score_mod(score, _b, _h, _q, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the bias looked up through the document id
            # of kv_idx.
            return score + doc_bias[doc_id[kv_idx]]

        q, k, v = create_test_tensors(seq_len=seq_len, device="cuda")
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @xfailIfSM120OrLater
    def test_captured_table_int64_index(self):
        # [FUNCTION DOC] Regression test: indexing a captured table with a kv index widened to
        # int64 must not break index-fragment materialization. Compares FLASH against TRITON.
        # Expected to fail on SM120+.
        """Widening index casts must not break index-fragment materialization (#188871)."""
        seq_len = 512
        tbl = torch.randn(seq_len, device="cuda")

        def score_mod(score, _b, _h, _q, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured table entry at the kv index
            # cast to int64.
            return score + tbl[kv_idx.to(torch.int64)]

        q, k, v = create_test_tensors(seq_len=seq_len, device="cuda")
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", SCORE_MOD_CASES, name_fn=score_case_name)
    def test_flash_attention_score_mod_cases(self, device, dtype, case):
        # [FUNCTION DOC] Parametrized over dtype (float16, bfloat16) and SCORE_MOD_CASES. Builds
        # q/k/v from the case's shape settings, then runs flash_vs_triton with the case's
        # score_mod (forward, and backward when the case requires grad). Expected to fail on
        # SM120+.
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=case.score_mod_factory(dtype, device),
        )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_SCORE_MOD_CASES, name_fn=score_case_name)
    def test_flash_attention_backward_deterministic_score_mod_cases(
        self, device, dtype, case
    ):
        # [FUNCTION DOC] Backward score_mod cases under deterministic mode. Builds tensors per
        # case. On SM120+ it expects an error: an AttributeError about the missing score-mod
        # support in the Sm120 forward class, or ValueError "Operation creation failed" for the
        # gqa_basic/mqa_basic cases; otherwise runs flash_vs_triton inside a deterministic guard
        # and expects success.
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        if SM120OrLater:
            cls, pattern = (
                AttributeError,
                r"'FlashAttentionForwardSm120' object has no attribute 'apply_score_mod'",
            )
            if "gqa_basic" in case.name or "mqa_basic" in case.name:
                cls, pattern = ValueError, "Operation creation failed"
            with self.assertRaisesRegex(cls, pattern), DeterministicGuard(True):
                flash_vs_triton(
                    q,
                    k,
                    v,
                    score_mod=case.score_mod_factory(dtype, device)
                    if case.score_mod_factory
                    else None,
                )
        else:
            with DeterministicGuard(True):
                flash_vs_triton(
                    q,
                    k,
                    v,
                    score_mod=case.score_mod_factory(dtype, device)
                    if case.score_mod_factory
                    else None,
                )

    @xfailIfSM120OrLater
    @decorateIf(
        unittest.expectedFailure,
        lambda params: (
            IS_SM8X
            and not params["case"].requires_grad
            and params["case"].score_mod_factory is not None
        ),
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_mask_mod_cases(self, device, dtype, case):
        # [FUNCTION DOC] Parametrized over dtype and MASK_MOD_CASES. Skips backward cases on
        # devices below SM90 (block-sparse backward unsupported for FLASH). Builds a block mask
        # with the device-adapted helper (using the case's block_mask_num_heads or num_heads) and
        # runs flash_vs_triton with the optional score_mod. Expected failure on SM120+, and on
        # SM8x for forward cases that have a score_mod.
        if case.requires_grad:
            major, _ = torch.cuda.get_device_capability()
            if major < 9:
                self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=(
                case.score_mod_factory(dtype, device)
                if case.score_mod_factory
                else None
            ),
            block_mask=_create_block_mask_for_device(
                case.mask_mod_factory(dtype, device),
                case.batch_size,
                case.block_mask_num_heads or case.num_heads,
                case.seq_len,
                case.seq_len,
                device=device,
            ),
        )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_backward_deterministic_block_mask_raises(
        self, device, dtype, case
    ):
        # [FUNCTION DOC] Deterministic mode with a block mask in FLASH backward must raise on SM90
        # (the test is skipped on SM100+, which supports it). Compiles flex_attention with
        # fullgraph, runs under a deterministic guard, and expects BackendCompilerFailed whose
        # message mentions that deterministic backward with block_mask is unsupported for the
        # FLASH backend.
        major, _ = torch.cuda.get_device_capability()
        if major >= 10:
            self.skipTest(
                "SM100+ supports deterministic block-sparse FLASH backward; "
                "this test validates the error on older architectures"
            )
        from torch._dynamo.exc import BackendCompilerFailed

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        block_mask = _create_block_mask_for_device(
            case.mask_mod_factory(dtype, device),
            case.batch_size,
            case.block_mask_num_heads or case.num_heads,
            case.seq_len,
            case.seq_len,
            device=device,
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with DeterministicGuard(True):
            with self.assertRaisesRegex(
                BackendCompilerFailed,
                "Deterministic backward for flex_attention with block_mask and BACKEND='FLASH'",
            ):
                out = compiled_fn(
                    q,
                    k,
                    v,
                    score_mod=(
                        case.score_mod_factory(dtype, device)
                        if case.score_mod_factory
                        else None
                    ),
                    block_mask=block_mask,
                    kernel_options={"BACKEND": "FLASH"},
                )
                out.sum().backward()

    @dtypes(torch.bfloat16)
    def test_flash_deterministic_block_mask_without_write_order_raises(
        self, device, dtype
    ):
        # [FUNCTION DOC] SM100+ only. A block mask created without the dq write order, run in
        # deterministic mode, must fail in backward: expects an InductorError naming the option
        # for computing the dq write order when creating the block mask. Forward runs first, error
        # is raised by the backward.
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("deterministic block-sparse FLASH backward requires SM100+")

        torch._dynamo.reset()
        q, k, v = create_test_tensors(
            seq_len=512,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        block_mask = _create_block_mask_for_device(
            _causal_mask, 2, 4, 512, 512, device=device
        )
        compiled_fn = torch.compile(flex_attention)

        from torch._inductor.exc import InductorError

        with DeterministicGuard(True):
            out = compiled_fn(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )
            with self.assertRaisesRegex(
                InductorError,
                "create_block_mask\\(\\.\\.\\., compute_dq_write_order=True\\)",
            ):
                out.sum().backward()

    @dtypes(torch.bfloat16)
    def test_flash_deterministic_block_mask_without_scheduler_order_raises(
        self, device, dtype
    ):
        # [FUNCTION DOC] SM100+ only. Creates a block mask with dq write order but clears the kv
        # scheduler order metadata; deterministic backward must raise InductorError saying
        # scheduler-order metadata is required.
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("deterministic block-sparse FLASH backward requires SM100+")

        torch._dynamo.reset()
        q, k, v = create_test_tensors(
            seq_len=512,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        block_mask = create_block_mask(
            _causal_mask,
            2,
            4,
            512,
            512,
            device=device,
            BLOCK_SIZE=(
                _DEFAULT_SPARSE_BLOCK_SIZE * 2,
                _DEFAULT_SPARSE_BLOCK_SIZE,
            ),
            compute_dq_write_order=True,
        )
        block_mask.dq_kv_order_spt = None
        compiled_fn = torch.compile(flex_attention)

        from torch._inductor.exc import InductorError

        with DeterministicGuard(True):
            out = compiled_fn(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )
            with self.assertRaisesRegex(
                InductorError,
                "requires dQ KV scheduler-order metadata",
            ):
                out.sum().backward()

    @dtypes(torch.bfloat16)
    def test_flash_deterministic_block_mask_warn_only_runs_nondeterministic(
        self, device, dtype
    ):
        # [FUNCTION DOC] SM90+ only. With deterministic mode in warn-only mode, a FLASH block-
        # sparse backward without write order must still run and emit a UserWarning that there is
        # no deterministic implementation for this configuration.
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        torch._dynamo.reset()
        q, k, v = create_test_tensors(
            seq_len=512,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        block_mask = _create_block_mask_for_device(
            _causal_mask, 2, 4, 512, 512, device=device
        )
        compiled_fn = torch.compile(flex_attention)

        with DeterministicGuard(True, warn_only=True):
            with self.assertWarnsRegex(
                UserWarning,
                "does not have a deterministic implementation for this configuration",
            ):
                out = compiled_fn(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    kernel_options={"BACKEND": "FLASH"},
                )
                out.sum().backward()

    @dtypes(torch.bfloat16)
    @parametrize("dq_kv_order", [False, True])
    def test_flash_attention_backward_deterministic_block_mask_with_write_order(
        self, device, dtype, dq_kv_order
    ):
        # [FUNCTION DOC] SM100+ only, parametrized over the dq_kv_order flag. Creates a block mask
        # with compute_dq_write_order enabled, asserts the write order exists and that the mask
        # reports the requested kv-order flag, then runs flash_vs_triton under deterministic mode
        # (forward and backward must match the references).
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("deterministic block-sparse FLASH backward requires SM100+")

        torch._dynamo.reset()
        q, k, v = create_test_tensors(
            seq_len=512,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        block_mask = create_block_mask(
            _causal_mask,
            2,
            4,
            512,
            512,
            device=device,
            BLOCK_SIZE=(
                _DEFAULT_SPARSE_BLOCK_SIZE * 2,
                _DEFAULT_SPARSE_BLOCK_SIZE,
            ),
            compute_dq_write_order=True,
            dq_kv_order=dq_kv_order,
        )
        self.assertIsNotNone(block_mask.dq_write_order)
        self.assertEqual(block_mask._dq_kv_order(), dq_kv_order)

        with DeterministicGuard(True):
            flash_vs_triton(q, k, v, block_mask=block_mask)

    @decorateIf(
        unittest.expectedFailure,
        lambda params: (
            SM120OrLater
            and params["case"].name
            in {
                "backward_block_mask_causal",
                "backward_block_mask_causal_rel_bias",
                "backward_block_mask_causal_score_squared",
                "backward_block_mask_causal_score_times_two",
                "mask_mod_view_buffer",
            }
        ),
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_backward_deterministic_warn_only_block_mask(
        self, device, dtype, case
    ):
        # [FUNCTION DOC] Deterministic warn-only mode for each deterministic mask case (SM90+).
        # Compiles flex_attention with fullgraph on FLASH, runs forward, and expects the backward
        # to emit a UserWarning about the lack of a deterministic implementation. Expected failure
        # on SM120+ for several named cases.
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        block_mask = _create_block_mask_for_device(
            case.mask_mod_factory(dtype, device),
            case.batch_size,
            case.block_mask_num_heads or case.num_heads,
            case.seq_len,
            case.seq_len,
            device=device,
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with DeterministicGuard(True, warn_only=True):
            out = compiled_fn(
                q,
                k,
                v,
                score_mod=(
                    case.score_mod_factory(dtype, device)
                    if case.score_mod_factory
                    else None
                ),
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )
            with self.assertWarnsRegex(
                UserWarning,
                "does not have a deterministic implementation for this configuration",
            ):
                out.sum().backward()

    @decorateIf(
        unittest.expectedFailure,
        lambda params: (
            SM120OrLater
            and params["case"].name
            in {
                "gqa_block_mask_causal",
                "gqa_block_mask_causal_per_head",
                "backward_gqa_block_mask_causal",
                "backward_gqa_block_mask_causal_per_head",
                "mqa_block_mask_causal",
                "mqa_block_mask_causal_per_head",
                "backward_mqa_block_mask_causal",
                "backward_mqa_block_mask_causal_per_head",
            }
        ),
    )
    @decorateIf(
        unittest.expectedFailure,
        lambda params: (
            IS_SM8X
            and not params["case"].requires_grad
            and params["case"].block_mask_num_heads == 1
        ),
    )
    @decorateIf(
        unittest.expectedFailure,
        lambda params: SM120OrLater and params["case"].name.endswith("_dim128"),
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", GQA_MQA_BLOCK_MASK_CASES, name_fn=mask_case_name)
    def test_flash_attention_gqa_mqa_block_mask_cases(self, device, dtype, case):
        # [FUNCTION DOC] Parametrized over dtype and GQA_MQA_BLOCK_MASK_CASES. Skips backward
        # cases below SM90. Runs flash_vs_triton with the device-adapted block mask built from the
        # case's mask_mod and block mask head count, plus an optional score_mod. Several expected-
        # failure decorators cover SM120+, SM8x with single-head block masks forward, and SM120+
        # with head dim 128.
        if case.requires_grad:
            major, _ = torch.cuda.get_device_capability()
            if major < 9:
                self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=(
                case.score_mod_factory(dtype, device)
                if case.score_mod_factory
                else None
            ),
            block_mask=_create_block_mask_for_device(
                case.mask_mod_factory(dtype, device),
                case.batch_size,
                case.block_mask_num_heads or case.num_heads,
                case.seq_len,
                case.seq_len,
                device=device,
            ),
        )

    @xfailIfSM120OrLater
    @parametrize("case", ["offset", "stride"])
    def test_cutedsl_captured_alias_views_keep_distinct_layouts(self, device, case):
        # [FUNCTION DOC] Parametrized with offset and stride. Two captured aliases (left and
        # right) of one base tensor are used in a score_mod; the generated code must contain
        # distinct reinterpret layouts with the expected sizes/strides/offsets for each view.
        # Checks numerical closeness to eager (atol/rtol 3e-2) and the substrings in the generated
        # source. Expected to fail on SM120+.
        make_tensor = functools.partial(
            torch.randn,
            (1, 2, 128, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        match case:
            case "offset":
                base = torch.randn((1, 2, 128, 256), device=device, dtype=torch.float32)
                expected_reinterpret_tensors = [
                    "(65536, 32768, 256, 1), 0)",
                    "(65536, 32768, 256, 1), 128)",
                ]
            case "stride":
                base = torch.randn((1, 2, 256, 128), device=device, dtype=torch.float32)
                expected_reinterpret_tensors = [
                    "(65536, 32768, 128, 1), 0)",
                    "(65536, 32768, 256, 1), 0)",
                ]
            case _:
                raise AssertionError(case)

        def fn(q, k, v, base):
            # [FUNCTION DOC] HELPER: function under test; slices base into two views according to
            # the case (offset: left and right halves of the last dim; stride: first 128 rows and
            # every other row), and runs flex_attention on FLASH with a score_mod adding both
            # views.
            match case:
                case "offset":
                    left = base[:, :, :, :128]
                    right = base[:, :, :, 128:]
                case "stride":
                    left = base[:, :, :128, :]
                    right = base[:, :, ::2, :]
                case _:
                    raise AssertionError(case)

            def score_mod(score, b, h, m, n):
                # [FUNCTION DOC] HELPER: score_mod adding the elements of the left and right
                # captured views.
                return score + left[b, h, m, n] + right[b, h, m, n]

            return flex_attention(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
            )

        expected = fn(q, k, v, base)
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), q, k, v, base
        )
        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)

        src = "\n".join(code)
        for expected_reinterpret_tensor in expected_reinterpret_tensors:
            self.assertIn(expected_reinterpret_tensor, src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10,
        "SM100+ only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    @parametrize(
        "case",
        SM100_AUX_SCORE_MOD_CASES,
        name_fn=sm100_aux_score_mod_case_name,
    )
    def test_flash_attention_sm100_aux_score_mod_vec_selection(self, case):
        # [FUNCTION DOC] SM100+ only, caches disabled. For each aux-bias case compiles a function
        # with a score_mod indexing biases in a pattern given by the case. First forces vector
        # size 1 and checks results against eager; then compiles with automatic selection and
        # requires bit-identical results to the scalar version, the expected vec size marker in
        # the generated code, and presence or absence of the autovec copy call depending on the
        # expected-autovec flag.
        seq_len, index_dim, bias_factory, expected_vec_size, expect_autovec = case
        torch.manual_seed(0)
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        biases = bias_factory(seq_len)

        def fn(q, k, v, *biases):
            # [FUNCTION DOC] HELPER: function under test taking q, k, v and bias tensors; defines
            # the score_mod and returns FLASH flex_attention output.
            def score_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod that adds bias values indexed according to the
                # case's index dimension (kv, kv mod 4, mixed stride, q, q and kv, or multiple
                # biases); returns None for unknown labels.
                match index_dim:
                    case "kv" | "kv_sliced_storage_offset":
                        return score + biases[0][kv_idx]
                    case "kv_mod_4":
                        return score + biases[0][kv_idx % 4]
                    case "kv_stride_mix":
                        return score + biases[0][2 * kv_idx - kv_idx % 4]
                    case "q":
                        return score + biases[0][q_idx]
                    case "q_kv":
                        return score + biases[0][q_idx, kv_idx]
                    case "multi":
                        return (
                            score
                            + biases[0][kv_idx]
                            + biases[1][q_idx]
                            + biases[2][q_idx, kv_idx]
                        )

            return flex_attention(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
            )

        fn_args = (q, k, v, *biases)
        expected = fn(*fn_args)
        with force_flex_flash_score_mod_vec_size(1):
            torch._dynamo.reset()
            scalar, scalar_code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), *fn_args
            )
        torch._dynamo.reset()
        auto, auto_code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False), *fn_args
        )

        self.assertEqual(scalar, expected, atol=3e-2, rtol=3e-2, msg=index_dim)
        self.assertEqual(auto, scalar, atol=0, rtol=0, msg=index_dim)
        self.assertIn("score_mod.__vec_size__ = 1", "\n".join(scalar_code))
        auto_src = "\n".join(auto_code)
        self.assertIn(f"score_mod.__vec_size__ = {expected_vec_size}", auto_src)
        if expect_autovec:
            self.assertIn("cute.autovec_copy", auto_src)
        else:
            self.assertNotIn("cute.autovec_copy", auto_src)

    def _assert_sm100_mask_vec_matches_scalar(
        self,
        fn,
        *args,
        expect_autovec: bool = True,
    ):
        # [FUNCTION DOC] HELPER: runs fn eagerly, then compiled with explicit mask_mod vec size
        # unset (scalar baseline) and compiled with automatic selection. Asserts scalar close to
        # eager, automatic bit-identical to scalar, no mask vec marker in scalar code, the marker
        # for vec size 32 in the automatic code and, if expect_autovec, the autovec copy call.
        # Returns the automatic code.
        expected = fn(*args)
        with use_explicit_flex_flash_mask_mod_vec_size(None):
            torch._dynamo.reset()
            scalar, scalar_code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), *args
            )
        torch._dynamo.reset()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False), *args
        )

        self.assertEqual(scalar, expected, atol=3e-2, rtol=3e-2)
        self.assertEqual(actual, scalar, atol=0, rtol=0)
        self.assertNotIn("mask_mod.__vec_size__", "\n".join(scalar_code))
        src = "\n".join(code)
        self.assertIn("mask_mod.__vec_size__ = 32", src)
        if expect_autovec:
            self.assertIn("cute.autovec_copy", src)
        return src

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_score_and_mask_mod_vec_selection(self):
        # [FUNCTION DOC] SM100/SM110 only. Builds a function with a score_mod (1-D kv bias) and a
        # mask_mod (causal or 2-D mask bias positive), compares eager, scalar-mask and automatic
        # compiled outputs, and asserts the generated code has score vec size 8, mask vec size 32
        # and autovec copies.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        score_bias = torch.randn(seq_len, device="cuda", dtype=torch.float16)
        mask_bias = torch.randn(seq_len, seq_len, device="cuda", dtype=torch.float16)

        def fn(q, k, v, score_bias, mask_bias):
            # [FUNCTION DOC] HELPER: function under test; builds a block mask from the nested
            # mask_mod and calls FLASH flex_attention with both mods.
            def score_mod(score, _b, _h, _q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the score bias at the kv index.
                return score + score_bias[kv_idx]

            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where causal or where the 2-D mask bias is
                # positive.
                return (q_idx >= kv_idx) | (mask_bias[q_idx, kv_idx] > 0)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )

        args = (q, k, v, score_bias, mask_bias)
        expected = fn(*args)
        with use_explicit_flex_flash_mask_mod_vec_size(None):
            torch._dynamo.reset()
            scalar_mask, scalar_code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), *args
            )
        torch._dynamo.reset()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False), *args
        )

        self.assertEqual(scalar_mask, expected, atol=3e-2, rtol=3e-2)
        self.assertEqual(actual, scalar_mask, atol=0, rtol=0)
        self.assertNotIn("mask_mod.__vec_size__", "\n".join(scalar_code))
        src = "\n".join(code)
        self.assertIn("score_mod.__vec_size__ = 8", src)
        self.assertIn("mask_mod.__vec_size__ = 32", src)
        self.assertIn("cute.autovec_copy", src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_mask_mod_rank1_aux_stays_scalar(self):
        # [FUNCTION DOC] SM100/SM110 only. A mask_mod using a rank-1 aux tensor indexed by kv must
        # not be vectorized: output matches eager and the generated code contains neither a mask
        # vec size marker nor autovec copies.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        col_mask = torch.randn(seq_len, device="cuda", dtype=torch.float16)

        def fn(q, k, v, col_mask):
            # [FUNCTION DOC] HELPER: function under test building the block mask from the nested
            # mask_mod and running FLASH flex_attention.
            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where causal and the column mask value is
                # positive.
                return (q_idx >= kv_idx) & (col_mask[kv_idx] > 0)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        expected = fn(q, k, v, col_mask)
        torch._dynamo.reset()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False), q, k, v, col_mask
        )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        src = "\n".join(code)
        self.assertNotIn("mask_mod.__vec_size__", src)
        self.assertNotIn("cute.autovec_copy", src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_mask_mod_vec_lane_uniform_no_aux(self):
        # [FUNCTION DOC] SM100/SM110 only. A mask_mod depending only on the query index (lane-
        # uniform, no aux tensors) is vectorized without autovec copies; uses the shared vec-
        # versus-scalar assert helper with expect_autovec disabled.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: function under test building a block mask and running FLASH
            # attention.
            def mask_mod(_b, _h, q_idx, _kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true for query indices below 64.
                return q_idx < 64

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        self._assert_sm100_mask_vec_matches_scalar(fn, q, k, v, expect_autovec=False)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    @parametrize("dynamic", [False, True], name_fn=lambda dynamic: str(dynamic))
    def test_flash_attention_sm100_sliding_window_uses_packed_shift_mask(self, dynamic):
        # [FUNCTION DOC] SM100/SM110 only, parametrized on dynamic shapes (static or dynamic with
        # the sequence dim marked dynamic at a fixed size). A sliding window causal mask_mod must
        # compile to the packed shift-based mask: output matches eager and the code contains mask
        # vec size 32 and the unsigned shift helpers, but no per-lane loop.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        if dynamic:
            for tensor in (q, k, v):
                torch._dynamo.mark_dynamic(tensor, 2, min=seq_len, max=seq_len)

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: function under test; takes lengths from tensor sizes when
            # dynamic, builds the block mask, and runs FLASH attention.
            q_len = q.size(2) if dynamic else seq_len
            kv_len = k.size(2) if dynamic else seq_len

            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where causal and within a window of 32
                # positions.
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= 32)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, q_len, kv_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        expected = fn(q, k, v)
        torch._dynamo.reset()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=dynamic), q, k, v
        )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        src = "\n".join(code)
        self.assertIn("mask_mod.__vec_size__ = 32", src)
        self.assertIn("utils.shr_u32", src)
        self.assertIn("utils.shl_u32", src)
        self.assertNotIn("for mask_lane_idx in cutlass.range_constexpr", src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_packed_mask_disabled_by_explicit_vec_config(self):
        # [FUNCTION DOC] SM100/SM110 only. With an explicit mask_mod vec size of 16 forced through
        # the config patch, the sliding-window mask must not use the packed shift path: output
        # matches eager, the code has mask vec size 16 and does not contain the unsigned shift
        # helper.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: function under test building the block mask and running FLASH
            # attention.
            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where causal and within a window of 32
                # positions.
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= 32)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        expected = fn(q, k, v)
        torch._dynamo.reset()
        with use_explicit_flex_flash_mask_mod_vec_size(16):
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), q, k, v
            )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        src = "\n".join(code)
        self.assertIn("mask_mod.__vec_size__ = 16", src)
        self.assertNotIn("utils.shr_u32", src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    @parametrize(
        "dynamic, expect_packed",
        [(False, True), (True, False)],
        name_fn=lambda dynamic, expect_packed: f"dynamic_{dynamic}",
    )
    def test_flash_attention_sm100_document_offsets_packed_mask(
        self, dynamic, expect_packed
    ):
        # [FUNCTION DOC] SM100/SM110 only, parametrized by dynamic and expect_packed (static
        # expects the packed mask path, dynamic does not). Mask looks up the document start offset
        # through captured document-id and offsets tensors; score_mod adds a kv bias. Compares to
        # eager and checks the generated code: packed case has the shift helper, score vec size 8
        # and no per-lane loop; the other case has no shift helper and has the per-lane loop.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        if dynamic:
            for tensor in (q, k, v):
                torch._dynamo.mark_dynamic(tensor, 2, min=seq_len, max=seq_len)
        positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)
        doc_ids = (positions // 32).expand(1, -1).contiguous()
        if dynamic:
            torch._dynamo.mark_dynamic(doc_ids, 1, min=seq_len, max=seq_len)
        offsets = torch.arange(0, seq_len + 32, 32, device="cuda", dtype=torch.int32)
        score_bias = torch.randn(seq_len, device="cuda", dtype=torch.float16)

        def fn(q, k, v, doc_ids, offsets, score_bias):
            # [FUNCTION DOC] HELPER: function under test; uses tensor sizes for lengths when
            # dynamic, defines the score and mask mods, builds the block mask and runs FLASH
            # attention.
            q_len = q.size(2) if dynamic else seq_len
            kv_len = k.size(2) if dynamic else seq_len

            def score_mod(score, _b, _h, _q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the score bias at the kv index.
                return score + score_bias[kv_idx]

            def mask_mod(b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod keeping kv positions between the start offset of
                # the query's document and the query position (document causal mask).
                doc = doc_ids[b, q_idx]
                start = offsets[doc]
                return (kv_idx >= start) & (kv_idx <= q_idx)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, q_len, kv_len, device="cuda"
            )
            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )

        args = (q, k, v, doc_ids, offsets, score_bias)
        expected = fn(*args)
        torch._dynamo.reset()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=dynamic), *args
        )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        src = "\n".join(code)
        if expect_packed:
            self.assertIn("utils.shr_u32", src)
            self.assertIn("score_mod.__vec_size__ = 8", src)
            self.assertNotIn("for mask_lane_idx in cutlass.range_constexpr", src)
        else:
            self.assertNotIn("utils.shr_u32", src)
            self.assertIn("for mask_lane_idx in cutlass.range_constexpr", src)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_mask_mod_explicit_vec_size(self):
        # [FUNCTION DOC] SM100/SM110 only. A mask_mod selecting only four lane positions out of 32
        # of the kv index (causal and lane in a fixed set) is compiled with an explicit mask vec
        # size of 8; output must match eager and the code must carry mask vec size 8.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: function under test building the block mask and running FLASH
            # attention.
            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where causal and the kv index modulo 32 is
                # 0, 15, 16 or 31.
                lane = kv_idx % 32
                return (q_idx >= kv_idx) & (
                    (lane == 0) | (lane == 15) | (lane == 16) | (lane == 31)
                )

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        expected = fn(q, k, v)
        torch._dynamo.reset()
        with use_explicit_flex_flash_mask_mod_vec_size(8):
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), q, k, v
            )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        self.assertIn("mask_mod.__vec_size__ = 8", "\n".join(code))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11),
        "SM100/SM110 only",
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flash_attention_sm100_aux_mask_mod_vec_mixed_gather_tail(self):
        # [FUNCTION DOC] SM100/SM110 only. Uses sequence length 130 with aux tensors padded to 160
        # (a non-multiple of the vector width) and a mask_mod mixing a 2-D gather with a rank-1
        # column load; checks via the shared assert helper that vectorized and scalar codegen
        # agree.
        seq_len = 130
        padded_len = 160
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=seq_len,
            dim=64,
            dtype=torch.float16,
            device="cuda",
        )
        mask_bias = torch.randn(
            padded_len, padded_len, device="cuda", dtype=torch.float16
        )
        col_mask = torch.randn(seq_len, device="cuda", dtype=torch.float16)

        def fn(q, k, v, mask_bias, col_mask):
            # [FUNCTION DOC] HELPER: function under test building the block mask and running FLASH
            # attention.
            def mask_mod(_b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where both the 2-D mask bias and the column
                # mask are positive.
                return (mask_bias[q_idx, kv_idx] > 0) & (col_mask[kv_idx] > 0)

            block_mask = _create_block_mask_for_device(
                mask_mod, 1, 1, seq_len, seq_len, device="cuda"
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        self._assert_sm100_mask_vec_matches_scalar(fn, q, k, v, mask_bias, col_mask)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_shared_captured_buffers(self, device, dtype):
        # [FUNCTION DOC] Two captured tensors (per-query rule masks and a prefix mask) are
        # combined into one boolean tensor that a mask_mod indexes. Compares FLASH and TRITON
        # compiled outputs and q/k/v gradients against an fp32 eager reference within 1e-2
        # tolerances. Expected to fail on SM120+.
        B, H, Q_LEN, KV_LEN, D = 2, 1, 512, 512, 64

        def model(q, k, v, rule_masks_per_query, prefix_mask, backend=None):
            # [FUNCTION DOC] HELPER: model combining the captured masks, creating the block mask
            # and running flex_attention with the requested backend option (default backend when
            # none is given).
            masks = rule_masks_per_query & prefix_mask.unsqueeze(0)

            def mask_mod(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod reading the combined mask tensor at query, batch
                # and kv indices.
                return masks[q_idx, b, kv_idx]

            block_mask = _create_block_mask_for_device(
                mask_mod,
                B,
                1,
                Q_LEN,
                KV_LEN,
                device=device,
            )
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": backend} if backend else {},
            )

        q = torch.randn(B, H, Q_LEN, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, KV_LEN, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, KV_LEN, D, device=device, dtype=dtype, requires_grad=True)
        rule_masks = torch.randint(
            0, 2, (Q_LEN, B, KV_LEN), dtype=torch.bool, device=device
        )
        prefix = torch.randint(0, 2, (B, KV_LEN), dtype=torch.bool, device=device)

        ref = model(q.float(), k.float(), v.float(), rule_masks, prefix)
        grad = torch.randn_like(ref)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad)

        torch._dynamo.reset()
        triton_out = torch.compile(model, dynamic=False)(
            q, k, v, rule_masks, prefix, "TRITON"
        )
        triton_grads = torch.autograd.grad(triton_out, (q, k, v), grad.to(dtype))

        torch._dynamo.reset()
        flash_out = torch.compile(model, dynamic=False)(
            q, k, v, rule_masks, prefix, "FLASH"
        )
        flash_grads = torch.autograd.grad(flash_out, (q, k, v), grad.to(dtype))

        torch.testing.assert_close(triton_out, ref.to(dtype), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(flash_out, ref.to(dtype), atol=1e-2, rtol=1e-2)
        for actual, expected in zip(triton_grads, ref_grads):
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        for actual, expected in zip(flash_grads, ref_grads):
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_viewable_captured_buffer_no_copy(self, device, dtype):
        # [FUNCTION DOC] A captured tensor that is just a view of a plain input must be passed to
        # the kernel as a reinterpreted view rather than materialized by a pointwise copy kernel.
        # Checks the generated wrapper code for the reinterpret call and the absence of a fused
        # view kernel name. Expected to fail on SM120+.
        """A capture that is a view of a plain input buffer should be passed to
        the CuteDSL kernel via reinterpret_tensor(...) instead of being
        materialized with a fresh pointwise copy."""
        from torch._inductor.utils import run_and_get_code

        B, H, Q_LEN, KV_LEN, D = 2, 1, 512, 512, 64

        def model(q, k, v, mask_flat):
            # [FUNCTION DOC] HELPER: model that views a flat boolean input as a 3-D mask, builds a
            # block mask from a mask_mod indexing it, and runs FLASH attention.
            masks = mask_flat.view(Q_LEN, B, KV_LEN)

            def mask_mod(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod reading the captured mask at query, batch and kv
                # indices.
                return masks[q_idx, b, kv_idx]

            block_mask = _create_block_mask_for_device(
                mask_mod, B, 1, Q_LEN, KV_LEN, device=device
            )
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )

        q = torch.randn(B, H, Q_LEN, D, device=device, dtype=dtype)
        k = torch.randn(B, H, KV_LEN, D, device=device, dtype=dtype)
        v = torch.randn(B, H, KV_LEN, D, device=device, dtype=dtype)
        mask_flat = torch.randint(
            0, 2, (Q_LEN * B * KV_LEN,), device=device, dtype=torch.bool
        )

        torch._dynamo.reset()
        out, codes = run_and_get_code(
            torch.compile(model, dynamic=False), q, k, v, mask_flat
        )
        wrapper_code = "\n".join(codes)
        self.assertIn("reinterpret_tensor(", wrapper_code)
        self.assertNotIn("triton_poi_fused_view", wrapper_code)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_kernel_called(self, device, dtype):
        # [FUNCTION DOC] Profiles CUDA kernels of a compiled FLASH run with a causal score_mod and
        # asserts a kernel named like the flash attention CuTe kernel is present; then runs with
        # the TRITON backend and asserts it is absent. Expected to fail on SM120+.
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        compiled_fn = torch.compile(flex_attention)

        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(q, k, v, score_mod=_causal, kernel_options={"BACKEND": "FLASH"})

        self.assertTrue(
            prof_result["found"],
            lambda msg: f"{msg}\nFlash attention kernel not found. Available kernels: {prof_result['kernel_names']}",
        )

        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(
                q, k, v, score_mod=_causal, kernel_options={"BACKEND": "TRITON"}
            )

        self.assertFalse(
            prof_result["found"],
            lambda msg: f"{msg}\nFlash attention kernel unexpectedly found when BACKEND='TRITON'. Kernels: {prof_result['kernel_names']}",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_impl_error_with_requires_grad(self, device, dtype):
        # [FUNCTION DOC] A score_mod capturing a bias tensor that requires grad must make the
        # FLASH backend raise RuntimeError (message says FLASH cannot be used since captured
        # tensors require gradients) at compile time.
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)

        def score_mod_with_grad(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured per-head bias that requires
            # gradients.
            return score + bias[h]

        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='FLASH' but flash attention cannot be used.*require gradients",
        ):
            compiled_fn(
                q,
                k,
                v,
                score_mod=score_mod_with_grad,
                kernel_options={"BACKEND": "FLASH"},
            )

    def test_mixed_dtypes(self, device):
        # [FUNCTION DOC] Checks that FLASH rejects mixed query/key/value dtypes: query in a higher
        # precision, key and value cast to a lower one (fp32 and fp16 normally, fp16 and fp8 when
        # the platform supports fp8). Expects InductorError saying mixed dtypes are unsupported.
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        """Ensure flash attention rejects mixed dtypes (e.g., fp32 Q with fp16 K/V)"""
        B, H, S, D = 2, 8, 512, 64

        query = torch.randn(B, H, S, D, dtype=dtype_high, device=device)
        key = torch.randn(B, H, S, D, dtype=dtype_high, device=device).to(dtype_low)
        value = torch.randn(B, H, S, D, dtype=dtype_high, device=device).to(dtype_low)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        from torch._inductor.exc import InductorError

        with self.assertRaisesRegex(
            InductorError,
            "Mixed query, key, and value dtype is not supported on this platform",
        ):
            compiled_fn(query, key, value, kernel_options={"BACKEND": "FLASH"})

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_rejects_captured_buffer_with_grad(
        self, device, dtype
    ):
        # [FUNCTION DOC] A captured bias requiring grad with q requiring grad must raise
        # RuntimeError (FLASH cannot be used with captured tensors requiring gradients) when
        # running forward and backward through the compiled function.
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)

        def score_mod_with_capture(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured per-head bias that requires
            # gradients.
            return score + bias[h]

        q.requires_grad_(True)
        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='FLASH' but flash attention cannot be used.*require gradients",
        ):
            compiled_fn(
                q,
                k,
                v,
                score_mod=score_mod_with_capture,
                kernel_options={"BACKEND": "FLASH"},
            ).sum().backward()

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_kernel_called(self, device, dtype):
        # [FUNCTION DOC] Runs a compiled FLASH forward and backward (head dim 128) under the CUDA
        # kernel profiler and asserts the flash backward kernel appears among the profiled
        # kernels. Expected to fail on SM120+.
        q, k, v = create_test_tensors(dim=128, dtype=dtype, device=device)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        def run_for_profile():
            # [FUNCTION DOC] HELPER: clones q, k, v as fresh leaf tensors requiring grad, runs the
            # compiled FLASH attention and its backward on the sum so the profiler captures both
            # kernels.
            q_run, k_run, v_run = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v)
            ]
            compiled_fn(
                q_run, k_run, v_run, kernel_options={"BACKEND": "FLASH"}
            ).sum().backward()

        with cuda_kernel_profiler("flash_attncuteflash_bwd") as prof_result:
            run_for_profile()

        self.assertTrue(
            prof_result["found"],
            lambda msg: f"{msg}\nFlash attention backward kernel not found. Kernels: {prof_result['kernel_names']}",
        )

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_forwards_deterministic_flag(self, device, dtype):
        # [FUNCTION DOC] Checks that the deterministic flag is baked into generated flash backward
        # code at compile time: with deterministic mode off the code contains the false setting
        # and, with it on, the true setting; in both cases it must not call the runtime query of
        # deterministic mode. Expected to fail on SM120+.
        q, k, v = create_test_tensors(dim=128, dtype=dtype, device=device)
        q.requires_grad_(True)

        torch._dynamo.reset()
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        with DeterministicGuard(False):
            _, code = run_fw_bw_and_get_code(
                lambda: compiled_fn(q, k, v, kernel_options={"BACKEND": "FLASH"})
            )
        code_str = "\n".join(code)
        self.assertIn(
            "deterministic=False",
            code_str,
            "Expected deterministic=False to be baked into flash backward codegen",
        )
        self.assertNotIn(
            "are_deterministic_algorithms_enabled()",
            code_str,
            "Expected flash backward deterministic setting to be compile-time baked",
        )

        torch._dynamo.reset()
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        with DeterministicGuard(True):
            _, code = run_fw_bw_and_get_code(
                lambda: compiled_fn(q, k, v, kernel_options={"BACKEND": "FLASH"})
            )
        code_str = "\n".join(code)
        self.assertIn(
            "deterministic=True",
            code_str,
            "Expected deterministic=True to be baked into flash backward codegen",
        )
        self.assertNotIn(
            "are_deterministic_algorithms_enabled()",
            code_str,
            "Expected flash backward deterministic setting to be compile-time baked",
        )

    @dtypes(torch.bfloat16)
    @parametrize("dq_kv_order", [False, True])
    def test_flash_attention_backward_wires_write_order_codegen(
        self, device, dtype, dq_kv_order
    ):
        # [FUNCTION DOC] SM100+ only, parametrized by dq_kv_order. Builds two block masks with dq
        # write order, one with the flag and one with its opposite, checks both have a write order
        # and that the orders differ. Runs the forward and backward with deterministic mode and
        # inspects generated code: block sparse kwargs must include the write order, the spt/kv-
        # order flag with the requested value, and the full write order only if the mask has one.
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("deterministic block-sparse FLASH backward requires SM100+")

        q, k, v = create_test_tensors(
            seq_len=512,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        block_mask = create_block_mask(
            _causal_mask,
            2,
            4,
            512,
            512,
            device=device,
            BLOCK_SIZE=(
                _DEFAULT_SPARSE_BLOCK_SIZE * 2,
                _DEFAULT_SPARSE_BLOCK_SIZE,
            ),
            compute_dq_write_order=True,
            dq_kv_order=dq_kv_order,
        )
        self.assertEqual(block_mask._dq_kv_order(), dq_kv_order)
        opposite_spt_block_mask = create_block_mask(
            _causal_mask,
            2,
            4,
            512,
            512,
            device=device,
            BLOCK_SIZE=(
                _DEFAULT_SPARSE_BLOCK_SIZE * 2,
                _DEFAULT_SPARSE_BLOCK_SIZE,
            ),
            compute_dq_write_order=True,
            dq_kv_order=not dq_kv_order,
        )
        self.assertIsNotNone(block_mask.dq_write_order)
        self.assertIsNotNone(opposite_spt_block_mask.dq_write_order)
        self.assertFalse(
            torch.equal(
                block_mask.dq_write_order, opposite_spt_block_mask.dq_write_order
            )
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        def run_for_code():
            # [FUNCTION DOC] HELPER: runs the compiled FLASH attention with the block mask inside
            # a deterministic guard, returning the output.
            with DeterministicGuard(True):
                return compiled_fn(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    kernel_options={"BACKEND": "FLASH"},
                )

        _, code = run_fw_bw_and_get_code(run_for_code)
        code_str = "\n".join(code)
        self.assertIn('block_sparse_kwargs["dq_write_order"]', code_str)
        self.assertTrue(
            f'block_sparse_kwargs["spt"] = {dq_kv_order}' in code_str
            or f'block_sparse_kwargs["dq_kv_order"] = {dq_kv_order}' in code_str
        )
        if block_mask.dq_write_order_full is not None:
            self.assertIn('block_sparse_kwargs["dq_write_order_full"]', code_str)
        else:
            self.assertNotIn('block_sparse_kwargs["dq_write_order_full"]', code_str)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_generates_cute_hash(self, device, dtype):
        # [FUNCTION DOC] Verifies the generated code for a FLASH call with a causal score_mod sets
        # a cute hash attribute on the score_mod for fast hashing. Expected to fail on SM120+.
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=_causal,
            kernel_options={"BACKEND": "FLASH"},
        )

        code_str = "\n".join(code)
        self.assertIn(
            "score_mod.__cute_hash__",
            code_str,
            "Generated code should set __cute_hash__ on score_mod for fast hashing",
        )

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_fused_qkv_reinterpret_view(self, device, dtype):
        # [FUNCTION DOC] Fused QKV projection followed by view, unbind and transposes produces
        # non-trivial q, k, v views; a compiled FLASH attention over them must run and return the
        # expected shape (batch, heads, seq, dim). Expected to fail on SM120+.
        B, M, H, D = 2, 256, 4, 64
        embed_dim = H * D

        def fn(x, weight):
            # [FUNCTION DOC] HELPER: projects x with weight, splits into q, k, v views per head,
            # transposes them and runs FLASH attention.
            qkv = x @ weight
            qkv = qkv.view(B, M, 3, H, D)
            q, k, v = qkv.unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            return flex_attention(q, k, v, kernel_options={"BACKEND": "FLASH"})

        x = torch.randn(B, M, embed_dim, device=device, dtype=dtype)
        weight = torch.randn(embed_dim, 3 * embed_dim, device=device, dtype=dtype)

        compiled_fn = torch.compile(fn)
        out = compiled_fn(x, weight)
        self.assertEqual(out.shape, (B, H, M, D))

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_gqa_expand_stride_zero_backward(self, device, dtype):
        # [FUNCTION DOC] GQA backward where k and v are expanded views with zero head stride
        # (regression for gradient buffer strides). Asserts the zero strides, then runs
        # flash_vs_triton with a causal block mask; on SM120+ instead expects a ValueError about
        # block sparsity sparse block size needing to match the tile size. Expected to fail on
        # SM120+ otherwise by decorator.
        """Test GQA backward with expand()-created K/V tensors (stride=0).

        Regression test for gradient buffer stride bug with expand().
        """
        batch_size = 1
        seqlen = 512
        headdim = 128
        n_heads = 4
        n_kv_heads = 1

        q, k_orig, v_orig = create_test_tensors(
            batch_size=batch_size,
            num_heads=n_heads,
            num_heads_kv=n_kv_heads,
            seq_len=seqlen,
            dim=headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        k = k_orig.expand(batch_size, n_heads, seqlen, headdim)
        v = v_orig.expand(batch_size, n_heads, seqlen, headdim)

        self.assertEqual(k.stride()[1], 0, "K should have stride=0 from expand()")
        self.assertEqual(v.stride()[1], 0, "V should have stride=0 from expand()")

        block_mask = _create_block_mask_for_device(
            _causal_mask, batch_size, n_heads, seqlen, seqlen, device=device
        )

        if SM120OrLater:
            # note: see [SM120 forward tile selection failure]
            with self.assertRaisesRegex(
                ValueError,
                r"Block sparsity requires sparse_block_size[1]=64 to match tile_n",
            ):
                flash_vs_triton(q, k, v, block_mask=block_mask)
        else:
            flash_vs_triton(q, k, v, block_mask=block_mask)

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_backend_return_lse_matches_triton_and_reference(self, device, dtype):
        # [FUNCTION DOC] With return_lse and unit scale, the logsumexp from FLASH and TRITON
        # compiled kernels must both match a float32 reference logsumexp of q times k transposed
        # and match each other (tolerance 1e-4). Expected to fail on SM120+.
        torch.manual_seed(0)
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=128,
            dim=32,
            dtype=dtype,
            device=device,
        )

        flash_flex = torch.compile(
            functools.partial(
                flex_attention,
                scale=1.0,
                return_lse=True,
                kernel_options={"BACKEND": "FLASH"},
            )
        )
        triton_flex = torch.compile(
            functools.partial(
                flex_attention,
                scale=1.0,
                return_lse=True,
                kernel_options={"BACKEND": "TRITON"},
            )
        )

        _, lse_flash = flash_flex(q, k, v)
        _, lse_triton = triton_flex(q, k, v)
        ref_lse = torch.logsumexp(
            torch.matmul(q.float(), k.float().transpose(-2, -1)), dim=-1
        )

        torch.testing.assert_close(lse_triton.float(), ref_lse, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_flash.float(), ref_lse, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            lse_flash.float(), lse_triton.float(), atol=1e-4, rtol=1e-4
        )

    @xfailIfSM120OrLater
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_backend_supports_grad_logsumexp(self, device, dtype):
        # [FUNCTION DOC] Checks that FLASH backward supports gradients flowing from both the
        # attention output and the logsumexp. Runs both backends with a score_mod doubling scores
        # on cloned leaf tensors, compares each to an fp32 eager reference, requires finiteness
        # and that the FLASH error is within twice the TRITON error plus a noise-floor tolerance.
        # Expected to fail on SM120+.
        torch.manual_seed(0)
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=1,
            seq_len=128,
            dim=32,
            dtype=dtype,
            device=device,
        )
        grad_out = torch.randn_like(q)
        grad_lse = torch.randn(1, 1, 128, device=device, dtype=torch.float32) * 3

        def run_backend(backend, q_in, k_in, v_in):
            # [FUNCTION DOC] HELPER: compiles flex_attention for the given backend with
            # return_lse, runs it and returns the gradients of q, k, v given the output and
            # logsumexp upstream gradients.
            compiled_flex = torch.compile(
                functools.partial(
                    flex_attention,
                    score_mod=_times_two,
                    scale=1.0,
                    return_lse=True,
                    kernel_options={"BACKEND": backend},
                )
            )
            out, lse = compiled_flex(q_in, k_in, v_in)
            return torch.autograd.grad(
                (out, lse), (q_in, k_in, v_in), (grad_out, grad_lse)
            )

        q_flash, k_flash, v_flash = [
            t.detach().clone().requires_grad_() for t in (q, k, v)
        ]
        q_triton, k_triton, v_triton = [
            t.detach().clone().requires_grad_() for t in (q, k, v)
        ]
        grads_flash = run_backend("FLASH", q_flash, k_flash, v_flash)
        grads_triton = run_backend("TRITON", q_triton, k_triton, v_triton)

        q_ref, k_ref, v_ref = [t.detach().float().requires_grad_() for t in (q, k, v)]
        out_ref, lse_ref = flex_attention(
            q_ref,
            k_ref,
            v_ref,
            score_mod=_times_two,
            scale=1.0,
            return_lse=True,
        )
        grads_ref = torch.autograd.grad(
            (out_ref, lse_ref),
            (q_ref, k_ref, v_ref),
            (grad_out.float(), grad_lse),
        )

        for grad_flash, grad_triton, grad_ref in zip(
            grads_flash, grads_triton, grads_ref
        ):
            self.assertTrue(torch.isfinite(grad_flash).all())
            self.assertTrue(torch.isfinite(grad_triton).all())
            self.assertTrue(torch.isfinite(grad_ref).all())
            atol = 2 * (grad_ref + 0.3 - 0.3 - grad_ref).abs().max().item()
            triton_error = (
                (grad_triton - grad_ref.to(grad_triton.dtype)).abs().max().item()
            )
            flash_error = (
                (grad_flash - grad_ref.to(grad_flash.dtype)).abs().max().item()
            )
            self.assertLessEqual(flash_error, 2 * triton_error + atol)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_backend_raises_on_return_max_scores(self, device, dtype):
        # [FUNCTION DOC] Requesting max scores through the auxiliary return request with the FLASH
        # backend must raise NotImplementedError with a message saying max scores are unsupported
        # for FLASH.
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Returning max scores is not supported with BACKEND='FLASH'",
        ):
            flex_attention(
                q,
                k,
                v,
                return_aux=AuxRequest(max_scores=True),
                kernel_options={"BACKEND": "FLASH"},
            )

    # 'FlashAttentionForwardSm120' object has no attribute 'apply_score_mod'
    @xfailIfSM120OrLater
    @decorateIf(
        unittest.expectedFailure,
        lambda params: IS_SM90,
    )
    @torch._inductor.config.patch(max_autotune=True)
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize(
        "score_mod_fn",
        [_times_two, _rel_bias],
        name_fn=lambda fn: fn.__name__,
    )
    def test_max_autotune_score_mod(self, device, dtype, score_mod_fn):
        # [FUNCTION DOC] Max-autotune enabled; parametrized over two score_mod functions (times
        # two and relative bias) and dtype. Runs flash_vs_triton with requires-grad inputs.
        # Expected to fail on SM120+ and on SM90.
        q, k, v = create_test_tensors(
            batch_size=2,
            num_heads=4,
            seq_len=257,
            dim=64,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        flash_vs_triton(q, k, v, score_mod=score_mod_fn)


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
class TestFlexFlashDynamicShapes(InductorTestCase):
    # [CLASS DOC] Test class for FLASH backend dynamic-shape compilation (score_mod captures,
    # block masks, backward, batch size, GQA); skipped if the flash library is missing. Each test
    # compares FLASH versus TRITON compiled with dynamic shapes over several sizes, via two helper
    # methods.
    """
    Dynamic-shape coverage for flex flash attention: score_mod captures and masks,
    plus backward, batch, and length variants.
    """

    def _run_dynamic_test(
        self, seq_lens, score_mod=None, block_mask_factory=None, requires_grad=False
    ):
        # [FUNCTION DOC] HELPER: for each sequence length, creates float16 CUDA q, k, v
        # (optionally requiring grad), builds the block mask from the optional factory called with
        # that length, and calls flash_vs_triton with dynamic compilation.
        """Helper to run dynamic=True tests across multiple sequence lengths."""

        for seq_len in seq_lens:
            q, k, v = create_test_tensors(
                seq_len=seq_len,
                device="cuda",
                dtype=torch.float16,
                requires_grad=requires_grad,
            )
            block_mask = block_mask_factory(seq_len) if block_mask_factory else None
            flash_vs_triton(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                dynamic=True,
            )

    def _flash_triton_dynamic(self, q, k, v, **kwargs):
        # [FUNCTION DOC] HELPER: calls flash_vs_triton with dynamic compilation enabled,
        # forwarding extra keyword arguments.
        flash_vs_triton(q, k, v, dynamic=True, **kwargs)

    # sm120: AttributeError: 'NoneType' object has no attribute '_trait'
    @xfailIfSM120OrLater
    def test_dynamic_seq_len_no_score_mod(self):
        # [FUNCTION DOC] Dynamic sequence lengths 128, 256, 512 without score_mod. Expected to
        # fail on SM120+.
        """Test dynamic sequence lengths without score_mod."""
        self._run_dynamic_test(seq_lens=[128, 256, 512])

    @xfailIfSM120OrLater
    def test_dynamic_seq_len_inline_literal(self):
        # [FUNCTION DOC] Dynamic sequence lengths with a score_mod that multiplies by a literal
        # constant (no captured values).
        """Test dynamic sequence lengths with inline literal score_mod."""

        def score_mod(score, _b, _h, _q, _k):
            # [FUNCTION DOC] HELPER: score_mod multiplying the score by the literal 2.0.
            return score * 2.0  # Inline literal, not captured

        self._run_dynamic_test(seq_lens=[128, 256, 512], score_mod=score_mod)

    @xfailIfSM120OrLater
    def test_dynamic_seq_len_captured_tensor_buffer(self):
        # [FUNCTION DOC] Dynamic sequence lengths with a score_mod capturing a per-head ALiBi
        # slope tensor.
        """Test dynamic sequence lengths with captured tensor buffer (ALiBi-style)."""
        num_heads = 4
        slopes = torch.exp2(
            -torch.linspace(1, 8, num_heads, device="cuda", dtype=torch.float16)
        )

        def alibi_score_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding kv-minus-q distance times the captured head
            # slope.
            return score + (kv_idx - q_idx) * slopes[h]

        self._run_dynamic_test(seq_lens=[128, 256, 512], score_mod=alibi_score_mod)

    @xfailIfSM120OrLater
    def test_dynamic_captured_table_negative_index_wrap(self):
        # [FUNCTION DOC] Regression test for a relative-position table gather with offset
        # depending on the sequence length (see docstring issue) under dynamic shapes for lengths
        # 512 and 1024: the negative-index wrap must cast the table length and offset to the index
        # dtype. Defines a new score_mod per length and compares with flash_vs_triton.
        """Rel-pos table gather with a shape-dependent offset (#188871).

        Once shapes go dynamic, the table length and offset are rendered as Int64
        aux scalars; the emitted negative-index wrap must cast them to the Int32
        index dtype or cute.where rejects the mixed operands.
        """
        for seq_len in [512, 1024]:
            tbl = torch.randn(2 * seq_len, device="cuda")
            offset = seq_len - 1

            def score_mod(score, _b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the captured table entry at the shifted
                # relative index.
                return score + tbl[q_idx - kv_idx + offset]

            q, k, v = create_test_tensors(seq_len=seq_len, device="cuda")
            flash_vs_triton(q, k, v, score_mod=score_mod, dynamic=True)

    @xfailIfSM120OrLater
    def test_dynamic_seq_len_with_block_mask(self):
        # [FUNCTION DOC] Dynamic sequence lengths with a causal block mask created per length.
        """Test dynamic sequence lengths with block mask."""

        def block_mask_factory(seq_len):
            # [FUNCTION DOC] HELPER: builds a causal block mask (batch 2, heads 4) for the given
            # length.
            return _create_block_mask_for_device(
                _causal_mask, 2, 4, seq_len, seq_len, device="cuda"
            )

        self._run_dynamic_test(
            seq_lens=[128, 256, 512], block_mask_factory=block_mask_factory
        )

    @xfailIfSM120OrLater
    def test_dynamic_batch_size(self):
        # [FUNCTION DOC] Runs FLASH versus TRITON with dynamic shapes for batch sizes 1, 2, 4 and
        # 8.
        """Test dynamic batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            q, k, v = create_test_tensors(
                batch_size=batch_size, seq_len=256, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    @xfailIfSM120OrLater
    def test_dynamic_backward(self):
        # [FUNCTION DOC] Backward through FLASH with dynamic sequence lengths (inputs require
        # grad).
        """Test backward with dynamic sequence lengths."""
        self._run_dynamic_test(seq_lens=[128, 256, 512], requires_grad=True)

    # 'FlashAttentionForwardSm120' object has no attribute 'apply_score_mod'
    @xfailIfSM120OrLater
    def test_dynamic_backward_with_score_mod(self):
        # [FUNCTION DOC] Backward with a doubling score_mod and dynamic sequence lengths.
        """Test backward with score_mod and dynamic sequence lengths."""

        def score_mod(score, _b, _h, _q, _k):
            # [FUNCTION DOC] HELPER: score_mod multiplying the score by the literal 2.0.
            return score * 2.0

        self._run_dynamic_test(
            seq_lens=[128, 256, 512], score_mod=score_mod, requires_grad=True
        )

    @xfailIfSM120OrLater
    def test_dynamic_backward_with_block_mask(self):
        # [FUNCTION DOC] Backward with a causal block mask and dynamic sequence lengths; skipped
        # below SM90.
        """Test backward with block mask and dynamic sequence lengths."""
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+")

        def block_mask_factory(seq_len):
            # [FUNCTION DOC] HELPER: builds a causal block mask (batch 2, heads 4) for the given
            # length.
            return _create_block_mask_for_device(
                _causal_mask, 2, 4, seq_len, seq_len, device="cuda"
            )

        self._run_dynamic_test(
            seq_lens=[128, 256, 512],
            block_mask_factory=block_mask_factory,
            requires_grad=True,
        )

    @xfailIfSM120OrLater
    def test_dynamic_gqa(self):
        # [FUNCTION DOC] Grouped-query attention (8 query heads, 2 kv heads) with dynamic shapes
        # over sequence lengths 128, 256, 512; random float16 q, k, v, no score_mod or block mask.
        """Test GQA with dynamic sequence lengths."""
        q_heads, kv_heads = 8, 2
        for seq_len in [128, 256, 512]:
            q = torch.randn(2, q_heads, seq_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            v = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v, score_mod=None, block_mask=None)

    @xfailIfSM120OrLater
    def test_dynamic_mqa(self):
        # [FUNCTION DOC] Multi-query attention (8 query heads, 1 kv head) with dynamic shapes over
        # sequence lengths 128, 256, 512. Expected to fail on SM120+.
        """Test MQA with dynamic sequence lengths."""
        q_heads, kv_heads = 8, 1
        for seq_len in [128, 256, 512]:
            q = torch.randn(2, q_heads, seq_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            v = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    @xfailIfSM120OrLater
    def test_dynamic_non_divisible_seq_len(self):
        # [FUNCTION DOC] Dynamic shapes with sequence lengths that are not multiples of the block
        # size (127, 255, 383, 511, 513). Expected to fail on SM120+.
        """Test non-block-divisible sequence lengths with dynamic shapes."""
        for seq_len in [127, 255, 383, 511, 513]:
            q, k, v = create_test_tensors(
                seq_len=seq_len, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    @xfailIfSM120OrLater
    def test_dynamic_asymmetric_qkv_lengths(self):
        # [FUNCTION DOC] Dynamic shapes with different query and kv lengths ((256, 512), (512,
        # 256), (128, 1024)). Expected to fail on SM120+.
        """Test asymmetric Q and KV lengths with dynamic shapes."""
        test_cases = [(256, 512), (512, 256), (128, 1024)]
        for q_len, kv_len in test_cases:
            q = torch.randn(2, 4, q_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(2, 4, kv_len, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(2, 4, kv_len, 64, device="cuda", dtype=torch.float16)
            self._flash_triton_dynamic(q, k, v)

    def test_captured_float_fails_with_dynamic(self):
        # [FUNCTION DOC] A score_mod capturing a Python float under dynamic shapes is rejected:
        # the compile must raise RuntimeError whose message says it captures a 0-dim CPU tensor.
        """Test that captured Python float still fails as a CPU scalar tensor."""
        val = 2.0

        def score_mod(score, _b, _h, _q, _k):
            # [FUNCTION DOC] HELPER: score_mod multiplying the score by the captured Python float.
            return score * val

        compiled_fn = torch.compile(flex_attention, dynamic=True)
        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, r"captures a 0-dim CPU tensor"):
            compiled_fn(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
            )

    @xfailIfSM120OrLater
    def test_captured_int_works_with_dynamic(self):
        # [FUNCTION DOC] A captured Python int is accepted under dynamic shapes; checked by
        # flash_vs_triton with dynamic compilation. Expected to fail on SM120+.
        """Captured Python int should work with dynamic=True."""
        val = 2

        def score_mod(score, _b, _h, _q, _k):
            # [FUNCTION DOC] HELPER: score_mod multiplying the score by the captured Python int.
            return score * val

        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        flash_vs_triton(q, k, v, score_mod=score_mod, dynamic=True)

    @xfailIfSM120OrLater
    def test_captured_float_works_with_static(self):
        # [FUNCTION DOC] A captured Python float is fine under static compilation; the FLASH
        # output must have the same shape as q. Expected to fail on SM120+.
        """Test that captured Python float works with dynamic=False."""
        val = 2.0  # Captured float

        def score_mod(score, _b, _h, _q, _k):
            # [FUNCTION DOC] HELPER: score_mod multiplying the score by the captured Python float.
            return score * val

        compiled_fn = torch.compile(flex_attention, dynamic=False)
        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        out = compiled_fn(
            q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
        )
        self.assertEqual(out.shape, q.shape)

    @xfailIfSM120OrLater
    def test_dynamic_mask_from_input_lengths_single_graph(self):
        # [FUNCTION DOC] A block mask created by a compiled create_block_mask whose mask_mod
        # depends on a per-batch input_lengths tensor, followed by compiled FLASH attention in a
        # small module, run three times with different lengths must produce only one compiled
        # graph (frame count of 1, counted by a compile counter backend). Expected to fail on
        # SM120+.
        """Dynamic mask creation driven by input lengths should stay single-graph."""
        counter = CompileCounterWithBackend("inductor")

        def _flex_attention_mask(b, h, q_idx, kv_idx, input_lengths):
            # [FUNCTION DOC] HELPER: mask_mod keeping positions where both query and kv indices
            # are below the batch element's length.
            return (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: small test module with a linear projection and head count; its
            # forward projects the input, reshapes into heads, builds a dynamic block mask from
            # the input lengths with a compiled mask-creation function, and runs compiled FLASH
            # attention on x as q, k, v.
            def __init__(self, dim=128, heads=4):
                # [FUNCTION DOC] HELPER: creates the linear projection layer (dim to dim) and
                # stores the head count.
                super().__init__()
                self.proj = torch.nn.Linear(dim, dim)
                self.heads: int = int(heads)

            def forward(self, x, input_lengths):
                # [FUNCTION DOC] HELPER: projects x, reshapes to (batch, heads, time, head_dim),
                # creates the block mask for the time length using the lengths tensor, and returns
                # compiled FLASH flex_attention output; the compile counter backend records graph
                # count.
                x = self.proj(x)
                B, T, C = x.shape
                head_dim = C // self.heads
                x = x.view(B, T, self.heads, head_dim).permute(0, 2, 1, 3)

                max_time = x.size(-2)
                mask = torch.compile(create_block_mask, dynamic=True, fullgraph=False)(
                    functools.partial(
                        _flex_attention_mask, input_lengths=input_lengths
                    ),
                    B=B,
                    H=None,
                    Q_LEN=max_time,
                    KV_LEN=max_time,
                    device=x.device,
                )

                return torch.compile(
                    flex_attention, dynamic=True, fullgraph=True, backend=counter
                )(x, x, x, block_mask=mask, kernel_options={"BACKEND": "FLASH"})

        model = Model().cuda().half()
        B, T, F = 8, 64, 128
        for _ in range(3):
            x = torch.randn(B, T, F, device="cuda", dtype=torch.float16)
            lens = torch.randint(1, T + 1, (B,), device="cuda")
            model(x, lens)

        self.assertEqual(
            counter.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 graph, got {counter.frame_count}",
        )

    @xfailIfSM120OrLater
    def test_dynamic_free_symbol_mask_single_graph(self):
        # [FUNCTION DOC] A dense random boolean mask captured as a free symbol inside a compiled
        # create_block_mask must not trigger recompilation under dynamic shapes: attention runs
        # for batch sizes 2, 4, 8 and the compile counter must report a single frame. Expected to
        # fail on SM120+.
        """Free-symbol dense mask under dynamic=True should not recompile."""
        counter = CompileCounterWithBackend("inductor")

        def make_mask(batch_shape, seq_len):
            # [FUNCTION DOC] HELPER: builds a random boolean mask of the given batch and sequence
            # size and returns a block mask from a compiled create_block_mask whose mask_mod
            # negates the mask entry.
            rand_mask = torch.randint(
                0, 2, (batch_shape, seq_len), device="cuda"
            ).bool()
            return torch.compile(create_block_mask, dynamic=True)(
                B=batch_shape,
                BLOCK_SIZE=128,
                mask_mod=lambda b, h, q_idx, kv_idx: ~rand_mask[b, q_idx],
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device="cuda",
            )

        @torch.compile(dynamic=True, fullgraph=True, backend=counter)
        def run(q, k, v, block_mask):
            # [FUNCTION DOC] HELPER: compiled (dynamic, full graph, counting backend) function
            # running FLASH flex_attention with the supplied block mask.
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        seq_len = 128
        for batch_shape in [2, 4, 8]:
            q, k, v = create_test_tensors(
                batch_shape, 4, seq_len, 64, device="cuda", dtype=torch.float16
            )
            block_mask = make_mask(batch_shape, seq_len)
            run(q, k, v, block_mask)

        self.assertEqual(
            counter.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 graph, got {counter.frame_count}",
        )

    @xfailIfSM120OrLater
    def test_dynamic_max_autotune_with_block_mask(self):
        # [FUNCTION DOC] Dynamic shapes with max-autotune (no cudagraphs) and a causal block mask
        # in bfloat16 must compile and run on the FLASH backend; asserts the output shape equals
        # q's. Expected to fail on SM120+.
        """Dynamic=True with max-autotune should succeed for FLASH backend."""
        q, k, v = create_test_tensors(
            batch_size=2,
            num_heads=4,
            seq_len=256,
            dim=64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        block_mask = _create_block_mask_for_device(
            _causal_mask, 2, 4, 256, 256, device="cuda"
        )

        compiled_fn = torch.compile(
            flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
        )
        out = compiled_fn(
            q,
            k,
            v,
            block_mask=block_mask,
            kernel_options={"BACKEND": "FLASH"},
        )
        self.assertEqual(out.shape, q.shape)

    @xfailIfSM120OrLater
    def test_dynamic_captured_buffer_varying_heads(self):
        # [FUNCTION DOC] Runs a captured per-head scale tensor score_mod under dynamic shapes with
        # head counts 4, 8, 4, 16, 4, comparing FLASH against TRITON each time. Resets dynamo
        # first. Expected to fail on SM120+.
        """Dynamic head_count with captured tensor buffer under FLASH/TRITON parity."""
        torch._dynamo.reset()

        def run_with_head_count(head_count):
            # [FUNCTION DOC] HELPER: creates a head-scale tensor, a score_mod multiplying by the
            # head's scale, random q, k, v requiring grad, and calls the dynamic comparison
            # helper.
            head_scale = torch.randn(
                head_count, device="cuda", dtype=torch.float16, requires_grad=False
            )

            def score_mod(score, batch, head, token_q, token_kv):
                # [FUNCTION DOC] HELPER: score_mod multiplying the score by the captured scale of
                # the head.
                return score * head_scale[head]

            q, k, v = create_test_tensors(
                batch_size=2,
                num_heads=head_count,
                seq_len=256,
                dim=64,
                dtype=torch.float16,
                device="cuda",
                requires_grad=True,
            )
            self._flash_triton_dynamic(q, k, v, score_mod=score_mod)

        for head_count in [4, 8, 4, 16, 4]:
            run_with_head_count(head_count)

    def test_dynamic_symbol_closure_in_score_mod(self):
        # [FUNCTION DOC] A score_mod closing over the symbolic batch size from the input shape
        # must lead to a single dynamic graph. A small attention module with a QKV projection is
        # compiled with the graph-recording backend, run with batch sizes 2, 4, 8, and exactly one
        # graph is expected.
        """Capturing a SymInt in score_mod should compile to one dynamic graph."""

        class SimpleAttention(torch.nn.Module):
            # [CLASS DOC] HELPER: attention module with a combined QKV linear layer; forward
            # reshapes into q, k, v heads and runs FLASH flex_attention with a score_mod adding
            # the batch size to scores.
            def __init__(self, dim=512, n_head=8):
                # [FUNCTION DOC] HELPER: creates the QKV linear layer and stores head count and
                # head dimension.
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head: int = int(n_head)
                self.head_dim: int = int(dim // n_head)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: computes q, k, v from the projection and returns FLASH
                # flex_attention with a score_mod that adds the symbolic batch size.
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                return flex_attention(
                    q,
                    k,
                    v,
                    score_mod=lambda s, b, h, q_idx, kv_idx: s + B,
                    kernel_options={"BACKEND": "FLASH"},
                )

        model = SimpleAttention().cuda()
        backend = EagerAndRecordGraphs()
        model.compile(mode="default", dynamic=True, backend=backend)

        torch._dynamo.reset()
        for batch_shape in [2, 4, 8]:
            x = torch.randn(batch_shape, 256, 512, device="cuda")
            model(x)

        self.assertEqual(len(backend.graphs), 1, "Expected a single dynamic graph")

    def _dynamic_multiple_scalar_closures_fn(self):
        # [FUNCTION DOC] HELPER: returns a function whose score_mod closes over two symbolic
        # scalars (batch and sequence length from tensor sizes) and adds them plus the position
        # difference.
        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: function taking q, k, v, reading batch and sequence sizes,
            # and running FLASH flex_attention with a score_mod capturing both.
            batch = q.size(0)
            seq_len = q.size(2)

            def score_mod(score, _b, _h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the two captured sizes and the q minus
                # kv distance.
                return score + batch + seq_len + q_idx - kv_idx

            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                kernel_options={"BACKEND": "FLASH"},
            )

        return fn

    def test_dynamic_multiple_scalar_closures_in_score_mod_codegen(self):
        # [FUNCTION DOC] With dynamic shapes and two captured scalar sizes, generated code must
        # pass both as 64-bit auxiliary scalars. Skipped on SM120+. Checks output against eager
        # and that the code contains a tuple of two cutlass Int64 values (regex) and uses elements
        # 0 and 1.
        if SM120OrLater:
            self.skipTest("score_mod is not supported on SM120")

        fn = self._dynamic_multiple_scalar_closures_fn()
        q, k, v = create_test_tensors(seq_len=128, device="cuda", dtype=torch.float16)
        expected = fn(q, k, v)
        actual, code = run_and_get_code(
            torch.compile(fn, dynamic=True, fullgraph=True), q, k, v
        )

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)
        src = "\n".join(code)
        self.assertRegex(
            src,
            r"aux_scalars = \(cutlass\.Int64\([^)]+\), cutlass\.Int64\([^)]+\)\)",
        )
        self.assertIn("aux_scalars[0]", src)
        self.assertIn("aux_scalars[1]", src)

    @xfailIfSM120OrLater
    @decorateIf(
        unittest.expectedFailure,
        lambda params: IS_SM90,
    )
    def test_dynamic_multiple_scalar_closures_with_max_autotune(self):
        # [FUNCTION DOC] Two captured scalar sizes with dynamic shapes under max-autotune (no
        # cudagraphs); output compared to eager. Expected to fail on SM120+ and SM90.
        fn = self._dynamic_multiple_scalar_closures_fn()
        q, k, v = create_test_tensors(seq_len=128, device="cuda", dtype=torch.float16)
        expected = fn(q, k, v)
        actual = torch.compile(
            fn,
            dynamic=True,
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )(q, k, v)

        self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)

    def _offset_block_mask(self, offset, q_len=128, kv_len=128):
        # [FUNCTION DOC] HELPER: builds a block mask with a causal-with-offset mask_mod (kv index
        # at most query index plus offset), batch 2, heads broadcast, for the given q and kv
        # lengths.
        def mask_mod(_b, _h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod keeping positions where kv index is at most query
            # plus offset.
            return kv_idx <= q_idx + offset

        return _create_block_mask_for_device(
            mask_mod, 2, None, q_len, kv_len, device="cuda"
        )

    def test_dynamic_scalar_closure_in_mask_mod(self):
        # [FUNCTION DOC] A mask_mod closing over a scalar offset under dynamic shapes with FLASH
        # and TRITON backends. Skipped without aux scalar support, on SM120+ or below SM90. For
        # offsets 0, 1, 5 compares outputs against an fp32 reference and each other and compares
        # gradients between backends; each backend must compile only once (frame count of 1).
        if not flash_supports_aux_scalars():
            self.skipTest(
                "Flash attention (CUTE) scalar capture support is not available"
            )

        major, _ = torch.cuda.get_device_capability()
        if SM120OrLater:
            self.skipTest("block sparse mask_mod is not supported on SM120")
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+")

        flash_counter = CompileCounterWithBackend("inductor")
        triton_counter = CompileCounterWithBackend("inductor")

        flash = torch.compile(
            lambda q, k, v, block_mask: flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            ),
            dynamic=True,
            fullgraph=True,
            backend=flash_counter,
        )
        triton = torch.compile(
            lambda q, k, v, block_mask: flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "TRITON"},
            ),
            dynamic=True,
            fullgraph=True,
            backend=triton_counter,
        )

        for offset in [0, 1, 5]:
            torch.manual_seed(1234)
            q = torch.randn(
                (2, 4, 1, 64),
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            k = torch.randn(
                (2, 4, 128, 64),
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            v = torch.randn(
                (2, 4, 128, 64),
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            q_ref, k_ref, v_ref = [
                x.detach().clone().requires_grad_() for x in (q, k, v)
            ]
            block_mask = self._offset_block_mask(offset, q_len=1, kv_len=128)

            with torch.no_grad():
                out_ref = flex_attention(
                    q.float(), k.float(), v.float(), block_mask=block_mask
                ).to(q.dtype)
            out_flash = flash(q, k, v, block_mask)
            out_triton = triton(q_ref, k_ref, v_ref, block_mask)
            self.assertEqual(out_flash, out_ref, atol=3e-2, rtol=3e-2)
            self.assertEqual(out_flash, out_triton, atol=3e-2, rtol=3e-2)

            grad = torch.randn_like(out_flash)
            grads_flash = torch.autograd.grad(out_flash, (q, k, v), grad)
            grads_triton = torch.autograd.grad(out_triton, (q_ref, k_ref, v_ref), grad)
            for actual, expected in zip(grads_flash, grads_triton):
                self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)

        self.assertEqual(flash_counter.frame_count, 1, "Expected one FLASH graph")
        self.assertEqual(triton_counter.frame_count, 1, "Expected one TRITON graph")

    def _compare_flash_triton_backward(self, score_mod_factory):
        # [FUNCTION DOC] HELPER: given a factory mapping batch size to a score_mod, compiles FLASH
        # and TRITON versions with dynamic shapes, runs them on identical inputs requiring grad,
        # and asserts outputs and q/k/v gradients are close (atol and rtol 3e-2).
        def run(q, k, v, backend):
            # [FUNCTION DOC] HELPER: reads the batch size from q and runs flex_attention with the
            # score_mod from the factory for the given backend name.
            batch = q.size(0)
            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod_factory(batch),
                kernel_options={"BACKEND": backend},
            )

        flash = torch.compile(lambda q, k, v: run(q, k, v, "FLASH"), dynamic=True)
        triton = torch.compile(lambda q, k, v: run(q, k, v, "TRITON"), dynamic=True)

        q, k, v = create_test_tensors(
            seq_len=128, device="cuda", dtype=torch.float16, requires_grad=True
        )
        q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        out_flash = flash(q, k, v)
        out_triton = triton(q_ref, k_ref, v_ref)
        self.assertEqual(out_flash, out_triton, atol=3e-2, rtol=3e-2)

        grad = torch.randn_like(out_flash)
        grads_flash = torch.autograd.grad(out_flash, (q, k, v), grad)
        grads_triton = torch.autograd.grad(out_triton, (q_ref, k_ref, v_ref), grad)
        for actual, expected in zip(grads_flash, grads_triton):
            self.assertEqual(actual, expected, atol=3e-2, rtol=3e-2)

    @xfailIfSM120OrLater
    def test_dynamic_symbol_closure_in_score_mod_backward(self):
        # [FUNCTION DOC] A score_mod using the symbolic batch size (scaling score by batch plus
        # one) must work through FLASH backward; compared against TRITON by the helper. Expected
        # to fail on SM120+.
        """Captured SymInt in score_mod should propagate through FLASH backward."""
        self._compare_flash_triton_backward(
            lambda batch: lambda score, _b, _h, _q, _kv: score * (batch + 1)
        )

    @xfailIfSM120OrLater
    def test_dynamic_symfloat_closure_in_score_mod_backward(self):
        # [FUNCTION DOC] A score_mod using a symbolic float expression (batch plus one, halved)
        # must lower in FLASH backward; compared against TRITON by the helper. Expected to fail on
        # SM120+.
        """SymFloat expressions in score_mod should lower inside FLASH backward."""
        self._compare_flash_triton_backward(
            lambda batch: lambda score, _b, _h, _q, _kv: score * ((batch + 1) / 2.0)
        )


class TestHierarchicalIndex(InductorTestCase):
    # [CLASS DOC] Unit tests for the hierarchical index machinery of the CuTe flex flash module
    # (HierarchicalIndex and its indexer builder), plus end-to-end checks of multi-dimensional
    # captured tensor loads in generated code and the install message of the unavailability error.
    def test_flash_attention_unavailable_message_has_install_link(self):
        # [FUNCTION DOC] Checks the unavailability message of flash attention includes the pip
        # install command for the flash-attn-4 prerelease and the PyPI project URL.
        message = _flash_attention_unavailable_message()
        self.assertIn("pip install --pre flash-attn-4", message)
        self.assertIn("https://pypi.org/project/flash-attn-4/", message)

    def test_hierarchical_index_preserves_args(self):
        # [FUNCTION DOC] A HierarchicalIndex built from two sympy symbols keeps them as its args
        # in order.
        from sympy import Symbol

        b = Symbol("b")
        q_idx = Symbol("q_idx")
        idx = HierarchicalIndex(b, q_idx)

        self.assertIsInstance(idx, HierarchicalIndex)
        self.assertEqual(idx.args, (b, q_idx))

    def test_hierarchical_indexer_single_dim_no_wrap(self):
        # [FUNCTION DOC] For a single-dimension size the indexer returns the lone index unchanged
        # (not wrapped).
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[10])
        q_idx = Symbol("q_idx")

        self.assertEqual(indexer([q_idx]), q_idx)

    def test_hierarchical_indexer_multi_dim_wraps(self):
        # [FUNCTION DOC] For a two-dimension size the indexer returns a HierarchicalIndex with the
        # given indices as args.
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[4, 128])
        b = Symbol("b")
        q_idx = Symbol("q_idx")

        result = indexer([b, q_idx])

        self.assertIsInstance(result, HierarchicalIndex)
        self.assertEqual(result.args, (b, q_idx))

    def test_hierarchical_indexer_3d_and_4d(self):
        # [FUNCTION DOC] Indexers for 3-D and 4-D sizes return HierarchicalIndex objects whose
        # args are the supplied indices in order.
        from sympy import Symbol

        b, h, q_idx, kv_idx = (
            Symbol("b"),
            Symbol("h"),
            Symbol("q_idx"),
            Symbol("kv_idx"),
        )

        indexer_3d = _hierarchical_indexer_cute(size=[2, 4, 512])
        result_3d = indexer_3d([b, h, q_idx])
        self.assertIsInstance(result_3d, HierarchicalIndex)
        self.assertEqual(result_3d.args, (b, h, q_idx))

        indexer_4d = _hierarchical_indexer_cute(size=[2, 4, 512, 512])
        result_4d = indexer_4d([b, h, q_idx, kv_idx])
        self.assertIsInstance(result_4d, HierarchicalIndex)
        self.assertEqual(result_4d.args, (b, h, q_idx, kv_idx))

    def test_isinstance_detection_for_load(self):
        # [FUNCTION DOC] Checks type-based detection: a HierarchicalIndex instance is recognized,
        # while a plain linear sympy expression combining symbols is not.
        from sympy import Symbol

        b = Symbol("b")
        q_idx = Symbol("q_idx")

        self.assertIsInstance(HierarchicalIndex(b, q_idx), HierarchicalIndex)
        self.assertNotIsInstance(b * Symbol("S") + q_idx, HierarchicalIndex)

    def test_hierarchical_indexer_rank_mismatch(self):
        # [FUNCTION DOC] Calling a 2-D indexer with only one index must raise AssertionError whose
        # message contains the text about a rank mismatch.
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[2, 4])
        b = Symbol("b")

        with self.assertRaises(AssertionError) as ctx:
            indexer([b])
        self.assertIn("Rank mismatch", str(ctx.exception))

    # 'FlashAttentionForwardSm120' object has no attribute 'apply_score_mod'
    @xfailIfSM120OrLater
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_2d(self):
        # [FUNCTION DOC] Score_mod indexing a captured 2-D bias (batch, heads) must lower to a
        # hierarchical (multi-index) load in FLASH codegen. Runs flash_vs_triton, then checks the
        # generated code contains the expected multi-index pointer access text. Skipped without
        # CUDA or the flash library; expected to fail on SM120+.
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_2d = torch.randn(batch_size, num_heads, device=device, dtype=dtype)

        def score_mod_2d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured 2-D bias at batch and head.
            return score + bias_2d[b, h]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_2d)

        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=score_mod_2d,
            kernel_options={"BACKEND": "FLASH"},
        )
        code_str = "\n".join(code)

        expected_pattern = "in_ptr4[tmp3[0], tmp4[0]]"
        self.assertIn(
            expected_pattern,
            code_str,
            lambda msg: f"{msg}\nExpected '{expected_pattern}' in generated code.\nExcerpt:\n{code_str[:2000]}",
        )

    # 'FlashAttentionForwardSm120' object has no attribute 'apply_score_mod'
    @xfailIfSM120OrLater
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_3d(self):
        # [FUNCTION DOC] Same as the 2-D variant, for a 3-D captured bias indexed by batch, head
        # and query index; checks the generated code for the corresponding three-index load.
        # Expected to fail on SM120+.
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_3d = torch.randn(
            batch_size, num_heads, seq_len, device=device, dtype=dtype
        )

        def score_mod_3d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured 3-D bias at batch, head and q
            # index.
            return score + bias_3d[b, h, q_idx]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_3d)

        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=score_mod_3d,
            kernel_options={"BACKEND": "FLASH"},
        )
        code_str = "\n".join(code)

        expected_pattern = "in_ptr4[tmp4[0], tmp5[0], tmp6[0]]"
        self.assertIn(
            expected_pattern,
            code_str,
            lambda msg: f"{msg}\nExpected '{expected_pattern}' in generated code.\nExcerpt:\n{code_str[:2000]}",
        )

    @xfailIfSM120OrLater
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_4d(self):
        # [FUNCTION DOC] Same, for a 4-D captured bias indexed by batch, head, query and kv index.
        # After flash_vs_triton, compiles with caches disabled and score_mod vector size 8 forced
        # and checks the generated code has vec size 8, autovec copy and a local tile over the
        # input with a four-element unit tile (regex). Expected to fail on SM120+.
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_4d = torch.randn(
            batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype
        )

        def score_mod_4d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured 4-D bias at batch, head, q and
            # kv index.
            return score + bias_4d[b, h, q_idx, kv_idx]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_4d)

        with torch._inductor.config.patch(force_disable_caches=True):
            with force_flex_flash_score_mod_vec_size(8):
                torch._dynamo.reset()
                _, code = run_and_get_code(
                    torch.compile(flex_attention),
                    q,
                    k,
                    v,
                    score_mod=score_mod_4d,
                    kernel_options={"BACKEND": "FLASH"},
                )
        code_str = "\n".join(code)

        self.assertIn("score_mod.__vec_size__ = 8", code_str)
        self.assertIn("cute.autovec_copy", code_str)
        self.assertRegex(
            code_str,
            r"cute\.local_tile\(in_ptr4, \(1, 1, 1, cute\.size\([^)]*\.shape\)\), "
            r"\([^,]+\[0\], [^,]+\[0\], [^,]+\[0\], [^)]* // cute\.size\([^)]*\.shape\)\)\)",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
