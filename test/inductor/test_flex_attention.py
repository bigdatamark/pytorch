# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for flex_attention (score_mod / mask_mod / BlockMask) under torch.compile.
# Defines device/dtype configs, shared score_mod and mask_mod fixtures, comparison helpers
# (compiled vs eager vs float64 golden), and large test classes covering forward, backward, paged
# attention, block masks, captured buffers, dynamic shapes, kernel options and error paths on
# CUDA, XPU, CPU and MPS. Import time: builds device_configs, picks the test_device tuple and sets
# LONG_COMPILATION_ON_CPU from the torch config string. Test classes are instantiated per device
# at the bottom of the file.

import functools
import gc
import json
import math
import os
import random
import string
import tempfile
import types
import unittest
import warnings
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import TypeVar
from unittest import expectedFailure, mock, skip, skipUnless
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import config, metrics
from torch._inductor.exc import InductorError
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention import SDPBackend
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _apply_kernel_options,
    _compute_dq_write_order_from_block_mask,
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
    flex_attention_hop,
    noop_mask,
    or_masks,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    PLATFORM_SUPPORTS_FP8,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    E4M3_MAX_POS,
    e4m3_type,
    flex_attention_supported_platform as supported_platform,
    instantiate_device_type_tests,
    IS_FLEX_ATTENTION_CPU_PLATFORM_SUPPORTED as TEST_ON_CPU,
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED as TEST_ON_CUDA,
    IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED as TEST_ON_XPU,
    largeTensorTest,
    skipCPUIf,
    skipCUDAIf,
    skipMPSIf,
    skipXPUIf,
)
from torch.testing._internal.common_quantized import _snr
from torch.testing._internal.common_utils import (  # noqa: F401
    IS_LINUX,
    isRocmArchAnyOf,
    MI200_ARCH,
    serialTest,
    skipIfRocm,
    skipIfRocmArch,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
)
from torch.testing._internal.inductor_utils import HAS_GPU, HAS_MPS
from torch.utils._triton import has_triton, has_triton_tma_device


# Use this decorator only when hitting Triton bugs on H100
running_on_a100_only = skipUnless(
    (
        (torch.cuda.is_available() and has_triton())
        and (torch.cuda.get_device_capability() == (8, 0) or torch.version.hip)
    )
    or (torch.xpu.is_available() and has_triton()),
    "Requires Triton + A100 or Triton + ROCm or Triton + Intel GPU",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] unittest module fixture: saves the current float32 matmul precision in a
    # module global and then sets precision to high for every test in this module.
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    # [FUNCTION DOC] unittest module fixture: restores the float32 matmul precision saved by
    # setUpModule (if any) and clears the saved global.
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


index = torch.ops.aten.index
Tensor = torch.Tensor


T = TypeVar("T")
M = TypeVar("M", bound=Callable)


def large_tensor_test_class(
    size: str, device: torch.device | str | None = None
) -> Callable[[type[T]], type[T]]:
    # [FUNCTION DOC] HELPER: builds a class decorator. Given a memory size string and an optional
    # device, the returned decorator wraps every callable class attribute whose name starts with
    # test_ in the large-tensor-test wrapper for the size and device, mutating the class in place
    # and returning it.
    def decorator(cls: type[T]) -> type[T]:
        # [FUNCTION DOC] HELPER: the class decorator; applies largeTensorTest to each test_ method
        # of cls via setattr and returns cls.
        for name, method in list(cls.__dict__.items()):
            if callable(method) and name.startswith("test_"):
                setattr(cls, name, largeTensorTest(size, device)(method))
        return cls

    return decorator


@contextmanager
def temp_float32_matmul_precision(precision: str):
    # [FUNCTION DOC] HELPER: context manager that sets the float32 matmul precision to the given
    # value ('highest', 'high' or 'medium') and, on XPU, also toggles oneDNN TF32. The original
    # precision is restored in a finally block, including on XPU.
    """
    Temporarily set the float32 matmul precision and restore it after the context is exited.

    Args:
    precision (str): The precision to set ('highest', 'high', or 'medium').
    """

    def set_float32_matmul_precision_xpu(precision: str):
        # [FUNCTION DOC] HELPER: XPU only; highest disables oneDNN TF32, high enables it, any
        # other value leaves the oneDNN setting unchanged.
        if precision == "highest":
            torch._C._set_onednn_allow_tf32(False)
        if precision == "high":
            torch._C._set_onednn_allow_tf32(True)

    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(original_precision)


def skip_on_cpu(test_func):
    # [FUNCTION DOC] HELPER: decorator that unconditionally skips the test on CPU using skipCPUIf.
    # Returns the wrapped test function.
    """Decorator to skip tests that are not supported on CPU."""
    decorated_func = skipCPUIf(True, "Not supported on CPU")(test_func)
    return decorated_func


def skip_on_cuda(test_func):
    # [FUNCTION DOC] HELPER: decorator that unconditionally skips the test on CUDA using
    # skipCUDAIf. Returns the wrapped test function.
    """Decorator to skip tests that are not supported on CUDA."""
    decorated_func = skipCUDAIf(True, "Not supported on CUDA")(test_func)
    return decorated_func


def skip_on_rocm(test_func):
    # [FUNCTION DOC] HELPER: decorator that skips the test on CUDA only when running on a ROCm
    # (HIP) build with CUDA available. Returns the wrapped test function.
    # KNOWN ISSUE: its docstring says CUDA but the skip only triggers on ROCm.
    """Decorator to skip tests that are not supported on CUDA."""
    IS_ROCM = torch.cuda.is_available() and torch.version.hip
    decorated_func = skipCUDAIf(IS_ROCM, "Not supported on ROCM")(test_func)
    return decorated_func


def skip_on_xpu(test_func):
    # [FUNCTION DOC] HELPER: decorator that unconditionally skips the test on XPU. Returns the
    # wrapped test function.
    """Decorator to skip tests that are not supported on Intel GPU."""
    decorated_func = skipXPUIf(True, "Not supported on Intel GPU")(test_func)
    return decorated_func


skip_on_mps = skipMPSIf(True, "Not supported on MPS")


def _is_not_implemented_exc(exc):
    # [FUNCTION DOC] HELPER: returns True if the exception, or any exception in its chain
    # (inner_exception, __cause__, __context__), is a NotImplementedError, has a type name
    # containing NotImplemented, or whose text contains the NotImplementedError marker. Cycle-safe
    # via a seen-id set.
    """True if `exc` is or wraps a NotImplementedError (Dynamo/Inductor wrap it)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if (
            isinstance(exc, NotImplementedError)
            or "NotImplemented" in type(exc).__name__
            or "NotImplementedError:" in str(exc)
        ):
            return True
        exc = getattr(exc, "inner_exception", None) or exc.__cause__ or exc.__context__
    return False


def expected_not_implemented_on_mps(test_func):
    # [FUNCTION DOC] HELPER: decorator for tests that must fail on MPS. On non-MPS devices the
    # test runs normally. On MPS the test must raise a NotImplementedError (possibly wrapped by
    # Dynamo/Inductor); any other exception is re-raised and silent success fails the test.
    """On MPS the test must raise NotImplementedError (possibly wrapped by
    Inductor/Dynamo); other exceptions or silent success fail the test."""

    @functools.wraps(test_func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: the decorated test; the device is taken from the device keyword
        # or the first positional argument after self, then dispatches as described above.
        device = kwargs.get("device", args[0] if args else None)
        if not _is_mps_device(device):
            return test_func(self, *args, **kwargs)
        try:
            test_func(self, *args, **kwargs)
        except Exception as exc:
            if _is_not_implemented_exc(exc):
                return
            raise
        self.fail("expected NotImplementedError on MPS but test passed")

    return wrapper


def rmse(ref, res):
    # [FUNCTION DOC] HELPER: root mean squared error between two tensors, computed in float64.
    # Returns a scalar tensor.
    """
    Calculate root mean squared error
    """
    ref = ref.to(torch.float64)
    res = res.to(torch.float64)
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def create_attention(score_mod, block_mask, enable_gqa=False, kernel_options=None):
    # [FUNCTION DOC] HELPER: returns a partial of flex_attention with score_mod, block_mask,
    # enable_gqa and kernel_options bound; the caller supplies q, k, v.
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )


def flex_attention_fwd(q, k, v, score_mod, block_mask, scale):
    # [FUNCTION DOC] HELPER: calls the flex attention higher-order op directly (not the public
    # API, because the backward op expects log2-scale lse) with an empty kernel options mapping.
    # Returns the output and the logsumexp, dropping the third result.
    # Uses the HOP directly because flex_attention_backward expects lse in log2
    # scale, but the public flex_attention API converts lse to natural log.
    out, lse, _ = flex_attention_hop(
        q,
        k,
        v,
        score_mod,
        block_mask.as_tuple(),
        scale,
        {},
    )
    return out, lse


def create_block_mask_test(score_mod, query, key):
    # [FUNCTION DOC] HELPER: creates a 1x1 batch/head block mask from the given mod, sized from
    # the second-to-last dimension of query and key, on the query's device.
    block_mask = create_block_mask(
        score_mod,
        1,
        1,
        query.shape[-2],
        key.shape[-2],
        query.device,
    )
    return block_mask


@dataclass
class DeviceConfig:
    # [CLASS DOC] HELPER: dataclass holding per-device test settings: the full list of dtypes to
    # test and a shorter fast list of dtypes.
    dtypes: list[torch.dtype]
    dtypes_fast: list[torch.dtype]


device_configs = {}
# Tests are skipped when no device is supported, so CPU as default is safe
test_device = ("cpu",)
if HAS_GPU:
    if TEST_ON_CUDA:
        if TEST_ON_CPU:
            test_device = (
                "cuda",
                "cpu",
            )
        else:
            test_device = ("cuda",)
    elif TEST_ON_XPU:
        torch._C._set_onednn_allow_tf32(True)
        test_device = ("xpu",)
elif HAS_MPS:
    test_device = ("mps",)


class SubstringSet:
    # [CLASS DOC] HELPER: set of device-type names whose membership test normalizes a device
    # string containing cuda, xpu or mps to the bare type name before looking it up.
    def __init__(self, items):
        # [FUNCTION DOC] HELPER: stores the given items as a set.
        self.items = set(items)

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: membership check; maps strings containing cuda, xpu or mps to the
        # bare device type, then tests membership in the stored set.
        if "cuda" in item:
            item = "cuda"
        if "xpu" in item:
            item = "xpu"
        if "mps" in item:
            item = "mps"
        return item in self.items


DEVICE_SUPPORTS_BACKWARDS = SubstringSet(["cuda", "xpu"])

device_configs["cuda"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if PLATFORM_SUPPORTS_BF16
        else [torch.float16, torch.float32]
    ),
    dtypes_fast=[torch.float16],
)
device_configs["xpu"] = DeviceConfig(
    dtypes=([torch.float32, torch.bfloat16, torch.float16]),
    dtypes_fast=[torch.float16],
)
device_configs["cpu"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    ),
    dtypes_fast=[torch.float32],
)
device_configs["mps"] = DeviceConfig(
    dtypes=[torch.float32, torch.float16, torch.bfloat16],
    dtypes_fast=[torch.float32],
)

torch_config_string = torch.__config__.show()
LONG_COMPILATION_ON_CPU = False

if "CLANG" in torch_config_string.upper():
    # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
    # TODO: check reason of long compile time
    LONG_COMPILATION_ON_CPU = True


# --------- Useful score mod functions for testing ---------
def _causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: score_mod that keeps the score where the query index is at least the
    # key/value index and gives negative infinity elsewhere (causal masking).
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: score_mod adding the relative position (query minus key/value index)
    # to the score.
    return score + (token_q - token_kv)


def _rel_causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: score_mod combining causal masking with a relative position bias;
    # masked positions get negative infinity.
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


def _generate_alibi_bias(num_heads: int):
    # [FUNCTION DOC] HELPER: factory returning an ALiBi score_mod for the given number of heads.
    def _alibi_bias(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        token_q: Tensor,
        token_kv: Tensor,
    ) -> Tensor:
        # [FUNCTION DOC] HELPER: score_mod adding a per-head slope times (kv index minus q index);
        # the slope is two raised to the power of minus (head+1)*8/num_heads.
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias


def _inverse_causal(score, b, h, m, n):
    # [FUNCTION DOC] HELPER: score_mod keeping scores where the q index is at most the kv index,
    # else minus infinity.
    return torch.where(m <= n, score, float("-inf"))


def _times_two(score, b, h, m, n):
    # [FUNCTION DOC] HELPER: score_mod that doubles the score.
    """Joint graph needed for correctness"""
    return score * 2


def _squared(score, b, h, m, n):
    # [FUNCTION DOC] HELPER: score_mod that squares the score.
    """Joint graph needed for correctness"""
    return score * score


def _head_offset(dtype: torch.dtype, device: str):
    # [FUNCTION DOC] HELPER: factory for a captured-buffer score_mod. Allocates a random tensor of
    # H elements (the module-level head count) in the dtype on the device and returns a score_mod
    # multiplying the score by the element for the current head.
    """Captured Buffer"""
    head_offset = torch.rand(H, device=device, dtype=dtype)

    def score_mod(score, b, h, m, n):
        # [FUNCTION DOC] HELPER: score_mod scaling the score by the captured per-head offset.
        return score * head_offset[h]

    return score_mod


def _trig(score, b, h, m, n):
    # [FUNCTION DOC] HELPER: score_mod returning sin of cos of the score plus tan of the
    # batch index.
    """Joint graph needed for correctness"""
    return torch.sin(torch.cos(score)) + torch.tan(b)


def _trig2(score, b, h, m, n):
    # [FUNCTION DOC] HELPER: score_mod with a branching graph: cosine of score times sine of
    # score plus tan of the batch index.
    """Branching joint graph"""
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    z = cos_score * sin_score + torch.tan(b)
    return z


# --------- Useful mask mod functions for testing ---------
def _causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: mask_mod returning True where q index at least kv index (causal).
    return token_q >= token_kv


def _inverse_causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: mask_mod returning True where q index at most kv index.
    return token_q <= token_kv


test_score_mods = [
    _identity,
    _times_two,
    _squared,
    _causal,
    _inverse_causal,
    _rel_bias,
    _rel_causal,
    _generate_alibi_bias(8),
]

test_score_mask_mod_map = {
    _identity: noop_mask,
    _times_two: noop_mask,
    _squared: noop_mask,
    _causal: _causal_mask,
    _inverse_causal: _inverse_causal_mask,
    _rel_bias: noop_mask,
    _rel_causal: _causal_mask,
    _generate_alibi_bias(8): noop_mask,
}

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 2
H = 4
S = 256
D = 64

test_Hq_Hkv = [
    (4, 2),
    (4, 1),
]

test_Bq_Bkv = [
    (3, 1),
    (4, 1),
    (5, 1),
]

test_block_size = [
    128,
    256,
    (128, 256),
    (256, 128),
]

test_strides = [
    ((H * S * D, S * D, D, 1), 997),  # offset
    ((H * D, D, B * H * D, 1), 499),  # transposed dimensions
    ((H * S * D, D, H * D, 1), 0),  # heads/sequence transposed
    (
        (S * (D + 1), B * S * (D + 1), (D + 1), 1),
        293,
    ),  # additional buffer on one dim
    (
        (1, D, (B + 1) * (H + 1) * D, 1),
        97,
    ),  # additional buffer on multiple dim + shared dimension
]


def _is_mps_device(device) -> bool:
    # [FUNCTION DOC] HELPER: True if the device (string or torch.device) is an MPS device.
    return str(device).startswith("mps") or (
        isinstance(device, torch.device) and device.type == "mps"
    )


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
):
    # [FUNCTION DOC] HELPER: returns detached clones of query, key and value cast to dtype
    # (default: query's dtype) with requires_grad copied from the originals. For a float64 request
    # on an MPS tensor the copy goes through CPU because MPS has no float64.
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    if dtype == torch.float64 and query.device.type == "mps":
        # MPS lacks fp64, run the high-precision reference on CPU
        query_ref = (
            query.detach().cpu().clone().to(dtype).requires_grad_(query.requires_grad)
        )
        key_ref = key.detach().cpu().clone().to(dtype).requires_grad_(key.requires_grad)
        value_ref = (
            value.detach().cpu().clone().to(dtype).requires_grad_(value.requires_grad)
        )
    else:
        query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
        key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
        value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def _relocate_captures(mod, device):
    # [FUNCTION DOC] HELPER: returns a copy of a score_mod/mask_mod function whose closure cells
    # holding tensors or BlockMasks are moved to the given device (BlockMask mask_mod captures are
    # relocated recursively; the BlockMask is moved with its to method). Builds a new function
    # object rather than mutating the original; returns the input unchanged if it has no closure.
    """Copy a score_mod / mask_mod with its captured tensors moved to `device`.

    A score_mod / mask_mod can reference a tensor from the surrounding scope:

        bias = torch.randn(S, S, device="mps")
        score_mod = lambda s, b, h, q, kv: s + bias[q, kv]   # captures `bias`

    The MPS tests check the compiled output against a reference run on CPU in
    float64 (MPS has no float64). That reference calls `score_mod` with CPU
    inputs, but `bias` is still on MPS, so `cpu_score + bias[q, kv]` mixes a CPU
    and an MPS tensor and raises a device-mismatch error.

    This returns a copy of `mod` with every captured tensor moved to `device`
    (and a captured BlockMask too, recursing into its mask_mod's captures). It
    builds a new function instead of editing `mod` in place, because the
    original is still used for the on-device run. A mod that captures nothing is
    returned unchanged.
    """
    cells = getattr(mod, "__closure__", None)
    if not cells:
        return mod
    relocated = []
    for cell in cells:
        val = cell.cell_contents
        if isinstance(val, BlockMask):
            val = val.to(device)
            val.mask_mod = _relocate_captures(val.mask_mod, device)
        elif isinstance(val, torch.Tensor):
            val = val.to(device)
        relocated.append(types.CellType(val))
    return types.FunctionType(
        mod.__code__, mod.__globals__, mod.__name__, mod.__defaults__, tuple(relocated)
    )


def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    # [FUNCTION DOC] HELPER: for every batch index, reserves pages in the PagedAttention for the
    # corresponding target sequence length (first dimension of target_seq_len).
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


@large_tensor_test_class("2GB", device=test_device[0])
class TestFlexAttention(InductorTestCase):
    # [CLASS DOC] Main flex_attention test suite (decorated to require large tensor memory on
    # the first test device). Provides comparison helpers (_check_equal, _check_out, run_test,
    # paged-attention helpers) and many tests covering score_mods, block masks, GQA, captured
    # buffers, backward, dynamic shapes and error cases. Device-specific classes are created
    # by instantiate_device_type_tests.
    def setUp(self):
        # [FUNCTION DOC] HELPER: calls the parent setUp, then calls skipCPUIf with the
        # CPU-compile flag.
        # KNOWN ISSUE: skipCPUIf returns a decorator; the result is discarded, so no skip happens.
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: str | None = None,
        fudge_atol: float = 0,
    ):
        # [FUNCTION DOC] HELPER: compares compiled and reference outputs against a float64 golden
        # one. Moves all to CPU if devices differ, fails on NaN, and asserts the compiled mean
        # absolute error vs golden is close to the reference error within rtol of fudge_factor
        # (atol 1e-7), with tensor_name in the message. fudge_atol is unused.
        if (
            golden_out.device != ref_out.device
            or golden_out.device != compiled_out.device
        ):
            golden_out = golden_out.cpu()
            ref_out = ref_out.cpu()
            compiled_out = compiled_out.cpu()
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.fail("Output/Grad with NaN")
        name = tensor_name if tensor_name is not None else ""
        msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
        torch.testing.assert_close(
            compiled_error, ref_error, rtol=fudge_factor, atol=1e-7, msg=msg
        )

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        is_paged_attention: bool = False,
    ):
        # [FUNCTION DOC] HELPER: checks forward output accuracy under no_grad. fudge factor is 10
        # for float32 (20 when is_paged_attention) and 1.1 for other dtypes.
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
                if is_paged_attention:
                    # paged attention is less accurate since it may reorder
                    # the blocks from block mask
                    fudge_factor = 20.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

    def _check_out_and_grad(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: checks output and the gradients of q, k and v against golden and
        # reference using _check_equal with fudge factor 10 for float32 and 1.1 otherwise.
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query"
            )
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key"
            )
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value"
            )

    def run_test(
        self,
        score_mod: _score_mod_signature,
        dtype: torch.dtype,
        device: str,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int | None = None,
        KV_H: int | None = None,
        KV_S: int | None = None,
        V_D: int | None = None,
        block_mask: BlockMask | None = None,
    ):
        # [FUNCTION DOC] HELPER: core comparison driver. Builds random q, k, v (KV sizes default
        # to the Q sizes; float16 on CPU is promoted to float32), a noop block mask if none is
        # given, runs eager reference, float64 golden (on CPU with relocated captures for MPS) and
        # the compiled attention, then checks forward only when the device lacks backward support,
        # else also runs backward with a random gradient and checks gradients.
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if KV_B is None:
            KV_B = Q_B
        if KV_H is None:
            KV_H = Q_H
        if KV_S is None:
            KV_S = Q_S
        if V_D is None:
            V_D = Q_D

        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, Q_B, Q_H, Q_S, KV_S, device=device
            )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        if _is_mps_device(device):
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=(Q_H != KV_H),
            )
        else:
            sdpa_partial_gold = sdpa_partial

        compiled_sdpa = torch.compile(sdpa_partial)
        compiled_out = compiled_sdpa(q, k, v)
        golden_out = sdpa_partial_gold(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        if not isinstance(golden_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(golden_out)}")
        if not isinstance(ref_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(ref_out)}")
        if not isinstance(compiled_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(compiled_out)}")

        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def preprocess_paged_attention(
        self,
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype,
        device: str,
        page_size: int = 128,
    ) -> tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:
        # [FUNCTION DOC] HELPER: prepares paged-attention inputs. Allocates empty k/v caches,
        # builds a PagedAttention with a larger batch capacity, reserves pages for several length
        # patterns to scatter pages, assigns k and v into the cache, and returns the caches, the
        # converted block mask and the converted score_mod. Raises if block_mask is None.
        if block_mask is None:
            raise AssertionError("Must provide block_mask")
        Q_B, Q_H, Q_S, _ = q.shape
        KV_B, KV_H, KV_S, QK_D = k.shape
        _, _, _, V_D = v.shape

        # test with different batch size
        max_batch_size = max(Q_B, KV_B) + 3

        n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

        # allocate cache
        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            QK_D,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device=device,
            dtype=dtype,
        )

        # For testing purposes, we randomly initialize the page table, which maps
        # (batch_idx, logical_block_idx) to physical_block_idx. Specifically, PagedAttention
        # maintains a stack empty_pages of unused physical_block_idx. The `batch_reserve`
        # function grabs physical_block_idx from the top of empty_pages until there are enough
        # pages for each batch index (i.e., num pages for batch_idx >= target_seq_len[batch_idx]).
        # For example, at the first batch_reserve call, physical block indices (1,...,KV_S//4)
        # are allocated to batch index 0, and physical block indices
        # (KV_S//4+1, ..., KV_S//4 + KV_S//2) are allocated to batch index 1, etc.
        # Thus, kv tensors of batch index 1 will be scattered in the kv cache, simulating
        # a real use case of paged attention.
        paged_attention = PagedAttention(
            n_pages, page_size, max_batch_size, device=device
        )

        def make_reserve_tensor(fractions):
            # [FUNCTION DOC] HELPER: returns an int64 tensor of per-batch target lengths cycling
            # through KV_S times each fraction.
            pattern = [KV_S * f for f in fractions]
            return torch.tensor(
                [pattern[i % len(pattern)] for i in range(KV_B)],
                device=device,
                dtype=torch.int64,
            )

        batch_reserve(
            paged_attention, make_reserve_tensor([1 / 4, 1 / 2, 1 / 4, 1 / 3])
        )
        batch_reserve(
            paged_attention, make_reserve_tensor([1 / 4, 1 / 2, 1 / 2, 1 / 2])
        )
        batch_reserve(paged_attention, make_reserve_tensor([1 / 2, 1, 1 / 2, 1]))
        batch_reserve(paged_attention, make_reserve_tensor([1, 1, 1, 1]))

        # update cache with k and v
        input_pos = (
            torch.arange(KV_S, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(KV_B, KV_S)
        )
        batch_idx = torch.arange(KV_B, device=device, dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        # convert block mask and score mod
        kv_len_tensor = torch.full((KV_B,), KV_S, device=device, dtype=torch.int64)
        converted_block_mask = paged_attention.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )
        converted_score_mod = paged_attention.get_score_mod(
            score_mod, kv_len=kv_len_tensor
        )
        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype,
        device: str,
        block_mask: BlockMask | None = None,
        kernel_options: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] HELPER: runs compiled flex_attention on paged k/v caches (building a noop
        # block mask if none is passed). Requests logsumexp when the device supports backward.
        # Returns the output and lse (None when lse not requested).
        B, Q_H, Q_S, KV_H, KV_S = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[1],
            k.shape[2],
        )

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, B, 1, Q_S, KV_S, device=device)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, device, block_mask.BLOCK_SIZE[1]
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        return_lse = True
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        else:
            return_lse = False
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Callable | None,
        dtype: torch.dtype,
        device,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
        block_mask: BlockMask | None = None,
    ):
        # [FUNCTION DOC] HELPER: paged-attention comparison driver. Requires the query head count
        # to be a multiple of the kv head count (raises AssertionError otherwise), builds random
        # q, k, v without grad (float16 on CPU promoted to float32), runs the paged compiled path
        # via run_paged_attention plus eager reference and float64 golden (CPU copies on MPS), and
        # checks the output, and the logsumexp when the device supports backward.
        if Q_H % KV_H != 0:
            raise AssertionError(
                f"Expected Q_H % KV_H == 0, got {Q_H} % {KV_H} = {Q_H % KV_H}"
            )
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D), dtype=dtype, device=device, requires_grad=False
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S, device=device)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, device, block_mask
        )
        if _is_mps_device(device):
            # MPS has no float64; the golden runs on CPU, so its score_mod
            # captures and block_mask must be relocated off-device too.
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=(Q_H != KV_H),
            )
        else:
            sdpa_partial_gold = sdpa_partial
        golden_out, golden_lse = sdpa_partial_gold(
            q_gold, k_gold, v_gold, return_lse=True
        )
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
            is_paged_attention=True,
        )
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            self._check_out(
                golden_lse,
                ref_lse,
                compiled_lse,
                is_paged_attention=True,
            )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        dtype: torch.dtype,
        device: str,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
    ):
        # [FUNCTION DOC] HELPER: like run_test but takes a ready-made attention callable. Builds
        # random q, k, v, runs compiled, eager and float64 golden versions (on MPS the golden
        # callable gets its captured tensors, block_mask and score_mod relocated to CPU), then
        # checks the forward output, plus gradients when the device supports backward.
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        compiled_out = compiled_sdpa(q, k, v)
        # On MPS the gold path runs on CPU at fp64; relocate any device-resident
        # captures (block_mask, score_mod buffers, or a closed-over BlockMask) to
        # CPU so the reference stays single-device.
        sdpa_call_gold = sdpa_call
        if _is_mps_device(device):
            if isinstance(sdpa_call, functools.partial):
                cpu_kwargs = dict(sdpa_call.keywords)
                if cpu_kwargs.get("block_mask") is not None:
                    bm = cpu_kwargs["block_mask"].to("cpu")
                    bm.mask_mod = _relocate_captures(bm.mask_mod, "cpu")
                    cpu_kwargs["block_mask"] = bm
                if cpu_kwargs.get("score_mod") is not None:
                    cpu_kwargs["score_mod"] = _relocate_captures(
                        cpu_kwargs["score_mod"], "cpu"
                    )
                sdpa_call_gold = functools.partial(
                    sdpa_call.func, *sdpa_call.args, **cpu_kwargs
                )
            else:
                sdpa_call_gold = _relocate_captures(sdpa_call, "cpu")
        golden_out = sdpa_call_gold(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def run_dynamic_test(
        self,
        score_mask_mod: tuple[Callable, Callable],
        dtype: torch.dtype,
        device,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        # [FUNCTION DOC] HELPER: dynamic-shape check. Computes eager and golden results for three
        # input shapes (original, then batch doubled with sequence halved, then sequence halved
        # again), then compiles each with the dynamic option on under one compile counter and
        # checks accuracy (and gradients on backward-capable devices) and that exactly one frame was
        # compiled in total after each call.
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        score_mod, mask_mod = score_mask_mod

        # First batch with original dimensions (B, H, S, D)
        block_mask1 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        # MPS gold runs on CPU at fp64, rebuild partial with CPU block_mask.
        sdpa_partial1_gold = (
            create_attention(score_mod, block_mask=block_mask1.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial1
        )
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1_gold(q1_gold, k1_gold, v1_gold)
        if requires_grad:
            backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out1.backward(backward_grad1.to(torch.float64))
            ref_out1.backward(backward_grad1)

        # Second batch with modified dimensions (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)

        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        sdpa_partial2_gold = (
            create_attention(score_mod, block_mask=block_mask2.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial2
        )
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2_gold(q2_gold, k2_gold, v2_gold)

        if requires_grad:
            backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out2.backward(backward_grad2.to(torch.float64))
            ref_out2.backward(backward_grad2)

        # Third batch with modified dimensions (B * 2, H, S / 4, D)
        S = int(S / 2)
        block_mask3 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)

        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q3_ref, k3_ref, v3_ref = query_key_value_clones(q3, k3, v3)
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        sdpa_partial3_gold = (
            create_attention(score_mod, block_mask=block_mask3.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial3
        )
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3_gold(q3_gold, k3_gold, v3_gold)

        if requires_grad:
            backward_grad3 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out3.backward(backward_grad3.to(torch.float64))
            ref_out3.backward(backward_grad3)

        # Clear dynamo counters
        torch._dynamo.reset()

        # First compilation with original dimensions
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_sdpa1 = torch.compile(sdpa_partial1, backend=backend, dynamic=True)
        compiled_out1 = compiled_sdpa1(q1, k1, v1)

        if requires_grad:
            compiled_out1.backward(backward_grad1)

            self._check_out_and_grad(
                golden_out1,
                ref_out1,
                compiled_out1,
                q1_gold,
                q1_ref,
                q1,
                k1_gold,
                k1_ref,
                k1,
                v1_gold,
                v1_ref,
                v1,
            )
        else:
            self._check_out(golden_out1, ref_out1, compiled_out1)
        self.assertEqual(backend.frame_count, 1)

        # Second compilation with new dimensions
        compiled_sdpa2 = torch.compile(sdpa_partial2, backend=backend, dynamic=True)
        compiled_out2 = compiled_sdpa2(q2, k2, v2)

        if requires_grad:
            compiled_out2.backward(backward_grad2)

            self._check_out_and_grad(
                golden_out2,
                ref_out2,
                compiled_out2,
                q2_gold,
                q2_ref,
                q2,
                k2_gold,
                k2_ref,
                k2,
                v2_gold,
                v2_ref,
                v2,
            )
        else:
            self._check_out(golden_out2, ref_out2, compiled_out2)
        self.assertEqual(backend.frame_count, 1)

        # Third compilation with new dimensions
        compiled_sdpa3 = torch.compile(sdpa_partial3, backend=backend, dynamic=True)
        compiled_out3 = compiled_sdpa3(q3, k3, v3)

        if requires_grad:
            compiled_out3.backward(backward_grad3)

            self._check_out_and_grad(
                golden_out3,
                ref_out3,
                compiled_out3,
                q3_gold,
                q3_ref,
                q3,
                k3_gold,
                k3_ref,
                k3,
                v3_gold,
                v3_ref,
                v3,
            )
        else:
            self._check_out(golden_out3, ref_out3, compiled_out3)
        self.assertEqual(backend.frame_count, 1)

    def run_automatic_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype,
        device: str,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        # [FUNCTION DOC] HELPER: automatic-dynamic check. Prepares three input shapes like the
        # dynamic test, resets dynamo, then compiles each with fullgraph and asserts accuracy; the
        # frame count must be 1 after the first call, 2 after the second (automatic dynamic
        # recompile) and still 2 after the third (no recompilation).
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        block_mask1 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)
        # MPS lacks fp64; gold path runs on CPU at fp64 with a CPU block_mask.
        sdpa_partial1_gold = (
            create_attention(score_mod, block_mask=block_mask1.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial1
        )
        # The first eager batch, shape (B, H, S, D)
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        golden_out1 = sdpa_partial1_gold(q1_gold, k1_gold, v1_gold)
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        sdpa_partial2_gold = (
            create_attention(score_mod, block_mask=block_mask2.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial2
        )
        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        golden_out2 = sdpa_partial2_gold(q2_gold, k2_gold, v2_gold)
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        sdpa_partial3_gold = (
            create_attention(score_mod, block_mask=block_mask3.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial3
        )
        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        golden_out3 = sdpa_partial3_gold(q3_gold, k3_gold, v3_gold)
        ref_out3 = sdpa_partial3(q3, k3, v3)

        # Need to clear dynamo counters, since flex attention eager mode also uses dynamo tracing.
        # We check dynamo counters["frames"]["ok"] to ensure:
        # 1, the first batch is compiled with static shape
        # 2, the second batch is compiled with dynamic shape
        # 3, no re-compilation in the third batch
        torch._dynamo.reset()

        # Note, it seems like we really are less accurate than the float32
        # computation, likely due to the online softmax
        if dtype == torch.float32:
            fudge_factor = 10.0
        else:
            fudge_factor = 1.1

        # The first batch.
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_out1 = torch.compile(sdpa_partial1, backend=backend, fullgraph=True)(
            q1, k1, v1
        )
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(backend.frame_count, 1)

        # The second batch (automatic dynamic).
        compiled_out2 = torch.compile(sdpa_partial2, backend=backend, fullgraph=True)(
            q2, k2, v2
        )
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

        # The third batch (no re-compilation).
        compiled_out3 = torch.compile(sdpa_partial3, backend=backend, fullgraph=True)(
            q3, k3, v3
        )
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods(self, device, dtype, score_mod: Callable):
        # [FUNCTION DOC] Runs each built-in score_mod through run_test and the paged attention
        # path for the parametrized dtypes, with a noop block mask.
        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @running_on_a100_only
    @common_utils.parametrize("score_mod", test_score_mods)
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_builtin_score_mods_seqlen_lt_default_sparse_block_size(
        self, device, dtype, score_mod: Callable
    ):
        # [FUNCTION DOC] On A100-class devices, runs flex_attention with each built-in score_mod
        # and the flex-attention path forced, using sequence length 64 which is below the default
        # sparse block size, and compares with reference via run_test_with_call.
        # _DEFAULT_SPARSE_BLOCK_SIZE is 128
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(attention, dtype, device, B, H, 64, D, B, H, 64, D)

    @running_on_a100_only
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_seqlen_lt_custom_sparse_block_size(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        # [FUNCTION DOC] Same as the previous test but with an explicit causal block mask built
        # with BLOCK_SIZE 256 for sequence length 64 (smaller than the custom block size).
        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod; True where the q index is at least the
            # kv index.
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, 1, 1, 64, 64, BLOCK_SIZE=256, device=device
        )
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(
            attention,
            dtype,
            device,
            B,
            H,
            64,
            D,
            B,
            H,
            64,
            D,
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mask_mod", test_score_mask_mod_map.items())
    def test_builtin_score_mods_dynamic(
        self, device, dtype: torch.dtype, score_mask_mod: tuple[Callable, Callable]
    ):
        # [FUNCTION DOC] Dynamic-shape test: runs each (score_mod, matching mask_mod) pair through
        # run_dynamic_test with sequence length 1024.
        self.run_dynamic_test(score_mask_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_automatic_dynamic(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        # [FUNCTION DOC] Automatic-dynamic test: runs each built-in score_mod through
        # run_automatic_dynamic_test with sequence length 1024.
        self.run_automatic_dynamic_test(score_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_different_seqlen(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        # [FUNCTION DOC] Runs each built-in score_mod with a query sequence length half the
        # key/value length through the regular and paged paths.
        inputs = (
            score_mod,
            dtype,
            device,
            B,
            H,
            S // 2,  # Seqlen of Q is different from seqlen of K/V
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("BLOCK_SIZE", test_block_size)
    def test_builtin_score_mods_different_block_size(
        self,
        device,
        dtype: torch.dtype,
        score_mod: Callable,
        BLOCK_SIZE: int | tuple[int, int],
    ):
        # [FUNCTION DOC] Runs each built-in score_mod with a noop block mask built for each
        # parametrized BLOCK_SIZE (int or pair) through the regular and paged paths.
        block_mask = create_block_mask(
            noop_mask, B, H, S, S, BLOCK_SIZE=BLOCK_SIZE, device=device
        )
        self.run_test(score_mod, dtype, block_mask=block_mask, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype, block_mask=block_mask, device=device
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast(
        self,
        device,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        # [FUNCTION DOC] Checks key/value batch broadcasting: query batch greater than 1 with a
        # key and value batch of 1, for GQA head configurations. Asserts the head and batch
        # preconditions, then compares via run_test with a noop block mask.
        Hq, Hkv = head_dims
        if Hq % Hkv != 0:
            raise AssertionError(
                f"Expected Hq % Hkv == 0, got {Hq} % {Hkv} = {Hq % Hkv}"
            )

        Bq, Bkv = batch_dims
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Expected Bq > 1 and Bkv == 1, got Bq={Bq}, Bkv={Bkv}"
            )

        block_mask = create_block_mask(noop_mask, Bq, 1, S, S, device=device)

        self.run_test(
            score_mod, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D, block_mask
        )

    @supported_platform
    @skip_on_cpu
    def test_small_block_mask(self, device):
        # [FUNCTION DOC] Smoke test (non-CPU): compiles block mask creation and builds masks from
        # two small sequence-id tensors (lengths 9 and 4) whose mask_mod matches positions with
        # equal, non -1 ids; only checks that it does not raise.
        compiled_create_block_mask = torch.compile(create_block_mask)

        def create_block_mask_from_seqlens(
            q_batch: torch.Tensor,
            kv_batch: torch.Tensor,
        ) -> BlockMask:
            # [FUNCTION DOC] HELPER: builds a block mask via the compiled creator from two id
            # tensors, with Q_LEN and KV_LEN taken from their lengths and no batch/head dims.
            B, H = None, None
            Q_LEN = q_batch.size(0)
            KV_LEN = kv_batch.size(0)

            def batch_mask_mod(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: mask_mod true when q and kv positions carry equal ids and
                # neither is -1.
                q_idx_batch = q_batch[q_idx]
                kv_idx_batch = kv_batch[kv_idx]
                batch_mask = (
                    (q_idx_batch == kv_idx_batch)
                    & (q_idx_batch != -1)
                    & (kv_idx_batch != -1)
                )

                return batch_mask

            return compiled_create_block_mask(
                batch_mask_mod,
                B=B,
                H=H,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
            )

        a = torch.tensor([2, 42, 18, 21, 4, 2, 7, 1, 1], device=device)
        b = torch.tensor([57, 21, 16, 8], device=device)

        for seqlen in [a, b]:
            create_block_mask_from_seqlens(seqlen, seqlen)

    @supported_platform
    @skip_on_cpu
    def test_create_block_mask_recompile_persistent_reduction_oob(self, device):
        # [FUNCTION DOC] Regression test (non-CPU): compiles create_block_mask with a large Q_LEN,
        # then recompiles with Q_LEN of 2 and block size 16, and asserts the compiled block mask
        # counts match the eager ones, guarding against an out-of-bounds read in the persistent
        # reduction block size.
        """Recompiling create_block_mask with a much smaller Q_LEN must not OOB.

        When Q_LEN shrinks enough that (Q_LEN+15)//16 == 1 for all guarded
        values, _get_persistent_RBLOCK must return R0_BLOCK=1 (not 2) so
        the constant all-True mask does not read out of bounds.
        """
        compiled_create_block_mask = torch.compile(create_block_mask)

        def causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod (q index at least kv index).
            return q_idx >= kv_idx

        KV_LEN = 2048

        compiled_create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=7896,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )

        compiled_bm = compiled_create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=2,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )
        eager_bm = create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=2,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )

        torch.testing.assert_close(
            compiled_bm.full_q_num_blocks,
            eager_bm.full_q_num_blocks,
        )
        torch.testing.assert_close(
            compiled_bm.q_num_blocks,
            eager_bm.q_num_blocks,
        )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # hardcodes kernel_options={"BACKEND": "TRITON"}
    def test_kv_order_invariance_padded_causal(self, device):
        # [FUNCTION DOC] Regression test (non-CPU, not MPS, bf16 required on CUDA): with a padded
        # causal mask, checks block mask indices with and without separated full blocks, then
        # asserts that compiled TRITON attention output over the shared prefix is bit-identical
        # whether the mask used the partial-block or full-block layout.
        if device == "cuda" and not PLATFORM_SUPPORTS_BF16:
            self.skipTest("bf16 is required for this regression test")

        batch_size = 1
        q_heads = 16
        kv_heads = 4
        seq_len = 256
        head_dim = 128
        shared_prefix = 201
        dtype = torch.bfloat16

        def make_block_mask(real_q_len: int, separate_full_blocks: bool):
            # [FUNCTION DOC] HELPER: builds a block mask for a padded causal pattern with the
            # given real query length and chosen full-block separation.
            def padded_causal_mask(b, h, q, kv):
                # [FUNCTION DOC] HELPER: mask_mod true for q below the real length and q at
                # least kv.
                return (q < real_q_len) & (q >= kv)

            return create_block_mask(
                padded_causal_mask,
                B=batch_size,
                H=1,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                separate_full_blocks=separate_full_blocks,
            )

        q = torch.randn(
            batch_size, q_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype
        )

        compiled_attention = torch.compile(
            flex_attention,
            fullgraph=True,
        )
        default_partial_block_mask = make_block_mask(shared_prefix, True)
        default_full_block_mask = make_block_mask(seq_len, True)
        self.assertEqual(default_partial_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertEqual(default_partial_block_mask.full_kv_num_blocks[0, 0, 1], 0)
        self.assertEqual(default_full_block_mask.kv_indices[0, 0, 1, :1], [1])
        self.assertEqual(default_full_block_mask.full_kv_indices[0, 0, 1, :1], [0])

        partial_block_mask = make_block_mask(shared_prefix, False)
        full_block_mask = make_block_mask(seq_len, False)
        self.assertEqual(partial_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertIsNone(partial_block_mask.full_kv_num_blocks)
        self.assertEqual(full_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertIsNone(full_block_mask.full_kv_num_blocks)

        out_partial = compiled_attention(
            q,
            k,
            v,
            score_mod=_identity,
            block_mask=partial_block_mask,
            enable_gqa=True,
            kernel_options={"BACKEND": "TRITON"},
        )
        out_full = compiled_attention(
            q,
            k,
            v,
            score_mod=_identity,
            block_mask=full_block_mask,
            enable_gqa=True,
            kernel_options={"BACKEND": "TRITON"},
        )
        self.assertTrue(
            torch.equal(
                out_partial[:, :, :shared_prefix], out_full[:, :, :shared_prefix]
            )
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast_causal_mask(
        self,
        device,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        # [FUNCTION DOC] Same as the batch broadcast test but with a causal block mask and
        # attention via run_test_with_call (GQA enabled when head counts differ).
        Hq, Hkv = head_dims
        if Hq % Hkv != 0:
            raise AssertionError(
                f"Expected Hq % Hkv == 0, got {Hq} % {Hkv} = {Hq % Hkv}"
            )

        Bq, Bkv = batch_dims
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Expected Bq > 1 and Bkv == 1, got Bq={Bq}, Bkv={Bkv}"
            )

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod (q index at least kv index).
            return q >= kv

        block_mask = create_block_mask(mask_mod, Bq, 1, S, S, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=(Hq != Hkv)
        )

        self.run_test_with_call(attention, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_GQA(self, device, dtype: torch.dtype, score_mod: Callable):
        # [FUNCTION DOC] Grouped-query attention test: query has four times as many heads as
        # key/value and a shorter sequence (one eighth); runs the regular and paged paths for
        # each score_mod.
        inputs = (
            score_mod,
            dtype,
            device,
            B,
            H * 4,  # Hq = 4*Hkv.
            S // 8,
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "q_s", test_strides[:2]
    )  # TODO: fix layout for query braodcasting
    @common_utils.parametrize(
        "k_s,v_s",
        [
            (test_strides[0], test_strides[0]),
            (test_strides[0], test_strides[1]),
            (test_strides[2], test_strides[3]),
            (test_strides[3], test_strides[1]),
            # (test_strides[2], test_strides[4]), # TODO: Doesn't work for
            # broadcasting reasons i think
        ],
    )
    @common_utils.parametrize("do_s", test_strides[:3])
    def test_strided_inputs(self, device, dtype: torch.dtype, q_s, k_s, v_s, do_s):
        # [FUNCTION DOC] Compiled attention with non-contiguous (as_strided) q, k, v and output
        # grad tensors laid out by the parametrized stride/offset pairs, with TMA kernel option
        # set and an alibi score_mod. Compares compiled to eager within 0.2 tolerances, including
        # gradients when supported, then runs the paged path without grad and compares again.
        q1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        k1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        v1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        do1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)

        q_shape = (B, H, S // 2, D)
        k_shape = (B, H, S, D)
        v_shape = (B, H, S, D)
        do_shape = (B, H, S // 2, D)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        def coerce_to_strides(val, shape, strides):
            # [FUNCTION DOC] HELPER: views a flat buffer with the given shape, strides and storage
            # offset, after checking (AssertionError) that the extent fits in the buffer and the
            # last stride is 1; sets requires_grad as the device allows.
            strides, offset = strides
            val_max = [x * (y - 1) for x, y in zip(strides, shape)]
            if sum(val_max) + offset >= B * H * S * D * 2:
                raise AssertionError(
                    f"Expected sum(val_max) + offset < B * H * S * D * 2, "
                    f"got {sum(val_max) + offset} >= {B * H * S * D * 2}"
                )
            if strides[-1] != 1:
                raise AssertionError(f"Expected strides[-1] == 1, got {strides[-1]}")
            return torch.as_strided(val, shape, strides, offset).requires_grad_(
                requires_grad
            )

        q = coerce_to_strides(q1, q_shape, q_s)
        k = coerce_to_strides(k1, k_shape, k_s)
        v = coerce_to_strides(v1, v_shape, v_s)
        do = coerce_to_strides(do1, do_shape, do_s)

        kernel_options = {"USE_TMA": True}

        block_mask = _create_empty_block_mask(q, k)
        score_mod = _generate_alibi_bias(8)
        sdpa_partial = create_attention(
            score_mod=score_mod, block_mask=block_mask, kernel_options=kernel_options
        )
        compiled_sdpa = torch.compile(sdpa_partial, fullgraph=True)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )
        if requires_grad:
            ref_out.backward(do)
            ref_grads = [q.grad, k.grad, v.grad]
            q.grad = None
            k.grad = None
            v.grad = None

            compiled_out.backward(do)
            compiled_grads = [q.grad, k.grad, v.grad]
            q.grad = None
            k.grad = None
            v.grad = None
            torch.testing.assert_close(
                compiled_grads[0],
                ref_grads[0],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[1],
                ref_grads[1],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[2],
                ref_grads[2],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )

        # test paged attention which does not support backward
        q.requires_grad, k.requires_grad, v.requires_grad = False, False, False
        paged_compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, device=device, kernel_options=kernel_options
        )
        torch.testing.assert_close(
            ref_out, paged_compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_doc_mask_sparse(self, device):
        # [FUNCTION DOC] Document masking expressed as a score_mod: ids of 256-token documents,
        # causal and same-document positions kept. Runs regular and paged paths in float16.
        document_id = torch.zeros(S, dtype=torch.int, device=device)
        for i in range(0, S, 256):
            document_id[i : i + 256] = i // 256

        def document_masking_causal(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod keeping scores only where causal and in the same
            # document; others become negative infinity.
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            return torch.where(causal_mask & document_mask, score, -float("inf"))

        self.run_test(document_masking_causal, torch.float16, device=device)
        self.run_test_with_paged_attention(
            document_masking_causal, torch.float16, device=device
        )

    @supported_platform
    def test_index_multiple(self, device):
        # [FUNCTION DOC] Captured two-dimensional bias indexed by batch then query position, added
        # to scores; run through regular and paged paths in float16.
        bias = torch.randn(B, S, device=device)

        def index_multiple(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured bias row for the batch at the
            # q index.
            return score + bias[b][q_idx]

        self.run_test(index_multiple, torch.float16, device=device)
        self.run_test_with_paged_attention(index_multiple, torch.float16, device=device)

    @supported_platform
    def test_index_weird1(self, device):
        # [FUNCTION DOC] Captured four-dimensional bias indexed with a constant first index, then
        # with batch and head tensors, then the q index; float16 regular and paged paths.
        bias = torch.randn(4, B, H, S, device=device)

        def index_weird1(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the captured bias selected by the unusual
            # index chain.
            return score + bias[0][b, h][q_idx]

        self.run_test(index_weird1, torch.float16, device=device)
        self.run_test_with_paged_attention(index_weird1, torch.float16, device=device)

    @supported_platform
    def test_index_weird2(self, device):
        # [FUNCTION DOC] Captured bias indexed by batch and head, then by a captured 0-d index
        # tensor and the q index (mixed tensor and scalar-tensor indices); float16 regular and
        # paged paths.
        bias = torch.randn(B, H, 4, S, device=device)
        which_bias = torch.tensor(0, device=device)

        def index_weird2(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod adding the bias gathered via the captured
            # index tensor.
            return score + bias[b][h][which_bias, q_idx]

        self.run_test(index_weird2, torch.float16, device=device)
        self.run_test_with_paged_attention(index_weird2, torch.float16, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_skip_odd_keys(self, device, dtype: torch.dtype):
        # [FUNCTION DOC] score_mod that masks odd key positions to negative infinity; regular and
        # paged paths.
        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: keeps scores where the kv index is even, else
            # negative infinity.
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_function_composition(self, device, dtype: torch.dtype):
        # [FUNCTION DOC] Composes two Python score_mod functions (relative bias, then causal
        # masking) into a third and checks it through the regular and paged paths.
        def score_mod_1(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: adds the relative position (m minus n) to the score.
            return score + (m - n)

        def score_mod_2(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: keeps scores where m at most n, else negative infinity.
            return torch.where(m <= n, score, float("-inf"))

        def composed_score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: applies the first score_mod and feeds its result to
            # the second.
            return score_mod_2(score_mod_1(score, b, h, m, n), b, h, m, n)

        self.run_test(composed_score_mod, dtype, device=device)
        self.run_test_with_paged_attention(composed_score_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_captured_buffers_all_dims(self, device, dtype: torch.dtype):
        # [FUNCTION DOC] Captures three buffers (per-head, per-batch, per-token scale) in one
        # score_mod and checks it through regular and paged paths.
        head_scale = torch.randn(H, device=device)
        batch_scale = torch.randn(B, device=device)
        tok_scale = torch.randn(S, device=device)

        def all_bias(score, batch, head, token_q, token_kv):
            # [FUNCTION DOC] HELPER: adds the token-q, batch and head scale values to the score.
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(all_bias, dtype, device=device)
        self.run_test_with_paged_attention(all_bias, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_bf16_score_mod_captured_grad_dtype(self, device):
        # [FUNCTION DOC] Non-CPU test (bf16 platform required, MPS must raise
        # NotImplementedError): the score_mod captures a bias computed by a bf16 linear layer.
        # Compares the weight gradient of eager and compiled runs with the same seed, checking
        # dtype matches the weight dtype and values agree within 0.1.
        """
        Tests with tensors that require gradients with bf16 dtype in the score_mod
        """
        if not PLATFORM_SUPPORTS_BF16:
            self.skipTest("Platform does not support bf16")

        B_local, H_local, S_local, D_local = 2, 4, 32, 64
        dtype = torch.bfloat16
        x = torch.randn(
            (B_local, S_local, D_local),
            device=device,
            dtype=dtype,
        )

        def forward(bias_proj, x):
            # [FUNCTION DOC] HELPER: builds q, k, v as head-split views of x, computes a
            # per-position bias by projecting x, and returns flex_attention with a score_mod
            # adding that bias.
            q = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            k = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            v = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            attn_bias = (
                bias_proj(x)
                .view(B_local, S_local, H_local, S_local)
                .permute(0, 2, 1, 3)
            )

            def bias_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the captured attention bias at batch,
                # head, q, kv.
                return score + attn_bias[b, h, q_idx, kv_idx]

            return flex_attention(q, k, v, score_mod=bias_mod)

        def run_and_get_weight_grad(bias_proj, x, compiled):
            # [FUNCTION DOC] HELPER: clears grads, runs the forward (compiled if requested) and a
            # random backward, and returns the projection weight gradient.
            bias_proj.zero_grad(set_to_none=True)
            fn = torch.compile(forward) if compiled else forward
            out = fn(bias_proj, x)
            grad = torch.randn_like(out)
            out.backward(grad)
            return bias_proj.weight.grad

        bias_proj_eager = nn.Linear(
            D_local, H_local * S_local, device=device, dtype=dtype, bias=False
        )
        bias_proj_compiled = nn.Linear(
            D_local, H_local * S_local, device=device, dtype=dtype, bias=False
        )
        bias_proj_compiled.load_state_dict(bias_proj_eager.state_dict())

        torch.manual_seed(0)
        eager_weight_grad = run_and_get_weight_grad(bias_proj_eager, x, compiled=False)
        self.assertEqual(eager_weight_grad.dtype, bias_proj_eager.weight.dtype)

        torch.manual_seed(0)
        compiled_weight_grad = run_and_get_weight_grad(
            bias_proj_compiled, x, compiled=True
        )
        self.assertEqual(compiled_weight_grad.dtype, bias_proj_compiled.weight.dtype)
        self.assertEqual(eager_weight_grad, compiled_weight_grad, atol=1e-1, rtol=1e-1)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_seq_masking(self, device, dtype):
        # [FUNCTION DOC] Score_mod masking cross-segment attention using a boolean per-position
        # segment id (first half versus second half); regular and paged paths.
        seq_idx = torch.zeros(S, device=device, dtype=torch.bool)
        seq_idx[S // 2 :] = 1

        def seq_mask_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: keeps scores where q and kv share the same segment id.
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        self.run_test(seq_mask_mod, dtype, device=device)
        self.run_test_with_paged_attention(seq_mask_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_seq_only(self, device, dtype):
        # [FUNCTION DOC] Adds a captured sequence-by-sequence bias indexed by q and kv; regular
        # and paged paths.
        bias = torch.randn(S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured bias at q and kv.
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_seq_batch(self, device, dtype):
        # [FUNCTION DOC] Adds a captured batch-by-sequence-by-sequence bias; regular and
        # paged paths.
        bias = torch.randn(B, S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured bias at batch, q and kv.
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_load_from_view_buffer(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS must raise NotImplementedError): compiles with
        # max-autotune and fullgraph a module whose score_mod captures a view of a tensor computed
        # under no_grad from the query, in float16, and runs forward and backward; only checks
        # that it completes.
        dtype = torch.float16
        W = 8

        class SimpleAttention(torch.nn.Module):
            # [CLASS DOC] HELPER: module owning a relative position table; its forward builds a
            # score_mod whose captured buffer is a reshaped result computed from the query.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the random relative position table on the
                # test device.
                super().__init__()
                self.rel_pos_h = torch.randn(2 * H - 1, D, device=device, dtype=dtype)

            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: reshapes q, builds the score_mod from it and
                # runs flex_attention.
                q = q.view(B * H, H * W, -1)
                score_mod = self.generate_score_mod(q)
                q = q.view(B, H, H * W, -1)
                return flex_attention(q, k, v, score_mod=score_mod)

            def generate_score_mod(self, q):
                # [FUNCTION DOC] HELPER: computes decomposed relative position values, reshapes
                # them into a batch/head/query/key-row tensor and returns a score_mod adding the
                # entry for the key row index.
                rel_h = self.add_decomposed_rel_pos(q)
                rel_h = rel_h.view(
                    B, H, rel_h.size(1), rel_h.size(2), rel_h.size(3)
                ).squeeze(-1)

                def score_mod(score, batch, head, q_idx, k_idx):
                    # [FUNCTION DOC] HELPER: adds the captured value indexed by batch, head, q and
                    # key row.
                    h_idx = k_idx // W
                    return score + rel_h[batch, head, q_idx, h_idx]

                return score_mod

            @torch.no_grad()
            def add_decomposed_rel_pos(self, q):
                # [FUNCTION DOC] HELPER: under no_grad, gathers relative position table rows by
                # coordinate differences and contracts them with the reshaped query (einsum);
                # returns a 4-D tensor.
                q_coords = torch.arange(H, device=self.rel_pos_h.device)[:, None]
                k_coords = torch.arange(H, device=self.rel_pos_h.device)[None, :]
                relative_coords = (q_coords - k_coords) + (H - 1)
                Rh = self.rel_pos_h[relative_coords.long()]
                r_q = q.reshape(B * H, H, W, D)
                rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
                return rel_h.reshape(B * H, H * W, H, 1)

        m = SimpleAttention().to(device).eval()
        m = torch.compile(m, mode="max-autotune", fullgraph=True)
        q = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        out = m(q, k, v)
        out.sum().backward()

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, device, dtype):
        # [FUNCTION DOC] Adds a captured batch/head/sequence/sequence bias; regular and
        # paged paths.
        bias = torch.randn(B, H, S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured bias at batch, head, q and kv.
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_rel_bias(self, device, dtype):
        # [FUNCTION DOC] Adds a relative bias looked up by q minus kv offset by the
        # sequence length.
        rel_bias = torch.randn(2 * S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured table entry at the shifted
            # relative position.
            return score + rel_bias[(q - kv) + S]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_dependent_causal_bidirectional(self, device, dtype):
        # [FUNCTION DOC] Score_mod mixing causal attention with a per-batch bidirectional prefix
        # length read from a captured int32 tensor.
        num_bidirectional = torch.randint(0, S, (B,), device=device, dtype=torch.int32)

        def bias_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: keeps scores that are causal or inside the batch's
            # bidirectional prefix; others negative infinity.
            causal_attention = q >= kv
            cur_num_bidirectional = num_bidirectional[b]
            bidirectional_attention_on_video = (q <= cur_num_bidirectional) & (
                kv <= cur_num_bidirectional
            )
            return torch.where(
                bidirectional_attention_on_video | causal_attention,
                score,
                -float("inf"),
            )

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_natten_2d(self, device, dtype):
        # [FUNCTION DOC] 2-D neighborhood (NATTEN-like) masking over a 32 x 8 grid with window 3.
        # Asserts the grid covers the sequence; regular and paged paths.
        H = 32
        W = S // H
        WINDOW = 3
        if W * H != S:
            raise AssertionError(f"Expected W * H == S, got {W * H} != {S}")

        def get_x_y(idx):
            # [FUNCTION DOC] HELPER: maps a flat index to grid x and y; x uses true division
            # (comment says floor divide is not properly supported), y uses the remainder.
            # This should be a floor divide, but we don't support that properly
            return idx / W, idx % W

        def natten_mask(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: keeps scores where either coordinate difference is within
            # the window.
            q_x, q_y = get_x_y(q)
            kv_x, kv_y = get_x_y(kv)
            return torch.where(
                ((q_x - kv_x).abs() <= WINDOW) | ((q_y - kv_y).abs() <= WINDOW),
                score,
                float("-inf"),
            )

        self.run_test(natten_mask, dtype, device=device)
        self.run_test_with_paged_attention(natten_mask, dtype, device=device)

    @supported_platform
    def test_subgraph_respect_decompostion(self, device):
        # [FUNCTION DOC] Traces flex_attention with make_fx under an empty and under the core ATen
        # decomposition table and checks (inline expected FX code) that floor division in the
        # score subgraph stays a floor_divide op only without decompositions and becomes a
        # division in floor rounding mode with them. Uses float32 on MPS, float64 otherwise.
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        def score_mod_func(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: score_mod subtracting the floor division of q by (1 + kv).
            return score - q // (1 + kv)

        # MPS doesn't support fp64; the test only inspects FX graph code,
        # which does not embed the input dtype in this assertion.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        # floor_div is not decomposed in decomposition_table is empty
        attention = functools.partial(flex_attention, score_mod=score_mod_func)
        gm = make_fx(attention, decomposition_table={})(query, key, value)
        self.assertExpectedInline(
            gm.sdpa_score0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    floor_divide = torch.ops.aten.floor_divide.default(arg3_1, add);  arg3_1 = add = None
    sub = torch.ops.aten.sub.Tensor(arg0_1, floor_divide);  arg0_1 = floor_divide = None
    return sub""",
        )

        # floor_div is decomposed for core_aten_decompositions
        gm = make_fx(attention, decomposition_table=core_aten_decompositions())(
            query, key, value
        )
        self.assertExpectedInline(
            gm.sdpa_score0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    div = torch.ops.aten.div.Tensor_mode(arg3_1, add, rounding_mode = 'floor');  arg3_1 = add = None
    sub = torch.ops.aten.sub.Tensor(arg0_1, div);  arg0_1 = div = None
    return sub""",
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_silu_on_score(self, device, dtype):
        # [FUNCTION DOC] Applies SiLU to the score inside the score_mod; regular and paged paths.
        def silu_score(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: score_mod returning silu of the score.
            return torch.nn.functional.silu(score)

        self.run_test(silu_score, dtype, device=device)
        self.run_test_with_paged_attention(silu_score, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_padded_dense_causal(self, device, dtype):
        # [FUNCTION DOC] Wraps the causal score_mod with a per-batch padding limit read from a
        # captured tensor and runs the regular path only.
        seq_len = torch.arange(B, device=device, dtype=torch.int32) + 1

        def create_padded_dense_wrapper(orig_score_mod):
            # [FUNCTION DOC] HELPER: returns a score_mod that wraps the given one and masks to
            # negative infinity where the score value is above the captured sequence length of
            # the batch.
            def njt_score_mod(qk, b, h, q, kv):
                # [FUNCTION DOC] HELPER: applies the wrapped score_mod, masking per
                # captured length.
                return torch.where(
                    qk <= seq_len[b], orig_score_mod(qk, b, h, q, kv), -float("inf")
                )

            return njt_score_mod

        causal_njt = create_padded_dense_wrapper(_causal)

        self.run_test(causal_njt, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_scale(self, device, dtype):
        # [FUNCTION DOC] Adds a captured 0-d int32 tensor to the score; regular and paged paths.
        scale = torch.ones((), device=device, dtype=torch.int32)

        def score_mod_scale(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured scalar to the score.
            return qk + scale

        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    @dtypes(*device_configs["cuda"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_scalar_grad(self, device, dtype):
        # [FUNCTION DOC] Non-CPU test: learnable scalar parameter of shape one with a literal
        # index captured by a score_mod; delegates to the shared helper.
        """Test learnable scalar parameter with shape (1,) using literal index."""
        self._run_test_captured_scalar_grad(device, dtype)

    def _run_test_captured_scalar_grad(self, device, dtype):
        # [FUNCTION DOC] HELPER: runs the scalar-grad scenario, then resets dynamo, collects
        # garbage, clears cuBLAS workspaces when CUDA exists and empties the accelerator cache
        # (twice) to release memory.
        def run():
            # [FUNCTION DOC] HELPER: builds a requires-grad scalar parameter and checks a
            # score_mod using it through regular and paged paths.
            scale = torch.ones((1,), device=device, dtype=dtype, requires_grad=True)

            def score_mod_scale(qk, b, h, q, kv):
                # [FUNCTION DOC] HELPER: adds the captured scalar element to the score.
                return qk + scale[0]

            self.run_test(score_mod_scale, dtype, device=device)
            self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

        run()
        torch._dynamo.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch._C._cuda_clearCublasWorkspaces()
        torch.accelerator.empty_cache()
        torch._dynamo.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch._C._cuda_clearCublasWorkspaces()
        torch.accelerator.empty_cache()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @dtypes(*device_configs["cuda"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("compiled", [False, True])
    def test_unused_captured_score_mod_grad(self, device, dtype, compiled):
        # [FUNCTION DOC] Non-CPU/MPS test (eager or compiled): a captured requires-grad tensor
        # that the score_mod indexes but never uses must get a None grad, while q, k, v get grads.
        # Resets dynamo at the end.
        """Unused captured grad slots should propagate None through backward."""
        B_local, H_local, S_local, D_local = 1, 4, 16, 64
        make_input = functools.partial(
            torch.randn,
            (B_local, H_local, S_local, D_local),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        q, k, v = make_input(), make_input(), make_input()
        unused = torch.zeros(
            (H_local, 2), device=device, dtype=dtype, requires_grad=True
        )

        def score_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: reads the captured tensor without using it; returns
            # the score.
            _ = unused[h]
            return score

        attention = torch.compile(flex_attention) if compiled else flex_attention
        attention(q, k, v, score_mod=score_mod).sum().backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertIsNone(unused.grad)
        torch._dynamo.reset()

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    @dtypes(torch.float16)
    @dtypesIfCUDA(torch.float16)
    @common_utils.parametrize("detach_temp", [False, True])
    @expected_not_implemented_on_mps
    def test_captured_0d_scalar_grad(self, device, dtype, detach_temp):
        # [FUNCTION DOC] Non-CPU/XPU float16 test (MPS must raise NotImplementedError): a module
        # with a 0-d learnable temperature captured by the score_mod (optionally detached).
        # Compares outputs and q, k, v grads of eager vs compiled; the temperature grad is None
        # when detached, otherwise compared as well.
        class M(nn.Module):
            # [CLASS DOC] HELPER: module with a scalar parameter whose forward captures it in a
            # score_mod passed to flex_attention.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the 0.7-valued float32 scalar parameter on
                # the device.
                super().__init__()
                self.temp = nn.Parameter(
                    torch.tensor(0.7, device=device, dtype=torch.float32)
                )

            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: runs flex_attention with a score_mod that uses the
                # temperature (detached, or as scale and shift depending on the detach_temp
                # test parameter).
                temp = self.temp

                def score_mod(score, b, h, q_idx, kv_idx):
                    # [FUNCTION DOC] HELPER: adds the detached temperature, or multiplies by it
                    # and adds it.
                    if detach_temp:
                        return score + temp.detach()
                    return score * temp + temp

                return flex_attention(q, k, v, score_mod=score_mod)

        torch.manual_seed(123)
        shape = (1, 2, 16, 16)
        q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        grad = torch.randn(shape, device=device, dtype=dtype)
        q2 = q.detach().clone().requires_grad_()
        k2 = k.detach().clone().requires_grad_()
        v2 = v.detach().clone().requires_grad_()
        m1 = M()
        m2 = M()
        m2.load_state_dict(m1.state_dict())

        out1 = m1(q, k, v)
        out1.backward(grad)
        out2 = torch.compile(m2)(q2, k2, v2)
        out2.backward(grad)
        if device == "cuda":
            torch.cuda.synchronize()

        pairs = [
            (out1, out2),
            (q.grad, q2.grad),
            (k.grad, k2.grad),
            (v.grad, v2.grad),
        ]
        if detach_temp:
            self.assertIsNone(m1.temp.grad)
            self.assertIsNone(m2.temp.grad)
        else:
            self.assertIsNotNone(m1.temp.grad)
            self.assertIsNotNone(m2.temp.grad)
            pairs.append((m1.temp.grad, m2.temp.grad))
        for a, b in pairs:
            self.assertEqual(a, b, atol=1e-2, rtol=1e-2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_recompile_changed_score_mod(self, device, dtype):
        # [FUNCTION DOC] Checks recompilation when the Python-level flag read by the score_mod
        # changes from addition to multiplication between two runs; each run uses regular and
        # paged paths.
        scale = torch.ones((), device=device, dtype=torch.int32)
        ADD = True

        def score_mod_scale(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds or multiplies the captured scale depending on the
            # enclosing flag.
            if ADD:
                return qk + scale
            else:
                return qk * scale

        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

        ADD = False
        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

    @supported_platform
    @expectedFailure  # If we capture a tensor then we can perform a reduction on it, and that shouldn't be allowed
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_reduction(self, device, dtype):
        # [FUNCTION DOC] Expected failure: a score_mod reducing over a captured tensor should not
        # be allowed but currently is not rejected.
        scale = torch.randn((B, 8), device=device)

        def score_mod_scale(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the sum of the captured row for the batch.
            return qk + scale[b].sum(dim=-1)

        self.run_test(score_mod_scale, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    @dtypes(torch.float16)
    @dtypesIfCUDA(torch.float16)
    @expected_not_implemented_on_mps
    def test_dynamic_captured_buffer(self, device, dtype):
        # [FUNCTION DOC] Non-CPU float16 test (MPS must raise NotImplementedError): compiles once
        # with dynamic shapes and fullgraph, then runs forward and backward with head counts
        # changing across calls; only verifies no error.
        def run_with_head_count(compiled_fa, head_count):
            # [FUNCTION DOC] HELPER: builds a requires-grad per-head scale and q, k, v for the
            # head count and runs compiled attention forward and backward.
            head_scale = torch.randn(
                head_count, device=device, dtype=dtype, requires_grad=True
            )

            def score_mod(score, batch, head, token_q, token_kv):
                # [FUNCTION DOC] HELPER: multiplies the score by the captured per-head scale.
                return score * head_scale[head]

            q = torch.randn(
                B, head_count, S, D, device=device, dtype=dtype, requires_grad=True
            )
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)

            block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

            out = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
            loss = out.sum()
            loss.backward()
            return out

        compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=True)

        head_counts = [4, 8, 4, 16, 4]
        for head_count in head_counts:
            run_with_head_count(compiled_fa, head_count)

    @supported_platform
    @skip_on_cpu
    @dtypes(torch.bfloat16)
    @dtypesIfCUDA(torch.bfloat16)
    @expected_not_implemented_on_mps  # uses requires_grad+autograd.grad; backward NIE on MPS
    def test_compacted_block_mask_matches_reference_after_recompile(
        self, device, dtype
    ):
        # [FUNCTION DOC] Non-CPU bf16 test (MPS raises NotImplementedError): builds document-based
        # block masks both as created and with indices trimmed to the largest block count, runs
        # eager reference first then a compiled run on both compact masks with dynamic off (second
        # call recompiles), and compares outputs and q, k, v grads to the reference.
        block_size = 128
        seq_len = 512
        head_dim = 64
        batch_size = 1

        def mask_fn(b, h, q_idx, kv_idx, document_id):
            # [FUNCTION DOC] HELPER: mask_mod true where causal or both positions share a
            # document id.
            causal = q_idx >= kv_idx
            same_document = document_id[b, q_idx] == document_id[b, kv_idx]
            return causal | same_document

        def make_block_mask(document_id, compact):
            # [FUNCTION DOC] HELPER: creates a block mask from the document ids; when compact is
            # requested returns a BlockMask whose index tensors are trimmed to the maximum used
            # block count.
            ref = create_block_mask(
                functools.partial(mask_fn, document_id=document_id),
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=block_size,
            )
            if not compact:
                return ref

            nums = [
                ref.kv_num_blocks,
                ref.full_kv_num_blocks,
                ref.q_num_blocks,
                ref.full_q_num_blocks,
            ]
            indices = [
                ref.kv_indices,
                ref.full_kv_indices,
                ref.q_indices,
                ref.full_q_indices,
            ]
            compact_indices = [
                index[..., : max(int(num.max()), 1)].contiguous()
                for num, index in zip(nums, indices, strict=True)
            ]
            return BlockMask(
                seq_lengths=ref.seq_lengths,
                kv_num_blocks=ref.kv_num_blocks,
                kv_indices=compact_indices[0],
                full_kv_num_blocks=ref.full_kv_num_blocks,
                full_kv_indices=compact_indices[1],
                q_num_blocks=ref.q_num_blocks,
                q_indices=compact_indices[2],
                full_q_num_blocks=ref.full_q_num_blocks,
                full_q_indices=compact_indices[3],
                BLOCK_SIZE=ref.BLOCK_SIZE,
                mask_mod=ref.mask_mod,
            )

        document_id_1 = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size, 1
        )
        document_id_1[:, :400] = 0
        document_id_2 = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size, 1
        )

        query = torch.randn(
            (batch_size, 1, seq_len, head_dim), device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        def run(mask, attention):
            # [FUNCTION DOC] HELPER: runs attention on detached requires-grad copies with the mask
            # and returns the output and the gradients with respect to q, k, v.
            q, k, v = [
                x.detach().clone().requires_grad_(True) for x in (query, key, value)
            ]
            out = attention(q, k, v, block_mask=mask)
            return out, torch.autograd.grad(out.sum(), (q, k, v))

        ref_mask_1 = make_block_mask(document_id_1, compact=False)
        ref_mask_2 = make_block_mask(document_id_2, compact=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_out_1, ref_grads_1 = run(ref_mask_1, flex_attention)
            ref_out_2, ref_grads_2 = run(ref_mask_2, flex_attention)

        compact_mask_1 = make_block_mask(document_id_1, compact=True)
        compact_mask_2 = make_block_mask(document_id_2, compact=True)

        torch._dynamo.reset()
        compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        out_1, grads_1 = run(compact_mask_1, compiled_fa)
        out_2, grads_2 = run(compact_mask_2, compiled_fa)

        torch.testing.assert_close(ref_out_1, out_1, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(ref_out_2, out_2, atol=2e-2, rtol=2e-2)
        for ref_grad, grad in zip(ref_grads_1, grads_1):
            torch.testing.assert_close(ref_grad, grad, atol=3e-2, rtol=3e-2)
        for ref_grad, grad in zip(ref_grads_2, grads_2):
            torch.testing.assert_close(ref_grad, grad, atol=3e-2, rtol=3e-2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    def test_return_max(self, device, dtype, score_mod):
        # [FUNCTION DOC] Non-CPU test of the max_scores auxiliary output: eager flex_attention
        # with no aux, max_scores only, or both must give identical outputs and max_scores;
        # compiled matches within tolerances (looser for float16 squared); max_scores shape is
        # batch, head, query, kept float32 without grad; and grads flow to q, k, v when supported.
        # KNOWN ISSUE: the message passed to assertIsNotNone is a lambda that is only stringified.
        # MPS has forward-only flex_attention
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 243, 16),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out_only = flex_attention(query, key, value, score_mod)
        out_max, aux_max = flex_attention(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )
        out_both, aux_both = flex_attention(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out_compiled, aux_compiled = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )

        torch.testing.assert_close(out_only, out_max, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out_only, out_both, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            aux_max.max_scores, aux_both.max_scores, atol=1e-6, rtol=1e-6
        )

        # we are calculating slightly different scores so add a lil fudge
        # Extra tolerance for squared score_mod with float16 due to limited dynamic range
        if score_mod.__name__ == "_squared" and dtype == torch.float16:
            atol, rtol = 2e-2, 2e-2
        else:
            atol, rtol = 5e-3, 5e-3

        torch.testing.assert_close(out_max, out_compiled, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            aux_max.max_scores, aux_compiled.max_scores, atol=atol, rtol=rtol
        )

        B, H, L = query.shape[:3]
        self.assertEqual(aux_max.max_scores.shape, (B, H, L))

        max_score_tensors = [
            aux_max.max_scores,
            aux_both.max_scores,
            aux_compiled.max_scores,
        ]
        for max_tensor in max_score_tensors:
            self.assertFalse(
                max_tensor.requires_grad, "max_scores should not require gradients"
            )
            self.assertEqual(
                max_tensor.dtype, torch.float32, "max_scores should be kept in fp32"
            )

        # Test gradient computation for both eager and compiled versions
        if requires_grad:
            test_cases = [
                ("eager", out_max, "eager mode"),
                ("compiled", out_compiled, "compiled mode"),
            ]

            for mode_name, output, description in test_cases:
                loss = output.sum()
                grads = torch.autograd.grad(loss, (query, key, value))

                # Verify gradients are computed for all inputs
                input_names = ["query", "key", "value"]
                for grad, input_name in zip(grads, input_names):
                    self.assertIsNotNone(
                        grad,
                        lambda msg: f"{msg}\n{input_name} should receive gradients in {description}",
                    )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_return_aux(self, device, dtype, score_mod):
        # [FUNCTION DOC] Non-CPU test (MPS raises NotImplementedError) of the return_aux API with
        # compiled fullgraph attention: no aux gives a tensor; lse only, max_scores only, and both
        # give an AuxOutput with the requested fields set (others None), shape (2, 2, 243) and
        # float32 dtype.
        """Test the new return_aux API with AuxRequest/Output"""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 243, 16),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        flex_compile_partial = torch.compile(flex_attention, fullgraph=False)

        # Test 1: No auxiliary outputs (default behavior)
        out_only = flex_compile(query, key, value, score_mod)
        self.assertIsInstance(out_only, torch.Tensor)

        # Test 2: Request only LSE
        out, aux_lse = flex_compile(
            query, key, value, score_mod, return_aux=AuxRequest(lse=True)
        )
        self.assertIsInstance(aux_lse, AuxOutput)
        self.assertIsInstance(aux_lse.lse, torch.Tensor)
        self.assertIsNone(aux_lse.max_scores)
        self.assertEqual(aux_lse.lse.shape, (2, 2, 243))
        self.assertEqual(aux_lse.lse.dtype, torch.float32)

        # Test 3: Request only max_scores
        out, aux_max = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )
        self.assertIsInstance(aux_max, AuxOutput)
        self.assertIsNone(aux_max.lse)
        self.assertIsInstance(aux_max.max_scores, torch.Tensor)
        self.assertEqual(aux_max.max_scores.shape, (2, 2, 243))
        self.assertEqual(aux_max.max_scores.dtype, torch.float32)

        # Test 4: Request both auxiliary outputs
        out, aux_both = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )
        self.assertIsInstance(aux_both, AuxOutput)
        self.assertIsInstance(aux_both.lse, torch.Tensor)
        self.assertIsInstance(aux_both.max_scores, torch.Tensor)
        self.assertEqual(aux_both.lse.shape, (2, 2, 243))
        self.assertEqual(aux_both.max_scores.shape, (2, 2, 243))

        # Test 5: Request no auxiliary outputs explicitly
        out, aux_none = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(),  # Default is lse=False, max_scores=False
        )
        self.assertIsInstance(aux_none, AuxOutput)
        self.assertIsNone(aux_none.lse)
        self.assertIsNone(aux_none.max_scores)

        # Test 6: Verify outputs are consistent with legacy API, can't fullgraph through warnings
        out_legacy, lse_legacy = flex_compile_partial(
            query, key, value, score_mod, return_lse=True
        )
        torch.testing.assert_close(out_only, out_legacy, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aux_lse.lse, lse_legacy, atol=1e-6, rtol=1e-6)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/162464",
    )
    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    @skip_on_mps  # tests deprecation-warning emission; flaky under test-suite ordering
    def test_return_aux_deprecation_warnings(self, device, dtype):
        # [FUNCTION DOC] Checks the legacy return_lse deprecation: with the shown-warnings
        # registry cleared, calling with return_lse emits a warning mentioning deprecation,
        # and passing both return_lse and return_aux raises ValueError ("Cannot specify both
        # ..."). The registry is restored in a finally block. KNOWN ISSUE: the skipIf
        # condition uses IS_LINUX with or, so the test is skipped on every Linux machine
        # regardless of other flags.
        """Test that deprecation warnings are issued for legacy parameters"""
        import warnings

        import torch.nn.attention.flex_attention as fa

        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 64, 16),
            device=device,
            dtype=dtype,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # Clear shown warnings to ensure we can test them
        original_shown = fa._WARNINGS_SHOWN.copy()
        fa._WARNINGS_SHOWN.clear()

        try:
            # Test deprecation warning for return_lse
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                flex_attention(query, key, value, return_lse=True)
                self.assertTrue(
                    any(
                        "return_lse is deprecated" in str(warning.message)
                        for warning in w
                    )
                )

            # Clear for next test
            fa._WARNINGS_SHOWN.clear()

            # Test error when both old and new API are used
            with self.assertRaises(ValueError) as cm:
                flex_attention(
                    query,
                    key,
                    value,
                    return_lse=True,
                    return_aux=AuxRequest(lse=True),
                )
            self.assertIn(
                "Cannot specify both return_lse and return_aux", str(cm.exception)
            )

        finally:
            # Restore original warnings state
            fa._WARNINGS_SHOWN.clear()
            fa._WARNINGS_SHOWN.update(original_shown)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @expected_not_implemented_on_mps
    def test_autocast(self, device, dtype):
        # [FUNCTION DOC] Under autocast to float16, flex_attention output must match scaled dot
        # product attention within small tolerances (looser on MI200 float16) and be float16. MPS
        # must raise NotImplementedError.
        """Test torch autocast functionality"""
        q = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype).to(torch.float16)
        k = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype)
        v = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype)

        atol = 1e-3
        rtol = 1e-3

        if isRocmArchAnyOf(MI200_ARCH) and dtype == torch.float16:
            # this behavior matches known subnormal (denormal) handling on MI200 for float16
            atol = 0.002
            rtol = 0.41  # relative difference can become large at small tensor values

        with torch.autocast(dtype=torch.float16, enabled=True, device_type=device):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            flex_output = flex_attention(q, k, v)
            torch.testing.assert_close(sdpa_output, flex_output, atol=atol, rtol=rtol)
            self.assertEqual(flex_output.dtype, torch.float16)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    @skip_on_mps  # asserts Triton "IS_DIVISIBLE : tl.constexpr" in generated code
    def test_dynamic_divisibility_guards(self, device, dtype):
        # [FUNCTION DOC] Non-CPU/MPS test: compiles attention for shapes 256, 255, 383 and 384 and
        # asserts the generated code sets the divisibility constant to True only when the sequence
        # is a multiple of 128, and that output has the expected shape.
        """Test guards for divisible/non-divisible shape transitions"""
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        def score_mod(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal score_mod; masked positions become negative infinity.
            return torch.where(q >= kv, qk, -float("inf"))

        def test_shape(S, backend):
            # [FUNCTION DOC] HELPER: compiles attention for one sequence length with the given
            # backend, checks the divisibility flag in the generated code and the output shape,
            # and returns the output and code.
            """Test a single shape configuration"""
            block_mask = create_block_mask(noop_mask, 1, 1, S, S, device=device)
            sdpa_partial = create_attention(score_mod, block_mask=block_mask)

            tensors = [
                torch.randn(
                    2, 4, S, 64, dtype=dtype, device=device, requires_grad=False
                )
                for _ in range(3)
            ]

            compiled_sdpa = torch.compile(sdpa_partial, backend=backend)
            out, code = run_and_get_code(compiled_sdpa, *tensors)

            # Check divisibility flag
            is_divisible = S % 128 == 0
            expected_flag = f"IS_DIVISIBLE : tl.constexpr = {is_divisible}"
            self.assertIn(
                expected_flag,
                str(code),
                lambda msg: f"{msg}\nS={S} should have {expected_flag}",
            )

            self.assertEqual(out.shape, (2, 4, S, 64))
            return out, code

        torch._dynamo.reset()
        backend = CompileCounterWithBackend("inductor")

        # Test divisible and non-divisible shapes
        test_shapes = [256, 255, 383, 384]
        _ = [test_shape(S, backend) for S in test_shapes]

    @supported_platform
    @skip_on_cpu
    def test_mask_mod_handles_symint_addition(self, device):
        # [FUNCTION DOC] Non-CPU test: block mask built inside a dynamic fullgraph compiled
        # function using a mask_mod that adds symbolic sequence sizes; eager and compiled outputs
        # must agree for two different shape sets.
        dtype = torch.float16

        def run(q, k, v):
            # [FUNCTION DOC] HELPER: builds a block mask from a mask_mod depending on the q/kv
            # lengths and runs flex_attention with it.
            ql = q.size(-2)
            kl = k.size(-2)
            frame = 32

            def _opaque_mask(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where q index is below a limit derived from
                # both lengths.
                ref = ql // frame
                mot = kl // frame  # codespell:ignore
                limit = (ref + mot) * frame  # codespell:ignore
                return q_idx < limit

            block_mask = create_block_mask(
                _opaque_mask,
                B=q.size(0),
                H=q.size(1),
                Q_LEN=ql,
                KV_LEN=kl,
                device=device,
            )
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled_run = torch.compile(run, fullgraph=True, dynamic=True)

        q = torch.randn(1, 2, 192, 32, device=device, dtype=dtype)
        k = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)
        v = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)

        eager_out = run(q, k, v)
        compiled_out = compiled_run(q, k, v)
        torch.testing.assert_close(eager_out, compiled_out, atol=1e-3, rtol=1e-3)

        # Exercise different dynamic shapes to ensure SymInt sums remain well-formed.
        q2 = torch.randn(1, 2, 160, 32, device=device, dtype=dtype)
        k2 = torch.randn(1, 2, 96, 32, device=device, dtype=dtype)
        v2 = torch.randn(1, 2, 96, 32, device=device, dtype=dtype)

        eager_out2 = run(q2, k2, v2)
        compiled_out2 = compiled_run(q2, k2, v2)
        torch.testing.assert_close(eager_out2, compiled_out2, atol=1e-3, rtol=1e-3)

    @supported_platform
    @skip_on_cpu
    def test_mask_mod_handles_derived_symint_closure(self, device):
        # [FUNCTION DOC] Non-CPU test: mask_mod capturing a value derived from an integer argument
        # (position plus one) under dynamic fullgraph compile; eager and compiled agree for two
        # different positions.
        dtype = torch.float16

        def run(q, k, v, current_pos):
            # [FUNCTION DOC] HELPER: builds a block mask from a closure on the derived position
            # and runs flex_attention.
            p_plus = current_pos + 1

            def _opaque_mask(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where kv index is at most the captured
                # derived value.
                return kv_idx <= p_plus

            block_mask = create_block_mask(
                _opaque_mask,
                B=None,
                H=None,
                Q_LEN=q.size(-2),
                KV_LEN=k.size(-2),
                device=device,
            )
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled_run = torch.compile(run, fullgraph=True, dynamic=True)

        q = torch.randn(1, 2, 192, 32, device=device, dtype=dtype)
        k = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)
        v = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)

        eager_out = run(q, k, v, 5)
        compiled_out = compiled_run(q, k, v, 5)
        torch.testing.assert_close(eager_out, compiled_out, atol=1e-3, rtol=1e-3)

        # Exercise a different captured value to ensure derived SymInt captures remain well-formed.
        eager_out2 = run(q, k, v, 9)
        compiled_out2 = compiled_run(q, k, v, 9)
        torch.testing.assert_close(eager_out2, compiled_out2, atol=1e-3, rtol=1e-3)

    @supported_platform
    def test_multiple_score_mod_calls(self, device):
        # [FUNCTION DOC] Two chained flex_attention calls with different score_mods, compiled vs
        # eager, within 0.2 tolerances.
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]

        def scoremod_1(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the relative position (q minus kv) to the score.
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal score_mod with negative infinity for masked positions.
            return torch.where(q >= kv, qk, -float("inf"))

        def f(q, k1, k2, v1, v2):
            # [FUNCTION DOC] HELPER: applies two attentions in sequence, feeding the first output
            # as the second's query.
            q2 = flex_attention(q, k1, v1, score_mod=scoremod_1)
            return flex_attention(q2, k2, v2, score_mod=scoremod_2)

        out = f(query, *keys, *values)
        out2 = torch.compile(f)(query, *keys, *values)
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_multiple_mask_calls(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): one function calling
        # flex_attention with a causal and a sliding-window block mask and summing; compiled
        # gradients w.r.t. q, k, v match eager.
        make_tensor = functools.partial(
            torch.randn,
            (1, 4, 512, 64),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        window_size = 32

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        def causal_mask_slidewindow_mod(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod for causal attention limited to the captured
            # window size.
            return (q_idx >= kv_idx) & (q_idx <= kv_idx + window_size)

        mask1 = create_block_mask(
            causal_mask, 1, None, 512, 512, _compile=False, device=device
        )
        mask2 = create_block_mask(
            causal_mask_slidewindow_mod,
            1,
            None,
            512,
            512,
            _compile=False,
            device=device,
        )

        def f(q, k, v):
            # [FUNCTION DOC] HELPER: sums the outputs of attention under the two block masks.
            out1 = flex_attention(q, k, v, block_mask=mask1)
            out2 = flex_attention(q, k, v, block_mask=mask2)
            return out1 + out2

        f_compiled = torch.compile(f, fullgraph=True)

        out = f(query, key, value)
        out_compiled = f_compiled(query, key, value)

        grads = torch.autograd.grad((out,), (query, key, value), torch.ones_like(out))
        grads_compile = torch.autograd.grad(
            (out_compiled,), (query, key, value), torch.ones_like(out_compiled)
        )

        for grad, grad_compiled in zip(grads, grads_compile):
            torch.testing.assert_close(grad, grad_compiled, atol=3e-2, rtol=3e-2)

    @supported_platform
    def test_multiple_score_mod_calls2(self, device):
        # [FUNCTION DOC] Three chained attention calls (one through a partial) compiled with
        # fullgraph; mean absolute difference from eager must be under 0.01.
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]

        def scoremod_1(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the relative position to the score.
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal score_mod with negative infinity for masked positions.
            return torch.where(q >= kv, qk, -float("inf"))

        attention1 = functools.partial(flex_attention, score_mod=scoremod_1)

        def f(q, k1, k2, k3, v1, v2, v3):
            # [FUNCTION DOC] HELPER: three chained attentions, each feeding the next as query.
            q2 = attention1(q, k1, v1)
            q3 = flex_attention(q2, k2, v2, score_mod=scoremod_2)
            return flex_attention(q3, k3, v3, score_mod=scoremod_1)

        out = f(query, *keys, *values)
        out2 = torch.compile(f, fullgraph=True)(query, *keys, *values)
        self.assertTrue((out - out2).abs().mean() < 1e-2)

    @supported_platform
    def test_multiple_score_mod_calls_paged_attention(self, device):
        # [FUNCTION DOC] Two chained attentions evaluated eagerly, then the same with each stage
        # converted to paged attention (separate caches, converted masks and score_mods), compiled
        # with fullgraph; compared within 0.2 tolerances.
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]

        def scoremod_1(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the relative position to the score.
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal score_mod with negative infinity for masked positions.
            return torch.where(q >= kv, qk, -float("inf"))

        def f(q, k1, k2, v1, v2):
            # [FUNCTION DOC] HELPER: eager reference for two chained attentions.
            q2 = flex_attention(q, k1, v1, score_mod=scoremod_1)
            return flex_attention(q2, k2, v2, score_mod=scoremod_2)

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024, device=device)

        (
            k_cache1,
            v_cache1,
            converted_block_mask1,
            converted_score_mod1,
        ) = self.preprocess_paged_attention(
            scoremod_1,
            query,
            keys[0],
            values[0],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2,
            query,
            keys[1],
            values[1],
            block_mask,
            torch.float32,
            device=device,
        )

        def paged_f(q, k1, k2, v1, v2):
            # [FUNCTION DOC] HELPER: two chained paged attentions with their converted masks
            # and score_mods.
            q2 = flex_attention(
                q,
                k1,
                v1,
                score_mod=converted_score_mod1,
                block_mask=converted_block_mask1,
            )
            return flex_attention(
                q2,
                k2,
                v2,
                score_mod=converted_score_mod2,
                block_mask=converted_block_mask2,
            )

        compiled_out = torch.compile(paged_f, fullgraph=True)(
            query, k_cache1, k_cache2, v_cache1, v_cache2
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_multiple_score_mod_calls2_paged_attention(self, device):
        # [FUNCTION DOC] Three chained attentions, eagerly and as paged attention (one via a
        # partial) compiled with fullgraph; compared within 0.2 tolerances.
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]

        def scoremod_1(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the relative position to the score.
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal score_mod with negative infinity for masked positions.
            return torch.where(q >= kv, qk, -float("inf"))

        attention1 = functools.partial(flex_attention, score_mod=scoremod_1)

        def f(q, k1, k2, k3, v1, v2, v3):
            # [FUNCTION DOC] HELPER: eager reference for three chained attentions.
            q2 = attention1(q, k1, v1)
            q3 = flex_attention(q2, k2, v2, score_mod=scoremod_2)
            return flex_attention(q3, k3, v3, score_mod=scoremod_1)

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024, device=device)
        (
            k_cache1,
            v_cache1,
            converted_block_mask1,
            converted_score_mod1,
        ) = self.preprocess_paged_attention(
            scoremod_1,
            query,
            keys[0],
            values[0],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2,
            query,
            keys[1],
            values[1],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache3,
            v_cache3,
            converted_block_mask3,
            converted_score_mod3,
        ) = self.preprocess_paged_attention(
            scoremod_1,
            query,
            keys[2],
            values[2],
            block_mask,
            torch.float32,
            device=device,
        )

        paged_attention1 = functools.partial(
            flex_attention,
            score_mod=converted_score_mod1,
            block_mask=converted_block_mask1,
        )

        def paged_f(q, k1, k2, k3, v1, v2, v3):
            # [FUNCTION DOC] HELPER: three chained paged attentions using the converted masks
            # and score_mods.
            q2 = paged_attention1(q, k1, v1)
            q3 = flex_attention(
                q2,
                k2,
                v2,
                score_mod=converted_score_mod2,
                block_mask=converted_block_mask2,
            )
            return flex_attention(
                q3,
                k3,
                v3,
                score_mod=converted_score_mod3,
                block_mask=converted_block_mask3,
            )

        compiled_out = torch.compile(paged_f, fullgraph=True)(
            query, k_cache1, k_cache2, k_cache3, v_cache1, v_cache2, v_cache3
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_inputs_are_realized(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): the function applies sin to q, a
        # score_mod reading a freshly computed non-input tensor and cos to the output. Compiled
        # output and q, k, v grads must be within 0.01 mean absolute error of eager.
        def f(q, k, v):
            # [FUNCTION DOC] HELPER: builds a scaled random tensor captured by a score_mod and
            # wraps the attention of the sine of q with cosine.
            x = torch.randn(1024, device=device)
            x = x * 2

            def func(qk, b, h, q, kv):
                # [FUNCTION DOC] HELPER: adds the captured tensor entry at q index to the score.
                return qk + x[q]

            return flex_attention(q.sin(), k, v, score_mod=func).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device=device, requires_grad=True)
            for _ in range(3)
        )
        ref = f(q, k, v)
        out = torch.compile(f)(q, k, v)
        self.assertTrue((ref - out).abs().mean() < 1e-2)
        gradOut = torch.randn_like(q)

        ref_grads = torch.autograd.grad(ref, (q, k, v), gradOut)
        out_grads = torch.autograd.grad(out, (q, k, v), gradOut)
        for ref, out in zip(ref_grads, out_grads):
            self.assertTrue((ref - out).abs().mean() < 1e-2)

    @supported_platform
    @skip_on_cpu
    def test_make_block_mask(self, device):
        # [FUNCTION DOC] Non-CPU test: a fullgraph compiled create_block_mask matches the eager
        # one on kv_num_blocks, kv_indices and q_num_blocks for a causal mask.
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask_a = torch.compile(create_block_mask, fullgraph=True)(
            causal_mask, 1, 1, 512, 512, device=device
        )
        block_mask_b = create_block_mask(causal_mask, 1, 1, 512, 512, device=device)
        self.assertEqual(block_mask_a.kv_num_blocks, block_mask_b.kv_num_blocks)
        self.assertEqual(block_mask_a.kv_indices, block_mask_b.kv_indices)
        self.assertEqual(block_mask_a.q_num_blocks, block_mask_b.q_num_blocks)

    @supported_platform
    def test_mask_mod_combiners(self, device):
        # [FUNCTION DOC] Tests and_masks and or_masks: causal AND sliding window gives an
        # inline-expected 28 kv blocks total and passes run_test_with_call; causal AND anti-causal
        # gives zero blocks; causal OR anti-causal has the same sparsity as the noop mask.
        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        def neg_causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where q < kv.
            return q < kv

        def sliding_window(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where q - kv at most 512.
            return (q - kv) <= 512

        local_s = 2048
        block_mask = create_block_mask(
            and_masks(causal_mask, sliding_window),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        self.assertExpectedInline(block_mask.kv_num_blocks.sum().item(), """28""")
        attention = functools.partial(flex_attention, block_mask=block_mask)
        self.run_test_with_call(
            attention, Q_S=local_s, KV_S=local_s, dtype=torch.float16, device=device
        )

        block_mask = create_block_mask(
            and_masks(causal_mask, neg_causal_mask),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        self.assertEqual(block_mask.kv_num_blocks.sum(), 0)

        block_mask1 = create_block_mask(
            or_masks(causal_mask, neg_causal_mask),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        block_mask2 = create_block_mask(
            noop_mask, 1, 1, local_s, local_s, device=device
        )
        self.assertEqual(block_mask1.sparsity(), block_mask2.sparsity())

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts "triton_tem_fused" in generated code
    def test_epilogue_fused(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: enables inductor metric logging, compiles attention
        # followed by cos, checks with FileCheck that code has a template kernel and no separate
        # pointwise cos kernel, and that bytes accessed is below four accesses plus one for
        # logsumexp. Resets logging afterwards.
        # set so that metrics appear
        torch._logging.set_logs(inductor_metrics=True)

        @torch.compile
        def f(q, k, v):
            # [FUNCTION DOC] HELPER: flex_attention output followed by a cos.
            out = flex_attention(q, k, v)
            return out.cos()

        q, k, v = (torch.randn(1, 8, 1024, 64, device=device) for _ in range(3))
        metrics.reset()
        _, code = run_and_get_code(f, q, k, v)
        fc = FileCheck()
        fc.check("triton_tem_fused")  # template call
        fc.check_not("poi_fused_cos")  # No cos pointwise operation
        fc.run(code[0])
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize
        num_accesses = 4  # q, k, v reads, one output.
        # TODO: Get rid of this fudge factor
        # We need this fudge factor for now as we write the extraneous logsumexp
        num_accesses += 1
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)
        torch._logging.set_logs()

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_njt_causal(self, device, dtype):
        # [FUNCTION DOC] Nested-jagged-tensor style causal score_mod: sequence boundaries come
        # from offsets and a captured per-position sequence index; checks regular and paged paths.
        offsets = torch.tensor(
            [0, 1024, 1024 + 512, S], device=device, dtype=torch.int32
        )
        seq_idx = torch.zeros(S, device=device, dtype=torch.int32)
        for idx in range(len(offsets) - 1):
            seq_idx[offsets[idx] : offsets[idx + 1]] = idx

        def create_njt_wrapper(orig_score_mod, offsets, seq_idx):
            # [FUNCTION DOC] HELPER: returns a score_mod that rebases q and kv indices to their
            # sequence start from the offsets table and calls the original.
            def njt_score_mod(qk, b, h, q, kv):
                # [FUNCTION DOC] HELPER: shifts q and kv by their sequence offsets and applies the
                # original mod.
                q_nested = q - offsets[seq_idx[q]]
                kv_nested = kv - offsets[seq_idx[kv]]
                return orig_score_mod(qk, b, h, q_nested, kv_nested)

            return njt_score_mod

        causal_njt = create_njt_wrapper(_causal, offsets, seq_idx)

        self.run_test(causal_njt, dtype, device=device)
        self.run_test_with_paged_attention(causal_njt, dtype, device=device)

    @supported_platform
    @skip_on_mps  # eager fp8 mixed-dtype path; MPS lacks the dispatch
    def test_mixed_dtypes_eager(self, device):
        # [FUNCTION DOC] Not on MPS: eager attention with float16 (or float32 without FP8) query
        # and fp8 (or float16) key/value; output shape and dtype equal the query's.
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        query = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        value = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        out = flex_attention(query, key, value, _identity)
        self.assertEqual(out.shape, query.shape)
        self.assertEqual(out.dtype, query.dtype)

    @supported_platform
    @expected_not_implemented_on_mps
    def test_mixed_dtypes_compiled(self, device):
        # [FUNCTION DOC] Compiled mixed query/key/value dtypes (fp8 when supported and not on
        # MPS). On CPU it must raise InductorError about unsupported mixed dtypes; elsewhere shape
        # and dtype of the output match the query. MPS must raise NotImplementedError.
        # MPS doesn't support fp8 dtype conversion; force the fp16/fp32 mix on MPS.
        use_fp8 = PLATFORM_SUPPORTS_FP8 and not _is_mps_device(device)
        dtype_high = torch.float16 if use_fp8 else torch.float32
        dtype_low = e4m3_type if use_fp8 else torch.float16
        query = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        value = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        if device == "cpu":
            with self.assertRaisesRegex(
                InductorError,
                "Mixed query, key, and value dtype is not supported on this platform",
            ):
                compiled_fn(query, key, value, _identity)
        else:
            out = compiled_fn(query, key, value, _identity)
            self.assertEqual(out.shape, query.shape)
            self.assertEqual(out.dtype, query.dtype)

    @skip_on_cpu
    @supported_platform
    @skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    @skip_on_mps  # requires PLATFORM_SUPPORTS_FP8
    def test_mixed_dtypes_sqnr_per_tensor(self, device):
        # [FUNCTION DOC] Needs FP8, non-CPU/MPS: per-tensor fp8 quantized key/value with the key
        # scale applied in the score_mod and value scale after attention; SQNR versus the bf16
        # reference must exceed 10.
        query_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
        )
        key_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
        )
        value_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
        )

        key_scale = torch.max(torch.abs(key_ref)) / E4M3_MAX_POS
        value_scale = torch.max(torch.abs(value_ref)) / E4M3_MAX_POS

        key_fp8 = (key_ref / key_scale).to(e4m3_type)
        value_fp8 = (value_ref / value_scale).to(e4m3_type)

        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: dequantizes keys by multiplying the score with the key scale.
            # Dequantize keys inside the attention score computation
            return score * key_scale

        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = compiled_fn(query_ref, key_fp8, value_fp8, score_mod) * value_scale
        out_ref = compiled_fn(query_ref, key_ref, value_ref, _identity)
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @skip_on_cpu
    @supported_platform
    @skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    @skip_on_mps  # requires PLATFORM_SUPPORTS_FP8
    def test_mixed_dtypes_sqnr_per_head(self, device):
        # [FUNCTION DOC] Needs FP8, non-CPU/MPS: like the per-tensor test but with per batch-head
        # scales (key scale looked up in the score_mod, value scale broadcast afterwards); SQNR
        # over 10.
        query_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
        )
        key_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
        )
        value_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
        )

        fp8_max = E4M3_MAX_POS
        key_scale = torch.amax(torch.abs(key_ref), dim=(-2, -1)) / fp8_max  # (B, H)
        value_scale = torch.amax(torch.abs(value_ref), dim=(-2, -1)) / fp8_max  # (B, H)

        key_scale_b = key_scale[..., None, None]  # (B, H, 1, 1) for broadcasting
        value_scale_b = value_scale[..., None, None]

        key_fp8 = (key_ref / key_scale_b).to(e4m3_type)
        value_fp8 = (value_ref / value_scale_b).to(e4m3_type)

        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: multiplies the score by the key scale of the batch and head.
            # Dequantize keys inside the attention score computation
            return score * key_scale[b, h]

        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = compiled_fn(query_ref, key_fp8, value_fp8, score_mod) * value_scale_b
        out_ref = compiled_fn(query_ref, key_ref, value_ref, _identity)
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # uses .to(fp8); MPS lacks fp8 dtype
    def test_mixed_dtype_backwards_eager(self, device):
        # [FUNCTION DOC] Non-CPU/MPS: backward through eager attention with mixed dtypes must
        # raise ValueError saying the backward with mixed dtypes is unsupported.
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        q = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        )
        k = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)
        v = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)
        out = flex_attention(q, k, v, _identity).mean()
        with self.assertRaisesRegex(
            ValueError,
            "Backward pass with mixed query, key, and value dtype is not supported",
        ):
            out.backward()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # assertRaisesRegex on Triton-specific backward error
    def test_mixed_dtype_backwards_compiled(self, device):
        # [FUNCTION DOC] Non-CPU/MPS: compiled mixed-dtype forward plus backward must raise
        # InductorError saying backward with mixed dtypes is unsupported.
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        q = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        )
        k = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)
        v = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with self.assertRaisesRegex(
            InductorError,
            "Backward pass with mixed query, key, and value dtype is not supported",
        ):
            out_mixed = (compiled_fn(q, k, v, _identity)).mean()
            out_mixed.backward()

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    @skip_on_mps  # uses Triton max_autotune (no autotuning on MPS)
    def test_max_autotune(self, device):
        # [FUNCTION DOC] Runs with max_autotune patched on (not on MPS): score_mod doubling the
        # score checked via run_test and paged attention (float16, plus bfloat16 with short KV).
        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: doubles the score.
            return score * 2

        self.run_test(score_mod, dtype=torch.float16, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype=torch.float16, device=device
        )
        self.run_test_with_paged_attention(
            score_mod=score_mod,
            dtype=torch.bfloat16,
            KV_S=64,
            device=device,
        )

    @supported_platform
    @skip("TODO: Figure out why this is erroring")
    @patch.object(torch._inductor.config, "max_autotune", True)
    @skip_on_mps  # uses Triton max_autotune
    def test_max_autotune_with_captured(self, device):
        # [FUNCTION DOC] Currently skipped (reason: error to investigate): max_autotune with
        # captured head, batch and token scales added to scores; not on MPS.
        head_scale = torch.randn(H, device=device)
        batch_scale = torch.randn(B, device=device)
        tok_scale = torch.randn(S, device=device)

        def bias_mod(score, batch, head, token_q, token_kv):
            # [FUNCTION DOC] HELPER: adds captured token, batch and head scales to the score.
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(bias_mod, dtype=torch.float32, device=device)

    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    def test_non_equal_head_dims(self, device, dtype, score_mod, head_dims):
        # [FUNCTION DOC] Query/key head dim differing from value head dim (half in either
        # direction), for each score_mod; regular and paged paths.
        qk_d, v_d = head_dims
        self.run_test(score_mod, dtype, device, B, H, S, qk_d, B, H, S, V_D=v_d)
        self.run_test_with_paged_attention(
            score_mod, dtype, device, B, H, S, qk_d, B, H, S, V_D=v_d
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_autograd_function_in_score_mod(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a score_mod applying a custom
        # autograd function (masking via where, with a vmap rule) under fullgraph compile; only
        # checks the forward runs. Comments note full correctness checks are an expected failure.
        class ApplyMask(torch.autograd.Function):
            # [CLASS DOC] HELPER: autograd function masking its input to negative infinity where
            # a boolean mask is false; vmap rule generated automatically; mask is
            # non-differentiable; backward passes the gradient to the first input and None for
            # the mask.
            generate_vmap_rule = True

            @staticmethod
            def forward(a, mask):
                # [FUNCTION DOC] HELPER: forward: keeps values where mask is true, else
                # negative infinity.
                return torch.where(mask, a, -float("inf"))

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: marks the mask input as non-differentiable.
                _, mask = inputs
                ctx.mark_non_differentiable(mask)

            @staticmethod
            def backward(ctx, i):
                # [FUNCTION DOC] HELPER: backward: returns the incoming gradient for the input and
                # None for the mask.
                return i, None

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: score_mod applying the custom autograd function with mask q
            # at most kv.
            return ApplyMask.apply(score, q <= kv)

        func = torch.compile(flex_attention, fullgraph=True)

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device=device, requires_grad=True)
            for _ in range(3)
        )

        # Just checking that it runs
        func(q, k, v)

        # expectedFailure
        # This doesn't work due to vmap + autograd.Function + torch.compile not composing
        # self.run_test(score_mod)

    @supported_platform
    def test_causal_block(self, device):
        # [FUNCTION DOC] Causal block mask in float16 checked via run_test_with_call.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, dtype=torch.float16, device=device)

    @supported_platform
    def test_causal_block_paged_attention(self, device):
        # [FUNCTION DOC] Causal block mask per batch in float16 checked through the paged
        # attention path.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)
        self.run_test_with_paged_attention(
            score_mod=_identity,
            dtype=torch.float16,
            device=device,
            block_mask=block_mask,
        )

    @supported_platform
    def test_new_empty_mask_mod(self, device):
        # [FUNCTION DOC] Compiles with fullgraph a score_mod and mask_mod that create zeros with
        # the same shape as the head index tensor (new_zeros) to index a captured boolean mask;
        # only checks it compiles and runs.
        S = 128
        q, k, v = (torch.randn(4, 1, S, 64, device=device) for _ in range(3))

        attn_mask = torch.ones(4, 1, S, S, dtype=torch.bool, device=device).tril()

        def score_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the captured mask entry (indexed with zero head) to
            # the score.
            h_ = h.new_zeros(h.shape)
            return score + attn_mask[b, h_, q_idx, kv_idx]

        def causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod reading the captured lower-triangular mask with
            # zero head.
            h_ = h.new_zeros(h.shape)
            return attn_mask[b, h_, q_idx, kv_idx]

        block_mask = create_block_mask(
            causal, B=4, H=None, Q_LEN=S, KV_LEN=S, device=device
        )
        torch.compile(flex_attention, fullgraph=True)(
            q, k, v, score_mod, block_mask=block_mask
        )

    @supported_platform
    @common_utils.parametrize("head_dim", [17, 24, 94, 121])
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_non_pow_2_headdim(self, device, dtype, head_dim):
        # [FUNCTION DOC] Head dims not a power of two (17, 24, 94, 121) with relative-bias
        # score_mod via run_test.
        self.run_test(_rel_bias, dtype, device, B, H, S, head_dim, B, H, S, head_dim)

    @supported_platform
    def test_GQA_causal_mask(self, device):
        # [FUNCTION DOC] GQA with causal block mask: query heads four times key/value heads, short
        # sequences (one eighth), checked via run_test_with_call in float16.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S // 8, S // 8, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=True
        )

        self.run_test_with_call(
            attention,
            torch.float16,
            device,
            B,
            H * 4,  # Hq = 4*Hkv.
            S // 8,
            D,
            B,
            H,
            S // 8,
            D,
        )

        self.run_test_with_paged_attention(
            _identity,
            dtype=torch.float16,
            device=device,
            Q_H=H * 4,
            Q_S=S // 8,
            KV_H=H,
            KV_S=S // 8,
            block_mask=block_mask,
        )

    @supported_platform
    def test_custom_block_mask_generator(self, device):
        # [FUNCTION DOC] A causal BlockMask built by hand with from_kv_blocks (block counts 1..n
        # and index rows per query block, block size 128) must have the same dense form as the one
        # generated from the causal mask_mod.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        auto_mask = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        BLOCK_SIZE = 128

        def causal_constructor(S):
            # [FUNCTION DOC] HELPER: builds the lower-triangular block counts and indices for a
            # sequence length and returns a BlockMask via from_kv_blocks with the causal mask_mod.
            num_blocks = torch.arange(S // BLOCK_SIZE, device=device) + 1
            indices = torch.arange(S // BLOCK_SIZE, device=device).expand(
                S // BLOCK_SIZE, S // BLOCK_SIZE
            )
            num_blocks = num_blocks[None, None, :]
            indices = indices[None, None, :]
            return BlockMask.from_kv_blocks(
                num_blocks, indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=mask_mod
            )

        manual_mask = causal_constructor(S)
        self.assertEqual(auto_mask.to_dense(), manual_mask.to_dense())

    @supported_platform
    @skip_on_cpu
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    @expected_not_implemented_on_mps  # backward path (return_lse / requires_grad); NIE on MPS
    def test_logsumexp_correctness(self, device, dtype, score_mod):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): the logsumexp returned by
        # compiled attention (float32) must match the aot_eager float64 reference (both outputs
        # and lse) within 0.02 for identity and causal score_mods.
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def sdpa_hop(q, k, v, score_mod):
            # [FUNCTION DOC] HELPER: compiled attention returning output and logsumexp.
            return flex_attention(q, k, v, score_mod, return_lse=True)

        @torch.compile(backend="aot_eager")
        def eager_sdpa_hop(q, k, v, score_mod):
            # [FUNCTION DOC] HELPER: same, compiled with aot_eager backend.
            return flex_attention(q, k, v, score_mod, return_lse=True)

        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v, torch.float64)
        ref_out, ref_lse = eager_sdpa_hop(q_ref, k_ref, v_ref, score_mod)
        compiled_out, compiled_lse = sdpa_hop(q, k, v, score_mod)

        self.assertTrue(ref_lse.dtype == torch.float64)
        self.assertTrue(compiled_lse.dtype == torch.float32)

        tolerance = Tolerances(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            ref_out.to(dtype=torch.float32),
            compiled_out.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
        torch.testing.assert_close(
            ref_lse.to(dtype=torch.float32),
            compiled_lse.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_logsumexp_only_return(self, device):
        # [FUNCTION DOC] Non-CPU test: a compiled function using only the logsumexp output
        # of attention must still generate the flex attention kernel exactly once (FileCheck
        # on code).
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def func(q, k, v, score_mod):
            # [FUNCTION DOC] HELPER: takes the lse of attention and doubles it.
            _, lse = flex_attention(q, k, v, score_mod, return_lse=True)
            lse_2 = lse * 2
            return lse_2

        _, code = run_and_get_code(func, q, k, v, _identity)
        # Ensure that we're still generating the flexattention kernel
        FileCheck().check_count(".run(primals_2, primals_1, primals_3", 1, True).run(
            code[0]
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize(
        "score_mod", [_identity, _causal, _times_two, _squared, _trig, _trig2]
    )
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_aot_eager_gradcheck(self, device, score_mod):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): runs gradcheck through aot_eager
        # fullgraph attention for several score_mods (float64; float32 on MPS).
        # MPS lacks fp64; on MPS we just need _validate_device to raise NIE,
        # so use fp32 — the call never gets far enough for precision to matter.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend="aot_eager", fullgraph=True)

        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_eager_backward_strides(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): runs forward and backward through
        # aot_eager on a module producing q, k, v from a linear projection and transposes
        # (non-contiguous layouts) feeding flex_attention; checks it does not raise.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a fused qkv linear layer that splits, reshapes and
            # transposes into non-contiguous q, k, v heads before calling flex_attention.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the 256 to 768 linear layer and head and
                # attention sizes.
                super().__init__()
                self.qkv_proj = torch.nn.Linear(256, 256 * 3)
                self.n_head = 256 // 64
                self.d_attn = 256

            def forward(self, x):
                # [FUNCTION DOC] HELPER: projects x, splits into q, k, v, reshapes to heads,
                # transposes and returns the attention output.
                n_batch, n_ctx, _ = x.shape
                q, k, v = self.qkv_proj(x).split(
                    [self.d_attn, self.d_attn, self.d_attn], dim=2
                )
                q = q.reshape(n_batch, n_ctx, self.n_head, -1)
                k = k.reshape(n_batch, n_ctx, self.n_head, -1)
                v = v.reshape(n_batch, n_ctx, self.n_head, -1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x = torch.nn.attention.flex_attention.flex_attention(q, k, v)
                return x

        model = Repro().to(device)
        x = torch.randn((1, 512, 256), device=device, requires_grad=True)
        out = torch.compile(model, backend="aot_eager", fullgraph=True)(x)
        out.backward(torch.ones_like(out))

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_differentiable_logsumexp_gradcheck(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): gradcheck of a compiled
        # (aot_eager) function returning only the logsumexp output, so lse is differentiable.
        # See test_aot_eager_gradcheck for the per-device dtype rationale.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def flex_attention_lse_only(q, k, v):
            # [FUNCTION DOC] HELPER: returns only the logsumexp from attention.
            return flex_attention(q, k, v, return_lse=True)[1]

        func = torch.compile(flex_attention_lse_only, backend="aot_eager")

        self.assertTrue(
            torch.autograd.gradcheck(func, (query, key, value), raise_exception=True)
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_differentiable_logsumexp_compiled(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): loss using both output and lse
        # weighted by a random mask; eager and compiled outputs, lse and grads agree within 0.1.
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        lse_mask = torch.randn(2, 2, 128, device=device)

        out, lse = flex_attention(q, k, v, return_lse=True)
        (out.mean() + (lse * lse_mask).sum()).backward()
        q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        out2, lse2 = torch.compile(flex_attention)(q, k, v, return_lse=True)
        (out2.mean() + (lse2 * lse_mask).sum()).backward()
        q_grad2, k_grad2, v_grad2 = q.grad, k.grad, v.grad
        tolerance = Tolerances(atol=1e-1, rtol=1e-1)

        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(lse, lse2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(
            q_grad, q_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            k_grad, k_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            v_grad, v_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_score_mod_without_score_gradient_errors(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a score_mod whose result does not
        # depend on the score makes compiled backward raise RuntimeError with a message that
        # backward requires dependence on score.
        def score_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod returning a boolean mask, ignoring the score.
            return q_idx >= kv_idx

        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 128, 16),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        with self.assertRaisesRegex(
            RuntimeError,
            "flex_attention backward requires the output of score_mod to depend on score",
        ):
            torch.compile(flex_attention, dynamic=False)(q, k, v, score_mod=score_mod)

    # Use weird mask to test reusing block_mask does work well.
    @supported_platform
    @skip_on_cpu
    def _test_block_mask_reuse_with_weird_mask(self, device):
        # [FUNCTION DOC] Disabled test (underscore name): checks compiled dynamic attention with a
        # block mask from a larger size reused with shorter q, k, v (mask adjusted) matches a run
        # using a mask built for the shorter length; compares output and grads.
        # KNOWN ISSUE: the leading underscore means the test runner never collects it.
        def mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod keeping kv positions below 256 or at least 2048.
            return (kv < 256) | (kv >= 2048)

        make_tensor = functools.partial(
            torch.randn,
            (4, 4, 4096, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        block_mask = create_block_mask(mask, None, None, 4096, 4096, device=device)
        # Compile 1st version with q/k/v(seqlen=4096) and block_mask(seqlen=4096)
        torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            make_tensor(), make_tensor(), make_tensor(), block_mask=block_mask
        )

        make_tensor2 = functools.partial(
            torch.randn,
            (4, 4, 2048, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor2(), make_tensor2(), make_tensor2()

        # Compile 2nd version with q/k/v(seqlen=2048) and block_mask(seqlen=4096),
        # The graph includes the BlockMask._adjust part.
        out = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            q, k, v, block_mask=block_mask
        )
        out.sum().backward()
        q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        block_mask2 = create_block_mask(mask, None, None, 2048, 2048, device=device)
        # Reuse the 1st version with q/k/v(seqlen=2048) and block_mask(seqlen=2048)
        out2 = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            q, k, v, block_mask=block_mask2
        )
        out2.sum().backward()
        q_grad2, k_grad2, v_grad2 = q.grad, k.grad, v.grad
        tolerance = Tolerances(atol=1e-3, rtol=1e-3)

        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(
            q_grad, q_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            k_grad, k_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            v_grad, v_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_float32_matmul_precision(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): with highest float32 matmul
        # precision (temp context), eager and compiled attention grads on constant inputs must
        # be close.
        make_tensor = functools.partial(
            torch.zeros,
            (2, 2, 128, 32),
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query.fill_(0.2)
        key.fill_(0.3)
        value.fill_(0.4)

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: doubles the score.
            return score * 2

        with temp_float32_matmul_precision("highest"):
            out_eager = flex_attention(query, key, value, score_mod)
            flex_compiled = torch.compile(flex_attention, fullgraph=True)
            out_compiled = flex_compiled(query, key, value, score_mod)

            grads_eager = torch.autograd.grad(out_eager.sum(), (query, key, value))
            grads_compile = torch.autograd.grad(out_compiled.sum(), (query, key, value))

        torch.testing.assert_close(grads_eager, grads_compile)

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("score_mod_name", ["_head_offset"])
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_captured_score_mod_aot_eager_gradcheck(
        self, device, score_mod_name: str, mode: str
    ):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): gradcheck with a captured
        # per-head buffer score_mod, in eager or aot_eager compile mode (float64; float32 on MPS).
        # See test_aot_eager_gradcheck for the per-device dtype rationale.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend=mode, fullgraph=True)
        score_mod = captured_buffers_map[score_mod_name](dtype, device)

        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_document_masking_edge_case(self, device, mode):
        # [FUNCTION DOC] Non-CPU test: document-mask block mask over two documents compiled
        # with the given backend (eager or aot_eager); runs forward (and backward if
        # supported). No numerics.
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        document_masks = torch.full((2, 128), 0, dtype=torch.int32, device=device)
        document_masks[:, 64:] = 1

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where q and kv belong to the same document.
            same_doc = document_masks[b, q] == document_masks[b, kv]
            return same_doc

        # MPS doesn't support fp64; this test has no numerical assertions.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 1, 128, 4),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        func = torch.compile(flex_attention, backend=mode, fullgraph=True)

        block_mask = create_block_mask(mask_mod, 2, 1, 128, 128, device=device)
        out = func(query, key, value, block_mask=block_mask)
        if requires_grad:
            out.sum().backward()

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_strided_backwards(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): attention with sliced (strided)
        # key and value, compiled dynamic; checks kernel names in forward and backward code and
        # that grads match eager within 9e-3 absolute.
        shape = (1, 2, 4096, 64)
        Q = torch.randn(shape, requires_grad=True, device=device)
        K = torch.randn(shape, requires_grad=True, device=device)
        V = torch.randn(shape, requires_grad=True, device=device)
        func = torch.compile(flex_attention, dynamic=True, fullgraph=True)

        K_sliced = K[:, :, :-128]
        V_sliced = V[:, :, :-128]

        out_eager = flex_attention(Q, K_sliced, V_sliced)

        out_compiled, code = run_and_get_code(func, Q, K_sliced, V_sliced)

        # Make sure flex attention kernels have flex_attention in name
        FileCheck().check_regex("triton_tem_fused_flex_attention.*").run(code[0])
        FileCheck().check_regex("triton_tem_fused_flex_attention_backward.*").run(
            code[1]
        )

        grad = torch.rand_like(out_eager)

        eager_grads = torch.autograd.grad(out_eager, (Q, K, V), grad)
        compiled_grads = torch.autograd.grad(out_compiled, (Q, K, V), grad)

        for eager, compiled in zip(eager_grads, compiled_grads):
            torch.testing.assert_close(eager, compiled, atol=9e-3, rtol=0)

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "inductor", "paged_attention"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),  # Default order
            (1, 0, 2, 3),  # Swap first two dims
            (0, 2, 1, 3),  # Mixed order
            (2, 0, 1, 3),  # Another mixed order
            (0, 1, 3, 2),  # Swap last two dims (non-contiguous last dim)
            (1, 0, 3, 2),  # Swap first two AND last two dims
        ],
    )
    @common_utils.parametrize("shape", [(2, 1, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_stride_ordering(self, device, mode, permute_order, shape):
        # [FUNCTION DOC] Non-CPU test: permutes q, k, v dims; runs eager, inductor or paged mode
        # and asserts the output last stride is 1 and, when the query last stride is 1, that the
        # stride order of output equals that of the query. Paged mode skipped on MPS.
        from torch._inductor.ir import get_stride_order

        if mode == "paged_attention" and _is_mps_device(device):
            self.skipTest(
                "paged attention captures the page table; not supported on MPS"
            )

        dtype = torch.float32
        # Setup
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.randn,
            shape,
            device=device,
            dtype=dtype,
            requires_grad=False if mode == "paged_attention" else requires_grad,
        )

        # Create and permute tensors
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query = query.permute(permute_order)
        key = key.permute(permute_order)
        value = value.permute(permute_order)

        if mode == "inductor":
            func = torch.compile(flex_attention, backend=mode, fullgraph=True)
            out = func(query, key, value)
        elif mode == "paged_attention":
            out, _ = self.run_paged_attention(
                _identity, query, key, value, dtype, device=device
            )
        else:
            func = flex_attention
            out = func(query, key, value)

        # We always enforce stride[-1]=1 for attention kernels
        self.assertEqual(out.stride()[-1], 1, "Output must have contiguous last dim")

        # When input already has stride[-1]=1, verify full stride order is preserved
        if query.stride()[-1] == 1:
            out_stride_order = get_stride_order(out.stride())
            query_stride_order = get_stride_order(query.stride())
            self.assertEqual(
                out_stride_order,
                query_stride_order,
                lambda msg: f"{msg}\nStride order mismatch: out {out_stride_order}, query {query_stride_order}",
            )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "inductor"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 2, 1, 3),
            (2, 0, 1, 3),
            (0, 1, 3, 2),
            (1, 0, 3, 2),
        ],
    )
    @common_utils.parametrize("shape", [(2, 5, 128, 16), (4, 2, 64, 16)])
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_flex_attention_backward_stride_ordering(
        self, device, mode, permute_order, shape
    ):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): same permutations with
        # backward; grads must exist, have no NaN and, when the input last stride is 1, keep its
        # stride order.
        from torch._inductor.ir import get_stride_order

        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn, shape, device=device, dtype=dtype, requires_grad=False
        )

        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query = query.permute(permute_order)
        key = key.permute(permute_order)
        value = value.permute(permute_order)

        query.requires_grad_()
        key.requires_grad_()
        value.requires_grad_()

        func = (
            torch.compile(flex_attention, backend=mode, fullgraph=True)
            if mode == "inductor"
            else flex_attention
        )
        out = func(query, key, value)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)

        for leaf, grad, name in [
            (query, query.grad, "query"),
            (key, key.grad, "key"),
            (value, value.grad, "value"),
        ]:
            self.assertIsNotNone(
                grad, lambda msg: f"{msg}\nGrad {name} should be computed"
            )
            self.assertFalse(
                torch.isnan(grad).any(), lambda msg: f"{msg}\nGrad {name} contains NaN"
            )

            # When input has stride[-1]=1, verify stride order is preserved
            if leaf.stride()[-1] == 1:
                input_stride_order = get_stride_order(grad.stride())
                orig_stride_order = get_stride_order(leaf.stride())
                self.assertEqual(
                    input_stride_order,
                    orig_stride_order,
                    lambda msg: f"{msg}\nMode: {mode}, Stride order mismatch for {name}: grad {input_stride_order}, input {orig_stride_order}.",
                )

    @supported_platform
    def test_non_contiguous_last_dim(self, device):
        # [FUNCTION DOC] Inputs whose last two dims are stored column-major (last stride not
        # 1), for sequence lengths 16 and 64, compiled dynamic vs eager and golden, with grads
        # when supported.
        """Test flex_attention with tensors having non contiguous last dimension."""
        B, H, D = 4, 8, 64
        dtype = torch.float16 if device in DEVICE_SUPPORTS_BACKWARDS else torch.float32
        for S in [16, 64]:

            def column_major_tensor():
                # [FUNCTION DOC] HELPER: random tensor made column-major in its last two dims.
                tensor = torch.randn(
                    (B, H, S, D),
                    dtype=dtype,
                    device=device,
                )
                # Column major in last 2 dims
                return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)

            q = column_major_tensor()
            k = column_major_tensor()
            v = column_major_tensor()

            requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
            if requires_grad:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)

            self.assertNotEqual(q.stride()[-1], 1)
            self.assertNotEqual(k.stride()[-1], 1)
            self.assertNotEqual(v.stride()[-1], 1)

            q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
            q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

            golden_out = flex_attention(q_gold, k_gold, v_gold)
            ref_out = flex_attention(q_ref, k_ref, v_ref)

            flex_compiled = torch.compile(flex_attention, fullgraph=True, dynamic=True)
            compiled_out = flex_compiled(q, k, v)

            self._check_out(golden_out, ref_out, compiled_out)

            if requires_grad:
                backward_grad = torch.randn_like(ref_out)

                golden_out.backward(backward_grad.to(torch.float64))
                ref_out.backward(backward_grad)
                compiled_out.backward(backward_grad)

                self._check_out_and_grad(
                    golden_out,
                    ref_out,
                    compiled_out,
                    q_gold,
                    q_ref,
                    q,
                    k_gold,
                    k_ref,
                    k,
                    v_gold,
                    v_ref,
                    v,
                )

    @supported_platform
    @common_utils.parametrize("compile", [True, False])
    def test_fully_masked_out_rows_0_check(self, device, compile: bool):
        # [FUNCTION DOC] Rows entirely masked out (q index from half the sequence) must not
        # produce NaNs: output zero there, lse negative infinity and q grad zero (when grads are
        # supported), in eager or compiled mode.
        # Ensure fully masked out rows won't cause NaNs.
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        query = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        key = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        value = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )

        M = S // 2

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true for the first half of the queries.
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)

        flex = (
            torch.compile(flex_attention, dynamic=False) if compile else flex_attention
        )
        if requires_grad:
            out, lse = flex(query, key, value, block_mask=block_mask, return_lse=True)
            self.assertEqual(out[:, :, M:, :].sum(), 0)
            self.assertTrue((lse[:, :, M:] == -float("inf")).all())

            loss = out.sum() + lse.sum()
            loss.backward()
            self.assertEqual(query.grad[:, :, M:, :].sum(), 0)
        else:
            out = flex(query, key, value, block_mask=block_mask, return_lse=False)

        self.assertEqual(out[:, :, M:, :].sum(), 0)

    @supported_platform
    def test_fully_masked_out_rows(self, device):
        # [FUNCTION DOC] Fully masked rows in run_test comparison with a score_mod that passes
        # scores through, against the golden reference.
        M = S // 2

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true for the first half of the queries.
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)

        def noop_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: score_mod returning the score unchanged.
            return score

        self.run_test(
            noop_mod, torch.float32, device, B, H, S, D, B, H, S, D, block_mask
        )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton "BLOCK_M : tl.constexpr" string in generated code
    def test_kernel_options_argument_is_respected(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: user kernel option for block M of 16 must appear in the
        # generated Triton code.
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        # Ensure we respect user's input kernel options.
        _, code = run_and_get_code(
            torch.compile(flex_attention, fullgraph=True),
            q,
            k,
            v,
            kernel_options={"BLOCK_M": 16},
        )
        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_backward_kernel_options_filtering(self, device):
        # [FUNCTION DOC] Non-CPU/MPS/XPU test (bf16 required): forward kernel options for
        # block sizes passed together must be filtered for backward without error; checks the
        # q grad shape.
        if not PLATFORM_SUPPORTS_BF16:
            self.skipTest("bf16 is required for this regression test")

        b, h, n, c = 1, 8, 234, 16
        block_size = 32

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where q > kv.
            return q > kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=n,
            KV_LEN=n,
            device=device,
            BLOCK_SIZE=block_size,
        )
        kernel_options = {
            "BLOCK_M": block_size,
            "BLOCK_M1": block_size,
            "BLOCK_M2": block_size,
            "BLOCK_N": block_size,
            "BLOCK_N1": block_size,
            "BLOCK_N2": block_size,
        }
        make_tensor = functools.partial(
            torch.randn,
            (b, h, n, c),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out = torch.compile(flex_attention)(
            query, key, value, block_mask=block_mask, kernel_options=kernel_options
        )
        out.mean().backward()

        self.assertEqual(query.grad.shape, query.shape)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_forward_kernel_options_error(self, device):
        # [FUNCTION DOC] Non-CPU/MPS/XPU test: forward kernel options that are inconsistent
        # (Triton backend with block sizes only) raise InductorError or ValueError whose message
        # names the invalid forward options.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=64,
            KV_LEN=64,
            device=device,
            BLOCK_SIZE=32,
        )
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 64, 16),
            device=device,
            dtype=torch.float16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention forward kernel options.*fwd_BLOCK_M.*fwd_num_stages",
        ):
            torch.compile(flex_attention)(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BACKEND": "TRITON", "BLOCK_M": 64, "BLOCK_N": 32},
            )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_backward_kernel_options_error(self, device):
        # [FUNCTION DOC] Non-CPU/MPS/XPU test: backward kernel options with oversize block values
        # raise InductorError or ValueError naming invalid backward options at backward time.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=64,
            KV_LEN=64,
            device=device,
            BLOCK_SIZE=32,
        )
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 64, 16),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out = torch.compile(flex_attention)(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "bwd_BLOCK_M1": 64,
                "bwd_BLOCK_N1": 64,
                "bwd_BLOCK_M2": 64,
                "bwd_BLOCK_N2": 64,
            },
        )
        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention backward kernel options.*"
            "bwd_BLOCK_M1.*bwd_num_stages",
        ):
            out.mean().backward()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_decode_kernel_options_error(self, device):
        # [FUNCTION DOC] Non-CPU/MPS/XPU test: single-query decode with invalid decode kernel
        # options for the decode backend raises InductorError or ValueError naming invalid
        # decode options.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=1,
            KV_LEN=256,
            device=device,
            BLOCK_SIZE=128,
        )
        query = torch.randn(1, 2, 1, 64, device=device, dtype=torch.float16)
        key = torch.randn(1, 2, 256, 64, device=device, dtype=torch.float16)
        value = torch.randn(1, 2, 256, 64, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention decode kernel options.*fwd_BLOCK_M.*fwd_num_stages",
        ):
            torch.compile(flex_attention)(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={
                    "BACKEND": "TRITON_DECODE",
                    "fwd_BLOCK_M": 96,
                    "fwd_BLOCK_N": 64,
                },
            )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_flex_flash_honors_sparse_metadata_with_trivial_mask(self, device):
        # [FUNCTION DOC] Non-CPU/MPS/XPU test (needs flash-attn-4 and SM100+ GPU): the FLASH
        # backend must honor BlockMask sparse metadata (both full-block and partial-block forms,
        # top-1 block per query block) without using the mask_mod; output compared with a dense
        # masked softmax reference at 0.05 tolerance.
        """FLASH should honor BlockMask metadata without relying on mask_mod."""
        from torch._inductor.kernel.flex.flex_flash_attention import (
            ensure_flash_available,
        )

        if not ensure_flash_available():
            self.skipTest("flash-attn-4 is not available")
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("FA4 sparse block path in this test targets SM100+")

        torch.manual_seed(0)
        batch = 1
        heads = 1
        q_blocks = 2
        kv_blocks = 2
        q_block_size = 256
        kv_block_size = 128
        head_dim = 128
        top_k = 1
        q = torch.randn(
            batch,
            heads,
            q_blocks * q_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            batch,
            heads,
            kv_blocks * kv_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch,
            heads,
            kv_blocks * kv_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        selected = torch.tensor(
            [[[[0], [1]]]],
            device=device,
            dtype=torch.int32,
        )
        num_blocks = torch.full(
            selected.shape[:-1], top_k, device=device, dtype=torch.int32
        )
        zero_num_blocks = torch.zeros_like(num_blocks)
        zero_indices = torch.zeros(
            *selected.shape[:-1], kv_blocks, device=device, dtype=torch.int32
        )
        sparse_indices = torch.zeros_like(zero_indices)
        sparse_indices[..., :top_k] = selected

        block_map = torch.zeros(
            batch, heads, q_blocks, kv_blocks, device=device, dtype=torch.bool
        )
        block_map.scatter_(-1, selected.long(), True)
        dense_mask = block_map.repeat_interleave(q_block_size, -2).repeat_interleave(
            kv_block_size, -1
        )
        ref = (
            torch.softmax(
                (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).masked_fill(
                    ~dense_mask, -float("inf")
                ),
                dim=-1,
            )
            @ v
        )

        block_masks = {
            "full": BlockMask.from_kv_blocks(
                kv_num_blocks=zero_num_blocks,
                kv_indices=zero_indices,
                full_kv_num_blocks=num_blocks,
                full_kv_indices=sparse_indices,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                seq_lengths=(q_blocks * q_block_size, kv_blocks * kv_block_size),
            ),
            "partial": BlockMask.from_kv_blocks(
                kv_num_blocks=num_blocks,
                kv_indices=sparse_indices,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                seq_lengths=(q_blocks * q_block_size, kv_blocks * kv_block_size),
            ),
        }
        flash = torch.compile(
            functools.partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
        )
        for name, block_mask in block_masks.items():
            with self.subTest(name=name):
                out = flash(q, k, v, block_mask=block_mask)
                torch.cuda.synchronize()
                self.assertEqual(out, ref, atol=5e-2, rtol=5e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    def test_backend_auto_matches_triton_large(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: on large shapes, compiled attention with the AUTO
        # backend must give bit-identical output to the TRITON backend, and the generated code of
        # both must mention flex_attention and not flex_decoding.
        """BACKEND='AUTO' should follow Triton heuristics on large shapes."""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 256, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        def compile_and_run(kernel_options):
            # [FUNCTION DOC] HELPER: compiles attention with the given kernel options and returns
            # the output with generated code.
            return run_and_get_code(
                torch.compile(flex_attention, fullgraph=True),
                q,
                k,
                v,
                kernel_options=kernel_options,
            )

        default_out, default_code = compile_and_run({"BACKEND": "AUTO"})
        triton_out, triton_code = compile_and_run({"BACKEND": "TRITON"})

        torch.testing.assert_close(default_out, triton_out, atol=0.0, rtol=0.0)

        default_src = "\n".join(default_code)
        FileCheck().check("flex_attention").check_not("flex_decoding").run(default_src)

        triton_src = "\n".join(triton_code)
        FileCheck().check("flex_attention").check_not("flex_decoding").run(triton_src)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # exercises Triton flex_decoding kernel; not present on MPS
    def test_backend_triton_decode_matches_auto(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: on a decode-friendly shape, the AUTO backend and the
        # explicit TRITON_DECODE backend must both call the flex decoding kernel creator (observed
        # through a wrapping mock); outputs have the expected shape and agree within 3e-3.
        """BACKEND='TRITON_DECODE' should match heuristics on decode-friendly shapes."""
        make_tensor = functools.partial(
            torch.randn,
            (1, 2, 64, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        def compile_and_run(kernel_options):
            # [FUNCTION DOC] HELPER: compiles attention with the given kernel options and returns
            # output and generated code.
            return run_and_get_code(
                torch.compile(flex_attention, fullgraph=True),
                q,
                k,
                v,
                kernel_options=kernel_options,
            )

        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        with mock.patch.object(
            flex_kernel_mod,
            "create_flex_decoding_kernel",
            wraps=flex_kernel_mod.create_flex_decoding_kernel,
        ) as decode_kernel:
            default_out, _ = compile_and_run({"BACKEND": "AUTO"})
            self.assertTrue(
                decode_kernel.called,
                "Expected heuristics to dispatch to flex decoding kernel.",
            )

        with mock.patch.object(
            flex_kernel_mod,
            "create_flex_decoding_kernel",
            wraps=flex_kernel_mod.create_flex_decoding_kernel,
        ) as decode_kernel:
            decode_out, _ = compile_and_run({"BACKEND": "TRITON_DECODE"})
            self.assertTrue(
                decode_kernel.called,
                "Expected explicit BACKEND='TRITON_DECODE' to use flex decoding kernel.",
            )

        self.assertEqual(decode_out.shape, (1, 2, 64, 64))
        torch.testing.assert_close(default_out, decode_out, atol=3e-3, rtol=3e-3)

    def test_unbacked_flex_decoding_eligibility_falls_back(self, device):
        # [FUNCTION DOC] Builds a fake graph context whose size allocator has an unbacked sequence
        # length symbol with a hint override, and asserts that the flex decoding eligibility check
        # returns False for such shapes (it falls back).
        from torch._inductor.kernel.flex.flex_decoding import _use_flex_decoding
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.virtualized import V
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class FakeBuffer:
            # [CLASS DOC] HELPER: stand-in for an IR buffer that only reports a given size.
            def __init__(self, size):
                # [FUNCTION DOC] HELPER: stores the size.
                self.size = size

            def get_size(self):
                # [FUNCTION DOC] HELPER: returns the stored size.
                return self.size

        shape_env = ShapeEnv()
        seq_len = shape_env.create_unbacked_symint().node.expr
        shape_env.var_to_hint_override[seq_len] = 64
        graph = mock.Mock(sizevars=SizeVarAllocator(shape_env))

        with V.set_graph_handler(graph):
            self.assertFalse(
                _use_flex_decoding(
                    FakeBuffer([1, 2, seq_len, 64]),
                    FakeBuffer([1, 1, 1, 1]),
                    FakeBuffer([1, 2, seq_len, 64]),
                    {},
                    False,
                )
            )

    def test_backward_fake_symbolic_query_key_batch_metadata(self, device):
        # [FUNCTION DOC] Checks the fake-tensor implementation of the flex attention backward for
        # symbolic batch sizes: shared unbacked batch gives grads with the key/value shapes;
        # transposed-batch key/value (batch 1) keep the strides of a permuted empty tensor; query
        # batch larger than a batch-1 key/value gives summed (broadcast) shapes and strides;
        # mismatched batch sizes raise RuntimeError containing the text batch must match.
        from torch._dynamo.source import LocalSource
        from torch._higher_order_ops.flex_attention import (
            _permute_strides,
            flex_attention_backward_fake_tensor_mode,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)

        def fakeify(t, name, shape_id=None):
            # [FUNCTION DOC] HELPER: converts a real tensor to a fake tensor in the shared fake
            # mode with static sizes, except dim 0 becomes unbacked and shares an id when shape_id
            # is given.
            dynamic_sizes = [DimDynamic.STATIC] * t.dim()
            shape_ids = {}
            if shape_id is not None:
                dynamic_sizes[0] = DimDynamic.UNBACKED
                shape_ids[0] = shape_id
            return fake_mode.from_tensor(
                t,
                source=LocalSource(name, is_input=True),
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=dynamic_sizes,
                    shape_ids=shape_ids,
                ),
            )

        B, H, S, D = 4, 2, 8, 16

        def run(query, key, value, out, logsumexp, grad_out, grad_logsumexp):
            # [FUNCTION DOC] HELPER: runs the fake backward under the fake mode and returns the
            # key and value gradients.
            with fake_mode:
                _, grad_key, grad_value, _ = flex_attention_backward_fake_tensor_mode(
                    query,
                    key,
                    value,
                    out,
                    logsumexp,
                    grad_out,
                    grad_logsumexp,
                    None,
                    None,
                    (),
                    1.0,
                    {},
                )
            return grad_key, grad_value

        def make_fake_inputs(Bq, Bkv, *, shared_batch=False, key_value_factory=None):
            # [FUNCTION DOC] HELPER: builds fake query, key, value, out, logsumexp and grads for
            # the given batch sizes (optionally sharing one unbacked batch symbol and using a
            # factory for the key/value layouts); also returns the real key and value tensors.
            shape_id = "batch" if shared_batch else None
            query = fakeify(torch.empty(Bq, H, S, D, device=device), "query", shape_id)
            if key_value_factory is None:
                key_real = torch.empty(Bkv, H, S, D, device=device)
                value_real = torch.empty(Bkv, H, S, D, device=device)
            else:
                key_real = key_value_factory(Bkv)
                value_real = key_value_factory(Bkv)
            key = fakeify(key_real, "key", shape_id)
            value = fakeify(value_real, "value", shape_id)
            out = fakeify(torch.empty(Bq, H, S, D, device=device), "out", shape_id)
            grad_out = fakeify(
                torch.empty(Bq, H, S, D, device=device), "grad_out", shape_id
            )
            logsumexp = fakeify(
                torch.empty(Bq, H, S, device=device), "logsumexp", shape_id
            )
            grad_logsumexp = fakeify(
                torch.empty(Bq, H, S, device=device), "grad_logsumexp", shape_id
            )
            return (
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                grad_logsumexp,
                key_real,
                value_real,
            )

        args = make_fake_inputs(B, B, shared_batch=True)
        grad_key, grad_value = run(*args[:7])

        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)

        def transposed_batch_tensor(B):
            # [FUNCTION DOC] HELPER: creates an empty tensor of the given batch whose batch
            # dimension is physically last, permuted back to batch-first.
            return torch.empty(H, S, D, B, device=device).permute(3, 0, 1, 2)

        args = make_fake_inputs(1, 1, key_value_factory=transposed_batch_tensor)
        grad_key, grad_value = run(*args[:7])
        key_real, value_real = args[7:]
        expected_key = _permute_strides(
            key_real.new_empty((1, H, S, D)), key_real.stride()
        )
        expected_value = _permute_strides(
            value_real.new_empty((1, H, S, D)), value_real.stride()
        )
        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)
        self.assertEqual(grad_key.stride(), expected_key.stride())
        self.assertEqual(grad_value.stride(), expected_value.stride())

        args = make_fake_inputs(B, 1, key_value_factory=transposed_batch_tensor)
        grad_key, grad_value = run(*args[:7])
        key_real, value_real = args[7:]
        expected_key = torch.sum(
            _permute_strides(key_real.new_empty((B, H, S, D)), key_real.stride()),
            dim=0,
            keepdim=True,
        )
        expected_value = torch.sum(
            _permute_strides(value_real.new_empty((B, H, S, D)), value_real.stride()),
            dim=0,
            keepdim=True,
        )
        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)
        self.assertEqual(grad_key.stride(), expected_key.stride())
        self.assertEqual(grad_value.stride(), expected_value.stride())

        args = make_fake_inputs(B, 2)
        with self.assertRaisesRegex(RuntimeError, "batch must match"):
            run(*args[:7])

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific BACKEND='TRITON_DECODE' error
    def test_backend_triton_decode_errors_when_not_supported(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: TRITON_DECODE on shapes decode does not support raises
        # RuntimeError saying flex_decoding cannot be used.
        """Requesting decode on unsupported shapes should raise a helpful error."""
        make_tensor = functools.partial(
            torch.randn,
            (1, 2, 256, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
        ):
            flex_compiled(q, k, v, kernel_options={"BACKEND": "TRITON_DECODE"})

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific BACKEND='TRITON_DECODE' error
    def test_backend_triton_decode_errors_with_non_power_of_two_gqa(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: TRITON_DECODE with a GQA ratio of three (not a power of
        # two) raises RuntimeError saying flex_decoding cannot be used.
        """BACKEND='TRITON_DECODE' should fail when GQA ratio is not a power of two."""
        q = torch.randn(
            1, 3, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )
        k = torch.randn(
            1, 1, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )
        v = torch.randn(
            1, 1, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
        ):
            flex_compiled(
                q,
                k,
                v,
                enable_gqa=True,
                kernel_options={"BACKEND": "TRITON_DECODE"},
            )

    @supported_platform
    @skip_on_cpu
    def test_backend_rejects_legacy_force_use_flag(self, device):
        # [FUNCTION DOC] Non-CPU test: giving BACKEND together with the legacy force flag raises
        # RuntimeError saying they cannot be combined.
        """Combining BACKEND with FORCE_USE_FLEX_ATTENTION should raise an error."""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND cannot be combined with legacy FORCE_USE_FLEX_ATTENTION",
        ):
            flex_compiled(
                q,
                k,
                v,
                kernel_options={
                    "BACKEND": "TRITON",
                    "FORCE_USE_FLEX_ATTENTION": True,
                },
            )

    @supported_platform
    def test_backend_defaults_and_rejects_invalid(self, device):
        # [FUNCTION DOC] Checks that applying empty kernel options sets BACKEND to AUTO and that
        # an invalid BACKEND value raises ValueError mentioning the invalid value.
        device = torch.device(device)
        query = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)
        key = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)
        value = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)

        kernel_options = _apply_kernel_options(
            query, key, value, return_lse=True, kernel_options={}
        )
        self.assertEqual(kernel_options["BACKEND"], "AUTO")

        with self.assertRaisesRegex(ValueError, r"Invalid BACKEND value 'INVALID'"):
            _apply_kernel_options(
                query,
                key,
                value,
                return_lse=True,
                kernel_options={"BACKEND": "INVALID"},
            )

    @supported_platform
    @skip_on_cpu
    def test_block_mask_non_divisible(self, device):
        # [FUNCTION DOC] Non-CPU test: sequence of 1023 (not divisible by block size) with a
        # segment-id mask_mod; block mask creation works eagerly and compiled, and attention
        # matches reference.
        seq = torch.arange(1023, device=device) // 128

        def mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where q and kv positions fall in the
            # same 128-block.
            return seq[q] == seq[kv]

        block_mask = create_block_mask(mod, None, None, 1023, 1023, device=device)
        torch.compile(create_block_mask)(mod, None, None, 1023, 1023, device=device)
        self.run_test_with_call(
            lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask),
            torch.float16,
            device,
            Q_S=1023,
            KV_S=1023,
        )

    @supported_platform
    def test_causal_block_non_divisible(self, device):
        # [FUNCTION DOC] Causal block mask on length S minus one (not block divisible) checked
        # through run_test_with_call.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S - 1, S - 1, device=device)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, torch.float16, device, Q_S=S - 1, KV_S=S - 1)

    @supported_platform
    @skip_on_cpu
    def test_modular_indexing(self, device):
        # [FUNCTION DOC] Non-CPU test: compiles a module whose score_mod indexes a captured bias
        # reshaped from a doubled view (large batch 100, bf16) to exercise modular indexing; only
        # checks it runs.
        B, H, N, D = 100, 12, 128, 64
        dtype = torch.bfloat16
        device = torch.device(device)

        class Attention(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding a random bias and running flex_attention with a
            # score_mod built from it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the random bias tensor on the device.
                super().__init__()
                self.bias = torch.randn(B, N, N, H, device=device, dtype=dtype)

            def forward(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: builds the score_mod from the bias and returns
                # flex_attention output.
                score_mod = generate_score_mod(self.bias)
                o = flex_attention(q, k, v, score_mod=score_mod)
                return o

        def generate_score_mod(bias):
            # [FUNCTION DOC] HELPER: doubles and reshapes the bias, returning a score_mod that
            # adds the bias entry for batch, head, q and kv.
            bias = (2 * bias).view(B, H, N, N).contiguous()

            def score_mod(score, batch, head, q_idx, k_idx):
                # [FUNCTION DOC] HELPER: adds the captured bias entry to the score.
                attn_bias = bias[batch, head, q_idx, k_idx]
                return score + attn_bias

            return score_mod

        m = Attention().to(device).eval().to(dtype)
        m = torch.compile(m, mode="default", fullgraph=False)

        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        m(q, k, v)

    @supported_platform
    @skip_on_cpu
    def test_force_write_lse(self, device):
        # [FUNCTION DOC] Non-CPU test: lse from eager, compiled and (when backward is supported)
        # paged attention agree within 3e-3.
        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        out_eager, lse_eager = flex_attention(query, key, value, return_lse=True)

        flex_compile = torch.compile(flex_attention)
        out_compiled, lse_compiled = flex_compile(query, key, value, return_lse=True)

        out_paged, lse_paged = self.run_paged_attention(
            score_mod=_identity, q=query, k=key, v=value, dtype=dtype, device=device
        )

        torch.testing.assert_close(lse_eager, lse_compiled, atol=3e-3, rtol=0)
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            torch.testing.assert_close(lse_eager, lse_paged, atol=3e-3, rtol=0)

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("backend", ["flex_attention", "flex_decode", "eager"])
    @expected_not_implemented_on_mps
    def test_lse_masked_output(self, device, backend):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): combines sliding-window and
        # global causal attention results via their logsumexp weights; the merged output and grads
        # must equal plain causal SDPA within tolerances. Parametrized by backend: forced flex
        # attention, flex decode (shorter context) or eager.
        if backend == "flex_decode":
            kernel_options = {"FORCE_USE_FLEX_ATTENTION": False}
            flex_call = torch.compile(flex_attention, fullgraph=True)
            N_CTX = 96
        elif backend == "flex_attention":
            kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
            flex_call = torch.compile(flex_attention, fullgraph=True)
            N_CTX = 196
        else:
            kernel_options = {}
            flex_call = flex_attention
            N_CTX = 196

        SLIDING_WINDOW = 64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, N_CTX, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def sliding_window_causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod for causal attention within the sliding window.
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= SLIDING_WINDOW
            return causal_mask & window_mask

        def global_causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod for causal positions beyond the sliding window.
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx > SLIDING_WINDOW
            return causal_mask & window_mask

        sliding_window_causal = torch.nn.attention.flex_attention.create_block_mask(
            sliding_window_causal,
            B=None,
            H=None,
            Q_LEN=N_CTX,
            KV_LEN=N_CTX,
            device=device,
        )
        global_causal = torch.nn.attention.flex_attention.create_block_mask(
            global_causal, B=None, H=None, Q_LEN=N_CTX, KV_LEN=N_CTX, device=device
        )

        local_attn = functools.partial(
            flex_call,
            block_mask=sliding_window_causal,
            return_lse=True,
            kernel_options=kernel_options,
        )
        global_attn = functools.partial(
            flex_call,
            block_mask=global_causal,
            return_lse=True,
            kernel_options=kernel_options,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        gradOut = make_tensor(requires_grad=False)

        x_local, lse_local = local_attn(q, k, v)
        x_global, lse_global = global_attn(q, k, v)

        max_lse = torch.maximum(lse_local, lse_global)
        lse_global = lse_global - max_lse
        lse_local = lse_local - max_lse
        lse_global = torch.exp(lse_global)
        lse_local = torch.exp(lse_local)
        x = ((x_local * lse_local[..., None]) + (x_global * lse_global[..., None])) / (
            lse_global[..., None] + lse_local[..., None]
        )
        x.backward(gradOut)
        flex_q_grad, flex_k_grad, flex_v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(gradOut)

        torch.testing.assert_close(x, out, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_q_grad, q.grad, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_k_grad, k.grad, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_v_grad, v.grad, atol=3e-3, rtol=2e-3)

    @supported_platform
    @skip_on_cpu
    def test_mixed_device_error_message(self, device):
        # [FUNCTION DOC] Non-CPU test: key on GPU but query and value on CPU makes flex_attention
        # raise ValueError with the same-device-type message.
        # Create tensors on different devices
        cpu_tensor = torch.randn(2, 2, 128, 16, device="cpu")
        gpu_tensor = torch.randn(2, 2, 128, 16, device=device)

        # Use different devices for query, key, and value
        query, key, value = cpu_tensor, gpu_tensor, cpu_tensor

        expected_error_message = (
            "Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )

        with self.assertRaisesRegex(ValueError, expected_error_message):
            flex_attention(query, key, value)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific "Buffers cannot be created" message
    def test_captured_wrong_device_error_message(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: a score_mod capturing a CPU buffer while q/k/v are on
        # the GPU raises RuntimeError saying buffers cannot be created.
        means = torch.randn(64, 3, device=device)
        length_scales = torch.logspace(0.001, 0.1, 8, device="cpu")

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            # [FUNCTION DOC] HELPER: score_mod weighting the score by exp of minus the distance
            # between captured positions divided by a captured per-head scale.
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device=device) for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific "Buffers cannot be created" message
    def test_cant_lower_error_message(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: a score_mod containing a 256-wide reduction over
        # captured buffers cannot be lowered; compile raises RuntimeError saying buffers cannot
        # be created.
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(64, 256, device=device)
        length_scales = torch.logspace(0.001, 0.1, 8, device=device)

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            # [FUNCTION DOC] HELPER: distance-based score_mod using a 256-element
            # coordinate reduction.
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device=device) for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # logspace out not implemented
    def test_reduction_unrolled(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a distance-based score_mod with a
        # 3-element reduction compiles and matches reference via run_test in bfloat16.
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(S, 3, device=device)
        length_scales = torch.logspace(0.001, 0.1, H, device=device)

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            # [FUNCTION DOC] HELPER: distance-based score_mod using captured positions and
            # head scales.
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        self.run_test(euclidean_dist_pos_embed, torch.bfloat16, device=device)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # exercises the Triton default-config tile clamping path
    def test_narrow_kv_block_size_uses_clamped_default_config(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: with sparse block size (128, 64), the default kernel
        # config must clamp tile sizes and not error; compiled output and grads match eager
        # within 0.02.
        """The single default config must clamp its tiles to fit a narrower KV
        sparse block size instead of erroring, in both forward and backward."""
        q, k, v = (
            torch.randn(
                2, 8, 2048, 128, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        )
        block_mask = create_block_mask(
            _causal_mask, None, None, 2048, 2048, BLOCK_SIZE=(128, 64), device=device
        )
        out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
        ref = flex_attention(q, k, v, block_mask=block_mask)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

        grad_out = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=True)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out, retain_graph=True)
        for g, rg in zip(grads, ref_grads):
            torch.testing.assert_close(g, rg, atol=2e-2, rtol=2e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # exercises the Triton IS_DIVISIBLE sparse-block guard
    def test_sparse_block_not_dividing_seqlen(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: sparse blocks of 256 over length 384 must not read past
        # the KV length; compiled output and grads match eager within 0.02.
        """Sparse blocks that overhang a 128-divisible seq len must not be
        walked unmasked past KV_LEN (silent OOB corruption)."""

        def tail_only(b, h, m, n):
            # [FUNCTION DOC] HELPER: mask_mod keeping only kv positions from 256 on.
            return n >= 256

        S = 384
        q, k, v = (
            torch.randn(2, 4, S, 64, device=device, requires_grad=True)
            for _ in range(3)
        )
        block_mask = create_block_mask(
            tail_only, None, None, S, S, BLOCK_SIZE=256, device=device
        )
        out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
        ref = flex_attention(q, k, v, block_mask=block_mask)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

        grad_out = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=True)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out, retain_graph=True)
        for g, rg in zip(grads, ref_grads):
            torch.testing.assert_close(g, rg, atol=2e-2, rtol=2e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific BLOCK_M/BLOCK_N divisibility error
    def test_invalid_block_size(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: sparse block size 96 is not divisible by the selected
        # tiles, so compile raises RuntimeError with the invalid forward kernel options message.
        # Create tensors on different devices
        q, k, v = (torch.randn(1, 8, 128, 64, device=device) for _ in range(3))

        expected_error_message = (
            "Invalid FlexAttention forward kernel options: Q and KV block sizes "
            "must be divisible by the selected tile sizes.*"
            "SPARSE_Q_BLOCK_SIZE=96.*SPARSE_KV_BLOCK_SIZE=96.*"
            "BLOCK_M=\\d+.*BLOCK_N=\\d+"
        )
        block_mask = create_block_mask(
            noop_mask, 1, 8, 128, 128, BLOCK_SIZE=96, device=device
        )

        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward
    def test_small_q_kv_len(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): length-one queries and keys of
        # ones with the forced flex attention path; compiled output and lse equal eager exactly
        # and grads agree.
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 1, 16),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
        out_eager, lse_eager = flex_attention(
            query, key, value, return_lse=True, kernel_options=kernel_options
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out_compiled, lse_compiled = flex_compile(
            query, key, value, return_lse=True, kernel_options=kernel_options
        )

        if not torch.equal(out_eager, out_compiled):
            raise AssertionError("out_eager and out_compiled are not equal")
        if not torch.equal(lse_eager, lse_compiled):
            raise AssertionError("lse_eager and lse_compiled are not equal")

        grads_eager = torch.autograd.grad(out_eager.sum(), (query, key, value))
        grads_compile = torch.autograd.grad(out_compiled.sum(), (query, key, value))

        torch.testing.assert_close(grads_eager, grads_compile)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward
    def test_dynamic_shapes_bug_dynamic_batch(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): model with dynamic batch and
        # padding lengths, compiling block mask creation and attention in dynamic mode over five
        # random batches; asserts only one graph was compiled by the counter backend.
        def _flex_attention_mask(b, h, q_idx, kv_idx, input_lengths):
            # [FUNCTION DOC] HELPER: mask_mod true where both q and kv are below the batch's
            # input length.
            padding_condition = (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])
            return padding_condition

        counter = CompileCounterWithBackend("inductor")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a conv subsampler and linear projector that builds
            # heads and runs compiled attention with a padding block mask.
            def __init__(self, dim=1024):
                # [FUNCTION DOC] HELPER: creates the convolution, projection and head count.
                super().__init__()
                self.subsampler = torch.nn.Conv1d(256, 256, 5)
                self.projector = torch.nn.Linear(256, dim)
                self.num_heads = 4

            def forward(self, x, input_lengths):
                # [FUNCTION DOC] HELPER: embeds the input, builds the padding block mask with
                # dynamic compile and runs compiled fullgraph attention counted by the backend.
                x = self.subsampler(x.transpose(-1, -2)).transpose(-1, -2)
                x = self.projector(x).transpose(0, 1)
                head_dim = x.size(-1) // self.num_heads
                x = x.view(-1, x.size(1), self.num_heads, head_dim)
                x = x.permute(1, 2, 0, 3)

                max_time = x.size(-2)
                mask = torch.compile(create_block_mask, dynamic=True, fullgraph=False)(
                    functools.partial(
                        _flex_attention_mask,
                        input_lengths=input_lengths,
                    ),
                    B=input_lengths.size(0),
                    H=None,
                    Q_LEN=max_time,
                    KV_LEN=max_time,
                    device=device,
                )

                x = torch.compile(
                    flex_attention, dynamic=True, fullgraph=True, backend=counter
                )(
                    query=x,
                    key=x,
                    value=x,
                    block_mask=mask,
                )
                return x

        model = Model(128).to(device)
        B, F, T = 16, 256, 12
        for _ in range(5):
            x = torch.randn(B, T, F, device=device)
            l = torch.randint(0, T, (B,), device=device)
            model(x, l)

        if counter.frame_count != 1:
            raise AssertionError(
                f"Expected 1 graph, but got {counter.frame_count} graphs"
            )

    @supported_platform
    @skip_on_cpu
    def test_dynamic_shapes_with_custom_kernel_options(self, device):
        # [FUNCTION DOC] Non-CPU test: custom BLOCK_M/BLOCK_N options with dynamic shapes;
        # compiled output matches eager within 3e-3.
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device=device,
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}
        out_eager = flex_attention(query, key, value, kernel_options=kernel_options)

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    def test_dynamic_shapes_with_max_autotune(self, device):
        # [FUNCTION DOC] Compiled with max-autotune and dynamic shapes using a causal block mask;
        # compiled output matches eager within 3e-3.
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device=device,
            dtype=torch.float if device == "cpu" else torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        block_mask = create_block_mask(
            _causal_mask, None, None, 1024, 1024, device=device
        )

        out_eager = flex_attention(query, key, value, block_mask=block_mask)

        flex_compile = torch.compile(
            flex_attention, fullgraph=True, dynamic=True, mode="max-autotune"
        )
        out_compiled = flex_compile(query, key, value, block_mask=block_mask)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific "Query length must be greater than 0"
    def test_zero_length_sequence_error(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: a zero-length sequence makes compile raise
        # InductorError saying query length must be greater than 0.
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 0, 64),  # Zero in sequence dimension
            device=device,
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # Test compiled mode - should also raise assertion error
        flex_compile = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError, "Query length must be greater than 0"
        ):
            flex_compile(query, key, value)

    @supported_platform
    def test_causal_block_non_divisible_with_captured_buffer(
        self,
        device,
    ):
        # [FUNCTION DOC] Causal block mask with sequence length S minus three and captured bias
        # buffers, run with bf16 via run_test_with_call.
        # KNOWN ISSUE: the score_mod defined here is never passed to attention, so it is unused.
        Q_S = S - 3
        KV_S = S - 3
        offset_q = torch.randn(Q_S, device=device, dtype=torch.bfloat16)
        offset_kv = torch.randn(KV_S, device=device, dtype=torch.bfloat16)

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: unused score_mod adding captured q and kv offsets.
            return score + offset_q[q] + offset_kv[kv]

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, Q_S, KV_S, device=device)

        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(
            attention, Q_S=Q_S, KV_S=KV_S, dtype=torch.bfloat16, device=device
        )

    @supported_platform
    def test_non_divisible_with_captured_buffer(self, device):
        # [FUNCTION DOC] Sequence length S plus three (not block divisible) with a captured
        # multiplier buffer in the score_mod, bf16, via run_test_with_call.
        Q_S = S + 3
        KV_S = S + 3

        multiplier = torch.randn(Q_S, device=device, dtype=torch.bfloat16)

        def apply_multiplicative_bias(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: multiplies the score by the captured per-query multiplier.
            return score * multiplier[q_idx]

        attention = functools.partial(
            flex_attention, score_mod=apply_multiplicative_bias
        )

        self.run_test_with_call(
            attention, Q_S=Q_S, KV_S=KV_S, dtype=torch.bfloat16, device=device
        )

    @supported_platform
    def test_num_warps_8_error(self, device):
        # [FUNCTION DOC] Head dimension 128 with identity score_mod in float16 (regression for the
        # number of warps); checked via run_test_with_call.
        attention = functools.partial(flex_attention, score_mod=_identity)
        self.run_test_with_call(
            attention,
            dtype=torch.float16,
            device=device,
            Q_S=128,
            KV_S=128,
            Q_D=128,
            V_D=128,
        )

    @supported_platform
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_qkv_and_block_mask_on_the_same_device(self, device):
        # [FUNCTION DOC] Needs multiple GPUs: q/k/v on the first CUDA device and the block mask on
        # the second makes compile raise RuntimeError that they must be on the same device.
        make_tensor = functools.partial(
            torch.ones,
            (2, 2, 256, 32),
            device="cuda:0",
            dtype=torch.float32,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, 256, 256, device="cuda:1")
        with self.assertRaisesRegex(
            RuntimeError, "Expect q/k/v and block_mask to be on the same device"
        ):
            torch.compile(flex_attention)(query, key, value, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @config.patch(
        {
            "triton.native_matmul": False,
        }
    )
    @expected_not_implemented_on_mps  # no backward yet
    def test_free_symbol_dynamic(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError, native matmul off): a dynamic
        # compiled module run with block masks for batch sizes 4, 16 and 32 built with dynamic
        # create_block_mask from a random dense mask; asserts aot_autograd compiled graphs counter
        # is 2 (one static and one dynamic compile).
        def batch_flip_causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: unused mask_mod for causal attention on even batch indices.
            return (q_idx >= kv_idx) & (b % 2 == 0)

        class SimpleAttention(torch.nn.Module):
            # [CLASS DOC] HELPER: attention module with a fused qkv linear layer and head split
            # calling flex_attention with an optional block mask.
            def __init__(self, dim=512, n_head=8):
                # [FUNCTION DOC] HELPER: creates the qkv projection and head sizes.
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head = n_head
                self.head_dim = dim // n_head

            def forward(self, x, block_mask=None):
                # [FUNCTION DOC] HELPER: projects, splits heads, runs flex_attention and merges
                # heads back.
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                y = flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                )
                return y.transpose(1, 2).contiguous().view(B, T, C)

        model = SimpleAttention().to(device)
        model.compile(mode="default", dynamic=True)
        sequence_len = 256

        # Test different batch shapes with dense masks
        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            # Create dense mask
            rand_mask = torch.randint(
                0, 2, (batch_shape, sequence_len), device=device
            ).bool()
            block_mask = torch.compile(create_block_mask, dynamic=True)(
                B=batch_shape,
                BLOCK_SIZE=128,
                mask_mod=lambda b, h, q_idx, kv_idx: ~rand_mask[b, q_idx],
                H=None,
                Q_LEN=sequence_len,
                KV_LEN=sequence_len,
                device=device,
            )

            # Run forward pass
            x = torch.randn(batch_shape, sequence_len, 512, device=device)
            model(x, block_mask=block_mask)

        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_symbol_closure_in_score_mod(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a score_mod closing over a
        # symbolic batch size is compiled with dynamic shapes; asserts one graph recorded and the
        # score_mod subgraph code (inline expected text) takes the symbolic int as an extra input.
        class SimpleAttention(torch.nn.Module):
            # [CLASS DOC] HELPER: attention module whose score_mod adds the batch size to scores.
            def __init__(self, dim=512, n_head=8):
                # [FUNCTION DOC] HELPER: creates the qkv projection and head sizes.
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head = n_head
                self.head_dim = dim // n_head

            def forward(self, x, block_mask=None):
                # [FUNCTION DOC] HELPER: runs flex_attention with a score_mod that adds the
                # batch size.
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                return flex_attention(
                    q,
                    k,
                    v,
                    score_mod=lambda s, b, h, q, k: s + B,
                    block_mask=block_mask,
                )

        model = SimpleAttention().to(device)
        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()
        model.compile(mode="default", dynamic=True, backend=backend)
        sequence_len = 256

        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            x = torch.randn(batch_shape, sequence_len, 512, device=device)
            model(x)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].score_mod_0.code.strip(),
            """\
def forward(self, child : torch.Tensor, child_1 : torch.Tensor, child_2 : torch.Tensor, child_3 : torch.Tensor, child_4 : torch.Tensor, getitem : torch.SymInt):
    add = child + getitem;  child = getitem = None
    return add""",
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_fw_bw_graph_correctness(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): compiles forward and backward
        # with the aot_eager backend and a squared score_mod and causal block mask; asserts
        # one frame and one graph, then compares the normalized readable graphs to inline
        # expected text.
        cnt = CompileCounterWithBackend("aot_eager")
        # MPS lacks fp64; the NIE fires before any assertion would care.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 1, 1, 128, 128, device=device)

        func = torch.compile(flex_attention, backend=cnt, fullgraph=True)
        out = func(query, key, value, _squared, block_mask=block_mask)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)
        graph = cnt.graphs[0]
        norm_graph = normalize_gm(graph.print_readable(print_output=False))
        self.assertExpectedInline(
            norm_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_block_mask_full_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_full_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_full_q_indices: "i32[1, 1, 1, 1]", L_block_mask_full_q_num_blocks: "i32[1, 1, 1]", L_block_mask_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_q_indices: "i32[1, 1, 1, 1]", L_block_mask_q_num_blocks: "i32[1, 1, 1]", L_key_: "f64[2, 2, 128, 4]", L_query_: "f64[2, 2, 128, 4]", L_value_: "f64[2, 2, 128, 4]"):
        l_block_mask_full_kv_indices = L_block_mask_full_kv_indices
        l_block_mask_full_kv_num_blocks = L_block_mask_full_kv_num_blocks
        l_block_mask_full_q_indices = L_block_mask_full_q_indices
        l_block_mask_full_q_num_blocks = L_block_mask_full_q_num_blocks
        l_block_mask_kv_indices = L_block_mask_kv_indices
        l_block_mask_kv_num_blocks = L_block_mask_kv_num_blocks
        l_block_mask_q_indices = L_block_mask_q_indices
        l_block_mask_q_num_blocks = L_block_mask_q_num_blocks
        l_key_ = L_key_
        l_query_ = L_query_
        l_value_ = L_value_

        mask_fn_0 = self.mask_fn_0
        score_mod_0 = self.score_mod_0
        flex_attention = torch.ops.higher_order.flex_attention(l_query_, l_key_, l_value_, score_mod_0, (128, 128, l_block_mask_kv_num_blocks, l_block_mask_kv_indices, l_block_mask_full_kv_num_blocks, l_block_mask_full_kv_indices, l_block_mask_q_num_blocks, l_block_mask_q_indices, l_block_mask_full_q_num_blocks, l_block_mask_full_q_indices, None, None, None, None, 128, 128, mask_fn_0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  l_query_ = l_key_ = l_value_ = score_mod_0 = l_block_mask_kv_num_blocks = l_block_mask_kv_indices = l_block_mask_full_kv_num_blocks = l_block_mask_full_kv_indices = l_block_mask_q_num_blocks = l_block_mask_q_indices = l_block_mask_full_q_num_blocks = l_block_mask_full_q_indices = mask_fn_0 = None
        getitem: "f64[2, 2, 128, 4]" = flex_attention[0];  flex_attention = None
        return (getitem,)

    class mask_fn_0(torch.nn.Module):
        def forward(self, child: "i32[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]"):
            ge: "b8[]" = child_2 >= child_3;  child_2 = child_3 = None
            return ge

    class score_mod_0(torch.nn.Module):
        def forward(self, child: "f64[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]", child_4: "i32[]"):
            mul: "f64[]" = child * child;  child = None
            return mul
""",
        )
        # Save the AOT graphs
        aot_graphs = []
        from torch._inductor import compile_fx

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            # [FUNCTION DOC] HELPER: replacement inner compile function for compile_fx that
            # appends each AOT graph to the enclosing list and returns the graph unchanged.
            aot_graphs.append(graph)
            return graph

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )
        func = torch.compile(func, backend=backend, fullgraph=True)
        out = func(query, key, value, _squared)
        out.sum().backward()

        joint_graph = normalize_gm(aot_graphs[1].print_readable(print_output=False))
        self.assertExpectedInline(
            joint_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[2, 2, 128, 4]", primals_2: "f64[2, 2, 128, 4]", primals_3: "f64[2, 2, 128, 4]", full: "i32[1, 1, 1]", full_default: "i32[1, 1, 1, 1]", convert_element_type: "i32[1, 1, 1, 1]", convert_element_type_1: "i32[1, 1, 1]", getitem_2: "f64[2, 2, 128, 4]", getitem_3: "f32[2, 2, 128]", tangents_1: "f64[2, 2, 128, 4]"):
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_2, primals_1, primals_3, getitem_2, getitem_3, tangents_1, None, fw_graph0, joint_graph0, (1, 1, full, full_default, None, None, convert_element_type_1, convert_element_type, None, None, None, None, None, None, 1073741824, 1073741824, mask_graph0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  primals_2 = primals_1 = primals_3 = getitem_2 = getitem_3 = tangents_1 = fw_graph0 = joint_graph0 = full = full_default = convert_element_type_1 = convert_element_type = mask_graph0 = None
        getitem_5: "f64[2, 2, 128, 4]" = flex_attention_backward[0]
        getitem_6: "f64[2, 2, 128, 4]" = flex_attention_backward[1]
        getitem_7: "f64[2, 2, 128, 4]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_6, getitem_5, getitem_7)

    class fw_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
            return mul

    class joint_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f64[]"):
            mul_1: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            mul_2: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1);  arg5_1 = arg0_1 = None
            add: "f64[]" = torch.ops.aten.add.Tensor(mul_2, mul_1);  mul_2 = mul_1 = None
            return [add, None, None, None, None]

    class mask_graph0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]"):
            full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='GPU_TYPE', index=0), pin_memory = False)
            return full_default
""".replace("GPU_TYPE", torch.device(device).type),
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_direct_backward_preserves_explicit_buffers(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): calls the flex attention backward
        # higher-order op directly (compiled) with a captured scalar mask buffer in the mask_mod
        # (shift-by-buffer causal pattern); grads must match a golden float64 reference within the
        # fudge factor of 10.
        mask_buffer = torch.full((), 128, device=device, dtype=torch.int32)

        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: score_mod returning the score unchanged.
            return score

        def mask_mod(b, h, m, n):
            # [FUNCTION DOC] HELPER: mask_mod true where the position plus the captured scalar is
            # at least kv.
            return m + mask_buffer >= n

        block_mask = create_block_mask(
            mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5

        dtype = torch.float32
        q = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            score_mod,
            block_mask,
            scale,
        )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: compiled fullgraph wrapper calling the flex attention
            # backward op with the closed-over score_mod, block mask tuple and scale.
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (),
                (),
            )

        with torch.no_grad():
            dq, dk, dv, _ = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")

    @supported_platform
    @skip_on_cpu
    def test_direct_backward_supports_symint_score_mod_buffers(self, device):
        # [FUNCTION DOC] Non-CPU test: the direct backward op accepts a score_mod with an extra
        # integer (head dim) buffer, passed once as a literal and once as a symbolic size under
        # dynamic compile with aot_eager; dq, dk, dv and buffer grads must be identical.
        def score_mod(score, b, h, m, n, head_dim):
            # [FUNCTION DOC] HELPER: score_mod with an extra head-dimension argument, returns
            # the score.
            return score

        def mask_mod(b, h, m, n):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return m >= n

        block_mask = create_block_mask(
            mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        k = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        v = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            _identity,
            block_mask,
            scale,
        )
        static_head_dim = q.shape[-1]

        @torch.compile(backend="aot_eager", fullgraph=True)
        def compiled_literal_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: compiled backward call passing the static head dim as the
            # score_mod buffer.
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (static_head_dim,),
                (),
            )

        @torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: dynamic compiled backward call passing the query's last
            # dimension (symbolic) as the score_mod buffer.
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (query.shape[-1],),
                (),
            )

        with torch.no_grad():
            ref_dq, ref_dk, ref_dv, ref_buffer_grads = compiled_literal_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )
            compiled_dq, compiled_dk, compiled_dv, compiled_buffer_grads = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        torch.testing.assert_close(compiled_dq, ref_dq)
        torch.testing.assert_close(compiled_dk, ref_dk)
        torch.testing.assert_close(compiled_dv, ref_dv)
        self.assertEqual(compiled_buffer_grads, ref_buffer_grads)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward Triton lowering
    def test_direct_backward_inductor_batch_broadcast_matches_eager(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): batch-broadcast key/value (batch
        # 1, query batch 4) with the direct backward op on the TRITON backend; compiled dk, dv
        # shapes and strides match the eager direct call and grads match golden within fudge 10.
        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: score_mod returning the score unchanged.
            return score

        def mask_mod(b, h, m, n):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return m >= n

        block_mask = create_block_mask(
            mask_mod,
            B=4,
            H=None,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((4, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((1, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((1, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        backward_grad = torch.randn_like(q)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            _identity,
            block_mask,
            scale,
        )

        def eager_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: calls the direct flex attention backward op with the TRITON
            # backend option and no extra buffers.
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {"BACKEND": "TRITON"},
                (),
                (),
            )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: compiled fullgraph wrapper over the eager backward helper.
            return eager_bw(query, key, value, fwd_out, lse, grad_out)

        with torch.no_grad():
            _, ref_dk, ref_dv, _ = eager_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )
            dq, dk, dv, _ = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)
        self.assertEqual(dk.stride(), ref_dk.stride())
        self.assertEqual(dv.stride(), ref_dv.stride())

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward Triton lowering
    def test_direct_backward_inductor_supports_scalar_mask_mod_buffers(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a mask_mod taking an extra scalar
        # shift buffer (zero) is passed to the forward op and the direct compiled backward op on
        # the TRITON backend; grads match golden within fudge 10 and no buffer grads are returned.
        def score_mod(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: score_mod returning the score unchanged.
            return score

        def base_mask_mod(b, h, m, n):
            # [FUNCTION DOC] HELPER: causal mask_mod used for the reference block mask.
            return m >= n

        def mask_mod(b, h, m, n, shift):
            # [FUNCTION DOC] HELPER: causal mask_mod that adds a buffer-provided shift to q.
            return (m + shift) >= n

        base_block_mask = create_block_mask(
            base_mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        block_mask = BlockMask.from_kv_blocks(
            base_block_mask.kv_num_blocks,
            base_block_mask.kv_indices,
            base_block_mask.full_kv_num_blocks,
            base_block_mask.full_kv_indices,
            BLOCK_SIZE=base_block_mask.BLOCK_SIZE,
            mask_mod=mask_mod,
            seq_lengths=base_block_mask.seq_lengths,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        ref_attention = create_attention(score_mod, base_block_mask)
        golden_out = ref_attention(q_gold, k_gold, v_gold)
        ref_out = ref_attention(q_ref, k_ref, v_ref)
        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)
        static_shift = 0

        out, logsumexp, _ = flex_attention_hop(
            q,
            k,
            v,
            _identity,
            block_mask.as_tuple(),
            scale,
            {},
            (),
            (static_shift,),
        )

        def eager_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: direct backward op call (TRITON backend) with the
            # shift buffer.
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {"BACKEND": "TRITON"},
                (),
                (static_shift,),
            )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            # [FUNCTION DOC] HELPER: compiled fullgraph wrapper over the eager backward helper.
            return eager_bw(query, key, value, fwd_out, lse, grad_out)

        with torch.no_grad():
            dq, dk, dv, buffer_grads = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")
        self.assertEqual(buffer_grads, ())

    @supported_platform
    def test_tensor_subclass_dispatch_order(self, device):
        # [FUNCTION DOC] Verifies that a tensor subclass implementation of the flex attention
        # higher-order op takes dispatch priority over fake and functional tensor modes: a wrapper
        # subclass raises on the strided-view op, compiled flex attention with aot_eager must
        # still work on subclass inputs, return subclass outputs of the right shapes and match the
        # plain-tensor results.
        """Test that tensor subclasses get proper dispatch priority over modes.

        This test verifies the fix that allows tensor subclasses' pyimpl to run before
        FakeTensorMode/FunctionalTensorMode implementations, preventing issues
        where subclasses that error on as_strided would fail in flex_attention.
        """
        import torch.utils._pytree as pytree
        from torch.utils._python_dispatch import return_and_correct_aliasing

        class AsStridedErrorTensor(torch.Tensor):
            # [CLASS DOC] HELPER: wrapper tensor subclass around an element tensor that raises
            # RuntimeError if the strided-view op is dispatched on it; other ops are unwrapped,
            # run and rewrapped.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: creates a wrapper subclass instance with the element's
                # metadata (shape, strides, offset, dtype, layout, device, requires_grad); raises
                # AssertionError if the element is not a tensor.
                if not isinstance(elem, torch.Tensor):
                    raise AssertionError(f"Expected torch.Tensor, got {type(elem)}")
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.shape,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )

            def __init__(self, elem):
                # [FUNCTION DOC] HELPER: stores the wrapped element tensor.
                self.elem = elem

            def __repr__(self):
                # [FUNCTION DOC] HELPER: string form showing the wrapped element.
                return f"AsStridedErrorTensor({self.elem})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: flatten protocol: returns the single inner tensor name
                # and no metadata.
                return ["elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: unflatten protocol: rebuilds the wrapper from the inner
                # tensor mapping; raises AssertionError if metadata is not None.
                if meta is not None:
                    raise AssertionError(f"Expected meta to be None, got {meta}")
                elem = inner_tensors["elem"]
                return AsStridedErrorTensor(elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: dispatch hook: raises RuntimeError for the strided-view
                # op; otherwise unwraps all subclass arguments, calls the op, wraps tensor results
                # back and fixes aliasing.
                # Error if as_strided is called
                if func is torch.ops.aten.as_strided.default:
                    raise RuntimeError("as_strided was called on AsStridedErrorTensor!")

                if kwargs is None:
                    kwargs = {}
                args_elem = pytree.tree_map_only(
                    AsStridedErrorTensor, lambda x: x.elem, args
                )
                kwargs_elem = pytree.tree_map_only(
                    AsStridedErrorTensor, lambda x: x.elem, kwargs
                )

                out = func(*args_elem, **kwargs_elem)

                def wrap_output(x):
                    # [FUNCTION DOC] HELPER: wraps tensor results in the subclass and leaves other
                    # values alone.
                    if isinstance(x, torch.Tensor):
                        return AsStridedErrorTensor(x)
                    return x

                out_wrapped = pytree.tree_map(wrap_output, out)
                return return_and_correct_aliasing(func, args, kwargs, out_wrapped)

        from torch._higher_order_ops.flex_attention import (
            flex_attention as flex_attention_hop,
        )

        @flex_attention_hop.py_impl(AsStridedErrorTensor)
        def flex_attention_as_strided_error_tensor(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers=(),
            mask_mod_other_buffers=(),
        ):
            # [FUNCTION DOC] HELPER: Python implementation of the flex attention higher-order op
            # for the subclass: unwraps the inner tensors, calls the op, and wraps out, lse and
            # max scores.
            inner_q, inner_k, inner_v = query.elem, key.elem, value.elem
            out, lse, max_scores = flex_attention_hop(
                inner_q,
                inner_k,
                inner_v,
                score_mod,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
            return (
                AsStridedErrorTensor(out),
                AsStridedErrorTensor(lse),
                AsStridedErrorTensor(max_scores),
            )

        # Test setup
        B, H, S, D = 2, 1, 128, 16
        dtype = torch.float32

        # Create regular tensors
        query_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)
        key_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)
        value_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Test 1: Verify as_strided raises error when called directly on AsStridedErrorTensor
        test_tensor = AsStridedErrorTensor(query_elem)
        with self.assertRaisesRegex(
            RuntimeError, "as_strided was called on AsStridedErrorTensor!"
        ):
            torch.as_strided(
                test_tensor, size=(B, H, S, D), stride=test_tensor.stride()
            )

        # Test 2: Run flex_attention with normal tensors first
        compiled_fn = torch.compile(flex_attention, backend="aot_eager")
        normal_out, normal_lse = compiled_fn(
            query_elem, key_elem, value_elem, return_lse=True
        )

        # Test 3: Wrap in our subclass
        query = AsStridedErrorTensor(query_elem)
        key = AsStridedErrorTensor(key_elem)
        value = AsStridedErrorTensor(value_elem)

        # This should NOT error with as_strided after the fix
        # Before the fix, it would error because FakeTensorMode would directly
        # call flex_attention_fake_impl which uses as_strided
        out, lse = compiled_fn(query, key, value, return_lse=True)
        # Verify we got valid output
        self.assertIsInstance(out, AsStridedErrorTensor)
        self.assertIsInstance(lse, AsStridedErrorTensor)
        self.assertEqual(out.shape, (B, H, S, D))
        self.assertEqual(lse.shape, (B, H, S))

        # Test 4: Compare outputs between normal tensors and subclassed tensors
        torch.testing.assert_close(out.elem, normal_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lse.elem, normal_lse, rtol=1e-5, atol=1e-5)

    @supported_platform
    @skip_on_cuda
    @skip_on_xpu
    def test_cpu_qk_chunk_same_addmm_buffer(self, device):
        # [FUNCTION DOC] CPU-only (skipped on CUDA/XPU) test: a module whose query and key come
        # from chunks of one linear output (shared addmm buffer) feeds flex_attention; compiled
        # no-grad output matches eager within 1e-4.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: module projecting hidden states into a shared query/key
            # tensor and a value tensor, splitting heads, and calling flex_attention with an
            # identity score_mod.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the head, dimension and sequence sizes and two
                # linear layers.
                super().__init__()
                self.H = 2
                self.D = 16
                self.S = 128
                self.project_qk = nn.Linear(self.H * self.D, self.H * self.D * 2)
                self.project_v = nn.Linear(self.H * self.D, self.H * self.D)

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: projects, chunks query and key, builds heads and
                # runs attention.
                B = hidden_states.size(0)
                qk = self.project_qk(hidden_states)
                query, key = qk.chunk(2, dim=-1)

                query = query.view(B, self.S, self.H, self.D).permute(0, 2, 1, 3)
                key = key.view(B, self.S, self.H, self.D).permute(0, 2, 1, 3)
                value = (
                    self.project_v(hidden_states)
                    .view(B, self.S, self.H, self.D)
                    .permute(0, 2, 1, 3)
                )

                return flex_attention(query, key, value, score_mod=_identity)

        torch.manual_seed(0)
        x = torch.randn(1, 128, 32, device=device)
        model = Model().to(device).eval()

        with torch.no_grad():
            eager_out = model(x)
            compiled_out = torch.compile(model)(x)

        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-4, atol=1e-4)

    @supported_platform
    @skip_on_cuda
    def test_cpu_error_message_return_lse(self, device):
        # [FUNCTION DOC] CPU-only test: compiling flex_attention with lse on CPU raises
        # InductorError saying CPU only supports inference without lse.
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device="cpu",
            dtype=torch.float32,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        attention = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            r"NotImplementedError: torch.compile on CPU only supports inference and `return_lse` is not supported yet.",
        ):
            attention(query, key, value, return_lse=True)

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    def test_nested_tensor_inputs_error(self, device):
        # [FUNCTION DOC] Non-CPU/XPU test: nested jagged tensors make eager flex_attention raise
        # NotImplementedError and compiled fullgraph raise a dynamo Unsupported error with the
        # same message about NestedTensor inputs.
        def make_nt(lengths, heads=2, dim=8):
            # [FUNCTION DOC] HELPER: builds a jagged-layout nested tensor from random per-sequence
            # tensors and transposes heads and sequence.
            elems = [
                torch.randn(s, heads, dim, device=device, dtype=torch.float32)
                for s in lengths
            ]
            return torch.nested.nested_tensor(elems, layout=torch.jagged).transpose(
                1, 2
            )

        q = make_nt([3, 2])
        k = make_nt([3, 2])
        v = make_nt([3, 2])
        msg = "flex_attention does not support NestedTensor inputs"

        with self.assertRaisesRegex(NotImplementedError, msg):
            flex_attention(q, k, v)

        compiled_flex = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            compiled_flex(q, k, v)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_device_cuda_1(self, device):
        # [FUNCTION DOC] Needs multiple GPUs: compiled module running attention with inputs and
        # block mask on the second CUDA device; output must stay on that device.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward just calls flex_attention with a
            # block mask.
            def forward(self, q, k, v, block_mask):
                # [FUNCTION DOC] HELPER: returns flex_attention of q, k, v with the block mask.
                return flex_attention(q, k, v, block_mask=block_mask)

        q = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        k = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=256,
            KV_LEN=256,
            device="cuda:1",
        )
        mod = torch.compile(TestModule())
        attn_output = mod(q, k, v, mask)
        self.assertEqual(attn_output.device, torch.device("cuda:1"))

    @supported_platform
    @skip_on_cpu
    def test_custom_score_mod_layout_freeze(self, device):
        # [FUNCTION DOC] Non-CPU test: module whose score_mod reads a learned MLP bias table
        # through captured index buffers (layout freeze scenario); compiled inductor (static)
        # output under flash SDPA context matches eager within 0.02.
        torch.manual_seed(0)

        class FlexAttentionCPB(nn.Module):
            # [CLASS DOC] HELPER: attention module with a learned relative-position bias: an MLP
            # over a log-scaled relative offset table, a gate from mu inputs and non-persistent
            # index buffers; bias applies only to positions at or after a cutoff.
            def __init__(self, N: int, R: int, H: int = 4, hidden: int = 32):
                # [FUNCTION DOC] HELPER: builds the MLP, gamma parameter, lookup tables and the
                # cutoff buffer.
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, H, bias=False),
                )
                self.gamma = nn.Parameter(torch.zeros(H))
                self.H = H
                self._init_tables(N, R)
                self.register_buffer(
                    "r_cutoff", torch.tensor(R, dtype=torch.long), persistent=False
                )

            def _init_tables(self, N: int, R: int) -> None:
                # [FUNCTION DOC] HELPER: builds the relative offset table and an index table
                # mapping pairs of positions after the cutoff to table rows; raises AssertionError
                # if the number of non-cutoff positions is not a perfect square. Registers both as
                # non-persistent buffers.
                P = N - R
                S = int(P**0.5)
                if S * S != P:
                    raise AssertionError(f"Expected S * S == P, got {S * S} != {P}")
                rng = torch.arange(-(S - 1), S, dtype=torch.float32)
                dY, dX = torch.meshgrid(rng, rng, indexing="ij")
                rel = torch.stack(
                    [dY / max(S - 1, 1), dX / max(S - 1, 1)], dim=-1
                ).reshape(-1, 2)
                rel_table = torch.sign(rel) * torch.log1p(rel.abs())
                self.register_buffer("rel_table", rel_table, persistent=False)

                yy, xx = torch.arange(S), torch.arange(S)
                Y, X = torch.meshgrid(yy, xx, indexing="ij")
                flat = torch.stack([Y, X], 0).flatten(1)
                d = flat[:, :, None] - flat[:, None, :]
                d = d.permute(1, 2, 0).contiguous()
                d[:, :, 0] += S - 1
                d[:, :, 1] += S - 1
                d[:, :, 0] *= 2 * S - 1
                l_idx = d.sum(-1).to(torch.long)
                idx = torch.full((N, N), 0, dtype=torch.long)
                idx[R:, R:] = l_idx
                self.register_buffer("idx_table", idx, persistent=False)

            def _score_mod(self, mu: torch.Tensor):
                # [FUNCTION DOC] HELPER: returns a score_mod adding a gated bias for pairs past
                # the cutoff, using the MLP output table and the mu-based gate.
                bt = self.mlp(self.rel_table)
                idx = self.idx_table
                mu_q, mu_k = mu.unbind(2)
                gam_sig = torch.sigmoid(self.gamma)

                def score_mod(score, b, h, q, kv):
                    # [FUNCTION DOC] HELPER: score_mod adding gate times bias where both positions
                    # pass the cutoff.
                    has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
                    l2 = idx[q, kv]
                    bias = bt[l2, h]
                    w_gate = gam_sig[h] * (mu_q[b, h, q] + mu_k[b, h, kv])
                    return score + has_bias.to(score.dtype) * w_gate * bias

                return score_mod

            def forward(self, q, k, v, mu):
                # [FUNCTION DOC] HELPER: runs flex_attention with the score_mod built from mu.
                return flex_attention(q, k, v, score_mod=self._score_mod(mu))

        dtype = torch.bfloat16 if PLATFORM_SUPPORTS_BF16 else torch.float16
        device_obj = torch.device(device)
        module = FlexAttentionCPB(N=18, R=2).to(device_obj)
        compiled_module = torch.compile(module, backend="inductor", dynamic=False)

        q = torch.randn(2, 4, 18, 32, device=device_obj, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mu = torch.randn(2, 4, 2, 18, device=device_obj)

        with torch.no_grad():
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                eager_out = module(q, k, v, mu)
                compiled_out = compiled_module(q, k, v, mu)

        self.assertEqual(compiled_out.shape, eager_out.shape)
        torch.testing.assert_close(
            compiled_out.float(), eager_out.float(), atol=2e-2, rtol=2e-2
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize(
        "ops_to_save",
        [
            [
                torch.ops.aten.mm.default,
            ],
            [
                flex_attention_hop,
            ],
            [torch.ops.aten.mm.default, flex_attention_hop],
        ],
    )
    @expected_not_implemented_on_mps  # no backward yet
    def test_selective_ac(self, device, ops_to_save):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): flex_attention inside a selective
        # activation checkpoint region saving the given ops (matmul, the flex op, or both);
        # compiled outputs and input grads match eager within 0.01.
        class FlexAttentionModule(nn.Module):
            # [CLASS DOC] HELPER: attention block with separate q, k, v and output linear layers
            # and flex_attention in the middle.
            def __init__(self, hidden_size, num_heads):
                # [FUNCTION DOC] HELPER: creates the four linear layers and the head size.
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads

                # In-projections (query, key, value)
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)

                # Out-projection
                self.out_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: projects to q, k, v heads, applies attention and projects
                # the output.
                batch_size, seq_len, _ = x.size()

                # Project queries, keys, and values
                q = (
                    self.q_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # Apply flex attention
                attn_output = flex_attention(
                    q,
                    k,
                    v,
                )

                # Reshape output
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.hidden_size)
                )

                # Out projection
                output = self.out_proj(attn_output)

                return output

        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        context_fn = functools.partial(
            create_selective_checkpoint_contexts, ops_to_save
        )

        # Define a model that uses FlexAttention with selective activation checkpointing
        class SacModule(nn.Module):
            # [CLASS DOC] HELPER: wrapper running the attention block under reentrant-free
            # checkpointing with the selective context function.
            def __init__(self, hidden_size, num_heads, context_fn):
                # [FUNCTION DOC] HELPER: stores the attention block and the context function.
                super().__init__()
                self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
                self.context_fn = context_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs the attention block via checkpoint.
                def flex_attn_fn(x):
                    # [FUNCTION DOC] HELPER: calls the attention block.
                    return self.flex_attn(x)

                output = checkpoint(
                    flex_attn_fn,
                    x,
                    use_reentrant=False,
                    context_fn=self.context_fn,
                )

                return output

        flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
            device, dtype=torch.bfloat16
        )
        x = torch.ones(8, 1024, 512, device=device, dtype=torch.bfloat16)

        # Run without compilation
        output_module = flex_module(x)
        compiled_module = torch.compile(flex_module)
        output_compiled = compiled_module(x)

        torch.testing.assert_close(output_module, output_compiled, rtol=1e-2, atol=1e-2)

        # Calculate gradients and compare them
        x.requires_grad_(True)
        output_module = flex_module(x)
        output_compiled = compiled_module(x)
        grad_output = torch.ones_like(output_module)

        grad_module = torch.autograd.grad(
            outputs=output_module, inputs=x, grad_outputs=grad_output, retain_graph=True
        )[0]

        grad_compiled = torch.autograd.grad(
            outputs=output_compiled, inputs=x, grad_outputs=grad_output
        )[0]

        torch.testing.assert_close(grad_module, grad_compiled, rtol=1e-2, atol=1e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # uses mode="max-autotune-no-cudagraphs"; Triton/CUDA-only
    def test_selective_ac_with_max_autotune_short_query(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: selective checkpoint policy saving mm and bmm, flex
        # attention compiled with max-autotune-no-cudagraphs inside the checkpoint on a short
        # query; backward must produce an input gradient.
        from functools import partial

        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        compute_intensive_ops = [
            torch.ops.aten.mm,
            torch.ops.aten.bmm,
        ]

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: checkpoint policy: must save matmul-like ops, prefer
            # recompute others.
            if op in compute_intensive_ops:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        class DummyAttentionModule(nn.Module):
            # [CLASS DOC] HELPER: attention module whose flex attention call is wrapped in
            # selective checkpointing and compiled with max-autotune without cudagraphs.
            def __init__(self, dim=64, num_heads=4):
                # [FUNCTION DOC] HELPER: creates projections, the checkpoint context function and
                # the compiled checkpointed attention.
                super().__init__()
                self.dim = dim
                self.num_heads = num_heads
                self.head_dim = dim // num_heads

                self.q_proj = nn.Linear(dim, dim)
                self.k_proj = nn.Linear(dim, dim)
                self.v_proj = nn.Linear(dim, dim)
                self.out_proj = nn.Linear(dim, dim)

                self._activation_checkpoint_context_fn = partial(
                    create_selective_checkpoint_contexts, policy_fn
                )

                self._flex_attention = torch.compile(
                    partial(
                        checkpoint,
                        flex_attention,
                        use_reentrant=False,
                        context_fn=self._activation_checkpoint_context_fn,
                    ),
                    mode="max-autotune-no-cudagraphs",
                )

            def forward(self, x, block_mask):
                # [FUNCTION DOC] HELPER: projects, runs the compiled checkpointed attention with
                # the block mask and projects the output.
                batch_size, seq_len, _ = x.shape

                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = q.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                k = k.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                v = v.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)

                attn_out = self._flex_attention(q, k, v, block_mask=block_mask)

                attn_out = (
                    attn_out.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.dim)
                )

                out = self.out_proj(attn_out)

                return out

        batch_size = 2
        seq_len = 64
        dim = 64
        num_heads = 4

        model = DummyAttentionModule(dim=dim, num_heads=num_heads).to(device)

        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        block_mask = create_block_mask(
            causal_mask,
            B=batch_size,
            H=num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        out = model(x, block_mask)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific NYI "embedding dimension" message
    def test_validate_small_embedding_size_error_message(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: head dim 8 works eagerly but compiled attention raises
        # InductorError with an NYI message that embedding dims must be at least 16.
        # KNOWN ISSUE: the second eager-style call is labeled compiled CPU but is a plain call,
        # and the final compile for dim 16 is never executed.
        # eager support for small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        flex_attention(q, k, v)

        # compiled cpu support for small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        flex_attention(q, k, v)

        # compiled gpu kernel does not support small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        compiled_fa = torch.compile(flex_attention)

        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "NYI: embedding dimension of the query, key, and value must be "
            "at least 16 but got E=8 and Ev=8",
        ):
            compiled_fa(q, k, v)

        # compiled gpu kernel supports large embedding size
        q, k, v = [torch.randn(2, 2, 128, 16, device=device) for _ in range(3)]
        compiled_fa = torch.compile(flex_attention)

    @unittest.skipIf(
        not has_triton() or not HAS_WARP_SPEC,
        reason="FBCODE Triton is required for this test",
    )
    def test_triton_template_warp_specialization(self, device):
        # [FUNCTION DOC] Skipped unless Triton with warp specialization is available: compiles
        # with kernel options for warp counts, consumer groups and buffers; generated code must
        # mention the specialization parameters and the result must match eager within 0.01
        # (raises AssertionError otherwise).
        def make_tensor():
            # [FUNCTION DOC] HELPER: random bf16 tensor of batch 4, 16 heads, length 4096, dim 64.
            return torch.rand(4, 16, 4096, 64, device=device, dtype=torch.bfloat16)

        q, k, v = make_tensor(), make_tensor(), make_tensor()
        flex_compiled = torch.compile(flex_attention, fullgraph=True)

        positional_args = (q, k, v)
        keyword_args = {
            "kernel_options": {
                "num_warps": 4,
                "num_consumer_groups": 2,
                "num_buffers_warp_spec": 3,
            }
        }

        # Check if kernel code contains warp specialization parameters
        _, kernel_code = run_and_get_code(
            flex_compiled,
            *positional_args,
            **keyword_args,
        )
        if kernel_code is None:
            raise AssertionError("Failed to retrieve compiled kernel code")
        if "num_consumer_groups" not in kernel_code[0]:
            raise AssertionError("num_consumer_groups missing in kernel definition")
        if "num_buffers_warp_spec" not in kernel_code[0]:
            raise AssertionError("num_buffers_warp_spec missing in kernel definition")

        # Validate correctness
        C1 = flex_compiled(q, k, v)
        C2 = flex_attention(q, k, v)

        if not torch.allclose(C1, C2, atol=1e-2, rtol=1e-2):
            raise AssertionError(
                "Warp specialized kernel result differs from reference"
            )

    @supported_platform
    @skip_on_cpu
    @skipCUDAIf(not has_triton_tma_device(), "Requires TMA enabled CUDA device")
    def test_tma_with_customer_kernel_options(self, device):
        # [FUNCTION DOC] Non-CPU test (needs TMA on CUDA): compiles with TMA off and on with small
        # block sizes (dynamic); outputs must match within 0.2 and, when backward is supported,
        # grads are compared.
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 256, 128),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        kernel_options_1 = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "USE_TMA": False,
        }
        kernel_options_2 = {"BLOCK_M": 32, "BLOCK_N": 32, "USE_TMA": True}

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options_1)
        out_tma_compiled = flex_compile(
            query, key, value, kernel_options=kernel_options_2
        )

        # vanilla compiled vs TMA compiled
        torch.testing.assert_close(out_tma_compiled, out_compiled, atol=2e-1, rtol=2e-1)

        if requires_grad:
            grad_output = torch.randn_like(out_compiled)

            out_compiled.backward(grad_output)
            compiled_grads = [query.grad, key.grad, value.grad]
            query.grad = None
            key.grad = None
            value.grad = None

            out_tma_compiled.backward(grad_output)
            tma_grads = [query.grad, key.grad, value.grad]
            query.grad = None
            key.grad = None
            value.grad = None

            torch.testing.assert_close(
                compiled_grads[0],
                tma_grads[0],
                atol=2e-1,
                rtol=2e-1,
            )
            torch.testing.assert_close(
                compiled_grads[1],
                tma_grads[1],
                atol=2e-1,
                rtol=2e-1,
            )
            torch.testing.assert_close(
                compiled_grads[2],
                tma_grads[2],
                atol=2e-1,
                rtol=2e-1,
            )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward yet
    def test_large_batch_heads_grid_dimension(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): very large batch times heads
        # (22720 by 3) in float16, compiled dynamic fullgraph forward and backward; output and
        # grad shapes must be correct and grads non-None (exercises the GPU grid dimension limit).
        B, H, S, D = 22720, 3, 64, 32

        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )

        query, key, value = make_tensor(), make_tensor(), make_tensor()

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value)

        self.assertEqual(out_compiled.shape, (B, H, S, D))

        grad_output = torch.randn_like(out_compiled)
        out_compiled.backward(grad_output)

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

    @supported_platform
    def test_debug_flag_disables_internal_compilation(self, device):
        # [FUNCTION DOC] Verifies the module debug flag that disables internal compilation: with
        # the flag off, eager flex_attention emits a UserWarning about being called without
        # compile; with it on, no warning is raised (warnings are errors) and outputs agree within
        # 1e-4. The flag and the shown-warnings registry are restored in a finally block.
        """Test that _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG flag bypasses internal compilation."""
        import torch.nn.attention.flex_attention as fa

        original_flag = fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG
        original_warnings_shown = fa._WARNINGS_SHOWN.copy()

        try:
            B, H, S, D = 1, 1, 128, 64
            query = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
            key = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
            value = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

            def simple_score_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod returning the score unchanged.
                return score

            # Test with debug flag False - should warn
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False
            fa._WARNINGS_SHOWN.clear()

            with self.assertWarns(UserWarning) as cm:
                out_compiled = fa.flex_attention(
                    query, key, value, score_mod=simple_score_mod
                )

            self.assertIn(
                "flex_attention called without torch.compile", str(cm.warning)
            )

            # Test with debug flag True - should NOT warn
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

            # Should not error
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                out_debug = fa.flex_attention(
                    query, key, value, score_mod=simple_score_mod
                )

            torch.testing.assert_close(out_compiled, out_debug, rtol=1e-4, atol=1e-4)

        finally:
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = original_flag
            fa._WARNINGS_SHOWN.clear()
            fa._WARNINGS_SHOWN.update(original_warnings_shown)

    @supported_platform
    def test_mask_mod_functools_partial(self, device):
        # [FUNCTION DOC] Regression test (issue 170489): a mask_mod made by binding an offset
        # argument with functools.partial builds a block mask and runs compiled attention; checks
        # only the output shape.
        """Test that functools.partial wrapped mask_mod works with flex_attention.

        Regression test for https://github.com/pytorch/pytorch/issues/170489
        """
        B, H, S, D = 1, 8, 64, 64

        def causal_mask_with_offset(b, h, q_idx, kv_idx, offset):
            # [FUNCTION DOC] HELPER: mask_mod true where q plus the given offset is at least kv.
            return (q_idx + offset) >= kv_idx

        offset = 31
        mask_mod = functools.partial(causal_mask_with_offset, offset=offset)

        query = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        key = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        value = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        block_mask = create_block_mask(mask_mod, B, H, S, S, device=device)

        flex_compiled = torch.compile(flex_attention)
        out = flex_compiled(query, key, value, block_mask=block_mask)

        self.assertEqual(out.shape, (B, H, S, D))

    @supported_platform
    def test_callable_class_mask_mod(self, device):
        # [FUNCTION DOC] A callable class instance (with a name, equality and hash) used as
        # mask_mod: creating the block mask inside a fullgraph compiled function and running
        # attention must compile without error.
        # Test that callable class instances work as mask_mod with flex_attention

        B, H, S, D = 1, 8, 64, 64

        class CausalOrWindowMask:
            # [CLASS DOC] HELPER: callable mask_mod object with a window size; true where causal
            # or kv is less than a window size ahead of q. Defines a name attribute, equality by
            # window size and a matching hash.
            __name__ = "causal_or_window"

            def __init__(self, window_size):
                # [FUNCTION DOC] HELPER: stores the window size.
                self.window_size = window_size

            def __call__(self, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask value: causal or inside the window.
                causal = q_idx >= kv_idx
                in_window = (kv_idx - q_idx) < self.window_size
                return causal | in_window

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: equal when other is the same class with the same window
                # size, else returns NotImplemented.
                if not isinstance(other, CausalOrWindowMask):
                    return NotImplemented
                return self.window_size == other.window_size

            def __hash__(self):
                # [FUNCTION DOC] HELPER: hash of the window size.
                return hash(self.window_size)

        mask_mod = CausalOrWindowMask(window_size=64)

        query = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        key = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        value = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        @torch.compile(fullgraph=True)
        def f(q, k, v):
            # [FUNCTION DOC] HELPER: compiled function that builds the block mask and
            # runs attention.
            block_mask = create_block_mask(mask_mod, B, H, S, S, device=q.device)
            return flex_attention(q, k, v, block_mask=block_mask)

        _ = f(query, key, value)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts TritonTemplateKernel layout-freezing internal
    def test_flex_attention_always_freezes_layout(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: patches the Triton template kernel method that gets
        # strides and maybe freezes layout with a wrapper; for every flexible-layout input node
        # seen, after the call the layout must be fixed (AssertionError otherwise), and at least
        # one such call must have happened.
        """Test that flex attention always freezes FlexibleLayout inputs.

        When always_freeze_layout=True on flex attention templates,
        get_stride_and_maybe_freeze_layout should freeze FlexibleLayout
        immediately rather than using layout constraints.
        """
        from torch._inductor import ir
        from torch._inductor.select_algorithm import TritonTemplateKernel

        B, H, S, D = 2, 4, 128, 64
        dtype = torch.float16

        query = torch.randn(B, H, S, D, device=device, dtype=dtype)
        key = torch.randn(B, H, S, D, device=device, dtype=dtype)
        value = torch.randn(B, H, S, D, device=device, dtype=dtype)

        flexible_layout_called = False
        orig_stride_call = TritonTemplateKernel.get_stride_and_maybe_freeze_layout

        def tracking_get_stride(self, node):
            # [FUNCTION DOC] HELPER: wrapper recording whether the node had a flexible layout,
            # calling the original, and checking the node layout became fixed.
            nonlocal flexible_layout_called
            flexible_layout = isinstance(node.data.layout, ir.FlexibleLayout)
            result = orig_stride_call(self, node)
            if flexible_layout:
                flexible_layout_called = True
                if not isinstance(node.data.layout, ir.FixedLayout):
                    raise AssertionError(
                        f"Expected FixedLayout, got {type(node.data.layout)}"
                    )
            return result

        with patch.object(
            TritonTemplateKernel,
            "get_stride_and_maybe_freeze_layout",
            tracking_get_stride,
        ):
            compiled_flex = torch.compile(flex_attention, fullgraph=True)
            compiled_flex(query, key, value)

        self.assertTrue(
            flexible_layout_called,
            "get_stride_and_maybe_freeze_layout should be called with FlexibleLayout nodes",
        )

    @unittest.skip(
        "Too large to run reliably in CI. Kept as a manual repro for #185262"
    )
    @supported_platform
    @skip_on_cpu
    @largeTensorTest("12GB", inductor=True)
    @serialTest()
    def test_large_kv_int64_pointer_math(self, device):
        # [FUNCTION DOC] Skipped test (too large for CI), kept as a manual repro for a 64-bit
        # pointer math bug: huge key/value storage (beyond 2 to the 31 elements) with a block mask
        # selecting only the last KV block, compared to a float32 reference on the tail.
        # Regression test for #185262: load_checked_2d (and a few inline
        # pointer-arithmetic sites in the backwards template) computed
        # `offs * stride` in int32 even when Inductor had already promoted
        # INDEX_DTYPE to int64 because some input buffer exceeded 2**31
        # elements. The product silently overflowed and caused an IMA.
        #
        # We construct K/V so that (a) each tensor's storage exceeds 2**31
        # elements (so Inductor uses int64 indexing), and (b) the last KV
        # index times its KV-dim stride also exceeds 2**31 (so the
        # in-kernel pointer math would overflow without the .to(INDEX_DTYPE)
        # cast). Without the fix this raises a CUDA illegal memory access.
        H, D = 1, 128
        KV_S = 2**24 + 1024
        Q_S = 128
        dtype = torch.bfloat16

        q = torch.randn(1, H, Q_S, D, device=device, dtype=dtype)
        k = torch.randn(1, H, KV_S, D, device=device, dtype=dtype)
        v = torch.randn(1, H, KV_S, D, device=device, dtype=dtype)

        def mask_mod(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod true for the last tail positions of the key/value.
            return kv_idx >= (KV_S - tail)

        BLOCK = _DEFAULT_SPARSE_BLOCK_SIZE
        tail = BLOCK
        last_kv_block = KV_S // BLOCK - 1
        kv_num_blocks = torch.ones(1, 1, 1, dtype=torch.int32, device=device)
        kv_indices = torch.full(
            (1, 1, 1, 1), last_kv_block, dtype=torch.int32, device=device
        )
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=BLOCK,
            mask_mod=mask_mod,
            seq_lengths=(Q_S, KV_S),
        )

        compiled = torch.compile(flex_attention, fullgraph=True)
        out = compiled(q, k, v, block_mask=block_mask)

        # Reference: only the last `tail` KV positions are unmasked.
        k_tail = k[:, :, -tail:, :].to(torch.float32)
        v_tail = v[:, :, -tail:, :].to(torch.float32)
        q_f32 = q.to(torch.float32)
        scale = D**-0.5
        scores = (q_f32 @ k_tail.transpose(-1, -2)) * scale
        ref = (torch.softmax(scores, dim=-1) @ v_tail).to(dtype)

        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


class TestBlockMask(InductorTestCase):
    # [CLASS DOC] Tests for the BlockMask data structure and create_block_mask: attributes,
    # sparsity, indexing and slicing, device moves, compile behavior, from_kv_blocks constructors
    # and constructor validation.
    def setUp(self):
        # [FUNCTION DOC] HELPER: calls the parent setUp.
        super().setUp()

    @supported_platform
    def test_block_mask_attributes(self, device):
        # [FUNCTION DOC] Checks shape, numel and sparsity (46.875 percent for causal) of a block
        # mask and its indexing results; with batch-dependent offsets, sparsity differs across
        # batches (29.1015625 overall) and decreases with batch index.
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod shifted by a captured per-batch offset
            # times 128.
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 2048, 2048, device=device)
        self.assertEqual(block_mask.shape, (4, 2, 2048, 2048))
        self.assertEqual(block_mask[0].shape, (1, 2, 2048, 2048))
        self.assertEqual(block_mask[0, 0].shape, (1, 1, 2048, 2048))
        self.assertEqual(block_mask.numel(), 4 * 2 * 2048 * 2048)
        self.assertEqual(block_mask.sparsity(), 46.875)
        self.assertEqual(block_mask[0].sparsity(), 46.875)
        self.assertEqual(block_mask[1, 0].sparsity(), 46.875)
        self.assertEqual(block_mask.sparsity(), block_mask[1].sparsity())

        offset = torch.arange(8, device=device)
        block_mask = create_block_mask(causal_mask, 8, 1, 2048, 2048, device=device)
        self.assertEqual(block_mask.sparsity(), 29.1015625)
        self.assertTrue(block_mask.sparsity() < block_mask[0].sparsity())
        self.assertTrue(block_mask[0].sparsity() > block_mask[1].sparsity())

    @supported_platform
    def test_block_mask_sparsity_with_partial_block(self, device):
        # [FUNCTION DOC] A causal-in-document mask of sequence 100 (not block aligned) yields zero
        # reported sparsity, also shown in the string form.
        document_id = torch.zeros(100, dtype=torch.int, device=device)
        document_id[10:20] = 1
        for i in range(20, 100, 20):
            document_id[i : i + 20] = i // 20 + 1

        def document_causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: mask_mod true where causal and the two positions share
            # a document.
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            return causal_mask & document_mask

        block_mask = create_block_mask(document_causal_mask, 1, 1, 100, 100, device)
        self.assertEqual(block_mask.sparsity(), 0.0)
        self.assertTrue("sparsity=0.00%" in str(block_mask))

    @supported_platform
    def test_adjust_block_mask_ignores_entries_past_num_blocks(self, device):
        # [FUNCTION DOC] Adjusting a block mask to smaller sizes must ignore index entries beyond
        # the valid block count: adjusted full block counts and dense form equal a mask built
        # directly at that size.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true for kv below 128 or in 384 to 511.
            return (kv < 128) | ((kv >= 384) & (kv < 512))

        block_mask = create_block_mask(mask_mod, 1, 1, 128, 640, device=device)
        adjusted = block_mask._adjust(128, 384)
        expected = create_block_mask(mask_mod, 1, 1, 128, 384, device=device)

        self.assertEqual(adjusted.full_kv_num_blocks, expected.full_kv_num_blocks)
        self.assertEqual(adjusted.to_dense(), expected.to_dense())

    @supported_platform
    @common_utils.parametrize("BLOCK_SIZE", [32, 64, 128, 256, (32, 64), (64, 32)])
    def test_block_size_changes(self, device, BLOCK_SIZE: int | tuple[int, int]):
        # [FUNCTION DOC] For several square or rectangular block sizes, create_block_mask records
        # the BLOCK_SIZE as a pair and the overall shape.
        B, H, Q_LEN, KV_LEN = 4, 2, 2048, 2048

        if isinstance(BLOCK_SIZE, int):
            Q_BLOCK_SIZE = BLOCK_SIZE
            KV_BLOCK_SIZE = BLOCK_SIZE
        else:
            Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

        block_mask = create_block_mask(
            noop_mask, B, H, Q_LEN, KV_LEN, BLOCK_SIZE=BLOCK_SIZE, device=device
        )

        self.assertEqual(block_mask.BLOCK_SIZE, (Q_BLOCK_SIZE, KV_BLOCK_SIZE))
        self.assertEqual(block_mask.shape, (B, H, Q_LEN, KV_LEN))

    @supported_platform
    def test_getitem_query_slice_updates_shape(self, device):
        # [FUNCTION DOC] Indexing a block mask along the query-block dimension (single index,
        # slice, negative index, stepped slice, tensor index) updates its shape and sequence
        # lengths and keeps the dense form of the selected rows; indexing the key dimension
        # raises IndexError; the tail block of an uneven sequence has smaller size and recomputed
        # q indices.
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=8,
            KV_LEN=9,
            BLOCK_SIZE=4,
            device=device,
        )

        first_q_block = block_mask[:, :, :1]
        self.assertEqual(first_q_block.shape, (1, 1, 4, 9))
        self.assertEqual(first_q_block.seq_lengths, (4, 9))
        self.assertEqual(block_mask[:, :, 0].shape, (1, 1, 4, 9))

        second_q_block = block_mask[:, :, 1]
        self.assertEqual(second_q_block.shape, (1, 1, 4, 9))
        self.assertEqual(second_q_block.to_dense(), block_mask.to_dense()[:, :, 1:2, :])

        q = torch.empty(1, 1, 8, 2, device=device)
        self.assertEqual(q[:, :, :4, :].shape[-2], first_q_block.shape[-2])
        self.assertEqual(q[:, :, 4:8, :].shape[-2], second_q_block.shape[-2])

        with self.assertRaisesRegex(IndexError, "batch, head, and Q-block"):
            block_mask[:, :, :, :1]

        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=9,
            KV_LEN=10,
            BLOCK_SIZE=4,
            device=device,
        )

        self.assertEqual(block_mask[:, :, :].shape, (1, 1, 9, 10))
        self.assertEqual(block_mask[:, :, 1].shape, (1, 1, 4, 10))
        self.assertEqual(block_mask[:, :, 1:].shape, (1, 1, 5, 10))
        self.assertEqual(block_mask[:, :, :0].shape, (1, 1, 0, 10))
        self.assertEqual(block_mask[:, :, ::2].shape, (1, 1, 5, 10))
        self.assertEqual(
            block_mask[:, :, ::2].to_dense(), block_mask.to_dense()[:, :, ::2, :]
        )

        tail_q_block = block_mask[:, :, 2]
        self.assertEqual(tail_q_block.shape, (1, 1, 1, 10))
        self.assertEqual(block_mask[:, :, -1].shape, tail_q_block.shape)
        self.assertEqual(tail_q_block.seq_lengths, (1, 10))
        self.assertEqual(tail_q_block.to_dense(), block_mask.to_dense()[:, :, 2:3, :])
        if tail_q_block.q_indices is None:
            raise AssertionError("Expected q_indices to be recomputed")
        self.assertEqual(
            tail_q_block.q_indices, torch.zeros_like(tail_q_block.q_indices)
        )

        tail_q_block_tensor_index = block_mask[:, :, torch.tensor([2], device=device)]
        self.assertEqual(tail_q_block_tensor_index.shape, (1, 1, 4, 10))

    @supported_platform
    def test_getitem(self, device):
        # [FUNCTION DOC] Indexing a block mask by batch, head, negative indices, slices and a
        # query tensor index gives the expected shapes of block counts and index tensors
        # (AssertionError otherwise) and matching values to the sliced originals, including full
        # block tensors.
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod shifted by a captured per-batch offset
            # times 128.
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 512, 512, device=device)
        if block_mask.kv_num_blocks.shape != (4, 2, 4):
            raise AssertionError(
                f"Expected shape (4, 2, 4), got {block_mask.kv_num_blocks.shape}"
            )
        if block_mask.kv_indices.shape != (4, 2, 4, 4):
            raise AssertionError(
                f"Expected shape (4, 2, 4, 4), got {block_mask.kv_indices.shape}"
            )

        # Index on batch dimension
        new_block_mask = block_mask[0]
        if new_block_mask.kv_num_blocks.shape != (1, 2, 4):
            raise AssertionError(
                f"Expected shape (1, 2, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 2, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 2, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # Index on batch and head dimension
        new_block_mask = block_mask[0, 1]
        if new_block_mask.kv_num_blocks.shape != (
            1,
            1,
            4,
        ):
            raise AssertionError(
                f"Expected shape (1, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # Index on batch and head dimension with -1 semantics
        new_block_mask = block_mask[-1, -2]
        if new_block_mask.kv_num_blocks.shape != (
            1,
            1,
            4,
        ):
            raise AssertionError(
                f"Expected shape (1, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # slicing on batch and head dimension
        new_block_mask = block_mask[0:2, 1:2]
        if new_block_mask.kv_num_blocks.shape != (2, 1, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (2, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # slicing on batch, head, and query dimension
        new_block_mask = block_mask[0:2, 1:2, torch.tensor([1], dtype=torch.int32)]
        if new_block_mask.kv_num_blocks.shape != (2, 1, 1):
            raise AssertionError(
                f"Expected shape (2, 1, 1), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (2, 1, 1, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 1, 4), got {new_block_mask.kv_indices.shape}"
            )

        # slicing on batch, head, and query dimension
        q_index = torch.tensor([0], dtype=torch.int32)
        new_block_mask = block_mask[:, :, q_index]

        self.assertEqual(new_block_mask.kv_num_blocks.ndim, 3)
        self.assertEqual(new_block_mask.kv_indices.ndim, 4)
        torch.testing.assert_close(
            new_block_mask.kv_num_blocks,
            block_mask.kv_num_blocks[:, :, q_index],
        )
        torch.testing.assert_close(
            new_block_mask.kv_indices, block_mask.kv_indices[:, :, q_index, :]
        )

        if block_mask.full_kv_num_blocks is not None:
            if new_block_mask.full_kv_num_blocks is None:
                raise AssertionError("Expected full_kv_num_blocks to not be None")
            if new_block_mask.full_kv_indices is None:
                raise AssertionError("Expected full_kv_indices to not be None")
            torch.testing.assert_close(
                new_block_mask.full_kv_num_blocks,
                block_mask.full_kv_num_blocks[:, :, q_index],
            )
            torch.testing.assert_close(
                new_block_mask.full_kv_indices,
                block_mask.full_kv_indices[:, :, q_index, :],
            )

    @supported_platform
    def test_block_mask_positional_constructor_preserves_block_size(self, device):
        # [FUNCTION DOC] Building a BlockMask from positional arguments taken from a reference
        # mask keeps the block size and mask_mod, and leaves dq_write_order as None.
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        ref = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, BLOCK_SIZE=64, device=device
        )
        block_mask = BlockMask(
            ref.seq_lengths,
            ref.kv_num_blocks,
            ref.kv_indices,
            ref.full_kv_num_blocks,
            ref.full_kv_indices,
            ref.q_num_blocks,
            ref.q_indices,
            ref.full_q_num_blocks,
            ref.full_q_indices,
            ref.BLOCK_SIZE,
            ref.mask_mod,
        )

        self.assertEqual(block_mask.BLOCK_SIZE, ref.BLOCK_SIZE)
        self.assertIs(block_mask.mask_mod, ref.mask_mod)
        self.assertIsNone(block_mask.dq_write_order)

    @supported_platform
    def test_block_mask_transform_preserves_dq_write_order(self, device):
        # [FUNCTION DOC] A block mask created with a dq write order keeps it (and the kv order
        # flag) after batch indexing, adjusting size and moving to a device.
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=2,
            H=2,
            Q_LEN=256,
            KV_LEN=256,
            BLOCK_SIZE=128,
            device=device,
            compute_dq_write_order=True,
            dq_kv_order=True,
        )

        sliced = block_mask[0]
        self.assertIsNotNone(sliced.dq_write_order)
        self.assertEqual(sliced._dq_kv_order(), True)

        adjusted = block_mask._adjust(128, 128)
        self.assertIsNotNone(adjusted.dq_write_order)
        self.assertEqual(adjusted._dq_kv_order(), True)

        moved = block_mask.to(device)
        self.assertIsNotNone(moved.dq_write_order)
        self.assertEqual(moved._dq_kv_order(), True)

    @supported_platform
    def test_sliced_blockmask_mask_mod_error(self, device):
        # [FUNCTION DOC] Attention with a block mask sliced to one query block compiled raises
        # RuntimeError that the mask_mod of a sliced BlockMask cannot be used.
        """Test that sliced BlockMask raises helpful error when used with flex_attention"""

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        base_mask = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, device=device
        )
        sliced_mask = base_mask[:, :, 0]

        q = torch.randn(1, 1, 1, 64, device=device)
        k = torch.randn(1, 1, 256, 64, device=device)
        v = torch.randn(1, 1, 256, 64, device=device)

        compiled_fa = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot use mask_mod from a sliced BlockMask"
        ):
            compiled_fa(q, k, v, block_mask=sliced_mask)

    @supported_platform
    def test_block_mask_device_change(self, device):
        # [FUNCTION DOC] Block mask index tensors are on the creation device, move to CPU with the
        # to method and back; raises AssertionError on any mismatch.
        device = torch.device(device)
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod shifted by a captured per-batch offset
            # times 128.
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 512, 512, device=device)
        if block_mask.kv_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_indices.device.type}"
            )
        if block_mask.kv_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_num_blocks.device.type}"
            )
        if block_mask.q_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_indices.device.type}"
            )
        if block_mask.q_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_num_blocks.device.type}"
            )

        block_mask = block_mask.to("cpu")
        if not block_mask.kv_indices.is_cpu:
            raise AssertionError("Expected kv_indices to be on CPU")
        if not block_mask.kv_num_blocks.is_cpu:
            raise AssertionError("Expected kv_num_blocks to be on CPU")
        if not block_mask.q_indices.is_cpu:
            raise AssertionError("Expected q_indices to be on CPU")
        if not block_mask.q_num_blocks.is_cpu:
            raise AssertionError("Expected q_num_blocks to be on CPU")

        block_mask = block_mask.to(device)
        if block_mask.kv_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_indices.device.type}"
            )
        if block_mask.kv_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_num_blocks.device.type}"
            )
        if block_mask.q_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_indices.device.type}"
            )
        if block_mask.q_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_num_blocks.device.type}"
            )

    @supported_platform
    def test_compiling_create_block_mask(self, device):
        # [FUNCTION DOC] Fullgraph compiled create_block_mask with a segment-aware causal mask
        # returns a BlockMask with expected block count and index shapes for 512 length.
        seq = torch.arange(512, device=device) // 127

        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: mask_mod true where causal and in the same segment.
            return (q >= kv) & (seq[q] == seq[kv])

        block_mask = torch.compile(create_block_mask, fullgraph=True)(
            mask_mod, 1, 1, 512, 512, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((1, 1, 4)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((1, 1, 4, 4)))

    @supported_platform
    def test_compiling_create_block_mask_no_recompile(self, device):
        # [FUNCTION DOC] Compiled create_block_mask is called with three sizes: the first compile,
        # a second one from automatic dynamic shapes (aot_autograd ok counter becomes 2), and a
        # third that reuses it (counter stays 2); shapes of results are checked.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        torch._dynamo.reset()
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 2, 4, 1024, 1024, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((2, 4, 8)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((2, 4, 8, 8)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 1)

        # automatic dynamic shapes triggered and recompilation.
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 4, 8, 2048, 2048, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((4, 8, 16)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((4, 8, 16, 16)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

        # no recompilation.
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 6, 16, 3072, 3072, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((6, 16, 24)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((6, 16, 24, 24)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    def test_block_mask_viz(self, device):
        # [FUNCTION DOC] Checks the string visualization of a causal block mask against an inline
        # expected text (with non-printable characters mapped to at-signs and spaces to s) and
        # that a batch-dependent mask reports 29.10 percent sparsity.
        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 2048, 2048, device=device)

        def replace_non_printable(s):
            # [FUNCTION DOC] HELPER: maps each non-printable character to an at-sign and spaces to
            # the letter s, so the visualization compares stably.
            def replace(c):
                # [FUNCTION DOC] HELPER: character mapping used by the joined replacement.
                if c not in string.printable:
                    return "@"
                elif c == " ":
                    return "s"
                return c

            return "".join(replace(c) for c in s)

        self.assertExpectedInline(
            replace_non_printable(str(block_mask)),
            """\
BlockMask(shape=(1,s1,s2048,s2048),ssparsity=46.88%,s
(0,s0)
@@ssssssssssssssssssssssssssssss
@@@@ssssssssssssssssssssssssssss
@@@@@@ssssssssssssssssssssssssss
@@@@@@@@ssssssssssssssssssssssss
@@@@@@@@@@ssssssssssssssssssssss
@@@@@@@@@@@@ssssssssssssssssssss
@@@@@@@@@@@@@@ssssssssssssssssss
@@@@@@@@@@@@@@@@ssssssssssssssss
@@@@@@@@@@@@@@@@@@ssssssssssssss
@@@@@@@@@@@@@@@@@@@@ssssssssssss
@@@@@@@@@@@@@@@@@@@@@@ssssssssss
@@@@@@@@@@@@@@@@@@@@@@@@ssssssss
@@@@@@@@@@@@@@@@@@@@@@@@@@ssssss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@ssss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
)""",
        )

        offset = torch.arange(8, device=device)

        def causal_offset_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod shifted by the per-batch offset times 128.
            return (q + offset[b] * 128) >= kv

        block_mask = create_block_mask(
            causal_offset_mask, 8, 1, 2048, 2048, device=device
        )
        str_block_mask = str(block_mask)
        self.assertTrue("sparsity=29.10" in str_block_mask)

    def generate_test_inputs(self, full_seq_len: bool, device):
        # [FUNCTION DOC] HELPER: builds small kv block metadata for one query block and two kv
        # blocks. With full_seq_len true: one partial block (index 1) and one full block (index
        # 0); otherwise two partial blocks and no full ones. Returns kv_num_blocks, kv_indices,
        # full_kv_num_blocks and full_kv_indices.
        if full_seq_len:
            kv_num_blocks = torch.tensor([1], dtype=torch.int32, device=device).view(
                1, 1, 1
            )
            kv_indices = torch.tensor([1, -1], dtype=torch.int32, device=device).view(
                1, 1, 1, 2
            )
            full_kv_num_blocks = torch.tensor(
                [1], dtype=torch.int32, device=device
            ).view(1, 1, 1)
            full_kv_indices = torch.tensor(
                [0, -1], dtype=torch.int32, device=device
            ).view(1, 1, 1, 2)
        else:
            kv_num_blocks = torch.tensor([2], dtype=torch.int32, device=device).view(
                1, 1, 1
            )
            kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device).view(
                1, 1, 1, 2
            )
            full_kv_indices = None
            full_kv_num_blocks = None
        return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices

    @supported_platform
    @common_utils.parametrize("full_indices", [False, True])
    def test_from_kv_blocks(self, device, full_indices: bool):
        # [FUNCTION DOC] BlockMask.from_kv_blocks keeps the kv metadata and derives the transposed
        # q metadata (with or without full blocks) checked against hand-written tensors.
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ) = self.generate_test_inputs(full_indices, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices
        )

        self.assertIsInstance(block_mask, BlockMask)
        torch.testing.assert_close(block_mask.kv_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.kv_indices, kv_indices)

        if full_indices:
            torch.testing.assert_close(
                block_mask.full_kv_num_blocks, full_kv_num_blocks
            )
            torch.testing.assert_close(block_mask.full_kv_indices, full_kv_indices)
            torch.testing.assert_close(
                block_mask.q_num_blocks,
                torch.tensor([0, 1], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )
            torch.testing.assert_close(
                block_mask.full_q_num_blocks,
                torch.tensor([1, 0], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.full_q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )

        else:
            torch.testing.assert_close(
                block_mask.q_num_blocks,
                torch.tensor([1, 1], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )
            self.assertIsNone(block_mask.full_kv_num_blocks)
            self.assertIsNone(block_mask.full_kv_indices)
            self.assertIsNone(block_mask.full_q_num_blocks)
            self.assertIsNone(block_mask.full_q_indices)

    @supported_platform
    def test_from_kv_blocks_unpadded_kv_indices_with_seq_lengths(self, device):
        # [FUNCTION DOC] from_kv_blocks with kv index rows of width one and explicit sequence
        # lengths 96 gives shape 96 by 96 and the derived q metadata matches the kv metadata.
        kv_num_blocks = torch.ones((1, 1, 3), dtype=torch.int32, device=device)
        kv_indices = torch.arange(3, dtype=torch.int32, device=device).view(1, 1, 3, 1)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=32,
            seq_lengths=(96, 96),
        )

        self.assertEqual(block_mask.shape, (1, 1, 96, 96))
        torch.testing.assert_close(block_mask.kv_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.kv_indices, kv_indices)
        torch.testing.assert_close(block_mask.q_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.q_indices[..., :1], kv_indices)

    @supported_platform
    def test_from_kv_blocks_fullgraph_compile_with_inferred_seq_lengths(self, device):
        # [FUNCTION DOC] from_kv_blocks inside a fullgraph compiled function with inferred
        # sequence lengths compiles into one frame/graph and returns the expected q metadata.
        def from_kv_blocks_q_metadata(kv_num_blocks, kv_indices):
            # [FUNCTION DOC] HELPER: builds a mask from kv metadata and returns its q block counts
            # and q indices.
            block_mask = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices)
            return block_mask.q_num_blocks, block_mask.q_indices

        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)
        counter = CompileCounterWithBackend("aot_eager")
        q_num_blocks, q_indices = torch.compile(
            from_kv_blocks_q_metadata, backend=counter, fullgraph=True
        )(kv_num_blocks, kv_indices)

        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(len(counter.graphs), 1)
        torch.testing.assert_close(
            q_num_blocks,
            torch.tensor([1, 1], dtype=torch.int32, device=device).view(1, 1, 2),
        )
        torch.testing.assert_close(
            q_indices,
            torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
        )

    @supported_platform
    def test_block_size(self, device):
        # [FUNCTION DOC] from_kv_blocks uses the default sparse block size unless BLOCK_SIZE
        # is given.
        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)
        block_mask = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices)
        self.assertEqual(
            block_mask.BLOCK_SIZE,
            (_DEFAULT_SPARSE_BLOCK_SIZE, _DEFAULT_SPARSE_BLOCK_SIZE),
        )

        custom_block_size = (64, 64)
        block_mask_custom = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, BLOCK_SIZE=custom_block_size
        )
        self.assertEqual(block_mask_custom.BLOCK_SIZE, custom_block_size)

    @supported_platform
    def test_upcast_appropriately(self, device):
        # [FUNCTION DOC] Compiles attention in float16 with a score_mod that takes the log of a
        # captured float16 scalar (checks upcast handling; no value assertions).
        q = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        k = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        v = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        mass = torch.ones((1), dtype=torch.float16, device=device)

        def score_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the log of the captured scalar to the score.
            return score + torch.log(mass[0])

        torch.compile(flex_attention)(q, k, v, score_mod=score_mod)

    @supported_platform
    def test_init_mismatched_full_kv(self, device):
        # [FUNCTION DOC] Constructing a BlockMask with full kv block counts but no full kv indices
        # must raise AssertionError.
        kv_num_blocks, kv_indices, full_kv_num_blocks, _ = self.generate_test_inputs(
            True, device
        )

        with self.assertRaises(AssertionError):
            BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=None,  # Mismatched, should raise error
                q_num_blocks=kv_num_blocks,
                q_indices=kv_indices,
                full_q_num_blocks=None,
                full_q_indices=None,
                BLOCK_SIZE=(64, 64),
                mask_mod=noop_mask,
                seq_lengths=(1, 1),
            )

    @supported_platform
    def test_init_mismatched_full_q(self, device):
        # [FUNCTION DOC] Constructing a BlockMask with full q block counts but no full q indices
        # must raise AssertionError.
        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)

        with self.assertRaises(AssertionError):
            BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=None,
                full_kv_indices=None,
                q_num_blocks=kv_num_blocks,
                q_indices=kv_indices,
                full_q_num_blocks=kv_num_blocks,
                full_q_indices=None,  # Mismatched, should raise error
                BLOCK_SIZE=(64, 64),
                mask_mod=noop_mask,
                seq_lengths=(1, 1),
            )

    @supported_platform
    @expected_not_implemented_on_mps  # no captured buffers yet
    def test_doc_mask_clamped_repro(self, device):
        # [FUNCTION DOC] Not on MPS (MPS raises NotImplementedError): builds a document mask_mod
        # from seeded random document lengths and checks that two compiled create_block_mask
        # calls give equal kv and full-kv indices; then, for five slightly different lengths,
        # builds an index-clamped document mask and runs compiled attention with it, checking
        # only that it runs.
        def _offsets_to_doc_ids_tensor(offsets):
            # [FUNCTION DOC] HELPER: converts document offsets to a per-token document id tensor
            # by repeating each id by its document length.
            device = offsets.device
            counts = offsets[1:] - offsets[:-1]
            return torch.repeat_interleave(
                torch.arange(len(counts), device=device, dtype=torch.int32), counts
            )

        def length_to_offsets(lengths: list[int], device: str | torch.device) -> Tensor:
            # [FUNCTION DOC] HELPER: returns the cumulative offsets (starting at zero) of the
            # given lengths.
            offsets = [0]
            offsets.extend(lengths)
            offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
            offsets = torch.cumsum(offsets, dim=-1)
            return offsets

        def generate_doc_mask_mod(offsets: Tensor) -> _mask_mod_signature:
            # [FUNCTION DOC] HELPER: builds a same-document mask_mod from offsets.
            document_id = _offsets_to_doc_ids_tensor(offsets)

            def doc_mask_mod(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: true where q and kv share a document id.
                same_doc = document_id[q_idx] == document_id[kv_idx]
                return same_doc

            return doc_mask_mod

        random.seed(0)

        def generate_random_lengths(total_length, num_documents):
            # [FUNCTION DOC] HELPER: random positive integer lengths summing to the total across
            # the requested number of documents using the seeded random module.
            lengths = [1] * num_documents
            remaining_length = total_length - num_documents
            for _ in range(remaining_length):
                index = random.randint(0, num_documents - 1)
                lengths[index] += 1
            return lengths

        max_seq_len, doc_count = 128, 4
        SEQ_LEN = max_seq_len

        lengths = generate_random_lengths(max_seq_len, doc_count)
        offsets = length_to_offsets(lengths, device)

        document_causal_mask = generate_doc_mask_mod(offsets)
        block_mask_compiled = torch.compile(create_block_mask)(
            document_causal_mask,
            1,
            1,
            SEQ_LEN,
            SEQ_LEN,
            device=device,
        )
        block_mask = torch.compile(create_block_mask)(
            document_causal_mask,
            1,
            1,
            SEQ_LEN,
            SEQ_LEN,
            device=device,
        )
        self.assertEqual(block_mask_compiled.kv_indices, block_mask.kv_indices)
        self.assertEqual(
            block_mask_compiled.full_kv_indices, block_mask.full_kv_indices
        )
        for i in range(5):
            lengths = generate_random_lengths(1024 + i, 5)
            offsets = length_to_offsets(lengths, device)
            doc_ids = _offsets_to_doc_ids_tensor(offsets)

            def doc_mask_mod(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: mask_mod true where the clamped q and kv positions share
                # a document id.
                return (
                    doc_ids[q_idx.clamp(0, doc_ids.shape[0] - 1)]
                    == doc_ids[kv_idx.clamp(0, doc_ids.shape[0] - 1)]
                )

            q, k, v = (
                torch.randn(1, 12, 1024 + i, 64, device=device) for _ in range(3)
            )
            block_mask = create_block_mask(
                doc_mask_mod, None, None, 1024 + i, 1024 + i, device=device
            )
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    def test_eager_tracing_correctness(self, device):
        # [FUNCTION DOC] Fullgraph compiles a function calling GQA flex_attention with a
        # value head dim twice the key dim and viewing the output into extra dims; checks the
        # output shape.
        qk_dims = 64
        v_dims = 128
        q_heads = 4
        kv_heads = 2
        seq_len = 256
        batch_size = 1

        make_tensor = functools.partial(torch.randn, device=device, dtype=torch.float16)
        q = make_tensor(*(batch_size, q_heads, seq_len, qk_dims))
        k = make_tensor(*(batch_size, kv_heads, seq_len, qk_dims))
        v = make_tensor(*(batch_size, kv_heads, seq_len, v_dims))

        def flex_attention_fn():
            # [FUNCTION DOC] HELPER: runs attention on the captured tensors and views the last dim
            # as 2 by 64.
            out = flex_attention(q, k, v, enable_gqa=True)
            return out.view(batch_size, q_heads, seq_len, 2, 64)

        # Run with compilation
        compiled_fn = torch.compile(flex_attention_fn, fullgraph=True)
        result = compiled_fn()

        # Assert expected output shape
        expected_shape = (batch_size, q_heads, seq_len, 2, 64)
        self.assertEqual(
            result.shape,
            expected_shape,
            lambda msg: f"{msg}\nExpected output shape {expected_shape}, but got {result.shape}",
        )

    @supported_platform
    @skip_on_xpu
    @skip_on_mps  # uses torch.cuda.CUDAGraph; CUDA-only
    def test_create_is_cuda_graphable(self, device):
        # [FUNCTION DOC] Not on XPU/MPS: captures create_block_mask inside a CUDA graph and
        # replays it.
        def mask_mod(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g):
            create_block_mask(mask_mod, None, None, 256, 256)

        g.replay()

    @common_utils.parametrize("compile", [False, True])
    @supported_platform
    @expected_not_implemented_on_mps  # no backward yet
    def test_block_mask_vs_sequence_lengths(self, device, compile):
        # [FUNCTION DOC] Eager or compiled attention rejects a block mask created for a different
        # sequence length (larger or smaller than the inputs) with RuntimeError "block_mask was
        # created for"; a matching length works. MPS raises NotImplementedError.
        if compile:
            flex_attention_call = torch.compile(flex_attention)
        else:
            flex_attention_call = flex_attention

        def mask_mod(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        def create_inputs(S):
            # [FUNCTION DOC] HELPER: returns requires-grad float16 q, k, v of the given
            # sequence length.
            q, k, v = (
                torch.randn(
                    1, 8, S, 64, dtype=torch.float16, requires_grad=True, device=device
                )
                for _ in range(3)
            )
            return q, k, v

        block_mask = create_block_mask(mask_mod, None, None, 1024, 1024, device=device)
        flex_attention_call(*create_inputs(1024), block_mask=block_mask)
        with self.assertRaisesRegex(RuntimeError, "block_mask was created for"):
            flex_attention_call(*create_inputs(2048), block_mask=block_mask)

        block_mask = create_block_mask(mask_mod, None, None, 1023, 1023, device=device)
        with self.assertRaisesRegex(RuntimeError, "block_mask was created for"):
            flex_attention_call(*create_inputs(1024), block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    def test_block_mask_check_does_not_specialize_backed_dynamic_length(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: compiles attention with block masks for lengths 320,
        # 256 and 192 (two frames total) while hooking dynamo guard creation to collect guard
        # code; the last guard set must tie the block mask sequence lengths to the q and k sizes
        # (not specialize), and a stale mask built for 320 used with 512 raises an error about a
        # smaller length. The guard hook is restored in a finally block.
        def mask_mod(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        dtype = device_configs[torch.device(device).type].dtypes_fast[0]

        def create_inputs(S):
            # [FUNCTION DOC] HELPER: builds q, k, v of length S and the matching block mask.
            q, k, v = (
                torch.randn(1, 1, S, 64, dtype=dtype, device=device) for _ in range(3)
            )
            block_mask = create_block_mask(mask_mod, None, None, S, S, device=device)
            return q, k, v, block_mask

        counter = CompileCounterWithBackend("eager")
        guard_code_parts = []

        @torch.compile(fullgraph=True, backend=counter)
        def flex_attention_call(q, k, v, block_mask):
            # [FUNCTION DOC] HELPER: compiled fullgraph attention call with a block mask argument.
            return flex_attention(q, k, v, block_mask=block_mask)

        def collect_guard_code_parts(guard_wrapper, f_locals, builder):
            # [FUNCTION DOC] HELPER: guard testing hook that appends the verbose code parts of all
            # epilogue lambda guards of the guard wrapper to the shared list.
            parts = []
            for guard in guard_wrapper.root.get_epilogue_lambda_guards():
                parts.extend(guard.verbose_code_parts())
            guard_code_parts.append(parts)

        old_hook = torch._dynamo.guards.guard_manager_testing_hook_fn
        torch._dynamo.guards.guard_manager_testing_hook_fn = collect_guard_code_parts
        try:
            with torch.no_grad():
                for S in (320, 256, 192):
                    self.assertEqual(
                        flex_attention_call(*create_inputs(S)).shape, (1, 1, S, 64)
                    )
        finally:
            torch._dynamo.guards.guard_manager_testing_hook_fn = old_hook

        self.assertEqual(counter.frame_count, 2)
        dynamic_guard_code = "\n".join(guard_code_parts[-1])
        self.assertIn(
            "L['block_mask'].seq_lengths[0] == L['q'].size()[2]",
            dynamic_guard_code,
        )
        self.assertIn(
            "L['block_mask'].seq_lengths[1] == L['k'].size()[2]",
            dynamic_guard_code,
        )

        stale_block_mask = create_block_mask(
            mask_mod, None, None, 320, 320, device=device
        )
        with self.assertRaisesRegex(
            Exception, "block_mask was created for a smaller length"
        ):
            flex_attention_call(*create_inputs(512)[:3], stale_block_mask)

    @supported_platform
    @common_utils.parametrize("full_indices", [False, True])
    def test_from_kv_blocks_without_q_computation(self, device, full_indices: bool):
        # [FUNCTION DOC] from_kv_blocks with q-block computation disabled keeps the kv metadata
        # and leaves all q metadata None (full kv metadata only when provided).
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ) = self.generate_test_inputs(full_indices, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            compute_q_blocks=False,
        )

        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks, kv_num_blocks)
        self.assertEqual(block_mask.kv_indices, kv_indices)

        self.assertIsNone(block_mask.q_num_blocks)
        self.assertIsNone(block_mask.q_indices)
        self.assertIsNone(block_mask.full_q_num_blocks)
        self.assertIsNone(block_mask.full_q_indices)

        if full_indices:
            self.assertEqual(block_mask.full_kv_num_blocks, full_kv_num_blocks)
            self.assertEqual(block_mask.full_kv_indices, full_kv_indices)
        else:
            self.assertIsNone(block_mask.full_kv_num_blocks)
            self.assertIsNone(block_mask.full_kv_indices)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific "BlockMask q_indices is None" runtime error
    def test_backward_error_with_none_q_indices(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test: a block mask without q metadata works under no_grad but
        # with grad enabled compiled attention raises RuntimeError that q_indices is None and the
        # mask must be created with q block computation.
        N_BLOCKS = 4
        B, H, S, D = 1, 1, 128, 64
        S_KV = N_BLOCKS * S

        kv_num_blocks = torch.tensor([[[N_BLOCKS]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )

        q = torch.randn(
            B, H, S, D, dtype=torch.float16, device=device, requires_grad=True
        )
        k = torch.randn(
            B, H, S_KV, D, dtype=torch.float16, device=device, requires_grad=True
        )
        v = torch.randn(
            B, H, S_KV, D, dtype=torch.float16, device=device, requires_grad=True
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)

        with torch.no_grad():
            out_no_grad = flex_compile(q, k, v, block_mask=block_mask)
            self.assertEqual(out_no_grad.shape, (B, H, S, D))

        # Forward pass with grad enabled should error immediately
        with self.assertRaisesRegex(
            RuntimeError,
            "BlockMask q_indices is None. Backward pass requires q_indices to be computed. "
            "Please create the BlockMask with compute_q_blocks=True",
        ):
            flex_compile(q, k, v, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backwards yet
    def test_flex_attention_poisoned_rel_logits(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): captured relative logits whose
        # unused second batch half is NaN; compiled attention output and q, k, v grads must stay
        # finite (raises AssertionError otherwise).
        B = 1
        H = 1
        S = 1025
        D = 64
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]
        rel_logits = torch.randn(2 * B, H, S, S, device=device)
        rel_logits[B:] = float("nan")

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: adds the captured relative logit for batch, head, q and kv.
            return score + rel_logits[b, h, q, kv]

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention)(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()

        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        if not k.grad.isfinite().all().item():
            raise AssertionError("k.grad contains non-finite values")
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backwards yet
    def test_flex_attention_poison_mod_fwd(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a score_mod dividing one by the
        # score can create non-finite values; compiled output, q grad and v grad must still be
        # finite (k grad check is commented out).
        """Div by score should cause our edge case handiling to NaN"""
        B = 1
        H = 1
        S = 257
        D = 16
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: score_mod returning the reciprocal of the score.
            return 1 / score

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention, backend="inductor")(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()
        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        # assert k.grad.isfinite().all().item()
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_flex_attention_poison_mod_bwd(self, device):
        # [FUNCTION DOC] Non-CPU test (MPS NotImplementedError): a score_mod taking the log of
        # positive scores can give NaN grads; compiled output, q grad and v grad must remain
        # finite (k grad check is commented out).
        """log score should cause our edge case handiling for NaN in grad score"""
        B = 1
        H = 1
        S = 257
        D = 16
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: score_mod returning the log of positive scores and the
            # score otherwise.
            return torch.where(score > 0, torch.log(score), score)

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention, backend="inductor")(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()
        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        # assert k.grad.isfinite().all().item()
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

    @supported_platform
    @skip_on_cpu
    def test_forward_pass_with_none_q_indices(self, device):
        # [FUNCTION DOC] Non-CPU test: forward only with a block mask lacking q metadata compiles
        # and returns a float16 tensor of the right shape.
        N_BLOCKS = 4
        B, H, S, D = 1, 1, 128, 64
        S_KV = N_BLOCKS * S

        kv_num_blocks = torch.tensor([[[N_BLOCKS]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )

        q = torch.randn(
            B,
            H,
            S,
            D,
            dtype=torch.float16,
            device=device,
        )
        k = torch.randn(
            B,
            H,
            S_KV,
            D,
            dtype=torch.float16,
            device=device,
        )
        v = torch.randn(
            B,
            H,
            S_KV,
            D,
            dtype=torch.float16,
            device=device,
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out = flex_compile(q, k, v, block_mask=block_mask)

        self.assertEqual(out.shape, (B, H, S, D))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.float16)

    @supported_platform
    def test_block_mask_operations_with_none_q_indices(self, device):
        # [FUNCTION DOC] A block mask built without q metadata still reports shape and block size,
        # and keeps q metadata None after indexing and after moving to CPU (when not on CPU).
        kv_num_blocks = torch.tensor([[[4]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )
        self.assertEqual(block_mask.shape, (1, 1, 128, 512))
        self.assertEqual(block_mask.BLOCK_SIZE, (128, 128))

        sliced_mask = block_mask[0]
        self.assertEqual(sliced_mask.shape, (1, 1, 128, 512))
        self.assertIsNone(sliced_mask.q_indices)
        self.assertIsNone(sliced_mask.q_num_blocks)

        # Test device movement
        if device != "cpu":
            cpu_mask = block_mask.to("cpu")
            self.assertEqual(cpu_mask.kv_num_blocks.device.type, "cpu")
            self.assertIsNone(cpu_mask.q_indices)

    @supported_platform
    def test_compute_dq_write_order_with_expanded_kv_indices(self, device):
        # [FUNCTION DOC] The dq write order computed from a block mask whose kv metadata is
        # expanded over the batch has batch size 2 and identical rows (and the same for the full
        # variant when present).
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, BLOCK_SIZE=128, device=device
        )
        expanded_kv_block_mask = BlockMask(
            seq_lengths=block_mask.seq_lengths,
            kv_num_blocks=block_mask.kv_num_blocks.expand(2, 1, -1),
            kv_indices=block_mask.kv_indices.expand(2, 1, -1, -1),
            full_kv_num_blocks=(
                block_mask.full_kv_num_blocks.expand(2, 1, -1)
                if block_mask.full_kv_num_blocks is not None
                else None
            ),
            full_kv_indices=(
                block_mask.full_kv_indices.expand(2, 1, -1, -1)
                if block_mask.full_kv_indices is not None
                else None
            ),
            q_num_blocks=block_mask.q_num_blocks,
            q_indices=block_mask.q_indices,
            full_q_num_blocks=block_mask.full_q_num_blocks,
            full_q_indices=block_mask.full_q_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=block_mask.mask_mod,
        )
        dq_write_order, dq_write_order_full = _compute_dq_write_order_from_block_mask(
            expanded_kv_block_mask
        )

        self.assertEqual(dq_write_order.shape[:2], (2, 1))
        self.assertEqual(dq_write_order[0], dq_write_order[1])
        if dq_write_order_full is not None:
            self.assertEqual(dq_write_order_full.shape[:2], (2, 1))
            self.assertEqual(dq_write_order_full[0], dq_write_order_full[1])

    @supported_platform
    def test_create_block_mask_defaults_to_spt_dq_write_order(self, device):
        # [FUNCTION DOC] create_block_mask with dq write order computation stores a dq write order
        # and defaults to the kv-ordered mode (true).
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=256,
            KV_LEN=256,
            BLOCK_SIZE=128,
            device=device,
            compute_dq_write_order=True,
        )

        self.assertIsNotNone(block_mask.dq_write_order)
        self.assertEqual(block_mask._dq_kv_order(), True)

    @supported_platform
    def test_create_block_mask_rejects_non_bool_dq_kv_order(self, device):
        # [FUNCTION DOC] create_block_mask rejects a non-bool dq_kv_order (an int tensor) with
        # ValueError saying it must be a bool.
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        dq_kv_order = torch.tensor([[[1, 0]]], dtype=torch.int32, device=device)
        with self.assertRaisesRegex(
            ValueError,
            "dq_kv_order must be a bool",
        ):
            create_block_mask(
                causal_mask,
                B=1,
                H=1,
                Q_LEN=256,
                KV_LEN=256,
                BLOCK_SIZE=128,
                device=device,
                compute_dq_write_order=True,
                dq_kv_order=dq_kv_order,
            )

    @supported_platform
    def test_from_kv_blocks_accepts_dq_metadata(self, device):
        # [FUNCTION DOC] from_kv_blocks accepts dq metadata (write order and a tensor or bool kv
        # order), stores it, and raises ValueError for an invalid kv order or for a full write
        # order without the matching data; indexing or adjusting a mask with a tensor kv order
        # raises NotImplementedError.
        kv_num_blocks = torch.tensor([[[1, 2]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor(
            [[[[0, 0], [0, 1]]]], dtype=torch.int32, device=device
        )
        dq_write_order = torch.zeros((1, 1, 2, 2), dtype=torch.int32, device=device)
        dq_kv_order = torch.tensor([[[1, 0]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            dq_write_order=dq_write_order,
            dq_kv_order=dq_kv_order,
        )
        spt_block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            dq_write_order=dq_write_order,
            dq_kv_order=True,
        )

        self.assertEqual(block_mask.dq_write_order, dq_write_order)
        self.assertEqual(block_mask.dq_kv_order, dq_kv_order)
        self.assertIsNone(block_mask.dq_write_order_full)
        self.assertEqual(spt_block_mask.dq_write_order, dq_write_order)
        self.assertEqual(spt_block_mask._dq_kv_order(), True)

        with self.assertRaisesRegex(ValueError, "dq_kv_order must be"):
            BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                dq_write_order=dq_write_order,
                dq_kv_order=1,
            )
        with self.assertRaisesRegex(ValueError, "dq_write_order_full requires"):
            BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                dq_write_order_full=dq_write_order,
            )

        with self.assertRaisesRegex(NotImplementedError, "tensor dq_kv_order"):
            block_mask[0]
        with self.assertRaisesRegex(NotImplementedError, "tensor dq_kv_order"):
            block_mask._adjust(128, 128)

    @supported_platform
    @skip_on_cpu
    def test_broadcasted_head_block_mask(self, device):
        # [FUNCTION DOC] Non-CPU test: decoding with a head-broadcast block mask sliced to one
        # query block, a mask_mod shifted by a position tensor and sequence length 1, compiled
        # dynamic; output equals SDPA over the cropped caches within 1e-3.
        torch.manual_seed(42)

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        def get_mask_mod_with_offset(mask_mod, offset_tensor):
            # [FUNCTION DOC] HELPER: wraps a mask_mod so the q index is shifted by a
            # tensor offset.
            def _mask_mod(b, h, q, kv):
                # [FUNCTION DOC] HELPER: calls the wrapped mask_mod with q plus the offset.
                return mask_mod(b, h, q + offset_tensor, kv)

            return _mask_mod

        B, T, H, D, current_pos = 4, 512, 8, 64, 128
        dtype = torch.float32

        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_cache = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v_cache = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Keep future tokens tiny to avoid numerical issues when using full caches
        k_cache[:, :, current_pos + 1 :, :] = (
            torch.randn_like(k_cache[:, :, current_pos + 1 :, :]) * 1e-10
        )
        v_cache[:, :, current_pos + 1 :, :] = (
            torch.randn_like(v_cache[:, :, current_pos + 1 :, :]) * 1e-10
        )

        k_cropped = k_cache[:, :, : current_pos + 1, :]
        v_cropped = v_cache[:, :, : current_pos + 1, :]
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q, k_cropped, v_cropped, attn_mask=None
        )

        base_mask = create_block_mask(
            causal_mask,
            B=B,
            H=None,  # broadcast across heads
            Q_LEN=T,
            KV_LEN=T,
            device=device,
            _compile=True,
        )

        q_block_size = base_mask.BLOCK_SIZE[0]
        block_offset = current_pos // q_block_size
        mask_slice = base_mask[:, :, block_offset]

        offset_tensor = torch.tensor(current_pos, device=device)
        mask_slice.mask_mod = get_mask_mod_with_offset(
            base_mask.mask_mod, offset_tensor
        )
        mask_slice.seq_lengths = (1, mask_slice.seq_lengths[1])

        fa = torch.compile(flex_attention, dynamic=True)
        flex_output = fa(q, k_cache, v_cache, block_mask=mask_slice)

        self.assertEqual(flex_output, sdpa_output, atol=1e-3, rtol=1e-3)

    @supported_platform
    def test_pytree_flatten_unflatten(self, device):
        # [FUNCTION DOC] A block mask flattened and unflattened by its class methods keeps shape,
        # sparsity, every tensor attribute (or None) and every context attribute.
        """Test that BlockMask can be correctly flattened and unflattened using class methods."""

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        # Create a BlockMask with various attributes set
        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        # Flatten and unflatten using class methods
        tensors, context = block_mask._flatten()
        reconstructed_mask = BlockMask._unflatten(tensors, context)

        # Verify the reconstructed mask has the same attributes
        self.assertEqual(reconstructed_mask.shape, block_mask.shape)
        self.assertEqual(reconstructed_mask.sparsity(), block_mask.sparsity())

        # Verify all tensor attributes are equal (using _TENSOR_ATTRS)
        for attr_name in BlockMask._TENSOR_ATTRS:
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            if original_value is None:
                self.assertIsNone(
                    reconstructed_value,
                    lambda msg: f"{msg}\nTensor attribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertIsInstance(
                    original_value,
                    torch.Tensor,
                    lambda msg: f"{msg}\nExpected {attr_name} to be a Tensor",
                )
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    lambda msg: f"{msg}\nTensor attribute {attr_name} not equal after reconstruction",
                )

        # Verify all context attributes are equal (using _CONTEXT_ATTRS)
        for attr_name in BlockMask._CONTEXT_ATTRS:
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            self.assertEqual(
                original_value,
                reconstructed_value,
                lambda msg: f"{msg}\nContext attribute {attr_name} not equal after reconstruction",
            )

    @supported_platform
    def test_pytree_flatten_with_keys(self, device):
        # [FUNCTION DOC] The keyed flatten of a block mask yields one attribute-key entry per
        # present tensor and a context with one more entry than the context attribute list.
        """Test that BlockMask._flatten_with_keys works correctly for tracing."""

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        tensors_with_keys, context_with_keys = block_mask._flatten_with_keys()

        n_regular = sum(
            getattr(block_mask, attr) is not None for attr in BlockMask._TENSOR_ATTRS
        )
        self.assertEqual(len(tensors_with_keys), n_regular)
        self.assertEqual(len(context_with_keys), len(BlockMask._CONTEXT_ATTRS) + 1)

        from torch.utils._pytree import GetAttrKey

        for key, _tensor in tensors_with_keys:
            self.assertIsInstance(key, GetAttrKey)
            self.assertIsNotNone(key)

    @supported_platform
    def test_pytree_preserves_new_attributes(self, device):
        # [FUNCTION DOC] Checks that every attribute listed in the tensor and context
        # attribute lists survives flatten and unflatten, with the expected counts of tensors
        # and context entries.
        """
        Test that BlockMask._TENSOR_ATTRS and _CONTEXT_ATTRS are correctly defined
        and that flatten/unflatten preserves all attributes in these lists.

        """

        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        # Flatten and unflatten using class methods
        tensors, context = block_mask._flatten()
        reconstructed_mask = BlockMask._unflatten(tensors, context)

        # Verify the number of tensors and context values matches the present attributes
        n_regular = sum(
            getattr(block_mask, attr) is not None for attr in BlockMask._TENSOR_ATTRS
        )
        self.assertEqual(
            len(tensors),
            n_regular,
            "Number of tensors should match present tensor attributes",
        )
        self.assertEqual(
            len(context),
            len(BlockMask._CONTEXT_ATTRS) + 1,
            "Context values should include optional_tensor_attrs",
        )

        # Verify all attributes from the lists exist and are equal after reconstruction
        for attr_name in BlockMask._TENSOR_ATTRS + BlockMask._CONTEXT_ATTRS:
            self.assertTrue(
                hasattr(reconstructed_mask, attr_name),
                lambda msg: f"{msg}\nReconstructed mask missing attribute: {attr_name}",
            )
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            if isinstance(original_value, torch.Tensor):
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    lambda msg: f"{msg}\nTensor attribute {attr_name} not equal after reconstruction",
                )
            elif original_value is None:
                self.assertIsNone(
                    reconstructed_value,
                    lambda msg: f"{msg}\nAttribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertEqual(
                    original_value,
                    reconstructed_value,
                    lambda msg: f"{msg}\nAttribute {attr_name} not equal after reconstruction",
                )


@large_tensor_test_class("2GB", device=test_device[0])
class TestPagedAttention(InductorTestCase):
    # [CLASS DOC] Paged attention tests: page allocation, erase, logical-to-physical block mask
    # and mask_mod conversion, cache assign, and end-to-end paged attention against reference.
    # Decorated to require large tensor memory on the first test device.
    def setUp(self):
        # [FUNCTION DOC] HELPER: calls the parent setUp, then calls skipCPUIf with the
        # CPU-compile flag.
        # KNOWN ISSUE: skipCPUIf returns a decorator; the result is discarded, so nothing
        # is skipped.
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: compares compiled and reference errors against golden after
        # moving to CPU if devices differ. Fails on NaN, and fails when the compiled mean absolute
        # error exceeds the reference error times the fudge factor.
        if (
            golden_out.device != ref_out.device
            or golden_out.device != compiled_out.device
        ):
            # MPS golden runs on CPU (no float64); compare everything on CPU.
            golden_out = golden_out.cpu()
            ref_out = ref_out.cpu()
            compiled_out = compiled_out.cpu()
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def allocate_page_cache(self, n_pages: int, page_size: int, device: str):
        # [FUNCTION DOC] HELPER: creates a PagedAttention with the given pages and page size and a
        # fixed maximum batch size of 3.
        max_batch_size = 3
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)
        return paged_cache

    def cdiv(self, x, y):
        # [FUNCTION DOC] HELPER: ceiling division of x by y.
        return (x + y - 1) // y

    def roundup(self, x, y):
        # [FUNCTION DOC] HELPER: rounds x up to the next multiple of y.
        return (x + y - 1) // y * y

    @supported_platform
    def test_page_allocation(self, device):
        # [FUNCTION DOC] Reserving pages for lengths 8, 24 and 16 uses all 12 pages; reserving
        # more for batch 0 raises AssertionError saying there are no empty pages; erasing batch 1
        # frees pages so the reserve then succeeds.
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size, device=device)

        batch_reserve(paged_cache, torch.tensor([8, 24, 16]))

        with self.assertRaisesRegex(
            AssertionError, "requested 2 pages but there are only 0 empty pages"
        ):
            paged_cache.reserve(
                torch.tensor([0], device=device),
                torch.tensor([16], device=device),
            )

        paged_cache.erase(torch.tensor([1], device=device))
        paged_cache.reserve(
            torch.tensor([0], device=device),
            torch.tensor([16], device=device),
        )

    @supported_platform
    def test_allocate(self, device):
        # [FUNCTION DOC] Allocates, erases and reallocates pages for sequence lengths; checks
        # capacity (rounded up to page size) and the free page count each time, and that erasing
        # all batches resets capacity to zero and frees all pages.
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size, device=device)

        target_seq_len = torch.tensor([3, 11, 8])
        batch_reserve(paged_cache, target_seq_len)

        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # deallocate batch 1
        paged_cache.erase(torch.tensor([1], device=device))
        target_seq_len = torch.tensor([3, 0, 8])
        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # re-allocate
        target_seq_len = torch.tensor([7, 2, 10])
        batch_reserve(paged_cache, target_seq_len)
        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # deallocate all batches
        paged_cache.erase(torch.tensor([0, 1, 2]))
        self.assertEqual(paged_cache.capacity, torch.tensor([0, 0, 0]))
        self.assertEqual(len(paged_cache.empty_pages), n_pages)

    @supported_platform
    def test_convert_logical_block_mask(self, device):
        # [FUNCTION DOC] After several reserves with two batches, checks the page table, then
        # converts a causal logical block mask to the physical layout and compares kv and full-kv
        # block counts and indices with expected physical page orders.
        n_pages, page_size, max_batch_size, max_seq_len = 8, 128, 2, 512
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        batch_reserve(paged_cache, torch.tensor([100, 200], device=device))
        batch_reserve(paged_cache, torch.tensor([150, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([300, 512], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device=device),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        # Get a block mask
        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len, device=device
        )
        kv_len_tensor = torch.full(
            (max_batch_size,), max_seq_len, device=device, dtype=torch.int64
        )
        new_block_mask = paged_cache.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )

        zeros = [0, 0, 0, 0]
        # Check that the new block mask is correct
        expected_kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], device=device, dtype=torch.int32
        )
        expected_kv_indices = torch.tensor(
            [
                [
                    [
                        [0, 3, 5, 7, *zeros],
                        [3, 0, 5, 7, *zeros],
                        [5, 0, 3, 7, *zeros],
                        [7, 0, 3, 5, *zeros],
                    ]
                ],
                [
                    [
                        [2, 1, 4, 6, *zeros],
                        [1, 2, 4, 6, *zeros],
                        [4, 2, 1, 6, *zeros],
                        [6, 2, 1, 4, *zeros],
                    ]
                ],
            ],
            device=device,
            dtype=torch.int32,
        )
        expected_full_kv_num_blocks = torch.tensor(
            [[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device=device, dtype=torch.int32
        )
        expected_full_kv_indices = torch.tensor(
            [
                [
                    [
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                    ]
                ],
                [
                    [
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                    ]
                ],
            ],
            device=device,
            dtype=torch.int32,
        )
        self.assertEqual(new_block_mask.kv_num_blocks, expected_kv_num_blocks)
        self.assertEqual(new_block_mask.kv_indices, expected_kv_indices)
        self.assertEqual(new_block_mask.full_kv_num_blocks, expected_full_kv_num_blocks)
        self.assertEqual(new_block_mask.full_kv_indices, expected_full_kv_indices)

    @supported_platform
    def test_convert_mask_mod(self, device):
        # [FUNCTION DOC] Checks the page table and physical-to-logical mapping after reserves,
        # then that the converted causal mask_mod gives true or false for physical positions
        # according to their logical mapping (False when no logical block is mapped).
        n_pages, page_size, max_batch_size = 8, 128, 2
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        batch_reserve(paged_cache, torch.tensor([100, 200], device=device))
        batch_reserve(paged_cache, torch.tensor([150, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([300, 512], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device=device),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        expected_physical_to_logical = torch.tensor(
            [[0, -1, -1, 1, -1, 2, -1, 3], [-1, 1, 0, -1, 2, -1, 3, -1]],
            device=device,
        )
        self.assertEqual(paged_cache.physical_to_logical, expected_physical_to_logical)

        # Get a block mask
        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        converted_causal_mask = paged_cache.get_mask_mod(causal_mask)

        # Equivalent to: causal_mask(0, 0, 256, 128)
        self.assertEqual(converted_causal_mask(0, 0, 256, 384), True)
        # Equivalent to: causal_mask(0, 1, 256, 128)
        self.assertEqual(converted_causal_mask(0, 1, 256, 384), True)
        # Not found corresponding logical block
        self.assertEqual(converted_causal_mask(1, 0, 256, 384), False)
        # Equivalent to: causal_mask(1, 0, 64, 14)
        self.assertEqual(converted_causal_mask(1, 0, 64, 270), True)

    @supported_platform
    def test_update(self, device):
        # [FUNCTION DOC] Reserves pages for two batches and checks the page table; assigns
        # sequential k values for both batches into the paged key cache and asserts it equals the
        # expected layout in which batch 0 fills pages 0, 3, 5 and batch 1 fills pages 2, 1, 4 for
        # each head.
        dtype = torch.float32

        n_pages, page_size, max_batch_size, max_seq_len = 6, 2, 2, 6
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        n_heads, head_dim = 2, 3
        cache_shape = (1, n_heads, n_pages * page_size, head_dim)
        k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        batch_reserve(paged_cache, torch.tensor([1, 3], device=device))
        batch_reserve(paged_cache, torch.tensor([4, 5], device=device))
        batch_reserve(paged_cache, torch.tensor([6, 6], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, -1, -1, -1], [2, 1, 4, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        batch_idx = torch.arange(max_batch_size, device=device, dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        k = torch.arange(
            max_batch_size * n_heads * max_seq_len * head_dim,
            device=device,
            dtype=dtype,
        ).view(max_batch_size, n_heads, max_seq_len, head_dim)

        v = k.detach().clone()
        v_cache = k_cache.detach().clone()

        paged_cache.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        expected_cache = torch.tensor(
            [
                [
                    # h = 0
                    [
                        # page = 0
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        # page = 1
                        [42.0, 43.0, 44.0],
                        [45.0, 46.0, 47.0],
                        # page = 2
                        [36.0, 37.0, 38.0],
                        [39.0, 40.0, 41.0],
                        # page = 3
                        [6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0],
                        # page = 4
                        [48.0, 49.0, 50.0],
                        [51.0, 52.0, 53.0],
                        # page = 5
                        [12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0],
                    ],
                    # h = 1
                    [
                        # page = 0
                        [18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0],
                        # page = 1
                        [60.0, 61.0, 62.0],
                        [63.0, 64.0, 65.0],
                        # page = 2
                        [54.0, 55.0, 56.0],
                        [57.0, 58.0, 59.0],
                        # page = 3
                        [24.0, 25.0, 26.0],
                        [27.0, 28.0, 29.0],
                        # page = 4
                        [66.0, 67.0, 68.0],
                        [69.0, 70.0, 71.0],
                        # page = 5
                        [30.0, 31.0, 32.0],
                        [33.0, 34.0, 35.0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assertEqual(k_cache, expected_cache)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_paged_builtin_score_mods(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        # [FUNCTION DOC] Paged attention with each built-in score_mod: four batches with various
        # reserved lengths are written into a paged cache; compiled attention over the cache with
        # the converted block mask and score_mod must match the eager reference and float64 golden
        # (golden on CPU for MPS) within the fudge factor.
        n_pages, page_size, max_batch_size, max_seq_len = 32, 128, 4, 512
        n_heads, head_dim = 4, 16

        def causal_mask(b, h, q, kv):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len, device=device
        )
        q = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        k = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        v = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=False)

        if _is_mps_device(device):
            # MPS has no float64; the golden runs on CPU, so its score_mod
            # captures and block_mask must be relocated off-device too.
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=False,
            )
        else:
            sdpa_partial_gold = sdpa_partial

        golden_out = sdpa_partial_gold(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=device,
            dtype=dtype,
        )

        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)
        batch_reserve(paged_cache, torch.tensor([100, 200, 50, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([100, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 512], device=device))

        batch_idx = torch.arange(max_batch_size, device=device, dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        paged_cache.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        kv_len_tensor = torch.full(
            (max_batch_size,), max_seq_len, device=device, dtype=torch.int64
        )
        new_block_mask = paged_cache.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )

        compiled_sdpa = torch.compile(
            create_attention(
                paged_cache.get_score_mod(score_mod, kv_len=kv_len_tensor),
                block_mask,
                enable_gqa=False,
            )
        )
        paged_out = compiled_sdpa(q, k_cache, v_cache, block_mask=new_block_mask)

        with torch.no_grad():
            dtype = ref_out.dtype
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, paged_out, fudge_factor, "Out")


@dataclass
class Params:
    # [CLASS DOC] HELPER: dataclass of one learnable-bias test configuration (batch, heads,
    # sequence length, head dim, dtype, optional config string) whose string form encodes those
    # fields for test names.
    batch_size: int
    num_heads: int
    seq_length: int
    head_dim: int
    dtype: torch.dtype
    config_str: str | None = None

    def __str__(self):
        # [FUNCTION DOC] HELPER: returns a name string with batch, head, sequence length, head dim
        # and dtype name.
        return f"batch:{self.batch_size}_head:{self.num_heads}_seq_len:{self.seq_length}_headdim:{self.head_dim}_dtype:{str(self.dtype).split('.')[-1]}"


def get_params(dtypes: list[torch.dtype]) -> list[Params]:
    # [FUNCTION DOC] HELPER: returns Params for each combination of sequence length 37, 256 or 277
    # and the given dtypes, with batch 2, 4 heads and head dim 16.
    params = []
    seq_lengths = [37, 256, 277]
    for seq_len, dtype in product(seq_lengths, dtypes):
        params.append(
            Params(
                batch_size=2, num_heads=4, seq_length=seq_len, head_dim=16, dtype=dtype
            )
        )
    return params


ROCM_FLAKY_LEARNABLE_BIAS_CASES = {
    "head_specific_gate": {
        (2, 4, 37, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
    "relative_1d_bias": {
        (2, 4, 37, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
    "symmetric_bias": {
        (2, 4, 37, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
}


supports_learnable_bias = unittest.skipUnless(
    (
        (torch.xpu.is_available() and has_triton())
        or (torch.cuda.is_available() and has_triton())
        and (torch.cuda.get_device_capability() >= (8, 0) or torch.version.hip)
    ),
    "Requires Triton + A100 or Triton + ROCm",
)


@supports_learnable_bias
@large_tensor_test_class("2GB", device=test_device[0])
class TestLearnableBiases(InductorTestCase):
    # [CLASS DOC] Tests flex_attention with learnable (requires-grad) captured bias tensors of
    # many shapes: eager, compiled and float64 golden outputs and gradients are compared. Only
    # defined when Triton and a suitable GPU are available (see the decorator). Stores default
    # dtype and tolerance attributes in setUp.
    def setUp(self):
        # [FUNCTION DOC] HELPER: calls the parent setUp, calls skipCPUIf with the CPU-compile
        # flag, and sets default dtype float32 and tolerances of 3e-2.
        # KNOWN ISSUE: skipCPUIf returns a decorator whose result is discarded; the stored dtype
        # and tolerances are not used by the tests in this class.
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )
        self.dtype = torch.float32
        self.atol = 3e-2
        self.rtol = 3e-2

    def _init_tensors(self, params: Params, device: str):
        # [FUNCTION DOC] HELPER: returns random requires-grad query, key and value tensors with
        # the shape and dtype given by the parameters.
        make_tensor = functools.partial(
            torch.randn,
            (params.batch_size, params.num_heads, params.seq_length, params.head_dim),
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )
        return (make_tensor(), make_tensor(), make_tensor())

    def skip_rocm_flaky_learnable_bias_case(
        self, test_name: str, params: Params, mode: str
    ):
        # [FUNCTION DOC] HELPER: skips the test on ROCm when the (shape, dtype, mode) case appears
        # in the flaky list for the named test.
        """Skips parameterized ROCm flakes tracked by disabled-test issues."""
        case = (
            params.batch_size,
            params.num_heads,
            params.seq_length,
            params.head_dim,
            params.dtype,
            mode,
        )
        if TEST_WITH_ROCM and case in ROCM_FLAKY_LEARNABLE_BIAS_CASES[test_name]:
            self.skipTest(
                "Flaky on ROCm: https://github.com/pytorch/pytorch/issues/167271 and https://github.com/pytorch/pytorch/issues/167296"
            )

    @torch.no_grad()
    def _gold_check(self, eager, compiled, gold, tensor_name, fudge_factor=1.35):
        # [FUNCTION DOC] HELPER: no-grad accuracy check of one tensor: computes RMSE of eager and
        # of compiled against gold; the allowed factor is multiplied by 10 for float32. For output
        # in float32, skips the test if compiled error exceeds the scaled reference error.
        # Otherwise asserts compiled error is at most the reference error times the factor.
        # KNOWN ISSUE: the reference error is already multiplied by the factor and then multiplied
        # by it again in the assertion, so the factor is applied twice.
        ref_error = rmse(eager, gold)
        comp_error = rmse(compiled, gold)
        # Note: This has been carefully tested that FlexAttention is within
        # 20% of the average error of SDPA! Do not bump this tolerance
        # unless you are absolutely sure you are not worsening the accuracy
        # of FlexAttention!
        if eager.dtype == torch.float32:
            fudge_factor = 10.0 * fudge_factor

        comp_error = comp_error.item()
        ref_error = ref_error.item() * fudge_factor

        if (
            tensor_name == "out"
            and eager.dtype == torch.float32
            and comp_error > ref_error
        ):
            self.skipTest("Compiled FlexAttention is less accurate than eager in fp32")

        self.assertLessEqual(
            comp_error,
            (ref_error * fudge_factor),
            lambda msg: f"{msg}\nTensor: {tensor_name}\nCompiled error ({comp_error:.8f}) exceeds "
            f"reference error ({ref_error:.8f}) * fudge_factor ({fudge_factor})",
        )

    def _check_outputs_and_grads(
        self, out_eager, out_compiled, out_gold, tensors, names=None
    ):
        # [FUNCTION DOC] HELPER: draws a random output gradient (made on CPU), computes the
        # gradients of eager, compiled and gold outputs with respect to the tensors, and compares
        # the outputs and each gradient through _gold_check using given or default names.
        backwards_grad = torch.randn_like(out_eager, device="cpu").to(out_eager.device)
        grads_eager = torch.autograd.grad((out_eager,), tensors, backwards_grad)
        grads_compiled = torch.autograd.grad((out_compiled,), tensors, backwards_grad)
        grads_gold = torch.autograd.grad((out_gold,), tensors, backwards_grad)

        tensor_names = (
            ["out", "grad_query", "grad_key", "grad_value", "grad_bias"]
            if names is None
            else names
        )

        eager_tensors = (out_eager, *grads_eager)
        compiled_tensors = (out_compiled, *grads_compiled)
        gold_tensors = (out_gold, *grads_gold)

        for eager, compiled, gold, name in zip(
            eager_tensors, compiled_tensors, gold_tensors, tensor_names, strict=True
        ):
            self._gold_check(eager, compiled, gold, name)

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_relative_1d_bias(self, device, params, mode: str):
        # [FUNCTION DOC] Learnable bias indexed by the absolute q-kv distance, for default and
        # max-autotune modes: compares outputs and grads including the bias grad. Skips known
        # flaky ROCm cases.
        self.skip_rocm_flaky_learnable_bias_case("relative_1d_bias", params, mode)
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            2 * params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at the absolute distance of q and kv.
            return score + bias[torch.abs(q_idx - kv_idx)]

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_absolute_2d_bias(self, device, params):
        # [FUNCTION DOC] Learnable absolute 2-D bias indexed by q and kv; outputs and
        # grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at q and kv.
            return score + bias[q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_head_specific_bias(self, device, params):
        # [FUNCTION DOC] Learnable per-head bias matrix indexed by head, q and kv; outputs and
        # grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at head, q and kv.
            return score + bias[h, q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_batch_head_bias(self, device, params):
        # [FUNCTION DOC] Learnable bias with batch, head, q and kv dimensions; outputs and
        # grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at batch, head, q and kv.
            return score + bias[b, h, q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_multiplicative_bias(self, device, params):
        # [FUNCTION DOC] Learnable bias multiplying the scores per query position; outputs and
        # grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: multiplies the score by the bias entry at q.
            return score * bias[q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_local_window_bias(self, device, params):
        # [FUNCTION DOC] Learnable float32 local window bias (17 entries) indexed by clamped q-kv
        # offset; outputs and grads compared.
        query, key, value = self._init_tensors(params, device=device)
        window_size = 8
        bias = torch.randn(
            2 * window_size + 1,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at the clamped window offset.
            window_idx = torch.clamp(q_idx - kv_idx + window_size, 0, 2 * window_size)
            return score + bias[window_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_global_tokens_bias(self, device, params):
        # [FUNCTION DOC] Learnable float32 per-key bias (global token style) indexed by kv;
        # outputs and grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at kv.
            return score + bias[kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_weird_bias(self, device, params):
        # [FUNCTION DOC] Learnable bias with an extra dimension of four indexed by batch, head, a
        # captured 0-d index tensor and q; outputs and grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            4,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )
        which_bias = torch.tensor(0, device=device)

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry selected with the captured index tensor.
            return score + bias[b, h, which_bias, q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_indirect_bias(self, device, params):
        # [FUNCTION DOC] Learnable bias reached through an index tensor (random offsets per q);
        # outputs and grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        offset = torch.randint(
            0,
            params.seq_length,
            (params.seq_length,),
            device=device,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at the offset chosen for q.
            return score + bias[offset[q_idx]]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_symmetric_bias(self, device, params, mode: str):
        # [FUNCTION DOC] Learnable bias indexed twice (by q and by kv) in one score_mod: the
        # backward must raise InductorError about multiple indexing operations on one tensor
        # requiring gradients. Skips known flaky ROCm cases.
        self.skip_rocm_flaky_learnable_bias_case("symmetric_bias", params, mode)
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entries at q and at kv.
            return score + bias[q_idx] + bias[kv_idx]

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )
        # Error in backwards
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "Using multiple indexing operations on the same tensor that requires gradients",
        ):
            self._check_outputs_and_grads(
                out_eager,
                out_compiled,
                out_gold,
                (query, key, value, bias),
            )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_flipped_indexed_bias(self, device, params):
        # [FUNCTION DOC] Learnable 2-D bias indexed with kv first and q second; outputs and
        # grads compared.
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at kv and q.
            return score + bias[kv_idx, q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_head_specific_gate(self, device, params, mode: str):
        # [FUNCTION DOC] Learnable per-head gate passed through sigmoid that multiplies
        # scores; compares outputs and grads in default and max-autotune modes. Skips known
        # flaky ROCm cases.
        self.skip_rocm_flaky_learnable_bias_case("head_specific_gate", params, mode)
        query, key, value = self._init_tensors(params, device=device)
        gate_score = torch.randn(
            params.num_heads,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: multiplies the score by the sigmoid of the head's gate.
            return score * torch.sigmoid(gate_score[h])

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, gate_score),
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_distinct_biases(self, device, params):
        # [FUNCTION DOC] Two separate learnable biases (one in the parameter dtype and one
        # float32), one indexed by q and one by kv; compares outputs and all grads with named
        # gradient entries.
        query, key, value = self._init_tensors(params, device=device)
        # Create two separate bias tensors
        bias1 = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )
        bias2 = torch.randn(
            params.seq_length,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds bias1 at q and bias2 at kv.
            return score + bias1[q_idx] + bias2[kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        # Include both bias tensors in the tuple for gradient checking
        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias1, bias2),
            names=[
                "out",
                "grad_query",
                "grad_key",
                "grad_value",
                "grad_bias1",
                "grad_bias2",
            ],
        )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @torch.compile
    def test_learnable_bias_global_compiled(self, device, params):
        # [FUNCTION DOC] Whole test body is itself compiled (decorated): attention with a
        # learnable full bias followed by a linear layer and mse loss; the bias gradient must
        # exist and be non-zero (AssertionError otherwise).
        batch_size = 1
        num_heads = 1
        seq_len = 128
        head_dim = 16
        d_model = num_heads * head_dim

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        out_proj = nn.Linear(d_model, d_model, device=device)

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        bias = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=device,
            requires_grad=True,
        )

        def bias_mod(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at batch, head, q and kv.
            return score + bias[b, h, q_idx, kv_idx]

        out = flex_attention(
            query=query,
            key=key,
            value=value,
            score_mod=bias_mod,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        attn_output = out_proj(out)
        random_target = torch.randn(batch_size, seq_len, d_model, device=device)
        loss = torch.nn.functional.mse_loss(attn_output, random_target)
        loss.backward()

        if bias.grad is None:
            raise AssertionError("No gradient computed for bias")
        if not torch.any(bias.grad != 0):
            raise AssertionError("Gradient for bias is 0")

    @skip_on_cpu
    def test_backprop_error_case(self, device):
        # [FUNCTION DOC] Regression test: a compiled function builds a bias matrix by indexing a
        # learnable vector and uses it in a score_mod of attention; backward must give non-zero
        # grads for both inputs.
        @torch.compile()
        def test(x, y):
            # [FUNCTION DOC] HELPER: compiled function building the bias matrix from y, repeating
            # x over heads and running attention with a score_mod that adds the bias.
            # Materialize a bias matrix
            B, L, device = x.shape[0], x.shape[1], x.device
            b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
            q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
            kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
            bias_mat = y[b, q_idx] + y[b, kv_idx]  # (B, L, L)

            # Dummy score_mod retrieving bias values
            def score_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: adds the materialized bias value at batch, q and kv.
                return score + bias_mat[b, q_idx, kv_idx]

            x_ = x[:, :, None].repeat(1, 1, 16, 1)
            # torch._dynamo.graph_break()
            return flex_attention(x_, x_, x_, score_mod=score_mod)

        B, L, D = 2, 16, 64

        x = torch.randn(B, L, D, device=device, requires_grad=True)
        y = torch.randn(B, L, device=device, requires_grad=True)

        _ = test(x, y).mean().backward()

        if not (x.grad.norm() > 0):
            raise AssertionError(f"Expected x.grad.norm() > 0, got {x.grad.norm()}")
        if not (y.grad.norm() > 0):
            raise AssertionError(f"Expected y.grad.norm() > 0, got {y.grad.norm()}")

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_relative_1d_bias_only_grad(self, device, params):
        # [FUNCTION DOC] Only the bias requires gradient (q, k, v do not): compares outputs and
        # the bias gradient across eager, compiled and golden.
        query, key, value = self._init_tensors(params, device=device)
        query = query.detach().requires_grad_(False)
        key = key.detach().requires_grad_(False)
        value = value.detach().requires_grad_(False)

        # Only bias requires gradients
        bias = torch.randn(
            2 * params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,  # Only bias needs gradients
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the bias entry at the absolute q-kv distance.
            return score + bias[torch.abs(q_idx - kv_idx)]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)

        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        # For gradient checking, we only pass the bias tensor since it's the only one requiring gradients
        self._check_outputs_and_grads(
            out_eager, out_compiled, out_gold, (bias,), names=["out", "bias"]
        )

    def _test_flex_attention_with_dynamic_max_autotune(self, device):
        # [FUNCTION DOC] HELPER: shared body compiling GQA attention with an alibi score_mod and a
        # compiled block mask using dynamic shapes and max-autotune without cudagraphs, running
        # backward and checking the output shape equals the query shape.
        query = torch.randn(2, 16, 512, 64, device=device)
        key = torch.randn(2, 16, 512, 64, device=device)
        value = torch.randn(2, 16, 512, 64, device=device)
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        shape = (2, 16, 512, 16, 512, 64)
        B, Hq, M, Hkv, N, D = shape

        score_mod = _generate_alibi_bias(8)

        def causal(b, h, m, n):
            # [FUNCTION DOC] HELPER: causal mask_mod.
            return m >= n

        mask_shape = (1, 1, M, N)
        block_mask = torch.compile(create_block_mask)(
            causal, *mask_shape, device=device
        )

        compiled_sdpa = torch.compile(
            flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
        )

        out = compiled_sdpa(
            query=query,
            key=key,
            value=value,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=None,
        )
        out.sum().backward()

        self.assertEqual(
            out.shape,
            query.shape,
            lambda msg: f"{msg}\nExpected shape {query.shape}, got {out.shape}",
        )

    @skip_on_cpu
    @skip_on_mps  # uses mode="max-autotune-no-cudagraphs"; Triton/CUDA-only
    def test_flex_attention_with_dynamic_max_autotune(self, device):
        # [FUNCTION DOC] Non-CPU/MPS test running the dynamic max-autotune helper.
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    @torch._inductor.config.patch("graph_partition", True)
    def test_flex_attention_with_dynamic_max_autotune_graph_partition(self, device):
        # [FUNCTION DOC] Non-CPU test running the dynamic max-autotune helper with the graph
        # partition config enabled.
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    def test_flex_attention_logging(self, device):
        # [FUNCTION DOC] Non-CPU test: sets the environment variable for the flex attention
        # autotune log file to a temporary path, clears the cached log-name function, compiles
        # forward and backward with dynamic max-autotune, then reads the produced JSON log: it
        # must hold two entries (forward and backward) with the query, key and value shapes, a
        # non-empty list of choices with type and time (sorted so the first is fastest), and for
        # Triton choices the num_warps, num_stages, problem sizes and the forward-only or
        # backward-only block size keys. The cache clear is registered as a cleanup.
        from torch._inductor.select_algorithm import get_flex_attention_log_filename

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "flex_attention_configs")

            with patch.dict(
                os.environ, {"TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE": log_file}
            ):
                get_flex_attention_log_filename.cache_clear()
                self.addCleanup(get_flex_attention_log_filename.cache_clear)

                query = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )
                key = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )
                value = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )

                def score_mod(score, b, h, q_idx, kv_idx):
                    # [FUNCTION DOC] HELPER: doubles the score.
                    return score * 2

                def causal_mask(b, h, q_idx, kv_idx):
                    # [FUNCTION DOC] HELPER: causal mask_mod.
                    return q_idx >= kv_idx

                block_mask = torch.compile(create_block_mask)(
                    causal_mask, 1, 1, 128, 128, device=device
                )

                compiled_flex = torch.compile(
                    flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
                )

                out = compiled_flex(
                    query=query,
                    key=key,
                    value=value,
                    score_mod=score_mod,
                    block_mask=block_mask,
                )

                out.sum().backward()

                json_file = log_file + ".json"
                self.assertTrue(
                    os.path.exists(json_file),
                    lambda msg: f"{msg}\nLog file {json_file} was not created",
                )

                with open(json_file) as f:
                    log_data = json.load(f)

                self.assertIsInstance(log_data, list)
                self.assertEqual(len(log_data), 2)

                # Check that we have both forward and backward entries
                kernel_types_seen = [entry["kernel_type"] for entry in log_data]
                self.assertIn("forward", kernel_types_seen)
                self.assertIn("backward", kernel_types_seen)

                # Expected values from the test inputs
                expected_query_shape = "[1, 2, 128, 64]"
                expected_key_shape = "[1, 2, 128, 64]"
                expected_value_shape = "[1, 2, 128, 64]"
                expected_B = 1
                expected_Hq = 2
                expected_Hkv = 2
                expected_seq_len_q = 128
                expected_seq_len_kv = 128
                expected_qk_head_dim = 64
                expected_v_head_dim = 64

                for entry in log_data:
                    self.assertIsInstance(entry, dict)
                    # New format has: query_shape, key_shape, value_shape, kernel_type, choices
                    self.assertIn("kernel_type", entry)
                    self.assertIn("choices", entry)
                    self.assertIn("query_shape", entry)
                    self.assertIn("key_shape", entry)
                    self.assertIn("value_shape", entry)

                    # Check shape values
                    self.assertEqual(entry["query_shape"], expected_query_shape)
                    self.assertEqual(entry["key_shape"], expected_key_shape)
                    self.assertEqual(entry["value_shape"], expected_value_shape)

                    kernel_type = entry["kernel_type"]
                    choices = entry["choices"]

                    self.assertIsInstance(choices, list)
                    self.assertGreater(len(choices), 0)

                    for i, choice in enumerate(choices):
                        self.assertIn("type", choice)
                        self.assertIn("time", choice)

                        if choice["type"] == "triton":
                            self.assertIn("num_warps", choice)
                            self.assertIn("num_stages", choice)

                            # Check numerical values in each choice
                            self.assertIn("B", choice)
                            self.assertIn("Hq", choice)
                            self.assertIn("Hkv", choice)
                            self.assertIn("seq_len_q", choice)
                            self.assertIn("seq_len_kv", choice)
                            self.assertIn("qk_head_dim", choice)
                            self.assertIn("v_head_dim", choice)

                            self.assertEqual(choice["B"], expected_B)
                            self.assertEqual(choice["Hq"], expected_Hq)
                            self.assertEqual(choice["Hkv"], expected_Hkv)
                            self.assertEqual(choice["seq_len_q"], expected_seq_len_q)
                            self.assertEqual(choice["seq_len_kv"], expected_seq_len_kv)
                            self.assertEqual(
                                choice["qk_head_dim"], expected_qk_head_dim
                            )
                            self.assertEqual(choice["v_head_dim"], expected_v_head_dim)

                            if kernel_type == "forward":
                                self.assertIn("BLOCK_M", choice)
                                self.assertIn("BLOCK_N", choice)
                                self.assertNotIn("BLOCK_M1", choice)
                            elif kernel_type == "backward":
                                self.assertIn("BLOCK_M1", choice)
                                self.assertIn("BLOCK_N1", choice)
                                self.assertIn("BLOCK_M2", choice)
                                self.assertIn("BLOCK_N2", choice)
                                self.assertNotIn("BLOCK_M", choice)
                                self.assertNotIn("BLOCK_N", choice)

                        if i > 0:
                            self.assertLessEqual(choices[0]["time"], choice["time"])

    @skip_on_cpu
    def test_inspect_bug(self, device):
        # [FUNCTION DOC] Regression test for issue 139374: a mask_mod made via functools.partial
        # binding a tensor argument creates a block mask eagerly and under fullgraph compile.
        # https://github.com/pytorch/pytorch/issues/139374
        def sliding_window(b, h, q_idx, kv_idx, val):
            # [FUNCTION DOC] HELPER: mask_mod true where the absolute q-kv distance is below the
            # given value.
            return (q_idx - kv_idx).abs() < val

        sliding_window2 = functools.partial(
            sliding_window, val=torch.randn((), device=device)
        )
        opt_fn = torch.compile(create_block_mask, fullgraph=True)
        create_block_mask(sliding_window2, None, None, 1024, 1024, device=device)
        # checks that the compile is working
        opt_fn(sliding_window2, None, None, 1024, 1024, device=device)

    @supported_platform
    @skip_on_cpu
    def test_head_bias_req_grad(self, device):
        # [FUNCTION DOC] Learnable per-head bias (float16, 4 heads) added in the score_mod;
        # compares flex attention outputs and grads, including the bias grad, with SDPA given the
        # equivalent broadcast attention bias via the shared helper.
        B, H, S, D = 1, 4, 256, 64
        bias = torch.randn(H, device=device, dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def head_bias(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the head's bias entry to the score.
            return score + bias_flex[h]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        implicit_bias_sdpa_ref = implicit_bias_sdpa_ref.view(H, 1, 1).expand(H, S, S)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold
        implicit_bias_sdpa_gold = implicit_bias_sdpa_gold.view(H, 1, 1).expand(H, S, S)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            head_bias,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

    @supported_platform
    @skip_on_cpu
    def test_comparison_vs_sdpa_with_learnable_bias(self, device):
        # [FUNCTION DOC] Runs the shared SDPA comparison helper for several learnable bias forms:
        # 1-D indexed by q plus kv, 2-D by q and kv, 2-D indexed in two steps, 2-D transposed, and
        # a 3-D per-head transposed bias.
        # 1-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(
            2 * S, device=device, dtype=torch.float16, requires_grad=True
        )

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_1d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the 1-D bias entry at the sum of q and kv.
            return score + bias_flex[q_idx + kv_idx]

        bias_indices = torch.arange(S)[:, None] + torch.arange(S)
        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref[bias_indices]
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold[bias_indices]

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_1d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

        # 2-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the 2-D bias entry at q and kv.
            return score + bias_flex[q_idx, kv_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

        # 2-dimensional bias + index multiple
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the 2-D bias via two successive indexings (q then kv).
            return score + bias_flex[q_idx][kv_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

        # 2-dimensional bias + transposed:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d_transposed(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the 2-D bias entry at kv and q (transposed).
            return score + bias_flex[kv_idx, q_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref.transpose(-1, -2)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold.transpose(-1, -2)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d_transposed,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

        # 3-dimensional bias + transposed
        B, H, S, D = 4, 8, 256, 64
        bias = torch.randn(
            H, S, S, device=device, dtype=torch.float16, requires_grad=True
        )

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_3d_transposed(score, b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: adds the 3-D bias entry at head, kv and q (transposed).
            return score + bias_flex[h, kv_idx, q_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref.transpose(-1, -2)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold.transpose(-1, -2)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_3d_transposed,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
            device,
        )

    def _test_learnable_bias_inner(
        self,
        B,
        H,
        S,
        D,
        score_mod,
        bias_flex,
        implicit_bias_sdpa_ref,
        bias_sdpa_ref,
        implicit_bias_sdpa_gold,
        bias_sdpa_gold,
        device,
    ):
        # [FUNCTION DOC] HELPER: runs SDPA in float16 (reference) and float64 (gold) with the
        # implicit attention bias, and flex attention with the score_mod, on all-ones inputs;
        # backward on all. For output, q, k, v grads and the bias grad, asserts that the flex RMSE
        # vs gold is at most 1.2 times the SDPA RMSE vs gold.
        make_tensor = functools.partial(
            torch.ones,
            (B, H, S, D),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        q_ref, k_ref, v_ref = make_tensor(), make_tensor(), make_tensor()
        q_gold, k_gold, v_gold = query_key_value_clones(
            q_ref, k_ref, v_ref, torch.float64
        )
        q_flex, k_flex, v_flex = query_key_value_clones(q_ref, k_ref, v_ref)

        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, attn_mask=implicit_bias_sdpa_ref
        )
        out_ref.sum().backward()
        out_gold = torch.nn.functional.scaled_dot_product_attention(
            q_gold, k_gold, v_gold, attn_mask=implicit_bias_sdpa_gold
        )
        out_gold.sum().backward()
        out_flex = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod)
        out_flex.sum().backward()

        name = score_mod.__name__
        for ref, flex, gold in [
            (out_ref, out_flex, out_gold),
            (q_ref.grad, q_flex.grad, q_gold.grad),
            (k_ref.grad, k_flex.grad, k_gold.grad),
            (v_ref.grad, v_flex.grad, v_gold.grad),
            (bias_sdpa_ref.grad, bias_flex.grad, bias_sdpa_gold.grad),
        ]:
            ref_error = rmse(ref, gold)
            flex_error = rmse(flex, gold)
            self.assertTrue(
                ref_error * 1.2 >= flex_error,
                lambda msg: f"{msg}\n{name} -> Ref error: {ref_error}, Flex eager Error: {flex_error}",
            )


instantiate_device_type_tests(
    TestFlexAttention,
    globals(),
    only_for=test_device,
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestPagedAttention,
    globals(),
    only_for=test_device,
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestBlockMask,
    globals(),
    only_for=(test_device[0] if (HAS_GPU or HAS_MPS) else "cuda",),
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestLearnableBiases, globals(), only_for=test_device, allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
