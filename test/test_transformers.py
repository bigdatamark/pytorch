# Owner(s): ["module: sdpa"]
# [MODULE DOC] Test suite for scaled dot product attention (SDPA) and the transformer modules
# (MultiheadAttention, TransformerEncoder/Decoder layers, nested-tensor fast paths). Defines
# shared helpers (tolerance checks, random/strided SDPA input builders, platform capability
# probes), and the test classes TestTransformers, TestSDPAFailureModes, TestSDPA,
# TestSDPACpuOnly, and device-specific variants. Import-time side effects: imports fairseq only
# when available, builds the namedtuples SdpaShape and Tolerances, computes CUDA
# compute-capability flags (isSM8XDevice etc.), and builds the list of platform-supported SDPA
# backends. Device-generic tests are instantiated at the bottom of the file.

import contextlib
from functools import partial
from collections import namedtuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
import itertools
import torch.optim as optim
from torch.backends.cuda import SDPAParams
from torch.testing._internal.common_device_type import expectedFailureMPS, instantiate_device_type_tests, onlyCUDA, largeTensorTest
import torch.utils.cpp_extension
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    skipIfTorchDynamo,
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck,
    make_tensor,
    NOTEST_CPU,
    TEST_WITH_TORCHDYNAMO,
    TEST_XPU,
    xfailIfNoAcceleratorTriton,
    xfailIf,
    setSdpaBackendsToDefaultFinally,
)
from torch._dynamo.testing import CompileCounterWithBackend


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import (
    IS_JETSON,
    SM80OrLater,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_CK_SDPA,
    tf32_off,
    tf32_on_and_off,
    tf32_enabled,
)
from torch.testing._internal.common_device_type import skipXPUIf
from torch.testing._internal.common_xpu import PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])


@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    # [FUNCTION DOC] HELPER: Context manager that sets the global deterministic-algorithms mode
    # and warn-only flag for the duration of the with-block. Saves the previous mode and
    # warn-only state, applies the requested ones, yields an empty dict, and always restores the
    # saved state on exit (even on exceptions).
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

isSM8XDevice = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 7), (8, 9)]
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
isSM120Device = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(12, 0), (12, 1)]
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5
isLessThanSM80Device = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8


def _cudnn_supports_d256_attention():
    # [FUNCTION DOC] HELPER: Returns True only if CUDA and cuDNN are available, the GPU is SM
    # 9.0 or any SM 10.x (except SM 10.7 with cuDNN 9.25.0 or older), and the cuDNN version is
    # newer than 9.22.0; otherwise False. Used to gate tests on cuDNN attention support for head
    # dim 256.
    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        return False
    cudnn_version = torch.backends.cudnn.version() or 0
    device_capability = torch.cuda.get_device_capability()
    is_unsupported_sm107 = device_capability == (10, 7) and cudnn_version <= 92500
    supports_d256 = (
        device_capability == (9, 0) or device_capability[0] == 10
    ) and not is_unsupported_sm107
    return supports_d256 and cudnn_version > 92200


def _cudnn_d256_mixed_head_dim_bprop_unsupported():
    # [FUNCTION DOC] HELPER: Returns True if CUDA and cuDNN are available and the platform
    # supports head dim 256 cuDNN attention (via the helper above); used to skip/xfail mixed
    # head-dim backward tests where cuDNN does not support them. Returns False when CUDA or
    # cuDNN is missing.
    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        return False
    return _cudnn_supports_d256_attention()


def _cudnn_supports_d192_attention():
    # [FUNCTION DOC] HELPER: Returns True only if CUDA and cuDNN are available, the GPU is SM
    # 9.0 or SM 10.x, and the cuDNN version is at least 9.11.0; gates tests that use head dim
    # 192 on the cuDNN backend.
    if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
        return False
    cudnn_version = torch.backends.cudnn.version() or 0
    device_capability = torch.cuda.get_device_capability()
    is_sm90_or_sm10x = device_capability == (9, 0) or device_capability[0] == 10
    return is_sm90_or_sm10x and cudnn_version >= 91100


def _check_equal(
    golden: torch.Tensor,
    reference: torch.Tensor,
    test: torch.Tensor,
    fudge_factor: float,
    tensor_name: str | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Compares a test tensor to a golden (highest precision) tensor,
    # using the error of a same-precision reference tensor as the yardstick. Nested tensors are
    # unbound and compared element-wise recursively. Computes max abs error of test and of
    # reference against golden, raises ValueError if the test error is NaN while the reference
    # error is not, or if the test error exceeds the larger of the float32 default atol and
    # reference error times fudge_factor. tensor_name is only used in the error message. Returns
    # None.
    """
    Compare test tensor against golden and reference tensors.
    Golden is the highest precision possible serving as the "ground truth"
    Reference is the same precision as test and should also serve as less precisie ground truth.
    We calcculate the "reference error" by comparing the golden to reference and use this as the
    measruing stick for the test tensor.

    Raises ValueError if compiled error exceeds threshold.

    Args:
        golden (torch.Tensor): The golden tensor to compare against.
        reference (torch.Tensor): The reference tensor for error calculation.
        test (torch.Tensor): The test tensor to be evaluated.
        fudge_factor (float): A multiplier for the reference error to determine the threshold.
        tensor_name (Optional[str], optional): Name of the tensor for error reporting. Defaults to None.

    Raises:
        ValueError: If the test tensor contains NaN values while the reference does not,
                    or if the test error exceeds the calculated threshold.

    Notes:
        - For nested tensors, the function recursively calls itself on each nested element.
        - The error threshold is calculated as the maximum of a default tolerance for float32
          and the product of the reference error and the fudge_factor.
        - If the test error exceeds the threshold, a ValueError is raised with a detailed message.
    """
    if golden.is_nested and reference.is_nested and test.is_nested:
        for gold, ref, tst in zip(golden.unbind(), reference.unbind(), test.unbind()):
            _check_equal(gold, ref, tst, fudge_factor, tensor_name)
        return

    # Compute error between golden
    test_error = (golden - test).abs().max()
    ref_error = (golden - reference).abs().max()

    if torch.isnan(test_error).any() and not torch.isnan(ref_error).any():
        raise ValueError("Output/Grad with NaN")

    # Calculate the error threshold as the maximum of:
    # 1. A predefined default tolerance for float32
    # 2. The reference error multiplied by the fudge factor
    threshold = max(default_atol[torch.float32], ref_error * fudge_factor)
    if test_error > threshold:
        name = tensor_name or ""
        msg = f"{name} Test error {test_error} is greater than threshold {threshold}!"
        raise ValueError(msg)


def check_out_and_grad(
    out_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_query_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_key_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_value_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_attn_mask_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    fudge_factors: dict[str, float] | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Validates attention output and the gradients of query, key, value
    # and optionally the attention mask. Each argument is a tuple of (golden reference,
    # low-precision reference, compiled/ test result). Under no_grad it calls _check_equal on
    # each, using per-tensor fudge factors from the optional dict with keys out, grad_query,
    # grad_key, grad_value, grad_attn_mask defaulting to 5.0. Raises ValueError (from
    # _check_equal) on mismatch; returns None.
    """
    Check output and gradients of attention mechanism tensors.
    Compares compiled results against reference and low-precision reference tensors.

    Args:
        out_tuple: Tuple of (ref, lp_ref, compiled) for output tensor
        grad_query_tuple: Tuple of (ref, lp_ref, compiled) for query gradient
        grad_key_tuple: Tuple of (ref, lp_ref, compiled) for key gradient
        grad_value_tuple: Tuple of (ref, lp_ref, compiled) for value gradient
        grad_attn_mask_tuple: Optional tuple of (ref, lp_ref, compiled) for attention mask gradient
        fudge_factors: Dictionary of fudge factors for each tensor type (default uses 5.0 for all)
    """
    default_fudge_factor = 5.0
    if fudge_factors is None:
        fudge_factors = {}

    out_ref, out_lp_ref, out = out_tuple

    with torch.no_grad():
        _check_equal(out_ref, out_lp_ref, out, fudge_factors.get('out', default_fudge_factor), "out")

        grad_checks = [
            (grad_query_tuple, "grad_query"),
            (grad_key_tuple, "grad_key"),
            (grad_value_tuple, "grad_value")
        ]

        for grad_tuple, name in grad_checks:
            ref_grad, lp_ref_grad, comp_grad = grad_tuple
            _check_equal(ref_grad, lp_ref_grad, comp_grad, fudge_factors.get(name, default_fudge_factor), name)

        if grad_attn_mask_tuple:
            attn_mask_ref_grad, attn_mask_ref_lp_grad, attn_mask_grad = grad_attn_mask_tuple
            _check_equal(
                attn_mask_ref_grad,
                attn_mask_ref_lp_grad,
                attn_mask_grad,
                fudge_factors.get("grad_attn_mask", default_fudge_factor),
                "grad_attn_mask",
            )


def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None):
    # [FUNCTION DOC] HELPER: Returns detached clones of query, key and value cast to dtype
    # (defaults to the query dtype), each with requires_grad matching its source, so that
    # reference computations get independent leaves. Returns a tuple of query, key, value
    # clones.
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

def get_platform_specific_sdpa():
    # [FUNCTION DOC] HELPER: Builds the list of SDPA backends supported by this platform: flash
    # attention, memory efficient attention and cuDNN attention, each added when its
    # PLATFORM_SUPPORTS flag is true. If none are supported it returns a single memory-efficient
    # placeholder so that parametrize never receives an empty list.
    ret = []
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    if not ret:
        # Add a placeholder, an empty list causes "An empty arg_values was passed to @parametrize"
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    return ret

PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()
# Indicate the Efficient attention backend can support:
# 1. sequence longher than 512
# 2. head dimsion larger than 64
MEM_EFF_CAPABILITY_MATCHES_SM80 = SM80OrLater or TEST_WITH_ROCM

def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Creates a random tensor for SDPA tests from an SdpaShape (batch,
    # num_heads, seq_len, head_dim) on the given device and dtype. If type is "nested", returns
    # a nested tensor of batch random components; seq_len may be a list of per-sequence lengths
    # or a single int; each component is of sequence, heads, head_dim, or sequence by three
    # times heads times head_dim when packed. Otherwise returns a dense random tensor of batch,
    # seq_len, heads, head_dim or the packed variant, and raises AssertionError if seq_len is
    # not an int. requires_grad is applied to all created tensors.
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
    if type == "nested":
        if isinstance(seq_len, list):
            def _size(i):
                # [FUNCTION DOC] HELPER: Nested function returning the shape tuple of the i-th
                # nested component: per-sequence length i with num_heads and head_dim, or the
                # packed shape of length by three times num_heads times head_dim when packed.
                return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

            return torch.nested.nested_tensor([
                torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                for i in range(batch)])
        else:
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
    else:
        if not isinstance(seq_len, int):
            raise AssertionError(f"seq_len should be int, got {type(seq_len)}")
        size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


def make_strided_sdpa_input(
    shape: tuple[int, int, int, int],
    layout: str,
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds a leaf SDPA input of the given batch, heads, seq, head_dim
    # shape whose last dimension has stride one but whose outer layout is non-contiguous. layout
    # selects: padded (view of wider last dim), offset (view starting at offset 8),
    # sequence_sliced, head_sliced (strided slices), transposed (batch/heads swapped view), or
    # batch_expanded (broadcast over batch). Raises AssertionError for an unknown layout.
    # Returns the detached tensor with requires_grad set as asked.
    """Create a leaf SDPA input with last-dimension stride one."""
    batch, num_heads, seq_len, head_dim = shape
    match layout:
        case "padded":
            tensor = torch.randn(
                batch, num_heads, seq_len, head_dim + 8, device=device, dtype=dtype
            )[..., :head_dim]
        case "offset":
            tensor = torch.randn(
                batch, num_heads, seq_len, head_dim + 16, device=device, dtype=dtype
            )[..., 8 : head_dim + 8]
        case "sequence_sliced":
            tensor = torch.randn(
                batch, num_heads, seq_len * 2, head_dim, device=device, dtype=dtype
            )[:, :, ::2]
        case "head_sliced":
            tensor = torch.randn(
                batch, num_heads * 2, seq_len, head_dim, device=device, dtype=dtype
            )[:, ::2]
        case "transposed":
            tensor = torch.randn(
                num_heads, batch, seq_len, head_dim, device=device, dtype=dtype
            ).transpose(0, 1)
        case "batch_expanded":
            tensor = torch.randn(
                1, num_heads, seq_len, head_dim, device=device, dtype=dtype
            ).expand(batch, -1, -1, -1)
        case _:
            raise AssertionError(f"Unknown layout: {layout}")
    return tensor.detach().requires_grad_(requires_grad)


class TestTransformers(NNTestCase):
    # [CLASS DOC] Device-generic test class (derived from NNTestCase) for the nn transformer
    # modules: MultiheadAttention fast path, TransformerEncoder/Decoder layers, nested-tensor
    # handling, masks, is_causal behavior, and scaled_dot_product_attention against a reference
    # implementation. Enables CUDA memory leak checking and non-default-stream checking via
    # class attributes. Each method takes a device argument injected by
    # instantiate_device_type_tests.
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    def test_self_attn_TxT_attn_mask(self, device):
        # [FUNCTION DOC] Skipped (4D masks unsupported) CUDA test: runs an eval-mode
        # MultiheadAttention in inference mode on random queries with a random square float mask
        # given once as a 2D mask and once as a 4D expanded mask, and checks the two outputs are
        # equal.
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        query = torch.rand(batch_size, tgt_len, embed_dim, device=device)  # [N, T, D]
        attn_mask = torch.randint(0, 2, (tgt_len, tgt_len)).cuda().float()  # [T, T]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

        attn_mask_4d = attn_mask.expand(batch_size, num_heads, tgt_len, tgt_len)

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
        mta_model.eval()

        # Generate 3D results
        with torch.inference_mode():
            output_mask_4d = mta_model(query, query, query, attn_mask=attn_mask_4d)[0]
            output_mask_4d = output_mask_4d.transpose(0, 1)  # [N, T, D]

            output_mask_TxT = mta_model(query, query, query, attn_mask=attn_mask)[0]
            output_mask_TxT = output_mask_TxT.transpose(0, 1)  # [N, T, D]

            self.assertEqual(output_mask_4d, output_mask_TxT)

    @slowTest
    def test_train_with_pad_and_catch_error(self, device):
        # [FUNCTION DOC] Slow test: trains a 2-layer TransformerEncoder for 100 iterations with
        # a boolean key padding mask and an SGD optimizer, and each iteration checks the encoder
        # rejects uint8 and int64 padding masks by raising AssertionError (the loop continues on
        # the expected error), then checks that train-mode and eval-mode outputs on the unpadded
        # positions differ by less than 1e-4 in L1 loss (reached only if the unsupported-type
        # calls do not raise).
        iters = 100
        pad_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool).to(device)
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=2,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()
        for _ in range(iters):
            encoder.train()
            optimizer.zero_grad()
            inputs = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

            outputs = encoder(inputs, src_key_padding_mask=pad_mask)

            loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

                # Expect uint8 type not supported
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.uint8))
                except AssertionError:
                    continue
                self.assertFalse(e, "Failed to catch unsupported uint8 type exception")

                test_train_bool = encoder(test, src_key_padding_mask=pad_mask)
                encoder.eval()

                # Expect long type not supported
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.int64))
                except AssertionError as e:
                    continue
                self.assertFalse(e, "Failed to catch unsupported Long type exception")

                test_eval_bool = encoder(test, src_key_padding_mask=pad_mask)
                l1_bool = nn.L1Loss()(test_train_bool[:, 0:2, :], test_eval_bool[:, 0:2, :]).item()
                self.assertTrue(l1_bool < 1e-4, "Eval/Train difference in pad_mask BOOL")

    @tf32_on_and_off(0.001)
    @parametrize("attn_mask_dim", [2, 3, None])
    @parametrize("key_padding_mask_dim", [2, None])
    @parametrize("mask_dtype", [torch.bool, torch.float32])
    def test_multiheadattention_fastpath_attn_mask(self, device, attn_mask_dim, key_padding_mask_dim, mask_dtype):
        # [FUNCTION DOC] Checks that MultiheadAttention gives the same result in train mode (no
        # fast path) and eval mode (fast path) under no_grad, for attention masks of 2D, 3D or
        # no mask, optional 2D key padding mask, and bool or float mask dtype. The fast-path
        # output is passed through nan_to_num because fully masked rows are NaN on the fast path
        # but zero on the SDPA path.
        # MHA converts all
        with torch.no_grad():
            B = 2
            L = 4
            D = 8
            H = 4

            if attn_mask_dim == 2:
                attn_mask = make_tensor((L, L), dtype=mask_dtype, device=device)
            elif attn_mask_dim == 3:
                attn_mask = make_tensor((B, 1, L, L), dtype=mask_dtype, device=device).expand(B, H, L, L).reshape(B * H, L, L)
            elif attn_mask_dim is None:
                attn_mask = None

            if key_padding_mask_dim == 2:
                key_padding_mask = make_tensor((B, L), dtype=mask_dtype, device=device)
            elif key_padding_mask_dim is None:
                key_padding_mask = None

            mha = nn.MultiheadAttention(D, H, batch_first=True, device=device)
            X = torch.randn(B, L, D, device=device)

            mha.train()  # disable fast path
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            mha.eval()  # enable fast path
            out_fp, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            # The FP kernel will return NaNs while the sdpa kernel which is ran when the fast path is turned off returns 0 instead
            # of NaNs for fully masked rows
            self.assertEqual(out, out_fp.nan_to_num())

    @onlyCUDA
    def test_multiheadattention_fastpath_fp16_head_dim_alignment(self, device):
        # [FUNCTION DOC] CUDA-only test: runs a float16 MultiheadAttention with embed dim 96, 16
        # heads, head dim 6 with large inputs under no_grad, once with the fast path enabled and
        # once disabled via the backend flag, and checks the fast-path output is finite and
        # close to the slow path within loose tolerances. Always restores the previous fast-path
        # setting in a finally block.
        previous_fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.manual_seed(0)
            mha = nn.MultiheadAttention(
                96,
                16,
                dropout=0.2,
                batch_first=True,
                dtype=torch.float16,
                device=device,
            )
            x = torch.randn(8, 32, 96, device=device, dtype=torch.float16) * 238.15560380049612
            with torch.no_grad():
                torch.backends.mha.set_fastpath_enabled(True)
                mha.eval()
                out, _ = mha(x, x, x, need_weights=False)
                torch.backends.mha.set_fastpath_enabled(False)
                out_ref, _ = mha(x, x, x, need_weights=False)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(out, out_ref, atol=2e-2, rtol=8e-2)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous_fastpath)

    @parametrize("nhead", [1, 4, 8])
    def test_transformerencoderlayer_src_mask(self, device, nhead):
        # [FUNCTION DOC] Smoke test: builds a batch-first TransformerEncoderLayer for the given
        # head count and runs it with an all-False boolean src_mask in train mode, then in eval
        # mode under no_grad. Only checks that no exception is raised.
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        model(src, src_mask=src_mask)
        model.eval()
        with torch.no_grad():
            model(src, src_mask=src_mask)

    @parametrize("nhead", [3, 4])
    def test_transformerencoderlayer_no_fastpath_with_hooks(self, device, nhead):
        # [FUNCTION DOC] Checks that a forward hook registered on the self-attention submodule
        # of an eval-mode TransformerEncoderLayer is still called exactly once during inference,
        # i.e. hooks disable the fast path. Raises AssertionError if the hook was not called
        # exactly once, and removes the hook.
        batch_size = 2
        seqlen = 4
        d_model = 12

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            batch_first=True).to(device).eval()
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model

        cache = []

        # forward hook to save output
        def hook(module, inputs, output):
            # [FUNCTION DOC] HELPER: Forward hook that appends a detached copy of the first
            # element of the hooked module's output to the enclosing list named cache.
            cache.append(output[0].detach())

        # register hook to get the output of the self-attention layer
        handle = model.self_attn.register_forward_hook(hook)

        # forward pass
        with torch.inference_mode():
            model(src)

        # output of the self-attention layer
        if len(cache) != 1:
            raise AssertionError(f"Expected 1 output, got {len(cache)}")

        # remove hook
        handle.remove()

    # Test asserts fastpath == slowpath, both running under the current
    # precision — so the @tf32_on_and_off tolerance applies to the mutual
    # agreement of two TF32-rounded outputs, not to TF32-vs-FP32. Measured
    # worst case on MI300 at d_model=12/seqlen<=1040 is ~1e-3, well under
    # 0.002. For d_model=256 the inner GEMMs have K=256 (vs K=12), so
    # summation-order drift between the two paths under hipBLASLt FAST_TF32
    # scales ~sqrt(256/12) ≈ 4.6x and compounds through 2 encoder layers;
    # use a looser tolerance on ROCm to cover that worst case while keeping
    # CUDA strict. See https://github.com/jeffdaily/tf32_analysis.
    @tf32_on_and_off(0.02 if TEST_WITH_ROCM else 0.002)
    @parametrize("use_torchscript", [False])
    @parametrize("enable_nested_tensor", [True, False])
    @parametrize("use_autocast", [True, False])
    @parametrize("d_model", [12, 256])
    def test_transformerencoder_fastpath(self, device, use_torchscript, enable_nested_tensor, use_autocast, d_model):
        # [FUNCTION DOC] Checks that a two-layer eval-mode TransformerEncoder gives the same
        # output on the fast path (under no_grad) and the slow path for several pairs of input
        # and boolean padding mask, including sequence lengths around 1024 where the softmax
        # kernel changes behavior. Optionally uses autocast to float16 and nested tensors. Fast
        # path outputs are padded back to the mask length and masked positions are zeroed before
        # comparing with assertEqual. Torchscript is only parametrized as False.
        """
        Test TransformerEncoder fastpath output matches slowpath output
        """
        torch.manual_seed(1234)
        nhead = 4
        dim_feedforward = d_model
        batch_first = True

        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=batch_first),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device).eval()

        if use_torchscript:
            model = torch.jit.script(model)

        # each input is (input, mask)
        input_mask_pairs = [
            (
                torch.rand(3, 2, d_model),
                [
                    [0, 1],
                    [0, 1],
                    [1, 1]
                ]
            ),
            (
                torch.rand(2, 100, d_model),
                [
                    [0] * 98 + [1] * 2,
                    [0] * 90 + [1] * 10
                ]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test 1024.
            (
                torch.rand(2, 1024, d_model),
                [
                    [0] * 1020 + [1] * 4,
                    [0] * 1024,
                ]
            ),
            (
                torch.rand(1, 1026, d_model),
                [[0] * 1024 + [1] * 2]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test range of masks above 1024.
            (
                torch.rand(4, 1040, d_model),
                [
                    [0] * 1024 + [1] * 16,
                    [0] * 1025 + [1] * 15,
                    [0] * 1031 + [1] * 9,
                    [0] * 1040,
                ]
            )
        ]
        input_mask_pairs = [
            (
                torch.tensor(pair[0], device=device, dtype=torch.get_default_dtype()),  # float input
                torch.tensor(pair[1], device=device, dtype=torch.bool)  # bool mask
            ) for pair in input_mask_pairs
        ]

        maybe_autocast = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        with maybe_autocast:
            for input, src_key_padding_mask in input_mask_pairs:
                with torch.no_grad():
                    fastpath_output = model(input, src_key_padding_mask=src_key_padding_mask)
                slowpath_output = model(input, src_key_padding_mask=src_key_padding_mask)  # reference
                # Make sure fastpath_output is same shape as slowpath_output and mask.
                # When enable_nested_tensor=true, fastpath_output may be smaller than input tensor.
                # Eg if input bs=1, seqlen=6, and we mask out 2 tokens, fastpath_output will have bs=1, seqlen=4.
                # Expand back to old size to match.
                bs, true_seqlen, embed_dim = fastpath_output.shape
                expanded_seqlen = src_key_padding_mask.shape[1]
                fastpath_output_expanded = torch.zeros(bs, expanded_seqlen, embed_dim, device=device)
                fastpath_output_expanded[:, :true_seqlen, :] = fastpath_output
                # no garauntees on output corresponding to masked tokens, so they may vary between slow/fast path. set all to 0.
                fastpath_output_expanded = fastpath_output_expanded.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                slowpath_output = slowpath_output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                self.assertEqual(fastpath_output_expanded, slowpath_output)

    @tf32_on_and_off(0.001)
    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [False])
    def test_transformerencoder_square_input(self, with_no_grad, training, enable_nested_tensor, device):
        # [FUNCTION DOC] Edge-case test for a two-layer TransformerEncoder given an input of
        # batch 2, length 2, embedding 4 where batch size equals sequence length. All parameters
        # are set to cosine-valued constants; with or without no_grad, in train or eval mode,
        # the output for a given boolean mask must match hard-coded reference values in shape
        # and value.
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for p in model.parameters():
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        if training:
            model = model.train()
        else:
            model = model.eval()
        x = torch.arange(0, 16).reshape(2, 2, 4).to(torch.get_default_dtype()).to(device)
        src_mask = torch.Tensor([[0, 1], [0, 0]]).to(torch.bool).to(device)

        if with_no_grad:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        with cm:
            result = model(x, mask=src_mask)

        ref_output = torch.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  ).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        self.assertEqual(result, ref_output)

    @parametrize("batch_first", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [True, False])
    def test_transformerencoder(self, batch_first, training, enable_nested_tensor, device):
        # [FUNCTION DOC] Deterministic test of TransformerEncoder against hard-coded reference
        # outputs, for batch_first or not, train or eval mode, and with or without nested
        # tensors. Runs in double default dtype, and under no_grad when evaluating (needed for
        # the fast path). Delegates to the nested function _test.
        def get_a_test_layer(activation, batch_first=False):
            # [FUNCTION DOC] HELPER: Builds a small TransformerEncoderLayer with d_model 4, 2
            # heads, feedforward 16 and no dropout with the given activation and batch_first on
            # the test device, with every parameter overwritten by deterministic cosine values,
            # and returns it.
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0

            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
            ).to(device)

            with torch.no_grad():
                # set constant weights of the model
                for p in layer.parameters():
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = torch.cos(torch.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerEncoder
        activation = F.relu

        def _test(batch_first, training, enable_nested_tensor):
            # [FUNCTION DOC] HELPER: Runs the encoder checks: builds one- two- and six-layer
            # encoders (with and without a final LayerNorm) over a fixed 5x2x4 input, with no
            # mask, an all-False source mask, and key padding masks, comparing each result's
            # shape and values with hard-coded references with rtol 1e-7 and atol 1e-5. Switches
            # to eval mode when training is False.
            def perm_fn(x):
                # [FUNCTION DOC] HELPER: Nested function that swaps the first two dimensions of
                # its argument when batch_first is set, otherwise returns it unchanged; used to
                # adapt reference data to the layout.
                return x.transpose(1, 0) if batch_first else x

            encoder_layer = get_a_test_layer(activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerEncoder(
                encoder_layer, 1, enable_nested_tensor=enable_nested_tensor
            ).to(device)

            if not training:
                model = model.eval()

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                 )).to(device)
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0 src_mask
            src_mask = torch.zeros([5, 5]).to(device) == 1
            result = model(encoder_input, mask=src_mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0
            mask = torch.zeros([2, 5]).to(device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249, -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643, -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 2, multiple layers no norm
            model = nn.TransformerEncoder(encoder_layer, 2, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419051, 0.017446, -0.608738, -0.085003],
                                                [2.419102, 0.017452, -0.608703, -0.085026]],
                                               [[2.419043, 0.017445, -0.608744, -0.084999],
                                                [2.419052, 0.017446, -0.608738, -0.085004]],
                                               [[2.419067, 0.017448, -0.608727, -0.085010],
                                                [2.419098, 0.017452, -0.608706, -0.085024]],
                                               [[2.419072, 0.017449, -0.608724, -0.085012],
                                                [2.419119, 0.017455, -0.608691, -0.085034]],
                                               [[2.419019, 0.017442, -0.608761, -0.084989],
                                                [2.419075, 0.017449, -0.608722, -0.085014]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 3, multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerEncoder(encoder_layer, 2, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695949, -0.357635, -0.893077, -0.445238],
                                                [1.695955, -0.357639, -0.893050, -0.445266]],
                                               [[1.695948, -0.357634, -0.893082, -0.445233],
                                                [1.695950, -0.357635, -0.893077, -0.445238]],
                                               [[1.695951, -0.357636, -0.893069, -0.445246],
                                                [1.695955, -0.357639, -0.893052, -0.445264]],
                                               [[1.695952, -0.357636, -0.893066, -0.445249],
                                                [1.695957, -0.357641, -0.893041, -0.445276]],
                                               [[1.695946, -0.357632, -0.893095, -0.445220],
                                                [1.695952, -0.357637, -0.893065, -0.445251]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # TODO: remove set default dtype to double by making ref_output more precise.
        # Added because this test was copied from test_nn.py, which has default
        # dtype double. If default dtype is float, tests will say tensors not close because
        # ref output precision too low
        with set_default_dtype(torch.double):
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()  # transformer fast path requires no grad
            with cm:
                _test(batch_first, training, enable_nested_tensor)

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_encoder_padding_and_src_mask_bool(self):
        # [FUNCTION DOC] Skipped before Python 3.11. Runs a two-layer batch-first
        # TransformerEncoder with a boolean causal source mask and a boolean padding mask, and
        # checks that no log messages are emitted (unless under dynamo), i.e. no mask-type
        # mismatch warning.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(16)
        encoder = nn.TransformerEncoder(
            encoder_layer, 2, encoder_norm
        )

        inputs = torch.randn(2, 3, 16)

        src_mask = torch.ones(3, 3, dtype=torch.bool).triu_(diagonal=1)
        input_seq_len = torch.tensor([3, 2])
        padding_mask = (
            torch.arange(3)[None, :].cpu() >= input_seq_len[:, None]
        )

        with (self.assertNoLogs(None) if not TEST_WITH_TORCHDYNAMO else contextlib.nullcontext()):
            encoder(
                inputs,
                mask=src_mask,
                src_key_padding_mask=padding_mask,
            )

    def test_transformer_encoder_layer_fwd_fake(self, device):
        # [FUNCTION DOC] Checks that a two-layer eval-mode TransformerEncoder compiled with
        # torch.compile (fullgraph, dynamic shapes) produces the same output as eager mode on a
        # random input.
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=256,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=2,
        ).to(device).eval()

        x = torch.rand(2, 10, 128, device=device)
        eager_out = model(x)

        compiled_model = torch.compile(model, fullgraph=True, dynamic=True)
        compiled_out = compiled_model(x)

        self.assertEqual(eager_out, compiled_out)

    def test_transformer_encoder_layer_fwd_unbacked(self, device):
        # [FUNCTION DOC] Regression test for the meta rule of the fused encoder layer forward
        # op: traces a function with make_fx in symbolic mode where the leading size of the
        # source is an unbacked value taken from a tensor. The test passes if tracing does not
        # raise a data-dependent guard error.
        # Regression for a data-dependent error in the meta rule for
        # _transformer_encoder_layer_fwd. The rule used to test
        # `src.numel() == 0` directly, which raises GuardOnDataDependentSymNode
        # when src has an unbacked (data-dependent) size during fake-tensor
        # tracing. Routing that check through guard_or_false lets an undecidable
        # size trace instead of erroring. Drive the fused op's meta with an
        # unbacked leading dim and assert it traces.
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(n_t):
            # [FUNCTION DOC] HELPER: Traced function: reads an integer from the input tensor,
            # constrains it non-negative, creates random source and weight tensors for a small
            # encoder layer with embed dim 16 and 2 heads, and calls the fused ATen
            # encoder-layer forward op with them, returning its output.
            n = n_t.item()
            torch._check(n >= 0)
            embed_dim, num_heads = 16, 2
            src = torch.randn(n, 10, embed_dim, device=device)
            qkv_weight = torch.randn(3 * embed_dim, embed_dim, device=device)
            qkv_bias = torch.randn(3 * embed_dim, device=device)
            proj_weight = torch.randn(embed_dim, embed_dim, device=device)
            proj_bias = torch.randn(embed_dim, device=device)
            norm_weight_1 = torch.randn(embed_dim, device=device)
            norm_bias_1 = torch.randn(embed_dim, device=device)
            norm_weight_2 = torch.randn(embed_dim, device=device)
            norm_bias_2 = torch.randn(embed_dim, device=device)
            ffn_weight_1 = torch.randn(4 * embed_dim, embed_dim, device=device)
            ffn_bias_1 = torch.randn(4 * embed_dim, device=device)
            ffn_weight_2 = torch.randn(embed_dim, 4 * embed_dim, device=device)
            ffn_bias_2 = torch.randn(embed_dim, device=device)
            return torch.ops.aten._transformer_encoder_layer_fwd(
                src, embed_dim, num_heads, qkv_weight, qkv_bias,
                proj_weight, proj_bias, False, False, 1e-5,
                norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2,
                ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, None, None,
            )

        # Without the fix this raises GuardOnDataDependentSymNode while tracing.
        make_fx(fn, tracing_mode="symbolic")(torch.tensor(3, device=device))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/101787")
    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_decoder_padding_and_src_mask_bool(self):
        # [FUNCTION DOC] Skipped before Python 3.11 and under dynamo. Runs a decoder with
        # boolean causal and padding masks and checks that no log messages are emitted (no mask
        # type warnings).

        def transformer_decoder(inputs, input_seq_len, memory):
            # [FUNCTION DOC] HELPER: Builds a two-layer batch-first TransformerDecoder with a
            # final LayerNorm, constructs a boolean upper-triangular causal mask and a boolean
            # padding mask from sequence lengths, and returns the decoder output for the inputs
            # and memory.
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=16,
                nhead=2,
                dim_feedforward=32,
                dropout=0.1,
                activation='relu',
                batch_first=True,
            )
            decoder_norm = nn.LayerNorm(16)
            decoder = nn.TransformerDecoder(
                decoder_layer, 2, decoder_norm
            )

            src_mask = torch.ones(
                inputs.shape[1], inputs.shape[1], dtype=torch.bool
            ).triu_(diagonal=1)
            padding_mask = (
                torch.arange(inputs.shape[1])[None, :].cpu()
                >= input_seq_len[:, None]
            )

            return decoder(
                inputs,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )

        inputs = torch.randn(2, 3, 16)
        memory = torch.randn(2, 3, 16)
        input_seq_len = torch.tensor([3, 2])

        with self.assertNoLogs(None):
            transformer_decoder(inputs, input_seq_len, memory)

    def test_encoder_is_causal(self):
        # [FUNCTION DOC] Checks that a TransformerEncoderLayer in eval mode gives identical
        # output when run with a generated square subsequent mask both with and without the
        # is_causal hint.

        d_model = 3
        layer = torch.nn.TransformerEncoderLayer(d_model, 1, 6, batch_first=True)
        layer.eval()
        x = torch.randn(1, 5, d_model)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        is_causal_output = layer(x, src_mask=mask, is_causal=True)
        masked_output = layer(x, src_mask=mask)

        self.assertEqual(masked_output, is_causal_output)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt pre-SM80 hardware"
    )
    def test_math_backend_high_precision(self):
        # [FUNCTION DOC] CUDA-only test (skipped when flash attention is unsupported). With
        # random bfloat16 inputs it runs the math SDPA backend with low-precision reductions
        # allowed and again with them disallowed, and against a double precision reference cast
        # to bfloat16. The high precision result must match the reference within 1e-2; the low
        # precision result must NOT match (an AssertionError with the message about tensors not
        # being close is expected). Mutates the global CUDA math-SDP reduction flag and leaves
        # it set to disallowed.
        xq = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xk = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xv = torch.randn([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16)
        mask = None

        def scaled_dot_product_attention(
            xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, mask: torch.Tensor | None, backend: SDPBackend
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Nested reference wrapper that transposes query/key/value to
            # head-major layout, repeats key and value once along the heads dimension, runs SDPA
            # restricted to the given backend with no dropout, and transposes the result back.
            n_rep = 1
            xq, xk, xv = (tensor.transpose(1, 2) for tensor in (xq, xk, xv))
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

            with sdpa_kernel(backends=[backend]):
                attn_output = F.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=mask, dropout_p=0.0
                )
            return attn_output.transpose(1, 2)

        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        sdp_math_low_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        sdp_math_high_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)

        sdp_math_fp64_out_ref = scaled_dot_product_attention(
            xq.double(), xk.double(), xv.double(), mask, SDPBackend.MATH
        ).bfloat16()

        torch.testing.assert_close(sdp_math_high_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
            torch.testing.assert_close(sdp_math_low_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

    @onlyCUDA
    @parametrize("nb_heads", [1, 8])
    @parametrize("bias", [True, False])
    def test_mha_native_args(self, nb_heads, bias):
        # [FUNCTION DOC] CUDA-only smoke test: runs an eval-mode batch-first MultiheadAttention
        # (embed dim 128, one or eight heads, with or without bias) on random input under
        # no_grad, passing a zeros boolean key padding mask when bias is enabled. Only checks
        # that the native fast-path call does not raise.

        B, L, F = 8, 100, 128
        batch_first = True
        fast_path = True
        use_pad_mask = (bias % 2) == 1

        mha = nn.MultiheadAttention(
            embed_dim=F,
            num_heads=nb_heads,
            batch_first=batch_first,
            bias=bias
        ).cuda()
        mha.eval()

        ctx = torch.no_grad if fast_path else contextlib.nullcontext
        with ctx():
            x = torch.randn(B, L, F).cuda()
            if not batch_first:
                x = x.transpose(0, 1)

            pad_mask = None
            if use_pad_mask:
                pad_mask = torch.zeros((B, L), dtype=torch.bool).cuda()

            mha(query=x, key=x, value=x, key_padding_mask=pad_mask)

    def test_kpm_mask_trailing_column_with_nested_tensor(self, device):
        # [FUNCTION DOC] Runs a three-layer encoder with nested tensors enabled on a
        # non-batch-first input of length 10, batch 6, embedding 256 with a key padding mask
        # whose first row is all False and the rest True, and checks the output keeps batch size
        # 6 in dimension one.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=False,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        x = torch.randn(10, 6, 256).to(device)
        mask = torch.ones(6, 10)
        mask[0, :] = 0  # here I masked 5 columns instead of just one
        mask = mask.bool().to(device)
        out = transformer_encoder(src=x, src_key_padding_mask=mask)
        self.assertEqual(out.shape[1], 6)

    # CPU unit test has_torch_functions in test environment,
    #   preventing successful completion
    @onlyCUDA
    def test_with_nested_tensor_input(self, device):
        # [FUNCTION DOC] CUDA-only test: builds a nested tensor from a padding mask over an
        # input of batch 6, length 10, embedding 256, passes it to an eval-mode three-layer
        # encoder under no_grad with no padding mask, and checks the output is also a nested
        # tensor.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        transformer_encoder.eval()
        with torch.no_grad():
            x = torch.randn(6, 10, 256).to(device)
            mask = torch.ones(6, 10)
            mask[0, 0:] = 0  # here I masked 5 columns instead of just one
            mask[2, 2:] = 0  # here I masked 5 columns instead of just one
            mask[4, 4:] = 0  # here I masked 5 columns instead of just one
            mask[5, 8:] = 0  # here I masked 5 columns instead of just one
            mask = mask.bool().to(device)
            x = torch._nested_tensor_from_mask(x, mask.logical_not(), mask_check=False)
            out = transformer_encoder(src=x, src_key_padding_mask=None)

        self.assertEqual(out.is_nested, True)



    def test_script_encoder_subclass(self, device):
        # [FUNCTION DOC] Checks that torch.jit.script succeeds on a TransformerEncoder whose
        # layers are a trivial subclass of TransformerEncoderLayer. No numerical checks.
        class MyCustomLayer(nn.TransformerEncoderLayer):
            # [CLASS DOC] HELPER: Empty subclass of TransformerEncoderLayer used to verify
            # scripting works for subclasses.
            pass

        encoder = nn.TransformerEncoder(
            MyCustomLayer(d_model=256, nhead=8), num_layers=6
        ).to(device=device)
        torch.jit.script(encoder)

    # brazenly adapted from test_transformerencoderlayer_src_mask to test execution of
    # torchscripted transformerencoderlayer subclass
    def test_transformerencoderlayer_subclass(self, device):
        # [FUNCTION DOC] Checks that a scripted trivial subclass of TransformerEncoderLayer
        # matches the eager one: with the same seed in train mode, and then in eval mode under
        # no_grad, outputs must be equal.
        class MyCustomLayer(nn.TransformerEncoderLayer):
            # [CLASS DOC] HELPER: Empty subclass of TransformerEncoderLayer used to verify
            # scripting works for subclasses.
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, src_mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, src_mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, src_mask=src_mask)
            scripted_result = script_model(src, src_mask=src_mask)
            self.assertEqual(result, scripted_result)


    def test_transformerencoderlayer_subclass_model(self, device):
        # [FUNCTION DOC] Same as the layer subclass test but wraps a six-layer
        # TransformerEncoder of the subclass: eager and torch.jit.script results with a boolean
        # mask must be equal in train mode (same seed) and in eval mode under no_grad.
        class MyCustomLayer(nn.TransformerEncoderLayer):
            # [CLASS DOC] HELPER: Empty subclass of TransformerEncoderLayer used to verify
            # scripting works for subclasses.
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        layer = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        model = nn.TransformerEncoder(
            layer, num_layers=6
        ).to(device=device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, mask=src_mask)
            scripted_result = script_model(src, mask=src_mask)
            self.assertEqual(result, scripted_result)


    @onlyCUDA
    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
    def test_decoder_only_layer(self):
        # [FUNCTION DOC] CUDA-only test, skipped without fairseq. Defines a fairseq decoder
        # wrapper class; the test body itself never instantiates it, so it checks nothing.
        # KNOWN ISSUE: the test only defines the nested class and has no assertions.
        class FairseqDecoder(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper around the fairseq TransformerDecoder configured from
            # the given sizes (no token positional embeddings, no embedding scaling, no encoder
            # attention), converted to half precision, on CUDA, in eval mode. Forward returns
            # the decoder features only.
            def __init__(
                self,
                embed_dim,
                attention_heads,
                ffn_embed_dim,
                num_layers,
                embedding_layer,  # torch.nn.Embedding. Must have a padding_idx field
                dropout=0,
                normalize_before=False,
                torch_encoder=None,  # torch encoder that you can map weights from
                activation="relu",
            ):
                # [FUNCTION DOC] HELPER: Builds a fairseq transformer config from embed_dim,
                # attention_heads, ffn_embed_dim, num_layers, dropout, normalize_before and
                # activation, creates the fairseq TransformerDecoder with the given embedding
                # layer, optionally maps weights from a torch encoder, and moves it to eval
                # mode, CUDA and half precision.
                # KNOWN ISSUE: torch_to_fairseq is not defined or imported in this file
                # (suppressed lint), so passing a torch_encoder would raise NameError.
                super().__init__()

                cfg = fairseq_transformer.TransformerConfig()
                cfg.decoder.embed_dim = embed_dim
                cfg.decoder.output_dim = embed_dim
                cfg.decoder.attention_heads = attention_heads
                cfg.decoder.ffn_embed_dim = ffn_embed_dim
                cfg.dropout = dropout
                cfg.decoder.normalize_before = normalize_before
                cfg.decoder.layers = num_layers
                # make embedding behavior same as other encoders
                cfg.no_token_positional_embeddings = True
                cfg.no_scale_embedding = True
                cfg.activation_fn = activation

                dictionary = {}  # TODO: verify what this is

                self.decoder = fairseq_transformer.TransformerDecoder(
                    cfg,
                    dictionary,
                    embedding_layer,
                    no_encoder_attn=True,
                    output_projection=None,
                )

                if torch_encoder is not None:
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)  # noqa: F821
                self.decoder = self.decoder.eval().cuda().half()

            def forward(
                self,
                tokens,
                src_lengths=None,
                with_triangle_mask=False,
                incremental_state=None,
            ):
                # [FUNCTION DOC] HELPER: Runs the wrapped decoder on the token tensor in
                # features-only mode, with full context alignment unless with_triangle_mask is
                # true, and returns the first output element.
                return self.decoder(
                    prev_output_tokens=tokens,
                    encoder_out=None,
                    incremental_state=incremental_state,
                    features_only=True,
                    full_context_alignment=not with_triangle_mask,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=src_lengths,
                    return_all_hiddens=False,
                )[0]

    @tf32_on_and_off(0.003)
    @parametrize("batch_size", [1, 5])
    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    @sdpa_kernel(backends=[SDPBackend.MATH])
    def test_scaled_dot_product_attention(self, device, batch_size, input_dim, attn_mask_dim, is_causal, dropout_p):
        # [FUNCTION DOC] Compares F.scaled_dot_product_attention (math backend) against a Python
        # reference for float64 and float32, with 3D or 4D inputs, no mask / random boolean mask
        # / causal lower-triangular mask of 2D or full dimension, several dropout rates, and
        # batch sizes. For a causal mask the explicit mask must be omitted, and passing both a
        # mask and is_causal must raise a RuntimeError. Fully masked rows must give NaN in the
        # reference and zeros in the actual result, then the test returns early. Otherwise the
        # outputs must be equal; when there is no mask a gradcheck of both implementations is
        # run.
        # KNOWN ISSUE: the nested test_incompatible_mask below is defined in this method body
        # and never called, so it never runs.
        def sdp_ref(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0):
            # [FUNCTION DOC] HELPER: Reference attention on 3D tensors: scale the query by the
            # inverse square root of the embedding size, add the optional float mask via batched
            # matrix multiply, apply softmax over the last dimension, optionally apply dropout,
            # and multiply with the value. Returns the output.
            E = q.size(-1)
            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            if attn_mask is not None:
                attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
            else:
                attn = torch.bmm(q, k.transpose(-2, -1))

            attn = torch.nn.functional.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = torch.nn.functional.dropout(attn, p=dropout_p)
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output = torch.bmm(attn, v)
            return output
        # TODO: Support cross-device / dtype testing properly when instantiate_device_type_tests() is used.
        dtypes = [torch.double, torch.float]
        for dtype in dtypes:
            N = batch_size

            def rand_tensor(*shape):
                # [FUNCTION DOC] HELPER: Returns a standard-normal tensor of the given shape on
                # the test device with the loop dtype.
                return torch.randn(shape, device=device, dtype=dtype)

            # This test compares python and C++ implementations of SDP.
            N_prime, L, S, E = 2, 4, 3, 6
            if input_dim == 3:
                query = rand_tensor(N, L, E)
                key = rand_tensor(N, S, E)
                value = rand_tensor(N, S, E)
            elif input_dim == 4:
                query = rand_tensor(N, N_prime, L, E)
                key = rand_tensor(N, N_prime, S, E)
                value = rand_tensor(N, N_prime, S, E)
            else:
                self.fail(f'Invalid input_dim {input_dim} encountered in SDP test')

            attn_mask = None
            if attn_mask_dim is not None:
                if attn_mask_dim not in [2, input_dim]:
                    raise AssertionError(f"attn_mask_dim should be 2 or {input_dim}, got {attn_mask_dim}")
                mask_size = (L, S) if attn_mask_dim == 2 else ((N, L, S) if input_dim == 3 else (N, N_prime, L, S))
                attn_mask = (torch.ones(mask_size, device=device, dtype=torch.bool).tril() if is_causal
                             else torch.randint(0, 2, size=mask_size, device=device, dtype=torch.bool))

            with freeze_rng_state():
                # Python impl only supports float mask and 3D inputs.
                attn_mask_float = attn_mask
                if attn_mask_float is not None:
                    attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
                    attn_mask_float.masked_fill_(attn_mask.logical_not(), float("-inf"))
                q, k, v = query.view(-1, L, E), key.view(-1, S, E), value.view(-1, S, E)
                a = attn_mask_float
                if a is not None and attn_mask_dim > 3:
                    a = a.view(-1, L, S)
                expected = sdp_ref(q, k, v, attn_mask=a, dropout_p=dropout_p)
                if input_dim > 3:
                    expected = expected.view(-1, N_prime, L, E)

            with freeze_rng_state():
                if is_causal:
                    # NB: Don't pass attn_mask here
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, None, dropout_p, is_causal)

                    # Error case: both explicit attn_mask and is_causal are set
                    with self.assertRaisesRegex(RuntimeError,
                                                "Explicit attn_mask should not be set when is_causal=True"):
                        torch.nn.functional.scaled_dot_product_attention(
                            query, key, value, attn_mask, dropout_p, is_causal)
                else:
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask, dropout_p, is_causal)
                    # This test the fully masked out rows case
                if torch.isnan(expected).any():
                    row_sums = attn_mask.sum(dim=-1)
                    masked_out_rows = (row_sums == 0)

                    for _ in range((input_dim - attn_mask_dim) - 1):
                        masked_out_rows = masked_out_rows.unsqueeze(0)

                    masked_out_rows = masked_out_rows.expand(expected.shape[:-1])
                    # Slice out the fully masked rows from expected and actual
                    expected_masked_out = expected[masked_out_rows]
                    actual_masked_out = actual[masked_out_rows]

                    expected_all_nan = torch.isnan(expected_masked_out).all()
                    actual_all_zero = (actual_masked_out.abs().sum() == 0)

                    self.assertTrue(expected_all_nan)
                    self.assertTrue(actual_all_zero)
                    return

                self.assertEqual(actual, expected)

        if attn_mask_dim is None:
            q = q.double().clone()
            k = k.double().clone()
            v = v.double().clone()
            q.requires_grad_()
            k.requires_grad_()
            v.requires_grad_()

            if not gradcheck(lambda *args, **kwargs: wrapper_set_seed(sdp_ref, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p)):
                raise AssertionError("gradcheck failed for sdp_ref")
            if not gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p)):
                raise AssertionError("gradcheck failed for scaled_dot_product_attention")

        def test_incompatible_mask(self, device):
            # [FUNCTION DOC] HELPER: Intended test (but never invoked, as it is nested inside
            # another test) that a MultiheadAttention given an incompatible float key padding
            # mask and boolean attention mask raises RuntimeError.
            def ones_tensor(*shape):
                # [FUNCTION DOC] HELPER: Returns a float32 tensor of ones with the given shape.
                return torch.ones(shape, dtype=torch.float32)
            S, L, E, H = 1, 2, 4, 1
            qkv = ones_tensor(S, L, E)

            mha = nn.MultiheadAttention(E, H)
            mha.in_proj_weight = Parameter(torch.ones((E * 3, E)))
            mha.out_proj.weight = Parameter(torch.ones((E, E)))
            qkv = qkv.to(float)
            kpm = ones_tensor(S, L) * float("-inf")
            am = ones_tensor(L, L).to(bool)

            def func():
                # [FUNCTION DOC] HELPER: Calls the attention module with the shared tensors, no
                # weights, and both masks, returning the result; expected to raise.
                return mha(qkv, qkv, qkv, need_weights=False, key_padding_mask=kpm, attn_mask=am)

            self.assertRaises(RuntimeError, func)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_mask_check_fastpath(self):
        # [FUNCTION DOC] Checks which encoder fast path is taken by patching the fused encoder
        # layer function: with nested tensors and mask checking, a left aligned padding mask
        # uses nested tensors and a non-aligned one uses regular tensors; without nested tensors
        # the fast path is always taken, also with both an attention mask and a padding mask;
        # with mask checking disabled nested tensors are always used.
        """
        Test that fastpath is executed independently of the masks that are passed.
        If the passed key padding mask is left aligned or mask_check=False, test that nested tensors are used
        (sparsity fastpath), otherwise use fastpath with traditional tensors.
        Also test that fast path is executed with both key padding mask and attention mask passed at the same time.
        """

        x = torch.Tensor([[[1, 2], [3, 4], [5, 6]]]).to(torch.float)

        def _test_fastpath(model, key_padding_mask, mock_return_value, attn_mask=None, nested_tensors=True):
            # [FUNCTION DOC] HELPER: Patches the fused layer function to return the given value,
            # runs the model on the shared input with the masks, asserts the patched function
            # was called, and that the first argument of each call is nested exactly when
            # nested_tensors is set.
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = mock_return_value
                model(x, src_key_padding_mask=key_padding_mask, mask=attn_mask)

                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

                # If mock was called with nested tensors, sparsity fastpath was taken
                for call_args, _ in fastpath_mock.call_args_list:
                    self.assertEqual(call_args[0].is_nested, nested_tensors)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        model.eval()

        aligned_key_padding_mask = torch.Tensor([[0, 0, 1]]).to(torch.bool)
        not_aligned_key_padding_mask = torch.Tensor([[1, 0, 1]]).to(torch.bool)
        attn_mask = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).to(torch.bool)
        nested_tensor_return_value = torch.nested.nested_tensor([torch.ones((2, 2), dtype=torch.float)])
        tensor_return_value = torch.ones((1, 3, 2), dtype=torch.float)

        # Left aligned mask results in sparsity fastpath
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

        # Not aligned mask results in fastpath
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False, mask_check=True)
        model.eval()

        # If nested tensor disabled, fastpath is always taken
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        # Fast path is taken if both attention mask and key padding mask are present
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, attn_mask=attn_mask, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=False)
        model.eval()

        # Mask check disabled results in sparisty fastpath, independently of the mask
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)
        _test_fastpath(model, not_aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

    # Test failing MHA when bias was NoneType
    def test_bias_is_none(self):
        # [FUNCTION DOC] Regression test: an eval-mode MultiheadAttention built with no bias
        # must run a forward without error.
        x = torch.rand((1, 5, 10))
        model = torch.nn.modules.activation.MultiheadAttention(10, 1, bias=False, batch_first=True)
        model.eval()
        model(x, x, x)
        # completes without error

    def test_transformer_bias_is_none(self, device):
        # [FUNCTION DOC] Checks that a TransformerEncoderLayer without bias runs in eval mode,
        # and that wrapping it in a TransformerEncoder or building a Transformer with no bias
        # emits a UserWarning that self_attn was passed no bias.
        batch_size = 2
        seqlen = 3
        d_model = 8
        nhead = 4

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, bias=False, batch_first=True, device=device)
        encoder_layer.eval()
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        # runs without error
        encoder_layer(x)

        with self.assertWarnsRegex(UserWarning, "encoder_layer.self_attn was passed bias=False"):
            encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1).eval()
            encoder(x)

        with self.assertWarnsRegex(UserWarning, "self_attn was passed bias=False"):
            transformer = torch.nn.Transformer(
                d_model=d_model, nhead=nhead, bias=False, batch_first=True, device=device
            ).eval()
            transformer(x, x)

    def test_train_with_is_causal(self, device):
        # [FUNCTION DOC] Trains a two-layer encoder one step with is_causal and a square
        # subsequent mask; then checks MultiheadAttention with a mask and is_causal works while
        # is_causal alone raises RuntimeError; checks that passing a boolean causal mask makes
        # the encoder call its second layer (replaced by a mock) with is_causal true; finally
        # runs the math-kernel numeric check via is_causal_kernels.
        # training with is_causal
        S, L, E, H = 1, 2, 2, 1
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=H,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()
        optimizer.zero_grad()
        inputs = torch.randn(S, L, E).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            inputs.size(1), device=device
        )

        outputs = encoder(inputs, mask=mask, is_causal=True)

        loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
        loss.backward()
        optimizer.step()

        # inference with is_causal
        t_qvk = torch.randn((S, L, E), device=device, dtype=torch.float32)
        mha = nn.MultiheadAttention(E, H).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            S, device=device
        )

        attn_out, _ = mha(t_qvk, t_qvk, t_qvk, attn_mask=mask, is_causal=True)

        # Can't give only is_causal
        with self.assertRaises(RuntimeError):
            mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # # Passing a causal mask sets is_causal to 1
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[1] = mock_layer
        outputs = encoder(inputs, mask=causal_mask)
        mock_layer.assert_called_with(ANY, src_mask=ANY, is_causal=True, src_key_padding_mask=ANY)

        # check expected numerical values with all kernels
        self.is_causal_kernels([SDPBackend.MATH], device)

    def is_causal_kernels(self, kernels, device):
        # [FUNCTION DOC] HELPER: Shared check (not a test by name): for each SDPA backend, runs
        # a MultiheadAttention with all-ones weights and inputs, a causal mask and is_causal,
        # and requires the exact expected output (all values 16). For non-math backends also
        # expects a RuntimeError saying no kernel is available when the embedding size (2) is
        # not a multiple of 4, then synchronizes CUDA.
        def ones_tensor(*shape):
            # [FUNCTION DOC] HELPER: Returns a float32 tensor of ones with the given shape on
            # the test device.
            return torch.ones(shape, device=device, dtype=torch.float32).to(device)
        S, L, E, H = 1, 2, 4, 1
        qkv = ones_tensor(S, L, E)

        mha = nn.MultiheadAttention(E, H).to(device)
        mha.in_proj_weight = Parameter(torch.ones((E * 3, E), device=device))
        mha.out_proj.weight = Parameter(torch.ones((E, E), device=device))
        expected = torch.ones(size=(S, L, E)).to(device) * 16
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            qkv.size(1), device=device
        )

        for kernel in kernels:
            with sdpa_kernel(backends=[kernel]):
                actual, _ = mha(qkv, qkv, qkv, attn_mask=mask, need_weights=False, is_causal=True)
                self.assertTrue(torch.equal(actual, expected))

                if kernel != SDPBackend.MATH:
                    # fails with embedding size not multiple of 4
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            qkv_f.size(1), device=device
                        )
                        _ = mha_f(qkv_f, qkv_f, qkv_f, attn_mask=mask, need_weights=False, is_causal=True)
                        torch.cuda.synchronize()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt fused SDPA or pre-SM80 hardware"
    )
    def test_is_causal_gpu(self):
        # [FUNCTION DOC] Runs the causal-kernel check on CUDA for the math and efficient
        # attention backends; skipped when flash attention is unsupported on the platform.
        device = 'cuda'
        self.is_causal_kernels([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], device)

    def test_script_mha_in_proj_weight_none(self):
        # [FUNCTION DOC] Checks that torch.jit.script succeeds for an eval-mode
        # MultiheadAttention with different key and value dimensions (so its input projection
        # weight is absent).
        mha = torch.nn.MultiheadAttention(
            embed_dim=128, num_heads=8, kdim=256, vdim=256
        ).eval()

        torch.jit.script(mha)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_disable_fastpath(self, device):
        # [FUNCTION DOC] Verifies the global MHA fast path switch: with patched native functions
        # it checks that the encoder, the full Transformer and MultiheadAttention call the fast
        # path functions when enabled, do not call them after disabling the fast path, and call
        # them again after re-enabling. Runs under no_grad and leaves the switch enabled.
        def _test_te_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            # [FUNCTION DOC] HELPER: Patches the fused encoder layer function with a mock
            # returning the given value, calls the model with the positional and keyword
            # arguments, and asserts that the mock's called state equals is_called.
            if kwargs is None:
                kwargs = {}
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        def _test_mha_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            # [FUNCTION DOC] HELPER: Same as the encoder helper but patches the native multi
            # head attention function.
            if kwargs is None:
                kwargs = {}
            with patch('torch._native_multi_head_attention') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        inp = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32, device=device)
        src_key_padding_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool, device=device)
        te_return_value = torch.ones((1, 3, 2), dtype=torch.float32)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)
        te = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        te = te.to(device).eval()

        t = torch.nn.Transformer(d_model=2, nhead=2, batch_first=True, device=device).eval()
        src = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]], dtype=torch.float32, device=device)
        t_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        mha = nn.MultiheadAttention(2, 2, batch_first=True, device=device).eval()
        q = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.float32, device=device)
        mha_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)

        torch.backends.mha.set_fastpath_enabled(False)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=False
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=False)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=False)

        torch.backends.mha.set_fastpath_enabled(True)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)


class TestSDPAFailureModes(NNTestCase):
    # [CLASS DOC] Test class for the error paths of scaled_dot_product_attention: invalid
    # shapes, dtypes, devices, strides, alignment, unsupported masks or head dims for the flash,
    # efficient and cuDNN backends, and zero-length or unbacked-size inputs. Most tests restrict
    # the backend with the sdpa_kernel context and expect RuntimeError (often with a UserWarning
    # explaining the reason). Enables CUDA leak and non-default-stream checks.
    """ Used to test the failure modes of scaled_dot_product_attention
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @parametrize(
        "backend",
        [
            None,
            SDPBackend.ERROR,
            SDPBackend.MATH,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.OVERRIDEABLE,
        ],
        name_fn=lambda backend: "default" if backend is None else backend.name.lower(),
    )
    def test_attn_mask_and_is_causal_errors_before_dispatch(self, device, backend):
        # [FUNCTION DOC] Checks that passing both an explicit attention mask and is_causal
        # raises a RuntimeError with the explicit-mask message before any backend is selected,
        # for the default dispatch and for each listed backend (including the error and
        # overrideable ones) on random 4D float inputs.
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        v = torch.randn(2, 4, 8, 16, device=device)
        attn_mask = torch.zeros(8, 8, device=device)

        expected_error = "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True"
        context = contextlib.nullcontext() if backend is None else sdpa_kernel(backends=[backend])
        with context:
            with self.assertRaisesRegex(RuntimeError, expected_error):
                F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice or not isSM120Device,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    @parametrize("head_dim", [193, 256])
    @parametrize("dropout_p", [0.0, 0.2])
    def test_flash_backward_failure_sm86plus(self, device, head_dim: int, dropout_p: float):
        # [FUNCTION DOC] CUDA test for flash attention backward limits on SM 8.6/8.9 style
        # hardware: with head dim 193 or 256, forward without gradients must match the math
        # backend; with inputs requiring gradients the call must raise RuntimeError for head
        # dims above 192 up to 224 (or above 224 with dropout) and succeed otherwise.
        # KNOWN ISSUE: the skip condition requires the device to be both SM8X and SM12x, which
        # cannot both hold, so the test is always skipped.
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype)
        # See check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89 in
        # pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
        size = (2, 2, 4, head_dim)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # Should not fail because inputs don't require grad
            flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

            self.assertEqual(math_ref, flash_ref, atol=1e-3, rtol=1e-3)

            # Should fail because inputs require grad
            q = make_tensor(size, requires_grad=True)
            k = make_tensor(size, requires_grad=True)
            v = make_tensor(size, requires_grad=True)
            if 192 < head_dim <= 224 or (head_dim > 224 and dropout_p != 0.0):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, None, dropout_p, False
                    ),
                )
            else:
                flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p, False)

    @onlyCUDA
    def test_dispatch_fails_no_backend(self, device):
        # [FUNCTION DOC] CUDA test: with the backend list set to the error backend, both the
        # fused backend choice query and F.scaled_dot_product_attention on half precision 3D
        # inputs raise RuntimeError saying no viable backend was found.
        dtype = torch.float16
        with sdpa_kernel(backends=[SDPBackend.ERROR]):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_fused_inputs_dim_3(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Checks that 3D (unbatched) half precision inputs are accepted by the
        # chosen fused backend: the fused backend choice must equal the requested backend, and
        # the result equals applying the op to inputs with a leading dimension added then
        # removed. Uses a random 3x3 mask except for flash.
        size = (2, 3, 8)
        dtype = torch.float16
        q = torch.randn(size, device=device, dtype=dtype)
        k = torch.randn(size, device=device, dtype=dtype)
        v = torch.randn(size, device=device, dtype=dtype)
        attn_mask = (
            None
            if kernel == SDPBackend.FLASH_ATTENTION
            else torch.randn(3, 3, device=device, dtype=dtype)
        )

        with sdpa_kernel(backends=[kernel]):
            self.assertEqual(
                torch._fused_sdp_choice(q, k, v, attn_mask), kernel.value
            )
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask
            ).squeeze(0)
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask
            )

        self.assertEqual(actual, expected)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_broadcast(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Checks that fused backends reject dense inputs where query batch size
        # is 1 and key/value batch size is 2 (no broadcasting), raising RuntimeError.
        with sdpa_kernel(backends=[kernel]):
            #  Fused Kernels don't support broadcasting for dense inputs
            dtype = torch.float16
            size = (2, 4, 3, 8)
            size_broadcast = (1, 4, 3, 8)
            q = torch.randn(size_broadcast, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    def test_zero_sequence_lengths(self, device):
        # [FUNCTION DOC] Checks SDPA on half precision inputs with sequence length zero returns
        # an empty output of shape batch 2, heads 2, length 0, head dim 8 equal to a zeros
        # tensor.
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype)
        size = SdpaShape(2, 2, 0, 8)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)
        self.assertEqual(out.shape, (2, 2, 0, 8))
        self.assertEqual(out, torch.zeros(2, 2, 0, 8, dtype=dtype, device=device))

    @skipIfTorchDynamo("mark_dynamic is not traceable")
    def test_sdpa_unbacked_no_dde(self, device):
        # [FUNCTION DOC] Regression test: compiles a plain SDPA function with torch.compile
        # (eager backend, full graph) after marking the batch dimension dynamic, and checks the
        # output shape matches eager mode, i.e. no data-dependent guard error arises while
        # checking for empty tensors.
        """
        Test that scaled_dot_product_attention with dynamic symbolic dimensions
        does not raise GuardOnDataDependentSymNode when checking for empty tensors.
        This tests the TORCH_GUARD_OR_FALSE fix in attention.cpp.
        Regression test for D94287846.
        """

        def func(q, k, v):
            # [FUNCTION DOC] HELPER: Returns scaled dot product attention of the three inputs.
            return F.scaled_dot_product_attention(q, k, v)

        dtype = torch.float32
        q = torch.rand(4, 2, 8, 16, dtype=dtype, device=device)
        k = torch.rand(4, 2, 8, 16, dtype=dtype, device=device)
        v = torch.rand(4, 2, 8, 16, dtype=dtype, device=device)

        torch._dynamo.mark_dynamic(q, 0)
        torch._dynamo.mark_dynamic(k, 0)
        torch._dynamo.mark_dynamic(v, 0)

        torch._dynamo.reset()
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(q, k, v)
        expected = func(q, k, v)
        self.assertEqual(result.shape, expected.shape)

    @skipIfTorchDynamo("mark_dynamic is not traceable")
    def test_sdpa_empty_unbacked_no_dde(self, device):
        # [FUNCTION DOC] Regression test: same as the unbacked test but with the sequence
        # dimension of zero length marked dynamic; the compiled result must have shape batch 2,
        # heads 2, length 0, head dim 8.
        """
        Test that scaled_dot_product_attention with zero sequence length and
        dynamic dimensions does not raise GuardOnDataDependentSymNode.
        This tests the early return path in attention.cpp with symbolic shapes.
        Regression test for D94287846.
        """

        def func(q, k, v):
            # [FUNCTION DOC] HELPER: Returns scaled dot product attention of the three inputs.
            return F.scaled_dot_product_attention(q, k, v)

        dtype = torch.float32
        q = torch.rand(2, 2, 0, 8, dtype=dtype, device=device)
        k = torch.rand(2, 2, 0, 8, dtype=dtype, device=device)
        v = torch.rand(2, 2, 0, 8, dtype=dtype, device=device)

        torch._dynamo.mark_dynamic(q, 2)
        torch._dynamo.mark_dynamic(k, 2)
        torch._dynamo.mark_dynamic(v, 2)

        torch._dynamo.reset()
        compiled_func = torch.compile(func, fullgraph=True, backend="eager")
        result = compiled_func(q, k, v)
        self.assertEqual(result.shape, (2, 2, 0, 8))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_last_dim_stride(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] CUDA test: after a strided view makes the last dimension of the query
        # have stride other than one, SDPA on a fused backend must emit a UserWarning that last
        # dimension stride 1 is required and raise RuntimeError.
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with last dim stride not equal to 1 will error
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            q.as_strided_(size, [2, 2, 2, 2])
            with self.assertWarnsRegex(UserWarning, "All fused kernels require the last dimension of the input to have stride 1."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
        or not PLATFORM_SUPPORTS_CUDNN_ATTENTION,
        "Efficient or cuDNN Attention was not built for this system",
    )
    @parametrize("kernel", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])
    def test_mask_invalid_last_dim_stride(self, device, kernel):
        # [FUNCTION DOC] CUDA test: with the efficient or cuDNN backend, an attention mask whose
        # last dimension stride is not one makes SDPA warn (GPU backends require stride 1) and
        # raise RuntimeError.
        with sdpa_kernel(backends=[kernel]):
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            attn_mask = make_tensor((2, 2, 8, 8))
            # Passing in a attn_mask with last dim stride not equal to 1 will error
            attn_mask.as_strided_(size, [2, 2, 2, 2])

            with self.assertWarnsRegex(
                UserWarning,
                "GPU backends require attn_mask's last dimension to have stride 1 while the CPU does not",
            ):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask, 0.0, False
                    ),
                )

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("fused_kernel", [SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_sdpa_kernel_grouped_query_attention_cuda(self, device, fused_kernel):
        # [FUNCTION DOC] CUDA test: grouped query shapes (8 query heads, 4 key/value heads)
        # without the GQA flag on the efficient backend must warn about dense input requirements
        # and raise a RuntimeError reporting no available kernel.
        rand_query = torch.rand(8, 8, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_key = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_value = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)

        with sdpa_kernel(fused_kernel):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                with self.assertWarnsRegex(UserWarning, "For dense inputs, both fused kernels require query, "
                                           "key and value to have"):
                    F.scaled_dot_product_attention(rand_query, rand_key, rand_value, dropout_p=0.0,
                                                   is_causal=False, enable_gqa=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not flash_attention fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_fused_inputs_head_dim(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] CUDA test: head dim 9 (efficient) or 257 (513 on ROCm) for half
        # precision inputs must make SDPA raise RuntimeError on the selected fused backend.
        with sdpa_kernel(backends=[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 3, 9) if kernel == SDPBackend.EFFICIENT_ATTENTION else SdpaShape(2, 2, 3, 257)
            if TEST_WITH_ROCM:  # On ROCM, FA and EA share the backend GPU kernels
                size = SdpaShape(2, 2, 3, 513)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_invalid_dtype(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] CUDA test: float64 inputs are invalid for the fused backends and make
        # SDPA raise RuntimeError.
        with sdpa_kernel(backends=[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, device=device, dtype=torch.float64)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION])
    def test_invalid_fused_inputs_attn_mask_present(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] CUDA test: passing any attention mask to the flash backend must raise
        # RuntimeError.
        with sdpa_kernel(backends=[kernel]):
            # Failures for unsupported SDP args
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, size, device=device, dtype=torch.float16)
            q, k, v = make_tensor(), make_tensor(), make_tensor()
            # Non-None attention mask
            mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_unaligned_tensors(self, device):
        # [FUNCTION DOC] CUDA test: half precision inputs with head dim 5 (unaligned) on the
        # efficient backend must raise RuntimeError.
        # The alignment is dependent on arch so we specify SM80OrLater
        dtype = torch.float16
        size = SdpaShape(2, 2, 8, 5)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            ctxmgr = self.assertRaises(RuntimeError)
            with ctxmgr:
                torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "CUTLASS mem efficient attention alignment check is CUDA-only")
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support mem efficient attention")
    def test_mem_efficient_attention_misaligned_data_ptr_sm80_or_later(self, device):
        # [FUNCTION DOC] CUDA test: query, key and value are views offset by one to three
        # elements into one float32 storage, so their data pointers are misaligned. Checks that
        # the backend choice falls back to math and matches the math result, and that
        # restricting to the efficient backend warns about storage offsets and raises
        # RuntimeError. Skipped on ROCm and pre-SM80 hardware.
        if torch.cuda.get_device_capability(device)[0] < 8:
            self.skipTest("sm80 or newer requires aligned mem efficient attention kernels")

        B, H, S, D = 6, 4, 64, 64
        storage = torch.zeros(B * H * S * D + 4, dtype=torch.float32, device=device)
        q = storage[1:1 + B * H * S * D].view(B, H, S, D)
        k = storage[2:2 + B * H * S * D].view(B, H, S, D)
        v = storage[3:3 + B * H * S * D].view(B, H, S, D)
        alignment_bytes = 4 * q.element_size()
        self.assertNotEqual(q.data_ptr() % alignment_bytes, 0)
        self.assertNotEqual(k.data_ptr() % alignment_bytes, 0)
        self.assertNotEqual(v.data_ptr() % alignment_bytes, 0)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            self.assertEqual(torch._fused_sdp_choice(q, k, v), SDPBackend.MATH.value)
            actual = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(actual, expected)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "storage offsets"):
                self.assertRaisesRegex(
                    RuntimeError,
                    "No available kernel|No viable backend",
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
                )

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_flash_fail_fp32(self, device):
        # [FUNCTION DOC] CUDA test: float32 inputs on the flash backend emit a UserWarning about
        # the expected dtypes (half or bfloat16) and raise RuntimeError.
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, BFloat16}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_float16(self, device):
        # [FUNCTION DOC] CUDA test: float32 inputs under float16 autocast must run on the flash
        # backend without error; the result is discarded.
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_bfloat16(self, device):
        # [FUNCTION DOC] CUDA test: float32 inputs under bfloat16 autocast must run on the flash
        # backend without error; the result is discarded.
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    # Note: do not truncate the list according to platforms. These tests should always raise errors.
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_datatypes(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Checks that query float32 with key and value float16 raises
        # RuntimeError for the math, flash or efficient backend, which are always tested
        # regardless of platform.
        with sdpa_kernel(backends=[kernel]):
            # Different datatypes
            shape = (1, 4, 8, 16)
            query = torch.randn(shape, dtype=torch.float32, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_devices(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] CUDA test: query on the CUDA device with key and value on the CPU
        # raises RuntimeError.
        # Different devices
        shape = (1, 4, 8, 16)
        query = torch.randn(shape, dtype=torch.float32, device=device)
        key = torch.randn(shape, dtype=torch.float16, device='cpu')
        value = torch.randn(shape, dtype=torch.float16, device='cpu')
        self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_1_dimensional_inputs(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Checks that a one-dimensional query with two-dimensional key and value
        # raises RuntimeError on the math, flash and efficient backends.
        with sdpa_kernel(backends=[kernel]):
            # 1 dimensional input
            shape = (1, 4)
            query = torch.randn(4, dtype=torch.float16, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_error_cases(self, device):
        # [FUNCTION DOC] CUDA test: a nested query with per-sequence lengths, a nested key of
        # batch size 1 and a nested value with different lengths cannot be broadcast, so the
        # efficient backend raises a RuntimeError reporting no available kernel.
        # one of k,v needs to be broadcasted and other has non consistent seq_len dim
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim = 32, 8, 64
        seq_lens_q = torch.randint(low=1, high=32, size=(batch,)).tolist()
        seq_lens_v = torch.randint(low=1, high=32, size=(batch,)).tolist()

        q_shape = SdpaShape(batch, num_heads, seq_lens_q, head_dim)
        k_shape = SdpaShape(1, num_heads, 1, head_dim)
        v_shape = SdpaShape(batch, num_heads, seq_lens_v, head_dim)

        query = rand_nested_tensor(q_shape).transpose(1, 2)
        key = rand_nested_tensor(k_shape).transpose(1, 2)
        value = rand_nested_tensor(v_shape).transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_nested_fails_on_padding_head_dim(self, device):
        # [FUNCTION DOC] CUDA test: nested bfloat16 inputs with head dim 57 on the flash backend
        # must warn that nested tensors need padded head dims and raise RuntimeError.
        dtype = torch.bfloat16
        seq_len_list = [2, 4, 5, 6, 7]
        shape = SdpaShape(5, 8, seq_len_list, 57)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type="nested", device=device, dtype=dtype)
        q, k, v = make_tensor().transpose(1, 2), make_tensor().transpose(1, 2), make_tensor().transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "For NestedTensor inputs, Flash attention requires"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION or not isLessThanSM80Device,
                     "Current platform does not support fused SDPA or is an SM80+ device.")
    def test_mem_efficient_fail_bfloat16_less_than_sm80(self, device):
        # [FUNCTION DOC] CUDA test, only on pre-SM80 hardware: bfloat16 inputs on the efficient
        # backend warn that only half and float are supported and raise RuntimeError.
        dtype = torch.bfloat16
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, Float}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_atteention_large_bf16_nan_values(self, device):
        # [FUNCTION DOC] CUDA test: with very large bfloat16 values (133120) in size-1 inputs,
        # the flash backend output must not contain NaN.
        query = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        key = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        value = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        self.assertFalse(torch.isnan(out).any(), "Output should not contain NaNs!")

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_0_inputs(self, device, fused_kernel):
        # [FUNCTION DOC] CUDA test: nested half precision inputs where several sequences have
        # zero length make the selected fused backend raise RuntimeError reporting no available
        # kernel.
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 0
        num_zeros = 10
        indices = torch.randint(low=0, high=batch, size=(num_zeros,))
        seq_lens.scatter_(0, indices, 0)

        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_requires_grad_failure(self, device):
        # [FUNCTION DOC] CUDA test: a dense query that requires grad with nested key/value that
        # need broadcasting makes the flash backend warn that training with broadcasted nested
        # inputs is unsupported and raise a RuntimeError reporting no available kernel.
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16, requires_grad=True)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = SdpaShape(1, num_heads, 1, head_dim)
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # create a dense query
        query = torch.randn(q_shape, device=device, dtype=torch.float16, requires_grad=True)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support training with broadcasted NT inputs"):
                with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                    torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_attention_fail_with_non_square_causal_attention(self, device):
        # [FUNCTION DOC] CUDA test: flash attention with causal masking and unequal query and
        # key lengths (8 versus 12) warns that the combination is unsupported and raises
        # RuntimeError.
        dtype = torch.bfloat16
        q_shape = SdpaShape(1, 1, 8, 16)
        kv_shape = SdpaShape(1, 1, 12, 16)
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        q, k, v = make_q(), make_kv(), make_kv()
        warning_str = "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k."
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, warning_str):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, is_causal=True))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_fail_with_batch_size_geq_65536(self):
        # [FUNCTION DOC] CUDA test: batch size 65536 on the efficient backend must still work:
        # forward and backward outputs and gradients are compared with a CPU computation with
        # atol 2e-3 and rtol 1e-4.
        batch_size = 2**16
        query = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        key = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        value = torch.rand([batch_size, 2, 2, 8], device='cuda', dtype=torch.float16, requires_grad=True)
        q_cpu, k_cpu, v_cpu = (query.detach().cpu().requires_grad_(True),
                               key.detach().cpu().requires_grad_(True),
                               value.detach().cpu().requires_grad_(True))
        with sdpa_kernel(backends=SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(query, key, value)
        out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        grad_out = torch.rand_like(out)
        out.backward(grad_out)
        out_cpu.backward(grad_out.cpu())

        self.assertEqual(out, out_cpu, atol=2e-3, rtol=1e-4)
        self.assertEqual(query.grad, q_cpu.grad, atol=2e-3, rtol=1e-4)
        self.assertEqual(key.grad, k_cpu.grad, atol=2e-3, rtol=1e-4)
        self.assertEqual(value.grad, v_cpu.grad, atol=2e-3, rtol=1e-4)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_fail_with_batch_size_geq_65536_error(self):
        # [FUNCTION DOC] CUDA test: calling the efficient attention op directly with batch size
        # 65536 and dropout and log-sumexp computing must raise RuntimeError saying seed and
        # offset outputs cannot be produced above 65535.
        query = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        key = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        value = torch.rand([2**16, 2, 2, 8], device='cuda', dtype=torch.float16)
        error_str = (r"Efficient attention cannot produce valid seed and offset outputs when "
                     r"the batch size exceeds \(65535\)\.")
        with self.assertRaisesRegex(RuntimeError, error_str):
            torch._scaled_dot_product_efficient_attention(query, key, value,
                                                          attn_bias=None, compute_log_sumexp=True,
                                                          dropout_p=0.01)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_backward_requires_bias_when_bias_grad(self):
        # [FUNCTION DOC] CUDA test: calling the efficient attention backward op with output mask
        # requesting a bias gradient but no bias must raise RuntimeError that bias_requires_grad
        # is true but no bias was provided.
        q = torch.rand(2, 8, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.rand(2, 8, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.rand(2, 8, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        out, lse, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q, k, v, None, True
        )
        grad_out = torch.rand_like(out)
        error_str = "bias_requires_grad is true but no bias was provided"
        with self.assertRaisesRegex(RuntimeError, error_str):
            torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                grad_out,
                q,
                k,
                v,
                None,
                out,
                lse,
                philox_seed,
                philox_offset,
                0.0,
                (True, True, True, True),
                False,
                scale=None,
            )

    @largeTensorTest("15GB", "cuda")
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_large_seq_len_uniform_attention(self):
        # [FUNCTION DOC] Large tensor CUDA test (15GB): 49999 queries with all zero query/key
        # and ones value and an all-True boolean mask on the efficient backend. Output must be
        # all ones, query and key gradients zero and value gradient ones.
        device = torch.device("cuda")
        dtype = torch.bfloat16

        num_queries = 49999
        num_heads = 2
        feature_dim = 16

        # Q and K are all zeros -> uniform attention
        query = torch.zeros(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        key = torch.zeros(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        value = torch.ones(1, num_heads, num_queries, feature_dim, device=device, dtype=dtype, requires_grad=True)
        mask = torch.ones((num_queries, num_queries), dtype=torch.bool, device=device)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
            )
            expected = torch.ones_like(output)
            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            self.assertTrue(torch.allclose(output, expected))
            self.assertTrue(torch.allclose(query.grad, torch.zeros_like(query)))
            self.assertTrue(torch.allclose(key.grad, torch.zeros_like(key)))
            # For value, since each input position contributed 1/num_queries to each output, the grad should sum accordingly
            # for all ones grad_output, each value position receives grad of 1 (because sum of all softmax weights per row is 1)
            self.assertTrue(torch.allclose(value.grad, torch.ones_like(value)))

    @largeTensorTest("12GB", "cuda")
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support Efficient Attention")
    def test_mem_eff_attention_large_seq_len_attn_mask_index_overflow(self):
        # [FUNCTION DOC] Large tensor CUDA test (12GB): regression for a 32-bit offset overflow
        # in the efficient kernel when an attention mask is used and sequence length squared
        # exceeds 2**32. Compares output with an explicit additive causal mask (sequence length
        # 66000) with that of the causal flag (atol and rtol 5e-3).
        # When an attn_mask is given and seq_len**2 > 2**32, the kernel used to
        # compute the per-row mask offset (query_start * bias_strideM) in
        # 32-bit arithmetic, which wraps around and silently corrupts the
        # output of all rows past 2**32 // seq_len.
        device = torch.device("cuda")
        dtype = torch.bfloat16

        seq_len = 66000  # seq_len**2 > 2**32
        head_dim = 16
        torch.manual_seed(0)
        make_tensor = partial(
            torch.randn, 1, 1, seq_len, head_dim, device=device, dtype=dtype
        )
        query, key, value = make_tensor() * 0.1, make_tensor() * 0.1, make_tensor()

        # additive causal mask: 0 on/below the diagonal, -inf above
        attn_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        ).triu_(1)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            ref = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask
            )
        # before the fix, all rows past 2**32 // seq_len were garbage
        self.assertEqual(out, ref, atol=5e-3, rtol=5e-3)


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # [FUNCTION DOC] HELPER: Mirrors the CUDA flash kernel key block size choice (must match the
    # kernel). Raises AssertionError for head dim above 256. Returns 128 for head dim up to 32;
    # 128 (64 with dropout) up to 64; 64 up to 96; up to 128: on SM 8.6/8.9 style devices 64
    # only for causal without dropout else 32, on others 64 without dropout else 32; up to 160:
    # 64 on SM 8.6/8.9 else 32; 64 up to 256.
    # This should match the block sizes in the CUDA kernel
    if head_dim > 256:
        raise AssertionError(f"head_dim should be <= 256, got {head_dim}")
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def pad_last_dim(input_tensor, alignment_size, slice: bool = False):
    # [FUNCTION DOC] HELPER: Pads the last dimension of a tensor with zeros up to the next
    # multiple of alignment_size. Returns a tuple of the tensor and the original last dimension
    # size. If already aligned returns the input unchanged; if slice is true, returns a view of
    # the padded tensor cut back to the original size (so the storage is padded but the shape is
    # not).
    last_dim_size = input_tensor.size(-1)
    if (last_dim_size % alignment_size == 0):
        return input_tensor, last_dim_size
    pad_count = alignment_size - (last_dim_size % alignment_size)
    padded_tensor = F.pad(input_tensor, (0, pad_count))
    if slice:
        return padded_tensor[..., :last_dim_size], last_dim_size
    return padded_tensor, last_dim_size


class TestSDPA(NNTestCase):
    # [CLASS DOC] Device-generic tests of core scaled_dot_product_attention behavior that must
    # hold on all devices: math backend gradients, negative scale, fp16 overflow, zero-sized
    # inputs, output shape from the value head dim, and export with unbacked mask shapes.
    # CUDA-only tests belong elsewhere.
    """ Used to test generic functionality of scaled_dot_product_attention
    Summary:
        If you are adding a new test to this class, make sure that it runs
        for both cpu and cuda. If you're test is only applicable to cuda,
        add it to TestSDPACudaOnly.
    """
    @expectedFailureMPS  # No double support
    @parametrize("contiguous_inputs", [True, False])
    def test_sdp_math_gradcheck(self, device, contiguous_inputs: bool):
        # [FUNCTION DOC] Gradcheck of the math SDPA backend in float64 for query/key/value taken
        # from a packed qkv tensor (chunked, reshaped and transposed), either contiguous or left
        # as views. Raises AssertionError if gradcheck fails. Expected failure on MPS (no double
        # support).

        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            if not gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (query, key, value, None, 0.0, False)
                             ):
                raise AssertionError("gradcheck failed for scaled_dot_product_attention")

    @parametrize("kernel", [SDPBackend.MATH])
    def test_scaled_dot_product_attention_math_with_negative_scale(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Regression test: math backend SDPA with a negative scale must match a
        # reference built from matmul, division by a small negative number, softmax and matmul.
        # https://github.com/pytorch/pytorch/issues/105190.
        def ref(x):
            # [FUNCTION DOC] HELPER: Reference attention for the negative-scale test: scores
            # from self matmul divided by a small negative constant, softmax over the last
            # dimension, then multiplied with the input.
            v1 = torch.matmul(x, x.transpose(-1, -2))
            v2 = v1 / -0.0001
            v3 = v2.softmax(dim=-1)
            v4 = torch.matmul(v3, x)
            return v4

        x = torch.randn(1, 3, 64, 64, device=device)
        ref_result = ref(x)
        with sdpa_kernel(backends=[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)

    def test_scaled_dot_product_attention_fp16_overflow(self, device):
        # [FUNCTION DOC] Regression test: SDPA on half precision inputs filled with 256 must not
        # return NaN.
        # Regression test for https://github.com/pytorch/pytorch/issues/160841
        x = torch.full((1, 32, 23, 80), 256.0, dtype=torch.half, device=device)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        self.assertFalse(y.isnan().any().item())

    @parametrize(
        "q_shape,k_shape,v_shape",
        [
            # B=0: no batches, output is empty
            ((0, 4, 10, 64), (0, 4, 10, 64), (0, 4, 10, 32)),
            # H=0: no heads, output is empty
            ((2, 0, 10, 64), (2, 0, 10, 64), (2, 0, 10, 32)),
            # L_q=0: no query positions, output is empty
            ((2, 4, 0, 64), (2, 4, 10, 64), (2, 4, 10, 32)),
            # L_k=0: softmax over empty set, undefined
            ((2, 4, 10, 64), (2, 4, 0, 64), (2, 4, 0, 32)),
            # head_dim_v=0: output last dim is 0
            ((2, 4, 10, 64), (2, 4, 10, 64), (2, 4, 10, 0)),
            # all zero
            ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ],
    )
    def test_sdpa_zero_output(self, device, q_shape, k_shape, v_shape):
        # [FUNCTION DOC] Checks SDPA for zero-sized shapes (zero batch, heads, query length, key
        # length, value head dim, or all zero): output shape is the query shape with the last
        # dim replaced by the value head dim, values are all zeros, and backward produces
        # gradients with the same shapes as the inputs.
        q = torch.randn(*q_shape, device=device, requires_grad=True)
        k = torch.randn(*k_shape, device=device, requires_grad=True)
        v = torch.randn(*v_shape, device=device, requires_grad=True)
        out = F.scaled_dot_product_attention(q, k, v)
        expected_shape = list(q_shape)
        expected_shape[-1] = v_shape[-1]
        self.assertEqual(out.shape, tuple(expected_shape))
        self.assertEqual(out, torch.zeros(expected_shape, device=device))
        out.sum().backward()
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)

    @parametrize("value_dim", [1, 8])
    @parametrize("scale", [None, 1.0])
    def test_sdpa_zero_qk_head_dim_matches_eager_attention(self, device, value_dim, scale):
        # [FUNCTION DOC] Checks SDPA with a zero-sized query/key head dim (default or unit
        # scale, no dropout) matches an eager matmul-softmax-matmul reference for the output and
        # for the query, key and value gradients.
        torch.manual_seed(0)
        query = torch.empty((1, 16, 16, 0), device=device, requires_grad=True)
        key = torch.empty((1, 16, 16, 0), device=device, requires_grad=True)
        value = torch.randn((1, 16, 16, value_dim), device=device, requires_grad=True)
        grad_output = torch.randn((1, 16, 16, value_dim), device=device)

        actual = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=scale)
        actual.backward(grad_output)
        actual_query_grad = query.grad.detach().clone()
        actual_key_grad = key.grad.detach().clone()
        actual_value_grad = value.grad.detach().clone()

        query_ref = query.detach().clone().requires_grad_()
        key_ref = key.detach().clone().requires_grad_()
        value_ref = value.detach().clone().requires_grad_()
        scores = torch.matmul(query_ref, key_ref.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_ref.dtype)
        expected = torch.matmul(probs, value_ref)
        expected.backward(grad_output)

        self.assertEqual(actual, expected)
        self.assertEqual(actual_query_grad, query_ref.grad)
        self.assertEqual(actual_key_grad, key_ref.grad)
        self.assertEqual(actual_value_grad, value_ref.grad)

    def test_sdpa_output_shape_uses_value_head_dim(self, device):
        # [FUNCTION DOC] Regression test: for 3D, 4D and 5D inputs where the value head dim
        # differs from the query/key head dim, the output shape must use the value head dim.
        # Regression test for https://github.com/pytorch/pytorch/issues/176767
        test_cases = [
            ((1, 16, 32), (1, 16, 32), (1, 16, 24)),
            ((1, 1, 16, 32), (1, 1, 16, 32), (1, 1, 16, 24)),
            ((2, 3, 1, 16, 32), (2, 3, 1, 16, 32), (2, 3, 1, 16, 24)),
        ]

        for q_shape, k_shape, v_shape in test_cases:
            q = torch.randn(*q_shape, device=device, dtype=torch.float32)
            k = torch.randn(*k_shape, device=device, dtype=torch.float32)
            v = torch.randn(*v_shape, device=device, dtype=torch.float32)

            actual = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            expected_shape = list(q_shape)
            expected_shape[-1] = v_shape[-1]
            self.assertEqual(actual.shape, torch.Size(expected_shape))

    def test_sdpa_export_unbacked_attn_mask(self, device):
        # [FUNCTION DOC] Checks that torch.export of a small model succeeds when SDPA receives a
        # boolean attention mask whose batch dimension is unbacked (data-dependent via boolean
        # indexing).
        """SDPA backend selection should not crash on unbacked symbolic mask shapes."""

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Small module with three linear projections (432 to 32) that
            # filters out all-zero batch entries by boolean indexing (giving an unbacked batch
            # size), projects to per-head query/key/value and calls SDPA with an all-True mask
            # of unbacked batch shape.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets 4 heads, head dim 8, hidden size 32 and creates
                # three linear layers from 432 to 32.
                super().__init__()
                self.num_heads = 4
                self.head_dim = 8
                self.hidden_size = self.num_heads * self.head_dim
                # 3 * 12 * 12 = 432
                self.proj_q = nn.Linear(432, self.hidden_size)
                self.proj_k = nn.Linear(432, self.hidden_size)
                self.proj_v = nn.Linear(432, self.hidden_size)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Keeps batch entries whose sum is non-zero, flattens
                # them, projects to query, key and value shaped as batch, heads, 1, head dim,
                # builds a ones boolean mask, and returns the SDPA output.
                keep = x.sum(dim=(-1, -2, -3)) != 0  # x: [B, 3, 12, 12]
                x = x[keep]  # [u0, 3, 12, 12].
                flat = x.flatten(1)  # [u0, 432]
                q = self.proj_q(flat).view(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.proj_k(flat).view(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.proj_v(flat).view(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
                attn_mask = torch.ones(
                    (x.shape[0], 1, 1, 1), dtype=torch.bool, device=x.device
                )
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )

        model = Model().eval().to(device)
        x = torch.randn(4, 3, 12, 12, device=device)
        x[0].zero_()

        # Should not crash during export with unbacked symbolic mask batch dim
        torch.export.export(model, args=(x,))

class TestSDPACpuOnly(NNTestCase):
    # [CLASS DOC] Test class for CPU-only SDPA behavior: flash attention backend selection and
    # numerics on CPU versus the math backend (masks, GQA/MQA, infinities, NaN propagation,
    # fully masked rows).
    """ Used to test CPU only functionality of scaled_dot_product_attention """

    @parametrize("type", ["dense", "nested"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_cpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # [FUNCTION DOC] Checks the backend selected on CPU: math for nested inputs or when
        # dropout is positive (unsupported dtypes also map to math), otherwise flash attention.
        # Raises AssertionError on mismatch.
        # Test that cpu and nestedtensor cpu return MATH backend
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if type == "nested" \
                or dropout > 0.0 \
                or dtype not in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            if torch._fused_sdp_choice(q, k, v, dropout_p=dropout) != SDPBackend.MATH.value:
                raise AssertionError("expected MATH backend")
        else:
            if torch._fused_sdp_choice(q, k, v, dropout_p=dropout) != SDPBackend.FLASH_ATTENTION.value:
                raise AssertionError("expected FLASH_ATTENTION backend")

    def test_fused_inputs_dim_3_cpu(self, device):
        # [FUNCTION DOC] Checks that unbatched 3D inputs on CPU with no mask, a 2D mask or a
        # batched 3D mask select the flash backend and give results equal to batched 4D inputs
        # with leading dimension added.
        size = (2, 3, 8)
        q = torch.randn(size, device=device)
        k = torch.randn(size, device=device)
        v = torch.randn(size, device=device)
        masks = (
            None,
            torch.randn(3, 3, device=device),
            torch.randn(2, 3, 3, device=device),
        )

        for attn_mask in masks:
            normalized_mask = attn_mask
            while normalized_mask is not None and normalized_mask.dim() < 4:
                normalized_mask = normalized_mask.unsqueeze(0)
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                self.assertEqual(
                    torch._fused_sdp_choice(q, k, v, attn_mask),
                    SDPBackend.FLASH_ATTENTION.value,
                )
                expected = F.scaled_dot_product_attention(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    normalized_mask,
                ).squeeze(0)
                actual = F.scaled_dot_product_attention(q, k, v, attn_mask)

            self.assertEqual(actual, expected)

    def _generate_fixed_qkv_helper(
        self,
        device,
        dtype,
        batch_size,
        q_n_head,
        kv_n_head,
        q_seq_len,
        kv_seq_len,
        head_dim
    ):
        # [FUNCTION DOC] HELPER: Seeds the generator with 777 and returns query, key, value
        # without grad as dense random tensors transposed so the layout is batch, seq, heads,
        # dim views. Query uses q_n_head heads and q_seq_len; key and value use kv_n_head and
        # kv_seq_len.
        torch.manual_seed(777)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, q_n_head, q_seq_len, head_dim)
        kv_shape = SdpaShape(batch_size, kv_n_head, kv_seq_len, head_dim)
        q = make_tensor(q_shape).transpose(1, 2)
        k = make_tensor(kv_shape).transpose(1, 2)
        v = make_tensor(kv_shape).transpose(1, 2)
        return q, k, v

    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
    @parametrize("batch_size", [2, 12])
    @parametrize("q_seq_len", [11, 514, 1030])
    @parametrize("kv_seq_len", [17, 514])
    @parametrize("n_head", [1, 3])
    @parametrize("head_dim", [8])
    @parametrize("mask_dim", [2, 4])
    @parametrize("bool_mask", [False, True])
    @parametrize("train", [True, False])
    @parametrize("casual", [True, False])
    @parametrize("set_attn_mask", [True, False])
    def test_scaled_dot_product_fused_attention_mask_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_seq_len,
        kv_seq_len,
        n_head,
        head_dim,
        mask_dim,
        bool_mask,
        train,
        casual,
        set_attn_mask,
    ):
        # [FUNCTION DOC] Compares CPU flash attention to the math backend over many parameter
        # combinations (dtype, batch, sequence lengths, heads, 2D or 4D mask shapes with
        # singleton broadcast dims, boolean or float mask, causal flag, train or eval). Outputs,
        # and gradients of the sum when training, must match within dtype specific tolerances
        # and contain no NaN.
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        tol_grad = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol_grad = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol_grad = Tolerances(1e-1, 1e-1)
        if dtype is torch.float32:
            tol_grad = Tolerances(1.25e-5, 5.25e-6)
        for mask_shape in itertools.product(
            [q_seq_len, 1], [kv_seq_len, 1]
        ) if mask_dim == 2 else itertools.product(
            [batch_size, 1], [n_head, 1], [q_seq_len, 1], [kv_seq_len, 1]
        ):
            q, k, v = self._generate_fixed_qkv_helper(
                device, dtype, batch_size, n_head, n_head, q_seq_len, kv_seq_len, head_dim)
            q2, k2, v2 = self._generate_fixed_qkv_helper(
                device, dtype, batch_size, n_head, n_head, q_seq_len, kv_seq_len, head_dim)
            if train:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)
                q2.requires_grad_(True)
                k2.requires_grad_(True)
                v2.requires_grad_(True)

            if set_attn_mask and not casual:
                if bool_mask:
                    attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
                else:
                    attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)
            else:
                attn_mask = None

            with sdpa_kernel(backends=[fused_kernel]):
                actual = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=casual)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                math_ref = torch.nn.functional.scaled_dot_product_attention(
                    q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=casual)

            if dtype in [torch.bfloat16, torch.float16]:
                math_ref = math_ref.to(dtype)

            self.assertFalse(torch.isnan(math_ref).any())
            self.assertFalse(torch.isnan(actual).any())

            self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)

            if train:
                actual.sum().backward()
                math_ref.sum().backward()

                grad_q_actual, grad_k_actual, grad_v_actual = q.grad, k.grad, v.grad
                grad_q_ref, grad_k_ref, grad_v_ref = q2.grad, k2.grad, v2.grad

                self.assertFalse(grad_q_actual is None)
                self.assertFalse(grad_k_actual is None)
                self.assertFalse(grad_v_actual is None)
                self.assertEqual(grad_q_actual, grad_q_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)
                self.assertEqual(grad_k_actual, grad_k_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)
                self.assertEqual(grad_v_actual, grad_v_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)

    def test_mqa_singleton_head_broadcast(self, device):
        # [FUNCTION DOC] Checks that single key/value head inputs against multiple query heads
        # (multi-query attention) give the same result on the flash backend as on the math
        # backend without enabling GQA.
        """Singleton KV heads should select CPU flash attention without GQA."""
        query = torch.randn(2, 8, 16, 32, device=device)
        key = torch.randn(2, 1, 32, 32, device=device)
        value = torch.randn(2, 1, 32, 32, device=device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = scaled_dot_product_attention(query, key, value)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            actual = scaled_dot_product_attention(query, key, value)

        self.assertEqual(actual, expected)

    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
    @parametrize("n_heads", [[65, 5], [16, 4], [27, 1], [5, 1]])
    @parametrize("train", [False, True])
    def test_scaled_dot_product_fused_attention_gqa_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        n_heads,
        train,
    ):
        # [FUNCTION DOC] Compares CPU flash attention with the GQA flag enabled against the math
        # backend for several query/key-value head counts, a random float mask, in train or eval
        # mode; outputs and (if training) gradients must match within dtype dependent
        # tolerances.
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        tol_grad = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol_grad = Tolerances(1e-1, 1e-1)
        if dtype is torch.float16:
            tol_grad = Tolerances(1e-1, 1e-1)
        q_n_head, kv_n_head = n_heads
        batch_size, q_seq_len, kv_seq_len, head_dim = 12, 17, 100, 8

        q, k, v = self._generate_fixed_qkv_helper(
            device, dtype, batch_size, q_n_head, kv_n_head, q_seq_len, kv_seq_len, head_dim)
        q2, k2, v2 = self._generate_fixed_qkv_helper(
            device, dtype, batch_size, q_n_head, kv_n_head, q_seq_len, kv_seq_len, head_dim)
        if train:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            q2.requires_grad_(True)
            k2.requires_grad_(True)
            v2.requires_grad_(True)

        mask_shape = [batch_size, q_n_head, q_seq_len, kv_seq_len]
        attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, enable_gqa=True)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, enable_gqa=True)

        if dtype in [torch.bfloat16, torch.float16]:
            math_ref = math_ref.to(dtype)

        self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)

        if train:
            actual.sum().backward()
            math_ref.sum().backward()

            grad_q_actual, grad_k_actual, grad_v_actual = q.grad, k.grad, v.grad
            grad_q_ref, grad_k_ref, grad_v_ref = q2.grad, k2.grad, v2.grad

            self.assertFalse(grad_q_actual is None)
            self.assertFalse(grad_k_actual is None)
            self.assertFalse(grad_v_actual is None)
            self.assertEqual(grad_q_actual, grad_q_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)
            self.assertEqual(grad_k_actual, grad_k_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)
            self.assertEqual(grad_v_actual, grad_v_ref, atol=tol_grad.atol, rtol=tol_grad.rtol)

    def test_sdpa_with_inf(self, device):
        # [FUNCTION DOC] Regression test: on CPU, SDPA with a float mask that contains negative
        # infinity entries (band structure) must give equal results on the math and flash
        # backends.
        # https://github.com/pytorch/pytorch/issues/127055.
        full = torch.full((600, 600), float("-inf"), device=device)
        mask = torch.triu(full, diagonal=1) + torch.tril(full, diagonal=-10)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, requires_grad=False)
        input_shape = SdpaShape(1, 600, 2, 8)
        q = make_tensor(input_shape)
        k = make_tensor(input_shape)
        v = make_tensor(input_shape)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(math_ref, actual)

    def test_sdpa_backward_with_gradient(self, device):
        # [FUNCTION DOC] Regression test: forward and backward through SDPA with a seeded random
        # upstream gradient must give equal outputs and query, key and value gradients on the
        # math and flash backends (CPU).
        # https://github.com/pytorch/pytorch/issues/133671.
        def sdpa_helper():
            # [FUNCTION DOC] HELPER: Seeds the generator, creates random query/key/value of
            # shape 2x2x49x32, runs SDPA with no mask or dropout, backpropagates a random
            # gradient retaining the graph, and returns the output and the three input
            # gradients.
            torch.manual_seed(777)
            query = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            key = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            value = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            res = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, None, 0.0, False
            )
            res_grad = (
                torch.empty_like(res, device=device)
                .uniform_(-1, 1)
            )
            res.backward(res_grad, retain_graph=True)
            return res, query.grad, key.grad, value.grad
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            res_ref, query_grad_ref, key_grad_ref, value_grad_ref = sdpa_helper()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            res_actual, query_grad_actual, key_grad_actual, value_grad_actual = sdpa_helper()
        self.assertEqual(res_ref, res_actual)
        self.assertEqual(query_grad_ref, query_grad_actual)
        self.assertEqual(key_grad_ref, key_grad_actual)
        self.assertEqual(value_grad_ref, value_grad_actual)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("backend", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])
    @parametrize("seq_len", [32, 64, 128])
    @parametrize("head_dim", [16, 32])
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_fully_masked_out_rows(self, backend, device, seq_len, head_dim, dtype):
        # [FUNCTION DOC] Checks that backends return zeros (with zero gradient for the query)
        # for fully masked query rows, and that outputs and gradients match the math backend
        # with no NaN. Returns early without checking for flash attention on CUDA devices and
        # for efficient attention on CPU.
        # KNOWN ISSUE: the early exits only construct a skip decorator and do not raise, so
        # those cases silently pass instead of being reported as skipped.
        def attention_inputs(seq_len, head_dim, device, dtype, mask_every_n_rows=4):
            # [FUNCTION DOC] HELPER: Builds random query, key, value (requiring grad) and a
            # boolean mask in which every fourth row is fully masked and the remaining entries
            # follow a fixed modulo-five pattern. Returns them.
            query = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)
            key = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)
            value = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)

            # Create a mask with deterministic row masking
            mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)

            # Mask every nth row
            mask[0, 0, ::mask_every_n_rows, :] = False

            # Create a fixed pattern for element-wise masking
            element_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            element_mask[torch.arange(seq_len)[:, None] % 5 == torch.arange(seq_len) % 5] = True

            # Combine row masking and element-wise masking
            mask = mask & element_mask.unsqueeze(0).unsqueeze(0)

            return query, key, value, mask

        def compute_output_and_grads(query, key, value, mask, backend):
            # [FUNCTION DOC] HELPER: Runs SDPA under the given backend with the mask, sums the
            # output as loss, computes gradients of the loss for query, key and value, and
            # returns the output and gradients.
            with sdpa_kernel(backend):
                masked_out = scaled_dot_product_attention(query, key, value, attn_mask=mask)
                loss = masked_out.sum()
            grads = torch.autograd.grad(loss, [query, key, value])
            return masked_out, grads

        if backend == SDPBackend.FLASH_ATTENTION and "cuda" in str(device):
            unittest.skip("FlashAttention does not support masks on cuda")
            return
        if backend == SDPBackend.EFFICIENT_ATTENTION and "cpu" in str(device):
            unittest.skip("EfficientAttention does not support masks on cpu")
            return
        query, key, value, mask = attention_inputs(seq_len, head_dim, device, dtype)

        # Compute results for the tested backend
        backend_out, backend_grads = compute_output_and_grads(query, key, value, mask, backend)

        # Compute results for the Math backend
        math_out, math_grads = compute_output_and_grads(query, key, value, mask, SDPBackend.MATH)

        # Compare outputs
        torch.testing.assert_close(backend_out, math_out, atol=5e-3, rtol=0)
        self.assertFalse(backend_out.isnan().any())
        self.assertFalse(math_out.isnan().any())
        # Compare gradients
        for bg, mg in zip(backend_grads, math_grads):
            torch.testing.assert_close(bg, mg, atol=3e-3, rtol=0)
            self.assertFalse(bg.isnan().any())
            self.assertFalse(mg.isnan().any())

        # Check if masked rows are zero in output
        mask_sum = mask.sum(dim=-1, keepdim=True)
        masked_rows = (mask_sum == 0).expand_as(backend_out)
        self.assertTrue((mask_sum == 0).sum() > 0, "No fully masked out rows found")
        if not torch.all(backend_out[masked_rows] == 0):
            raise AssertionError(f"Non-zero values in fully masked rows for {backend=}")

        # Check if gradients for masked rows are zero
        grad_query = backend_grads[0]
        if not torch.all(grad_query[masked_rows] == 0):
            raise AssertionError(f"Non-zero gradients in fully masked rows for {backend=}")

    @parametrize("dtype", [torch.float32, torch.float16])
    @parametrize("fill_val", [float("inf")])
    def test_non_masked_rows_nan_props(self, device, dtype, fill_val):
        # [FUNCTION DOC] Checks that an infinite value in one query element makes SDPA output
        # contain NaN and the query gradient contain NaN after a backward pass of the output
        # sum.
        # KNOWN ISSUE: comments in the test body call the value a NaN, but the parametrization
        # uses infinity.
        query = torch.randn(1, 2, 4, 16, device=device, dtype=dtype)
        # a single NaN in the query input
        query[0, 1, 2, 3] = fill_val
        query = query.detach().requires_grad_(True)
        key = torch.randn(1, 2, 4, 16, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(1, 2, 4, 16, device=device, dtype=dtype, requires_grad=True)

        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        self.assertTrue(torch.isnan(out).any())
        out.sum().backward()
        self.assertTrue(torch.isnan(query.grad).any())

    @parametrize("dtype", [torch.float32, torch.float16])
    def test_cpu_flash_attn_nan_propagation(self, dtype):
        # [FUNCTION DOC] Checks that CPU flash attention with an all-NaN query returns an
        # all-NaN output.
        # Setup tensors
        query = torch.full((1, 1, 16, 16), torch.nan, dtype=dtype)
        key = torch.randn(1, 1, 16, 16, dtype=dtype)
        value = torch.randn(1, 1, 16, 16, dtype=dtype)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )

            # Check that output contains NaN
            self.assertTrue(torch.isnan(out).all())

    @parametrize("kernel", [SDPBackend.MATH])
    def test_scaled_dot_product_attention_math_with_negative_scale(self, device, kernel: SDPBackend):
        # [FUNCTION DOC] Same negative scale regression check as in the generic class: math
        # backend SDPA with a negative scale must match a matmul-softmax reference.
        # https://github.com/pytorch/pytorch/issues/105190.
        def ref(x):
            # [FUNCTION DOC] HELPER: Reference attention for the negative-scale test: self
            # matmul divided by a small negative constant, softmax, and multiplication with the
            # input.
            v1 = torch.matmul(x, x.transpose(-1, -2))
            v2 = v1 / -0.0001
            v3 = v2.softmax(dim=-1)
            v4 = torch.matmul(v3, x)
            return v4

        x = torch.randn(1, 3, 64, 64, device=device)
        ref_result = ref(x)
        with sdpa_kernel(backends=[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)

class TestSDPACudaOnly(NNTestCase):
    # [CLASS DOC] Tests specific to CUDA behavior of SDPA (flash, memory-efficient, cuDNN and
    # math backends), with availability depending on GPU architecture and build flags (see the
    # platform support flags at the top of the file). Enables CUDA memory leak and
    # non-default-stream checks.
    """ Used to test CUDA only functionality of scaled_dot_product_attention
    Quarks:
        There is some trickiness with this function. Its runtime behavior
        is dependent on the CUDA architecture you are testing it on. See
        `PLATFORM_SUPPORTS_FUSED_ATTENTION` at the top of the file.
        Summary:
            Math: always supported
            FlashAttention: Supported on sm80 or newer hardware
            MemEfficientAttention: Supported on sm50 or newer hardware
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_decode_disabled_on_affected_blackwell(self, device):
        # [FUNCTION DOC] Checks that cuDNN decode attention (single query position) is disabled
        # on affected Blackwell GPUs (SM 10.x and 11.x with cuDNN 9.19 up to 9.25.0 except
        # 9.24.1): the can-use check must return False and the cuDNN op must raise RuntimeError
        # saying decode is disabled. Skipped elsewhere.
        # See #193893 and #194927 for reasoning
        # TODO: Remove this test when fixed and disable is no longer needed
        cudnn_version = torch.backends.cudnn.version() or 0
        device_capability = torch.cuda.get_device_capability()
        affected_arch = device_capability[0] in (10, 11)
        # cuDNN versions 9.19-9.25.0 (except 9.24.1) on SM 10.x and 11.x are disabled
        # This check also allows possible future 9.24 patch versions without a rewrite
        if not (affected_arch and (91900 <= cudnn_version <= 92500) and not (92400 < cudnn_version < 92500)):
            self.skipTest("Requires cuDNN 9.19-9.25.0 (except 9.24.1) on SM 10.x or 11.x")

        query = torch.randn(2, 8, 1, 64, device=device, dtype=torch.float16)
        key = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
        value = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
        params = SDPAParams(query, key, value, None, 0.0, False, False)

        self.assertFalse(torch.backends.cuda.can_use_cudnn_attention(params))
        with self.assertRaisesRegex(RuntimeError, "cuDNN SDPA decode is disabled"):
            torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                query, key, value, None, False, 0.0, False, False
            )
        with self.assertRaisesRegex(RuntimeError, "cuDNN SDPA decode is disabled"):
            torch.ops.aten._cudnn_attention_forward.default(
                query, key, value, None, None, None, 1, 128, False, 0.0, False, False
            )

    # TODO USED FOR TESTING THE SCORES, e.g. testing ALIBI we don't need this now
    def normalize_flash_attn_S(
        self,
        attn_unnorm,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        is_dropout=False,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        scale=None,
    ):
        # [FUNCTION DOC] HELPER: Recomputes normalized attention probabilities from the
        # unnormalized flash attention score matrix. Transposes inputs, builds scaled scores in
        # float32, applies key padding and local window masks and optional bias, computes the
        # log-sum-exp over key blocks of the kernel block size and the running block maxima,
        # rescales each block of attn_unnorm by the exponential of its maximum minus the
        # log-sum-exp (infinite log-sum-exp of fully masked rows is replaced by positive
        # infinity to avoid NaN), zeroes padded query rows, and returns the result in the dtype
        # of attn_unnorm.
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k, v: (batch_size, seqlen_k, nheads, head_dim)
            key_padding_mask: (batch_size, seqlen_q)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        Output:
            softmax_lse: (batch_size, nheads, seqlen_q)
            softmax_max: (batch_size, nheads, seqlen_q)
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if causal:
            window_size = (window_size[0], 0)
        q, k, v = q.float(), k.float(), v.float()
        _, seqlen_q, _, head_dim = q.shape
        seqlen_k = k.shape[1]
        b = q.shape[0]
        from torch.nn.attention.bias import _calculate_scale
        scale = _calculate_scale(head_dim, scale)
        scores = torch.matmul(q.transpose(1, 2) * scale, k.permute(0, 2, 3, 1))
        if key_padding_mask is not None:
            scores.masked_fill_(~key_padding_mask.view(b, 1, 1, -1), float("-inf"))
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias.to(dtype=scores.dtype)
        block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
        scores_block = scores.split(block_size_n, dim=-1)
        lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
        lse = torch.logsumexp(lse_block, dim=-1)
        # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
        # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
        lse[lse == float("-inf")] = float("inf")
        scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
        cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
        attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
        attn_norm = torch.cat(
            [
                a * (torch.exp(m - lse)).unsqueeze(-1)
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )
        if query_padding_mask is not None:
            attn_norm.masked_fill_(~query_padding_mask.view(b, 1, -1, 1), 0.0)
            # attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
        return attn_norm.to(dtype=attn_unnorm.dtype)

    def construct_local_mask(self, seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, device):
        # [FUNCTION DOC] HELPER: Builds a boolean mask (True means masked out) for sliding
        # window attention from the sequence lengths and window sizes left and right, taking
        # per-batch true lengths from the optional padding masks. A negative left window means
        # unbounded to the left. Returns the mask tensor.
        # row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).view(-1, 1)
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else key_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else query_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )

    def convert_flash_attn_S_to_softmax(
        self,
        S,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
    ):
        # [FUNCTION DOC] HELPER: Converts the score matrix S returned by the flash kernel
        # (padded to rounded lengths and stored in kernel specific layout) into the logical
        # softmax layout. Returns S unchanged on ROCm. Otherwise masks out entries outside the
        # local window or the padding masks with zeros, pads or slices to the original sequence
        # lengths, and returns the top-left block of size seqlen_q by seqlen_k.
        """FlashAttention stores the S matrix in a different way.
        Arguments:
            S: (batch_size, nheads, seqlen_q, seqlen_k)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
        """
        if TEST_WITH_ROCM:
            return S
        b = S.shape[0]

        if causal:
            window_size = (window_size[0], 0)
        seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
        S_converted = S
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                S.device,
            )
            local_mask = F.pad(
                local_mask,
                (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
                value=True,
            )
            S_converted = S_converted.masked_fill(local_mask, 0.0)

        # Need to zero out things not in attention_mask in case S was initialized with random values
        # and some of those values aren't overwritten.
        seqlen_q_og = (
            query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
        )
        if query_padding_mask is not None:
            query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
            # S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
            S_converted = S_converted.masked_fill(~query_padding_mask.view(b, 1, -1, 1), 0.0)
        seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
            S_converted = S_converted.masked_fill(~key_padding_mask.view(b, 1, 1, -1), 0.0)
            # S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
        return S_converted[:, :, :seqlen_q, :seqlen_k]

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_different_dk_dv(self, device):
        # [FUNCTION DOC] cuDNN attention test: bfloat16 inputs with query/key head dim 128 and
        # value head dim 64 must give the same result as the math backend on float32 copies with
        # atol 1e-3 and rtol 1e-2.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Memory efficient attention is not supported on this system")
    @parametrize("enable_gqa", [False, True])
    def test_mqa_singleton_head_broadcast(self, device, enable_gqa):
        # [FUNCTION DOC] Checks that a single key/value head broadcast against multiple query
        # heads gives, on the efficient backend, forward results and gradients equal to the math
        # backend, with or without the GQA flag (float32). Skipped on ROCm and when memory
        # efficient attention is unsupported.
        """Singleton KV heads should use fused attention with either GQA mode."""
        make_tensor = partial(
            torch.randn, device=device, dtype=torch.float32, requires_grad=True
        )
        query = make_tensor(2, 8, 16, 32)
        key = make_tensor(2, 1, 32, 32)
        value = make_tensor(2, 1, 32, 32)
        grad_out = torch.randn(2, 8, 16, 32, device=device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = scaled_dot_product_attention(
                query, key, value, enable_gqa=enable_gqa
            )
            expected_grads = torch.autograd.grad(
                expected, (query, key, value), grad_out
            )

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = scaled_dot_product_attention(
                query, key, value, enable_gqa=enable_gqa
            )
            actual_grads = torch.autograd.grad(
                actual, (query, key, value), grad_out
            )

        self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertEqual(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Memory efficient attention is not supported on this system")
    @parametrize("dtype", [torch.float32, torch.float16])
    @parametrize(
        "num_query_heads,num_kv_heads,is_causal,use_attn_mask,q_layout,k_layout,v_layout",
        [
            (8, 2, False, True, "padded", "sequence_sliced", "transposed"),
            (8, 4, True, False, "transposed", "offset", "head_sliced"),
            (6, 2, False, False, "sequence_sliced", "batch_expanded", "padded"),
        ],
    )
    def test_mem_efficient_attention_gqa(
        self,
        device,
        dtype,
        num_query_heads,
        num_kv_heads,
        is_causal,
        use_attn_mask,
        q_layout,
        k_layout,
        v_layout,
    ):
        # [FUNCTION DOC] Checks efficient-attention grouped query attention for float32 and
        # float16 with several head counts, causal or masked configurations, and non-contiguous
        # layouts of query, key, value and mask built with the strided input helper: output and
        # gradients (including the mask gradient if present) must match the math backend within
        # dtype dependent tolerances.
        """Map query heads to KV groups while keeping masks query-head indexed."""
        query = make_strided_sdpa_input(
            (2, num_query_heads, 16, 32), q_layout, device, dtype
        )
        key = make_strided_sdpa_input(
            (2, num_kv_heads, 32, 32), k_layout, device, dtype
        )
        value = make_strided_sdpa_input(
            (2, num_kv_heads, 32, 16), v_layout, device, dtype
        )
        attn_mask = (
            make_strided_sdpa_input(
                (2, num_query_heads, 16, 32), "offset", device, dtype
            )
            if use_attn_mask
            else None
        )
        grad_out = torch.randn(
            2, num_query_heads, 16, 16, device=device, dtype=dtype
        )
        inputs = (query, key, value, attn_mask) if attn_mask is not None else (query, key, value)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=True,
            )
            expected_grads = torch.autograd.grad(expected, inputs, grad_out)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=True,
            )
            actual_grads = torch.autograd.grad(actual, inputs, grad_out)

        forward_tolerance = 1e-2 if dtype == torch.float16 else 1e-5
        grad_tolerance = 1e-1 if dtype == torch.float16 else 1e-4
        self.assertEqual(
            actual,
            expected,
            atol=forward_tolerance,
            rtol=forward_tolerance,
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertEqual(
                actual_grad,
                expected_grad,
                atol=grad_tolerance,
                rtol=grad_tolerance,
            )

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Memory efficient attention is not supported on this system")
    def test_mem_efficient_attention_gqa_dropout(self, device):
        # [FUNCTION DOC] Checks GQA with dropout and mask on the efficient backend: compares
        # grouped key/value against repeated (expanded) key/value under the same CUDA seed
        # (dropout 0.2, scale 0.37), and the key and value gradients against the group-summed
        # gradients of the expanded version.
        """GQA should preserve query-head dropout and mask semantics."""
        batch, num_query_heads, num_kv_heads = 2, 6, 2
        group_size = num_query_heads // num_kv_heads
        query = make_strided_sdpa_input(
            (batch, num_query_heads, 17, 32), "sequence_sliced", device, torch.float32
        )
        key = make_strided_sdpa_input(
            (batch, num_kv_heads, 33, 32), "offset", device, torch.float32
        )
        value = make_strided_sdpa_input(
            (batch, num_kv_heads, 33, 24), "transposed", device, torch.float32
        )
        attn_mask = make_strided_sdpa_input(
            (batch, num_query_heads, 17, 33), "padded", device, torch.float32
        )
        grad_out = torch.randn(
            batch, num_query_heads, 17, 24, device=device, dtype=torch.float32
        )

        query_expanded = query.detach().clone().requires_grad_()
        key_expanded = (
            key.detach().repeat_interleave(group_size, dim=1).requires_grad_()
        )
        value_expanded = (
            value.detach().repeat_interleave(group_size, dim=1).requires_grad_()
        )
        attn_mask_expanded = attn_mask.detach().clone().requires_grad_()

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            torch.cuda.manual_seed(1234)
            actual = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p=0.2,
                scale=0.37,
                enable_gqa=True,
            )
            torch.cuda.manual_seed(1234)
            expected = scaled_dot_product_attention(
                query_expanded,
                key_expanded,
                value_expanded,
                attn_mask_expanded,
                dropout_p=0.2,
                scale=0.37,
            )

        actual_grads = torch.autograd.grad(
            actual, (query, key, value, attn_mask), grad_out
        )
        expanded_grads = torch.autograd.grad(
            expected,
            (query_expanded, key_expanded, value_expanded, attn_mask_expanded),
            grad_out,
        )
        expected_grads = (
            expanded_grads[0],
            expanded_grads[1]
            .unflatten(1, (num_kv_heads, group_size))
            .sum(2),
            expanded_grads[2]
            .unflatten(1, (num_kv_heads, group_size))
            .sum(2),
            expanded_grads[3],
        )

        self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertEqual(actual_grad, expected_grad, atol=4e-4, rtol=3e-4)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Memory efficient attention is not supported on this system")
    def test_mem_efficient_attention_gqa_split_key(self, device):
        # [FUNCTION DOC] Checks the split-key backward of the low level efficient attention ops
        # with grouped key/value heads (six query and two key/value heads, key length 4096): the
        # forward output and the query, key and value gradients must match the math backend
        # results.
        """Split-key backward should reduce per-query-head KV gradients."""
        batch, num_query_heads, num_kv_heads = 1, 6, 2
        query = torch.randn(
            batch, num_query_heads, 17, 64, device=device, requires_grad=True
        )
        key = torch.randn(
            batch, num_kv_heads, 4096, 64, device=device, requires_grad=True
        )
        value = torch.randn(
            batch, num_kv_heads, 4096, 40, device=device, requires_grad=True
        )
        grad_out = torch.randn(batch, num_query_heads, 17, 40, device=device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = scaled_dot_product_attention(
                query, key, value, enable_gqa=True
            )
        expected_grads = torch.autograd.grad(expected, (query, key, value), grad_out)

        query_bmhd = query.detach().transpose(1, 2)
        key_bmhd = key.detach().transpose(1, 2)
        value_bmhd = value.detach().transpose(1, 2)
        output, logsumexp, seed, offset, max_q, max_k = (
            torch.ops.aten._efficient_attention_forward.default(
                query_bmhd,
                key_bmhd,
                value_bmhd,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0,
                True,
            )
        )
        grad_query, grad_key, grad_value, _ = (
            torch.ops.aten._efficient_attention_backward.default(
                grad_out.transpose(1, 2),
                query_bmhd,
                key_bmhd,
                value_bmhd,
                None,
                output,
                None,
                None,
                max_q,
                max_k,
                logsumexp,
                0.0,
                seed,
                offset,
                0,
                False,
                num_splits_key=4,
            )
        )
        actual_grads = (
            grad_query.transpose(1, 2),
            grad_key.transpose(1, 2),
            grad_value.transpose(1, 2),
        )

        self.assertEqual(output.transpose(1, 2), expected, atol=3e-5, rtol=2e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertEqual(actual_grad, expected_grad, atol=5e-4, rtol=4e-4)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Memory efficient attention is not supported on this system")
    def test_mem_efficient_attention_zero_heads(self, device):
        # [FUNCTION DOC] Checks that the low level efficient attention forward and backward ops
        # accept zero-length inputs: output has shape batch 2, heads 3, length 0, head dim 16,
        # the log-sum-exp has leading sizes 2 and 0, and the gradients are empty with the shapes
        # of the inputs.
        """Zero-head low-level attention should return empty outputs and gradients."""
        query = torch.empty(2, 3, 0, 8, device=device)
        key = torch.empty(2, 4, 0, 8, device=device)
        value = torch.empty(2, 4, 0, 16, device=device)
        output, logsumexp, seed, offset, max_q, max_k = (
            torch.ops.aten._efficient_attention_forward.default(
                query,
                key,
                value,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0,
                True,
            )
        )
        grad_query, grad_key, grad_value, _ = (
            torch.ops.aten._efficient_attention_backward.default(
                torch.empty_like(output),
                query,
                key,
                value,
                None,
                output,
                None,
                None,
                max_q,
                max_k,
                logsumexp,
                0.0,
                seed,
                offset,
                0,
                False,
            )
        )

        self.assertEqual(output.shape, (2, 3, 0, 16))
        self.assertEqual(logsumexp.shape[:2], (2, 0))
        for grad, source in zip(
            (grad_query, grad_key, grad_value), (query, key, value)
        ):
            self.assertEqual(grad.shape, source.shape)
            self.assertEqual(grad.numel(), 0)

    @parametrize(
        "fused_kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION],
    )
    def test_mqa_singleton_head_broadcast_native_cuda(self, device, fused_kernel):
        # [FUNCTION DOC] Checks that flash attention or cuDNN attention (whichever is
        # parametrized and supported, else the test is skipped) accept implicit multi-query
        # broadcasting of one key/value head and match the math backend on float16 inputs.
        """Native GQA backends should accept implicit MQA broadcasting."""
        if fused_kernel == SDPBackend.FLASH_ATTENTION and not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("Flash attention is not supported on this system")
        if fused_kernel == SDPBackend.CUDNN_ATTENTION and not PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            self.skipTest("cuDNN attention is not supported on this system")

        query = torch.randn(2, 8, 16, 32, device=device, dtype=torch.float16)
        key = torch.randn(2, 1, 32, 32, device=device, dtype=torch.float16)
        value = torch.randn(2, 1, 32, 32, device=device, dtype=torch.float16)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = scaled_dot_product_attention(query, key, value)
        with sdpa_kernel(backends=[fused_kernel]):
            actual = scaled_dot_product_attention(query, key, value)

        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_gqa(self, device):
        # [FUNCTION DOC] cuDNN test: with 32 query heads, 8 key heads and 4 value heads and a
        # causal mask, the cuDNN backend must raise RuntimeError (no available kernel) when GQA
        # is not enabled and must match the math backend result when enabled.
        batch = 4
        seq_len_q = 512
        seq_len_kv = 1024
        D = 128
        # Sample call to SDPA - GQ
        query = torch.rand(batch, 32, seq_len_q, D, device='cuda', dtype=torch.bfloat16)
        key = torch.rand(batch, 8, seq_len_kv, D, device='cuda', dtype=torch.bfloat16)
        # cuDNN supports h_k != h_v
        value = torch.rand(batch, 4, seq_len_kv, D, device='cuda', dtype=torch.bfloat16)
        with sdpa_kernel([SDPBackend.MATH]):
            output_math = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)

        with self.assertRaisesRegex(RuntimeError, "No available kernel."):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                output_cudnn = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=False)

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            output_cudnn = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)

        self.assertEqual(output_math, output_cudnn)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    @xfailIf(_cudnn_d256_mixed_head_dim_bprop_unsupported())
    def test_cudnn_attention_d256_heuristic(self, device):
        # [FUNCTION DOC] cuDNN test: bfloat16 with query/key head dim 256 and value head dim 64
        # under priority ordering with forward and backward. Expected failure marker is set
        # where mixed head dim backward is unsupported. When the helper says the mixed shape is
        # unsupported, the nested test must pass (result close to the math backend), otherwise a
        # RuntimeError reporting no available kernel is expected.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 256, 64
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        def test():
            # [FUNCTION DOC] HELPER: Runs the cuDNN backend with priority on shared inputs,
            # backpropagates random gradients, computes a float32 math reference, and asserts
            # the forward results are close with atol 1e-3 and rtol 1e-2.
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION], set_priority=True):
                actual = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
                actual.backward(torch.randn_like(actual))
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                math_ref = torch.nn.functional.scaled_dot_product_attention(
                    query.contiguous().to(torch.float32),
                    key.contiguous().to(torch.float32),
                    value.contiguous().to(torch.float32),
                    attn_mask=None, dropout_p=0.0, is_causal=False)
            self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

        # cuDNN d_qk=256, d_v=64 bprop is unsupported on the d256 support surface.
        if _cudnn_d256_mixed_head_dim_bprop_unsupported():
            test()
        else:
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                test()

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_dense_d256_sm90_sm10x(self, device):
        # [FUNCTION DOC] cuDNN test for head dim 256 on SM 9.0 and SM 10.x: skips if unsupported
        # by the platform or cuDNN version, selects the cuDNN backend (for SM 10.7 with old
        # cuDNN checks that cuDNN is not selected and falls back to flash/math), runs forward
        # and backward, and compares with a float32 math reference.
        cudnn_version = torch.backends.cudnn.version() or 0
        is_unsupported_sm107 = (
            torch.cuda.get_device_capability() == (10, 7)
            and cudnn_version <= 92500
        )
        if not is_unsupported_sm107 and not _cudnn_supports_d256_attention():
            self.skipTest(
                "head_dim=256 cuDNN attention requires SM90 or supported SM10.x with cuDNN > 9.22.0"
            )

        dtype = torch.bfloat16
        batch, num_heads, seq_len, head_dim = 2, 4, 128, 256
        query = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)

        backends = [SDPBackend.CUDNN_ATTENTION]
        if is_unsupported_sm107:
            backends.extend([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
        with sdpa_kernel(backends=backends, set_priority=is_unsupported_sm107):
            backend = torch._fused_sdp_choice(query, key, value)
            if is_unsupported_sm107:
                self.assertNotEqual(backend, SDPBackend.CUDNN_ATTENTION.value)
            elif backend != SDPBackend.CUDNN_ATTENTION.value:
                self.skipTest("head_dim=256 cuDNN attention requires cuDNN frontend support")
            actual = F.scaled_dot_product_attention(query, key, value)
            actual.backward(torch.randn_like(actual))

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                query.detach().to(torch.float32),
                key.detach().to(torch.float32),
                value.detach().to(torch.float32),
            )

        self.assertEqual(actual, math_ref.to(dtype), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_d192_heuristic(self, device):
        # [FUNCTION DOC] cuDNN test: bfloat16 with query/key head dim 192 and value head dim
        # 128, forward and backward under priority ordering, compared against the math backend;
        # the nested helper does the work and is called depending on the platform support of
        # head dim 192 (described by the helper above).
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 192, 128
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        def test():
            # [FUNCTION DOC] HELPER: Runs the cuDNN backend with priority on shared inputs,
            # backpropagates random gradients, computes a float32 math reference, and asserts
            # forward results are close with atol 1e-3 and rtol 1e-2.
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION], set_priority=True):
                actual = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
                actual.backward(torch.randn_like(actual))
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                math_ref = torch.nn.functional.scaled_dot_product_attention(
                    query.contiguous().to(torch.float32),
                    key.contiguous().to(torch.float32),
                    value.contiguous().to(torch.float32),
                    attn_mask=None, dropout_p=0.0, is_causal=False)
            self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

        # head_dim=192 support on SM 9.0 (Hopper) and SM 10.x (Blackwell) requires cuDNN >= 9.11.0 (91100)
        # This is a special case for DeepSeek model dimensions
        if _cudnn_supports_d192_attention():
            test()
        else:
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                test()

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_fused_attention_different_dk_dv(self, device):
        # [FUNCTION DOC] cuDNN test: query length 1, key/value length 2, query/key head dim 128,
        # value head dim 64 in bfloat16 are an unsupported cuDNN case; without forcing a backend
        # the default dispatch must not use cuDNN and the result must match a float32 math
        # reference with atol 1e-3 and rtol 1e-2.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        q_shape = SdpaShape(batch, num_heads, 1, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, 2, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, 2, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        # test that we do not dispatch to cuDNN for an unsupported case
        actual = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)


    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    @unittest.skipIf(True, "broken as of cuDNN 9.10")
    def test_cudnn_attention_fail_d128(self, device):
        # [FUNCTION DOC] Skipped unconditionally (broken as of cuDNN 9.10). Intended to check
        # that the cuDNN backend rejects value head dim 144 with RuntimeError (no available
        # kernel), except on SM 9.0 or 10.0 with cuDNN 9.5.1 or newer where the call should
        # succeed.
        # Test that cuDNN attention dispatching correctly bails out on d > 128
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 128, 144

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        device_cap = torch.cuda.get_device_capability()
        ISSM90 = device_cap == (9, 0)
        ISSM100 = device_cap == (10, 0)
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            if (ISSM90 or ISSM100) and torch.backends.cudnn.version() >= 90501:
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    @parametrize("shape", [(65536, 1, 4, 8), (1, 65536, 4, 8)])
    def test_cudnn_attention_fail_large_batch_or_num_heads(self, device, shape):
        # [FUNCTION DOC] cuDNN test: half precision inputs with a batch size or a head count of
        # 65536 make the cuDNN backend raise RuntimeError reporting no available kernel.
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_trivial_output_transpose(self, device):
        # [FUNCTION DOC] Regression test (issue 134001): a cuDNN SDPA on a transposed size-1
        # sequence input, followed by a transpose and reshape of the output and backward, must
        # give input gradients close to the CPU computation (atol and rtol 7e-3).
        # see also: https://github.com/pytorch/pytorch/issues/134001
        x = torch.randn(2, 4, 1, 64, device='cuda', dtype=torch.float16, requires_grad=True)
        x2 = x.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            o = torch.nn.functional.scaled_dot_product_attention(x2, x2, x2).transpose(1, 2).reshape(2, 64, 4)
        o.backward(o)
        x_cpu = x.clone().cpu().detach()
        x_cpu.requires_grad = True
        x2_cpu = x_cpu.transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(x2_cpu, x2_cpu, x2_cpu).transpose(1, 2).reshape(2, 64, 4)
        o.backward(o)
        torch.testing.assert_close(x.grad, x_cpu.grad.cuda(), atol=7e-3, rtol=7e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_nonmodulo64seqlen(self, device):
        # [FUNCTION DOC] Regression test (issue 137347): cuDNN attention with a random boolean
        # mask, query length 157 and key length 6404 (not multiples of 64) in half precision;
        # the query, key and value gradients must match the CPU computation with atol 3e-3 and
        # rtol 2e-3.
        # see also: https://github.com/pytorch/pytorch/issues/137347
        mask = torch.randint(0, 2, (2, 1, 157, 6404)).to(device="cuda", dtype=torch.bool)
        q = torch.randn(2, 32, 157, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, 32, 6404, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(2, 32, 6404, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        q_cpu = q.detach().clone().cpu()
        k_cpu = k.detach().clone().cpu()
        v_cpu = v.detach().clone().cpu()
        q_cpu.requires_grad = True
        k_cpu.requires_grad = True
        v_cpu.requires_grad = True
        mask_cpu = mask.detach().clone().cpu()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            out = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        out_cpu = nn.functional.scaled_dot_product_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            attn_mask=mask_cpu,
            dropout_p=0.0,
            is_causal=False,
        )

        out.sum().backward()
        out_cpu.sum().backward()

        torch.testing.assert_close(q.grad, q_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(k.grad, k_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(v.grad, v_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_preserves_query_layout(self, device):
        # [FUNCTION DOC] Checks that cuDNN attention outputs preserve the memory layout of the
        # query: for every permutation of batch, heads and sequence dimensions (head dim last),
        # eager and torch.compile runs must return an output that is contiguous in the permuted
        # order, and backward must run. Expected failure when no accelerator Triton is
        # available.

        def test_attention(backend: SDPBackend, permute_order: list[list[int]], use_compile: bool):
            # [FUNCTION DOC] HELPER: For one backend and one dimension permutation, builds
            # bfloat16 query/key/value views whose logical shape is batch 4, heads 16, query
            # length 256, key length 512, head dim 64 but whose memory order follows the
            # permutation, runs SDPA (optionally compiled) under the backend, asserts the output
            # is contiguous in the permuted order, and runs backward on its sum. Resets dynamo
            # first.
            torch._dynamo.reset()
            BHSqD = [4, 16, 256, 64]
            BHSkvD = [4, 16, 512, 64]

            shape_q = [BHSqD[idx] for idx in permute_order]
            shape_kv = [BHSkvD[idx] for idx in permute_order]
            reverse = [permute_order.index(idx) for idx in range(4)]
            q = torch.randn(*shape_q, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            k = torch.randn(*shape_kv, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            v = torch.randn(*shape_kv, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            self.assertEqual(q.shape, BHSqD)
            self.assertEqual(k.shape, BHSkvD)
            self.assertEqual(v.shape, BHSkvD)

            def fn(q, k, v):
                # [FUNCTION DOC] HELPER: Runs SDPA on the three inputs inside the selected
                # backend context and returns the result.
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q, k, v)

            if use_compile:
                fn = torch.compile(fn)

            out = fn(q, k, v)
            self.assertTrue(out.permute(permute_order).is_contiguous())
            out.sum().backward()

        permute_orders = list()
        permutable = [0, 1, 2]
        permute_orders = itertools.permutations(permutable)

        for permute_order in permute_orders:
            for use_compile in [False, True]:
                test_attention(SDPBackend.CUDNN_ATTENTION, list(permute_order) + [3], use_compile)

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cudnn_attention_broadcast_stride_zero(self, device, dtype):
        # [FUNCTION DOC] Regression test: query/key/value with zero strides over batch and heads
        # (broadcast via as_strided) must run on the cuDNN backend with causal masking, produce
        # an output of the expected shape with last dimension stride one, and match a
        # computation on contiguous copies (atol/rtol 3e-3).
        # Regression test: Q/K/V with zero strides (broadcast via as_strided)
        # caused alloc_with_matching_layout to produce output with stride[-1] != 1,
        # which cuDNN Frontend rejects.
        N, n_head, L, S, E, Ev = 8, 2, 64, 16, 128, 64
        q = torch.randn(N, n_head, L, E, dtype=dtype, device=device)
        k = torch.randn(N, n_head, S, E, dtype=dtype, device=device)
        v = torch.randn(N, n_head, S, Ev, dtype=dtype, device=device)

        q_bc = torch.as_strided(q, size=q.shape, stride=(0, 0, E, 1))
        k_bc = torch.as_strided(k, size=k.shape, stride=(0, 0, E, 1))
        v_bc = torch.as_strided(v, size=v.shape, stride=(0, 0, Ev, 1))

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q_bc, k_bc, v_bc, is_causal=True)

        self.assertEqual(out.shape, (N, n_head, L, Ev))
        self.assertEqual(out.stride(-1), 1)

        out_ref = F.scaled_dot_product_attention(
            q_bc.contiguous(), k_bc.contiguous(), v_bc.contiguous(), is_causal=True
        )
        torch.testing.assert_close(out, out_ref, atol=3e-3, rtol=3e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    @unittest.skipIf(
        not _cudnn_supports_d192_attention(),
        "head_dim=192,d_v=128 requires SM90 or SM10.x with cuDNN >= 9.11.0"
    )
    def test_cudnn_attention_interleaved_layout_compile(self, device):
        # [FUNCTION DOC] Checks cuDNN attention under torch.compile when the query has an
        # interleaved head layout (heads permuted from sequence-major storage, so it is
        # non-contiguous): the compiled function must return the expected output shape.
        # Regression for a stride mismatch between inductor prediction and the runtime
        # allocation. Requires head dims 192 and 128 support.
        """
        Test cudnn attention with interleaved head layout under torch.compile.

        This tests the specific case where query has strides like (B*S*H*D, D, H*D, 1)
        meaning heads are interleaved within the sequence dimension. This layout
        triggered a bug where the C++ alloc_with_matching_layout had an operator
        precedence issue causing stride mismatches between Inductor's meta prediction
        and runtime allocation.
        """
        torch._dynamo.reset()

        B, H, S, D_q, D_v = 1, 128, 512, 192, 128

        q = torch.randn(B, S, H, D_q, dtype=torch.bfloat16, device='cuda').permute(0, 2, 1, 3)
        k = torch.randn(B, S, H, D_q, dtype=torch.bfloat16, device='cuda').permute(0, 2, 1, 3)
        v = torch.randn(B, S, H, D_v, dtype=torch.bfloat16, device='cuda').permute(0, 2, 1, 3)

        self.assertEqual(q.shape, (B, H, S, D_q))
        self.assertEqual(q.stride(1), D_q)
        self.assertEqual(q.stride(2), H * D_q)
        self.assertFalse(q.is_contiguous())

        @torch.compile
        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: Compiled function running SDPA with the cuDNN backend on
            # the three inputs.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v)

        out = fn(q, k, v)
        self.assertEqual(out.shape, (B, H, S, D_v))

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_compiles(self):
        # [FUNCTION DOC] Compiles a function that runs cuDNN SDPA forward and backward on a half
        # precision query used as query, key and value, and compares the output (atol/rtol 1e-3)
        # and the gradient (atol 7e-3, rtol 5e-3) with a float32 CPU computation. Expected
        # failure without accelerator Triton.
        q = torch.randn(2, 8, 1024, 128, dtype=torch.half, device='cuda', requires_grad=True)
        grad = torch.randn_like(q)

        @torch.compile()
        def func():
            # [FUNCTION DOC] HELPER: Compiled function that runs cuDNN SDPA on the enclosing
            # query, backpropagates the enclosing upstream gradient and returns the output.
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                out.backward(grad)
            return out

        out = func()

        q_cpu = q.float().cpu().detach().clone()
        q_cpu.requires_grad = True
        grad_cpu = grad.cpu().float()
        out_cpu = torch.nn.functional.scaled_dot_product_attention(q_cpu, q_cpu, q_cpu)
        out_cpu.backward(grad_cpu)
        self.assertEqual(out, out_cpu.cuda().half(), atol=1e-3, rtol=1e-3)
        self.assertEqual(q.grad, q_cpu.grad.cuda().half(), atol=7e-3, rtol=5e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_seqlen1_dropout_heuristic(self):
        # [FUNCTION DOC] Smoke test: half precision SDPA with query length 1, dropout 0.5 and
        # both cuDNN and flash backends enabled must run forward and backward without error.
        q = torch.randn(2, 8, 1, 128, dtype=torch.half, device='cuda', requires_grad=True)
        grad = torch.randn_like(q)

        with torch.nn.attention.sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=0.5)
            out.backward(grad)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_low_dropout(self):
        # [FUNCTION DOC] Checks that dropout probability of 1e-11 yields an output equal to the
        # no-dropout output: the inequality assertion is expected to fail, which the test
        # requires via assertRaisesRegex.
        q = torch.randn(2, 8, 128, 128, dtype=torch.half, device='cuda')
        dropout_p = 0.00000000001
        out1 = torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=dropout_p)
        out2 = torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=0.)
        with self.assertRaisesRegex(AssertionError, "AssertionError not raised"):
            self.assertNotEqual(out1, out2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_broken_166211(self):
        # [FUNCTION DOC] Regression test for issue 166211: 100 iterations of cuDNN SDPA forward
        # and gradient computation on bfloat16 inputs of shape 20 by 4 by 4 by 32 scaled by 10
        # must produce no NaN in any gradient.
        # https://github.com/pytorch/pytorch/issues/166211#issue-3551350377
        shape = (20, 4, 4, 32)
        scale = 10
        for _ in range(100):
            q = torch.randn(*shape, device='cuda', dtype=torch.bfloat16) * scale
            k = torch.randn(*shape, device='cuda', dtype=torch.bfloat16) * scale
            v = torch.randn(*shape, device='cuda', dtype=torch.bfloat16) * scale
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True

            grad_attn_output = torch.randn(*shape, device='cuda', dtype=torch.bfloat16) * scale

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                dq, dk, dv = torch.autograd.grad(outputs=attn_output, inputs=(q, k, v), grad_outputs=grad_attn_output)

            self.assertFalse(dq.isnan().any())
            self.assertFalse(dk.isnan().any())
            self.assertFalse(dv.isnan().any())

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_mask_broken_177842(self):
        # [FUNCTION DOC] Regression test for issue 177842: cuDNN SDPA with GQA on bfloat16
        # inputs and three kinds of boolean masks (lower triangular, upper triangular, full
        # rows) restricted to a block of the mask, must match the math backend within atol 5e-3
        # and rtol 3e-3.
        # https://github.com/pytorch/pytorch/issues/177842
        q = torch.randn(1, 10, 8, 8, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(1, 10, 1, 8, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(1, 10, 1, 8, dtype=torch.bfloat16, device='cuda')

        attention_mask_custom = torch.zeros(10, 10, dtype=torch.bool).to("cuda")
        attention_mask_custom[:7, :7] = torch.tril(torch.ones(7, 7, dtype=torch.bool), diagonal=0)

        with sdpa_kernel(SDPBackend.MATH):
            attn_output_math = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output_cudnn = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        self.assertEqual(attn_output_math, attn_output_cudnn, atol=5e-3, rtol=3e-3)

        attention_mask_custom = torch.zeros(10, 10, dtype=torch.bool).to("cuda")
        attention_mask_custom[:7, :7] = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=0)

        with sdpa_kernel(SDPBackend.MATH):
            attn_output_math = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output_cudnn = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        self.assertEqual(attn_output_math, attn_output_cudnn, atol=5e-3, rtol=3e-3)

        attention_mask_custom = torch.zeros(10, 10, dtype=torch.bool).to("cuda")
        attention_mask_custom[:7, :10] = torch.ones(7, 10, dtype=torch.bool)

        with sdpa_kernel(SDPBackend.MATH):
            attn_output_math = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output_cudnn = torch.nn.functional.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=attention_mask_custom,
                is_causal=False,
                enable_gqa=True,
            )
        self.assertEqual(attn_output_math, attn_output_cudnn, atol=5e-3, rtol=3e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("mask_dtype", [torch.float32, None])
    def test_cudnn_attention_attn_mask_dtype(self, device, dtype, mask_dtype):
        # [FUNCTION DOC] Regression test for the cuDNN bias dtype: SDPA with a float32 mask or a
        # mask matching the query dtype (control), alternating dtypes with identical strides to
        # exercise the execution plan cache, must be finite and match the math backend (the
        # forward within dtype tolerances, gradients within four times the tolerance). Gradients
        # are taken outside the backend context.
        # scaled_dot_product_attention accepts an attn_mask that is float32 as well as one matching
        # query's dtype. The cuDNN graph declared the bias with the graph-wide io data type, so an
        # fp32 mask reached cuDNN described as half and was reinterpreted element by element:
        # silently wrong output and gradients. mask_dtype=None is the matched-dtype control.
        #
        # Both dtypes are issued back to back at one shape. They have identical sizes and strides,
        # so before the bias dtype was added to the execution-plan cache key they shared one
        # cached plan and the result depended on which dtype the process saw first.
        batch, num_heads, seq_len, head_dim = 2, 8, 128, 64
        make_tensor = partial(torch.randn, device=device, dtype=dtype, requires_grad=True)
        shape = SdpaShape(batch, num_heads, seq_len, head_dim)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        mask = torch.randn(batch, 1, seq_len, seq_len, device=device, dtype=dtype)
        if mask_dtype is None:
            masks = (mask,)   # control: the matched-dtype path must stay correct
        else:
            # the same logical mask in the other dtype, with identical sizes and strides, so
            # before the dtype entered the key these two shared one cached plan. Alternate them
            # so both orders are exercised regardless of the order pytest runs the variants in.
            other = mask.to(mask_dtype)
            self.assertEqual(mask.stride(), other.stride())
            masks = (mask, other, mask, other)

        atol, rtol = (5e-3, 5e-3) if dtype == torch.float16 else (2e-2, 2e-2)
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

        for m in masks:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=m)
            self.assertTrue(out.isfinite().all(), f"non-finite output for a {m.dtype} mask")
            self.assertEqual(out, ref, atol=atol, rtol=rtol, msg=f"wrong result for a {m.dtype} mask")

        # autograd.grad is intentionally outside the sdpa_kernel context: the backward op is fixed
        # by the autograd node recorded during the forward, so it still dispatches to
        # _scaled_dot_product_cudnn_attention_backward.
        grad_out = torch.randn_like(out)
        grads = torch.autograd.grad(out, (query, key, value), grad_out)
        grads_ref = torch.autograd.grad(ref, (query, key, value), grad_out)
        for grad, grad_ref in zip(grads, grads_ref):
            self.assertTrue(grad.isfinite().all())
            self.assertEqual(grad, grad_ref, atol=4 * atol, rtol=4 * rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("mask_dim", [1, 2, 3, 4])
    def test_mem_efficient_attention_mask_variants(self, device, mask_dim: list[int]):
        # [FUNCTION DOC] Runs efficient attention forward and backward on half precision inputs
        # with 1D, 2D, 3D or 4D attention bias shapes (key length 15, not aligned); only checks
        # that no error occurs.
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 15
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        if mask_dim == 1:
            mask = torch.randn((seq_len_kv,), device=device, dtype=dtype)
        elif mask_dim == 2:
            mask = torch.randn((seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 3:
            mask = torch.randn((num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 4:
            mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_mask_only_requires_grad(self, device):
        # [FUNCTION DOC] Checks efficient attention when only the mask requires grad: output and
        # mask gradient must match the math backend within 1e-5, after a CUDA synchronize.
        torch.manual_seed(0)
        query, key, value = (torch.randn((1, 1, 64, 16), device=device) for _ in range(3))
        mask = torch.randn((1, 1, 64, 64), device=device)

        actual_mask = mask.detach().clone().requires_grad_()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = F.scaled_dot_product_attention(query, key, value, actual_mask)
        actual.sum().backward()
        torch.cuda.synchronize()

        expected_mask = mask.detach().clone().requires_grad_()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(query, key, value, expected_mask)
        expected.sum().backward()

        self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(actual_mask.grad, expected_mask.grad, atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("mask_factory", ["ones", "new_ones"])
    def test_mem_efficient_attention_vmap_attn_mask(self, device, mask_factory: str):
        # [FUNCTION DOC] Checks that vmap over SDPA with an all-True mask created via ones or
        # via new_ones on the input matches a loop of per-sample calls, for output and input
        # gradient, on the efficient backend.
        """Exercise vmap masking through the fused memory-efficient SDPA batch rule."""
        batch, vmap_batch, num_heads, seq_len, embed_dim = 2, 8, 2, 4, 64
        x = torch.randn(
            batch,
            vmap_batch,
            seq_len,
            embed_dim,
            device=device,
            requires_grad=True,
        )

        def run_attention(x):
            # [FUNCTION DOC] HELPER: Reshapes the input to batch, heads, sequence, head dim,
            # builds an all-True boolean mask (by ones or by the new_ones method of the input
            # depending on mask_factory), runs SDPA with it, and returns the output merged back
            # to batch, sequence, embedding.
            qkv = x.view(batch, seq_len, num_heads, embed_dim // num_heads).transpose(1, 2)
            if mask_factory == "new_ones":
                attn_mask = x.new_ones(seq_len, seq_len, dtype=torch.bool)
            else:
                attn_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            out = F.scaled_dot_product_attention(qkv, qkv, qkv, attn_mask=attn_mask)
            return out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.vmap(run_attention, in_dims=1, out_dims=1)(x)
            expected = torch.stack([run_attention(x[:, i]) for i in range(vmap_batch)], dim=1)
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grad, expected_grad)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_efficient_attention_vmap_attn_mask_only(self, device):
        # [FUNCTION DOC] Checks vmap over SDPA when only the attention mask is batched
        # (efficient backend): output and mask gradient equal those of a loop of per-sample
        # calls.
        """Exercise vmap when only the SDPA attention mask is batched."""
        vmap_batch, batch, num_heads, seq_len, head_dim = 8, 2, 2, 4, 32
        query = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        attn_masks = torch.randn(
            vmap_batch, seq_len, seq_len, device=device, requires_grad=True
        )

        def run_attention(attn_mask):
            # [FUNCTION DOC] HELPER: Runs SDPA with the enclosing query as query, key and value
            # and the given mask.
            return F.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.vmap(run_attention)(attn_masks)
            expected = torch.stack([run_attention(attn_masks[i]) for i in range(vmap_batch)])
        actual_grad = torch.autograd.grad(actual.sum(), attn_masks)[0]
        expected_grad = torch.autograd.grad(expected.sum(), attn_masks)[0]
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grad, expected_grad)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_efficient_attention_vmap_gqa_backward(self, device):
        # [FUNCTION DOC] Checks vmap over GQA SDPA with per-query-head masks on the efficient
        # backend: the output matches a loop of per-sample calls and the gradients of query,
        # key, value and mask match (atol and rtol 1e-4).
        """Exercise GQA backward with a per-query-head mask under vmap."""
        vmap_batch, batch, num_query_heads, num_kv_heads = 3, 2, 6, 2
        make_tensor = partial(
            torch.randn,
            vmap_batch,
            batch,
            device=device,
            requires_grad=True,
        )
        query = make_tensor(num_query_heads, 7, 32)
        key = make_tensor(num_kv_heads, 11, 32)
        value = make_tensor(num_kv_heads, 11, 24)
        attn_mask = make_tensor(num_query_heads, 7, 11)
        grad_out = torch.randn(
            vmap_batch, batch, num_query_heads, 7, 24, device=device
        )

        def run_attention(query, key, value, attn_mask):
            # [FUNCTION DOC] HELPER: Runs SDPA with the GQA flag enabled on the given tensors
            # and mask.
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask, enable_gqa=True
            )

        inputs = (query, key, value, attn_mask)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.vmap(run_attention)(*inputs)
            expected = torch.stack(
                [
                    run_attention(query[i], key[i], value[i], attn_mask[i])
                    for i in range(vmap_batch)
                ]
            )
        actual_grads = torch.autograd.grad(actual, inputs, grad_out)
        expected_grads = torch.autograd.grad(expected, inputs, grad_out)

        self.assertEqual(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertEqual(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_non_contiguous_mask(self, device, dtype):
        # [FUNCTION DOC] Smoke test: efficient attention forward and backward with a broadcast
        # attention mask built as a zero-stride (expanded) view, in float32 and float16; only
        # checks no error occurs.
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 16
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        mask = torch.as_strided(mask, (batch, num_heads, seq_len_q, seq_len_kv), (0, 0, 0, 1))
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_long_sequence_mask(self, device, dtype):
        # [FUNCTION DOC] Smoke test: efficient attention forward and backward with a large of
        # 8192 by 8192 with 32 heads mask. Returns early when the GPU has under 80GiB of memory.
        # KNOWN ISSUE: the low memory branch only builds a skip object without raising, so the
        # test silently passes instead of being reported as skipped.
        if torch.cuda.get_device_properties('cuda').total_memory < 80 * 2**30:
            unittest.skip("This test requires substatnial GPU memory.")
            return
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 1, 32, 64
        seq_len_q, seq_len_kv = 8192, 8192
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_non_contig_mask_bug(self, device):
        # [FUNCTION DOC] Regression test: efficient attention with strided (partly broadcast)
        # query/key/value and a non-contiguous bias must agree with the result for a contiguous
        # copy of the bias (mean absolute difference below 1e-7).
        # Without the fix this produces `AssertionError: assert 0.07352933287620544 < 1e-07`
        # Shapes taken from repro
        query_size = (3, 16, 1, 128)
        query_strides = (2304, 128, 2048, 1)
        key_size = (3, 16, 14, 128)
        key_strides = (3584, 0, 256, 1)
        value_size = (3, 16, 14, 128)
        value_strides = (3584, 0, 256, 1)
        attention_mask_size = (3, 1, 1, 14)
        attn_mask_strides = (14, 14, 14, 1)

        # Calculate the number of elements needed for each tensor
        query_num_elements = max(size * stride for size, stride in zip(query_size, query_strides))
        key_num_elements = max(size * stride for size, stride in zip(key_size, key_strides))
        value_num_elements = max(size * stride for size, stride in zip(value_size, value_strides))
        attention_mask_num_elements = max(size * stride for size, stride in zip(attention_mask_size, attn_mask_strides))

        # Create the tensors with the specified sizes and strides
        query = torch.randn(query_num_elements, device=device).as_strided(query_size, query_strides)
        key = torch.randn(key_num_elements, device=device).as_strided(key_size, key_strides)
        value = torch.randn(value_num_elements, device=device).as_strided(value_size, value_strides)
        bias = torch.randn(attention_mask_num_elements, device=device).as_strided(attention_mask_size, attn_mask_strides)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, bias)
            out_contig = F.scaled_dot_product_attention(query, key, value, bias.contiguous())

        max_diff = (out - out_contig).abs().mean()
        self.assertTrue(max_diff.item() < 1e-7)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_single_query_tail_mask(self, device):
        # [FUNCTION DOC] Checks efficient attention against the math backend (bfloat16, scale 1,
        # head dim 512, query length 289, broadcast key/value, lower-triangular boolean mask),
        # exercising a single-query tail block. Skipped on ROCm.
        seq_len, num_heads, head_dim = 289, 16, 512
        make_tensor = partial(torch.randn, device=device, dtype=torch.bfloat16)
        query = make_tensor(1, seq_len, num_heads, head_dim).transpose(1, 2)
        key = make_tensor(seq_len, head_dim).expand(1, num_heads, -1, -1)
        value = make_tensor(seq_len, head_dim).expand(1, num_heads, -1, -1)
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()[None, None]

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(query, key, value, mask, scale=1.0)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = F.scaled_dot_product_attention(query, key, value, mask, scale=1.0)

        self.assertEqual(actual, expected, atol=2e-2, rtol=2e-2)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @unittest.skipIf(not SM80OrLater, "bfloat16 requires SM80 or later")
    @parametrize("kv_len,num_heads,is_causal", [(289, 40, False), (400, 16, True)])
    def test_mem_eff_attention_single_query_tail(self, device, kv_len, num_heads, is_causal):
        # [FUNCTION DOC] Checks efficient attention against the math backend (bfloat16, head dim
        # 512, 289 queries, 40 or 16 heads, causal or not) when the last query block has a
        # single query; tolerance 2e-2. Skipped on ROCm and pre-SM80.
        # head_dim > 128 gives 32 queries per block, so 289 queries leave a single-query
        # tail block, which iterates over heads instead of queries: 40 heads reach a
        # second head block, and the causal case must still bound keys at the tail
        # row's own diagonal.
        q_len, head_dim = 289, 512
        make_tensor = partial(torch.randn, device=device, dtype=torch.bfloat16)
        query = make_tensor(1, q_len, num_heads, head_dim).transpose(1, 2)
        key = make_tensor(kv_len, head_dim).expand(1, num_heads, -1, -1)
        value = make_tensor(kv_len, head_dim).expand(1, num_heads, -1, -1)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=1.0)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=1.0)

        self.assertEqual(actual, expected, atol=2e-2, rtol=2e-2)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @unittest.skipIf(not SM80OrLater, "bfloat16 requires SM80 or later")
    def test_mem_eff_attention_dropout_rng_offset_no_wraparound(self, device):
        # [FUNCTION DOC] Regression test: with 65536 queries and keys and dropout, the random
        # offset must not wrap around at 32 bits, so two heads with identical inputs must give
        # different outputs.
        # num_queries * num_keys == 2^32, which wraps a 32-bit offset so that head 1
        # replays head 0's mask. Inputs are identical per head, so a shared mask shows
        # up as identical outputs.
        seq_len, num_heads, head_dim = 1 << 16, 2, 64
        base = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        query, key, value = (base.expand(1, num_heads, -1, -1).contiguous().requires_grad_() for _ in range(3))

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.5, scale=1.0)

        self.assertFalse(torch.equal(out[0, 0], out[0, 1]))

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @unittest.skipIf(not SM80OrLater, "bfloat16 requires SM80 or later")
    def test_mem_eff_attention_dropout_single_query_tail(self, device):
        # [FUNCTION DOC] Regression test: with dropout 0.5 and key length 1, the last query row
        # of heads one onward must not equal the rows of head one (no replay of the same dropout
        # mask in the single-query tail remap).
        # The tail remap makes rows heads while the RNG offset still treats them as
        # queries, so head h's tail row would replay head 1's row h - 1. kv_len == 1
        # makes each output element either zero or v/(1 - p), exposing the mask.
        q_len, kv_len, num_heads, head_dim = 65, 1, 32, 64
        make_tensor = partial(torch.randn, device=device, dtype=torch.bfloat16)
        query = make_tensor(1, num_heads, q_len, head_dim)
        key = make_tensor(kv_len, head_dim).expand(1, num_heads, -1, -1)
        value = make_tensor(kv_len, head_dim).expand(1, num_heads, -1, -1)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.5, scale=1.0)

        tail_row = out[0, 1:, q_len - 1, 0]
        head_one_rows = out[0, 1, : num_heads - 1, 0]
        self.assertFalse(torch.equal(tail_row, head_one_rows))

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_save_on_cpu_unaligned_bias_stride(self, device):
        # [FUNCTION DOC] Regression test: efficient attention with a float16 bias of unaligned
        # stride and with the autograd saved-tensor hook that moves saved tensors to CPU must
        # give the same output and the same query, key, value and mask gradients as without the
        # hook.
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, seq_len_q, seq_len_kv, head_dim = 1, 8, 1, 361, 64
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        attn_mask = torch.randn((batch, 1, seq_len_q, seq_len_kv), device=device, dtype=dtype, requires_grad=True)
        self.assertEqual(attn_mask.stride(1), 361)
        ref_query = query.detach().clone().requires_grad_()
        ref_key = key.detach().clone().requires_grad_()
        ref_value = value.detach().clone().requires_grad_()
        ref_attn_mask = attn_mask.detach().clone().requires_grad_()

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            ref_out = F.scaled_dot_product_attention(ref_query, ref_key, ref_value, ref_attn_mask)
            ref_out.sum().backward()
            with torch.autograd.graph.save_on_cpu():
                out = F.scaled_dot_product_attention(query, key, value, attn_mask)
        out.sum().backward()

        self.assertEqual(out, ref_out)
        for actual, expected in (
            (query.grad, ref_query.grad),
            (key.grad, ref_key.grad),
            (value.grad, ref_value.grad),
            (attn_mask.grad, ref_attn_mask.grad),
        ):
            self.assertEqual(actual, expected)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_singelton_head_dim_stride_ne_1(self, device):
        # [FUNCTION DOC] Smoke test: flash attention with a size-1 head dim whose query stride
        # is not one (via transpose) on float16 inputs using the legacy sdp_kernel context; only
        # checks no error occurs.
        query = torch.tensor([[[[1, 2]]]], dtype=torch.float16, device=device)
        query = query.transpose(-1, -2)
        key = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)
        value = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)

        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
            scaled_dot_product_attention(query, key, value)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        # [FUNCTION DOC] Compares efficient attention on query/key/value taken from a packed qkv
        # tensor (dense or nested, contiguous or as transposed views, half precision, batch 32,
        # 16 heads, head dim 64) against the math backend on contiguous copies with atol 2e-3
        # and rtol 1e-2.
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "Fused SDPA was not built for this system")
    @unittest.skipIf("TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED" not in os.environ, "cuDNN Nested Tensor support not enabled")
    @parametrize("type", ["nested"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_cudnn_nested(self, device, type: str, is_contiguous: bool):
        # [FUNCTION DOC] Like the packed efficient attention test but for the cuDNN backend on
        # nested packed inputs (batch 8); skipped unless the environment variable enabling cuDNN
        # nested tensor support is set.
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 8, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)
        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_scaled_dot_product_attention_fused_kernels_packed_accuracy(self, device, type: str, fused_kernel: str):
        # [FUNCTION DOC] Accuracy check of flash or efficient attention on packed float16 inputs
        # (dense or nested) built from values in the range -3 to 3. Compares the fused result
        # with a float32 math reference and the float16 math reference with it (nested results
        # are first converted to padded tensors); tolerances are 8e-3 and 7e-3 respectively.
        def rand_nt(shape):
            # [FUNCTION DOC] HELPER: Builds a list of random float32 packed tensors of sequence
            # by three times heads times head dim in the range -3 to 3 and returns a pair of
            # nested tensors, in float32 and in float16, made from that list.
            batch, seq_len, num_heads, head_dim = shape
            tensors = [6 * torch.rand((seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
                       for _ in range(batch)]
            return (torch.nested.nested_tensor(tensors, device=device, dtype=torch.float32),
                    torch.nested.nested_tensor(tensors, device=device, dtype=torch.float16))

        def rand_tensor(shape):
            # [FUNCTION DOC] HELPER: Builds one random float32 packed dense tensor of batch,
            # sequence, three times heads times head dim in the range -3 to 3 and returns it
            # with its float16 copy.
            batch, seq_len, num_heads, head_dim = shape
            tensor = 6 * torch.rand((batch, seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
            return tensor, tensor.to(dtype=torch.float16)

        batch_size, seq_len, num_heads, head_dim = 16, 8, 4, 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Test Packed
        qkv, qkv_low_precision = rand_tensor(shape) if type == "dense" else rand_nt(shape)
        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_low_precision.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp.contiguous(), key_lp.contiguous(), value_lp.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

            math_query = query.contiguous()
            math_key = key.contiguous()
            math_value = value.contiguous()

            math_ref = torch.nn.functional.scaled_dot_product_attention(
                math_query, math_key, math_value, attn_mask=None, dropout_p=0.0, is_causal=False)

        actual_test = actual
        math_ref_test = math_ref
        math_ref_lp_test = math_ref_lp

        if actual_test.is_nested:
            actual_test = torch.nested.to_padded_tensor(actual_test.contiguous(), padding=0.0)
            math_ref_test = torch.nested.to_padded_tensor(math_ref_test, padding=0.0)
            math_ref_lp_test = torch.nested.to_padded_tensor(math_ref_lp_test, padding=0.0)

        actual_test = actual_test.to(dtype=torch.float32).contiguous()
        math_ref_test = math_ref_test.to(dtype=torch.float32).contiguous()
        math_ref_lp_test = math_ref_lp_test.to(dtype=torch.float32).contiguous()

        self.assertEqual(math_ref_test, math_ref_lp_test, atol=8e-3, rtol=7e-3)
        self.assertEqual(actual_test, math_ref_test, atol=7e-3, rtol=7e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Efficient Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    def test_sdp_mem_efficient_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool):
        # [FUNCTION DOC] Compares the gradients of efficient attention in float32 against the
        # math backend in float64 for packed qkv inputs (contiguous or not, causal or not):
        # after backward with the same random upstream gradient, the packed qkv gradients must
        # agree within 1e-5.
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
        qkv_lp = qkv.detach().clone().to(torch.float32).requires_grad_()

        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(torch.float32)

        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # Cast up and compare
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_flash_attention_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool, dtype: torch.dtype):
        # [FUNCTION DOC] Compares the gradients of flash attention in float16 or bfloat16 with
        # the float64 math backend for packed qkv inputs (contiguous or not, causal or not);
        # tolerance depends on dtype (7e-4 for float16, 7e-3 for bfloat16; slightly larger on
        # ROCm).
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
        qkv_lp = qkv.detach().clone().to(dtype).requires_grad_()

        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(dtype)

        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # Cast up and compare
        # Since we are doing the compute on fp16 we have to bump the tolerance
        # Bump down the tolerance for blfoat16
        atol = 7e-4 if dtype == torch.float16 else 7e-3
        rtol = 7e-4 if dtype == torch.float16 else 7e-3
        if TEST_WITH_ROCM:
            atol = 9e-4 if dtype == torch.float16 else 9e-3
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("type", ["dense", "nested"])
    def test_fused_sdp_choice(self, device, type: str):
        # [FUNCTION DOC] Checks the automatic backend selection for packed half precision
        # inputs, dense or nested: cuDNN when it is preferred (environment flag, Hopper or SM 10
        # class GPU, cuDNN newer than 9.15.0) for dense inputs, else flash when supported, else
        # efficient (with cuDNN selectable explicitly when supported but not preferred). With
        # float32 inputs the efficient backend must be chosen (AssertionError otherwise).
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float16, packed=True, requires_grad=True)

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        device_capability = None
        if "cuda" in str(device):
            device_capability = torch.cuda.get_device_capability()
        prefer_cudnn = "TORCH_CUDNN_SDPA_PREFERRED" not in os.environ or bool(os.environ["TORCH_CUDNN_SDPA_PREFERRED"])
        # cuDNN prioritization requires cuDNN > 9.15.0 (91500) per sdp_utils.cpp:83
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 0
        is_hopper_or_newer = device_capability and (device_capability[0] == 9 or device_capability[0] == 10)
        prefer_cudnn = prefer_cudnn and is_hopper_or_newer and cudnn_version > 91500

        # cuDNN is enabled by default on SM 9.0/10.0 with cuDNN > 9.15.0 (per #169849)
        # For older cuDNN versions or other architectures, Flash Attention is preferred
        if type != "nested" and PLATFORM_SUPPORTS_CUDNN_ATTENTION and prefer_cudnn:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.CUDNN_ATTENTION.value)
        elif PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.FLASH_ATTENTION.value)
        elif type != "nested" and PLATFORM_SUPPORTS_CUDNN_ATTENTION and not prefer_cudnn:  # e.g., we're on Windows
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.EFFICIENT_ATTENTION.value)
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
                self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.CUDNN_ATTENTION.value)
        else:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.EFFICIENT_ATTENTION.value)

        # Change dtype to float32 so that efficient attention should get chosen
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float32, packed=True)

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if torch._fused_sdp_choice(query, key, value) != SDPBackend.EFFICIENT_ATTENTION.value:
            raise AssertionError("expected EFFICIENT_ATTENTION backend")

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("warn_only", [True, False])
    def test_sdp_choice_with_determinism(self, device, warn_only):
        # [FUNCTION DOC] Checks that with deterministic algorithms enabled (also warn-only) and
        # only efficient and math backends allowed, the fused choice still selects efficient
        # attention for float32 inputs. Raises AssertionError otherwise.
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                if torch._fused_sdp_choice(query, key, value) != SDPBackend.EFFICIENT_ATTENTION.value:
                    raise AssertionError("expected EFFICIENT_ATTENTION backend")

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("use_compile", [True, False])
    def test_fused_sdp_priority_order(self, device, use_compile):
        # [FUNCTION DOC] Performance-ordering test of backend priority: runs SDPA, eagerly or
        # compiled, with four priority orders and times a second run of each; asserts cuDNN is
        # faster than math, efficient is faster than math, flash is faster than math, cuDNN is
        # faster than efficient, and that the priority order is restored to the default
        # afterwards. Timing based, expected to fail without accelerator Triton.
        @torch.compile
        def compiled_func(order):
            # [FUNCTION DOC] HELPER: Compiled function that runs SDPA on the enclosing CUDA
            # query under a priority ordered backend list and returns the result.
            with sdpa_kernel(order, set_priority=True):
                out = scaled_dot_product_attention(q, q, q)
            return out

        q = torch.randn(64, 8, 1024, 64, dtype=torch.half, device='cuda')
        default_order = torch._C._get_sdp_priority_order()
        orders = [[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
                  [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                  [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
                  [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]]
        import time
        times = list()
        for order in orders:
            if use_compile:
                compiled_func(order)
            else:
                with sdpa_kernel(order, set_priority=True):
                    scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_compile:
                compiled_func(order)
            else:
                with sdpa_kernel(order, set_priority=True):
                    scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        self.assertTrue(times[0] < times[1], "expected cuDNN SDPA to be faster than Math backend.")
        self.assertTrue(times[1] > times[2], "expected Eff Attn backend to faster than Math backend.")
        self.assertTrue(times[3] < times[2], "expected Flash Attn backend to faster than Math backend.")
        self.assertTrue(times[0] < times[2], "expected cuDNN Attn backend to faster than Eff Attn backend.")
        reset_order = torch._C._get_sdp_priority_order()
        self.assertEqual(default_order, reset_order, "expected SDPA context manager to reset priority order.")

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    @parametrize("warn_only", [True, False])
    def test_fused_backwards_throws_determinism_warning(self, device, warn_only, fused_kernel):
        # [FUNCTION DOC] Checks that backward of fused kernels under deterministic mode warns
        # that the kernel defaults to a non-deterministic algorithm when warn-only is set; with
        # warn-only off the same call succeeds for flash and efficient attention, and cuDNN
        # attention raises RuntimeError.
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float16, packed=False, requires_grad=True)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        kernel_name = "Memory Efficient attention" if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else \
            "Flash Attention" if fused_kernel == SDPBackend.FLASH_ATTENTION else "cuDNN Attention"
        warning_context = (
            self.assertWarnsRegex(
                UserWarning,
                f"{kernel_name} defaults to a non-deterministic algorithm.",
            )
            if warn_only
            else contextlib.nullcontext()
        )
        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[fused_kernel]):
                with warning_context:
                    if warn_only or fused_kernel != SDPBackend.CUDNN_ATTENTION:
                        torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward()
                    else:
                        # cuDNN attention has no deterministic fallback
                        self.assertRaises(RuntimeError, lambda:
                                          torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward())

    @unittest.skip("This test is not behaving deterministaclly non-deterministaclly on CI/CD")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not support fused SDPA")
    def test_mem_eff_backwards_determinism(self, device):
        # [FUNCTION DOC] Skipped unconditionally. Intended to check that efficient attention
        # backward on long sequences is non-deterministic by default (some of 100 reruns differ
        # in query gradient) and deterministic when deterministic algorithms are enabled (all
        # reruns equal).
        # Need big seq_len to ensure that num_splits > 1
        dtype = torch.float32
        batch_size, seq_len, n_heads, head_dim = 1, 1024, 8, 64
        query = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # Run once to establish baseline
            out = F.scaled_dot_product_attention(query, key, value)
            upward_grad = torch.rand_like(out)
            out.backward(upward_grad)
            initial_query_grad = query.grad

            # Re-run the op with the same upward grad and check that the backward is
            # not deterministic
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(initial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            self.assertTrue(diff_anwser_once)

        with use_deterministic_algorithims(True, warn_only=False):
            query.grad = None
            out = F.scaled_dot_product_attention(query, key, value)
            upward_grad = torch.rand_like(out)
            out.backward(upward_grad)
            initial_query_grad = query.grad

            # Re-run the op with the same upward grad and check that the backward is
            # deterministic now that we have enforced it
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(initial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            self.assertFalse(diff_anwser_once)

    # verified passing successfully on H100
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize(
        "seq_len_q",
        [8, 103, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [4, 8, 256, 512],
    )
    @parametrize(
        "seq_len_k",
        [8, 103, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [4, 8, 256, 512],
    )
    @parametrize(
        "head_dim",
        [8, 16, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80 and not isSM120Device else [8, 16, 32, 64],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize(
        "dtype",
        (
            [torch.float16, torch.bfloat16, torch.float32]
            if MEM_EFF_CAPABILITY_MATCHES_SM80
            else [torch.float16, torch.float32]
        ),
    )
    @parametrize("scale", [None, "l1"])
    @tf32_enabled()
    def test_mem_efficient_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                       head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                                       scale: str):
        # [FUNCTION DOC] Compares efficient attention forward and gradients against a float64
        # math reference and a low precision math reference over many parameter combinations
        # (batch, lengths, head dim, causal flag, dropout, dtype, scale), using
        # check_out_and_grad with per-tensor fudge factors (larger on ROCm). With dropout the
        # reference reuses the mask generated by the dedicated mem-eff mask fill function.
        # Returns early for huge sequences on GPUs under 40GiB, and skips if sequence length
        # exceeds 1024 with dropout.
        # KNOWN ISSUE: the low memory branch only builds a skip object without raising, so those
        # cases silently pass instead of being reported as skipped.
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            # [FUNCTION DOC] HELPER: Generates the dropout keep mask matching the efficient
            # attention kernel for the given shape, probability, seed and offset by filling a
            # float32 tensor via the fill function and thresholding the returned uniform values
            # (threshold zero on ROCm where the fill encodes keep or drop as plus or minus a
            # half). Returns the float32 keep mask.
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        if TEST_WITH_ROCM and seq_len_q * seq_len_k * head_dim * batch_size > 1024 * 1024 * 128:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        higher_precision_dtype = torch.float64
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # Create real output
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                with tf32_off():
                    out_ref = F.scaled_dot_product_attention(
                        query_ref, key_ref, value_ref,
                        dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value,
                                                            dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        else:
            if seq_len_q > 1024:
                self.skipTest("Will call _fill_mem_eff_dropout_mask with too many threads!")
            # Create the dropout_mask
            torch.manual_seed(seed)
            dropout_mask = _get_mem_eff_drop_mask(batch_size, n_heads, seq_len_q, seq_len_k, dropout_p, seed, 0, device=device)
            # High Precision Math Reference
            with tf32_off():
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                    dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
        with tf32_off():
            grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

        fudge_factors = {
            'out': 3.0 ,
            'grad_query': 150.0 ,
            'grad_key': 25.0,
            'grad_value': 8.5,
        }
        if TEST_WITH_ROCM:
            fudge_factors['out'] = 5.0
            fudge_factors['grad_key'] = 45.0
            fudge_factors['grad_query'] = 360.0
            if seq_len_k >= 1024:
                fudge_factors['grad_key'] = 70.0
            if seq_len_k >= 2048:
                fudge_factors['grad_key'] = 160.0
                fudge_factors['grad_query'] = 670.0
            if dtype == torch.float32:
                fudge_factors['grad_key'] = 90.0
            if "gfx95" in torch.cuda.get_device_properties(0).gcnArchName:
                fudge_factors['grad_value'] = 16.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize(
        "seq_len_q",
        [8, 312, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [8, 152, 512],
    )
    @parametrize(
        "seq_len_k",
        [8, 408, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [8, 37, 512],
    )
    @parametrize(
        "head_dim",
        [8, 16, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80 and not isSM120Device else [8, 16, 32, 64],
    )
    @parametrize("is_causal", [False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize(
        "dtype",
        (
            [torch.float16, torch.bfloat16, torch.float32]
            if MEM_EFF_CAPABILITY_MATCHES_SM80
            else [torch.float16, torch.float32]
        ),
    )
    @parametrize("scale", [None, "l1"])
    @tf32_enabled()
    def test_mem_efficient_attention_attn_mask_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int,
                                                                 seq_len_k: int, head_dim: int, is_causal: bool,
                                                                 dropout_p: float, dtype: torch.dtype,
                                                                 scale: str):
        # [FUNCTION DOC] Like the efficient attention gradient comparison but with a learnable
        # additive 2D attention mask (non-causal only): compares output and query, key, value
        # and mask gradients against a higher precision math reference and a low precision
        # reference with dtype-based fudge factors (larger on ROCm), with the same dropout mask
        # handling and memory based early return.
        # KNOWN ISSUE: the low memory branch only builds a skip object without raising, so those
        # cases silently pass instead of being reported as skipped.
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            # [FUNCTION DOC] HELPER: Generates the dropout keep mask matching the efficient
            # attention kernel for the given shape, probability, seed and offset by filling a
            # float32 tensor via the fill function and thresholding the returned uniform values
            # (threshold zero on ROCm). Returns the float32 keep mask.
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        if TEST_WITH_ROCM and seq_len_q * seq_len_k * head_dim * batch_size > 1024 * 1024 * 128:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        attn_mask = torch.rand(seq_len_q, seq_len_k, device=device, dtype=dtype, requires_grad=True)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)
        attn_mask_ref = attn_mask.detach().to(higher_precision_dtype).requires_grad_(True)

        # Create real output
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p=dropout_p,
                                                 is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                with tf32_off():
                    out_ref = F.scaled_dot_product_attention(
                        query_ref, key_ref, value_ref, attn_mask_ref,
                        dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value, attn_mask,
                                                            dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        else:
            if seq_len_q > 1024:
                self.skipTest("Will call _fill_mem_eff_dropout_mask with too many threads!")
            # Create the dropout_mask
            torch.manual_seed(seed)
            dropout_mask = _get_mem_eff_drop_mask(batch_size, n_heads, seq_len_q,
                                                  seq_len_k, dropout_p, seed, 0, device=device)
            # High Precision Math Reference
            with tf32_off():
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, attn_mask_ref, dropout_p=dropout_p, is_causal=is_causal,
                    scale=scale, dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, attn_mask,
                dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        grads = torch.autograd.grad(out, (query, key, value, attn_mask), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value, attn_mask), upstream_grad)
        with tf32_off():
            grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref, attn_mask_ref), upstream_grad)

        fudge_factors = {
            "out": 4,
            "grad_query": 160.0,
            "grad_key": 25.0,
            "grad_value": 8.0,
            "grad_attn_mask": 45.0,
        }
        if TEST_WITH_ROCM:
            fudge_factors['out'] = 6.0
            fudge_factors['grad_key'] = 45.0
            fudge_factors['grad_query'] = 360.0
            if seq_len_k >= 1024:
                fudge_factors['grad_key'] = 70.0
            if seq_len_k >= 2048:
                fudge_factors['grad_key'] = 160.0
                fudge_factors['grad_query'] = 670.0  # gfx90a
            if dtype == torch.float32:
                fudge_factors['grad_key'] = 90.0
                if "gfx95" in torch.cuda.get_device_properties(0).gcnArchName:
                    fudge_factors['grad_value'] = 16.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_CK_SDPA,
        "Regression is specific to the ROCm CK SDPA backend",
    )
    @setSdpaBackendsToDefaultFinally
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("n_heads", [[8, 2], [16, 8]])
    @parametrize("seqlen_k", [4, 256, 1024])
    def test_flash_attention_ck_gqa_seqlen_q_1(self, device, dtype: torch.dtype,
                                               n_heads: list[int], seqlen_k: int):
        # [FUNCTION DOC] ROCm CK flash attention regression test: forces the CK library (skips
        # if unavailable) and runs grouped query attention with a single query position (decode)
        # at several key lengths in half and bfloat16; the output must be finite and match the
        # math reference (see the comment in the body).
        # Regression: the CK flash-attention host wrapper mis-ported the
        # seqlenq_ngroups_swapped optimization, which triggers for GQA
        # (num_heads_q > num_heads_kv) with seqlen_q == 1 (single-query decode).
        # It fed the un-swapped q / original-shaped output to the kernel, so the
        # kernel strided out of bounds and returned finite garbage (up to ~FLT_MAX)
        # that overflowed to NaN/Inf downstream. Verify the CK output is finite and
        # matches the math reference for this previously-untested shape.
        torch.backends.cuda.preferred_rocm_fa_library("ck")
        if (
            torch.backends.cuda.preferred_rocm_fa_library()
            != torch._C._ROCmFABackend.Ck
        ):
            self.skipTest("CK SDPA backend not available")
        batch_size, head_dim, seqlen_q = 8, 128, 1
        num_heads_q, num_heads_kv = n_heads
        q = torch.rand(
            batch_size, num_heads_q, seqlen_q, head_dim,
            device=device, dtype=dtype,
        )
        k = torch.rand(
            batch_size, num_heads_kv, seqlen_k, head_dim,
            device=device, dtype=dtype,
        )
        v = torch.rand(
            batch_size, num_heads_kv, seqlen_k, head_dim,
            device=device, dtype=dtype,
        )
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=True
            )
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            ref = F.scaled_dot_product_attention(
                q.double(), k.double(), v.double(),
                dropout_p=0.0, is_causal=False, enable_gqa=True,
            )
        msg = (
            f"dtype={dtype}, num_heads_q={num_heads_q}, "
            f"num_heads_kv={num_heads_kv}, seqlen_k={seqlen_k}"
        )
        self.assertFalse(
            torch.isnan(out).any().item(),
            lambda _m: f"{_m}\nCK GQA seqlen_q==1 produced NaN ({msg})",
        )
        self.assertFalse(
            torch.isinf(out).any().item(),
            lambda _m: f"{_m}\nCK GQA seqlen_q==1 produced Inf ({msg})",
        )
        self.assertEqual(out, ref.to(out.dtype), atol=2e-2, rtol=2e-2)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @setSdpaBackendsToDefaultFinally
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 143, 2048])
    @parametrize("seq_len_k", [4, 127, 579, 2048])
    @parametrize("head_dim", [8, 203, 256])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22, 0.48])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("scale", [None, "l1"])
    @parametrize("enable_gqa", [True, False])
    @parametrize("n_heads", [[16, 8], [10, 2]])
    @parametrize("sdpa_backend", ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"])
    @tf32_enabled()
    def test_flash_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                               head_dim: int, is_causal: bool, dropout_p: float,
                                               dtype: torch.dtype, scale: str, enable_gqa: bool,
                                               n_heads: list[int], sdpa_backend: str):
        # [FUNCTION DOC] Compares flash attention (with dropout, GQA, causal and scale variants,
        # head dims up to 256) against a higher precision math reference and a same precision
        # math reference for output and gradients of query/key/value using check_out_and_grad
        # with fudge factors that depend on ROCm and CK backends. With dropout the inputs are
        # padded to a multiple of 8 and the dropout mask is rebuilt from the debug mask returned
        # by the flash op (CK uses a different threshold rule). Skips for causal with unequal
        # lengths, CK with head dim over 128, and some SM 8.x/12.x head dim cases; on SM 8.x
        # style devices with large head dims backward is expected to raise RuntimeError.
        # KNOWN ISSUE: the SM8X skip and backward-failure conditions combine the SM8X flag with
        # the head-dim range through operator precedence, so on SM 8.x devices every head dim is
        # skipped.
        # KNOWN ISSUE: the memory-based early exit only builds a skip object without raising, so
        # those cases silently pass instead of being reported as skipped.
        if isSM8XDevice or isSM120Device and head_dim in range(193, 256 + 1):
            self.skipTest("Flash attention on sm86, sm87, and sm89 for headdim > 192 currently disabled")
        if is_causal and seq_len_q != seq_len_k:
            self.skipTest("Flash V2 does not accept is_casual when seq_len_q != seq_len_k")
        if TEST_WITH_ROCM and seq_len_q >= 1024 and seq_len_k >= 1024 and batch_size > 1:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return

        # ROCm now supports 2 different backends for SDPA that require different set up.
        TEST_WITH_CK = False
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)
            # When no args are given to preferred_rocm_fa_library, it acts as a getter
            TEST_WITH_CK = (torch.backends.cuda.preferred_rocm_fa_library() == torch._C._ROCmFABackend.Ck)

        if TEST_WITH_CK and head_dim > 128:
            self.skipTest("CK does not support head dims over 128")

        scale = scale if scale is None else (1 / head_dim)
        num_heads_q = num_heads_kv = 4
        if enable_gqa:
            num_heads_q = n_heads[0]
            num_heads_kv = n_heads[1]

        query = torch.rand(batch_size, num_heads_q, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, num_heads_kv, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, num_heads_kv, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        is_dropout = dropout_p > 0.0

        if not is_dropout:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                with tf32_off():
                    out_ref = F.scaled_dot_product_attention(
                        query_ref, key_ref, value_ref, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        else:
            # Problem: We pad sizes in the composite region of the top level SDPA. But we need the
            # Debug mask when have dropout. So I am going to manually pad up here when testing dropout
            q_padded, q_og_size = pad_last_dim(query, 8)
            k_padded, k_og_size = pad_last_dim(key, 8)
            v_padded, v_og_size = pad_last_dim(value, 8)
            # scale needs to be calculated on the og head_size
            if scale is None:
                scale = 1 / math.sqrt(q_og_size)
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                q_padded, k_padded, v_padded, dropout_p=dropout_p, is_causal=is_causal, scale=scale, return_debug_mask=is_dropout)
            out = output_tuple[0]
            out = out[..., :v_og_size]
            # Build dropout_mask
            dbug_mask = output_tuple[-1]
            query_padding_mask = torch.ones(
                batch_size, seq_len_q, device=device, dtype=torch.bool)
            key_padding_mask = torch.ones(
                batch_size, seq_len_k, device=device, dtype=torch.bool)

            softmax_mask = self.convert_flash_attn_S_to_softmax(
                dbug_mask, seq_len_q, seq_len_k, query_padding_mask, key_padding_mask,
                causal=is_causal)[:, :, :seq_len_q, :seq_len_k]

            # This is the default implementation for the mask but we need to match CK if we are using it
            dropout_mask = softmax_mask >= 0

            # This logic matches how CK calculates the dropout mask.
            # This is necessary because CK doesn't support passing in custom dropout masks
            # So we use this logic to ensure we are comparing apples to apples.
            if TEST_WITH_CK:
                dropout_mask = (softmax_mask <= int((1.0 - dropout_p) * 255.0)).to(torch.float32)

            # High Precision Math Reference
            with tf32_off():
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal,
                    scale=scale, dropout_mask=dropout_mask, enable_gqa=enable_gqa)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, dropout_mask=dropout_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        # backward for flash attention on sm86, sm87, and sm89 for headdim >= 193 currently disabled
        if isSM8XDevice or isSM120Device and head_dim in range(193, 256):
            self.assertRaises(RuntimeError, lambda: out.backward(upstream_grad))
            return

        grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
        with tf32_off():
            grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

        fudge_factors = {
            'out': 4,
            'grad_query': 180.0,
            'grad_key': 16,
            'grad_value': 4,
        }
        if TEST_WITH_ROCM:

            fudge_factors['grad_value'] = 6.0
            if TEST_WITH_CK:
                fudge_factors['out'] = 5.0
                fudge_factors['grad_key'] = 145.0
                fudge_factors['grad_query'] = 855.0  # ck min = 855.0
                if seq_len_k >= 1024:
                    fudge_factors['grad_key'] = 70.0
                if seq_len_k >= 2048:
                    fudge_factors['grad_key'] = 190.0
                    fudge_factors['grad_query'] = 1550.0  # NEW CK MIN
                    if seq_len_q >= 2048:
                        fudge_factors['grad_query'] = 1100.0
                if dtype == torch.float32:
                    fudge_factors['grad_key'] = 90.0
            else:
                fudge_factors['out'] = 6.0
                fudge_factors['grad_key'] = 45.0
                fudge_factors['grad_query'] = 360.0
                if seq_len_k >= 1024:
                    fudge_factors['grad_key'] = 70.0
                if seq_len_k >= 2048:
                    fudge_factors['grad_key'] = 190.0
                    fudge_factors['grad_query'] = 650.0
                    if seq_len_q >= 2048:
                        fudge_factors['grad_query'] = 1100.0
                if dtype == torch.float32:
                    fudge_factors['grad_key'] = 90.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [256, 1024])
    @parametrize("seq_len_k", [256, 1024])
    @parametrize("head_dim", [32, 64])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize("dtype", [torch.float16])
    @parametrize("scale", [None, "l1"])
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    @tf32_enabled()
    def test_fused_attention_vs_math_ref_grads_cudagraph(self, device, batch_size: int,
                                                         seq_len_q: int, seq_len_k: int,
                                                         head_dim: int,
                                                         is_causal: bool,
                                                         dropout_p: float,
                                                         dtype: torch.dtype,
                                                         scale: str,
                                                         fused_kernel: SDPBackend):
        # [FUNCTION DOC] Compares fused kernels (flash, efficient or cuDNN) captured in CUDA
        # graphs against math references: runs the op on a side stream for warmup, captures
        # forward and backward into graphs, replays the forward twice (equal results without
        # dropout, different results with dropout), builds the dropout mask from the efficient
        # seed and offset or from the flash debug mask, and compares output and gradients with
        # check_out_and_grad. cuDNN with dropout skips the reference comparison since it does
        # not return a mask. Skips causal flash with unequal lengths.
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, dropout_p, seed, offset, device=device):
            # [FUNCTION DOC] HELPER: Generates the dropout keep mask matching the efficient
            # attention kernel for the given shape, probability, seed and offset by filling a
            # float32 tensor and thresholding the uniform values (threshold zero on ROCm).
            # Returns the float32 keep mask.
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, dropout_p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = dropout_p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask

        def get_dropout_mask(output, fused_kernel, batch_size, n_heads, q_len, kv_len, dropout_p, device=device):
            # [FUNCTION DOC] HELPER: Returns the dropout keep mask of the fused op call. For
            # efficient attention it reads the seed and offset from the enclosing output tuple
            # and regenerates the mask; for flash attention it converts the debug mask in the
            # tuple to softmax layout using full padding masks and returns the boolean mask of
            # entries that are not dropped.
            if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
                output_seed, output_offset = output_tuple[2], output_tuple[3]
                output_seed = output_seed.item()
                output_offset = output_offset.item()
                return _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len,
                                              dropout_p, output_seed, output_offset, device=device)
            else:
                # Build dropout_mask
                dbug_mask = output_tuple[-1]
                query_padding_mask = torch.ones(
                    batch_size, seq_len_q, device=device, dtype=torch.bool)
                key_padding_mask = torch.ones(
                    batch_size, seq_len_k, device=device, dtype=torch.bool)

                softmax_mask = self.convert_flash_attn_S_to_softmax(
                    dbug_mask, seq_len_q, seq_len_k, query_padding_mask, key_padding_mask,
                    causal=is_causal)[:, :, :seq_len_q, :seq_len_k]
                dropout_mask = softmax_mask >= 0
                return dropout_mask

        if fused_kernel == SDPBackend.FLASH_ATTENTION and is_causal and seq_len_q != seq_len_k:
            self.skipTest("Flash V2 does not accept is_casual when seq_len_q != seq_len_k")

        seed = 42
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        fused_op = (torch.ops.aten._scaled_dot_product_efficient_attention
                    if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else torch.ops.aten._scaled_dot_product_flash_attention
                    if fused_kernel == SDPBackend.FLASH_ATTENTION else torch.ops.aten._scaled_dot_product_cudnn_attention)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # Set the global seed before capture
        torch.manual_seed(seed)
        kwargs = {"dropout_p": dropout_p, "is_causal": is_causal}
        if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
            kwargs["compute_log_sumexp"] = True
            kwargs["attn_bias"] = None
        if fused_kernel == SDPBackend.FLASH_ATTENTION:
            kwargs['return_debug_mask'] = dropout_p > 0.0
        if fused_kernel == SDPBackend.CUDNN_ATTENTION:
            kwargs["compute_log_sumexp"] = True
            kwargs["attn_bias"] = None
            if "return_debug_mask" in kwargs:
                kwargs.pop("return_debug_mask")
        with torch.cuda.stream(s):
            # Create real output
            output_tuple = fused_op(query, key, value, **kwargs)

        torch.cuda.current_stream().wait_stream(s)
        out = output_tuple[0]
        upstream_grad = torch.rand_like(out, requires_grad=False)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            out.backward(upstream_grad)
        for x in (query, key, value):
            x.grad = None
        g = torch.cuda.CUDAGraph()
        # Create real output
        with torch.cuda.graph(g):
            torch.rand_like(query, device=query.device)  # test non-zero intragraph offset
            # Create real output
            output_tuple = fused_op(query, key, value, **kwargs)
            if not all(not isinstance(o, torch.Tensor) or o.is_cuda for o in output_tuple):
                raise AssertionError("expected all tensor outputs to be on cuda")
        g.replay()
        out_first = output_tuple[0].clone()
        g.replay()
        out = output_tuple[0]
        if dropout_p == 0.0:
            self.assertEqual(out_first, out, atol=0, rtol=0)
        else:
            # replays produce different results
            self.assertNotEqual(out_first, out)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            if dropout_p == 0.0:
                # High Precision Math Reference
                with tf32_off():
                    out_ref = F.scaled_dot_product_attention(
                        query_ref, key_ref, value_ref,
                        dropout_p=dropout_p, is_causal=is_causal)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value,
                                                            dropout_p=dropout_p, is_causal=is_causal)
            # cuDNN attention doesn't support returning dropout mask
            elif fused_kernel != SDPBackend.CUDNN_ATTENTION:
                # Create the dropout_mask
                dropout_mask = get_dropout_mask(output_tuple, fused_kernel, batch_size,
                                                n_heads, seq_len_q, seq_len_k, dropout_p, device)
                # High Precision Math Reference
                with tf32_off():
                    out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                        query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal,
                        dropout_mask=dropout_mask)[0]
                # Low Precision Math Reference
                out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal,
                    dropout_mask=dropout_mask)[0]

        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        g1.replay()
        if fused_kernel != SDPBackend.CUDNN_ATTENTION or dropout_p == 0.0:
            grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
            with tf32_off():
                grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

            fudge_factors = {
                'out': 3.0,
                'grad_query': 110.0,
                'grad_key': 8.0,
                'grad_value': 3.0,
            }
            if TEST_WITH_ROCM:
                fudge_factors['out'] = 6.0
                fudge_factors['grad_value'] = 6.0
            check_out_and_grad(
                (out_ref, out_lp_ref, out),
                *zip(grads_ref, grads_ref_lp, grads),
                fudge_factors=fudge_factors
            )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_1_inputs(self, device, fused_kernel):
        # [FUNCTION DOC] Compares flash or efficient attention on nested half precision inputs
        # with some sequences of length one against a float32 math reference on contiguous
        # copies with atol 1e-3 and rtol 1e-2.
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 1
        num_ones = 10
        indices = torch.randint(low=0, high=batch, size=(num_ones,))
        seq_lens.scatter_(0, indices, 1)

        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(torch.float16), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    @parametrize("expand_q_batch", [True, False])
    @parametrize("expand_k_batch", [True, False])
    @parametrize("expand_v_batch", [True, False])
    @parametrize("expand_q_num_heads", [True, False])
    @parametrize("expand_k_num_heads", [True, False])
    @parametrize("expand_v_num_heads", [True, False])
    def test_fused_kernels_nested_broadcasting(
        self,
        device,
        kernel,
        expand_q_batch,
        expand_k_batch,
        expand_v_batch,
        expand_q_num_heads,
        expand_k_num_heads,
        expand_v_num_heads,
    ):
        # [FUNCTION DOC] Compares nested-tensor SDPA on flash or efficient attention (float16 or
        # float32) when any of query, key, value batch size or head count is broadcast from size
        # one, against the math backend run on explicitly expanded nested tensors with atol
        # 1.5e-3 and rtol 1e-2. Value head dim differs (32) for the efficient backend.
        is_efficient = kernel == SDPBackend.EFFICIENT_ATTENTION
        dtype = torch.float32 if is_efficient else torch.float16
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=dtype)
        batch, num_heads, head_dim = 32, 8, 64
        head_dim_v = 32 if is_efficient else head_dim
        seq_lens_q = (torch.randint(low=1, high=5, size=(1,)).item()
                      if expand_q_batch
                      else torch.randint(low=1, high=32, size=(batch,)).tolist())
        seq_lens_kv = (torch.randint(low=1, high=5, size=(1,)).item()
                       if (expand_k_batch or expand_v_batch)
                       else torch.randint(low=1, high=32, size=(batch,)).tolist())

        batch_q = 1 if expand_q_batch else batch
        batch_k = 1 if expand_k_batch else batch
        batch_v = 1 if expand_v_batch else batch

        # handle case where all batch_sizes are 1
        batch = max(batch_q, batch_k, batch_v)

        num_heads_q = 1 if expand_q_num_heads else num_heads
        num_heads_k = 1 if expand_k_num_heads else num_heads
        num_heads_v = 1 if expand_v_num_heads else num_heads

        # handle case where all num_heads are 1
        num_heads = max(num_heads_q, num_heads_k, num_heads_v)

        q_shape = SdpaShape(batch_q, num_heads_q, seq_lens_q, head_dim)
        k_shape = SdpaShape(batch_k, num_heads_k, seq_lens_kv, head_dim)
        v_shape = SdpaShape(batch_v, num_heads_v, seq_lens_kv, head_dim_v)

        query = rand_nested_tensor(q_shape)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        def _broadcast(t, batch_broadcasted, num_heads_broadcasted):
            # [FUNCTION DOC] HELPER: Expands a nested tensor with a singleton batch and/or
            # singleton head dimension to the full batch and head count, returning a float32
            # nested tensor; returns a float32 conversion if nothing needs expanding.
            if batch_broadcasted and num_heads_broadcasted:
                # (1, seq_len, 1, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor(
                    [t[0].expand(-1, num_heads, t.size(-1)) for _ in range(batch)], dtype=torch.float32)
            elif batch_broadcasted:
                # (1, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor([t[0] for _ in range(batch)], dtype=torch.float32)
            elif num_heads_broadcasted:
                # (batch, seq_len, 1, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor([x.expand(-1, num_heads, t.size(-1))
                                                    for x in t.unbind()], dtype=torch.float32)
            else:
                result = t.to(torch.float32)
            return result

        query_expanded = _broadcast(query, expand_q_batch, expand_q_num_heads).transpose(1, 2)
        key_expanded = _broadcast(key, expand_k_batch, expand_k_num_heads).transpose(1, 2)
        value_expanded = _broadcast(value, expand_v_batch, expand_v_num_heads).transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key_expanded.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1.5e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_query_dense(self, device):
        # [FUNCTION DOC] Efficient attention test with a dense one-sequence query, nested key
        # (batch 32, 16 heads) and nested value with a single head and head dim 96: compared
        # with the math backend on explicitly expanded nested tensors with atol 1e-3 and rtol
        # 1e-2.
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 96
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = (1, 1, num_heads, head_dim)
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # create a dense query
        query = torch.randn(q_shape, device=device, dtype=torch.float32)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        # (1, 1, num_heads, head_dim) -> (batch, 1, num_heads, head_dim)
        query_expanded = torch.nested.nested_tensor([query.squeeze(0) for _ in range(batch)]).transpose(1, 2)
        # (batch, seq_lens, 1, head_dim) -> (batch, seq_lens, num_heads, head_dim)
        value_expanded = torch.nested.nested_tensor(
            [t.expand(-1, num_heads, head_dim_v) for t in value.unbind()]).transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("batch_size", [8, 32])
    @parametrize("max_seq_len_q", [32, 256])
    @parametrize("max_seq_len_kv", [32, 256])
    @parametrize("head_dim", [8, 40, 64, 160])
    @parametrize("dropout_p", [0.0, 0.1])
    @parametrize("dtype", [torch.float16])
    @parametrize("scale", [None, "l1"])
    @parametrize("is_causal", [True, False])
    def test_flash_attention_vs_math_ref_grads_nestedtensor(self, device, batch_size: int, max_seq_len_q: int, max_seq_len_kv: int,
                                                            head_dim: int, dropout_p: float, dtype: torch.dtype,
                                                            scale: str, is_causal: bool):
        # [FUNCTION DOC] Compares flash attention on nested tensor inputs (half precision, with
        # or without dropout) against float32 and low precision math references for output and
        # gradients using fudge factors (scaled up for dropout). With dropout, nested dropout
        # masks are cut from the flash debug mask. Returns immediately for the causal case.
        # KNOWN ISSUE: in the causal case it only calls assertRaisesRegex without a callable or
        # context manager, so nothing is checked and the test returns.
        if is_causal:
            # TODO we should support this
            self.assertRaisesRegex(RuntimeError, "Nested tensors for query / key are not supported when is_causal=True")
            return
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        seq_lens_q = torch.randint(low=1, high=max_seq_len_q, size=(batch_size,))
        # Set one entry to max length
        seq_lens_q[torch.randint(0, batch_size, size=(1,))] = max_seq_len_q
        seq_lens_kv = torch.randint(low=1, high=max_seq_len_kv, size=(batch_size,))
        seq_lens_kv[torch.randint(0, batch_size, size=(1,))] = max_seq_len_kv

        def rand_nt(sequence_list, num_heads, head_dim):
            # [FUNCTION DOC] HELPER: Creates a nested tensor requiring grad, on the test device
            # and dtype, from random tensors of shape heads, length, head dim for each length in
            # the sequence list.
            tensors = [torch.rand((num_heads, seq_len, head_dim)) for seq_len in sequence_list]
            return torch.nested.nested_tensor(tensors, requires_grad=True, device=device, dtype=dtype)

        query = rand_nt(seq_lens_q, n_heads, head_dim)
        key = rand_nt(seq_lens_kv, n_heads, head_dim)
        value = rand_nt(seq_lens_kv, n_heads, head_dim)

        # Run the math kernel on low precision references
        query_ref_lp = query.detach().clone().requires_grad_(True)
        key_ref_lp = key.detach().clone().requires_grad_(True)
        value_ref_lp = value.detach().clone().requires_grad_(True)

        query_ref = query.detach().clone().to(torch.float32).requires_grad_(True)
        key_ref = key.detach().clone().to(torch.float32).requires_grad_(True)
        value_ref = value.detach().clone().to(torch.float32).requires_grad_(True)

        is_dropout = dropout_p > 0.0

        if not is_dropout:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                with tf32_off():
                    out_ref = F.scaled_dot_product_attention(
                        query_ref, key_ref, value_ref, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_ref_lp, key_ref_lp, value_ref_lp, is_causal=is_causal, scale=scale)
        else:
            # Create real output
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, return_debug_mask=is_dropout)
            out = output_tuple[0]
            dbug_mask = output_tuple[-1]

            query_padding_mask = torch.arange(max_seq_len_q).unsqueeze(0).expand(
                batch_size, max_seq_len_q
            ) < seq_lens_q.unsqueeze(-1)
            query_padding_mask = query_padding_mask.to("cuda")

            key_padding_mask = torch.arange(max_seq_len_kv).unsqueeze(0).expand(
                batch_size, max_seq_len_kv
            ) < seq_lens_kv.unsqueeze(-1)
            key_padding_mask = key_padding_mask.to("cuda")

            softmax_mask = self.convert_flash_attn_S_to_softmax(
                dbug_mask, max_seq_len_q, max_seq_len_kv, query_padding_mask, key_padding_mask, causal=is_causal)
            dropout_mask = softmax_mask >= 0
            nt_stack = []
            for tensor_component in range(batch_size):
                batch_stack = []
                for head in range(n_heads):
                    batch_stack.append(dropout_mask[tensor_component, head,
                                                    0:seq_lens_q[tensor_component],
                                                    0:seq_lens_kv[tensor_component]].unsqueeze(0))
                nt_stack.append(torch.cat(batch_stack))
            nested_dropout_mask = torch.nested.nested_tensor(nt_stack)
            # High Precision Math Reference
            with tf32_off():
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, dropout_mask=nested_dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=nested_dropout_mask)[0]

        upstream_grad = out.detach().clone().contiguous()

        out.backward(upstream_grad)
        with tf32_off():
            out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        dropout_fudge_factor = 1.0 if dropout_p == 0.0 else 2.0
        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            (query_ref, query_ref_lp, query),
            (key_ref, key_ref_lp, key),
            (value_ref, value_ref_lp, value),
            fudge_factors={
                'out': 1.5 * dropout_fudge_factor,
                'grad_query': 12.0 * dropout_fudge_factor,
                'grad_key': 1.5 * dropout_fudge_factor,
                'grad_value': 2.0 * dropout_fudge_factor,
            }
        )

class TestSDPAXpuOnly(NNTestCase):
    # [CLASS DOC] Test class for the XPU (Intel GPU) backends of SDPA, mostly migrated from the
    # CUDA-only class: backend choice, fused kernels with oneDNN attention, unsupported shapes,
    # layout preservation, GQA and mask comparisons against the math backend.
    """ Used to test XPU only functionality of scaled_dot_product_attention
    Mostly migrate from TestSDPACudaOnly in test/test_transformers.py
    """


    @parametrize("type", ["dense"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_xpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # [FUNCTION DOC] Checks the XPU backend choice for dense inputs: the math backend is
        # chosen when dropout is positive or the dtype is float64, otherwise the overrideable
        # (oneDNN) backend. Raises AssertionError on mismatch.
        # Migrate from test_fused_sdp_choice_cpu
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if dropout > 0.0 or dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            if torch._fused_sdp_choice(q, k, v, dropout_p=dropout) != SDPBackend.MATH.value:
                raise AssertionError("expected MATH backend")
        else:
            if torch._fused_sdp_choice(q, k, v, dropout_p=dropout) != SDPBackend.OVERRIDEABLE.value:
                raise AssertionError("expected OVERRIDEABLE backend")

    def test_fused_inputs_dim_3_xpu(self, device):
        # [FUNCTION DOC] Checks that unbatched 3D inputs on XPU with no mask, 2D mask or 3D mask
        # select the overrideable backend and match batched 4D inputs with a leading dimension
        # added and removed.
        size = (2, 3, 8)
        q = torch.randn(size, device=device)
        k = torch.randn(size, device=device)
        v = torch.randn(size, device=device)
        masks = (
            None,
            torch.randn(3, 3, device=device),
            torch.randn(2, 3, 3, device=device),
        )

        for attn_mask in masks:
            normalized_mask = attn_mask
            while normalized_mask is not None and normalized_mask.dim() < 4:
                normalized_mask = normalized_mask.unsqueeze(0)
            with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
                self.assertEqual(
                    torch._fused_sdp_choice(q, k, v, attn_mask),
                    SDPBackend.OVERRIDEABLE.value,
                )
                expected = F.scaled_dot_product_attention(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    normalized_mask,
                ).squeeze(0)
                actual = F.scaled_dot_product_attention(q, k, v, attn_mask)

            self.assertEqual(actual, expected)

    def test_backends_set_to_math(self, device):
        # [FUNCTION DOC] Checks that restricting the backend list to math enables math SDP and
        # disables overrideable SDP in the global flags, and that SDPA still runs on bfloat16
        # inputs.
        dtype = torch.bfloat16
        q_shape = SdpaShape(1, 1, 8, 16)
        kv_shape = SdpaShape(1, 1, 12, 16)
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        q, k, v = make_q(), make_kv(), make_kv()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            self.assertTrue(torch._C._get_math_sdp_enabled())
            self.assertFalse(torch._C._get_overrideable_sdp_enabled())
            _ = F.scaled_dot_product_attention(q, k, v)

    def test_default_priority_order(self, device):
        # [FUNCTION DOC] Checks the default XPU backend priority order: runs a backend choice to
        # initialize it, reads the current ordered backend list, and requires overrideable to
        # precede flash and flash to precede math.
        # The default priority order of xpu is overridable, math, flash, efficient, cudnn
        # For xpu backend, we need to make sure that overridable > flash > math
        dtype = torch.bfloat16
        shape = SdpaShape(1, 1, 1, 1)
        make_tensor = partial(torch.rand, shape, device=device, dtype=dtype)
        t = make_tensor()
        # run sdp_choice to make sure priority_order is set by XPU default priority_order
        torch._fused_sdp_choice(t, t, t)
        from torch.nn.attention import _cur_sdpa_kernel_backends
        default_priority = _cur_sdpa_kernel_backends(with_priority=True)
        flash_index = default_priority.index(SDPBackend.FLASH_ATTENTION)
        overrideable_index = default_priority.index(SDPBackend.OVERRIDEABLE)
        math_index = default_priority.index(SDPBackend.MATH)
        self.assertTrue(overrideable_index < flash_index < math_index,
                        lambda msg: f"{msg}\nExpected overrideable < flash < math, got {overrideable_index}, {flash_index}, {math_index}")

    def test_onednn_attention_different_dk_dv(self, device):
        # [FUNCTION DOC] XPU test: bfloat16 inputs with query/key head dim 128 and value head
        # dim 64, query length 1 and key length 2, overrideable backend vs math backend with
        # atol 1e-3 and rtol 1e-2.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        q_shape = SdpaShape(batch, num_heads, 1, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, 2, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, 2, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel([SDPBackend.OVERRIDEABLE]):
            actual = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdpa_kernel([SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    def test_onednn_attention_fail_d576(self, device):
        # [FUNCTION DOC] XPU test: head dims of 1024 exceed the oneDNN limit, so the
        # overrideable backend must raise RuntimeError reporting no available kernel.
        # Test that onednn graph attention dispatching correctly bails out on d > 576
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 1024, 1024

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                _ = F.scaled_dot_product_attention(q, k, v)

    def test_onednn_attention_broadcasted_input(self, device):
        # [FUNCTION DOC] XPU test: an attention mask of shape one by seqlen expanded to a four
        # dimensional broadcast view must give the same result on the overrideable backend
        # (which should fall back for the unsupported case) as on the math backend.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, seqlen, head_dim = 32, 16, 128, 32
        q_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        k_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        v_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        attn_mask_shape = (1, seqlen)
        attn_mask = make_tensor(attn_mask_shape)
        attn_mask = attn_mask.expand(1, 1, seqlen, seqlen)

        # test that we do not dispatch to onednn for an unsupported case
        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    def test_onednn_attention_preserves_query_layout(self, device):
        # [FUNCTION DOC] XPU test: for every permutation of batch, heads and sequence dimensions
        # the output of the overrideable backend on bfloat16 inputs must be contiguous in the
        # same permuted order as the inputs.

        def test_attention(permute_order: list[list[int]]):
            # [FUNCTION DOC] HELPER: Builds bfloat16 query/key/value views of logical shape
            # batch 4, heads 16, sequence 256 or 512, head dim 64 laid out in memory following
            # the given permutation, runs SDPA on the overrideable backend, and asserts the
            # output keeps the permuted contiguous layout.
            BHSqD = [4, 16, 256, 64]
            BHSkvD = [4, 16, 512, 64]

            shape_q = [BHSqD[idx] for idx in permute_order]
            shape_kv = [BHSkvD[idx] for idx in permute_order]
            reverse = [permute_order.index(idx) for idx in range(4)]
            q = torch.randn(*shape_q, dtype=torch.bfloat16, device=device, requires_grad=False).permute(reverse)
            k = torch.randn(*shape_kv, dtype=torch.bfloat16, device=device, requires_grad=False).permute(reverse)
            v = torch.randn(*shape_kv, dtype=torch.bfloat16, device=device, requires_grad=False).permute(reverse)
            self.assertEqual(q.shape, BHSqD)
            self.assertEqual(k.shape, BHSkvD)
            self.assertEqual(v.shape, BHSkvD)

            with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
                out = F.scaled_dot_product_attention(q, k, v)
            self.assertTrue(out.permute(permute_order).is_contiguous())

        permutable = [0, 1, 2]
        permute_orders = itertools.permutations(permutable)

        for permute_order in permute_orders:
            test_attention(list(permute_order) + [3])

    def test_onednn_attention_fused_kernels_safe_softmax(self, device):
        # [FUNCTION DOC] XPU test: with an all negative infinity float mask (every row fully
        # masked) the overrideable backend must return the same result as the math backend (safe
        # softmax behavior).
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, seqlen, head_dim = 32, 16, 32, 64
        q_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        k_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        v_shape = SdpaShape(batch, num_heads, seqlen, head_dim)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        attn_mask = torch.full((seqlen, seqlen), float('-inf'), device=device, dtype=torch.bfloat16)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    @parametrize("type", ["dense"])
    @parametrize("is_contiguous", [True, False])
    def test_onednn_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        # [FUNCTION DOC] XPU test: packed qkv half precision inputs (contiguous or transposed
        # views) on the overrideable backend must match the math backend on contiguous copies
        # with atol 2e-3 and rtol 1e-2.
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @parametrize("dtype", [torch.half, torch.bfloat16])
    @parametrize("batch_size,n_head,n_head_kv,q_size,kv_size,head_dim", [
        (2, 64, 16, 9216, 77, 64),
        (2, 32, 4, 2304, 2304, 64),
        (2, 32, 2, 2304, 77, 64),
        (2, 20, 2, 576, 576, 64),
        (2, 20, 2, 576, 77, 64),
        (2, 20, 2, 144, 144, 64),
        (2, 20, 2, 144, 77, 64),
        (1, 32, 2, 1, 32, 128),
        (4, 32, 4, 1, 32, 128),
        (1, 32, 2, 32, 32, 128),
        (4, 32, 4, 32, 32, 128),
        (1, 32, 2, 2016, 2016, 128),
        (4, 32, 4, 2016, 2016, 128),
    ])
    @parametrize("is_causal", [True, False])
    def test_onednn_attention_gqa_vs_math(self, device, dtype, batch_size, n_head, n_head_kv, q_size, kv_size, head_dim, is_causal):
        # [FUNCTION DOC] XPU test: grouped query attention with several shapes, causal or not,
        # in half and bfloat16; overrideable backend result must match the float32 math backend
        # result within dtype tolerances.
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, n_head, q_size, head_dim)
        k_shape = SdpaShape(batch_size, n_head_kv, kv_size, head_dim)
        v_shape = SdpaShape(batch_size, n_head_kv, kv_size, head_dim)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal, enable_gqa=True)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), attn_mask=None, dropout_p=0.0, is_causal=is_causal, enable_gqa=True)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=tol.atol, rtol=tol.rtol)

    @parametrize("fused_kernel", [SDPBackend.OVERRIDEABLE])
    @parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
    @parametrize("batch_size,n_head,q_size,kv_size,head_dim", [
        (2, 5, 9216, 9216, 64),
        (2, 5, 9216, 77, 64),
        (2, 10, 2304, 2304, 64),
        (2, 10, 2304, 77, 64),
        (2, 20, 576, 576, 64),
        (2, 20, 576, 77, 64),
        (2, 20, 144, 144, 64),
        (2, 20, 144, 77, 64),
        (1, 32, 1, 32, 128),
        (4, 32, 1, 32, 128),
        (1, 32, 32, 32, 128),
        (4, 32, 32, 32, 128),
        (1, 32, 2016, 2016, 128),
        (4, 32, 2016, 2016, 128),
    ])
    @parametrize("mask_type", ["float", "causal"])
    @parametrize("train", [False])
    def test_onednn_attention_mask_vs_math(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_size,
        kv_size,
        n_head,
        head_dim,
        mask_type,
        train,
    ):
        # [FUNCTION DOC] XPU test (forward only): compares the overrideable fused op (or math
        # op) with a float32 math reference for a float or causal mask (a boolean mask branch
        # exists but is not parametrized) over several shapes and dtypes; the result converted
        # to float32 must match within dtype tolerances.
        # Migrate from TestSDPACpuOnly
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        mask_shape = [batch_size, 1, 1, kv_size]
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, n_head, q_size, head_dim)
        kv_shape = SdpaShape(batch_size, n_head, kv_size, head_dim)
        q = make_tensor(q_shape)
        k = make_tensor(kv_shape)
        v = make_tensor(kv_shape)
        q2, k2, v2 = q.clone(), k.clone(), v.clone()

        if train:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            q2.requires_grad_(True)
            k2.requires_grad_(True)
            v2.requires_grad_(True)

        # (B, nh, T, hs)
        q = q.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask = None
        is_causal = False
        if mask_type == "bool":
            attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
        elif mask_type == "float":
            attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)
        elif mask_type == "causal":
            is_causal = True

        q2, k2, v2 = q2.float(), k2.float(), v2.float()
        q2 = q2.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask2 = attn_mask.float() if attn_mask is not None else None

        if fused_kernel == SDPBackend.MATH:
            actual = torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)[0]
        elif fused_kernel == SDPBackend.OVERRIDEABLE:
            actual = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                q, k, v, attn_bias=attn_mask, dropout_p=0.0, is_causal=is_causal)[0]

        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q2, k2, v2, attn_mask=attn_mask2, dropout_p=0.0, is_causal=is_causal)[0]

        self.assertEqual(actual.float(), math_ref, atol=tol.atol, rtol=tol.rtol)

    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    @parametrize("dtype", [torch.float32, torch.float64])
    def test_flash_attention_unsupport_dtypes(self, device, dtype):
        # [FUNCTION DOC] XPU test: the flash attention backend does not support float32 or
        # float64 inputs and must raise RuntimeError reporting no available kernel.
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, seqlen, head_dim = 32, 16, 32, 64
        q_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        k_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        v_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        q, k, v = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                F.scaled_dot_product_attention(q, k, v)

    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    def test_flash_attention_unsupport_dropout(self, device):
        # [FUNCTION DOC] XPU test: flash attention with nonzero dropout is unsupported and must
        # raise RuntimeError reporting no available kernel.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, seqlen, head_dim = 32, 16, 32, 64
        q_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        k_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        v_shape = SdpaShape(batch, seqlen, num_heads, head_dim)
        q, k, v = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    def test_flash_attention_headdim_size(self, device):
        # [FUNCTION DOC] XPU test: flash attention accepts head dim 256 but head dim 257 must
        # raise RuntimeError reporting no available kernel.
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
        batch, num_heads, seqlen = 32, 2, 32

        max_supported_head_dim = 256
        q_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim)
        k_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim)
        v_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim)
        q, k, v = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(q, k, v)

        q_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim + 1)
        k_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim + 1)
        v_shape = SdpaShape(batch, seqlen, num_heads, max_supported_head_dim + 1)
        q, k, v = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                F.scaled_dot_product_attention(q, k, v)

    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    def test_flash_attention_fail_with_non_square_causal_attention(self, device):
        # [FUNCTION DOC] XPU test: causal flash attention with unequal query and key lengths
        # must warn that the combination is unsupported on XPU and raise RuntimeError.
        dtype = torch.bfloat16
        q_shape = SdpaShape(1, 1, 8, 16)
        kv_shape = SdpaShape(1, 1, 12, 16)
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        q, k, v = make_q(), make_kv(), make_kv()
        warning_str = "Flash attention XPU does not support the is_causal flag when seqlen_q != seqlen_k."
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, warning_str):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, is_causal=True))

    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    @parametrize("dtype", [torch.half, torch.bfloat16])
    @parametrize("batch_size", [1, 2, 4])
    @parametrize("n_head", [[3, 1], [4, 2], [10, 2]])
    @parametrize("q_size", [1, 32, 77, 128, 144, 512, 576])
    @parametrize("kv_size", [1, 32, 77, 128, 144, 512, 576])
    @parametrize("head_dim", [64, 96, 128, 160, 192])
    @parametrize("mask_type", [None, "causal"])
    @parametrize("train", [True, False])
    @parametrize("layout", ["bshd", "bhsd"])
    @parametrize("enable_gqa", [True, False])
    def test_flash_attention_vs_math(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_size,
        kv_size,
        n_head,
        head_dim,
        mask_type,
        train,
        layout,
        enable_gqa,
    ):
        # [FUNCTION DOC] XPU test comparing flash attention with the float32 math backend over
        # batch, GQA head counts, sequence lengths, head dims, causal or none, training or not,
        # and two memory layouts. A causal case is compared with a bottom-right aligned causal
        # mask; outputs and (if training) query, key, value gradients of the mean must match
        # within dtype tolerances. Skips causal with unequal lengths.
        if mask_type == "causal" and q_size != kv_size:
            self.skipTest("Flash Attention V2 does not accept is_causal when seq_len_q != seq_len_k")

        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)

        if enable_gqa:
            n_head_q, n_head_kv = n_head[0], n_head[1]
        else:
            n_head_q = n_head_kv = n_head[0]

        q_shape = SdpaShape(batch_size, n_head_q, q_size, head_dim)
        kv_shape = SdpaShape(batch_size, n_head_kv, kv_size, head_dim)
        q = make_tensor(q_shape)
        k = make_tensor(kv_shape)
        v = make_tensor(kv_shape)

        # (B, S, H, D) by default
        q = q.view(batch_size, q_size, n_head_q, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_size, n_head_kv, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_size, n_head_kv, head_dim).transpose(1, 2)
        if layout == "bhsd":
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        is_causal = False
        if mask_type == "causal":
            is_causal = True

        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2, k2, v2 = q2.float(), k2.float(), v2.float()

        if train:
            q = q.detach().clone().requires_grad_(True)
            k = k.detach().clone().requires_grad_(True)
            v = v.detach().clone().requires_grad_(True)
            q2 = q2.detach().clone().requires_grad_(True)
            k2 = k2.detach().clone().requires_grad_(True)
            v2 = v2.detach().clone().requires_grad_(True)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal, enable_gqa=enable_gqa)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            if is_causal:
                bottom_right_mask = causal_lower_right(q_size, kv_size)
                math_ref = F.scaled_dot_product_attention(
                    q2, k2, v2, dropout_p=0.0, attn_mask=bottom_right_mask, enable_gqa=enable_gqa)
            else:
                math_ref = F.scaled_dot_product_attention(
                    q2, k2, v2, dropout_p=0.0, is_causal=is_causal, enable_gqa=enable_gqa)

        if dtype in [torch.float16, torch.bfloat16]:
            math_ref = math_ref.to(dtype)

        self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)

        if train:
            loss = torch.mean(actual)
            loss_ref = torch.mean(math_ref)
            loss.backward()
            loss_ref.backward()

            grad_q_actual, grad_k_actual, grad_v_actual = q.grad, k.grad, v.grad
            grad_q_ref, grad_k_ref, grad_v_ref = q2.grad, k2.grad, v2.grad
            if dtype in [torch.float16, torch.bfloat16]:
                grad_q_ref = grad_q_ref.to(dtype)
                grad_k_ref = grad_k_ref.to(dtype)
                grad_v_ref = grad_v_ref.to(dtype)

            self.assertEqual(grad_q_actual, grad_q_ref, atol=tol.atol, rtol=tol.rtol)
            self.assertEqual(grad_k_actual, grad_k_ref, atol=tol.atol, rtol=tol.rtol)
            self.assertEqual(grad_v_actual, grad_v_ref, atol=tol.atol, rtol=tol.rtol)

class TestAttnBias(NNTestCase):
    # [CLASS DOC] Test class for attention bias helpers (causal upper-left and lower-right
    # variants): checks equivalence with explicit masks in eager and compiled mode, equivalence
    # of the causal flag with the upper-left bias, and the error when both are given.

    def run_test(
        self,
        device,
        make_q,
        make_kv,
        attn_bias=None,
        forw_tolerances: Tolerances | None = None,
        grad_tolerances: Tolerances | None = None,
        backend=None,
        causal_variant=None,
    ):
        # [FUNCTION DOC] HELPER: Shared checker. Builds query, key, value and clones,
        # materializes the attention bias into a dense mask for a reference SDPA call, runs SDPA
        # with the bias object (compiled with the given backend if any, resetting dynamo first),
        # backpropagates the same random gradient through both and asserts output and
        # query/key/value gradients are close using the optional tolerances (dtype defaults
        # otherwise).
        if backend is not None:
            torch._dynamo.reset()

        query, key, value = make_q(), make_kv(), make_kv()
        query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)

        realized = attn_bias._materialize(device) if attn_bias is not None else None
        pytorch_output = scaled_dot_product_attention(
            query, key, value, attn_mask=realized, dropout_p=0.0, is_causal=False
        )

        sdpa_op = (
            torch.compile(scaled_dot_product_attention, backend=backend)
            if backend is not None
            else scaled_dot_product_attention
        )
        sdpa_output = sdpa_op(
            query_prototype,
            key_prototype,
            value_prototype,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )

        dOut = torch.randn_like(pytorch_output)
        pytorch_output.backward(dOut)
        sdpa_output.backward(dOut)

        # Use default assert_close tolerances for dtypes
        if forw_tolerances is None:
            forw_tolerances = Tolerances(atol=None, rtol=None)
        if grad_tolerances is None:
            grad_tolerances = Tolerances(atol=None, rtol=None)

        torch.testing.assert_close(pytorch_output, sdpa_output, rtol=forw_tolerances.rtol, atol=forw_tolerances.atol)
        torch.testing.assert_close(query.grad, query_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        torch.testing.assert_close(key.grad, key_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        torch.testing.assert_close(value.grad, value_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)

    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    def test_causal_variants(self, device, causal_variant: CausalVariant, shape: list[tuple[int]]):
        # [FUNCTION DOC] Checks that SDPA with an upper-left or lower-right causal bias object
        # gives the same output and gradients as SDPA with the equivalent dense mask, for
        # several shapes in float16 with several backends enabled. Skips lower-right with more
        # queries than keys (NaN outputs).
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )
        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

        forw_tol = Tolerances(1e-3, 1e-3)
        grad_tol = Tolerances(5e-3, 5e-3)

        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION,
                                   SDPBackend.FLASH_ATTENTION,
                                   SDPBackend.MATH,
                                   SDPBackend.CUDNN_ATTENTION]):
            self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=None)

    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    @skipIfTorchDynamo("This function already calls torch.compile.")
    @skipXPUIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "XPU Flash Attention is not supported")
    def test_causal_variants_compile(self, device, causal_variant: CausalVariant, shape: list[tuple[int]]):
        # [FUNCTION DOC] Like the causal variants test but with the SDPA call compiled by the
        # aot_eager counting backend; additionally requires exactly one compiled frame. Skipped
        # under dynamo.
        cnts = CompileCounterWithBackend("aot_eager")
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )
        forw_tol = Tolerances(1e-3, 1e-3)
        grad_tol = Tolerances(5e-3, 5e-3)

        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION,
                                   SDPBackend.FLASH_ATTENTION,
                                   SDPBackend.MATH,
                                   SDPBackend.CUDNN_ATTENTION]):
            self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=cnts)
        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")

    @parametrize("shape", [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)])
    def test_is_causal_equals_upper_left(self, device, shape: list[tuple[int]]):
        # [FUNCTION DOC] Checks that SDPA with an upper-left causal bias object produces the
        # same output as the causal flag for float16 inputs of several shapes (tolerance 1e-3).
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))

        forw_tol = Tolerances(1e-3, 1e-3)

        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()
        attn_bias = causal_upper_left(seq_len_q, seq_len_kv)

        out_attn_bias = scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=0.0)
        out_is_causal = scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
        torch.testing.assert_close(out_attn_bias, out_is_causal, rtol=forw_tol.rtol, atol=forw_tol.atol)

    def test_is_causal_and_mask_fails(self, device):
        # [FUNCTION DOC] Checks that passing a causal bias object together with the causal flag
        # raises ValueError saying the causal bias should not be used with the causal flag.
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )
        make_q_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))
        make_kv_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))

        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()
        attn_bias = causal_upper_left(128, 128)

        with self.assertRaisesRegex(ValueError, "CausalBias should not be used with causal=True"):
            scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, is_causal=True, dropout_p=0.0)

if NOTEST_CPU:
    device_types = ("cuda", "mps", "mtia")
else:
    device_types = ("cpu", "cuda", "mps", "mtia")

if TEST_XPU:
    device_types += ("xpu", )

instantiate_device_type_tests(TestTransformers, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPAFailureModes, globals(), only_for=device_types, allow_mps=True)
instantiate_device_type_tests(TestSDPA, globals(), only_for=device_types, allow_mps=True, allow_xpu=True)
instantiate_device_type_tests(TestSDPACudaOnly, globals(), only_for=("cuda"))
instantiate_device_type_tests(TestSDPACpuOnly, globals(), only_for=("cpu"))
instantiate_device_type_tests(TestAttnBias, globals(), only_for=device_types, allow_xpu=True)
instantiate_device_type_tests(TestSDPAXpuOnly, globals(), only_for="xpu", allow_xpu=True)

if __name__ == '__main__':
    run_tests()
