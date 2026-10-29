# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
# [MODULE DOC]
# TorchScript exporter symbolic functions for ONNX opset 17: layer_norm (LayerNormalization),
# its quantized variant, and stft (STFT op with window preparation, batching and
# normalization), with one private helper for window padding sizes. Import-time side effects:
# registers the handlers in the global symbolic registry for opset 17.
"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

import functools
from collections.abc import Sequence

import torch
from torch import _C
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = ["layer_norm", "stft", "quantized_layer_norm"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
):
    # [FUNCTION DOC]
    # Symbolic for layer_norm via LayerNormalization. The normalization axis is the negative
    # length of normalized_shape. When weight or bias is none, substitutes a constant of ones or
    # zeros of normalized_shape in the input dtype (float if unknown). Emits the op with epsilon
    # and axis attributes; cudnn_enable is ignored.
    # normalized_shape: input shape from an expected input of size
    # axis: The first normalization dimension.
    # layer_norm normalizes on the last D dimensions,
    # where D is the size of normalized_shape
    axis = -len(normalized_shape)
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    dtype = scalar_type.dtype()
    if symbolic_helper._is_none(weight):
        weight_value = torch.ones(normalized_shape, dtype=dtype)
        weight = g.op("Constant", value_t=weight_value)
    if symbolic_helper._is_none(bias):
        bias_value = torch.zeros(normalized_shape, dtype=dtype)
        bias = g.op("Constant", value_t=bias_value)
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )


@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # [FUNCTION DOC]
    # Symbolic for quantized layer_norm: dequantizes x, applies layer_norm, then requantizes with
    # the given scale and zero point. Returns the quantized value.
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


def _compute_edge_sizes(n_fft, window_size):
    # [FUNCTION DOC] HELPER: Returns the left and right zero padding sizes that center a window of
    # window_size inside n_fft; left is half the difference rounded down, right is the remainder.
    """Helper function to compute the sizes of the edges (left and right)
    of a given window centered within an FFT size."""
    left = (n_fft - window_size) // 2
    right = n_fft - left - window_size
    return left, right


@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b", "b")
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _C.Value | None = None,
    normalized: bool = False,
    onesided: bool | None = True,
    return_complex: bool | None = False,
    align_to_window: bool | None = None,
) -> _C.Value:
    # [FUNCTION DOC]
    # Symbolic for stft via the STFT op. Raises a symbolic value error for complex output, an
    # align_to_window option, a signal rank that is unknown or above 2, or a window longer than
    # n_fft; raises an assertion error when a given window size mismatches win_length or n_fft.
    # A rank 1 signal gets a batch dim added and removed afterwards. A given window shorter than
    # n_fft is zero padded on both sides; a missing window is built as a constant (rectangular or
    # centered ones of win_length). The window is cast to the signal type, the result is
    # transposed to match torch layout, and divided by the square root of n_fft if normalized.
    # Hop length defaults to a quarter of n_fft. Returns the result value.
    """Associates `torch.stft` with the `STFT` ONNX operator.
    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Args:
        g: Graph to write the ONNX representation into
        input: Input tensor for the transformation
        n_fft: FFT size
        hop_length: Size of the hop. Defaults to `floor(n_fft // 4)`
        win_length: Size of the analysis window. Defaults to `n_fft`
        window: Analysis window. Defaults to a window of all ones
        normalized: Whether to return a normalized STFT
        onesided: Whether to return only half (+1) of the results, given the
            symmetry of the STFT
        return_complex: Whether to return the complex value (Note: Must be
            `False` or `None`)

    Returns:
        op: Operator for torch.stft associated with STFT (ONNX)
    """
    # Checks
    if return_complex:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support complex types", value=input
        )

    if align_to_window is not None:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support the align_to_window option",
            value=input,
        )  # TODO(#145944): add compatibility with align_to_window option.

    # Get STFT sizes
    frame_step_value = hop_length if hop_length is not None else n_fft // 4
    frame_step_const = g.op(
        "Constant", value_t=torch.tensor(frame_step_value, dtype=torch.int64)
    )
    frame_length_const = g.op(
        "Constant", value_t=torch.tensor(n_fft, dtype=torch.int64)
    )

    # Pre-process input if needed
    signal = input
    signal_rank = symbolic_helper._get_tensor_rank(signal)
    if signal_rank == 1:
        # Add batch dimension
        signal = g.op(
            "Unsqueeze",
            signal,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
    elif signal_rank is None or signal_rank > 2:
        raise errors.SymbolicValueError(
            msg="STFT can only take inputs of 1 [signal] or 2 [batch, signal] dimensions. "
            f"Current rank of signal is {signal_rank}, please reduce it.",
            value=input,
        )

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    # pyrefly: ignore [bad-argument-type]
    n_win = symbolic_helper._get_tensor_dim_size(window, dim=0)
    if n_win is not None:
        win_length_default = win_length if win_length else n_fft
        if n_win != win_length_default:
            raise AssertionError(
                "Analysis window size must equal `win_length` or `n_fft`. "
                f"Please, set `win_length` or `n_fft` to match `window` size ({n_win})"
            )

        # Center window around zeros if needed (required by ONNX's STFT)
        if n_win < n_fft:
            left, right = _compute_edge_sizes(n_fft, n_win)
            left_win = g.op("Constant", value_t=torch.zeros(left))
            right_win = g.op("Constant", value_t=torch.zeros(right))
            # pyrefly: ignore [bad-argument-type]
            window = g.op("Concat", left_win, window, right_win, axis_i=0)

    # Create window, if needed
    if symbolic_helper._is_none(window):
        if win_length:
            if win_length > n_fft:
                raise errors.SymbolicValueError(
                    msg="The analysis window can't be longer than the size of the FFT. "
                    f"Please set `win_length` ({win_length}) to `n_fft` ({n_fft}) or less.",
                    value=input,
                )

            # Center window, if needed
            left, right = _compute_edge_sizes(n_fft, win_length)
            torch_window = torch.hstack(
                (torch.zeros(left), torch.ones(win_length), torch.zeros(right))
            )
        else:
            # Rectangle window
            torch_window = torch.ones(n_fft)
        if torch_window.shape[0] != n_fft:
            raise AssertionError(
                f"torch_window.shape[0]={torch_window.shape[0]} != n_fft={n_fft}"
            )
        window = g.op("Constant", value_t=torch_window)
    window = g.op(
        "Cast",
        # pyrefly: ignore [bad-argument-type]
        window,
        to_i=_type_utils.JitScalarType.from_value(signal).onnx_type(),
    )

    # Run STFT
    result = g.op(
        "STFT",
        signal,
        frame_step_const,
        window,
        frame_length_const,
        onesided_i=1 if onesided is None or onesided else 0,
    )

    # Transpose to mimic torch.stft's behavior
    result = g.op("Transpose", result, perm_i=[0, 2, 1, 3])

    # Remove batch dimension, if needed
    if signal_rank == 1:
        result = g.op(
            "Squeeze",
            result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # Normalize, if needed
    if normalized:
        sqrt_nfft = torch.sqrt(torch.tensor(n_fft, dtype=signal.type().dtype()))
        result = g.op("Div", result, g.op("Constant", value_t=sqrt_nfft))

    return result
