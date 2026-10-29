# mypy: allow-untyped-defs
# [MODULE DOC] Defines the 'quantized_decomposed' custom operator library: decomposed
# (functional) quantize, dequantize, choose-qparams and fake-quant ops whose
# quantization parameters are passed as explicit arguments instead of being stored on a
# quantized tensor. Each op is declared with a schema on the library and given a
# reference implementation (CompositeExplicitAutograd, or Autograd for
# fake_quant_per_channel) plus a Meta implementation that only computes output shape and
# dtype. Variants: per tensor (scalar, tensor and tensor2 parameter forms), per channel,
# per token, per channel group, plus a no-fuse dtype conversion. Import-time side
# effects: creates the Library named quantized_decomposed, builds the dtype to (min,
# max) bounds table, and registers all schemas and kernels globally.
import math

import torch
from torch._refs import _unsqueeze_multiple
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from torch.library import impl, Library


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_INTEGER_DTYPES = [torch.uint8, torch.int8, torch.uint16, torch.int16, torch.int32]
_FLOAT_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]

_DTYPE_TO_QVALUE_BOUNDS = {
    k: (torch.iinfo(k).min, torch.iinfo(k).max) for k in _INTEGER_DTYPES
}
_DTYPE_TO_QVALUE_BOUNDS.update(
    {k: (int(torch.finfo(k).min), int(torch.finfo(k).max)) for k in _FLOAT_DTYPES}
)


# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    # [FUNCTION DOC] HELPER: Validates that quant_min and quant_max fit inside the
    # representable range of dtype (table lookup). Raises ValueError for an unsupported
    # dtype and AssertionError if quant_min is below or quant_max above the dtype
    # bounds. Returns nothing.
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    if quant_min < quant_min_lower_bound:
        raise AssertionError(
            "quant_min out of bounds for dtype, "
            f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
        )

    if quant_max > quant_max_upper_bound:
        raise AssertionError(
            "quant_max out of bounds for dtype, "
            f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
        )


quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Reference per-tensor affine quantization. Casts float16/bfloat16
    # input to float32, requires float32 (AssertionError otherwise), checks the quant
    # range against dtype, then computes round(input / scale) + zero_point, clamps to
    # [quant_min, quant_max] and casts to dtype. Returns the quantized tensor; qparams
    # are not stored on it.
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for quantize_per_tensor: same input dtype checks
    # (float16/bfloat16 upcast, then must be float32, else AssertionError); returns an
    # empty tensor with input's shape and the requested dtype. No range check on
    # quant_min/quant_max.
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Overload of quantize_per_tensor taking scale and zero_point as
    # one-element tensors. Raises AssertionError if either does not have exactly one
    # element, then extracts Python scalars and delegates to quantize_per_tensor.
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for the tensor overload: upcasts float16/bfloat16
    # input, requires scale and zero_point to have one element and input to be float32
    # (AssertionError otherwise), returns an empty tensor of input shape with dtype.
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    return torch.empty_like(input, dtype=dtype)


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Overload of quantize_per_tensor where scale, zero_point, quant_min
    # and quant_max are all one-element tensors. Raises AssertionError if scale or
    # zero_point is not one element (quant_min/quant_max are not checked here), converts
    # all to scalars and delegates to quantize_per_tensor.
    # KNOWN ISSUE: the docstring is copied from the tensor overload and does not mention
    # quant_min/quant_max being tensors.
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for the tensor2 overload; simply forwards all arguments
    # to quantize_per_tensor_tensor_meta.
    return quantize_per_tensor_tensor_meta(
        input,
        scale,
        zero_point,  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Reference per-tensor affine dequantization. Requires input.dtype to
    # equal dtype (AssertionError otherwise); out_dtype defaults to float32. If dtype is
    # a supported quantized dtype returns (input cast to out_dtype minus zero_point)
    # times scale; otherwise raises ValueError. quant_min and quant_max are unused (kept
    # as metadata for pattern matching).
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(out_dtype) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for dequantize_per_tensor: returns an empty tensor with
    # input's shape and out_dtype (default float32). Performs no validation.
    # KNOWN ISSUE: scale and zero_point are annotated as Tensor although the op schema
    # takes float and int.
    if out_dtype is None:
        out_dtype = torch.float32
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Overload of dequantize_per_tensor with one-element tensor scale and
    # zero_point. Raises AssertionError if either is not a single element; extracts
    # scalars and delegates to dequantize_per_tensor, passing out_dtype through.
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,
        quant_max,
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for the tensor overload: out_dtype defaults to float32;
    # raises AssertionError if zero_point or scale is not one element or input.dtype
    # differs from dtype; raises ValueError for unsupported dtype; returns an empty
    # tensor of input shape and out_dtype.
    if out_dtype is None:
        out_dtype = torch.float32
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor2",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Overload of dequantize_per_tensor where scale, zero_point,
    # quant_min and quant_max are tensors. Checks scale and zero_point have one element
    # (AssertionError), converts all to scalars and delegates to dequantize_per_tensor
    # with out_dtype.
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for the tensor2 dequantize overload; forwards to
    # dequantize_per_tensor_tensor_meta including out_dtype.
    return dequantize_per_tensor_tensor_meta(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype
    )


quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Derives per-tensor affine qparams from input. Requires
    # float32/float16/bfloat16 input and a dtype with known bounds (AssertionError
    # otherwise), validates qmin/qmax, takes the global min and max of input and returns
    # the (scale, zero_point) tensors from determine_qparams using eps and no custom
    # qrange.
    # KNOWN ISSUE: the docstring names the range arguments quant_min/quant_max while the
    # parameters are qmin/qmax.
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)

    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
    )


quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_symmetric.tensor",
    "CompositeExplicitAutograd",
)
def choose_qparams_symmetric_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Same as choose_qparams_tensor but passes the per_tensor_symmetric
    # qscheme to determine_qparams, so the zero_point is symmetric. Same dtype checks
    # (AssertionError) and qmin/qmax validation; returns (scale, zero_point).
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise AssertionError(
            f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
        )
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric,
    )


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Meta kernel for choose_qparams.tensor: requires
    # float32/float16/bfloat16 input and quant_min < quant_max (AssertionError
    # otherwise); returns a one-element double tensor (scale) and one-element int64
    # tensor (zero_point) on input's device.
    if input.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
        )
    if quant_min >= quant_max:
        raise AssertionError(
            f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"
        )
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
def choose_qparams_symmetric_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Meta kernel for choose_qparams_symmetric.tensor: no validation;
    # returns one-element double scale and int64 zero_point tensors on input's device.
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    # [FUNCTION DOC] HELPER: Swaps dimension axis with dimension 0 of x. Returns the
    # permuted tensor and the permutation list (a swap, so it is its own inverse and can
    # be used to permute back).
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list


quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Reference per-channel affine quantization along axis. Upcasts
    # half/bfloat16 input, requires float32 and axis < input.dim() (AssertionError),
    # checks quant range for dtype, moves axis to the front, reshapes scales and
    # zero_points to broadcast over it, computes round(input / scale) + zero_point
    # clamped to [quant_min, quant_max], permutes back and casts to dtype.
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    zero_points = zero_points.view(new_shape)

    res = torch.clamp(
        torch.round(input * (1.0 / scales)) + zero_points, quant_min, quant_max
    )
    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for quantize_per_channel: same dtype, axis and
    # quant-range checks (AssertionError / ValueError); returns an empty tensor of input
    # shape and dtype.
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Reference per-channel dequantization along axis. Requires
    # input.dtype == dtype and axis < input.dim() (AssertionError), out_dtype defaults
    # to float32, checks quant range. Moves axis to the front, computes (input -
    # zero_points) * scales (zero_points may be None, meaning no shift), casts to
    # out_dtype and permutes back.
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    if zero_points is not None:
        res = (input - zero_points.view(new_shape)) * scales
    else:
        res = input * scales

    res = res.to(out_dtype)

    out = res.permute(tuple(permute_axis_list))
    return out


@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for dequantize_per_channel: same dtype/axis/quant-range
    # checks; returns an empty tensor of input shape and out_dtype (default float32).
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if axis >= input.dim():
        raise AssertionError(f"Expecting axis to be < {input.dim()}")
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Symmetric int8 per-token qparams: for input of shape (M1..Mn, N)
    # takes the absolute max over the last dimension (kept as size 1), upcasts float16
    # scales to float32, and returns scales = max(absmax, 1e-5) / 127 with all-zero
    # zero_points of the same shape. Only torch.int8 is supported; any other dtype
    # raises a bare Exception.
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(  # noqa: TRY002
            f"unsupported dtype in choose_qparams_per_token: {dtype}"
        )

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Meta kernel: returns a double scale tensor and an int64 zero_point
    # tensor of shape (leading dims of input, 1) on input's device.
    # KNOWN ISSUE: dtypes (double / int64) differ from the reference kernel, which
    # returns scales in the input float dtype and zero_points as floats.
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


quantized_decomposed_lib.define(
    "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "_choose_qparams_per_token_asymmetric_impl",
    "CompositeImplicitAutograd",
)
def _choose_qparams_per_token_asymmetric_impl(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Asymmetric int8 (range -128..127) per-token qparams
    # following the XNNPACK scheme. Per last-dimension row it takes min and max extended
    # to include zero, scale = (max - min) / 255 clamped to float32 eps, picks the
    # zero_point from whichever end gives the smaller error, clamps and rounds it into
    # range. Returns (scale as float64, zero_point as int64), each with trailing
    # dimension 1. The dtype argument is unused.
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float64), zero_point.to(torch.int64)


quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Op entry for asymmetric per-token qparams; delegates to
    # _choose_qparams_per_token_asymmetric_impl and returns its (scale, zero_point).
    return _choose_qparams_per_token_asymmetric_impl(input, dtype)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Meta kernel: returns double scale and int64 zero_point tensors of
    # shape (leading dims of input, 1) on input's device.
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    # [FUNCTION DOC] HELPER: Checks that scales and zero_points each contain exactly one
    # element per token (product of all but the last input dimension). Raises
    # AssertionError on a mismatch.
    num_tokens = math.prod(list(input.size())[:-1])
    if num_tokens != scales.numel():
        raise AssertionError(f"num_tokens: {num_tokens} scales: {scales.size()}")
    if num_tokens != zero_points.numel():
        raise AssertionError(
            f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"
        )


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    # [FUNCTION DOC] Per-token affine quantization: validates the quant range for dtype
    # and the number of scales/zero_points (AssertionError), then computes input * (1 /
    # scales) + zero_points, rounds, clamps to [quant_min, quant_max] and casts to
    # dtype. Scales broadcast over the last dimension.
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        input.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp(quant_min, quant_max)
        .to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    # [FUNCTION DOC] Meta kernel for quantize_per_token: checks only the quant range for
    # dtype, returns an empty tensor of input shape and dtype.
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    # [FUNCTION DOC] Per-token dequantization: subtracts zero_points, multiplies by
    # scales (broadcast over the last dimension) and casts the result to output_dtype
    # (default float32; scales are float64 so the cast is needed). quant_min, quant_max
    # and dtype are unused.
    # KNOWN ISSUE: the docstring describes mapping floating point to quantized values,
    # which is the quantize direction.
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float64 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int64 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input * scales
    # Since scales are of float64 type, we need to cast it to output dtype requested
    return input.to(output_dtype)


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    # [FUNCTION DOC] Meta kernel for dequantize_per_token: checks the quant range for
    # dtype and returns an empty tensor of input shape and output_dtype (fp16 not
    # specially handled, see TODO).
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return torch.empty_like(input, dtype=output_dtype)


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    # [FUNCTION DOC] Groupwise quantization of a 2-d tensor: every group_size
    # consecutive elements of a row share one scale and zero_point. Raises
    # AssertionError if group_size <= 1, the last dimension is not divisible by
    # group_size, input is not 2-d, or input contains NaN. When group_size exceeds the
    # last dimension and there is a single scale (GPTQ single column case) it is clamped
    # to the last dimension. Computes round(x / scale + zero_point), clamps to
    # [quant_min, quant_max], casts to dtype and restores the input shape.
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")

    # TODO: check for dtype, currently we can't express torch.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    if torch.isnan(to_quant).sum() != 0:
        raise AssertionError("to_quant must not contain NaNs")

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    # [FUNCTION DOC] Meta kernel for quantize_per_channel_group: performs the same
    # group_size / divisibility / 2-d checks (AssertionError) and returns an empty
    # tensor of input shape and dtype.
    # KNOWN ISSUE: the docstring is placed here but really describes the quantize
    # operation; the reference kernel has none.
    """Groupwise quantization within each channel for a 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    if input.shape[-1] % group_size != 0:
        raise AssertionError("input.shape[-1] must be divisible by group_size")
    if input.dim() != 2:
        raise AssertionError("input must be 2-dimensional")
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    # [FUNCTION DOC] Groupwise dequantization of a 2-d integer tensor w_int8: same
    # group_size rules as quantize_per_channel_group (AssertionError on group_size <= 1,
    # non-divisible last dimension, or non-2-d input; group_size clamped for the
    # single-scale case). Reshapes into groups, subtracts zero_points (an int32 zero
    # when None), multiplies by scales, restores shape and casts to output_dtype.
    # KNOWN ISSUE: the docstring describes mapping floating point to quantized values
    # and names the first argument input instead of w_int8.
    """Groupwise dequantization within each channel for a 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    if group_size <= 1:
        raise AssertionError("group_size must be > 1")
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    if w_int8.shape[-1] % group_size != 0:
        raise AssertionError("w_int8.shape[-1] must be divisible by group_size")
    if w_int8.dim() != 2:
        raise AssertionError("w_int8 must be 2-dimensional")

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    if zero_points is not None:
        zp = zero_points.reshape(-1, 1)
    else:
        zp = torch.zeros([], dtype=torch.int32, device=scales.device)
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    return w_dq


quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor"
)


class FakeQuantPerChannel(torch.autograd.Function):
    # [CLASS DOC] autograd Function implementing per-channel fake quantization (quantize
    # then dequantize in float32) along an axis. The forward saves a boolean mask of
    # elements that fell inside [quant_min, quant_max]; the backward is a
    # straight-through estimator that passes the gradient only where the mask is true.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        # [FUNCTION DOC] Fake-quantizes input along axis: casts scales to float32 and
        # zero_points to int32, requires float32 input and axis < input.dim()
        # (AssertionError), broadcasts scales/zero_points over the other dimensions,
        # computes (clamp(round(input / scale) + zp, quant_min, quant_max) - zp) *
        # scale. Saves the in-range mask on ctx and returns the result.
        if scales.dtype != torch.float32:
            scales = scales.to(torch.float32)
        if zero_points.dtype != torch.int32:
            zero_points = zero_points.to(torch.int32)
        if input.dtype != torch.float32:
            raise AssertionError(
                f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
            )
        if axis >= input.dim():
            raise AssertionError(f"Expecting axis to be < {input.dim()}")
        broadcast_dims = list(range(axis)) + list(range(axis + 1, input.ndim))
        unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
        unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
        temp = torch.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
        out = (
            torch.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points
        ) * unsqueeze_scales
        mask = torch.logical_and((temp >= quant_min), (temp <= quant_max))

        ctx.save_for_backward(mask)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, gy):
        # [FUNCTION DOC] Straight-through gradient: multiplies gy by the saved in-range
        # mask; returns None for the five non-tensor-gradient inputs (scales,
        # zero_points, axis, quant_min, quant_max).
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Autograd")
def fake_quant_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Autograd-registered op entry that applies FakeQuantPerChannel to
    # its arguments and returns the fake-quantized tensor.
    return FakeQuantPerChannel.apply(
        input, scales, zero_points, axis, quant_min, quant_max
    )


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel for fake_quant_per_channel: returns an empty tensor
    # with the same shape and dtype as input.
    return torch.empty_like(input)


quantized_decomposed_lib.define(
    "convert_element_type.no_fuse(Tensor input, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "convert_element_type.no_fuse",
    "CompositeExplicitAutograd",
)
def convert_element_type(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC] Op that casts input to dtype by calling the prims
    # convert_element_type op; registered as a no-fuse variant so compilers do not fuse
    # the cast away.
    return torch.ops.prims.convert_element_type.default(input, dtype)


@impl(quantized_decomposed_lib, "convert_element_type.no_fuse", "Meta")
def convert_element_type_meta(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC] Meta kernel: returns an empty tensor of input shape with the
    # requested dtype.
    return torch.empty_like(input, dtype=dtype)
