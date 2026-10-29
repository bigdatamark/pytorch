# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that wrap collective ops (all_gather, all_to_all, all_to_all_single) with
# automatic fp16/bfp16 quantization of inputs and dequantization of outputs. Defines the half-precision
# min/max constants (computed at import), the DQuantType enum, tensor/list (de)quantize helpers and auto_quantize.
import functools
from enum import Enum

import torch
import torch.distributed as dist


TORCH_HALF_MIN = torch.finfo(torch.float16).min
TORCH_HALF_MAX = torch.finfo(torch.float16).max


class DQuantType(Enum):
# [CLASS DOC] Enum of supported quantization methods for auto_quantize: FP16 and BFP16.
# KNOWN ISSUE: FP16's value is a one-element tuple while BFP16's is a string, so str() of FP16 returns a tuple.
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """

    FP16 = ("fp16",)
    BFP16 = "bfp16"

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the enum member's raw value.
        return self.value


def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Clamp a float tensor to the float16 representable range and convert to half, avoiding infinities.
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def _quantize_tensor(tensor, qtype):
    # [FUNCTION DOC] HELPER: Quantize one tensor by qtype (clamped fp16, or the bfloat16 quantization op).
    # Raises RuntimeError if input is not a tensor or qtype is unsupported.
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _quantize_tensor_list(tensor_list, qtype):
    # [FUNCTION DOC] HELPER: Quantize every tensor in a list; returns a new list. Raises RuntimeError
    # if the argument is not a list of tensors.
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list


def _dequantize_tensor(tensor, qtype, quant_loss=None):
    # [FUNCTION DOC] HELPER: Convert a quantized tensor back to float. FP16: requires half dtype, returns
    # float, divided by quant_loss when given. BFP16: calls the bfloat16-to-float op. Raises RuntimeError on
    # non-tensor input, wrong dtype or unsupported qtype.
    # KNOWN ISSUE: the BFP16 branch demands a float16 input dtype although BFP16 data is not float16 (quantized via a bfloat16 op).
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        elif tensor.dtype == torch.float16 and quant_loss is None:
            return tensor.float()
        else:
            # pyrefly: ignore [unsupported-operation]
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _dequantize_tensor_list(tensor_list, qtype, quant_loss=None):
    # [FUNCTION DOC] HELPER: Dequantize each tensor in a list and return a new list; raises RuntimeError
    # if not a list of tensors.
    # KNOWN ISSUE: quant_loss is accepted but not forwarded to the per-tensor dequantization, so it is ignored.
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_dequantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    dequantized_tensor_list = [_dequantize_tensor(t, qtype) for t in tensor_list]
    return dequantized_tensor_list


def auto_quantize(func, qtype, quant_loss=None):
    # [FUNCTION DOC] Decorator-style factory: returns a wrapper around a collective func (all_gather,
    # all_to_all or all_to_all_single) that quantizes inputs and output buffers by qtype, runs the
    # collective, dequantizes results and writes them back into the caller's output tensor list/tensor.
    # Args: func, qtype (DQuantType), optional quant_loss.
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] Run the quantized collective in place. Reads group, async_op, and (for
        # all_to_all_single) in_splits/out_splits from keyword args; raises RuntimeError if async_op
        # is true or func is unsupported. Mutates the output tensors; returns None.
        group = kwargs.get("group")
        async_op = kwargs.get("async_op", False)
        if async_op is True:
            raise RuntimeError("The async_op=True mode is not supported yet.")
        if func is dist.all_gather:
            tensors = args[0]
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all:
            tensors = args[0]
            input_tensors = _quantize_tensor_list(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_to_all(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all_single:
            tensors = args[0]
            out_splits = kwargs.get("out_splits")
            in_splits = kwargs.get("in_splits")
            # Quantizing the input/output tensor
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor(tensors, qtype)
            dist.all_to_all_single(
                out_tensors, input_tensors, out_splits, in_splits, group=group
            )
            for i, t in enumerate(
                _dequantize_tensor(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t
        else:
            raise RuntimeError(f"The collective op {func} is not supported yet")

    return wrapper
