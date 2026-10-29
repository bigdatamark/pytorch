# [MODULE DOC] Registers a Triton override of aten::bmm for outer-product-shaped batched inputs (b,m,1) x (b,1,n)
# on CUDA (and XPU when compiled). Contains the condition, implementation and registration functions.
import torch

from ... import triton_utils as tu


def _is_outer_product(a: torch.Tensor, b: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: True if both tensors are 3-D with a trailing singleton in a and a middle singleton in b,
    # both non-empty, and a is not complex (i.e. the bmm is an outer product per batch).
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1
        and b.shape[1] == 1
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
    )


def _bmm_outer_product_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Override implementation for bmm: runs the Triton outer-product kernel on a and b with the
    # accelerator device of a active, and returns the result. Extra positional/keyword arguments are ignored.
    from .triton_kernels import bmm_outer_product

    with torch.accelerator.device_index(a.get_device()):
        return bmm_outer_product(a, b)


def _is_acc_tensor(t: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: True if the tensor lives on the current accelerator's device type (False if no accelerator).
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type == t.device.type


def _bmm_outer_product_cond(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> bool:
    # [FUNCTION DOC] HELPER: Dispatch condition: True only when a is on the accelerator, both tensors are on the same
    # device and the shapes form an outer product. Extra arguments are ignored. Inputs are only read, so copy-on-write is fine.
    # a and b are read-only here: the kernel wraps them in ConstTensorWrapper and
    # reads through const_data_ptr(), so copy-on-write inputs are not
    # materialized and need not be excluded.
    if _is_acc_tensor(a) and a.device == b.device and _is_outer_product(a, b):
        return True
    return False


def _register_for_dispatch_key(dispatch_key: str) -> None:
    # [FUNCTION DOC] HELPER: Register the outer-product bmm override for aten::bmm under the given dispatch key via the
    # triton registry helper, allowing multiple overrides. Side effect: global dispatch registration.
    tu.register_op_override(
        "aten",
        "bmm",
        dispatch_key,
        cond=_bmm_outer_product_cond,
        impl=_bmm_outer_product_impl,
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    # [FUNCTION DOC] Register the override for CUDA and, if PyTorch is compiled with XPU, for XPU. Called on package import.
    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
