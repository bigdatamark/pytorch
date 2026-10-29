# [MODULE DOC] Defines ConstTensorWrapper, a thin wrapper used to pass read-only tensors to Triton kernels without
# materializing copy-on-write tensors. No import-time side effects.
import torch


class ConstTensorWrapper:
    # [CLASS DOC] Wraps one tensor and exposes the attributes Triton reads from kernel arguments (data_ptr, dtype, device,
    # shape, ndim, dim, size, stride, element_size). data_ptr returns the const data pointer; the rest forward to the tensor.
    # Methods that dispatch run with torch function handling disabled. Only for arguments the kernel never writes.
    """Read-only, zero-copy view of a tensor for Triton kernel arguments.

    Triton extracts a kernel pointer argument by duck typing: any object with a
    ``data_ptr()`` method and a ``dtype`` is treated as a pointer (see Triton's
    ``jit.py``). ``ConstTensorWrapper`` overrides ``data_ptr()`` to return
    ``const_data_ptr()``, so passing a copy-on-write tensor to an argument the
    kernel only reads does not materialize it. The remaining attributes Triton
    inspects during specialization and launch are forwarded to the wrapped
    tensor.

    Only wrap arguments the kernel treats as read-only (loads, never stores).
    Triton does not enforce this; wrapping a written argument is a bug.

    Accessors run under DisableTorchFunctionSubclass so the wrapper never
    re-enters __torch_function__: const_data_ptr() and friends are dispatched
    methods, and re-entering while an override is mid-redispatch trips the
    "cannot skip two levels of __torch_function__" guard.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] Store the wrapped tensor.
        self._tensor = tensor

    def data_ptr(self) -> int:
        # [FUNCTION DOC] Return the const (non-materializing) data pointer of the tensor as an int, with torch function disabled.
        with torch._C.DisableTorchFunctionSubclass():
            # const_data_ptr() exists at runtime but is absent from the stubs.
            return self._tensor.const_data_ptr()  # type: ignore[attr-defined]

    @property
    def dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Property: dtype of the wrapped tensor.
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Property: device of the wrapped tensor.
        return self._tensor.device

    @property
    def shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: shape of the wrapped tensor.
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Property: number of dimensions of the wrapped tensor.
        return self._tensor.ndim

    def dim(self) -> int:
        # [FUNCTION DOC] Return the number of dimensions, with torch function disabled.
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.dim()

    def size(self, dim: int | None = None) -> torch.Size | int:
        # [FUNCTION DOC] Return the full size, or the size of dimension dim if given, with torch function disabled.
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.size() if dim is None else self._tensor.size(dim)

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        # [FUNCTION DOC] Return the full strides, or the stride of dimension dim if given, with torch function disabled.
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.stride() if dim is None else self._tensor.stride(dim)

    def element_size(self) -> int:
        # [FUNCTION DOC] Return the element size in bytes, with torch function disabled.
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.element_size()
