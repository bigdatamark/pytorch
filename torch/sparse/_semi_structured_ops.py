# mypy: allow-untyped-defs
# [MODULE DOC] Dispatch handlers for torch ops applied to SparseSemiStructuredTensor
# (values, indices, transpose, view, detach, mm, addmm, linear, scaled_mm, clone, to, _to_copy),
# plus a fallback dispatcher and a helper that disables torch dispatch. Handlers share the
# torch-function style signature (func, types, args, kwargs). Exports are listed in __all__.
# No import-time side effects.
import contextlib

import torch


__all__ = [
    "fallback_dispatcher",
    "semi_sparse_values",
    "semi_sparse_indices",
    "semi_sparse_t",
    "semi_sparse_view",
    "semi_sparse_detach",
    "semi_sparse_mm",
    "semi_sparse_addmm",
    "semi_sparse_linear",
    "semi_sparse_scaled_mm",
    "semi_sparse_clone",
    "semi_sparse_to",
    "semi_sparse_to_copy",
]


@contextlib.contextmanager
def no_dispatch():
    # [FUNCTION DOC] HELPER: Context manager that disables torch dispatch interception for its
    # body by holding a C++ guard, and releases the guard on exit even after an exception.
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def fallback_dispatcher(func, types, args, kwargs):
    # [FUNCTION DOC] Run func on the positional args with dispatch disabled, so the operation
    # executes on the underlying tensor data. kwargs and types are ignored (kwargs are dropped).
    # Returns the op result.
    with no_dispatch():
        return func(*args)


def semi_sparse_values(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler returning the kept (compressed) values of one semi-structured
    # tensor. If no separate metadata exists, the values are the first half of the packed
    # buffer reshaped to (m, k/2); otherwise a detached view of the packed tensor.
    # Raises AssertionError on wrong argument count, type, or missing packed data.
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        return A.packed.ravel()[:num_kept_elements:].view(m, -1)
    else:
        return A.packed.detach()


def semi_sparse_indices(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler returning the sparsity metadata of one semi-structured tensor.
    # Without a separate meta tensor, the metadata is the tail of the packed buffer reshaped to
    # m rows and viewed as int32 (when the tensor dtype is int32) or int16; else returns meta.
    # Raises AssertionError on wrong argument count, type, or missing packed data.
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        metadata = A.packed.ravel()[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if A.dtype == torch.int32 else torch.int16)
    else:
        return A.meta


def semi_sparse_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for transpose of a 2D semi-structured tensor. No data is moved:
    # returns a new instance of the same class with swapped shape, packed/meta swapped with
    # their transposed counterparts, the bitmask transposed, and cuSPARSELt options copied.
    # Raises AssertionError if not given one 2D semi-structured tensor.
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    if not isinstance(self, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(self).__name__}"
        )
    if len(self.shape) != 2:
        raise AssertionError(f"expected 2D tensor, got {len(self.shape)}D")
    # Because we cannot go from the compressed representation back to the dense representation currently,
    # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
    # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
    # pyrefly: ignore [no-matching-overload]
    return self.__class__(
        torch.Size([self.shape[-1], self.shape[0]]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        compressed_swizzled_bitmask=(
            self.compressed_swizzled_bitmask.transpose(0, 1)
            if self.compressed_swizzled_bitmask is not None
            else None
        ),
        fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
        alg_id_cusparselt=args[0].alg_id_cusparselt,
    )


def semi_sparse_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for view: only the no-op case (requested shape equals the current
    # shape) is supported and returns the same tensor. Raises AssertionError for a wrong
    # argument count and NotImplementedError for any real reshape.
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for SparseSemiStructuredTensor, except for the dummy case (shape={shape})"
        )
    return self


def semi_sparse_detach(func, types, args, kwargs) -> torch.Tensor:
    # [FUNCTION DOC] Handler for detach: returns a new instance of the same class sharing all
    # buffers and options of the input but with requires_grad False.
    # Raises AssertionError unless exactly one argument is given.
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        compressed_swizzled_bitmask=self.compressed_swizzled_bitmask,
        fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
        alg_id_cusparselt=self.alg_id_cusparselt,
        requires_grad=False,
    )


def semi_sparse_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for matrix multiply of two 2D operands where one is semi-structured.
    # If the left operand is sparse, calls its multiply method with the right operand;
    # otherwise transposes the right operand (which must then be sparse) and multiplies with
    # the dense left operand flagged for transposition. Raises NotImplementedError for
    # non-2D inputs (no broadcasting) and AssertionError for a bad count or no sparse operand.
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        return A._mm(B)
    else:
        B_t = B.t()
        if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
            raise AssertionError(
                f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
            )
        return B_t._mm(A, should_transpose_dense=True)


def semi_sparse_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for addmm with arguments (bias, A, B) where only B may be sparse.
    # Requires 2D A and B and a 1D bias; transposes B and calls the sparse multiply with the
    # bias fused and the dense operand flagged for transposition. Raises NotImplementedError
    # for non-2D operands, non-1D bias, or sparse A; AssertionError for a bad count or dense B.
    if len(args) != 3:
        raise AssertionError(f"expected 3 args, got {len(args)}")
    bias, A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if bias.ndim != 1:
        raise NotImplementedError(
            f"`SparseSemiStructuredTensor` matmul: only bias dim=1 supported. Shape={bias.shape}"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: only operand B of `addmm` can be sparse"
        )
    B_t = B.t()
    if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
        )
    row, _col = A.shape
    return B_t._mm(A, bias=bias, should_transpose_dense=True)


def semi_sparse_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for linear with arguments (input, weight[, bias]) where weight is
    # semi-structured. Flattens input to 2D, then multiplies by the transposed weight (or uses
    # the addmm handler when bias is given), and restores the leading dimensions.
    # Raises AssertionError unless two or three arguments are passed; returns a tensor.
    if len(args) not in [2, 3]:
        raise AssertionError(f"expected 2 or 3 args, got {len(args)}")
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None

    shape = A.shape
    A_2d = A.view(-1, shape[-1])
    if bias is None:
        res = A_2d @ B.t()
    else:
        res = semi_sparse_addmm(
            func=None,
            types=None,
            args=[bias, A_2d, B.t()],
        )
    return res.view(*shape[:-1], -1)


def semi_sparse_scaled_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for scaled matrix multiply with float8 (e4m3) operands, using
    # cuSPARSELt. Unpacks (A, B, A_scale, B_scale, bias, scale_result, out_dtype) from the
    # first seven args; A must be a cuSPARSELt semi-structured tensor, scales must be float32
    # single-element tensors. The product of scales is passed as alpha to the sparse matmul.
    # Raises AssertionError on violated preconditions. Returns the product tensor.
    # KNOWN ISSUE: bias and scale_result are accepted but ignored; kwargs are unused.
    # pull all args, excluding use_fast_accum flag if set.
    A, B, A_scale, B_scale, bias, scale_result, out_dtype = args[:7]

    if A.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected A.dtype float8_e4m3fn, got {A.dtype}")
    if B.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected B.dtype float8_e4m3fn, got {B.dtype}")
    # only cuSPARSELt supports float8_e4m3fn currently
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensorCUSPARSELT):
        raise AssertionError(
            f"expected SparseSemiStructuredTensorCUSPARSELT, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    # Currently we only support per-tensor scaling, with float32 scales
    if A_scale.numel() != 1 or B_scale.numel() != 1:
        raise AssertionError(
            f"expected A_scale and B_scale to have numel 1, got {A_scale.numel()} and {B_scale.numel()}"
        )
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise AssertionError(
            f"expected A_scale and B_scale dtype float32, got {A_scale.dtype} and {B_scale.dtype}"
        )

    # cuSPARSELt lacks the A and B operand scaling support, so instead we use alpha to scale the result.
    # Note that this limits us to per-tensor scaling only.
    sparse_result = torch._cslt_sparse_mm(
        A.packed,
        B,
        alpha=A_scale * B_scale,
        out_dtype=out_dtype,
    )
    return sparse_result


def semi_sparse_clone(func, types, args=(), kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for clone: returns a new instance of the same class with every
    # present buffer (packed, meta, transposed variants, bitmask) cloned and options and
    # requires_grad preserved. Raises AssertionError unless given one semi-structured tensor.
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")

    self = args[0]
    if not isinstance(self, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(self).__name__}"
        )

    # pyrefly: ignore [no-matching-overload]
    return self.__class__(
        shape=self.shape,
        packed=None if self.packed is None else self.packed.clone(),
        meta=None if self.meta is None else self.meta.clone(),
        packed_t=None if self.packed_t is None else self.packed_t.clone(),
        meta_t=None if self.meta_t is None else self.meta_t.clone(),
        compressed_swizzled_bitmask=(
            None
            if self.compressed_swizzled_bitmask is None
            else self.compressed_swizzled_bitmask.clone()
        ),
        fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
        alg_id_cusparselt=self.alg_id_cusparselt,
        requires_grad=self.requires_grad,
    )


def semi_sparse_to_copy(func, types, args, kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for _to_copy: only conversion to a CPU device (given as a keyword)
    # is supported; the tensor is densified and func is applied to it with the keywords.
    # Raises NotImplementedError otherwise. Returns the converted dense tensor.
    self = args[0]
    kwargs = kwargs or {}

    device = kwargs.get("device", None)

    if device is not None and torch.device(device).type == "cpu":
        dense = self.to_dense()
        return func(dense, **kwargs)

    raise NotImplementedError(
        f"`_to_copy()` with kwargs={kwargs} is not implemented "
        "for SparseSemiStructuredTensor. Only converting to CPU is supported currently."
    )


def semi_sparse_to(func, types, args, kwargs=None) -> torch.Tensor:
    # [FUNCTION DOC] Handler for to: finds a target device from the first extra positional
    # argument (device or string, ignored if unparsable) or the device keyword (which wins).
    # Only a CPU target is supported: the tensor is densified and func is applied with the
    # remaining args. Raises NotImplementedError otherwise.
    self = args[0]
    remaining_args = args[1:]
    kwargs = kwargs or {}

    # Determine the target device from args/kwargs
    device = None
    if remaining_args:
        first_arg = remaining_args[0]
        if isinstance(first_arg, (torch.device, str)):
            try:
                device = torch.device(first_arg)
            except RuntimeError:
                pass
    if "device" in kwargs:
        device = torch.device(kwargs["device"])

    if device is not None and device.type == "cpu":
        dense = self.to_dense()
        return func(dense, *remaining_args, **kwargs)

    raise NotImplementedError(
        f"`to()` with args={remaining_args}, kwargs={kwargs} is not implemented "
        "for SparseSemiStructuredTensor. Only `to('cpu')` is supported currently."
    )
