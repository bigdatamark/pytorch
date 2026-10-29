# mypy: allow-untyped-defs
# [MODULE DOC] Semi-structured (2:4) sparse tensor support for CUDA tensors, as Tensor subclasses.
# Defines SparseSemiStructuredTensor (abstract base), CUTLASS and cuSPARSELt backend subclasses,
# the public to_sparse_semi_structured converter, and lazy registration of the custom ops
# semi_structured::cutlass_mm and semi_structured::cusparselt_mm.
# Import-time effects: only defines classes/globals; the config namedtuple is created here. Op
# registration and the dispatch table are deferred until first use to keep import torch cheap.
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import torch
from torch.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
)
from torch.sparse._semi_structured_ops import (
    fallback_dispatcher,
    semi_sparse_addmm,
    semi_sparse_clone,
    semi_sparse_detach,
    semi_sparse_indices,
    semi_sparse_linear,
    semi_sparse_mm,
    semi_sparse_scaled_mm,
    semi_sparse_t,
    semi_sparse_to,
    semi_sparse_to_copy,
    semi_sparse_values,
    semi_sparse_view,
)


__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)


class SparseSemiStructuredTensor(torch.Tensor):
    # [CLASS DOC] Abstract wrapper-subclass holding a 2:4 compressed matrix: packed values,
    # optional metadata,
    # and optional transposed copies (packed_t, meta_t) plus a compressed swizzled bitmask.
    # Also records cuSPARSELt options (fuse_transpose_cusparselt, alg_id_cusparselt). Class-level
    # state: _FORCE_CUTLASS, _FUSE_TRANSPOSE, _PROTOTYPE_WARNING_SHOWN, and the SPARSE_DISPATCH
    # mapping from aten overload packets to handlers. Only ops in that mapping are supported
    # (torch function overrides are disabled; everything goes through torch dispatch).
    # Subclasses must define BACKEND, _DTYPE_SHAPE_CONSTRAINTS, from_dense and _mm.
    """
    This class implements semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    There are two backends available for semi_structured sparsity, either cuSPARSELt or CUTLASS.
    This class is meant to serve as a base class for both implementations. SparseSemiStructuredCUTLASS
    and SparseSemiStructuredCUSPARSELT both inherit from this class and define three backend-specific items.
    Note that as such, this class cannot be instantiated directly.

    -`_DTYPE_SHAPE_CONSTRAINTS` - A dictionary holding backend specific dense/sparse min shape constraints
    - `def from_dense()` - backend specific compression routines
    - `def _mm()` - backend specific mm op (either torch._cslt_sparse_mm or torch._sparse_semi_structured_(mm|addmm))
    """

    _DEFAULT_ALG_ID: int = 0
    _DTYPE_SHAPE_CONSTRAINTS: dict[torch.dtype, _SEMI_STRUCTURED_SPARSE_CONFIG]
    _FORCE_CUTLASS: bool = False
    _FUSE_TRANSPOSE: bool = False
    _PROTOTYPE_WARNING_SHOWN: bool = False

    BACKEND: str
    SPARSE_DISPATCH: dict[Callable, Callable]

    packed: torch.Tensor | None
    meta: torch.Tensor | None
    packed_t: torch.Tensor | None
    meta_t: torch.Tensor | None
    compressed_swizzled_bitmask: torch.Tensor | None
    fuse_transpose_cusparselt: bool
    alg_id_cusparselt: int

    __slots__ = ["packed", "meta", "packed_t", "meta_t", "compressed_swizzled_bitmask"]

    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        packed: torch.Tensor | None,
        meta: torch.Tensor | None,
        packed_t: torch.Tensor | None,
        meta_t: torch.Tensor | None,
        compressed_swizzled_bitmask: torch.Tensor | None,
        fuse_transpose_cusparselt: bool = False,
        alg_id_cusparselt: int = 0,
        requires_grad: bool = False,
    ):
        # [FUNCTION DOC] Builds the wrapper subclass from compressed parts; shape is the
        # logical dense shape.
        # Device, dtype and layout come from packed, or from packed_t if packed is None.
        # Raises ValueError if both packed and packed_t are None.
        # Side effects: on the first construction (class flag unset) emits a prototype UserWarning,
        # loads the dispatch table, and registers the class with dynamo as allowed in graph.
        # Stores all parts and the cuSPARSELt options as attributes on the new tensor.
        """
        Create a new instance of the tensor subclass from the compressed sparse representation.

        We have the option to create the subclass with the compressed representations of both X and X', for training.
        For inference, we only need a single representation (either X or X'), while the corresponding other set will be None.

        Depending on the backend selected, certain fields will be set to None. (CUSPARSELT vs CUTLASS)

        Args:
            shape: The shape of the original dense tensor
            packed: The compressed representation of the original dense tensor
            meta: The metadata of the original dense tensor, if it is stored separately
            packed_t: The compressed representation of the transposed original dense tensor
            meta_t: The metadata of the transposed original dense tensor, if it is stored separately
            compressed_swizzled_bitmask: The masks used by the CUTLASS backend to determine which threads should
                                         participate in the computation. Used for pointwise ops.
            fuse_transpose_cusparselt: When running with cuSPARSELt, we have the option to fuse a transposition
                                       with a matmul, which is useful in the case of 2:4 sparse training.
            alg_id_cusparselt: The algorithm id to use when using cuSPARSELT, will have an effect on performance

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.
        """
        if not cls._PROTOTYPE_WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
                stacklevel=2,
            )
            cls._PROTOTYPE_WARNING_SHOWN = True

            # Because this only runs once, we also load the dispatch table here as well.
            # We can't define the dispatch table explicitly because of torch.ops import errors, so we do this instead
            # But this is useful since it allows users to overload the dispatch table for debugging / testing.
            cls._load_dispatch_table()

            # we can also register the classes with dynamo when the warning is shown.
            torch._dynamo.allow_in_graph(cls)

        if packed is not None:
            previous_tensor = packed
        elif packed_t is not None:
            previous_tensor = packed_t
        else:
            raise ValueError("At least one of packed or packed_t must be provided")

        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=previous_tensor.device,
            dtype=previous_tensor.dtype,
            layout=previous_tensor.layout,
            requires_grad=requires_grad,
        )

        tensor.packed = packed
        tensor.meta = meta
        tensor.packed_t = packed_t
        tensor.meta_t = meta_t
        tensor.compressed_swizzled_bitmask = compressed_swizzled_bitmask
        tensor.fuse_transpose_cusparselt = fuse_transpose_cusparselt
        tensor.alg_id_cusparselt = alg_id_cusparselt
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC] Returns a short string with the class name and logical shape; raises
        # AssertionError if
        # the tensor has no shape attribute.
        if not hasattr(self, "shape"):
            raise AssertionError("tensor has no shape attribute")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(
        self,
    ) -> tuple[list[str], tuple[torch.Size, bool, int, bool]]:
        # [FUNCTION DOC] Tensor-subclass flatten protocol: returns the names of non-None inner
        # tensor slots and a
        # tuple of metadata (shape, fuse_transpose_cusparselt, alg_id_cusparselt, requires_grad).
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (
            self.shape,
            self.fuse_transpose_cusparselt,
            self.alg_id_cusparselt,
            self.requires_grad,
        )
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: tuple[torch.Size, bool, int, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Inverse of the flatten protocol: rebuilds an instance from the inner
        # tensor mapping and the
        # metadata tuple; missing slots become None. outer_size and outer_stride are ignored.
        shape, fuse_transpose_cusparselt, alg_id_cusparselt, requires_grad = tensor_meta
        # pyrefly: ignore [no-matching-overload]
        return cls(
            shape=shape,
            packed=inner_tensors.get("packed", None),
            meta=inner_tensors.get("meta", None),
            packed_t=inner_tensors.get("packed_t", None),
            meta_t=inner_tensors.get("meta_t", None),
            compressed_swizzled_bitmask=inner_tensors.get(
                "compressed_swizzled_bitmask", None
            ),
            fuse_transpose_cusparselt=fuse_transpose_cusparselt,
            alg_id_cusparselt=alg_id_cusparselt,
            requires_grad=requires_grad,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[assignment]

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:  # type: ignore[override]
        # [FUNCTION DOC] Dispatch hook: looks up the op's overload packet in SPARSE_DISPATCH
        # and calls the handler
        # with (func, types, args, kwargs). Raises NotImplementedError for any unsupported op.
        if func._overloadpacket not in cls.SPARSE_DISPATCH:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        return cls.SPARSE_DISPATCH[func._overloadpacket](func, types, args, kwargs)

    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None) -> None:
        # [FUNCTION DOC] HELPER: Populates the class SPARSE_DISPATCH mapping (aten op to
        # handler) if it is unset,
        # then merges the optional custom_dispatch_table over it. If the mapping already exists,
        # does nothing, including ignoring the custom table. Mutates class state; returns None.
        """
        Loads the op overload sparse dispatch table for the current class.
        """
        if getattr(cls, "SPARSE_DISPATCH", None) is None:
            cls.SPARSE_DISPATCH = {
                torch.ops.aten.values: semi_sparse_values,
                torch.ops.aten.indices: semi_sparse_indices,
                torch.ops.aten.is_same_size: fallback_dispatcher,
                torch.ops.aten.detach_: fallback_dispatcher,
                torch.ops.aten.detach: semi_sparse_detach,
                torch.ops.aten.t: semi_sparse_t,
                torch.ops.aten.view: semi_sparse_view,
                torch.ops.aten.mm: semi_sparse_mm,
                torch.ops.aten.matmul: semi_sparse_mm,
                torch.ops.aten.addmm: semi_sparse_addmm,
                torch.ops.aten.linear: semi_sparse_linear,
                torch.ops.aten._to_copy: semi_sparse_to_copy,
                torch.ops.aten._scaled_mm: semi_sparse_scaled_mm,
                torch.ops.aten.clone: semi_sparse_clone,
                torch.ops.aten.to: semi_sparse_to,
            }
            if custom_dispatch_table is not None:
                cls.SPARSE_DISPATCH.update(custom_dispatch_table)

    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Checks a dense tensor can be compressed: must be CUDA, 2-D,
        # contiguous, of a dtype in
        # _DTYPE_SHAPE_CONSTRAINTS, and both dims at least and a multiple of the sparse minimums.
        # Raises RuntimeError describing the first violated condition; returns None.
        """
        Assert that the given tensor is valid for semi-structured sparse compression.
        """
        # check device
        if not original_tensor.is_cuda:
            raise RuntimeError(
                f"Error original_tensor.device= {original_tensor.device} is not supported! "
                "Only CUDA tensors are currently supported."
            )

        # check dim
        if original_tensor.dim() != 2:
            raise RuntimeError(
                f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                "Only 2d tensors are currently supported."
            )

        # check contiguous
        if not original_tensor.is_contiguous():
            raise RuntimeError(
                "Error original_tensor is not contiguous!"
                "Only contiguous tensors are currently supported."
            )

        # check dtype
        if original_tensor.dtype not in cls._DTYPE_SHAPE_CONSTRAINTS:
            raise RuntimeError(
                f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype for {cls}!"
            )

        # check shape
        m, n = original_tensor.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_cols
        if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
            # TODO in the future we can add in padding to support sparse dimensions that aren't perfect multiples
            raise RuntimeError(
                f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                f"Both dimensions must be larger than or equal to and a multiple of ({min_rows}, {min_cols})"
            )

    def to_dense(self):  # type:ignore[override]
        # [FUNCTION DOC] Default dense conversion: multiplies the sparse tensor by an identity
        # matrix of the column
        # count, using the same dtype and device. Returns the dense 2-D tensor.
        col = self.shape[-1]
        return torch.mm(self, torch.eye(col, dtype=self.dtype, device=self.device))

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = _DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensor":
        # [FUNCTION DOC] Abstract: backend subclasses compress original_tensor (using alg_id)
        # into an instance.
        # Base version always raises NotImplementedError.
        raise NotImplementedError

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Abstract: backend subclasses multiply this sparse matrix by dense B
        # (optional bias).
        # Base version always raises NotImplementedError.
        raise NotImplementedError


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    transposed: bool = False,
    alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
) -> SparseSemiStructuredTensor:
    # [FUNCTION DOC] Public converter: picks the CUTLASS subclass if _FORCE_CUTLASS is set,
    # otherwise the
    # cuSPARSELt subclass, and returns its from_dense result for original_tensor and alg_id.
    # If transposed is true, only emits a FutureWarning (the argument has no other effect).
    # KNOWN ISSUE: docstring says Raises None, but from_dense validation raises RuntimeError.
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of the minimum sparse block size, given in
    `_DTYPE_TO_SHAPE_CONSTRAINTS` for each dtype (float32, float16, bfloat16, int8).

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): deprecated arg to be removed in another release. Do not use.
        alg_id (int, optional): the algorithm id to use for cuSPARSELt matmul. Defaults to 0.
            Can be obtained via ``torch._cslt_sparse_mm_search``.
    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor
    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]))
        >>> A_sparse.values()
        tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
        >>> A_sparse.indices()
        tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0', dtype=torch.int16))
    """
    if transposed:
        warnings.warn(
            "Setting transpose from `to_sparse_semi_structured` is deprecated "
            "and will be removed in a future release. "
            "`SparseSemiStructuredTensor` only support contiguous input tensors.",
            FutureWarning,
            stacklevel=2,
        )

    # set from _FORCE_CUTLASS flag
    SPARSE_SUBCLASS = (
        torch.sparse.SparseSemiStructuredTensorCUTLASS
        if SparseSemiStructuredTensor._FORCE_CUTLASS
        else torch.sparse.SparseSemiStructuredTensorCUSPARSELT
    )

    return SPARSE_SUBCLASS.from_dense(original_tensor, alg_id=alg_id)


class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    # [CLASS DOC] CUTLASS backend: values are in packed and indices in a separate meta tensor;
    # the dense
    # min-shape table covers int8, float16, bfloat16 and float32. Matmul goes through the
    # lazily registered cutlass_mm custom op. Supports training-style pruning with transposed
    # copies via prune_dense_static_sort.
    """
    This class implements semi-structured sparsity for the CUTLASS backend.


    In this implementation, the specified elements and metadata are stored separately,
    in packed and meta respectively.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_(mm|addmm) and
    sparse_semi_structured_from_dense for conversion to the compressed format.
    """

    BACKEND = "cutlass"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4),
    }

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensorCUTLASS":
        # [FUNCTION DOC] Validates the dense tensor, converts it to packed values and meta with
        # the CUTLASS
        # conversion routine, and returns an instance preserving shape and requires_grad.
        # alg_id is unused. Raises RuntimeError on invalid input.
        cls._validate_device_dim_dtype_shape(original_tensor)
        (
            sparse_tensor_cutlass,
            meta_tensor_cutlass,
        ) = sparse_semi_structured_from_dense_cutlass(original_tensor)
        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=sparse_tensor_cutlass,
            meta=meta_tensor_cutlass,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            requires_grad=original_tensor.requires_grad,
        )

    def to_dense(self):  # type: ignore[override]
        # [FUNCTION DOC] Converts back to dense: uses the CUTLASS reverse conversion when meta
        # is 2-D, else falls
        # back to the base multiply-by-identity path. Raises AssertionError if meta or packed
        # is None.
        if self.meta is None or self.packed is None:
            raise AssertionError("meta and packed must not be None")
        return (
            sparse_semi_structured_to_dense_cutlass(
                self.packed,
                self.meta,
            )
            if self.meta.ndim == 2
            else super().to_dense()
        )

    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=""
    ) -> "SparseSemiStructuredTensor":
        # [FUNCTION DOC] Prunes a dense tensor to 2:4 over 4x4 tiles via the tile op (CUTLASS
        # layout), producing
        # packed, meta, packed_t, meta_t and the compressed swizzled bitmask, and returns an
        # instance
        # with requires_grad False. The algorithm string selects the tile pruning algorithm.
        """
        This function takes in a unpruned dense tensor and runs a (branchless) static sort across a 4x4 tile.

        It greedily picks the largest values in the tile, upholding the 2:4 sparsity constraint across both rows and columns.
        The algorithm used to prune the matrix is implemented in `_sparse_semi_structured_tile`.

        Then it creates the packed and meta tensors for the compressed sparse representation of the pruned dense tensor.
        It also calculates the packed_t and meta_t tensors for the compressed sparse representation of the transposed
        pruned dense tensor.
        Since we cannot transpose the compressed representations, we store both for the fw/bw pass respectively.

        Finally, this function also computes a compressed swizzled bitmask that encodes the sparsity pattern.
        This can be used in the backward pass to mask the gradients.

        ::

            [9 1 7 4]                       [9 0 7 0]
            [1 2 3 0]                       [0 2 0 0]
            [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to CUTLASS semi-structured -> packed
            [1 2 6 2]                       [0 0 6 2]                                    -> metadata

                                                      -> pack to transposed CUTLASS      -> packed_t
                                                         semi-structured representation  -> metadata_t

                                                      -> compute swizzled bitmask        -> compressed_swizzled_bitmask

        The equivalent PyTorch code to create the same five outputs from the dense tensor can be found below::

            from torch.sparse import SparseSemiStructuredTensorCUTLASS
            from torch.sparse._semi_structured_conversions import (
                _sparse_semi_structured_tile,
                _compute_compressed_swizzled_bitmask,
            )

            pruned = _sparse_semi_structured_tile(dense)
            packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(
                pruned
            )
            packed_t_cutlass, meta_t_cutlass = (
                sparse_semi_structured_from_dense_cutlass(pruned.t().contiguous())
            )
            bitmask = _compute_compressed_swizzled_bitmask(pruned)

            SparseSemiStructuredTensorCUTLASS(
                dense.shape,
                packed_cutlass,
                meta_cutlass,
                packed_t_cutlass,
                meta_t_cutlass,
                bitmask,
            )
        """
        # We can either pack to the CUTLASS or cuSPARSELt representation, depending on the use_cutlass flag.
        (
            packed,
            meta,
            packed_t,
            meta_t,
            compressed_swizzled_bitmask,
        ) = torch._sparse_semi_structured_tile(
            original_tensor, algorithm=algorithm, use_cutlass=True
        )

        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        should_transpose_dense: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Sparse @ dense B (plus optional bias) via cutlass_mm. Raises
        # ValueError if B is itself
        # sparse; NotImplementedError if either operand is not 2-D or packed/meta are missing.
        # Registers the custom op on first use. Uses dense min-shape constraints keyed by B dtype.
        # KNOWN ISSUE: constraints are looked up by B.dtype, so an unsupported dtype gives a
        # KeyError.
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        cls_name = self.__class__.__name__
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{cls_name}` matmul: Broadcasting is not implemented"
            )
        if self.packed is None or self.meta is None:
            raise NotImplementedError(
                f"`{cls_name}` matmul: operation is not supported"
            )
        else:
            _ensure_cutlass_mm_registered()
            constraints = self._DTYPE_SHAPE_CONSTRAINTS[B.dtype]
            return torch.ops.semi_structured.cutlass_mm(
                B,
                self.packed,
                self.meta,
                bias,
                self.shape[0],
                constraints.dense_min_rows,
                constraints.dense_min_cols,
                should_transpose_dense,
            )


class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    # [CLASS DOC] cuSPARSELt backend: values and metadata are stored together in one packed
    # tensor (meta
    # stays None), with packed_t for a transposed copy. Supports float8_e4m3fn, int8, float16
    # and bfloat16, optional transpose fusion and an algorithm id. Matmul goes through the
    # lazily registered cusparselt_mm custom op.
    """
    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
    packed = [ specified elements of original tensor | metadata ]
    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. Since there is only one tensor, we only use the packed and packed_t
    attributes respectively.

    cuSPARSELt also supports transposition fusion, which is necessary for performant 2:4 sparse training, as well
    as specifying alg_id, a config that affects the performance of the matmul depending on matmul sizes.
    """

    BACKEND = "cusparselt"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.float8_e4m3fn: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
    }

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensorCUSPARSELT":
        # [FUNCTION DOC] Validates the dense tensor, compresses it with the cuSPARSELt compress
        # op, and returns an
        # instance carrying the class-level fuse-transpose flag and alg_id. Raises RuntimeError
        # on invalid input.
        cls._validate_device_dim_dtype_shape(original_tensor)
        # pyrefly: ignore [no-matching-overload]
        return cls(
            shape=original_tensor.shape,
            packed=torch._cslt_compress(original_tensor),
            meta=None,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            fuse_transpose_cusparselt=SparseSemiStructuredTensor._FUSE_TRANSPOSE,
            alg_id_cusparselt=alg_id,
            requires_grad=original_tensor.requires_grad,
        )

    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=""
    ) -> "SparseSemiStructuredTensor":
        # [FUNCTION DOC] Prunes a dense tensor to 2:4 over 4x4 tiles with the cuSPARSELt
        # layout, then reshapes
        # packed and packed_t to 2-D (rows by remaining), and returns an instance with the bitmask.
        # KNOWN ISSUE: the docstring example refers to packed_cutlass names; and a TODO notes the
        # metadata layout is unverified.
        """
        This function does the same thing as described in SparseSemiStructuredCUTLASS, but uses the cuSPARSELt metadata
        layout and sparse matmul.

        The only functional difference is that cuSPARSELt stores `metadata` and `packed` together into a single tensor.

        ::

            [9 1 7 4]                       [9 0 7 0]
            [1 2 3 0]                       [0 2 0 0]
            [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to cuSPARSELT semi-structured -> packed
            [1 2 6 2]                       [0 0 6 2]

                                                      -> pack to transposed cuSPARSELt      -> packed_t
                                                         semi-structured representation

                                                      -> compute swizzled bitmask           -> compressed_swizzled_bitmask

        The equivalent PyTorch code to create the same three outputs from the dense tensor can be found below::

            from torch.sparse import SparseSemiStructuredTensorCUSPARSELT
            from torch.sparse._semi_structured_conversions import (
                _sparse_semi_structured_tile,
                _compute_compressed_swizzled_bitmask,
            )

            pruned = _sparse_semi_structured_tile(dense)
            packed_cusparselt = torch._cslt_compress(pruned)
            packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
            bitmask = _compute_compressed_swizzled_bitmask(pruned)

            SparseSemiStructuredTensorCUSPARSELT(
                dense.shape, packed_cutlass, None, packed_t_cutlass, None, bitmask
            )
        """
        (
            packed,
            meta,
            packed_t,
            meta_t,
            compressed_swizzled_bitmask,
        ) = torch._sparse_semi_structured_tile(
            original_tensor, algorithm=algorithm, use_cutlass=False
        )

        # Map this two 2-dim view of packed data.
        # TODO: is this proper cuSPARSELt metadata?
        packed = packed.view(original_tensor.shape[0], -1)
        packed_t = packed_t.view(original_tensor.shape[1], -1)

        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        should_transpose_dense: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Sparse @ dense B (plus optional bias) via cusparselt_mm. Raises
        # ValueError if B is sparse;
        # NotImplementedError for non-2-D operands, dtype mismatch of B or bias with self, float8
        # inputs, or missing packed. Registers the custom op on first use.
        # KNOWN ISSUE: should_transpose_dense is passed for both the fuse_transpose and the
        # should_transpose_dense arguments, ignoring fuse_transpose_cusparselt; the bias error
        # message reports B dtype as the bias dtype.
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if B.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype={self.dtype} and B.dtype={B.dtype}. "
                "This operation is only supported when A and B have the same data type."
            )
        if bias is not None and bias.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, "
                f"with A.dtype=B.dtype={self.dtype} and C.dtype={B.dtype}. "
                "This operation is only supported when A, B and C have the same data type."
            )
        # Force fp8 mm to error to be consistent with torch
        if self.dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype=B.dtype={self.dtype}. "
                "mm is not supported for float8_e4m3fn, please use `torch._scaled_mm` instead."
            )
        if self.packed is None:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: operation is not supported"
            )
        else:
            _ensure_cusparselt_mm_registered()
            constraints = self._DTYPE_SHAPE_CONSTRAINTS[B.dtype]
            return torch.ops.semi_structured.cusparselt_mm(
                B,
                self.packed,
                bias,
                self.shape[0],
                constraints.dense_min_rows,
                constraints.dense_min_cols,
                should_transpose_dense,
                self.alg_id_cusparselt,
                should_transpose_dense,
            )


_cutlass_mm_registered = False


def _ensure_cutlass_mm_registered():
    # [FUNCTION DOC] HELPER: One-time lazy registration of the semi_structured::cutlass_mm
    # custom op, with a
    # fake (shape-only) implementation and a decomposition under FunctionalTensorMode. Guarded by
    # a module-global flag set before registering; later calls return immediately.
    """Lazily register the cutlass_mm custom op.

    Registration is deferred to avoid importing torch.library at module load
    time, since torch.sparse is imported early during ``import torch``.
    """
    global _cutlass_mm_registered
    if _cutlass_mm_registered:
        return
    _cutlass_mm_registered = True

    from torch.library import custom_op

    def _cutlass_mm_impl(
        dense: torch.Tensor,
        packed: torch.Tensor,
        meta: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        should_transpose_dense: bool,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Pads dense up to the min-row/column multiples, optionally
        # transposes it, runs the
        # CUTLASS sparse mm (or addmm with bias), then trims padding to out_features and the real
        # column count. Returns a contiguous result, transposed back if should_transpose_dense.
        m, n = dense.shape
        to_pad_m = (-m) % min_rows
        to_pad_n = (-n) % min_cols
        need_pad = to_pad_m != 0 or to_pad_n != 0
        dense_padded = dense
        if need_pad:
            dense_padded = torch.nn.functional.pad(dense, (0, to_pad_n, 0, to_pad_m))
        mm_input = dense_padded.t() if should_transpose_dense else dense_padded
        if bias is None:
            res = torch._sparse_semi_structured_mm(packed, meta, mm_input)
        else:
            res = torch._sparse_semi_structured_addmm(bias, packed, meta, mm_input)
        if need_pad:
            out_cols = m if should_transpose_dense else n
            res = res[:out_features].narrow(1, 0, out_cols)
            return res.t().contiguous() if should_transpose_dense else res.contiguous()
        return res.t().contiguous() if should_transpose_dense else res

    cutlass_mm = custom_op("semi_structured::cutlass_mm", mutates_args=())(
        _cutlass_mm_impl
    )

    @cutlass_mm.register_fake
    def _cutlass_mm_fake(
        dense: torch.Tensor,
        packed: torch.Tensor,
        meta: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        transpose_dense: bool,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Fake implementation: returns an empty tensor of the output
        # shape (transposed layout
        # if transpose_dense) with dense dtype and device.
        out_cols = dense.shape[0] if transpose_dense else dense.shape[1]
        output_shape = (
            (out_cols, out_features) if transpose_dense else (out_features, out_cols)
        )
        return torch.empty(
            output_shape,
            dtype=dense.dtype,
            device=dense.device,
        )

    from torch._subclasses.functional_tensor import (
        _has_unrecognized_tensor_types,
        FunctionalTensorMode,
    )

    # Unlike triton custom ops (which guard with custom_triton_ops_decomposition_disabled
    # to keep the opaque op in exported programs), we decompose unconditionally here.
    # The decomposition targets (aten._sparse_semi_structured_mm, pad, narrow) are real
    # ATen ops with proper meta registrations and are fully serializable, so export
    # consumers benefit from seeing the underlying ops rather than an opaque custom op.
    def _cutlass_mm_decomp(mode, _op, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Decomposition for FunctionalTensorMode: returns NotImplemented
        # if unrecognized tensor
        # types are present, else runs the implementation under the mode.
        if _has_unrecognized_tensor_types(types):
            return NotImplemented
        with mode:
            return _cutlass_mm_impl(*args, **kwargs)

    cutlass_mm.register_torch_dispatch(FunctionalTensorMode, _cutlass_mm_decomp)


_cusparselt_mm_registered = False


def _ensure_cusparselt_mm_registered():
    # [FUNCTION DOC] HELPER: One-time lazy registration of the semi_structured::cusparselt_mm
    # custom op and its
    # fake implementation, guarded by a module-global flag; later calls return immediately.
    """Lazily register the cusparselt_mm custom op."""
    global _cusparselt_mm_registered
    if _cusparselt_mm_registered:
        return
    _cusparselt_mm_registered = True

    from torch.library import custom_op

    @custom_op("semi_structured::cusparselt_mm", mutates_args=())
    def cusparselt_mm(
        dense: torch.Tensor,
        packed: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        fuse_transpose: bool,
        alg_id: int,
        should_transpose_dense: bool = False,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Pads dense to the min-shape multiples, optionally transposes
        # it, runs the cuSPARSELt
        # sparse mm with bias, fuse_transpose and alg_id, then narrows away padding. Returns a
        # contiguous tensor when padded, otherwise the raw result.
        m, n = dense.shape
        to_pad_m = (-m) % min_rows
        to_pad_n = (-n) % min_cols
        need_pad = to_pad_m != 0 or to_pad_n != 0
        dense_padded = dense
        if need_pad:
            dense_padded = torch.nn.functional.pad(dense, (0, to_pad_n, 0, to_pad_m))
        mm_input = dense_padded.t() if should_transpose_dense else dense_padded
        res = torch._cslt_sparse_mm(
            packed,
            mm_input,
            bias=bias,
            transpose_result=fuse_transpose,
            alg_id=alg_id,
        )
        if need_pad:
            out_cols = m if should_transpose_dense else n
            if fuse_transpose:
                return res.narrow(0, 0, out_cols).contiguous()
            else:
                return res.narrow(1, 0, out_cols).contiguous()
        return res

    @cusparselt_mm.register_fake
    def _cusparselt_mm_fake(
        dense: torch.Tensor,
        packed: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        fuse_transpose: bool,
        alg_id: int,
        should_transpose_dense: bool,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Fake implementation: returns an empty tensor shaped by
        # fuse_transpose and
        # should_transpose_dense, with dense dtype and device.
        out_cols = dense.shape[0] if should_transpose_dense else dense.shape[1]
        output_shape = (
            (out_cols, out_features) if fuse_transpose else (out_features, out_cols)
        )
        return torch.empty(
            output_shape,
            dtype=dense.dtype,
            device=dense.device,
        )
