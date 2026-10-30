# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Variable-length (ragged batch) support for the quack GEMM kernels. Defines the
# argument bundle holding optional cumulative sequence-length tensors and a gather index, and a
# manager class that, per batch index, computes the per-group M or K length and slices the A, B,
# epilogue and block-scale-factor tensors accordingly. Import-time side effects: none beyond
# imports (the named tuple is registered through the mlir_namedtuple decorator).

from typing import Optional, NamedTuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr

from . import copy_utils
from .cute_dsl_utils import mlir_namedtuple


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class VarlenArguments(NamedTuple):
    # [CLASS DOC] Named tuple grouping the optional varlen inputs passed to a GEMM call: cumulative
    # sequence lengths along M, cumulative sequence lengths along K, and a row-gather index for A.
    # All fields default to None (meaning the feature is disabled).
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None


class VarlenManager:
    # [CLASS DOC] Compile-time-specialized helper that handles ragged (variable length) batches in
    # a GEMM. Holds a Params object plus static M and K lengths, and the previous batch index and
    # group-changed flag. The flags varlen_m, varlen_k and gather_A are derived from which Params
    # fields are not None. Methods return per-batch lengths and per-batch views of tensors. It can
    # be flattened to and rebuilt from MLIR values so it can cross jit boundaries.
    @dataclass
    class Params:
        # [CLASS DOC] Dataclass holding the optional device tensors used by VarlenManager:
        # cumulative sequence lengths for M and K and the gather index for A (all default None).
        # Built from a VarlenArguments via the create static method.
        cu_seqlens_m: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        mAIdx: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(args: VarlenArguments, *, loc=None, ip=None) -> "VarlenManager.Params":
            # [FUNCTION DOC] Jit-compiled factory that copies the three tensors of a
            # VarlenArguments (M cumulative lengths, K cumulative lengths, A gather index)
            # into a new Params object and returns it. No validation here. The loc and ip
            # arguments are unused.
            return VarlenManager.Params(
                cu_seqlens_m=args.mCuSeqlensM,
                cu_seqlens_k=args.mCuSeqlensK,
                mAIdx=args.mAIdx,
            )

    def __init__(
        self,
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        last_batch_idx: Int32 = Int32(-1),
        is_group_changed: Boolean = Boolean(True),
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Stores the params, static M and K lengths, last batch index and group-
        # changed flag (defaults: minus one and true), plus loc and ip. Derives three compile-time
        # constant flags: varlen_m if the M cumulative lengths exist, varlen_k if the K ones
        # exist, gather_A if the A gather index exists. Does not check that varlen_m and varlen_k
        # are not both set.
        self.params = params
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._last_batch_idx = last_batch_idx
        self._is_group_changed = is_group_changed
        self.varlen_m = const_expr(params.cu_seqlens_m is not None)
        self.varlen_k = const_expr(params.cu_seqlens_k is not None)
        self.gather_A = const_expr(params.mAIdx is not None)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: VarlenArguments, *, loc=None, ip=None) -> Params:
        # [FUNCTION DOC] Converts VarlenArguments to a Params object. Raises AssertionError if both
        # the M and K cumulative-length tensors are provided (only one varlen mode is supported).
        # Returns the Params built by Params.create.
        assert not (args.mCuSeqlensM is not None and args.mCuSeqlensK is not None), (
            "Only support either varlen_m or varlen_k"
        )
        return VarlenManager.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenManager":
        # [FUNCTION DOC] Jit-compiled factory returning a new VarlenManager from params and the
        # static M and K lengths. The loc and ip arguments are accepted but not forwarded.
        return VarlenManager(params, len_m_static=len_m_static, len_k_static=len_k_static)

    def len_m(self, batch_idx: Int32) -> Int32:
        # [FUNCTION DOC] Returns the M extent for a batch: the difference of adjacent entries of
        # the M cumulative-length tensor when varlen_m is enabled, otherwise the static M length.
        if const_expr(self.varlen_m):
            return self.params.cu_seqlens_m[batch_idx + 1] - self.params.cu_seqlens_m[batch_idx]
        else:
            return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        # [FUNCTION DOC] Returns the K extent for a batch: the difference of adjacent entries of
        # the K cumulative-length tensor when varlen_k is enabled, otherwise the static K length.
        if const_expr(self.varlen_k):
            return self.params.cu_seqlens_k[batch_idx + 1] - self.params.cu_seqlens_k[batch_idx]
        else:
            return self._len_k_static

    def offset_batch_A(self, mA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the 2D (M, K) view of tensor A for one batch. With varlen_m,
        # shifts the M origin by the batch's cumulative M offset. With varlen_k, shifts the K
        # origin by the K offset when the tensor has rank 2; for higher rank it builds a ragged
        # tensor through copy_utils.offset_ragged_tensor on dimension 1 (pointer shifted only when
        # rank is 3). Without varlen, selects the batch index along the last mode.
        params = self.params
        if const_expr(self.varlen_m):
            mA_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx], None), mA_mkl)
        elif const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mA_mkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mA_mk = cute.domain_offset((None, offset), mA_mkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                # rank 3 = 1-extra-dim (ptr_shift), rank 4 = 2-extra-dim
                ptr_shift = const_expr(ragged_rank == 3)
                mA_mk = copy_utils.offset_ragged_tensor(
                    mA_mkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mA_mk = mA_mkl[None, None, batch_idx]
        return mA_mk

    def offset_batch_AIdx(self, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the per-batch slice of the A gather-index tensor: offset by the
        # batch's cumulative M offset (varlen_m) or K offset (varlen_k), otherwise the column
        # selected by the batch index in the 2D index tensor.
        params = self.params
        if const_expr(self.varlen_m):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_k):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_k[batch_idx],), params.mAIdx)
        else:
            mAIdx_mk = params.mAIdx[None, batch_idx]
        return mAIdx_mk

    def offset_batch_SFA(self, mSFA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the batch view of the block scale-factor tensor for A. For
        # varlen_m or varlen_k the offset is the cumulative length divided by 128 (integer
        # division) plus the batch index, applied as a tile-unit shift on the M (resp. K) mode via
        # a compound coordinate; otherwise selects the batch index along the last mode.
        """Offset SFA by padded per-expert offset (dQaccum-style).

        The padded offset, in tile units (128 source-M or source-K per tile),
        is simply `cu_seqlens[b] // 128 + b`. (Algebraically identical to
        `(cu_seqlens[b] + b*128) // 128 * 128` / 128.) We pass it as a
        compound coord `(0, offset_tile)` to `domain_offset` so the outer
        rm/rk mode is shifted in tile units — no `* 128` needed, and the
        compiler sees the tile alignment natively.
        """
        params = self.params
        tile = 128
        if const_expr(self.varlen_m):
            offset_tile = params.cu_seqlens_m[batch_idx] // tile + batch_idx
            return cute.domain_offset(((0, offset_tile), None), mSFA_mkl)
        elif const_expr(self.varlen_k):
            offset_tile = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, (0, offset_tile)), mSFA_mkl)
        else:
            return mSFA_mkl[None, None, batch_idx]

    def offset_batch_SFB(self, mSFB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the batch view of the block scale-factor tensor for B. Only
        # varlen_k shifts it (tile-unit offset of cumulative K length divided by 128 plus
        # batch index on the K mode); in all other cases it simply selects the batch index
        # along the last mode.
        """Offset SFB by padded per-expert K offset (varlen_k only)."""
        params = self.params
        tile = 128
        if const_expr(self.varlen_k):
            offset_tile = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, (0, offset_tile)), mSFB_nkl)
        else:
            return mSFB_nkl[None, None, batch_idx]

    def offset_batch_B(self, mB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the 2D (N, K) view of tensor B for one batch. Only varlen_k
        # changes behavior: rank-2 tensors get a K-origin shift, higher-rank ones go through
        # copy_utils.offset_ragged_tensor on dimension 1 (pointer shift only for rank 3). Otherwise
        # selects the batch index along the last mode.
        params = self.params
        if const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mB_nkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mB_nk = cute.domain_offset((None, offset), mB_nkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mB_nk = copy_utils.offset_ragged_tensor(
                    mB_nkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mB_nk = mB_nkl[None, None, batch_idx]
        return mB_nk

    def offset_batch_epi(self, mD_mnl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        # [FUNCTION DOC] Returns the 2D (M, N) view of an epilogue tensor (such as the output) for
        # one batch. Only varlen_m changes behavior: rank-2 tensors get an M-origin shift, higher-
        # rank ones go through copy_utils.offset_ragged_tensor on dimension 0 (pointer shift only
        # for rank 3). Otherwise selects the batch index along the last mode.
        params = self.params
        if const_expr(self.varlen_m):
            offset = params.cu_seqlens_m[batch_idx]
            ragged_rank = const_expr(cute.rank(mD_mnl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mD_mn = cute.domain_offset((offset, None), mD_mnl)
            else:
                length = params.cu_seqlens_m[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mD_mn = copy_utils.offset_ragged_tensor(
                    mD_mnl,
                    offset,
                    length,
                    ragged_dim=0,
                    ptr_shift=ptr_shift,
                )
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        return mD_mn

    def __extract_mlir_values__(self):
        # [FUNCTION DOC] Flattens params and the four scalar fields into one list of MLIR values
        # and returns it. Side effect: records on the instance how many values each member
        # contributed (stored in _values_pos) so they can be split again later.
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self._len_m_static,
            self._len_k_static,
            self._last_batch_idx,
            self._is_group_changed,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        # [FUNCTION DOC] Rebuilds a VarlenManager from a flat list of MLIR values, splitting it
        # using the per-member counts saved by the extract method (so that must have been called
        # first on this instance). Returns a new instance of the same class, passing the saved loc.
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self._len_m_static,
                self._len_k_static,
                self._last_batch_idx,
                self._is_group_changed,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
