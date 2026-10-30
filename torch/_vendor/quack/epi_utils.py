# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Shared helpers for GEMM epilogue mixin classes: stride-alignment assumptions for
# broadcast vectors and construction of TMA copy atoms plus staged shared-memory layouts for extra
# epilogue tensors. No import-time side effects beyond imports.
"""Epilogue utilities: shared helpers for epilogue mixin classes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils

from . import sm90_utils
from . import copy_utils


def assume_stride_divisibility(tensor):
    # [FUNCTION DOC] Return a copy of the tensor whose dynamic strides carry a divisibility hint
    # of 32 bits worth of elements (32 divided by element width); static strides are left
    # unchanged. Returns None when given None. The data pointer is reused, so no data is copied.
    """Assume all strides are divisible by 32 bits (except static strides).

    Used for broadcast vectors and similar tensors where stride alignment is guaranteed.
    Returns a new tensor with the assumed strides.
    """
    if tensor is None:
        return None
    new_stride = tuple(
        cute.assume(s, divby=32 // tensor.element_type.width) if not cute.is_static(s) else s
        for s in tensor.stride
    )
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride))


def assume_broadcast_strides(*tensors):
    # [FUNCTION DOC] Map the stride divisibility helper over several tensors and return a list of
    # results, with None entries preserved as None.
    """Apply stride divisibility assumptions to multiple broadcast vectors.

    Returns a list with None preserved for None inputs.
    """
    return [assume_stride_divisibility(t) for t in tensors]


def setup_epi_tensor(gemm, tensor, epi_tile=None, op_type="store", stage=None):
    # [FUNCTION DOC] Build the TMA atom, TMA tensor and staged epilogue shared-memory layout for
    # one extra tensor. Epi tile and stage count default to the gemm values; the smem layout
    # helper is taken from the sm100 module when gemm.arch is at least 100, else from sm90. For
    # store ops on rank-2 tensors the tensor is first converted to a ragged tensor for TMA (row
    # offset baked into the descriptor); loads skip this. Returns a 4-tuple of TMA atom, TMA
    # tensor, staged smem layout and the epi tile used.
    """Create copy metadata + smem layout for a supplemental epilogue tensor.

    Args:
        gemm: The GEMM object (provides arch, epi_stage, and epilogue layout helpers).
        tensor: The global memory tensor to set up for the epilogue.
        epi_tile: Epilogue tile shape. Defaults to gemm.epi_tile.
        op_type: "store" or "load".

    Returns:
        (copy_atom, tensor, smem_layout_staged, epi_tile). copy_atom is None for pre-TMA archs.
    """
    if epi_tile is None:
        epi_tile = gemm.epi_tile
    if stage is None:
        stage = gemm.epi_stage
    dtype = tensor.element_type
    layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
    utils_cls = sm100_utils if gemm.arch >= 100 else sm90_utils
    smem_layout_staged = utils_cls.make_smem_layout_epi(dtype, layout, epi_tile, stage)
    # Ragging-for-TMA is for varlen_m stores that need a per-batch row offset baked
    # into the TMA descriptor. Loads don't currently support varlen_m, so skip the
    # ragging conversion.
    tma_input = (
        copy_utils.create_ragged_tensor_for_tma(tensor, ragged_dim=0, ptr_shift=True)
        if op_type != "load" and cute.rank(tensor) == 2
        else tensor
    )
    tma_atom, tma_tensor = gemm._make_tma_epi_atoms_and_tensors(
        tma_input,
        smem_layout_staged,
        epi_tile,
        op_type=op_type,
    )
    return tma_atom, tma_tensor, smem_layout_staged, epi_tile
