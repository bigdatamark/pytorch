# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# [MODULE DOC] Layout and register-permutation helpers for quack CuTe DSL GEMM epilogues.
# Contains tensor view utilities (transpose, select, expand, interleave), warp-shuffle based
# permutations of accumulator registers, conversions of MMA accumulator layouts to row/column
# (MN) views and to A-operand fragment layouts, vector partitioning helpers that map a shared
# memory vector onto MMA or copy partitions, and a scale-factor layout builder.
# No import-time side effects beyond imports.


import cutlass
import cutlass.cute as cute

from cutlass import Int32, const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    # [FUNCTION DOC] Return a view of a tensor with its first two modes swapped (rest unchanged).
    # Built by composing the tensor with an ordered layout of the swapped shape. No data moves.
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    # [FUNCTION DOC] Return a view of a tensor keeping only the listed top-level modes of its
    # layout, in the given order. Shares the iterator, so no data moves.
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def concat_to_interleave(a: cute.Tensor, dim: int) -> cute.Tensor:
    # [FUNCTION DOC] Reinterpret the given dimension, of even size 2N, as a hierarchical (2, N)
    # mode so that elements of the two halves alternate. Used to turn a concatenated gate/up
    # weight layout into an interleaved one. Returns a view with the same iterator; the new
    # inner stride is the half-size step and the outer one is the original stride.
    """Reshape a concat [first_half; second_half] layout to interleaved along `dim`.

    Splits dimension `dim` (size 2N) into hierarchical (2, N) so that elements
    from the first half and second half alternate: [first_0, second_0, first_1, ...].
    Used to convert gated MLP weight layout from concat [gate; up] to interleaved.
    """
    half = cute.size(a, mode=[dim]) // 2
    shape = (*a.shape[:dim], (2, half), *a.shape[dim + 1 :])
    stride = (*a.stride[:dim], (half * a.stride[dim], a.stride[dim]), *a.stride[dim + 1 :])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    # [FUNCTION DOC] Insert a new broadcast mode of the given size (static or Int32) at position
    # dim, with stride zero, so every index along it aliases the same data. Returns a view.
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    # [FUNCTION DOC] In-place permutation of a 16-bit accumulator register tensor across the four
    # lanes of a quad, used for gated epilogues. The tensor is recast as 32-bit words taken in
    # pairs; each lane swaps the pair order depending on whether its quad index is 0/3 or 1/2,
    # exchanges words with a computed partner lane by warp shuffle restricted to the quad, then
    # recombines 16-bit halves with byte-permute selectors that also depend on the lane class.
    # Fails an assertion if the element width is not 16 or the element count is not a
    # multiple of 4.
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = cute.arch.prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = cute.arch.prmt(upper0, lower0, selector_lower)


@cute.jit
def permute_Cregs_b32_for_stsm(t: cute.Tensor) -> None:
    # [FUNCTION DOC] In-place permutation of a 32-bit register tensor within each quad of four
    # lanes so the per-thread layout of consecutive pairs becomes a spread layout, allowing a
    # matrix store instruction instead of 64-bit stores without bank conflicts. Processes groups
    # of four elements: two shuffle rounds on element pairs, then swaps the middle two elements.
    # Fails an assertion if the element width is not 32 or the element count is not a
    # multiple of 4.
    """Permute and shuffle within 4 threads to change the layout from
     T0 | T1  | T2  | T3
    a b | c d | e f | g h
    to
    T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3
    a  | b  | c  | d  | e  | f  | g  | h
    This is so that we can use STSM (instead of STS.64) to store C registers without bank conflict.
    """

    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    # left_map = [0, 2, 1, 3]
    # right_map = [2, 0, 3, 1]
    # indexing isn't supported so we have to do arithmetic
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b10

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            # a b | c d | e f | g h -> a b | c d | f e | h g
            left0 = left if quad_idx < 2 else right
            right0 = right if quad_idx < 2 else left
            # a b | c d | f e | h g -> a b | f d | c e | h g
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            # a b | f d | c e | h g -> a e | f b | c g | h d
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            # a e | f b | c g | h d -> a e | b f | c g | d h
            t[i * 4 + r * 2 + 0] = left0 if quad_idx % 2 == 0 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx % 2 == 0 else left0
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]


@cute.jit
def permute_Cregs_b32_for_ldsm(t: cute.Tensor) -> None:
    # [FUNCTION DOC] Inverse of the store-oriented permutation: in-place rearrangement of a 32-bit
    # register tensor within each lane quad, restoring paired per-thread layout after a matrix
    # load. For each group of four elements it first swaps the middle two, then does two shuffle
    # rounds on element pairs. Fails an assertion if the element width is not 32 or the element
    # count is not a multiple of 4.
    """Permute and shuffle within 4 threads to change the layout from
    T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3
    a  | b  | c  | d  | e  | f  | g  | h
    to
     T0 | T1  | T2  | T3
    a b | c d | e f | g h
    This is so that we can use LDSM (instead of LDS.64) to store C registers without bank conflict.
    """

    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    # left_map = [0, 2, 1, 3]
    # right_map = [1, 3, 0, 2]
    # indexing isn't supported so we have to do arithmetic
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b01

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    # This is just the inverse of permute_Cregs_b32_for_stsm
    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            # a e | b f | c g | d h -> a e | f b | c g | h d
            left0 = left if quad_idx % 2 == 0 else right
            right0 = right if quad_idx % 2 == 0 else left
            # a e | f b | c g | h d -> a b | f d | c e | h g
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            # a b | f d | c e | h g -> a b | c d | f e | h g
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            # a b | c d | f e | h g -> a b | c d | e f | g h
            t[i * 4 + r * 2 + 0] = left0 if quad_idx < 2 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx < 2 else left0


@cute.jit
def concat_layout(*layouts: cute.Layout) -> cute.Layout:
    # [FUNCTION DOC] Combine several layouts into one whose modes are the given layouts: the shape
    # is the tuple of the input shapes and the stride is the tuple of the input strides.
    return cute.make_layout(
        tuple(l.shape for l in layouts),
        stride=tuple(l.stride for l in layouts),
    )


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    # [FUNCTION DOC] Convert an MMA accumulator layout into a row/column view. Mode zero of the
    # input holds the per-instruction values; its second element joins MMA_M into the row mode,
    # and its remaining elements join MMA_N into the column mode, with trailing modes kept.
    # With transpose set, the two resulting modes are exchanged. The result is the composition
    # of the input layout with this column-major-derived mapping.
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    # [FUNCTION DOC] Return a tensor view of an accumulator with its layout converted to the
    # row/column form by the layout conversion helper; transpose swaps row and column modes.
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    # [FUNCTION DOC] Reshape an accumulator tensor to its row/column view using the same layout
    # conversion as the make-view helper (identical behavior); transpose swaps the two modes.
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # [FUNCTION DOC] Convert an accumulator layout of a first GEMM into the layout accepted as the
    # A operand fragment of a back-to-back second GEMM. If mode zero has rank 3 (SM90 style) the
    # third element is split by two when even (else by one) and the inner part is moved into the
    # first mode; otherwise (SM80 style) the last mode must be even (assertion) and is split into
    # a pair, moving one factor into the first mode. Returns the new layout. FP8 on SM90 is a TODO.
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # If N / 8 is odd, we'll convert to ((2, 2, 1), MMA_M, N / 8, MMA_N).
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(
            acc_layout, ((None, None, div), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        assert acc_layout.shape[2] % 2 == 0
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    # [FUNCTION DOC] Return a view of the accumulator tensor with the A-operand fragment layout
    # given by the accumulator-to-fragment layout conversion.
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def convert_layout_zero_stride(
    input: cute.Tensor | cute.Layout, ref_layout: cute.Layout
) -> cute.Layout:
    # [FUNCTION DOC] Regroup the modes of a layout (or of a tensor, returned as a tensor) into two
    # groups using a reference layout: modes whose reference stride is nonzero first, then those
    # with zero stride. Both inputs are flattened and are assumed to have matching rank. If no
    # mode has a nonzero reference stride, the first group is a single mode of size one with
    # stride zero.
    layout = input.layout if const_expr(isinstance(input, cute.Tensor)) else input
    # Group the modes with non-zero stride in the ref_layout together,
    # and the modes with zero stride together
    layout_flat = cute.flatten(layout)
    ref_layout_flat = cute.flatten(ref_layout)
    nonzero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride != 0]
    zero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride == 0]
    # There's an edge case when all modes are zero stride
    new_shape = (
        tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    out_layout = cute.make_layout(new_shape, stride=new_stride)
    if const_expr(isinstance(input, cute.Tensor)):
        return cute.make_tensor(input.iterator, out_layout)
    else:
        return out_layout


def mma_partition_C_vec(
    sVec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    # [FUNCTION DOC] Partition a shared-memory vector across an MMA thread partition of C.
    # The vector must be 2D (length, stage) with unit stride on mode zero (asserted). It is
    # broadcast with a zero-stride mode of size expand_shape, as a column vector (length rows) or
    # row vector, then partitioned for C by the thread MMA and viewed in row/column form.
    # Returns the slice at index zero of the broadcast mode, keeping the other two modes.
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def mma_partition_A_vec(
    sVec: cute.Tensor, thr_mma: cute.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    # [FUNCTION DOC] Same as the C-partition vector helper but uses the A-operand partition of the
    # thread MMA. Requires a 2D vector with unit stride on mode zero (asserted); broadcasts it
    # over expand_shape as a column or row vector, partitions it, converts to the row/column
    # view, and returns the slice keeping the length and stage modes.
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_A(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_S_vec(
    sVec: cute.Tensor, thr_copy: cute.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    # [FUNCTION DOC] Partition a shared-memory vector with the source side of a thread copy.
    # Requires a 2D vector with unit stride on mode zero (asserted); broadcasts it over
    # expand_shape as a column or row vector, applies the thread copy source partition, converts
    # to row/column view and returns the slice at index zero of the broadcast mode.
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_S(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_D_vec(
    sVec: cute.Tensor, thr_copy: cute.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    # [FUNCTION DOC] Destination-side counterpart of the source copy-partition helper: broadcasts a
    # 2D vector (unit stride on mode zero, asserted) over expand_shape as column or row vector,
    # applies the thread copy destination partition, converts to row/column view, and returns the
    # slice at index zero of the broadcast mode.
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_D(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def tile_atom_to_shape_SF_strided(
    shape: cute.Shape,
    sf_vec_size: int,
    sf_strides,
) -> cute.Layout:
    # [FUNCTION DOC] Build a block-scaled scale-factor layout for an operand shape using the real
    # strides of the scale tensor. Starts from the hardware 512-element atom for the vector
    # size, tiles it over ceil-divided row blocks (128) and K blocks (4 vectors), taking outer
    # strides from entries one and two of the scale strides divided by 512. For a rank-3 shape a
    # batch mode with stride from entry zero is appended. Returns the layout.
    """Build an SFA/SFB layout matching `shape` (A or B operand shape) but
    honoring the scale tensor's actual strides instead of hardcoded packed
    ones.

    Mirrors `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF(shape,
    sf_vec_size)`, except outer-mode strides come from `sf_strides` (pass
    `mSFA.stride` / `mSFB.stride` directly). The inner 512-B atom
    `((32, 4), (sf_vec_size, 4)) : ((16, 4), (0, 1))` is hardware-fixed.

    Implementation uses `cute.blocked_product(atom, outer)`; `blocked_product`
    scales the outer layout's strides by `cosize(atom) == 512`, so we divide
    the byte strides by 512 (one tile) before handing them in.

    Args:
        shape: A/B operand shape. Rank-3 `(m/n, k, l)` or rank-2
            `(total_mn, k)` (varlen_m).
        sf_vec_size: Scale factor vector size (16 or 32).
        sf_strides: Strides of the scale tensor, which has logical shape
            `(L, rmn, rk, 512)` (rank 4). Only `sf_strides[0..2]` are used:
            `sf_strides[1]` as the rmn stride, `sf_strides[2]` as the rk
            stride, and `sf_strides[0]` as the L stride (only for rank-3
            `shape`).
    """
    from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk

    atom = BlockScaledBasicChunk(sf_vec_size).layout
    rmn = cute.ceil_div(shape[0], 128)
    rk = cute.ceil_div(shape[1], sf_vec_size * 4)
    outer = cute.make_layout((rmn, rk), stride=(sf_strides[1] // 512, sf_strides[2] // 512))
    sf_layout = cute.blocked_product(atom, outer)
    if const_expr(len(shape) == 3):
        sf_layout = cute.append(sf_layout, cute.make_layout(shape[2], stride=sf_strides[0]))
    return sf_layout
