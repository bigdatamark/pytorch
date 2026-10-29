# [MODULE DOC] Geometry helpers shared by NVGEMM reduction kernels written in CuTeDSL.
# Exports get_lane_warp_layouts (derives lane and warp layouts of an epilogue tiled copy)
# and partition_for_epilogue (thin forwarder to the vendored quack sm90 helper).
# No import-time side effects; cutlass and quack are imported lazily inside the functions.
"""CuTeDSL geometry helpers shared by NVGEMM reduction kernels."""

__all__ = ["get_lane_warp_layouts", "partition_for_epilogue"]


def get_lane_warp_layouts(tiled_copy, reference_src=True):
    # [FUNCTION DOC] Computes, for a tiled copy, which lane and which warp owns each (m, n)
    # element of the epilogue tile. Uses the source thread-value layout when reference_src
    # is true, otherwise the destination one. Inverts that layout, restricts it to the tile
    # extent, then composes with a lane-index and a warp-index mapping over the copy's warps.
    # Returns a pair (lane_layout_mn, warp_layout_mn). Imports cutlass/quack lazily.
    """Derive two-dimensional lane and warp layouts for an epilogue copy."""
    import cutlass.cute as cute

    from torch._vendor.quack import layout_utils

    layout_tv = (
        tiled_copy.layout_src_tv_tiled
        if reference_src
        else tiled_copy.layout_dst_tv_tiled
    )
    ref_layout = cute.right_inverse(layout_tv)
    tile_m = cute.size(tiled_copy.tiler_mn[0])
    tile_n = cute.size(tiled_copy.tiler_mn[1])
    ref_layout_mn = cute.composition(ref_layout, cute.make_layout((tile_m, tile_n)))

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
    tv_to_lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref_to_lane = cute.composition(tv_to_lane, ref_layout_mn)
    lane_m = cute.filter(cute.select(ref_to_lane, [0]))
    lane_n = cute.filter(cute.select(ref_to_lane, [1]))
    lane_layout_mn = layout_utils.concat_layout(lane_m, lane_n)

    tv_to_warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref_to_warp = cute.composition(tv_to_warp, ref_layout_mn)
    warp_m = cute.filter(cute.select(ref_to_warp, [0]))
    warp_n = cute.filter(cute.select(ref_to_warp, [1]))
    warp_layout_mn = layout_utils.concat_layout(warp_m, warp_n)
    return lane_layout_mn, warp_layout_mn


def partition_for_epilogue(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Lazily imports and forwards all arguments to the quack sm90_utils
    # partition_for_epilogue function and returns its result unchanged.
    from torch._vendor.quack.sm90_utils import partition_for_epilogue as partition

    return partition(*args, **kwargs)
