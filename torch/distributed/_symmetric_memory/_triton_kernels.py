# [MODULE DOC] Triton kernels for symmetric-memory collectives. Defines, only when
# Triton is available, a kernel that sums (optionally averages) per-rank partial
# buffers into one output.
from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_partials_first_dim_kernel(
        partials,
        out,
        shard_elems: tl.constexpr,
        group_size: tl.constexpr,
        do_avg: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # [FUNCTION DOC] Triton JIT kernel. Each program handles a block of elements of one
        # shard. For every rank in the group it loads that rank's slice from the partials
        # buffer (laid out as consecutive per-rank shards of shard_elems), masked to bounds
        # with zero fill, and accumulates in float32. If do_avg is set, divides by group_size.
        # Stores the result to out once, cast to out element dtype, masked to bounds. Args:
        # partials, out pointers, shard_elems, group_size, do_avg, BLOCK (compile-time
        # constants).
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < shard_elems
        acc = tl.zeros((BLOCK,), tl.float32)
        for rank in tl.static_range(0, group_size):
            vals = tl.load(
                partials + rank * shard_elems + offsets,
                mask=mask,
                other=0.0,
            )
            acc += vals.to(tl.float32)
        if do_avg:
            acc /= group_size
        # The accumulation stays in fp32 and is rounded exactly once, here.
        tl.store(out + offsets, acc.to(out.dtype.element_ty), mask=mask)
