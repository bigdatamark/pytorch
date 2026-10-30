# [MODULE DOC] Benchmark driver for block-sparse (bsr) matrix multiplication kernels, comparing
# dense matmul, torch matmul, bsr dense mm, bsr scatter mm variants and linear against random
# blocked-sparse operands. Provides a generator of random block-sparse test data and one timing
# function per operation; the main block (requires triton) parses a large set of comma-separated
# sweep options (sizes, block sizes, sparsity, kernel tuning parameters, dtype, device, repeats,
# output file, and a star mode that sweeps around heuristic kernel parameters), loops over all
# valid combinations, prints time and TFLOPS lines marking the best result and results close to
# the dense baseline, and prints the ten best lines at exit.
# KNOWN ISSUE: the shared timing worker reads the sizes m, k and n from module globals that
#     exist only when the main block has run, so the timing functions cannot be used after
#     a plain import.
import torch
from torch._inductor.runtime.benchmarking import benchmarker


def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    # [FUNCTION DOC] Builds a random tensor of 0 and 1 blocks with a requested fraction of zero
    # blocks. Shape is batch by M by N (the batch dimension is dropped when the batch size is
    # zero); M and N must be divisible by the block size. First draws each block as one with
    # probability one minus the sparsity, then randomly flips entries so the number of nonzero
    # blocks equals the expected count (integer part of the density times the number of blocks),
    # and finally expands every block to the block size along the last two dimensions. Raises an
    # assertion error if the sparsity is outside zero to one or M or N is not divisible by the
    # block size.
    if not (sparsity <= 1.0 and sparsity >= 0.0):
        raise AssertionError(
            f"sparsity should be a value between 0 and 1, but got {sparsity}"
        )
    if M % blocksize[0] != 0:
        raise AssertionError(
            f"M ({M}) must be divisible by blocksize[0] ({blocksize[0]})"
        )
    if N % blocksize[1] != 0:
        raise AssertionError(
            f"N ({N}) must be divisible by blocksize[1] ({blocksize[1]})"
        )
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0) :]
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    expected_nnz = int((1 - sparsity) * M * N / (blocksize[0] * blocksize[1]))
    nonzero_indices = A.flatten().nonzero()
    actual_nnz = nonzero_indices.shape[0]
    if actual_nnz > expected_nnz:
        selected_nonzeros = torch.randperm(actual_nnz)[: actual_nnz - expected_nnz]
        A.flatten()[nonzero_indices[selected_nonzeros]] = 0
    elif actual_nnz < expected_nnz:
        zero_indices = (A == 0).flatten().nonzero()
        selected_zeros = torch.randperm(zero_indices.shape[0])[
            : expected_nnz - actual_nnz
        ]
        A.flatten()[zero_indices[selected_zeros]] = 1
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    return A


def _test_worker(test_func):
    # [FUNCTION DOC] HELPER: Times a callable on the gpu with the inductor benchmarker (500
    # warmup and 100 timed units) and uses the time in milliseconds it returns. Computes the
    # throughput in TFLOPS as two times m times k times n over that time, using the module
    # global sizes. Returns the pair of time and TFLOPS.
    ms, ms_min, ms_max = benchmarker.benchmark_gpu(test_func, warmup=500, rep=100)

    tflops = 2 * m * k * n * 1e-12 / (ms * 1e-3)
    return ms, tflops


def test_dense_dense_mm(x, y, **meta):
    # [FUNCTION DOC] Benchmark of dense matmul: converts the first operand to dense once, then
    # times matmul with the second operand. Extra meta arguments are ignored. Returns time in
    # milliseconds and TFLOPS.
    def test_func(x=x.to_dense(), y=y):
        # [FUNCTION DOC] HELPER: Timed callable computing the dense matrix product of the
        # bound operands.
        return torch.matmul(x, y)

    return _test_worker(test_func)


def test_torch_matmul(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the generic matmul call with the operands as given (the first
    # may be sparse). Extra meta arguments are ignored. Returns time in milliseconds and TFLOPS.
    def test_func(x=x, y=y):
        # [FUNCTION DOC] HELPER: Timed callable computing the product of the bound operands with
        # the generic matmul.
        return torch.matmul(x, y)

    return _test_worker(test_func)


def test_bsr_dense_mm(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the triton bsr dense mm kernel with fixed tuning parameters
    # (group size 4, one stage, four warps), ignoring extra meta arguments. Imports the kernel
    # lazily. Returns time in milliseconds and TFLOPS.
    from torch.sparse._triton_ops import bsr_dense_mm

    def test_func(x=x, y=y):
        # [FUNCTION DOC] HELPER: Timed callable running the bsr dense mm kernel with the fixed
        # tuning parameters.
        return bsr_dense_mm(
            x, y, meta=dict(GROUP_SIZE_ROW=4, num_stages=1, num_warps=4)
        )

    return _test_worker(test_func)


def test_bsr_dense_mm_with_meta(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the triton bsr dense mm kernel using the tuning parameters
    # passed in as extra keyword arguments. Imports the kernel lazily. Returns time in
    # milliseconds and TFLOPS.
    from torch.sparse._triton_ops import bsr_dense_mm

    def test_func(x=x, y=y, meta=meta):
        # [FUNCTION DOC] HELPER: Timed callable running the bsr dense mm kernel with the bound
        # tuning parameters.
        return bsr_dense_mm(x, y, meta=meta)

    return _test_worker(test_func)


def test_bsr_scatter_mm2(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the bsr scatter mm kernel with index data in the scatter
    # format, computed once outside the timed region using the extra keyword arguments as
    # parameters. Imports lazily. Returns time in milliseconds and TFLOPS.
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    indices_data = bsr_scatter_mm_indices_data(
        x, y, indices_format="scatter_mm", **meta
    )

    def test_func(x=x, y=y):
        # [FUNCTION DOC] HELPER: Timed callable running the bsr scatter mm kernel with
        # precomputed index data.
        return bsr_scatter_mm(x, y, indices_data=indices_data)

    return _test_worker(test_func)


def test_bsr_scatter_mm6(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the bsr scatter mm kernel with index data in the compressed
    # strided format, computed once outside the timed region using the extra keyword arguments
    # as parameters. Imports lazily. Returns time in milliseconds and TFLOPS.
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    indices_data = bsr_scatter_mm_indices_data(
        x, y, indices_format="bsr_strided_mm_compressed", **meta
    )

    def test_func(x=x, y=y):
        # [FUNCTION DOC] HELPER: Timed callable running the bsr scatter mm kernel with
        # precomputed index data.
        return bsr_scatter_mm(x, y, indices_data=indices_data)

    return _test_worker(test_func)


def test_bsr_scatter_mm(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the bsr scatter mm kernel including index data construction
    # (compressed strided format, extra keyword arguments as parameters) inside the timed
    # region. Imports lazily. Returns time in milliseconds and TFLOPS.
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    def test_func(x=x, y=y):
        # [FUNCTION DOC] HELPER: Timed callable that builds the index data and then runs the bsr
        # scatter mm kernel.
        indices_data = bsr_scatter_mm_indices_data(
            x, y, indices_format="bsr_strided_mm_compressed", **meta
        )
        return bsr_scatter_mm(x, y, indices_data=indices_data)

    return _test_worker(test_func)


def test_linear(x, y, **meta):
    # [FUNCTION DOC] Benchmark of the functional linear op with the transposed second operand as
    # input and the first operand as weight. Extra meta arguments are ignored. Returns time in
    # milliseconds and TFLOPS.
    import torch.nn.functional as F

    def test_func(x=x, y=y.transpose(-2, -1)):
        # [FUNCTION DOC] HELPER: Timed callable applying linear to the bound operands.
        return F.linear(y, x)

    return _test_worker(test_func)


if __name__ == "__main__":
    import argparse
    import atexit
    import itertools
    import sys

    import triton

    from torch.testing import make_tensor

    torch.manual_seed(0)

    def integer_list(a):
        # [FUNCTION DOC] HELPER: Parses a comma-separated string into a list of integers.
        return list(map(int, a.split(",")))

    def float_list(a):
        # [FUNCTION DOC] HELPER: Parses a comma-separated string into a list of floats. Not used
        # elsewhere in this script.
        return list(map(float, a.split(",")))

    def integer_or_float_list(a):
        # [FUNCTION DOC] HELPER: Parses a comma-separated list where each item is an integer, a
        # float (item contains a decimal point), a half-open range written as start colon end,
        # or a stepped range written as start colon end colon step. Ranges expand to the
        # integers they cover. Returns the flat list.
        lst = []
        for n in a.split(","):
            if n.count(":") == 1:
                start, end = map(int, n.split(":"))
                lst.extend(range(start, end))
            elif n.count(":") == 2:
                start, end, step = map(int, n.split(":"))
                lst.extend(range(start, end, step))
            elif "." in n:
                lst.append(float(n))
            else:
                lst.append(int(n))
        return lst

    parser = argparse.ArgumentParser(description="SpTritonOps")

    parser.add_argument(
        "--ops",
        default="dense_dense_mm,bsr_dense_mm,bsr_scatter_mm6",
        type=str,
    )
    parser.add_argument("--b", default="0", type=int)

    parser.add_argument("--m", default="1024", type=integer_list)
    parser.add_argument("--k", default=None, type=integer_list)
    parser.add_argument("--n", default=None, type=integer_list)
    parser.add_argument("--bm", default="16", type=integer_list)
    parser.add_argument("--bk", default=None, type=integer_list)
    parser.add_argument("--tile_m", default=None, type=integer_list)
    parser.add_argument("--tile_n", default=None, type=integer_list)
    parser.add_argument("--split_n", default=None, type=integer_list)
    parser.add_argument("--group_size", default=None, type=integer_list)
    parser.add_argument("--num_warps", default=None, type=integer_list)
    parser.add_argument("--num_stages", default=None, type=integer_list)
    parser.add_argument("--sparsity", default="0.5", type=integer_or_float_list)
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--repeat", default="1", type=int)
    parser.add_argument("--outfile", default="stdout", type=str)
    parser.add_argument("--star", default=False, action="store_true")

    args = parser.parse_args()

    if args.outfile == "stdout":
        outfile = sys.stdout
        need_close = False
    elif args.outfile == "stderr":
        outfile = sys.stderr
        need_close = False
    else:
        outfile = open(args.outfile, "a")  # noqa: SIM115
        need_close = True

    ops = args.ops.split(",")

    b = args.b

    m_list = args.m or [1024]
    n_list = args.n or [None]
    k_list = args.k or [None]
    bm_list = args.bm or [16]
    bk_list = args.bk or [None]
    split_n_list = args.split_n or [None]
    tile_m_list = args.tile_m or [None]
    tile_n_list = args.tile_n or [None]
    group_size_list = args.group_size or [None]
    num_warps_list = args.num_warps or [None]
    num_stages_list = args.num_stages or [None]
    sparsity_list = args.sparsity or [0.5]
    dtype = getattr(torch, args.dtype)

    if args.star > 0:
        import torch.sparse._triton_ops

        if {len(m_list), len(n_list), len(k_list), len(bm_list), len(bk_list)} != {1}:
            raise AssertionError(
                "When --star is set, m, n, k, bm, bk lists must all have length 1"
            )
        m = m_list[0]
        n = n_list[0] or m
        k = k_list[0] or m
        bm = bm_list[0]
        bk = bk_list[0] or bm
        if "bsr_scatter_mm6" in ops:
            meta = torch.sparse._triton_ops.scatter_mm_meta(m, k, n, bm, bk)
        elif "bsr_dense_mm_with_meta" in ops:
            meta = torch.sparse._triton_ops.bsr_dense_mm_meta(m, k, n, bm, bk)
        else:
            raise NotImplementedError(f"--star not implemented for operations in {ops}")
        if "bsr_scatter_mm6" in ops:
            if split_n_list[0] is None:
                split_n_list = [
                    meta["SPLIT_N"] // 2,
                    meta["SPLIT_N"],
                    meta["SPLIT_N"] * 2,
                ][int(meta["SPLIT_N"] == 1) :]
            elif split_n_list[0] == 0:
                split_n_list = [meta["SPLIT_N"]]
            if tile_m_list[0] is None:
                tile_m_list = [meta["TILE_M"] // 2, meta["TILE_M"], meta["TILE_M"] * 2][
                    int(meta["TILE_M"] == 16) :
                ]
            elif tile_m_list[0] == 0:
                tile_m_list = [meta["TILE_M"]]
            if tile_n_list[0] is None:
                tile_n_list = [meta["TILE_N"] // 2, meta["TILE_N"], meta["TILE_N"] * 2][
                    int(meta["TILE_N"] == 16) :
                ]
            elif tile_n_list[0] == 0:
                tile_n_list = [meta["TILE_N"]]
            if group_size_list[0] is None:
                group_size_list = [
                    meta["GROUP_SIZE"] - 1,
                    meta["GROUP_SIZE"],
                    meta["GROUP_SIZE"] + 1,
                ][int(meta["GROUP_SIZE"] == 1) :]
            elif group_size_list[0] == 0:
                group_size_list = [meta["GROUP_SIZE"]]
        if "bsr_dense_mm_with_meta" in ops:
            if group_size_list[0] is None:
                group_size_list = [
                    meta["GROUP_SIZE_ROW"] - 1,
                    meta["GROUP_SIZE_ROW"],
                    meta["GROUP_SIZE_ROW"] + 1,
                ][int(meta["GROUP_SIZE_ROW"] == 1) :]
            elif group_size_list[0] == 0:
                group_size_list = [meta["GROUP_SIZE_ROW"]]
        if num_warps_list[0] is None:
            num_warps_list = [
                meta["num_warps"] // 2,
                meta["num_warps"],
                meta["num_warps"] * 2,
            ][int(meta["num_warps"] == 1) :]
        elif num_warps_list[0] == 0:
            num_warps_list = [meta["num_warps"]]
        if num_stages_list[0] is None:
            num_stages_list = [
                meta["num_stages"] - 1,
                meta["num_stages"],
                meta["num_stages"] + 1,
            ][int(meta["num_stages"] == 1) :]
        elif num_stages_list[0] == 0:
            num_stages_list = [meta["num_stages"]]

    device = args.device
    dense_dense_mm_sizes = set()
    target_performance = None
    performance_rtol = 1e-2

    best_messages = []

    @atexit.register
    def show_best_messages(best_messages=best_messages):
        # [FUNCTION DOC] HELPER: Exit handler registered with the atexit module. Prints a header
        # and the last ten collected best-result lines (the list bound at definition time), then
        # flushes standard output.
        print("TOP 10:")
        for m in best_messages[-10:]:
            print(m)
        sys.stdout.flush()

    for m, k, n, bm, bk, sparsity in itertools.product(
        m_list, k_list, n_list, bm_list, bk_list, sparsity_list
    ):
        k = k or m
        n = n or m
        bk = bk or bm

        if bm > m or bk > k:
            # Skip invalid parameter combinations
            continue

        blocksize = (bm, bk)

        if isinstance(sparsity, int):
            # integer sparsity value corresponds to desired nnz value
            sparsity = 1 - bk * bm * sparsity / (m * k)

        if sparsity > 1 or sparsity < 0:
            continue

        x = create_blocked_tensor(
            b, m, k, blocksize, sparsity, dtype, device
        ).to_sparse_bsr(blocksize)

        # recompute sparsity
        sparsity = 1 - bk * bm * x._nnz() / (m * k)

        y = make_tensor(k, n, dtype=dtype, device=device)

        bsr_size = f"{b}x{m}x{k}" if b > 0 else f"{k}x{n}"

        for op in ops:
            if op == "dense_dense_mm":
                if (m, k, n) in dense_dense_mm_sizes:
                    # Skip already benchmarked cases
                    continue
                dense_dense_mm_sizes.add((m, k, n))
            best_tflops = 0
            for (
                split_n,
                num_warps,
                num_stages,
                tile_m,
                tile_n,
                group_size,
            ) in itertools.product(
                split_n_list,
                num_warps_list,
                num_stages_list,
                tile_m_list,
                tile_n_list,
                group_size_list,
            ):
                if (
                    (tile_m or 0) > bm
                    or (tile_n or 0) > n // (split_n or 1)
                    or n % (split_n or 1) != 0
                    or (split_n or 0) > n
                ):
                    # Skip invalid parameter combinations
                    continue
                test_func = globals()["test_" + op]
                meta = dict(
                    bsr_scatter_mm6=dict(
                        SPLIT_N=split_n,
                        TILE_M=tile_m,
                        TILE_N=tile_n,
                        GROUP_SIZE=group_size,
                        num_stages=num_stages,
                        num_warps=num_warps,
                    ),
                    bsr_dense_mm_with_meta=dict(
                        GROUP_SIZE_ROW=group_size,
                        num_stages=num_stages,
                        num_warps=num_warps,
                    ),
                ).get(op, {})

                meta_str = ";".join(
                    f"{k}={v}" for k, v in meta.items() if v is not None
                )
                time_ms_lst = []
                performance_tflops_lst = []
                for r in range(args.repeat):
                    try:
                        time_ms, performance_tflops = test_func(x, y, **meta)
                    except triton.compiler.OutOfResources:
                        print(
                            f"op={op}[{meta_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})"
                            f" blocksize={bm}x{bk} OutOfResources",
                            file=outfile,
                        )
                        continue
                    except AssertionError:
                        raise
                    except Exception as msg:
                        msg = str(msg).split("\n", 1)[0]
                        print(
                            f"op={op}[{meta_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})"
                            f" blocksize={bm}x{bk} {msg}",
                            file=outfile,
                        )
                        continue
                    time_ms_lst.append(time_ms)
                    performance_tflops_lst.append(performance_tflops)
                    mark = ""
                    if op == "dense_dense_mm":
                        if target_performance is None:
                            target_performance = performance_tflops
                    elif target_performance is not None:
                        if (
                            abs(1 - performance_tflops / target_performance)
                            < performance_rtol
                        ):
                            mark += " @@@"
                    if best_tflops < performance_tflops:
                        best_tflops = performance_tflops
                        best_message = (
                            f"op={op}[{meta_str}]({bsr_size},x{n}) dtype={args.dtype} {sparsity=:.4f}(nnz={x._nnz()})"
                            f" blocksize={bm}x{bk} time={time_ms:.3f} ms performance={performance_tflops:.3f} TFLOPS"
                        )
                        if best_message not in best_messages:
                            best_messages.append(best_message)
                        mark += " !!!"
                    print(
                        f"op={op}[{meta_str}]({bsr_size},x{n}) dtype={args.dtype} {sparsity=:.4f}(nnz={x._nnz()})"
                        f" blocksize={bm}x{bk}"
                        f" time={time_ms:.3f} ms performance={performance_tflops:.3f} TFLOPS{mark}",
                        file=outfile,
                    )
                    outfile.flush()
                if args.repeat > 1:
                    avg_time_ms = sum(time_ms_lst) / len(time_ms_lst)
                    avg_performance_tflops = sum(performance_tflops_lst) / len(
                        performance_tflops_lst
                    )
                    print(
                        f"op={op}[{meta_str}]({bsr_size},{k}x{n}) dtype={args.dtype} {sparsity=}(nnz={x._nnz()})"
                        f" blocksize={bm}x{bk}"
                        f" time={time_ms:.3f} ms performance={performance_tflops:.3f} TFLOPS [AVERAGE]",
                        file=outfile,
                    )
                    outfile.flush()
                if op not in {"bsr_scatter_mm6", "bsr_dense_mm_with_meta"}:
                    # Break on operations that do not consume parameters
                    break
    if need_close:
        outfile.close()
