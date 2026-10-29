# Owner(s): ["module: mps"]
# Collection of op level benchmarks for MPS
# Useful as reference tool when migrating ops from MPS to Metal

# [MODULE DOC] Command-line benchmark script for MPS operators. Provides timing helpers built on
# torch.utils.benchmark, benchmark builders for unary, binary, reduction, scan and grid_sample ops
# (comparing eager against torch.compile where relevant), and a main entry point that selects
# benchmarks by name from the command line and prints comparison tables. No import-time side
# effects; running as a script raises the dynamo cache size limit before calling main.
import itertools
import sys
import timeit
import warnings

import torch
from torch.utils.benchmark import Compare, Measurement, Timer


def bench_unary_op(func, x, label) -> Measurement:
    # [FUNCTION DOC] Times calling func on tensor x with an adaptive-duration timer, appending an
    # MPS synchronize to the timed statement when x lives on an mps device. The label becomes the
    # measurement description; the sub-label holds the function name and dtype. Returns the
    # Measurement. func must have a name attribute.
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x);{sync_cmd}",
        globals={"f": func, "x": x},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_binary_op(func, x, y, label) -> Measurement:
    # [FUNCTION DOC] Same as bench_unary_op for a two-argument call func(x, y); synchronization
    # is decided from x's device only, and the sub-label lists both dtypes. Returns Measurement.
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x, y);{sync_cmd}",
        globals={"f": func, "x": x, "y": y},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)}, {str(y.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_unary(
    unary_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    # [FUNCTION DOC] Benchmarks a unary op on a 1024x1024 tensor in four layouts: dense,
    # transposed, strided (every other column of a wider tensor) and strided plus transposed.
    # Returns the list of four Measurements.
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dtype)
    x_s = torch.testing.make_tensor(1024, 2048, device=device, dtype=dtype)[::, ::2]
    rc = []
    rc.append(bench_unary_op(unary_func, x, "dense"))
    rc.append(bench_unary_op(unary_func, x.t(), "transposed"))
    rc.append(bench_unary_op(unary_func, x_s, "strided"))
    rc.append(bench_unary_op(unary_func, x_s.t(), "strided + transposed"))
    return rc


def bench_binary(
    binary_func,
    device: str = "mps",
    dt_a: torch.dtype = torch.float32,
    dt_b: torch.dtype | None = None,
) -> list[Measurement]:
    # [FUNCTION DOC] Benchmarks a binary op on 1024x1024 operands (second dtype defaults to the
    # first) for six combinations: dense-dense, both transposed, mixed dense/transposed in both
    # orders, tensor with a 0-d scalar tensor, and tensor with a broadcast row. Returns the list.
    dt_b = dt_b if dt_b is not None else dt_a
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_a)
    y = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_b)
    s = torch.testing.make_tensor((), device=device, dtype=dt_b)
    rc = []
    rc.append(bench_binary_op(binary_func, x, y, "dense-dense"))
    rc.append(bench_binary_op(binary_func, x.t(), y.t(), "transp-transp"))
    rc.append(bench_binary_op(binary_func, x, y.t(), "dense-transp"))
    rc.append(bench_binary_op(binary_func, x.t(), y, "transp-dense"))
    rc.append(bench_binary_op(binary_func, x, s, "dense-scalar"))
    rc.append(bench_binary_op(binary_func, x, y[0], "dense-bcast"))
    return rc


def check_eager_vs_compile(rc_c, rc_e, func, dtype):
    # [FUNCTION DOC] HELPER: Emits a warning (no exception) if the two result tensors are not
    # close, reporting the function name, dtype and maximum absolute difference.
    # KNOWN ISSUE: the warning text always says "reduction" even when called from the scan
    # benchmarks; callers also pass the eager and compiled results in inconsistent order.
    if not torch.allclose(rc_c, rc_e):
        mdiff = (rc_c - rc_e).abs().max()
        warnings.warn(
            f"Eager and compile reduction do not match for {func.__name__} and {dtype} max_diff={mdiff}",
            stacklevel=2,
        )


def bench_reduction(
    reduction_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    # [FUNCTION DOC] Benchmarks a reduction over dim 0 of square tensors of side 512 to 4096, both
    # eager and compiled (non-dynamic, full graph). For each size it also checks the eager and
    # compiled results agree (warning only; tuple results use the first element) and appends two
    # Measurements. Returns the list.
    rc = []

    # Bench 2D with reduction over dim=0
    def f(t):
        # [FUNCTION DOC] HELPER: Applies the reduction over dim 0.
        return reduction_func(t, dim=0)

    f.__name__ = reduction_func.__name__
    f_c = torch.compile(f, dynamic=False, fullgraph=True)

    for size in (512, 1024, 2048, 4096):
        x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
        rc_c, rc_e = f(x), f_c(x)
        rc_c, rc_e = (rc_c[0], rc_e[0]) if isinstance(rc_c, tuple) else (rc_c, rc_e)
        check_eager_vs_compile(rc_c, rc_e, reduction_func, dtype)
        rc.append(bench_unary_op(f, x, f"eager-{size}x{size}"))
        rc.append(bench_unary_op(f_c, x, f"compile-{size}x{size}"))
    return rc


def bench_scan(
    scan_func,
    device: str = "mps",
    dtype: torch.dtype = torch.float32,
    with_indices: bool = False,
) -> list[Measurement]:
    # [FUNCTION DOC] Benchmarks a scan (cumulative) op eager versus compiled: on square 2D tensors
    # for dim 0 and 1 (sizes 32 to 1024) and on 1D tensors of sizes 100 to one million. Sets a
    # descriptive name on each function for labeling, warns on eager/compile mismatch (checking
    # both values and indices when with_indices is true), and returns all Measurements.
    rc = []

    # Bench cumsum along different dimensions
    for dim in [0, 1]:

        def f(t):
            # [FUNCTION DOC] HELPER: Applies the scan along the current loop dim.
            return scan_func(t, dim=dim)

        f_c = torch.compile(f, dynamic=False, fullgraph=True)

        for size in (32, 128, 512, 1024):
            f.__name__ = f"{scan_func.__name__}-dim{dim}-{size}x{size}"
            f_c.__name__ = f.__name__
            x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
            rc_c, rc_e = f(x), f_c(x)
            if with_indices:
                check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
                check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
            else:
                check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
            rc.append(bench_unary_op(f, x, "eager"))
            rc.append(bench_unary_op(f_c, x, "compile"))

    # Bench 1D cumsum for different sizes
    def f_1d(t):
        # [FUNCTION DOC] HELPER: Applies the scan along dim 0 of a 1D tensor.
        return scan_func(t, dim=0)

    f_1d_c = torch.compile(f_1d, dynamic=False, fullgraph=True)

    for size in (100, 10000, 1000000):
        f_1d.__name__ = f"{scan_func.__name__}-1d-{size}"
        f_1d_c.__name__ = f_1d.__name__
        x = torch.testing.make_tensor(size, device=device, dtype=dtype)
        rc_c, rc_e = f_1d(x), f_1d_c(x)
        if with_indices:
            check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
            check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
        else:
            check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
        rc.append(bench_unary_op(f_1d, x, "eager"))
        rc.append(bench_unary_op(f_1d_c, x, "compile"))

    return rc


def bench_grid_sampler_2d(
    device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    # [FUNCTION DOC] Benchmarks 2D grid_sample (zeros padding, no corner alignment) for bilinear,
    # nearest and bicubic modes and sizes 64 to 512 with batch 4 and 16 channels, eager versus
    # compiled. Warns if results differ beyond a loose tolerance. Returns all Measurements.
    rc = []
    sync_cmd = "torch.mps.synchronize()" if device == "mps" else ""

    for mode in ("bilinear", "nearest", "bicubic"):
        for size in (64, 128, 256, 512):
            N, C = 4, 16
            inp = torch.randn(N, C, size, size, device=device, dtype=dtype)
            grid = torch.rand(N, size, size, 2, device=device, dtype=dtype) * 2 - 1

            def f(inp, grid, mode=mode):
                # [FUNCTION DOC] HELPER: Runs grid_sample with the mode bound at definition time.
                return torch.nn.functional.grid_sample(
                    inp, grid, mode=mode, padding_mode="zeros", align_corners=False
                )

            f_c = torch.compile(f, dynamic=False, fullgraph=True)

            rc_e, rc_c = f(inp, grid), f_c(inp, grid)
            if not torch.allclose(rc_e, rc_c, atol=1e-2, rtol=1e-2):
                mdiff = (rc_e - rc_c).abs().max()
                warnings.warn(
                    f"Eager and compile grid_sample do not match for {mode} {dtype} max_diff={mdiff}",
                    stacklevel=2,
                )

            sub_label = f"grid_sample-{mode}-{size}x{size} ({str(dtype)})"
            for label, fn in [("eager", f), ("compile", f_c)]:
                t = Timer(
                    stmt=f"f(inp, grid);{sync_cmd}",
                    globals={"f": fn, "inp": inp, "grid": grid},
                    language="python",
                    timer=timeit.default_timer,
                    sub_label=sub_label,
                    description=label,
                    env=torch.__version__,
                )
                rc.append(t.blocked_autorange())

    return rc


def main() -> None:
    # [FUNCTION DOC] Entry point: defines named benchmark groups (index, unary, reduction, scan,
    # scan_with_indices, binary, grid_sampler_2d), picks those named on the command line (all if
    # none), prints an error and exits with status 1 on an unknown name, otherwise runs each and
    # prints its comparison table to stdout.
    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    def bench_index_ops():
        # [FUNCTION DOC] HELPER: Benchmarks indexing a 3D tensor with a random index tensor for
        # several dtypes and sizes on mps, then prints the comparison table.
        B = 11
        rc = []
        for dtype, N in itertools.product(
            [torch.int8, torch.float16, torch.float32], [50, 100, 500, 1000, 2000]
        ):
            x = torch.testing.make_tensor((B, N, N), device="mps", dtype=dtype)
            y = torch.randint(0, B, (3,))
            rc.append(bench_binary_op(torch.Tensor.__getitem__, x, y, f"{B}x{N}x{N}"))
        Compare(rc).print()

    def bench_unary_ops():
        # [FUNCTION DOC] HELPER: Benchmarks sqrt and sin across the dtypes; prints the table.
        rc = []
        for op, dtype in itertools.product([torch.sqrt, torch.sin], dtypes):
            rc.extend(bench_unary(op, dtype=dtype))
        Compare(rc).print()

    def bench_reduction_ops():
        # [FUNCTION DOC] HELPER: Benchmarks sum and max reductions; prints the table.
        rc = []
        for op in [torch.sum, torch.max]:
            rc.extend(bench_reduction(op))
        Compare(rc).print()

    def bench_scan_ops():
        # [FUNCTION DOC] HELPER: Benchmarks cumsum across the dtypes; prints the table.
        rc = []
        for dtype in dtypes:
            rc.extend(bench_scan(torch.cumsum, dtype=dtype))
        Compare(rc).print()

    def bench_scan_with_indices_ops():
        # [FUNCTION DOC] HELPER: Benchmarks cummin (values and indices) across dtypes; prints it.
        rc = []
        for dtype in dtypes:
            rc.extend(bench_scan(torch.cummin, dtype=dtype, with_indices=True))
        Compare(rc).print()

    def bench_binary_ops():
        # [FUNCTION DOC] HELPER: Benchmarks fmax and add across dtypes, plus a float32 with
        # float16 mixed-dtype case; prints the table.
        rc = []
        ops = [torch.fmax, torch.add]
        for op, dtype in itertools.product(ops, dtypes):
            rc.extend(bench_binary(op, dt_a=dtype))
            if dtype == torch.float32:
                rc.extend(bench_binary(op, dt_b=torch.float16))
        Compare(rc).print()

    def bench_grid_sampler_2d_ops():
        # [FUNCTION DOC] HELPER: Benchmarks 2D grid_sample across the dtypes; prints the table.
        rc = []
        for dtype in dtypes:
            rc.extend(bench_grid_sampler_2d(dtype=dtype))
        Compare(rc).print()

    benchmarks = {
        "index": bench_index_ops,
        "unary": bench_unary_ops,
        "reduction": bench_reduction_ops,
        "scan": bench_scan_ops,
        "scan_with_indices": bench_scan_with_indices_ops,
        "binary": bench_binary_ops,
        "grid_sampler_2d": bench_grid_sampler_2d_ops,
    }

    selected = sys.argv[1:]
    if not selected:
        selected = list(benchmarks.keys())
    for name in selected:
        if name not in benchmarks:
            print(
                f"Unknown benchmark: {name}. Available: {', '.join(benchmarks.keys())}"
            )
            sys.exit(1)

    for name in selected:
        print(f"Running {name}...")
        benchmarks[name]()


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 2**16
    main()
