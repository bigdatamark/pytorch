# [MODULE DOC] Command-line benchmark of sparse-by-dense matrix multiplication time for coo
# and csr layouts on cpu, using random sparse matrices and the cpu event shim from the utils
# module. The main block parses options, runs the chosen format, and prints one result line to
# standard output, standard error or an appended file. Timings are in seconds, averaged over
# the repeat count.
# KNOWN ISSUE: the main block calls the combined coo-and-csr benchmark with three arguments
# while it requires five, so the both format fails with a type error.
import argparse
import sys

from utils import Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr

import torch


def test_sparse_csr(m, n, k, nnz, test_count):
    # [FUNCTION DOC] Times matrix multiplication of a random csr matrix (rows m, columns k, with
    # the given nonzero count) by a random dense double matrix (k by n). Runs it test_count
    # times, timing each run with the event shim, and returns the mean time in seconds.
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    csr = gen_sparse_csr((m, k), nnz)
    mat = torch.randn(k, n, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        csr.matmul(mat)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo(m, n, k, nnz, test_count):
    # [FUNCTION DOC] Same as the csr variant but for a random coo matrix: times test_count
    # multiplications by a random dense matrix and returns the mean time in seconds.
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    coo = gen_sparse_coo((m, k), nnz)
    mat = torch.randn(k, n, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        coo.matmul(mat)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo_and_csr(m, n, k, nnz, test_count):
    # [FUNCTION DOC] Generates one sparse matrix in both coo and csr form, then times test_count
    # multiplications by the same dense matrix for each layout. Returns a pair of the mean coo
    # time and the mean csr time, in seconds.
    start = Event(enable_timing=True)
    stop = Event(enable_timing=True)

    coo, csr = gen_sparse_coo_and_csr((m, k), nnz)
    mat = torch.randn((k, n), dtype=torch.double)

    times = []
    for _ in range(test_count):
        start.record()
        coo.matmul(mat)
        stop.record()
        times.append(start.elapsed_time(stop))

    coo_mean_time = sum(times) / len(times)

    times = []
    for _ in range(test_count):
        start.record()
        csr.matmul(mat)
        stop.record()
        times.append(start.elapsed_time(stop))

    csr_mean_time = sum(times) / len(times)

    return coo_mean_time, csr_mean_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMM")

    parser.add_argument("--format", default="csr", type=str)
    parser.add_argument("--m", default="1000", type=int)
    parser.add_argument("--n", default="1000", type=int)
    parser.add_argument("--k", default="1000", type=int)
    parser.add_argument("--nnz-ratio", "--nnz_ratio", default="0.1", type=float)
    parser.add_argument("--outfile", default="stdout", type=str)
    parser.add_argument("--test-count", "--test_count", default="10", type=int)

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

    test_count = args.test_count
    m = args.m
    n = args.n
    k = args.k
    nnz_ratio = args.nnz_ratio

    nnz = int(nnz_ratio * m * k)
    if args.format == "csr":
        time = test_sparse_csr(m, n, k, nnz, test_count)
    elif args.format == "coo":
        time = test_sparse_coo(m, n, k, nnz, test_count)
    elif args.format == "both":
        time_coo, time_csr = test_sparse_coo_and_csr(m, nnz, test_count)

    if args.format == "both":
        print(
            "format=coo",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_coo,
            file=outfile,
        )
        print(
            "format=csr",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_csr,
            file=outfile,
        )
    else:
        print(
            "format=",
            args.format,
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time,
            file=outfile,
        )
    if need_close:
        outfile.close()
