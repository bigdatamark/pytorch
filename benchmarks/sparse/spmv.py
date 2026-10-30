# [MODULE DOC] Command-line benchmark of sparse-matrix by dense-vector multiplication time for
# coo and csr layouts, using random square sparse matrices and the cpu event shim from the utils
# module. The main block parses options, runs the chosen format, and prints one result line (two
# for both formats) to standard output, standard error or an appended file. Timings are in
# seconds, averaged over the repeat count.
# KNOWN ISSUE: the helper import is relative, so running this file directly as a script fails;
#     it works only when imported as part of the package.
import argparse
import sys

import torch

from .utils import Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr


def test_sparse_csr(m, nnz, test_count):
    # [FUNCTION DOC] Times matrix-vector multiplication of a random square csr matrix (the given
    # size and nonzero count) by a random double vector. Repeats test_count times, timing each
    # run with the event shim, and returns the mean time in seconds.
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    csr = gen_sparse_csr((m, m), nnz)
    vector = torch.randn(m, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        csr.matmul(vector)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo(m, nnz, test_count):
    # [FUNCTION DOC] Same as the csr variant but for a random square coo matrix: returns the
    # mean time in seconds over test_count multiplications by a random double vector.
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    coo = gen_sparse_coo((m, m), nnz)
    vector = torch.randn(m, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        coo.matmul(vector)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo_and_csr(m, nnz, test_count):
    # [FUNCTION DOC] Generates one random square sparse matrix in both coo and csr form and
    # times test_count matrix-vector products for each layout with the same vector. Returns a
    # pair of the mean coo time and the mean csr time, in seconds.
    start = Event(enable_timing=True)
    stop = Event(enable_timing=True)

    coo, csr = gen_sparse_coo_and_csr((m, m), nnz)
    vector = torch.randn(m, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start.record()
        coo.matmul(vector)
        stop.record()

        times.append(start.elapsed_time(stop))

    coo_mean_time = sum(times) / len(times)

    times = []
    for _ in range(test_count):
        start.record()
        csr.matmul(vector)
        stop.record()
        times.append(start.elapsed_time(stop))

    csr_mean_time = sum(times) / len(times)

    return coo_mean_time, csr_mean_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMV")

    parser.add_argument("--format", default="csr", type=str)
    parser.add_argument("--m", default="1000", type=int)
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
    nnz_ratio = args.nnz_ratio

    nnz = int(nnz_ratio * m * m)
    if args.format == "csr":
        time = test_sparse_csr(m, nnz, test_count)
    elif args.format == "coo":
        time = test_sparse_coo(m, nnz, test_count)
    elif args.format == "both":
        time_coo, time_csr = test_sparse_coo_and_csr(m, nnz, test_count)

    if args.format != "both":
        print(
            "format=",
            args.format,
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " time=",
            time,
            file=outfile,
        )
    else:
        print(
            "format=coo",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
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
            " time=",
            time_csr,
            file=outfile,
        )
    if need_close:
        outfile.close()
