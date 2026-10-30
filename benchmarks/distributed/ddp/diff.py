#!/usr/bin/env python3
#
# Computes difference between measurements produced by ./benchmark.py.
#
# [MODULE DOC] Command line tool comparing two JSON result files written by the DDP
# benchmark script. Prints differing metadata side by side, then for each matching
# model/batch-size pair the 75th and 95th percentile times and relative change.

import argparse
import json

import numpy as np


def load(path):
    # [FUNCTION DOC] HELPER: Reads the given path and returns the parsed JSON content.
    with open(path) as f:
        return json.load(f)


def main():
    # [FUNCTION DOC] Entry point. Parses exactly two file arguments (baseline, test),
    # prints the non-result metadata keys of both side by side, then for each pair of
    # benchmark entries with identical model and batch size prints, per GPU count (the
    # no-DDP first round and rank-count mismatches are skipped), p75/p95 seconds per
    # iteration, examples per second and percent improvement (negated relative change).
    # Reads files and writes to stdout; raises RuntimeError if not two files.
    parser = argparse.ArgumentParser(description="PyTorch distributed benchmark diff")
    parser.add_argument("file", nargs=2)
    args = parser.parse_args()

    if len(args.file) != 2:
        raise RuntimeError("Must specify 2 files to diff")

    ja = load(args.file[0])
    jb = load(args.file[1])

    keys = (set(ja.keys()) | set(jb.keys())) - {"benchmark_results"}
    print(f"{'':20s} {'baseline':>20s}      {'test':>20s}")
    print(f"{'':20s} {'-' * 20:>20s}      {'-' * 20:>20s}")
    for key in sorted(keys):
        va = str(ja.get(key, "-"))
        vb = str(jb.get(key, "-"))
        print(f"{key + ':':20s} {va:>20s}  vs  {vb:>20s}")
    print()

    ba = ja["benchmark_results"]
    bb = jb["benchmark_results"]
    for ra, rb in zip(ba, bb):
        if ra["model"] != rb["model"]:
            continue
        if ra["batch_size"] != rb["batch_size"]:
            continue

        model = ra["model"]
        batch_size = int(ra["batch_size"])
        name = f"{model} with batch size {batch_size}"
        print(f"Benchmark: {name}")

        # Print header
        print()
        print(f"{'':>10s}", end="")
        for _ in [75, 95]:
            print(f"{'sec/iter':>16s}{'ex/sec':>10s}{'diff':>10s}", end="")
        print()

        # Print measurements
        for i, (xa, xb) in enumerate(zip(ra["result"], rb["result"])):
            # Ignore round without ddp
            if i == 0:
                continue
            # Sanity check: ignore if number of ranks is not equal
            if len(xa["ranks"]) != len(xb["ranks"]):
                continue

            ngpus = len(xa["ranks"])
            ma = sorted(xa["measurements"])
            mb = sorted(xb["measurements"])
            print(f"{ngpus:>4d} GPUs:", end="")
            for p in [75, 95]:
                va = np.percentile(ma, p)
                vb = np.percentile(mb, p)
                # We're measuring time, so lower is better (hence the negation)
                delta = -100 * ((vb - va) / va)
                print(
                    f"  p{p:02d}: {vb:8.3f}s {int(batch_size / vb):7d}/s {delta:+8.1f}%",
                    end="",
                )
            print()
        print()


if __name__ == "__main__":
    main()
