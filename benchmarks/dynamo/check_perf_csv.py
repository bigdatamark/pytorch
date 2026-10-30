# [MODULE DOC] CLI that checks a dynamo performance CSV: every model's speedup must reach a
# threshold (optionally scaled to relax the check). Prints a one-line summary per model and
# exits with status 1 if any model is below the threshold.
import argparse
import sys
import textwrap

import pandas as pd


def check_perf_csv(filename, threshold, threshold_scale):
    # [FUNCTION DOC] Reads the CSV (prints an error and exits with status 1 if the file does
    # not exist), computes the effective threshold as threshold times the scale, and prints
    # speedup, latency, compile time and memory ratio (inverse of compression ratio, with
    # peak memory) per model, omitting values that are NaN. Models with speedup under the
    # threshold are collected; if any, prints them sorted from worst with percent distance
    # from target and exits with status 1, otherwise prints a pass message.
    """
    Basic performance checking.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)

    effective_threshold = threshold * threshold_scale
    print(f"Checking {filename} (speedup threshold >= {effective_threshold:.2f}x)\n")

    failed = []
    for _, row in df.iterrows():
        model_name = row["name"]
        speedup = float(row["speedup"])
        abs_latency = float(row["abs_latency"])
        compilation_latency = float(row["compilation_latency"])
        compression_ratio = float(row["compression_ratio"])
        eager_peak_mem = float(row["eager_peak_mem"])
        dynamo_peak_mem = float(row["dynamo_peak_mem"])

        perf_summary = f"{model_name:34} speedup={speedup:.3f}x"
        if pd.notna(abs_latency):
            perf_summary += f", latency={abs_latency:.1f} ms/iter"
        if pd.notna(compilation_latency):
            perf_summary += f", compile={compilation_latency:.3f}s"
        if pd.notna(compression_ratio):
            perf_summary += f", mem_ratio={1 / compression_ratio:.2f}x"
            if pd.notna(eager_peak_mem) and pd.notna(dynamo_peak_mem):
                perf_summary += (
                    f" (eager={eager_peak_mem:.1f} GB, dynamo={dynamo_peak_mem:.1f} GB)"
                )

        if speedup < effective_threshold:
            failed.append((model_name, speedup))

        print(perf_summary)

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} model(s) performance regressed
                    {" ".join([name for name, _ in failed])}
                """
            )
        )
        for name, sp in sorted(failed, key=lambda x: x[1]):
            pct_from_target = (sp / effective_threshold - 1.0) * 100.0
            print(
                f"  - {name}: {sp:.3f}x (< {effective_threshold:.2f}x; {pct_from_target:.1f}% from target)"
            )
        sys.exit(1)
    else:
        print(
            f"\nAll {len(df)} model(s) passed threshold check (>= {effective_threshold:.2f}x)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    parser.add_argument(
        "--threshold", "-t", type=float, help="threshold speedup value to check against"
    )
    parser.add_argument(
        "--threshold-scale",
        "-s",
        type=float,
        default=1.0,
        help="multiply threshold by this value to relax the check",
    )
    args = parser.parse_args()
    check_perf_csv(args.file, args.threshold, args.threshold_scale)
