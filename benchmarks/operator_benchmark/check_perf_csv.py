# [MODULE DOC] Script that compares a CSV of benchmark results against a baseline CSV and
# fails (exit status one) when any case regressed, improved beyond the threshold, or has
# no baseline. Contains a list of known-bad cases to skip, a per-case lookup helper, the
# comparison routine and the command line main. Requires pandas.
import argparse
import sys
import textwrap

import pandas as pd


SKIP_TEST_LISTS = [
    # https://github.com/pytorch/pytorch/issues/143852
    "channel_shuffle_batch_size4_channels_per_group64_height64_width64_groups4_channel_lastTrue",
    "batchnorm_N3136_C256_cpu_trainingTrue_cudnnFalse",
    "index_add__M256_N512_K1_dim1_cpu_dtypetorch.float32",
    "interpolate_input_size(1,3,600,400)_output_size(240,240)_channels_lastTrue_modelinear",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits4_cpu",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits8_cpu",
]


def get_field(csv, case: str, field: str):
    # [FUNCTION DOC] HELPER: Returns the value of a column for the row
    # whose case name matches. Returns None if the case is absent,
    # duplicated or the column is missing (any error is swallowed).
    try:
        return csv.loc[csv["Case Name"] == case][field].item()
    except Exception:
        return None


def check_perf(actual_csv, expected_csv, expected_filename, threshold):
    # [FUNCTION DOC] Compares actual against expected execution times. Cases in the skip list are
    # dropped. For each remaining case the speed up is expected time divided by actual time. A
    # case with no baseline is reported as Baseline Not Found; speed up within the threshold band
    # (inclusive lower, exclusive upper) passes; at or above the upper edge is improved; below the
    # lower edge fails. Prints one line per case. Builds a message describing regressed, improved
    # and baseline-missing cases and a hint to update the expected file. Returns a pair: the first
    # non-empty list among failed, improved and missing (or an empty list) and the message.
    failed = []
    improved = []
    baseline_not_found = []

    actual_csv = actual_csv[~actual_csv["Case Name"].isin(set(SKIP_TEST_LISTS))]

    for case in actual_csv["Case Name"]:
        perf = get_field(actual_csv, case, "Execution Time")
        expected_perf = get_field(expected_csv, case, "Execution Time")

        if expected_perf is None:
            status = "Baseline Not Found"
            print(f"{case:34}  {status}")
            baseline_not_found.append(case)
            continue

        speed_up = expected_perf / perf

        if (1 - threshold) <= speed_up < (1 + threshold):
            status = "PASS"
            print(f"{case:34}  {status}")
            continue
        elif speed_up >= 1 + threshold:
            status = "IMPROVED:"
            improved.append(case)
        else:
            status = "FAILED:"
            failed.append(case)
        print(f"{case:34}  {status:9} perf={perf}, expected={expected_perf}")

    msg = ""
    if failed or improved or baseline_not_found:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have performance status regressed:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have performance status improved:
                {" ".join(improved)}

            """
            )

        if baseline_not_found:
            msg += textwrap.dedent(
                f"""
            Baseline Not Found: {len(baseline_not_found)} models don't have the baseline data:
                {" ".join(baseline_not_found)}

            """
            )

        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        """
        )
    return failed or improved or baseline_not_found, msg


def main():
    # [FUNCTION DOC] Command line entry. Requires actual and expected CSV paths and takes a
    # threshold (default 0.5). Reads both with pandas, drops duplicate case rows in the
    # actual data keeping the first, runs the comparison, and if anything failed, improved
    # or lacked a baseline prints the message and exits with status one.
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold to define regression/improvement",
    )
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    actual.drop_duplicates(subset=["Case Name"], keep="first", inplace=True)
    expected = pd.read_csv(args.expected)

    failed, msg = check_perf(actual, expected, args.expected, args.threshold)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
