# [MODULE DOC] CI script that compares PR-time benchmark results against an expected-results CSV
# and fails on regressions or unexpected improvements. Defines two small dataclasses for rows of
# the expected and result files, a number-rounding helper, and main, which reads three file
# paths from the command line. Reports outcomes to the open source signpost logger and prints
# instructions for refreshing the expected file.
import copy
import csv
import json
import sys
from dataclasses import dataclass

import torch._logging.scribe as scribe


@dataclass
class ExpectedFileEntry:
    # [CLASS DOC] One row of the expected-results file: benchmark name, metric name, expected
    # integer value and the allowed noise margin given as a fraction (not a percentage).
    benchmark_name: str
    metric_name: str
    expected_value: int
    noise_margin: float


@dataclass
class ResultFileEntry:
    # [CLASS DOC] One row of a results file produced by a benchmark run: benchmark name, metric
    # name and the measured integer value.
    benchmark_name: str
    metric_name: str
    actual_value: int


def replace_with_zeros(num):
    # [FUNCTION DOC] Rounds an integer down to four significant digits by zeroing the lower
    # digits using floor division (so negative numbers round toward negative infinity); numbers
    # with four or fewer digits are returned unchanged. Raises ValueError if the input is not an
    # int.
    # KNOWN ISSUE: the docstring says the first three digits are kept, but the code keeps four.
    """
    Keeps the first three digits of an integer and replaces the rest with zeros.

    Args:
        num (int): The number to modify.

    Returns:
        int: The modified number.

    Raises:
        ValueError: If the input is not an integer.
    """
    # Check if input is an integer
    if not isinstance(num, int):
        raise ValueError("Input must be an integer")

    # Calculate the number of digits to remove
    digits_to_remove = len(str(abs(num))) - 4

    # Replace digits with zeros
    if digits_to_remove > 0:
        modified_num = (num // 10**digits_to_remove) * 10**digits_to_remove
    else:
        modified_num = num

    return modified_num


def main():
    # [FUNCTION DOC] Entry point. Command line arguments are the expected-results CSV path, the
    # results CSV path and an output path for a regenerated expected-results CSV. Both inputs
    # are loaded into mappings keyed by benchmark and metric names; blank rows are skipped only
    # in the expected file, and a duplicate key in either file raises an AssertionError. For
    # each expected entry, a missing result exits with status 1 immediately. Otherwise the
    # percent change is computed and the result is classified as a regression (above the
    # margin), a win (below it) or a pass, a message is printed and a signpost event is logged.
    # The new expected value is replaced with the rounded result only when the change exceeds
    # one fifth of the margin. Results without an expected entry are reported and logged as
    # missing regression tests. The regenerated expected rows (each followed by three empty rows
    # to ease merge conflicts) are written to the output path, and shell instructions to
    # overwrite the expected file are printed. If any benchmark failed, reproduction
    # instructions are printed and the process exits with status 1; otherwise it prints a
    # success message.
    # KNOWN ISSUE: the comment describing the result file format mentions an expected value and
    #     noise margin, but result rows only carry three columns; a blank row in the results
    #     file would raise an IndexError.
    # Expected file is the file that have the results that we are comparing against.
    # Expected has the following format:
    # benchmark_name, metric name, expected value, noise margin (as percentage)
    # Example:
    # add_loop_eager,compile_time_instruction_count,283178305, 0.01 (1% noise margin)
    expected_file_path = sys.argv[1]

    # Result file is the file that have the results of the current run. It has the following format:
    # benchmark_name, metric name, expected value, noise margin (as percentage)
    # Example:
    # add_loop_eager,compile_time_instruction_count,283178305
    result_file_path = sys.argv[2]

    # A path where a new expected results file will be written that can be used to replace expected_results.csv
    # in case of failure. In case of no failure the content of this file will match expected_file_path.
    reference_expected_results_path = sys.argv[3]

    # Read expected data file.
    expected_data: dict[str, ExpectedFileEntry] = {}

    with open(expected_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            entry = ExpectedFileEntry(
                benchmark_name=row[0].strip(),
                metric_name=row[1].strip(),
                expected_value=int(row[2]),
                noise_margin=float(row[3]),
            )
            key = (entry.benchmark_name, entry.metric_name)
            if key in expected_data:
                raise AssertionError(f"Duplicate entry for {key}")
            expected_data[key] = entry

    # Read result data file.
    result_data: dict[str, ResultFileEntry] = {}

    with open(result_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            entry = ResultFileEntry(
                benchmark_name=row[0].strip(),
                metric_name=row[1].strip(),
                actual_value=int(row[2]),
            )

            key = (entry.benchmark_name, entry.metric_name)
            if key in result_data:
                raise AssertionError(f"Duplicate entry for {key}")
            result_data[key] = entry

    fail = False
    new_expected = copy.deepcopy(expected_data)
    for key, entry in expected_data.items():
        if key not in result_data:
            print(f"Missing entry for {key} in result file")
            sys.exit(1)

        low = entry.expected_value - entry.expected_value * entry.noise_margin
        high = entry.expected_value + entry.expected_value * entry.noise_margin
        result = result_data[key].actual_value
        ratio = float(result - entry.expected_value) * 100 / entry.expected_value

        def log(event_name):
            # [FUNCTION DOC] HELPER: Sends a signpost event with the given name under the
            # pr_time_benchmarks subsystem, carrying the current benchmark's name, metric,
            # actual and expected values, noise margin and change ratio as a JSON parameter.
            # Uses the loop variables of the enclosing scope.
            scribe.open_source_signpost(
                subsystem="pr_time_benchmarks",
                name=event_name,
                parameters=json.dumps(
                    {
                        "benchmark_name": entry.benchmark_name,
                        "metric_name": entry.metric_name,
                        "actual_value": result,
                        "expected_value": entry.expected_value,
                        "noise_margin": entry.noise_margin,
                        "change_ratio": ratio,
                    }
                ),
            )

        new_entry = copy.deepcopy(entry)
        # only change if abs(ratio) > entry.noise_margin /5.
        new_entry.expected_value = (
            replace_with_zeros(result)
            if abs(ratio) > entry.noise_margin * 100 / 5
            else entry.expected_value
        )
        new_expected[key] = new_entry

        if result > high:
            fail = True
            print(
                f"REGRESSION: benchmark {key} failed, actual result {result} "
                f"is {ratio:.2f}% higher than expected {entry.expected_value} ±{entry.noise_margin * 100:+.2f}% "
                f"if this is an expected regression, please update the expected results.\n"
            )
            print(
                "please update all results that changed significantly, and not only the failed ones"
            )

            log("fail_regression")

        elif result < low:
            fail = True

            print(
                f"WIN: benchmark {key} failed, actual result {result} is {ratio:+.2f}% lower than "
                f"expected {entry.expected_value} ±{entry.noise_margin * 100:.2f}% "
                f"please OPEN THE TEST RESULTS update ALL BENCHMARKS RESULT with the new printed expected results. ALL ALL ALL\n"
            )
            print(
                "please update all results that changed significantly, and not only the failed ones"
            )

            log("fail_win")

        else:
            print(
                f"PASS: benchmark {key} pass, actual result {result} {ratio:+.2f}% is within "
                f"expected {entry.expected_value} ±{entry.noise_margin * 100:.2f}%\n"
            )

            log("pass")

    # Log all benchmarks that do not have a regression test enabled for them.
    for key, entry in result_data.items():
        if key not in expected_data:
            print(
                f"MISSING REGRESSION TEST: benchmark {key} does not have a regression test enabled for it.\n"
            )
            scribe.open_source_signpost(
                subsystem="pr_time_benchmarks",
                name="missing_regression_test",
                parameters=json.dumps(
                    {
                        "benchmark_name": entry.benchmark_name,
                        "metric_name": entry.metric_name,
                    }
                ),
            )

    with open(reference_expected_results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for entry in new_expected.values():
            # Write the data to the CSV file
            # print(f"{entry.benchmark_name},{entry.metric_name,},{round(entry.expected_value)},{entry.noise_margin}")
            writer.writerow(
                [
                    entry.benchmark_name,
                    entry.metric_name,
                    entry.expected_value,
                    entry.noise_margin,
                ]
            )
            # Three empty rows for merge conflicts.
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])

    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print("To update expected results, run the following command:")
    print()
    print("cat > benchmarks/dynamo/pr_time_benchmarks/expected_results.csv << EOF")
    with open(reference_expected_results_path) as f:
        print(f.read().rstrip())
    print("EOF")
    print()
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)

    if fail:
        print(
            f"There was some failures you can use the new reference expected result stored at path:"
            f"{reference_expected_results_path} and printed above\n"
        )
        print(
            "To reproduce locally follow the following instructions, note that absolute instructions count are going "
            "to be different than on the CI, hence you might want to run locally with and without your change:\n"
            "cd benchmarks/dynamo/pr_time_benchmarks/ \n"
            "python benchmarks/BENCHMARK.py result.csv \n"
            "note that BENCHMARK.py is the name of the file containing the failing benchmark."
        )
        sys.exit(1)
    else:
        print("All benchmarks passed")


if __name__ == "__main__":
    main()
