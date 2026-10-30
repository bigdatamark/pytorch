# [MODULE DOC] CI script comparing per-model memory compression ratios between an actual and
# an expected CSV. Import-time side effect: builds the argument parser and parses the process
# command line (required actual and expected paths) into a module-level namespace; the check
# itself only runs when executed as a script.
import argparse
import sys
import textwrap

import pandas as pd


def main(args):
    # [FUNCTION DOC] Reads the actual and expected CSVs from the namespace paths and, for each
    # model in the actual file, compares compression ratios: PASS if the actual value is at
    # least 95 percent of the expected one, otherwise FAIL. Models missing from the expected
    # file are reported and skipped. Prints per-model detail; if any failed, prints an error
    # naming them and exits the process with status 1.
    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)
    failed = []

    for name in actual["name"]:
        actual_memory_compression = float(
            actual.loc[actual["name"] == name]["compression_ratio"]
        )
        try:
            expected_memory_compression = float(
                expected.loc[expected["name"] == name]["compression_ratio"]
            )
        except TypeError:
            print(f"{name:34} is missing from {args.expected}")
            continue
        if actual_memory_compression >= expected_memory_compression * 0.95:
            status = "PASS"
        else:
            status = "FAIL"
            failed.append(name)
        print(
            f"""
            {name:34}:
                actual_memory_compression={actual_memory_compression:.2f},
                expected_memory_compression={expected_memory_compression:.2f},
                {status}
            """
        )

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error: {len(failed)} models below expected memory compression ratio:
                    {" ".join(failed)}
                If this drop is expected, you can update `{args.expected}`.
                """
            )
        )
        sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--actual", type=str, required=True)
parser.add_argument("--expected", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
