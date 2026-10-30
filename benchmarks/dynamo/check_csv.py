# [MODULE DOC] Small CLI that reads a benchmark accuracy CSV and fails (exit status 1) if
# any model's accuracy field does not contain the text pass.
import argparse
import sys
import textwrap

import pandas as pd


def check_csv(filename):
    # [FUNCTION DOC] Reads the CSV, prints each model name with its accuracy value, and
    # collects models whose accuracy text lacks the substring pass. If any, prints an error
    # listing them and exits the process with status 1. A non-text (missing) accuracy value
    # would raise a TypeError.
    """
    Basic accuracy checking.
    """

    df = pd.read_csv(filename)

    failed = []
    for _, row in df.iterrows():
        model_name = row["name"]
        status = row["accuracy"]
        if "pass" not in status:
            failed.append(model_name)

        print(f"{model_name:34} {status}")

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models failed
                    {" ".join(failed)}
                """
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    args = parser.parse_args()
    check_csv(args.file)
