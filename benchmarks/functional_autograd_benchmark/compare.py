# [MODULE DOC] Command line tool comparing two result files of the functional autograd benchmark.
# Reads the before and after markdown tables, computes the speedup per model and task, and prints
# (and optionally writes) a combined markdown table. Imports helpers from the sibling utils
# module.
import argparse
from collections import defaultdict

from utils import from_markdown_table, to_markdown_table


def main():
    # [FUNCTION DOC] Entry point. Parses flags for the before file, after file and output file
    # path, loads both markdown tables, and builds a combined mapping where each model and task
    # has (speedup as mean before over mean after, before mean, before variance, after mean,
    # after variance); entries missing on one side get no values for the missing side and no
    # speedup. Prints the table and writes it to the output file when a path is given. Raises
    # FileNotFoundError for a missing input file.
    parser = argparse.ArgumentParser(
        "Main script to compare results from the benchmarks"
    )
    parser.add_argument(
        "--before",
        type=str,
        default="before.txt",
        help="Text file containing the times to use as base",
    )
    parser.add_argument(
        "--after",
        type=str,
        default="after.txt",
        help="Text file containing the times to use as new version",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Text file where to write the output"
    )
    args = parser.parse_args()

    with open(args.before) as f:
        content = f.read()
    res_before = from_markdown_table(content)

    with open(args.after) as f:
        content = f.read()
    res_after = from_markdown_table(content)

    diff = defaultdict(defaultdict)
    for model in res_before:
        for task in res_before[model]:
            mean_before, var_before = res_before[model][task]
            if task not in res_after[model]:
                diff[model][task] = (None, mean_before, var_before, None, None)
            else:
                mean_after, var_after = res_after[model][task]
                diff[model][task] = (
                    mean_before / mean_after,
                    mean_before,
                    var_before,
                    mean_after,
                    var_after,
                )
    for model in res_after:
        for task in res_after[model]:
            if task not in res_before[model]:
                mean_after, var_after = res_after[model][task]
                diff[model][task] = (None, None, None, mean_after, var_after)

    header = (
        "model",
        "task",
        "speedup",
        "mean (before)",
        "var (before)",
        "mean (after)",
        "var (after)",
    )
    out = to_markdown_table(diff, header=header)

    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
