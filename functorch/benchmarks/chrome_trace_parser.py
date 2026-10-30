#!/usr/bin/env python3
# [MODULE DOC] Command line tool that computes GPU utilization and matmul/convolution
# time utilization for PyTorch profiler chrome trace json files. Takes a runtimes csv
# file (columns name and runtime, in seconds) and either trace file names or a folder
# of json traces, and prints one line per model with the results. Import-time side
# effect: creates a module logger.
import argparse
import logging
import os

import pandas as pd

from torch._functorch.benchmark_utils import compute_utilization


log = logging.getLogger(__name__)

# process the chrome traces output by the pytorch profiler
# require the json input file's name to be in format {model_name}_chrome_trace_*.json
# the runtimes file should have format (model_name, runtime)


def get_model_name(filename):
    # [FUNCTION DOC] Returns the model name: the part of the file base name before the
    # chrome trace marker. If the marker is absent the search result is -1, so the last
    # character is dropped.
    """
    Get model name from a file in format {model_name}_chrome_trace_*.json
    """
    _, tail = os.path.split(filename)
    modelname = tail[: tail.find("_chrome_trace")]
    return modelname


def get_total_length(run_times_df, modelname):
    # [FUNCTION DOC] Looks up the runtime column of the data frame row whose name matches
    # the model name and converts it to float. Fails if there is not exactly one matching
    # row (the conversion of a series is deprecated or errors in recent pandas).
    return float(run_times_df[run_times_df["name"] == modelname]["runtime"])


def main():
    # [FUNCTION DOC] Parses arguments (required runtime csv, plus a mutually exclusive set
    # of repeated trace file names or a folder scanned for json files), prints a header,
    # reads the runtimes csv and for each trace prints model name, GPU utilization and
    # matmul/convolution utilization; errors per file are logged and printed as an error
    # line instead of aborting.
    # KNOWN ISSUE: compute_utilization receives the whole list of file names rather than
    # the current file name, although it expects a single file; so with several files it
    # will fail and be reported as errors. Also the final else branch (no inputs) is
    # unreachable since the group is required, and would leave the file list undefined.
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--runtime", "-runf", help="file name of the runtime file", required=True
    )
    group.add_argument(
        "--filename",
        "-f",
        action="append",
        help="a filename of the json file to process",
    )
    group.add_argument("--folder", "-fd", help="a folder of the json files to process")
    args = parser.parse_args()

    if args.filename:
        filenames = args.filename
    elif args.folder:
        filenames = []
        directory = args.folder
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith(".json"):
                filenames.append(f)
    else:
        print("Please provide a filename or a folder name")

    print("modelname, GPU Utilization, MM and Conv time")

    run_times_df = pd.read_csv(args.runtime)
    for filename in filenames:
        try:
            modelname = get_model_name(filename)
            total_length = get_total_length(run_times_df, modelname) * 1e6
            utilization, mm_conv_utilization = compute_utilization(
                filenames, total_length
            )
            print(f"{modelname}, {utilization}, {mm_conv_utilization}")
        except BaseException:
            log.exception("%s, ERROR", filename)
            print(f"{filename}, ERROR")


if __name__ == "__main__":
    main()
