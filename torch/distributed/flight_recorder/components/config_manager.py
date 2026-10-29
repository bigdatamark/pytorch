# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Command-line configuration for the flight recorder analyzer: defines JobConfig, which builds an argument parser and validates arguments. Creates the shared FlightRecorderLogger at import.
import argparse
import logging
from collections.abc import Sequence

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = ["JobConfig"]


logger: FlightRecorderLogger = FlightRecorderLogger()


class JobConfig:
    # [CLASS DOC] Holds the argparse parser for the trace analysis script. State: parser, with options for trace_dir, selected ranks, partial-rank analysis, process group filters, output path, filename prefix, print/verbosity flags, mismatch cap, and torchft transform with group world size. Key method: parse_args.
    """
    A helper class to manage the script configuration.
    """

    def __init__(self: "JobConfig"):
        # [FUNCTION DOC] Build the argument parser and register all analyzer options (trace_dir positional, --selected-ranks, --allow-incomplete-ranks, --pg-filters, -o/--output, -p/--prefix, -j/--just_print_entries, -v/--verbose, --print_stack_trace, --mismatch_cap default 10, --transform-ft, --group-world-size).
        self.parser = argparse.ArgumentParser(
            description="PyTorch Flight recorder analyzing script."
        )
        self.parser.add_argument(
            "trace_dir",
            nargs="?",
            help="Directory containing one trace file per rank, named with <prefix>_<rank>.",
        )
        self.parser.add_argument(
            "--selected-ranks",
            default=None,
            nargs="+",
            type=int,
            help="List of ranks we want to show traces for.",
        )
        self.parser.add_argument(
            "--allow-incomplete-ranks",
            action="store_true",
            help=(
                "FR trace require all ranks to have dumps for analysis. "
                "This flag allows best-effort partial analysis of results "
                "and printing of collected data."
            ),
        )
        self.parser.add_argument(
            "--pg-filters",
            default=None,
            nargs="+",
            type=str,
            help=(
                "List of filter strings, it could be pg name or pg desc. "
                "If specified, only show traces for the given pg."
            ),
        )
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument(
            "-p",
            "--prefix",
            help=(
                "Common filename prefix to strip such that rank can be extracted. "
                "If not specified, will attempt to infer a common prefix."
            ),
            default=None,
        )
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")
        self.parser.add_argument("--print_stack_trace", action="store_true")
        self.parser.add_argument(
            "--mismatch_cap",
            type=int,
            default=10,
            help="Maximum number of mismatches we print (from earliest).",
        )
        self.parser.add_argument(
            "--transform-ft",
            action="store_true",
            help="Transform PG config to use global ranks to analyze traces produced by torchft",
        )
        self.parser.add_argument(
            "--group-world-size",
            type=int,
            default=None,
            help="The number of ranks in 1 torchft replica group. Must be specified if --transform-ft is True",
        )

    def parse_args(self: "JobConfig", args: Sequence[str] | None) -> argparse.Namespace:
        # [FUNCTION DOC] Parse the given argument list (or the process command line when None) and return the namespace.
        # Raises AssertionError if selected ranks or pg filters are given without the just-print-entries flag. Side effect: with verbose set, switches the flight recorder logger to debug level. Argparse may exit on invalid arguments.
        # pyrefly: ignore [bad-assignment]
        args = self.parser.parse_args(args)
        # pyrefly: ignore [missing-attribute]
        if args.selected_ranks is not None:
            # pyrefly: ignore [missing-attribute]
            if not args.just_print_entries:
                raise AssertionError(
                    "Not support selecting ranks without printing entries"
                )
        # pyrefly: ignore [missing-attribute]
        if args.pg_filters is not None:
            # pyrefly: ignore [missing-attribute]
            if not args.just_print_entries:
                raise AssertionError(
                    "Not support selecting pg filters without printing entries"
                )
        # pyrefly: ignore [missing-attribute]
        if args.verbose:
            logger.set_log_level(logging.DEBUG)
        # pyrefly: ignore [bad-return]
        return args
