# [MODULE DOC]
# Defines BenchmarkRunner, the abstract base class for scripts that collect AutoHeuristic
# training data (or exercise a learned heuristic) by repeatedly running generated benchmarks.
# Importing has no side effects beyond imports (tqdm and torch are required).
import argparse
import random
import time
from abc import abstractmethod
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

import torch


class BenchmarkRunner:
    # [CLASS DOC] Base class for benchmark runners. Holds the heuristic name (self.name), an
    # argument parser preloaded with common flags (self.parser) and the parsed args (self.args,
    # initially unset). Subclasses implement create_input and run_benchmark; run drives the
    # whole collection or heuristic-usage loop.
    """
    BenchmarkRunner is a base class for all benchmark runners. It provides an interface to run benchmarks in order to
    collect data with AutoHeuristic.
    """

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Stores the heuristic name, creates the argument parser, registers the
        # base command line arguments and initializes args to None. No I/O.
        self.name = name
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self) -> None:
        # [FUNCTION DOC] Registers the common flags on the parser: --device (int GPU index,
        # default none), --use-heuristic (boolean), -o (log file path, default ah_data.txt),
        # --num-samples (default 1000) and --num-reps (default 3). Mutates self.parser only.
        self.parser.add_argument(
            "--device",
            type=int,
            default=None,
            help="torch.cuda.set_device(device) will be used",
        )
        self.parser.add_argument(
            "--use-heuristic",
            action="store_true",
            help="Use learned heuristic instead of collecting data.",
        )
        self.parser.add_argument(
            "-o",
            type=str,
            default="ah_data.txt",
            help="Path to file where AutoHeuristic will log results.",
        )
        self.parser.add_argument(
            "--num-samples",
            type=int,
            default=1000,
            help="Number of samples to collect.",
        )
        self.parser.add_argument(
            "--num-reps",
            type=int,
            default=3,
            help="Number of measurements to collect for each input.",
        )

    def run(self) -> None:
        # [FUNCTION DOC] Entry point. Sets the default torch device to cuda, parses the command
        # line, and sets the autoheuristic USE / COLLECT environment variables: with
        # --use-heuristic, USE is the runner name and COLLECT is empty; otherwise the reverse.
        # Also sets the inductor autoheuristic log path to the -o value, selects the CUDA
        # device if given, seeds python random from the current time, then calls main.
        # Side effects: global torch default device, environment, inductor config, RNG seed.
        # KNOWN ISSUE: self.args is never assigned; the parsed args stay in a local variable.
        torch.set_default_device("cuda")
        args = self.parser.parse_args()
        # Set environment variables to control autoheuristic behavior
        import os

        if args.use_heuristic:
            os.environ["TORCHINDUCTOR_AUTOHEURISTIC_USE"] = self.name
            os.environ["TORCHINDUCTOR_AUTOHEURISTIC_COLLECT"] = ""
        else:
            os.environ["TORCHINDUCTOR_AUTOHEURISTIC_USE"] = ""
            os.environ["TORCHINDUCTOR_AUTOHEURISTIC_COLLECT"] = self.name
        torch._inductor.config.autoheuristic_log_path = args.o
        if args.device is not None:
            torch.cuda.set_device(args.device)
        random.seed(time.time())
        self.main(args.num_samples, args.num_reps)

    @abstractmethod
    # [FUNCTION DOC] Abstract hook: run one benchmark for the unpacked input tuple.
    # (Placed above the one-line def to leave the def line itself untouched.)
    def run_benchmark(self, *args: Any) -> None: ...

    @abstractmethod
    # [FUNCTION DOC] Abstract hook: produce one random input tuple for run_benchmark.
    # (Placed above the one-line def to leave the def line itself untouched.)
    def create_input(self) -> tuple[Any, ...]: ...

    def main(self, num_samples: int, num_reps: int) -> None:
        # [FUNCTION DOC] Loops num_samples times (with a progress bar): creates one input and
        # runs the benchmark on it num_reps times. Returns nothing.
        for _ in tqdm(range(num_samples)):
            input = self.create_input()
            for _ in range(num_reps):
                self.run_benchmark(*input)
