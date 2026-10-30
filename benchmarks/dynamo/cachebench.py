# [MODULE DOC] Benchmark of inductor compile-time caching. For each model of the torchbench
# or huggingface suite it runs the suite script in a subprocess twice per repeat (cold cache,
# then warm cache, inside a fresh cache directory), reads the compilation latency from the
# CSV output, computes the warm-start speedup percentage, and writes the records as JSON.
import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

from torch._inductor.utils import fresh_cache


logger: logging.Logger = logging.getLogger(__name__)

TIMEOUT: int = 2000


# Keep in sync with .ci/pytorch/test.sh
TORCHBENCH_MODELS: list[str] = [
    "nanogpt",
    "BERT_pytorch",
    "resnet50",
    "moco",
    "llama",
]
HUGGINGFACE_MODELS: list[str] = [
    "AllenaiLongformerBase",
    "BertForMaskedLM",
    "GPT2ForSequenceClassification",
]


@dataclasses.dataclass
class RunResult:
    # [CLASS DOC] Plain record of one model's result: model name, mode, benchmark suite,
    # whether dynamic shapes were used, device, lists of cold and warm compile times in
    # seconds (one per repeat), and the percent speedup of warm over cold.
    model: str
    mode: str  # inference or training
    benchmark: str
    dynamic: bool
    device: str  # cuda or cpu
    cold_compile_s: list[float]
    warm_compile_s: list[float]
    speedup_pct: float


def get_compile_time(file: tempfile._TemporaryFileWrapper) -> float:
    # [FUNCTION DOC] Parses the CSV file that a benchmark run wrote (read from the open
    # temporary file in binary mode): finds the compilation latency column in the header row
    # and returns the value from the first data row as a float. Raises ValueError if the
    # column is missing and IndexError if there is no data row.
    lines = file.readlines()
    # Decode from byte string, remove new lines, parse csv
    lines = [line.decode("utf-8").strip().split(",") for line in lines]
    compilation_time_idx = lines[0].index("compilation_latency")
    compilation_time = lines[1][compilation_time_idx]
    return float(compilation_time)


def _run_torchbench_from_args(
    cmd_args: argparse.Namespace,
    model: str,
    args: list[str],
) -> tuple[list[float], list[float]]:
    # [FUNCTION DOC] HELPER: For each repeat, inside a fresh inductor cache, runs the command
    # once for a cold start and once for a warm start, each with an output CSV in a temporary
    # file, and collects the compile times. Mutates the passed argument list by appending and
    # removing the output option. Subprocess failures or timeouts propagate. Returns the cold
    # and warm lists of compile times.
    cold_compile_time: list[float] = []
    warm_compile_time: list[float] = []

    for _ in range(cmd_args.repeat):
        with fresh_cache():
            env = os.environ.copy()
            with tempfile.NamedTemporaryFile(suffix=".csv") as file:
                args.append("--output=" + file.name)
                logger.info(f"Performing cold-start run for {model}")  # noqa: G004
                subprocess.check_call(args, timeout=TIMEOUT, env=env)
                cold_compile_time.append(get_compile_time(file))

            args.pop()
            with tempfile.NamedTemporaryFile(suffix=".csv") as file:
                args.append("--output=" + file.name)
                logger.info(f"Performing warm-start run for {model}")  # noqa: G004
                subprocess.check_call(args, timeout=TIMEOUT, env=env)
                warm_compile_time.append(get_compile_time(file))

    return cold_compile_time, warm_compile_time


MODE_ARGS_DICT = {
    "inference": ["--inference", "--bfloat16"],
    "training": ["--training", "--amp"],
}


BENCHMARK_FILE = {
    "torchbench": "torchbench.py",
    "huggingface": "huggingface.py",
}


def _run_torchbench_model(
    cmd_args: argparse.Namespace,
    results: list[RunResult],
    model: str,
) -> None:
    # [FUNCTION DOC] HELPER: Runs one model in the chosen suite: checks the suite script
    # exists (AssertionError otherwise), builds the command line for inductor performance
    # mode with mode and dynamic specific flags, runs the cold/warm measurement, and appends
    # a RunResult to the results list. Any exception during the run is logged and swallowed.
    # The speedup is one minus the ratio of summed warm to summed cold time, in percent.
    cur_file = os.path.abspath(__file__)
    torchbench_file = os.path.join(
        os.path.dirname(cur_file), BENCHMARK_FILE[cmd_args.benchmark]
    )
    if not os.path.exists(torchbench_file):
        raise AssertionError(f"Torchbench does not exist at {torchbench_file}")

    dynamic = cmd_args.dynamic
    dynamic_args = ["--dynamic-shapes", "--dynamic-batch-only"] if dynamic else []

    args = (
        [
            sys.executable,
            torchbench_file,
            f"--only={model}",
            "--repeat=1",
            "--performance",
            "--backend=inductor",
            f"--device={cmd_args.device}",
        ]
        + MODE_ARGS_DICT[cmd_args.mode]
        + dynamic_args
    )

    logger.info(f"Command: {args}")  # noqa: G004
    try:
        cold_compile_t, warm_compile_t = _run_torchbench_from_args(
            cmd_args, model, args
        )
        speedup_pct = (1 - (sum(warm_compile_t) / sum(cold_compile_t))) * 100
        results.append(
            RunResult(
                model=model,
                mode=cmd_args.mode,
                benchmark=cmd_args.benchmark,
                dynamic=dynamic,
                device=cmd_args.device,
                cold_compile_s=cold_compile_t,
                warm_compile_s=warm_compile_t,
                speedup_pct=speedup_pct,
            )
        )
    except Exception:
        logger.info("fail", exc_info=True)
        return None


def _write_results_to_json(
    cmd_args: argparse.Namespace,
    results: list[RunResult],
) -> None:
    # [FUNCTION DOC] HELPER: Writes the results to the output path as a JSON list of records,
    # three per result (cold times, warm times, speedup), each with benchmark, model and
    # metric sections. Does nothing and writes no file if the results list is empty.
    if len(results) == 0:
        # do not write empty results
        return

    records = []
    for result in results:
        for metric_name, value in [
            ("Cold compile time (s)", result.cold_compile_s),
            ("Warm compile time (s)", result.warm_compile_s),
            ("Speedup (%)", [result.speedup_pct]),
        ]:
            records.append(
                {
                    "benchmark": {
                        "name": "TorchCache Benchmark",
                        "mode": result.mode,
                        "extra_info": {
                            "is_dynamic": result.dynamic,
                            "device": result.device,
                        },
                    },
                    "model": {
                        "name": result.model,
                        "backend": "inductor",
                        "origins": [result.benchmark],
                    },
                    "metric": {
                        "name": metric_name,
                        "type": "OSS model",
                        "benchmark_values": value,
                    },
                }
            )
    with open(cmd_args.output, "w") as f:
        json.dump(records, f)


def parse_cmd_args() -> argparse.Namespace:
    # [FUNCTION DOC] Parses command line options (model, dynamic flag, required benchmark
    # suite, mode, device, required output path, repeat count 1 to 9), ignoring unknown
    # arguments, and returns the namespace.
    parser = argparse.ArgumentParser(description="Run a TorchCache benchmark.")
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to run",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Whether to run with dynamic enabled",
    )
    parser.add_argument(
        "--benchmark",
        choices=("torchbench", "huggingface"),
        required=True,
        help="Name of benchmark suite to run",
    )
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="training",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The output filename (json)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Number of times to repeat the compilation (reduce noise)",
    )
    args, _ = parser.parse_known_args()
    return args


Dispatch_fn_t = Callable[[argparse.Namespace, list[RunResult], str], None]


def main() -> None:
    # [FUNCTION DOC] Entry point. Chooses the default model list for the selected suite, or
    # only the one named model, runs each via the dispatcher, and writes the JSON results.
    cmd_args = parse_cmd_args()

    dispatcher: dict[str, tuple[Dispatch_fn_t, list[str]]] = {
        "torchbench": (_run_torchbench_model, TORCHBENCH_MODELS),
        "huggingface": (_run_torchbench_model, HUGGINGFACE_MODELS),
    }
    fn, models = dispatcher[cmd_args.benchmark]
    if cmd_args.model is not None:
        models = [cmd_args.model]

    results: list[RunResult] = []
    for model in models:
        fn(cmd_args, results, model)

    _write_results_to_json(cmd_args, results)


if __name__ == "__main__":
    main()
