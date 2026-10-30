# [MODULE DOC] Command-line benchmark comparing the dispatch overhead of torch.add on a plain
# tensor and on the subclass or protocol-implementing classes from the common module. Module
# globals hold the repeat counts, which main may overwrite from command-line options. Nothing
# runs at import time; main runs when executed as a script.
import argparse
import time

from common import SubTensor, SubWithTorchFunction, WithTorchFunction

import torch


NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


def bench(t1, t2):
    # [FUNCTION DOC] Times repeated calls of the add op on the two given operands. Runs the
    # inner loop for the module-wide number of repeats, once per outer repetition, recording
    # wall-clock seconds for each outer repetition. Returns a pair: the minimum and the standard
    # deviation of those times, each divided by 1000 (a per-call figure only when the inner
    # repeat count is 1000). Reads module globals only.
    bench_times = []
    for _ in range(NUM_REPEAT_OF_REPEATS):
        time_start = time.time()
        for _ in range(NUM_REPEATS):
            torch.add(t1, t2)
        bench_times.append(time.time() - time_start)

    bench_time = float(torch.min(torch.tensor(bench_times))) / 1000
    bench_std = float(torch.std(torch.tensor(bench_times))) / 1000

    return bench_time, bench_std


def main():
    # [FUNCTION DOC] Script entry point. Parses command-line options for the inner repeat count
    # (nreps) and the outer repetition count (nrepreps), stores them into the module globals,
    # then benchmarks the add op for a plain tensor constructor, the tensor subclass, the
    # torch-function class and the subclass overriding torch-function, each on one-element float
    # inputs. Prints one line per type with the minimum time and standard deviation in
    # microseconds. Exits via the parser on bad arguments.
    global NUM_REPEATS
    global NUM_REPEAT_OF_REPEATS

    parser = argparse.ArgumentParser(
        description="Run the __torch_function__ benchmarks."
    )
    parser.add_argument(
        "--nreps",
        "-n",
        type=int,
        default=NUM_REPEATS,
        help="The number of repeats for one measurement.",
    )
    parser.add_argument(
        "--nrepreps",
        "-m",
        type=int,
        default=NUM_REPEAT_OF_REPEATS,
        help="The number of measurements.",
    )
    args = parser.parse_args()

    NUM_REPEATS = args.nreps
    NUM_REPEAT_OF_REPEATS = args.nrepreps

    types = torch.tensor, SubTensor, WithTorchFunction, SubWithTorchFunction

    for t in types:
        tensor_1 = t([1.0])
        tensor_2 = t([2.0])

        bench_min, bench_std = bench(tensor_1, tensor_2)
        print(
            f"Type {t.__name__} had a minimum time of {10**6 * bench_min} us"
            f" and a standard deviation of {(10**6) * bench_std} us."
        )


if __name__ == "__main__":
    main()
