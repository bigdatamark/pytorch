# [MODULE DOC] Script for profiling torch.add dispatch with an external sampling profiler. Run
# with a class name (a name defined in the common module, or Tensor for the plain constructor)
# and optional repeat count (default one million); it builds two one-element inputs of that
# class and calls the add op in a loop. The class is looked up by name in the module globals, so
# an unknown name raises a lookup error. All work happens under the main guard; nothing runs on
# import. The common-module imports are kept only so the names resolve.
import argparse

from common import SubTensor, SubWithTorchFunction, WithTorchFunction  # noqa: F401

import torch


Tensor = torch.tensor

NUM_REPEATS = 1000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the torch.add for a given class a given number of times."
    )
    parser.add_argument(
        "tensor_class", metavar="TensorClass", type=str, help="The class to benchmark."
    )
    parser.add_argument(
        "--nreps", "-n", type=int, default=NUM_REPEATS, help="The number of repeats."
    )
    args = parser.parse_args()

    TensorClass = globals()[args.tensor_class]
    NUM_REPEATS = args.nreps

    t1 = TensorClass([1.0])
    t2 = TensorClass([2.0])

    for _ in range(NUM_REPEATS):
        torch.add(t1, t2)
