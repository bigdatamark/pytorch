# [MODULE DOC] Script using the external pyarkbench harness to time saving and loading of tensor
# lists with torch.save and torch.load, either with the new zipfile format or the legacy one,
# selected by a module-level flag. Defines one benchmark class; the main block builds it from
# default harness arguments, runs it and prints mean and median statistics. Writes and reads
# files in the current directory.
from pyarkbench import Benchmark, default_args, Timer

import torch


use_new = True


class Basic(Benchmark):
    # [CLASS DOC] Harness benchmark with one measurement routine covering save and load of a big
    # and a small tensor collection.
    def benchmark(self):
        # [FUNCTION DOC] Builds a list of thirty 200x200 float tensors and times saving it to a
        # zip file and loading it back, then does the same for a list of two hundred 10x10
        # tensors using separate files. Files are created in the working directory and not
        # deleted. Returns a mapping from four labels (big and small, save and load) to the
        # elapsed milliseconds reported by the timers. The save uses the module flag to pick the
        # serialization format.
        x = [torch.ones(200, 200) for i in range(30)]
        with Timer() as big1:
            torch.save(x, "big_tensor.zip", _use_new_zipfile_serialization=use_new)

        with Timer() as big2:
            torch.load("big_tensor.zip")

        x = [torch.ones(10, 10) for i in range(200)]
        with Timer() as small1:
            torch.save(x, "small_tensor.zip", _use_new_zipfile_serialization=use_new)

        with Timer() as small2:
            torch.load("small_tensor.zip")

        return {
            "Big Tensors Save": big1.ms_duration,
            "Big Tensors Load": big2.ms_duration,
            "Small Tensors Save": small1.ms_duration,
            "Small Tensors Load": small2.ms_duration,
        }


if __name__ == "__main__":
    bench = Basic(*default_args.bench())
    print("Use zipfile serialization:", use_new)
    results = bench.run()
    bench.print_stats(results, stats=["mean", "median"])
