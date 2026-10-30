# [MODULE DOC] Command line benchmark of batched matrix multiply on CUDA nested tensors. For
# float32 and then float16 it sweeps the number of constituent tensors (4 up to 256), times a
# nested bmm with CUDA events, and prints one comma separated row per setting (count, dtype,
# min/mean/max length, runtime). Requires a CUDA device. Running as a script seeds the random
# generator with 123 and accepts an iteration count option (default 10).
import argparse
import random

import torch


def bench(nt_a, nt_b, niter):
    # [FUNCTION DOC] Times a nested tensor bmm between two nested tensors on the GPU.
    # Runs one warmup call, synchronizes, then records CUDA events around the given
    # number of iterations and returns the mean time per iteration in milliseconds (as
    # reported by the CUDA event elapsed time). Requires CUDA.
    # Warmup
    nt_a.bmm(nt_b)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for iter in range(niter):
        nt_a.bmm(nt_b)
    end_event.record()
    torch.cuda.synchronize()
    runtime = (start_event.elapsed_time(end_event)) / niter
    return runtime


def sweep_n(niter, dtype):
    # [FUNCTION DOC] Runs the benchmark sweep for one dtype. For each tensor count in a
    # fixed list it builds random 2D tensors with 256 rows and a random length between 100
    # and 200, makes a CUDA nested tensor from them and another from their transposes, times
    # the bmm and prints a comma separated line: count, dtype, minimum, mean and maximum of
    # the second dimension of the first nested tensor's sizes, and the runtime from the
    # timing helper. Output goes to stdout; returns nothing.
    for ntensor in [4, 8, 16, 32, 64, 128, 256]:
        tensors = [torch.randn(256, random.randint(100, 200)) for t in range(ntensor)]
        nt_a = torch.nested.nested_tensor(
            tensors,
            dtype=dtype,
            device="cuda",
        )
        nt_b = torch.nested.nested_tensor(
            [t.t() for t in tensors],
            dtype=dtype,
            device="cuda",
        )
        runtime = bench(nt_a, nt_b, niter)
        nt_a_size = torch.ops.aten._nested_tensor_size(nt_a)
        lengths = nt_a_size[:, 1]
        print(
            ",".join(
                map(
                    str,
                    [
                        ntensor,
                        dtype,
                        lengths.min().item(),
                        lengths.float().mean().item(),
                        lengths.max().item(),
                        runtime,
                    ],
                )
            )
        )


if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser(description="Nested Tensor BMM Benchmark")
    parser.add_argument("--niter", default="10", type=int)

    args = parser.parse_args()
    niter = args.niter

    print("ntensor,dtype,min_length,mean_length,max_length,runtime")
    sweep_n(niter, torch.float32)
    sweep_n(niter, torch.float16)
