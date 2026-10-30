# [MODULE DOC] Microbenchmark of Dynamo symbolic conversion overhead using a function whose loop
# is unrolled 100000 times during tracing. Running as a script times the compile (best of three)
# and then writes a cProfile dump. Import-time side effect: defines a function compiled with the
# eager backend and full graph.
import cProfile
import pstats
import timeit

import torch


@torch.compile(backend="eager", fullgraph=True)
def symbolic_convert_overhead_stress_test(x, y, n):
    # [FUNCTION DOC] Stress workload compiled with the eager backend and full graph. Loops n
    # times, decrementing n and swapping x and y each time, then returns the sum of x and y.
    # Because n is a plain integer, tracing unrolls the loop, which stresses the bytecode
    # interpreter of Dynamo.
    while n > 0:
        n -= 1
        x, y = y, x
    return x + y


def main():
    # [FUNCTION DOC] Time the compile of the stress function for 100000 iterations: each of
    # three repeats resets Dynamo first so every run recompiles, and the minimum time is printed
    # in seconds.
    def fn():
        # [FUNCTION DOC] HELPER: Reset Dynamo and run the stress function on the enclosing x and
        # y tensors with 100000 iterations.
        torch._dynamo.reset()
        symbolic_convert_overhead_stress_test(x, y, 100000)

    x = torch.randn(16)
    y = torch.randn(16)
    t = min(timeit.repeat(fn, number=1, repeat=3))
    print(f"symbolic_convert_overhead_stress_test: {t:.1f}s")


def profile():
    # [FUNCTION DOC] Profile one compile of the stress function with 33000 iterations under
    # cProfile after resetting Dynamo, write the stats file named for this script into the
    # current directory, and print the command used to view it. Side effect is writing that
    # file.
    x = torch.randn(16)
    y = torch.randn(16)
    torch._dynamo.reset()
    pr = cProfile.Profile()
    pr.enable()
    # 100k > 33k roughly cancels out the overhead of cProfile
    symbolic_convert_overhead_stress_test(x, y, 33000)
    pr.disable()
    ps = pstats.Stats(pr)
    ps.dump_stats("dynamo_microbenchmarks.prof")
    print("snakeviz dynamo_microbenchmarks.prof")


if __name__ == "__main__":
    main()
    profile()
