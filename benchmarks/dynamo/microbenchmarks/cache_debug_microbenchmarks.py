# [MODULE DOC] Microbenchmark measuring how fast Inductor's FxGraphHashDetails can produce its
# debug lines over a very large traced FX graph (10000 nodes, repeated 100 times). Run as a
# script; prints nodes per second. Defines the constants N and K at import; no other side
# effects.
import timeit

import torch.fx
from torch._inductor.codecache import FxGraphHashDetails


N = 10000
K = 100


def huge_graph():
    # [FUNCTION DOC] Build and return an FX GraphModule by symbolically tracing a function that
    # applies the sin method N times in a loop to its input.
    def fn(x):
        # [FUNCTION DOC] HELPER: Traced function applying sin to the input N times in sequence
        # and returning the result.
        for _ in range(N):
            x = x.sin()
        return x

    return torch.fx.symbolic_trace(fn)


def main():
    # [FUNCTION DOC] Build the huge graph, wrap it in a hash-details object with empty example
    # inputs, config and constants lists, time the debug-line generation (best of three repeats
    # of K calls), and print the number of nodes iterated and nodes per second.
    g = huge_graph()
    details = FxGraphHashDetails(g, [], {}, [])

    def fn():
        # [FUNCTION DOC] HELPER: Return the debug lines of the captured hash-details object; the
        # timed callable.
        return details.debug_lines()

    t = min(timeit.repeat(fn, number=K, repeat=3))
    print(f"iterating over {N * K} FX nodes took {t:.1f}s ({N * K / t:.0f} nodes/s)")


if __name__ == "__main__":
    main()
