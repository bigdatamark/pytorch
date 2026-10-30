# [MODULE DOC] Microbenchmark of how fast the nodes of a very large FX graph (100000 sin nodes)
# can be iterated. Run as a script; prints best-of-three nodes per second. Defines constants N
# and K at import.
import timeit

import torch.fx


N = 100000
K = 1000


def huge_graph():
    # [FUNCTION DOC] Build and return an FX GraphModule by symbolically tracing a function
    # applying sin to its input N times.
    def fn(x):
        # [FUNCTION DOC] HELPER: Traced function applying sin to the input N times in sequence
        # and returning it.
        for _ in range(N):
            x = x.sin()
        return x

    return torch.fx.symbolic_trace(fn)


def main():
    # [FUNCTION DOC] Build the huge graph, time K full iterations over its node list (best of
    # three repeats), and print the total node count iterated and nodes per second.
    g = huge_graph()

    def fn():
        # [FUNCTION DOC] HELPER: Walk every node of the graph once, doing nothing; the timed
        # callable.
        for n in g.graph.nodes:
            pass

    t = min(timeit.repeat(fn, number=K, repeat=3))
    print(f"iterating over {N * K} FX nodes took {t:.1f}s ({N * K / t:.0f} nodes/s)")


if __name__ == "__main__":
    main()
