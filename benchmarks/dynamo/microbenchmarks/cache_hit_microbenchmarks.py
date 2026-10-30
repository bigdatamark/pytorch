# [MODULE DOC] Microbenchmark of the Inductor FX graph cache hit path for a compiled function
# that applies sin 10000 times. Requires CUDA. Run as a script; enables the local graph cache
# and disables the remote one, uses a fresh cache directory, and prints best-of-three time for K
# calls.
import os
import timeit

import torch.fx
from torch._dynamo.utils import counters
from torch._inductor.utils import clear_caches, fresh_cache


N = 10000
K = 100


def huge_graph(x):
    # [FUNCTION DOC] HELPER: Uncompiled workload applying sin to x N times in sequence and
    # returning it; compiled by the benchmark.
    for _ in range(N):
        x = x.sin()
    return x


def main():
    # [FUNCTION DOC] Run the cache hit benchmark. Sets the Inductor config, enters a fresh cache
    # context, compiles once to populate the cache and checks exactly one cache miss was
    # recorded (raises an error otherwise). Then times K calls, where each repetition first
    # resets Dynamo, clears in-memory caches, deletes the generated Python cache module files
    # and clears counters, and each call verifies zero misses and one hit. Prints the best total
    # time.
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.fx_graph_remote_cache = False

    with fresh_cache():
        a = torch.randn(4).cuda()
        compiled_fn = torch.compile(huge_graph, backend="inductor")

        # write to cache
        compiled_fn(a)
        if counters["inductor"]["fxgraph_cache_miss"] != 1:
            raise AssertionError(
                f"expected fxgraph_cache_miss == 1, got {counters['inductor']['fxgraph_cache_miss']}"
            )

        def setup():
            # [FUNCTION DOC] HELPER: Untimed per-repetition setup: reset Dynamo, clear Inductor
            # caches, remove the on-disk module files of the cached Python code, and clear the
            # counters so the next call must load from the FX graph cache.
            torch._dynamo.reset()
            clear_caches()
            for m in torch._inductor.codecache.PyCodeCache.cache.values():
                os.remove(m.__file__)
            counters.clear()

        def fn():
            # [FUNCTION DOC] HELPER: Timed callable. Runs the compiled function and raises an
            # error unless the counters show zero cache misses and exactly one cache hit.
            # Returns the result.
            result = compiled_fn(a)
            if counters["inductor"]["fxgraph_cache_miss"] != 0:
                raise AssertionError(
                    f"expected fxgraph_cache_miss == 0, got {counters['inductor']['fxgraph_cache_miss']}"
                )
            if counters["inductor"]["fxgraph_cache_hit"] != 1:
                raise AssertionError(
                    f"expected fxgraph_cache_hit == 1, got {counters['inductor']['fxgraph_cache_hit']}"
                )
            return result

        t = min(timeit.repeat(fn, setup=setup, number=K, repeat=3))
        print(f"took {t:.1f}s")


if __name__ == "__main__":
    main()
