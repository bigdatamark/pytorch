# [MODULE DOC] Script measuring Dynamo compile time with the eager backend for a chain of 101
# nested functions, each calling the next between helper additions, where the innermost one has
# an explicit graph break. Compares the nested graph break config on and off. Import-time side
# effects: builds the chain, raises the Python and Dynamo recursion limits.
import sys
import time

import torch


DEPTH = 100


def gn(x):
    # [FUNCTION DOC] Add one to x 100 times in a loop and return it; the filler workload used
    # around each call level.
    for _ in range(DEPTH):
        x = x + 1
    return x


def make_fn(next_fn):
    # [FUNCTION DOC] Build one level of the chain. With no next function it returns a function
    # that runs the helper, forces a Dynamo graph break, then runs the helper again. Otherwise
    # it returns a function that applies the helper, then the next function, then the helper
    # again. The returned function gets a copy of its own code object so each level is a
    # distinct code object, which avoids recompilation limits and fallback to eager.
    if next_fn is None:

        def fn(x):
            # [FUNCTION DOC] HELPER: Innermost level; helper, explicit graph break, helper
            # again, returning the tensor.
            x = gn(x)
            torch._dynamo.graph_break()
            return gn(x)
    else:

        def fn(x):
            # [FUNCTION DOC] HELPER: Middle level; applies the helper, calls the enclosed next
            # function, applies the helper again, and returns the result.
            return gn(next_fn(gn(x)))

    # to prevent recompilation + fallback to eager
    fn.__code__ = fn.__code__.replace()
    return fn


fns = [make_fn(None)]
for _ in range(DEPTH):
    fns.append(make_fn(fns[-1]))

top_fn = fns[-1]

sys.setrecursionlimit(100000)
torch._dynamo.set_recursion_limit(1000000)


def main():
    # [FUNCTION DOC] Time the plain eager call of the outermost function, then compile it with
    # the eager backend and time the first call with nested graph breaks enabled, reset the
    # compiler, and repeat with it disabled. Sets the Dynamo config flag globally as a side
    # effect and prints each elapsed time.
    start = time.perf_counter()
    print(top_fn(torch.ones(3)))
    end = time.perf_counter()

    print(f"eager total time: {end - start:.2f}s")

    opt_fn = torch.compile(top_fn, backend="eager")

    torch._dynamo.config.nested_graph_breaks = True
    start = time.perf_counter()
    opt_fn(torch.ones(3))
    end = time.perf_counter()

    print(f"nested_graph_breaks=True total time: {end - start:.2f}s")

    torch.compiler.reset()

    torch._dynamo.config.nested_graph_breaks = False
    start = time.perf_counter()
    opt_fn(torch.ones(3))
    end = time.perf_counter()

    print(f"nested_graph_breaks=False total time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
