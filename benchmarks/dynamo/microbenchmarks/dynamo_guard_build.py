# [MODULE DOC] Script measuring Dynamo compile time (eager backend) for a function whose guards
# must walk a very deep structure: an object with 2000 integer attributes wrapped in 2000 nested
# single-element lists. Import-time side effects: builds that structure, raises the Python
# recursion limit and the Dynamo recursion limit, and defines the compiled function. Running as
# a script compiles once and prints the elapsed time.
import sys
import time

import torch


class Foo:
    # [CLASS DOC] Empty plain class used only as a container that receives many dynamically
    # assigned integer attributes (named attr0 up to attr1999) at import time.
    pass


obj = Foo()

DEPTH = 2000

attrs = [f"attr{i}" for i in range(DEPTH)]

for i, attr in enumerate(attrs):
    setattr(obj, attr, i)

lst = obj

for _ in range(DEPTH):
    lst = [lst]

sys.setrecursionlimit(100000)
torch._dynamo.set_recursion_limit(1000000)


@torch.compile(backend="eager")
def fn(x):
    # [FUNCTION DOC] Compiled with the eager backend. Unwraps the module-level nested list by
    # taking the first element 2000 times to reach the attribute holder, then adds each of its
    # 2000 integer attributes in order to the input tensor and returns the sum. Reads globals so
    # guards on the deep structure are generated.
    unpacked = lst
    for _ in range(DEPTH):
        unpacked = unpacked[0]
    for i in range(DEPTH):
        x = x + getattr(unpacked, f"attr{i}")
    return x


def main():
    # [FUNCTION DOC] Wrap the already compiled function in another eager compile, time a single
    # call on a random three element tensor (this includes tracing and guard construction), and
    # print the elapsed seconds.
    opt_fn = torch.compile(fn, backend="eager")

    start = time.perf_counter()
    opt_fn(torch.randn(3))
    end = time.perf_counter()

    print(f"total time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
