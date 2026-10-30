# [MODULE DOC] Microbenchmarks of Dynamo guard evaluation cost. One case runs a generated
# function with 100 tensor arguments that is compiled statically and called with ten different
# argument shapes (so several cache entries and guard sets are walked). The other cases compile
# stacks of repeated nn.Module layers sharing one object and time the guard manager check
# directly. Import-time side effects: generates the function source and executes it into the
# module globals, and sets the Dynamo recompile limit to 16. Prints median microsecond timings.
import time
import timeit

import numpy as np

import torch
import torch._dynamo.config
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


# to satisfy linter complaining about undefined variable
foo = None

args = [f"x{i}" for i in range(100)]
fn_str = f"""\
def foo({", ".join(args)}):
    n = {" + ".join(arg + ".shape[0]" for arg in args)}
    return x0 + n
"""

exec(fn_str, globals())
torch._dynamo.config.recompile_limit = 16


def bench(name, fn):
    # [FUNCTION DOC] Benchmark calling a many-argument function. Resets Dynamo, builds ten input
    # sets (each a list of 100 random vectors, with lengths 10 to 100 in steps of 10), warms up
    # with three passes over all input sets, then takes the median of ten timing repeats of one
    # thousand passes and prints name, microseconds per pass and warmup seconds. Args are a
    # label and the callable.
    torch._dynamo.reset()
    inps = [[torch.randn(i) for _ in range(100)] for i in range(10, 101, 10)]

    def run_fn():
        # [FUNCTION DOC] HELPER: Call the benchmarked function once for each prepared input set.
        for inp in inps:
            fn(*inp)

    start = time.perf_counter()
    for _ in range(3):
        run_fn()
    end = time.perf_counter()

    results = timeit.repeat(lambda: run_fn(), number=1000, repeat=10)
    print(f"{name} {np.median(results) * 1000:.1f}us (warmup={end - start:.1f}s)")


# Models with many repeated submodules (e.g. decoder stacks) exercise guard
# evaluation that scales with the number of inlined nn.Modules: per-layer scalar
# attribute guards, object-aliasing guards from shared objects, and relational
# symbolic-shape guards. See https://github.com/pytorch/pytorch/issues/185886.
#
# The same module pattern is mirrored in
# benchmarks/dynamo/pr_time_benchmarks/benchmarks/nn_module_guard_eval.py; keep
# the two in sync.
class _Shared:
    # [CLASS DOC] Tiny plain object holding a single boolean attribute, set to true, that is
    # shared by every repeated layer so Dynamo sees the same object reached through many
    # modules.
    def __init__(self):
        # [FUNCTION DOC] Create the shared object with its flag attribute set to true.
        self.flag = True


class _RepeatedLayer(torch.nn.Module):
    # [CLASS DOC] Module layer with a float epsilon, a per-layer float scale that depends on its
    # index, a registered random bias buffer of the hidden size, and a reference to a shared
    # object (kept by reference, not copied). Forward applies the scale and bias to the input.
    def __init__(self, idx, hidden, shared):
        # [FUNCTION DOC] Store a constant epsilon, a scale equal to one half plus a tiny
        # multiple of the layer index, a random bias buffer of the hidden size, and the shared
        # object reference.
        super().__init__()
        self.eps = 1e-6
        self.scale = 0.5 + idx * 1e-3
        self.register_buffer("bias", torch.randn(hidden))
        self.shared = shared

    def forward(self, x):
        # [FUNCTION DOC] Add epsilon to the input when the shared flag is true (always the
        # case), then multiply by the scale and add the bias. The read of the shared flag is
        # deliberate so that aliasing guards are generated. Returns the result.
        # Load-bearing: reading the shared object is what makes Dynamo emit the
        # object-aliasing guards (layers[i].shared is layers[j].shared) this
        # benchmark exercises. `flag` is always True, so do not "simplify" the
        # branch away -- doing so silently removes those guards.
        if self.shared.flag:
            x = x + self.eps
        return x * self.scale + self.bias


class _RepeatedModules(torch.nn.Module):
    # [CLASS DOC] Module holding a module list of the given number of repeated layers, all
    # constructed with one shared object so each layer references the same instance. Forward
    # applies the layers in sequence.
    def __init__(self, n_layers, hidden):
        # [FUNCTION DOC] Build one shared object and a module list with one repeated layer per
        # index, passing the hidden size and the shared object to each.
        super().__init__()
        shared = _Shared()
        self.layers = torch.nn.ModuleList(
            [_RepeatedLayer(i, hidden, shared) for i in range(n_layers)]
        )

    def forward(self, x):
        # [FUNCTION DOC] Pass the input through every layer in order and return the final
        # tensor.
        for layer in self.layers:
            x = layer(x)
        return x


def bench_guard_check(name, n_layers, hidden=64):
    # [FUNCTION DOC] Benchmark guard evaluation for the repeated module model with the given
    # layer count and hidden size (default 64). Resets Dynamo, compiles the model with dynamic
    # shapes and full graph on a sliced view input, runs it once, fetches the first cache entry
    # guard manager from the forward code, and raises RuntimeError if the guard check on the
    # live locals fails. Then takes the median of twenty repeats of one thousand checks and
    # prints microseconds per check plus the compile time.
    # Time the compiled guard tree in isolation (guard_manager.check) so the
    # measurement reflects guard-eval cost, not the trivial per-layer compute.
    torch._dynamo.reset()
    model = _RepeatedModules(n_layers, hidden)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    x = torch.randn(128, hidden)[:64]  # a view, so it has a ._base

    start = time.perf_counter()
    compiled(x)
    end = time.perf_counter()

    guard_manager = _debug_get_cache_entry_list(type(model).forward.__code__)[
        0
    ].guard_manager
    f_locals = {"self": model, "x": x}
    if not guard_manager.check(f_locals):
        raise RuntimeError("guard check failed; cannot benchmark guard eval")

    results = timeit.repeat(
        lambda: guard_manager.check(f_locals), number=1000, repeat=20
    )
    print(f"{name} {np.median(results) * 1000:.1f}us (warmup={end - start:.1f}s)")


def main():
    # [FUNCTION DOC] Run the many-argument static compile benchmark, then the guard check
    # benchmark for 10, 40 and 80 repeated layers.
    bench("compiled", torch.compile(foo, dynamic=False))  # type: ignore[F821]
    for n_layers in (10, 40, 80):
        bench_guard_check(f"repeated_modules[{n_layers}]", n_layers)


if __name__ == "__main__":
    main()
