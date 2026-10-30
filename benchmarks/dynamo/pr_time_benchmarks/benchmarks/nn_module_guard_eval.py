# [MODULE DOC] PR-time benchmark measuring the runtime cost of evaluating dynamo guards for a
# model made of many repeated submodules that share one object, in static and dynamic shape
# variants. The model is compiled once during setup and only the guard check on the cached entry
# is measured. Uses the private cache-entry debugging helper of the eval frame module. Run as a
# script with one argument, the CSV result path.
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


# Guard evaluation for models with many repeated submodules (e.g. decoder
# stacks) scales with the number of inlined nn.Modules: per-layer scalar
# attribute guards, object-aliasing guards from shared objects, and relational
# symbolic-shape guards. This benchmark guards against regressions in that
# runtime guard-eval cost. See https://github.com/pytorch/pytorch/issues/185886.
#
# The same module pattern is mirrored in
# benchmarks/dynamo/microbenchmarks/dynamo_guard_eval.py; keep the two in sync.
class Shared:
    # [CLASS DOC] HELPER: Plain object with one boolean attribute that is always true; shared
    # between all layers so dynamo emits object-aliasing guards.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Sets the flag attribute to true.
        self.flag = True


class GuardEvalLayer(nn.Module):
    # [CLASS DOC] HELPER: Layer with a scalar epsilon, an index-dependent scalar scale, a random
    # bias buffer of the hidden size and a reference to the shared object.
    def __init__(self, idx, hidden, shared):
        # [FUNCTION DOC] HELPER: Stores a small epsilon, a scale of one half plus a small
        # multiple of the layer index, registers a random bias buffer of the hidden size, and
        # keeps the shared object.
        super().__init__()
        self.eps = 1e-6
        self.scale = 0.5 + idx * 1e-3
        self.register_buffer("bias", torch.randn(hidden))
        self.shared = shared

    def forward(self, x):
        # [FUNCTION DOC] HELPER: When the shared flag is true (always) adds epsilon to the
        # input; returns the input times the scale plus the bias. Reading the shared flag is
        # intentional, it is what creates the aliasing guards.
        # Load-bearing: reading the shared object is what makes Dynamo emit the
        # object-aliasing guards (layers[i].shared is layers[j].shared) this
        # benchmark exercises. `flag` is always True, so do not "simplify" the
        # branch away -- doing so silently removes those guards.
        if self.shared.flag:
            x = x + self.eps
        return x * self.scale + self.bias


class RepeatedModules(nn.Module):
    # [CLASS DOC] HELPER: Model holding a list of layers that all reference one shared object.
    def __init__(self, n_layers, hidden):
        # [FUNCTION DOC] HELPER: Creates one shared object and a module list with n_layers
        # layers of the given hidden size, each given its index.
        super().__init__()
        shared = Shared()
        self.layers = nn.ModuleList(
            [GuardEvalLayer(i, hidden, shared) for i in range(n_layers)]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Passes the input through every layer in order and returns
        # the result.
        for layer in self.layers:
            x = layer(x)
        return x


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with category nn_module_guard_eval, eager backend, cpu device,
    # parameterized by dynamic. Class constants fix 20 layers and hidden size 64. State: model,
    # input, guard manager and the frame locals mapping used for guard checks.
    N_LAYERS = 20
    HIDDEN = 64

    def __init__(self, dynamic):
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend and cpu
        # device, and the dynamic flag.
        super().__init__(
            category="nn_module_guard_eval",
            backend="eager",
            device="cpu",
            dynamic=dynamic,
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category with a dynamic suffix when dynamic.
        prefix = self.category()
        if self.is_dynamic():
            prefix += "_dynamic"
        return prefix

    def description(self):
        # [FUNCTION DOC] Returns a fixed description string.
        return "runtime guard-eval cost for a stack of repeated nn.Modules"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: resets dynamo, builds the repeated-module model, and
        # with C++ symbolic shape guards disabled compiles it (eager backend, full graph,
        # dynamic setting) and calls it once on a view slice of a larger random tensor (so the
        # input has a base tensor). Then fetches the first cache entry guard manager for the
        # model forward code, builds the locals mapping with self and x, and raises RuntimeError
        # if the guards do not pass on those locals.
        torch._dynamo.reset()
        self._model = RepeatedModules(self.N_LAYERS, self.HIDDEN)
        # Pin enable_cpp_symbolic_shape_guards so the dynamic baseline is robust
        # to a future default flip: with it True the symbolic-shape guards move
        # into a compiled C++ guard, shifting the instruction count with no
        # change to expected_results.csv. (The static variant has no
        # symbolic-shape guards and is unaffected.)
        with torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=False):
            compiled = torch.compile(
                self._model,
                fullgraph=True,
                backend=self.backend(),
                dynamic=self.is_dynamic(),
            )
            # a view, so the input has a ._base (exercises relational shape guards)
            self._x = torch.randn(2 * self.HIDDEN, self.HIDDEN)[: self.HIDDEN]
            compiled(self._x)

        self._guard_manager = _debug_get_cache_entry_list(
            type(self._model).forward.__code__
        )[0].guard_manager
        self._f_locals = {"self": self._model, "x": self._x}
        if not self._guard_manager.check(self._f_locals):
            raise RuntimeError("guard check failed; cannot benchmark guard eval")

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: does nothing.
        pass

    def _work(self):
        # [FUNCTION DOC] Measured work: evaluates the guard manager on the stored locals,
        # result discarded.
        # Time guard evaluation in isolation so the metric reflects guard-eval
        # cost rather than the trivial per-layer compute.
        self._guard_manager.check(self._f_locals)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, runs
    # the static and then dynamic benchmark with plain instruction counting and appends results.
    result_path = sys.argv[1]
    for dynamic in (False, True):
        Benchmark(dynamic).enable_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
