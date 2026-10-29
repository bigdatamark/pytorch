# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]
# [MODULE DOC] Python wrapper around the TorchScript static runtime module
# (torch._C._jit_to_static_module). Defines StaticModule. No import-time side effects.

import torch


class StaticModule:
    # [CLASS DOC] Thin wrapper holding a static runtime module (attribute static_module)
    # built from a scripted module or graph; forwards calls, async runs and benchmarks.
    def __init__(self, scripted):
        # [FUNCTION DOC] Builds static_module from scripted: if it has a _c attribute (a
        # scripted module) its C++ module is converted, otherwise its graph attribute is.
        # Raises whatever the C++ conversion raises on unsupported input.
        # this is an nn.Module
        if hasattr(scripted, "_c"):
            self.static_module = torch._C._jit_to_static_module(scripted._c)
        else:
            self.static_module = torch._C._jit_to_static_module(scripted.graph)

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the static module on the given arguments and returns its output.
        return self.static_module(*args, **kwargs)

    def benchmark(self, args, kwargs, warmup_runs, main_runs):
        # [FUNCTION DOC] Runs the static module's benchmark with the given args list, kwargs
        # mapping and warmup/main run counts. Returns nothing (result of the call is dropped).
        self.static_module.benchmark(args, kwargs, warmup_runs, main_runs)

    def runAsync(self, args, kwargs):
        # [FUNCTION DOC] Starts an asynchronous run with the args list and kwargs mapping and
        # returns the resulting future from the underlying module.
        return self.static_module.runAsync(args, kwargs)

    def benchmark_individual_ops(self, args, kwargs, warmup_runs, main_runs):
        # [FUNCTION DOC] Benchmarks each operator separately with the given inputs and run
        # counts; returns the underlying module's per-operator timing results.
        return self.static_module.benchmark_individual_ops(
            args, kwargs, warmup_runs, main_runs
        )
