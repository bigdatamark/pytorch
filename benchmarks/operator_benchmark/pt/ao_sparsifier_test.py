# [MODULE DOC] Operator benchmark for the weight norm sparsifier step from the ao
# pruning package. Defines a short config list (explicit mask shape, sparsity level,
# block shape and zeros per block rows) and a long cross product of those parameters,
# plus the benchmark class. Import-time side effect: registers forward tests for both
# lists. Runs the benchmark runner when executed as a script.
# KNOWN ISSUE: the module name used here (weight_norm_sparsifier_step) is listed in the core
# SKIP_OP_LISTS, so the runner filters these tests out and they never actually run.
import operator_benchmark as op_bench

import torch
from torch import nn
from torch.ao import pruning


"""Microbenchmarks for sparsifier."""

sparse_configs_short = op_bench.config_list(
    attr_names=["M", "SL", "SBS", "ZPB"],
    attrs=[
        [(32, 16), 0.3, (4, 1), 2],
        [(32, 16), 0.6, (1, 4), 4],
        [(17, 23), 0.9, (1, 1), 1],
    ],
    tags=("short",),
)

sparse_configs_long = op_bench.cross_product_configs(
    M=((128, 128), (255, 324)),  # Mask shape
    SL=(0.0, 1.0, 0.3, 0.6, 0.9, 0.99),  # Sparsity level
    SBS=((1, 4), (1, 8), (4, 1), (8, 1)),  # Sparse block shape
    ZPB=(0, 1, 2, 3, 4, None),  # Zeros per block
    tags=("long",),
)


class WeightNormSparsifierBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark that times one step of a weight norm sparsifier applied to a module
    # with a single all-ones weight buffer. It has no tensor inputs (the inputs mapping is empty).
    def init(self, M, SL, SBS, ZPB):
        # [FUNCTION DOC] Creates a bare module holding a buffer named weight filled with ones of
        # the given mask shape, builds a weight norm sparsifier from the sparsity level, block
        # shape and zeros per block, prepares it on the module for that buffer, sets an empty
        # inputs mapping and the module name weight_norm_sparsifier_step. Mutates only local
        # objects.
        weight = torch.ones(M)
        model = nn.Module()
        model.register_buffer("weight", weight)

        sparse_config = [{"tensor_fqn": "weight"}]
        self.sparsifier = pruning.WeightNormSparsifier(
            sparsity_level=SL,
            sparse_block_shape=SBS,
            zeros_per_block=ZPB,
        )
        self.sparsifier.prepare(model, config=sparse_config)
        self.inputs = {}  # All benchmarks need inputs :)
        self.set_module_name("weight_norm_sparsifier_step")

    def forward(self):
        # [FUNCTION DOC] Runs one sparsifier step, which recomputes
        # the mask of the prepared weight. Returns nothing.
        self.sparsifier.step()


all_tests = sparse_configs_short + sparse_configs_long
op_bench.generate_pt_test(all_tests, WeightNormSparsifierBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
