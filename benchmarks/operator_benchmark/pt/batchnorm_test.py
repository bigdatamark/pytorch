# [MODULE DOC] Operator benchmarks for batch norm on three dimensional inputs (batch,
# channels, length) and on two dimensional inputs, each in training and inference mode on cpu
# and cuda, optionally with cudnn enabled. Defines a config post-processor that appends a
# cudnn flag to every config (an extra cudnn-enabled variant only for cuda configs), short and
# long config lists, and two benchmark classes. Registers forward and backward tests at import
# time and runs the benchmark runner when executed as a script.
import operator_benchmark as op_bench

import torch
import torch.nn.functional as F


"""Microbenchmarks for batchnorm operator."""

# Benchmark cudnn if available
if torch.backends.cudnn.is_available:

    def cudnn_benchmark_configs(configs):
        # [FUNCTION DOC] HELPER: Used when the cudnn availability check is truthy. Returns a
        # new list of configs where each config (a sequence of single-entry mappings) is
        # followed by a cudnn flag entry. Configs that target cuda are listed twice, once with
        # cudnn enabled and once disabled; others only with cudnn disabled.
        # KNOWN ISSUE: the availability check tests the function object rather than calling it, so
        # it is always true and the other branch with the cudnn-less version is never used.
        result = []
        for config in configs:
            is_cuda = any("cuda" in attr.values() for attr in config)
            if is_cuda:
                result.append((*config, dict(cudnn=True)))
            result.append((*config, dict(cudnn=False)))
        return result

else:

    def cudnn_benchmark_configs(configs):
        # [FUNCTION DOC] HELPER: Fallback used when cudnn is unavailable. Returns a
        # list with every config extended by a cudnn-disabled flag entry.
        return [(*config, dict(cudnn=False)) for config in configs]


batchnorm_configs_short = cudnn_benchmark_configs(
    op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=[
            [1, 256, 3136],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],
            "training": [True, False],
        },
        tags=["short"],
    )
)

batchnorm_configs_long = cudnn_benchmark_configs(
    op_bench.cross_product_configs(
        M=[2, 128],
        N=[8192, 2048],
        K=[1],
        device=["cpu", "cuda"],
        training=[True, False],
        tags=["long"],
    )
)


class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for functional batch norm on a three dimensional
    # random input with per-channel running mean, variance, weight and bias, plus
    # training and cudnn flags kept in the inputs mapping. Only the data input
    # gets requires_grad from auto_set. Module name is batchnorm.
    def init(self, M, N, K, device, training, cudnn):
        # [FUNCTION DOC] Fills the inputs mapping with a random input of batch, channel
        # and length sizes and four random per-channel vectors on the device, plus the
        # training and cudnn flags, and sets the module name to batchnorm.
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "mean": torch.rand(N, device=device),
            "var": torch.rand(N, device=device),
            "weight": torch.rand(N, device=device),
            "bias": torch.rand(N, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        self.set_module_name("batchnorm")

    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        # [FUNCTION DOC] Runs functional batch norm on the input with the given
        # statistics, scale and shift in the given training mode, with cudnn temporarily
        # enabled or disabled according to the flag. Returns the normalized tensor.
        with torch.backends.cudnn.flags(enabled=cudnn):
            return F.batch_norm(input_one, mean, var, weight, bias, training)


op_bench.generate_pt_test(
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)
op_bench.generate_pt_gradient_test(
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)


batchnorm1d_configs_short = cudnn_benchmark_configs(
    op_bench.config_list(
        attr_names=["N", "C"],
        attrs=[
            [3136, 256],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],
            "training": [True, False],
        },
        tags=["short"],
    )
)

batchnorm1d_configs_long = cudnn_benchmark_configs(
    op_bench.cross_product_configs(
        N=[2, 128],
        C=[8192, 2048],
        device=["cpu", "cuda"],
        training=[True, False],
        tags=["long"],
    )
)


class BatchNorm1dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Same as the previous batch norm benchmark but for a two
    # dimensional input of batch by channels. Module name is also batchnorm.
    def init(self, N, C, device, training, cudnn):
        # [FUNCTION DOC] Fills the inputs mapping with a random matrix of the
        # given sizes, four random per-channel vectors, and the training and
        # cudnn flags; sets the module name to batchnorm.
        self.inputs = {
            "input_one": torch.rand(N, C, device=device, requires_grad=self.auto_set()),
            "mean": torch.rand(C, device=device),
            "var": torch.rand(C, device=device),
            "weight": torch.rand(C, device=device),
            "bias": torch.rand(C, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        self.set_module_name("batchnorm")

    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        # [FUNCTION DOC] Runs functional batch norm in the given training mode with
        # cudnn enabled or disabled per the flag and returns the result.
        with torch.backends.cudnn.flags(enabled=cudnn):
            return F.batch_norm(input_one, mean, var, weight, bias, training)


op_bench.generate_pt_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)
op_bench.generate_pt_gradient_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
