# [MODULE DOC] Operator microbenchmarks for normalization modules on CUDA: LayerNorm and RMSNorm,
# BatchNorm1d/2d/3d (training and eval modes) and GroupNorm. Each has a cross-product config set
# tagged long; forward and gradient tests are registered at import time. Running as a script starts
# the op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""Microbenchmarks for normalization operators."""


# ==============================================================================
# LayerNorm and RMSNorm Benchmarks
# ==============================================================================

layernorm_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["LayerNorm", nn.LayerNorm],
        ["RMSNorm", nn.RMSNorm],
    ],
)

layernorm_configs = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    K=[64, 128, 512],
    device=["cuda"],
    tags=["long"],
)


class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for LayerNorm or RMSNorm over the last dimension of a BxMxK tensor.
    # State: the instantiated module in op_func and the input in the inputs mapping.
    def init(self, op_func, device, B, M, K):
        # [FUNCTION DOC] Create a random BxMxK input (requires_grad from auto_set) and instantiate
        # the given norm class with K as normalized size on the device. The module name is the
        # class name.
        self.inputs = {
            "input": torch.rand(B, M, K, device=device, requires_grad=self.auto_set())
        }
        # normalized_shape is the last dimension (hidden dim K)
        self.op_func = op_func(K, device=device)
        self.set_module_name(op_func.__name__)

    def forward(self, input):
        # [FUNCTION DOC] Apply the norm module to input and return the output.
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(
    layernorm_list,
    layernorm_configs,
    LayerNormBenchmark,
)

op_bench.generate_pt_gradient_tests_from_op_list(
    layernorm_list,
    layernorm_configs,
    LayerNormBenchmark,
)


# ==============================================================================
# BatchNorm1d Benchmarks (training + eval)
# ==============================================================================

batchnorm1d_configs = op_bench.cross_product_configs(
    B=[8, 32],
    C=[64, 128, 256],
    M=[256, 1024],
    device=["cuda"],
    training=[True, False],
    tags=["long"],
)


class BatchNorm1dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a BatchNorm1d module over a BxCxM input, in training or eval mode.
    def init(self, device, B, C, M, training):
        # [FUNCTION DOC] Create a random BxCxM input (requires_grad from auto_set) and a
        # BatchNorm1d module with C features on the device, switched to training or eval mode
        # according to the training flag. Sets the module name to BatchNorm1d.
        self.inputs = {
            "input": torch.rand(B, C, M, device=device, requires_grad=self.auto_set())
        }
        self.op_func = nn.BatchNorm1d(C, device=device)
        self.op_func.train(training)
        self.set_module_name("BatchNorm1d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the batch norm module to input and return the output.
        return self.op_func(input)


op_bench.generate_pt_test(
    batchnorm1d_configs,
    BatchNorm1dBenchmark,
)

op_bench.generate_pt_gradient_test(
    batchnorm1d_configs,
    BatchNorm1dBenchmark,
)


# ==============================================================================
# BatchNorm2d Benchmarks (training + eval)
# ==============================================================================

batchnorm2d_configs = op_bench.cross_product_configs(
    B=[8, 32],
    C=[64, 128, 256],
    H=[28, 56],
    W=[28, 56],
    device=["cuda"],
    training=[True, False],
    tags=["long"],
)


class BatchNorm2dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a BatchNorm2d module over a BxCxHxW input, in training or
    # eval mode.
    def init(self, device, B, C, H, W, training):
        # [FUNCTION DOC] Create a random BxCxHxW input (requires_grad from auto_set) and a
        # BatchNorm2d module with C features on the device, switched to training or eval mode
        # according to the training flag. Sets the module name to BatchNorm2d.
        self.inputs = {
            "input": torch.rand(
                B, C, H, W, device=device, requires_grad=self.auto_set()
            )
        }
        self.op_func = nn.BatchNorm2d(C, device=device)
        self.op_func.train(training)
        self.set_module_name("BatchNorm2d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the batch norm module to input and return the output.
        return self.op_func(input)


op_bench.generate_pt_test(
    batchnorm2d_configs,
    BatchNorm2dBenchmark,
)

op_bench.generate_pt_gradient_test(
    batchnorm2d_configs,
    BatchNorm2dBenchmark,
)


# ==============================================================================
# BatchNorm3d Benchmarks (training + eval)
# ==============================================================================

batchnorm3d_configs = op_bench.cross_product_configs(
    B=[8, 32],
    C=[64, 128, 256],
    D=[4, 8],
    H=[14, 28],
    W=[14, 28],
    device=["cuda"],
    training=[True, False],
    tags=["long"],
)


class BatchNorm3dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a BatchNorm3d module over a BxCxDxHxW input, in training or
    # eval mode.
    def init(self, device, B, C, D, H, W, training):
        # [FUNCTION DOC] Create a random BxCxDxHxW input (requires_grad from auto_set) and a
        # BatchNorm3d module with C features on the device, switched to training or eval mode
        # according to the training flag. Sets the module name to BatchNorm3d.
        self.inputs = {
            "input": torch.rand(
                B, C, D, H, W, device=device, requires_grad=self.auto_set()
            )
        }
        self.op_func = nn.BatchNorm3d(C, device=device)
        self.op_func.train(training)
        self.set_module_name("BatchNorm3d")

    def forward(self, input):
        # [FUNCTION DOC] Apply the batch norm module to input and return the output.
        return self.op_func(input)


op_bench.generate_pt_test(
    batchnorm3d_configs,
    BatchNorm3dBenchmark,
)

op_bench.generate_pt_gradient_test(
    batchnorm3d_configs,
    BatchNorm3dBenchmark,
)


# ==============================================================================
# GroupNorm Benchmarks
# ==============================================================================

groupnorm_configs = op_bench.cross_product_configs(
    B=[8, 32],
    C=[64, 128, 256],
    H=[28, 56],
    W=[28, 56],
    num_groups=[8, 16, 32],
    device=["cuda"],
    tags=["long"],
)


class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a GroupNorm module over a BxCxHxW input.
    def init(self, device, B, C, H, W, num_groups):
        # [FUNCTION DOC] Create a random BxCxHxW input (requires_grad from auto_set) and a
        # GroupNorm module with num_groups groups over C channels on the device. Sets the module
        # name to GroupNorm.
        self.inputs = {
            "input": torch.rand(
                B, C, H, W, device=device, requires_grad=self.auto_set()
            )
        }
        self.op_func = nn.GroupNorm(num_groups, C, device=device)
        self.set_module_name("GroupNorm")

    def forward(self, input):
        # [FUNCTION DOC] Apply the group norm module to input and return the output.
        return self.op_func(input)


op_bench.generate_pt_test(
    groupnorm_configs,
    GroupNormBenchmark,
)

op_bench.generate_pt_gradient_test(
    groupnorm_configs,
    GroupNormBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
