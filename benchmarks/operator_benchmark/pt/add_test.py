# [MODULE DOC] Operator benchmarks for the add and addr operators. Defines long (cross
# product) and short (explicit rows crossed with devices) config lists for add and a cross
# product of sizes, devices and dtypes (double, half) for addr. Import-time side effect:
# registers forward and backward tests for both operators. Test names follow the pattern of
# module name, size attributes and optional backward suffixes, which can be used for
# filtering. Runs the benchmark runner when executed as a script.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
add_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)


add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for elementwise add of two random three dimensional tensors on a
    # device; both inputs get requires_grad from auto_set. Module name is add.
    def init(self, M, N, K, device):
        # [FUNCTION DOC] Fills the inputs mapping with two random tensors of the given sizes on
        # the given device, each with requires_grad from auto_set, and sets the module name to
        # add.
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "input_two": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
        }
        self.set_module_name("add")

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Returns the elementwise sum of the two inputs.
        return torch.add(input_one, input_two)


# The generated test names based on add_short_configs will be in the following pattern:
# add_M8_N16_K32_devicecpu
# add_M8_N16_K32_devicecpu_bwdall
# add_M8_N16_K32_devicecpu_bwd1
# add_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddBenchmark)

"""Mircobenchmark for addr operator."""


class AddrBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for the outer product add: a random matrix plus the
    # outer product of two random vectors, on a device with a given dtype. All
    # three inputs get requires_grad from auto_set. Module name is addr.
    def init(self, M, N, device, dtype):
        # [FUNCTION DOC] Fills the inputs mapping with a matrix of the given rows and
        # columns and vectors of lengths rows and columns, on the given device and dtype,
        # each with requires_grad from auto_set, and sets the module name to addr.
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec1": torch.rand(
                (M,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec2": torch.rand(
                (N,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addr")

    def forward(self, input_one, vec1, vec2):
        # [FUNCTION DOC] Returns the matrix plus the outer product of the two vectors.
        return torch.addr(input_one, vec1, vec2)


addr_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    dtype=[torch.double, torch.half],
    tags=["addr"],
)

op_bench.generate_pt_test(addr_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_configs, AddrBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
