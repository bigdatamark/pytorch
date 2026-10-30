# [MODULE DOC] Operator microbenchmark for torch.mm. Short configs run on CPU in float; long
# configs run on CUDA with half, bfloat16 and float32. Forward tests for all configs and gradient
# tests for long configs are registered at import time. Includes a memory traffic estimate. Running
# as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.mm."""

# Benchmark ops performance without broadcast
ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["mm", torch.mm]],
)

mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)

mm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)


class MmOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a two-operand matrix multiply op on random MxN and NxK matrices.
    def init(self, M, N, K, device, dtype, op_func):
        # [FUNCTION DOC] Create standard-normal tensors of shape MxN and NxK with the dtype on the
        # device (requires_grad from auto_set), store them in the inputs mapping and keep op_func.
        self.inputs = {
            "input_one": torch.randn(
                M, N, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "input_two": torch.randn(
                N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Apply the matrix multiply op to the two inputs and return the result.
        return self.op_func(input_one, input_two)

    def get_memory_traffic_bytes(self):
        # [FUNCTION DOC] Estimate bytes moved by one call as elements read from both inputs plus
        # elements written to the MxK output, times the element size of input_one. Raises
        # AssertionError if the inner dimensions of the two inputs differ.
        """Override for matmul: (M, N) @ (N, K) -> (M, K)
        Memory traffic: read(M*N + N*K) + write(M*K)
        """
        input_one = self.inputs["input_one"]
        input_two = self.inputs["input_two"]
        M, N = input_one.shape
        N_check, K = input_two.shape
        if N != N_check:
            raise AssertionError(
                f"Matrix dimensions must match for matmul: N={N}, N_check={N_check}"
            )

        bytes_per_element = input_one.element_size()
        total_elements = M * N + N * K + M * K
        return total_elements * bytes_per_element


op_bench.generate_pt_tests_from_op_list(
    ops_list, mm_short_configs + mm_long_configs, MmOpBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    ops_list, mm_long_configs, MmOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
