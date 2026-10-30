# [MODULE DOC] Operator microbenchmark for concatenation of two quantized tensors through a
# QFunctional module. Varies the dtype and whether none, one or both operands are non-contiguous.
# Short and long config sets; tests are registered at import time. Running as a script starts the
# op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq


"""Microbenchmarks for quantized Cat operator"""

# Configs for PT Cat operator
qcat_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "L", "dim"],
    attrs=[
        [256, 512, 1, 2, 0],
        [512, 512, 2, 1, 1],
    ],
    cross_product_configs={
        "contig": ("all", "one", "none"),
        "dtype": (torch.quint8, torch.qint8, torch.qint32),
    },
    tags=["short"],
)

qcat_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    L=[5, 7],
    dim=[0, 1, 2],
    contig=["all", "one", "none"],
    dtype=[torch.quint8],
    tags=["long"],
)


class QCatBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for QFunctional cat on a pair of quantized MxNxK tensors. State: the
    # QFunctional module with scale and zero point set, and the tensor pair plus dim in the
    # inputs mapping.
    def init(self, M, N, K, L, dim, contig, dtype):
        # [FUNCTION DOC] Create a random MxNxK float tensor, quantize it per tensor with scale 1.0
        # and zero point 0 in the dtype, and configure a QFunctional with the same scale and zero
        # point. Builds a non-contiguous copy of the same shape by reversing dimensions, making
        # contiguous and reversing back. The pair is both contiguous for contig all, one of each
        # for one, and both non-contiguous for none. Stores the pair and dim in the inputs mapping
        # and sets the module name to qcat. Raises AssertionError for any other contig value.
        # KNOWN ISSUE: the config attribute L is accepted but never used.
        f_input = (torch.rand(M, N, K) - 0.5) * 256
        self.qf = nnq.QFunctional()
        scale = 1.0
        zero_point = 0
        self.qf.scale = scale
        self.qf.zero_point = zero_point

        if contig not in ("none", "one", "all"):
            raise AssertionError(
                f"contig must be 'none', 'one', or 'all', but got '{contig}'"
            )
        q_input = torch.quantize_per_tensor(f_input, scale, zero_point, dtype)
        permute_dims = tuple(range(q_input.ndim - 1, -1, -1))
        q_input_non_contig = q_input.permute(permute_dims).contiguous()
        q_input_non_contig = q_input_non_contig.permute(permute_dims)
        if contig == "all":
            self.input = (q_input, q_input)
        elif contig == "one":
            self.input = (q_input, q_input_non_contig)
        elif contig == "none":
            self.input = (q_input_non_contig, q_input_non_contig)

        self.inputs = {"input": self.input, "dim": dim}
        self.set_module_name("qcat")

    def forward(self, input: list[torch.Tensor], dim: int):
        # [FUNCTION DOC] Concatenate the pair of quantized tensors along dim with the QFunctional
        # module and return the quantized result.
        return self.qf.cat(input, dim=dim)


op_bench.generate_pt_test(qcat_configs_short + qcat_configs_long, QCatBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
