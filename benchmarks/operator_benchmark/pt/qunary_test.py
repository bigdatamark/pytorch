# [MODULE DOC] Microbenchmarks for unary operators on quantized tensors. One group covers
# point-wise and reduction ops (currently argsort, clone, mean, relu, in-place relu and sort;
# many others are commented out until supported for quantized tensors), and another covers
# topk. Config sets, op list and benchmarks are registered at import time. Running as a script
# starts the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized unary operators (point-wise and reduction)."""


# Configs for pointwise and reduction unary ops
qunary_ops_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        "dtype": [torch.quint8],
    },
    tags=["short"],
)

qunary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    dtype=[torch.quint8, torch.qint8, torch.qint32],
    tags=["long"],
)


class QUnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a single unary op applied to a quantized 2D tensor. The op comes
    # from the op list.
    def init(self, M, N, dtype, op_func):
        # [FUNCTION DOC] Args: M, N (shape), dtype, op_func. Quantizes a random float tensor
        # with scale 1.0 and zero point 0, stores it in self.inputs, and keeps op_func.
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {
            "q_input": torch.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            )
        }
        self.op_func = op_func

    def forward(self, q_input):
        # [FUNCTION DOC] Applies the op to the quantized input and returns its result.
        return self.op_func(q_input)


# TODO: Uncomment the ops whenever they are implemented for quantized tensor.
qunary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        # ['q_abs', torch.abs],
        # ['q_abs_', torch.abs_],
        # ['q_acos', torch.acos],
        # ['q_acos_', torch.acos_],
        ["q_argsort", torch.argsort],
        # ['q_asin', torch.asin],
        # ['q_asin_', torch.asin_],
        # ['q_atan', torch.atan],
        # ['q_atan_', torch.atan_],
        # ['q_ceil', torch.ceil],
        # ['q_ceil_', torch.ceil_],
        ["q_clone", torch.clone],
        # ['q_cos', torch.cos],
        # ['q_cos_', torch.cos_],
        # ['q_cosh', torch.cosh],
        # ['q_digamma', torch.digamma],
        # ['q_erf', torch.erf],
        # ['q_erf_', torch.erf_],
        # ['q_erfc', torch.erfc],
        # ['q_erfc_', torch.erfc_],
        # ['q_erfinv', torch.erfinv],
        # ['q_exp', torch.exp],
        # ['q_exp_', torch.exp_],
        # ['q_expm1', torch.expm1],
        # ['q_expm1_', torch.expm1_],
        # ['q_floor', torch.floor],
        # ['q_floor_', torch.floor_],
        # ['q_frac', torch.frac],
        # ['q_frac_', torch.frac_],
        # ['q_hardshrink', torch.hardshrink],
        # ['q_lgamma', torch.lgamma],
        # ['q_log', torch.log],
        # ['q_log10', torch.log10],
        # ['q_log10_', torch.log10_],
        # ['q_log1p', torch.log1p],
        # ['q_log1p_', torch.log1p_],
        # ['q_log2', torch.log2],
        # ['q_log2_', torch.log2_],
        # ['q_log_', torch.log_],
        ["q_mean", torch.mean],
        # ['q_neg', torch.neg],
        # ['q_neg_', torch.neg_],
        # ['q_reciprocal', torch.reciprocal],
        # ['q_reciprocal_', torch.reciprocal_],
        ["q_relu", torch.relu],
        ["q_relu_", torch.relu_],
        # ['q_round', torch.round],
        # ['q_round_', torch.round_],
        # ['q_rsqrt', torch.rsqrt],
        # ['q_rsqrt_', torch.rsqrt_],
        # ['q_sigmoid', torch.sigmoid],
        # ['q_sigmoid_', torch.sigmoid_],
        # ['q_sign', torch.sign],
        # ['q_sin', torch.sin],
        # ['q_sin_', torch.sin_],
        # ['q_sinh', torch.sinh],
        ["q_sort", torch.sort],
        # ['q_sqrt', torch.sqrt],
        # ['q_sqrt_', torch.sqrt_],
        # ['q_tan', torch.tan],
        # ['q_tan_', torch.tan_],
        # ['q_tanh', torch.tanh],
        # ['q_tanh_', torch.tanh_],
        # ['q_trunc', torch.trunc],
        # ['q_trunc_', torch.trunc_],
        # ['q_unique', torch.unique],
        # ['q_zero_', torch.zero_],
        # ['q_bernoulli_', lambda t: t.bernoulli_()],
        # ['q_cauchy_', lambda t: t.cauchy_()],
        # ['q_digamma_', lambda t: t.digamma_()],
        # ['q_exponential_', lambda t: t.exponential_()],
        # ['q_normal_', lambda t: t.normal_()],
        # ['q_random_', lambda t: t.random_()],
        # ['q_sign_', lambda t: t.sign_()],
        # ['q_uniform_', lambda t: t.uniform_()],
        # ['q_half', lambda t: t.half()],
        # ['q_long', lambda t: t.long()],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    qunary_ops_list,
    qunary_ops_configs_short + qunary_ops_configs_long,
    QUnaryOpBenchmark,
)


# === Other unary ops (i.e. the ones that need parameters as args) ===

# Configs for pointwise and reduction unary ops
qunary_ops_topk_configs_short = op_bench.config_list(
    attr_names=["M", "N", "k"],
    attrs=[
        [512, 512, 5],
    ],
    cross_product_configs={
        "dtype": [torch.quint8],
    },
    tags=["short"],
)

qunary_ops_topk_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    k=[1, 3, 5],
    dtype=[torch.quint8, torch.qint8, torch.qint32],
    tags=["long"],
)


class QTopkOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of topk on a quantized 2D tensor with a configurable k.
    def init(self, M, N, dtype, k):
        # [FUNCTION DOC] Args: M, N (shape), dtype, k. Quantizes a random float tensor with
        # scale 1.0 and zero point 0 and stores it with k in self.inputs. Sets the module name.
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {
            "q_input": torch.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "k": k,
        }
        self.set_module_name("qtopk")

    def forward(self, q_input, k: int):
        # [FUNCTION DOC] Returns the topk values and indices of the quantized input along the
        # last dimension.
        return torch.topk(q_input, k)


op_bench.generate_pt_test(
    qunary_ops_topk_configs_short + qunary_ops_topk_configs_long, QTopkOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
