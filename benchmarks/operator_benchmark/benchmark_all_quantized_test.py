# [MODULE DOC] Entry script that registers the quantized operator benchmarks by importing the
# quantized test modules from the pt directory (imports are only for their registration side
# effect). When run as a script it calls the operator_benchmark runner main function.
from pt import (  # noqa: F401
    qactivation_test,
    qarithmetic_test,
    qatembedding_ops_test,
    qbatchnorm_test,
    qcat_test,
    qcomparators_test,
    qconv_test,
    qembedding_pack_test,
    qembeddingbag_test,
    qgroupnorm_test,
    qinstancenorm_test,
    qinterpolate_test,
    qlayernorm_test,
    qlinear_test,
    qobserver_test,
    qpool_test,
    qrnn_test,
    qtensor_method_test,
    quantization_test,
    qunary_test,
)

import operator_benchmark as op_bench


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
