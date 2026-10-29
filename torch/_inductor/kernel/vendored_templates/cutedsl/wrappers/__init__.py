# [PACKAGE DOC] Wrappers exposing vendored CuTeDSL templates as Cutlass API operators; importing
# this package imports dense_blockscaled_gemm_kernel, which registers its kernels as a side effect.
# CuTeDSL Cutlass API registrations for PyTorch Inductor.
from . import dense_blockscaled_gemm_kernel
