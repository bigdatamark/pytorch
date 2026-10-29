# [MODULE DOC] Tiny helper module for building symbolic fake tensors used when compiling CuTe DSL kernels ahead of time.
# Imports cutlass.cute; no other import-time side effects.
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional

import cutlass.cute as cute


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    # [FUNCTION DOC] Build a fake cute tensor with the given dtype and shape for compilation.
    # Returns None when dtype is None. A negative leading_dim counts from the end. All strides are symbolic
    # (with the given divisibility) except the leading dimension, whose stride is 1. Alignment assumed is
    # divisibility times dtype width in bytes.
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )
