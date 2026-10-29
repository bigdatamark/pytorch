# [MODULE DOC]
# Helpers shared by AutoHeuristic data-generation scripts: random transpose flags, memory
# fit check, matmul input tensor creation, float32 precision selection and random sizes
# between powers of two. No import-time side effects.
import random
from typing import Any

import torch


def transpose_tensors(p_transpose_both: float = 0.05) -> tuple[bool, bool]:
    # [FUNCTION DOC] Randomly picks (transpose_left, transpose_right) flags. With probability
    # p_transpose_both returns both true; otherwise picks uniformly among left only, right
    # only and neither. Uses python random; no other side effects.
    transpose_both = random.choices(
        [True, False], [p_transpose_both, 1 - p_transpose_both]
    )[0]
    if transpose_both:
        return (True, True)
    transpose_left = (True, False)
    transpose_right = (False, True)
    no_transpose = (False, False)
    return random.choices([transpose_left, transpose_right, no_transpose])[0]


def fits_in_memory(dtype: Any, m: int, k: int, n: int) -> Any:
    # [FUNCTION DOC] Returns whether the three matmul operands (m by k, k by n, m by n) of the
    # given dtype take less than a quarter of the total memory of CUDA device 0. Queries the
    # device properties; requires CUDA.
    threshold_memory = torch.cuda.get_device_properties(0).total_memory / 4
    # dividing by 4 because we otherwise sometimes run out of memory, I assume because
    # inductor creates copies of tensors for benchmarking?
    return dtype.itemsize * (m * k + k * n + m * n) < threshold_memory


def get_mm_tensors(
    m: int,
    k: int,
    n: int,
    transpose_left: bool,
    transpose_right: bool,
    dtype_left: Any,
    dtype_right: Any,
) -> tuple[Any, Any]:
    # [FUNCTION DOC] Creates random normal matmul operands: left of logical shape m by k and
    # right of shape k by n, with the given dtypes. A transposed operand is allocated with
    # swapped dimensions and returned as a transposed (non-contiguous) view. Allocates on the
    # current default device.
    if transpose_left:
        a = torch.randn(k, m, dtype=dtype_left).t()
    else:
        a = torch.randn(m, k, dtype=dtype_left)

    if transpose_right:
        b = torch.randn(n, k, dtype=dtype_right).t()
    else:
        b = torch.randn(k, n, dtype=dtype_right)
    return (a, b)


def set_precision(dtype: Any, p_float32_prec_highest: float = 0.8) -> None:
    # [FUNCTION DOC] Sets the global float32 matmul precision. For float32 it randomly picks
    # "highest" with probability p_float32_prec_highest, else "high"; for other dtypes it
    # always uses "high". Side effect: mutates global torch state.
    if dtype == torch.float32:
        precisions = ["high", "highest"]
        weights = [1 - p_float32_prec_highest, p_float32_prec_highest]
        precision = random.choices(precisions, weights)[0]
    else:
        precision = "high"
    torch.set_float32_matmul_precision(precision)


def get_random_between_pow2(min_power2: int, max_power2: int) -> int:
    # [FUNCTION DOC] Picks a random exponent i in [min_power2, max_power2 - 1] and returns a
    # random integer strictly between 2^i and 2^(i+1). Raises AssertionError if the range is
    # empty (cannot happen for the exponents produced here).
    i = random.randint(min_power2, max_power2 - 1)
    lower = 2**i + 1
    upper = 2 ** (i + 1) - 1
    if lower > upper:
        raise AssertionError("lower must not be greater than upper")
    return random.randint(lower, upper)
