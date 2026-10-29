# mypy: ignore-errors
# [MODULE DOC]
# Data collection script for the mixed_mm AutoHeuristic: defines a benchmark runner that
# compiles a mixed-dtype matmul with max-autotune on random shapes and dtypes so results get
# logged. Appends the parent directory to sys.path at import time; running it as a script
# executes the runner (needs CUDA).
import random
import sys
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).absolute().parents[1]))

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    get_random_between_pow2,
)

import torch
from torch._inductor.utils import fresh_cache


class BenchmarkRunnerMixedMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC] Runner named "mixed_mm" producing inputs for mixed matmul (float16 or
    # bfloat16 left operand, int8 or uint8 right operand). Stateless beyond the base class.
    # Key methods: create_input (random shapes/dtypes), run_benchmark (compile and run).
    """
    BenchmarkRunner for mixed mm. Used to generate collect training data with AutoHeuristic to learn a heuristic.
    Currently, we are generating inputs with the following restrictions:
    - m <= 128, and n and k >= 1024 (for these inputs one of the triton kernels wins in most cases)
    - k % 256 == 0 (if k is not a multiple of the block size, this can have a huge negative impact on performance)
    - mat1 not transposed
    - mat2 transposed
    This allows us to learn a heuristic that works well e.g. for gpt-fast.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base runner with the name "mixed_mm".
        super().__init__("mixed_mm")

    def create_input(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Draws dtypes and sizes and returns (m, k, n, transpose_left,
        # transpose_right, dtype_left, dtype_right); left is never transposed, right always is.
        dtype1, dtype2 = self.get_dtypes()
        m, k, n = self.get_m_k_n(dtype1)
        transpose_left, transpose_right = False, True
        return (m, k, n, transpose_left, transpose_right, dtype1, dtype2)

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        transpose_left: bool,
        transpose_right: bool,
        dtype_left: Any,
        dtype_right: Any,
    ) -> Any:
        # [FUNCTION DOC] Builds the operands (right operand generated as float32, then cast to
        # dtype_right), then inside a fresh inductor cache compiles a function that casts the
        # right operand to the left dtype and multiplies, using max-autotune without cudagraphs,
        # runs it once and resets the compiler. Autotuning results get logged by AutoHeuristic.
        a, b = get_mm_tensors(
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype_left=dtype_left,
            dtype_right=torch.float32,
        )
        b = b.to(dtype=dtype_right)

        with fresh_cache():

            def mixed_mm(A, B):
                # [FUNCTION DOC] HELPER: Casts B to the dtype of A and returns the matmul.
                return torch.mm(A, B.to(A.dtype))

            cf = torch.compile(mixed_mm, mode="max-autotune-no-cudagraphs")
            cf(a, b)
            torch.compiler.reset()

    def random_multiple_of_128(self, min_num=7, max_num=17):
        # [FUNCTION DOC] HELPER: Picks a random exponent in [min_num, max_num - 1] and returns a
        # random multiple of 128 between 2^exponent and 2^(exponent+1) (integer division).
        ran_pow2 = random.randint(min_num, max_num - 1)
        start = (2**ran_pow2) // 128
        end = (2 ** (ran_pow2 + 1)) // 128
        random_multiple = random.randint(start, end)
        return random_multiple * 128

    def get_random_pow2(self, min_power2: int, max_power2: int):
        # [FUNCTION DOC] HELPER: Returns 2 raised to a random exponent in the inclusive range.
        return 2 ** random.randint(min_power2, max_power2)

    def get_distr_type(self) -> str:
        # [FUNCTION DOC] HELPER: Randomly chooses the distribution name used for a dimension:
        # mult_128, pow2, uniform or uniform-between-pow2 (weights 0.85, 0.1, 0.04, 0.01).
        # 85%: choose a random multiple of 128 between 2^10 and 2^17
        # 10%: choose a random power of 2 between 2^10 and 2^17 favoring larger values
        #  4%: choose a random number between 1024 and 131072
        #  1%: choose a random number between 2^i and 2^(i+1) with i in [10, 16]
        return random.choices(
            ["mult_128", "pow2", "uniform", "uniform-between-pow2"],
            [0.85, 0.1, 0.04, 0.01],
        )[0]

    def get_random_dim(self):
        # [FUNCTION DOC] HELPER: Returns a random large dimension (1024 to 131072) drawn from a
        # randomly chosen distribution. Exits the process with status 1 on an unknown type.
        distr_type = self.get_distr_type()
        if distr_type == "mult_128":
            return self.random_multiple_of_128(min_num=10, max_num=17)
        if distr_type == "pow2":
            return self.get_random_pow2(min_power2=10, max_power2=17)
        elif distr_type == "uniform-between-pow2":
            return get_random_between_pow2(min_power2=10, max_power2=17)
        elif distr_type == "uniform":
            return random.randint(1024, 131072)
        print(f"random_type {distr_type} not supported")
        sys.exit(1)

    def get_random_num_small(self) -> int:
        # [FUNCTION DOC] HELPER: Returns a small value for m: with probability 0.75 a power of
        # two between 2 and 128, otherwise a random number between two consecutive powers of two
        # in that range.
        pow2 = random.choices([True, False], [0.75, 0.25])[0]
        if pow2:
            return 2 ** random.randint(1, 7)
        else:
            return get_random_between_pow2(1, 7)

    def get_m_k_n(self, dtype: Any) -> tuple[int, int, int]:
        # [FUNCTION DOC] Repeatedly samples (m, k, n) until k is a multiple of 256, no operand
        # has 2^31 or more elements, and the operands fit in GPU memory for the given dtype;
        # returns the first valid triple. Raises AssertionError if k or n were below 1024.
        numel_max = 2**31

        # repeat until tensors fit in memory
        while True:
            m = self.get_random_num_small()
            k = self.get_random_dim()
            n = self.get_random_dim()
            if k % 256 != 0:
                continue

            if not (k >= 1024 and n >= 1024):
                raise AssertionError("k and n must be at least 1024")

            if m * k >= numel_max or m * n >= numel_max or k * n >= numel_max:
                # autotuning will not happen for tensors that are this large
                continue

            if fits_in_memory(dtype, m, k, n):
                return (m, k, n)

    def get_dtypes(self) -> Any:
        # [FUNCTION DOC] Randomly picks (left, right) dtypes: left from float16/bfloat16, right
        # from int8/uint8, rejecting the bfloat16 with uint8 combination.
        while True:
            dtype_floats = [torch.float16, torch.bfloat16]
            dtype_ints = [torch.int8, torch.uint8]
            mat1_dtype = random.choices(dtype_floats)[0]
            mat2_dtype = random.choices(dtype_ints)[0]
            if mat1_dtype == torch.bfloat16 and mat2_dtype == torch.uint8:
                # this combination seems to cause issues with mixed_mm
                continue
            return (mat1_dtype, mat2_dtype)


if __name__ == "__main__":
    runner = BenchmarkRunnerMixedMM()
    runner.run()
