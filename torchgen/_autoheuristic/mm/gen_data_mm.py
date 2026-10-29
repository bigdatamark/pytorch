# [MODULE DOC]
# Data collection script for the mm AutoHeuristic: defines a benchmark runner that compiles
# plain matmuls with max-autotune on random shapes and dtypes so results get logged. Appends
# the parent directory to sys.path at import time; running it as a script executes the runner
# (needs CUDA).
import itertools
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
    set_precision,
)

import torch
from torch._inductor.utils import fresh_cache


class BenchmarkRunnerMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC] Runner named "mm" that benchmarks torch.mm over random shapes, a random
    # dtype (float32, float16 or bfloat16) and all four transpose combinations. Stateless
    # beyond the base class.
    """
    BenchmarkRunner for mm.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base runner with the name "mm".
        super().__init__("mm")

    def create_input(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Picks a random dtype, sets the global float32 matmul precision for
        # it, samples a shape that fits in memory and returns (m, k, n, dtype).
        dtype = random.choices([torch.float32, torch.float16, torch.bfloat16])[0]
        set_precision(dtype)
        m, k, n = self.get_m_k_n(dtype)
        return (m, k, n, dtype)

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        dtype: Any,
    ) -> Any:
        # [FUNCTION DOC] For each of the four (transpose_left, transpose_right) combinations,
        # prints the configuration, builds operands, and inside a fresh inductor cache compiles
        # a matmul with max-autotune without cudagraphs, runs it once and resets the compiler.
        # for a given shape, test all possible combinations of transpose_left and transpose_right
        for transpose_left, transpose_right in itertools.product(
            [False, True], repeat=2
        ):
            print(
                f"m: {m}, k: {k}, n: {n}, transpose_left: {transpose_left}, transpose_right: {transpose_right}, dtype: {dtype}"
            )
            a, b = get_mm_tensors(
                m,
                k,
                n,
                transpose_left,
                transpose_right,
                dtype_left=dtype,
                dtype_right=dtype,
            )

            with fresh_cache():

                def mixed_mm(A: Any, B: Any) -> Any:
                    # [FUNCTION DOC] HELPER: Plain matmul of A and B (name is a leftover from
                    # the mixed_mm script).
                    return torch.mm(A, B)

                cf = torch.compile(mixed_mm, mode="max-autotune-no-cudagraphs")
                cf(a, b)
                torch.compiler.reset()

    def random_multiple_of_128(self, min_num: int = 7, max_num: int = 17) -> int:
        # [FUNCTION DOC] HELPER: Returns a random multiple of 128 lying between 2^e and
        # 2^(e+1), where e is a random exponent in [min_num, max_num - 1].
        # generates a random number ran_pow2 between min_num and max_num -1
        # and returns a random multiple of 128 between 2^ran_pow2 and 2^(ran_pow2+1)
        ran_pow2 = random.randint(min_num, max_num - 1)
        start = (2**ran_pow2) // 128
        end = (2 ** (ran_pow2 + 1)) // 128
        random_multiple = random.randint(start, end)
        return random_multiple * 128

    def get_distr_type(self) -> str:
        # [FUNCTION DOC] HELPER: Randomly chooses the dimension distribution name: mult_128,
        # pow2, uniform or uniform-between-pow2 (weights 0.85, 0.1, 0.04, 0.01).
        # 85%: choose a random multiple of 128 between 2^10 and 2^17
        # 10%: choose a random power of 2 between 2^0 and 2^17
        #  4%: choose a random number between 1 and 131072
        #  1%: choose a random number between 2^i and 2^(i+1) with i in [1, 16]
        return random.choices(
            ["mult_128", "pow2", "uniform", "uniform-between-pow2"],
            [0.85, 0.1, 0.04, 0.01],
        )[0]

    def get_random_dim(self) -> int:
        # [FUNCTION DOC] HELPER: Returns a random dimension (1 to 131072) from a randomly chosen
        # distribution. Prints a message and exits the process with status 1 on an unknown type.
        distr_type = self.get_distr_type()
        if distr_type == "mult_128":
            return self.random_multiple_of_128(min_num=10, max_num=17)
        if distr_type == "pow2":
            return int(2 ** random.randint(0, 17))
        elif distr_type == "uniform-between-pow2":
            # TODO(AlnisM): make mypy work for torchgen/_autoheuristic/
            return int(get_random_between_pow2(min_power2=1, max_power2=17))
        elif distr_type == "uniform":
            return random.randint(1, 131072)
        print(f"random_type {distr_type} not supported")
        sys.exit(1)

    def get_m_k_n(self, dtype: Any) -> tuple[int, int, int]:
        # [FUNCTION DOC] Repeatedly samples random (m, k, n) until no operand has 2^31 or more
        # elements and the operands fit in GPU memory for dtype; returns that triple.
        numel_max = 2**31

        # repeat until tensors fit in memory
        while True:
            m = self.get_random_dim()
            k = self.get_random_dim()
            n = self.get_random_dim()

            if m * k >= numel_max or m * n >= numel_max or k * n >= numel_max:
                # autotuning will not happen for tensors that are this large
                continue

            if fits_in_memory(dtype, m, k, n):
                return (m, k, n)


if __name__ == "__main__":
    runner = BenchmarkRunnerMM()
    runner.run()
