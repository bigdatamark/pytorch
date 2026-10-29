# [MODULE DOC] Thin entry point for a remote GEMM autotune cache. Exposes gen_best_config,
# which delegates to an fbcode-internal implementation and is unimplemented in OSS.
# No import-time side effects beyond imports.
import asyncio
from typing import TypeVar

import torch._inductor.config as config
from torch._inductor import ir


_T = TypeVar("_T")


def gen_best_config(mat1: ir.StorageBox, mat2: ir.StorageBox) -> asyncio.Task[_T]:
    # [FUNCTION DOC] Returns an asyncio task producing the best GEMM autotune config for the two
    # input matrices (mat1, mat2 as StorageBox). In fbcode, forwards to the internal implementation.
    # Raises NotImplementedError outside fbcode.
    """
    Generate the best GEMM autotune config for the given matrices.
    """
    if config.is_fbcode():
        from torch._inductor.fb.remote_gemm_autotune_cache import gen_best_config

        return gen_best_config(mat1, mat2)
    else:
        raise NotImplementedError("Function gen_best_config is not yet implemented")
