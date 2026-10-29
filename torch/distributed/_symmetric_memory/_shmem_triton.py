# [MODULE DOC] Backend-agnostic dispatch layer for SHMEM Triton support. Exposes
# get_shmem_backend_module, which picks the rocSHMEM module on ROCm builds and the
# NVSHMEM module otherwise, and requires_shmem, a decorator that applies the matching
# backend decorator to a Triton JIT function. Imports are lazy so only the selected
# backend module is loaded; no import-time side effects.
"""
SHMEM Triton backend dispatch.

Thin abstraction layer that routes to the correct platform-specific module:
  - ROCm  - _rocshmem_triton
  - CUDA  - _nvshmem_triton


End users import from here to get backend-agnostic functionality:
  - get_shmem_backend_module(): returns the active backend module
  - requires_shmem(): backend-agnostic @requires_shmem decorator
"""

import torch


def get_shmem_backend_module():
    # [FUNCTION DOC] Returns the platform SHMEM Triton module: _rocshmem_triton when the
    # torch build has HIP, else _nvshmem_triton. Imports the chosen module lazily on each
    # call. Takes no arguments.
    if torch.version.hip is not None:
        from torch.distributed._symmetric_memory import _rocshmem_triton

        return _rocshmem_triton
    from torch.distributed._symmetric_memory import _nvshmem_triton

    return _nvshmem_triton


def requires_shmem(jit_func):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Decorates a Triton JIT function for SHMEM use by delegating to
    # requires_rocshmem on ROCm or requires_nvshmem on CUDA. Returns the wrapped kernel
    # object produced by the backend. Errors (for example a non-JIT input) are raised by
    # the backend decorator. Side effects: loading the backend module and registering the
    # kernel.
    """
    Backend-agnostic Triton decorator for SHMEM kernels.

    Delegates to ``@requires_rocshmem`` on ROCm and ``@requires_nvshmem``
    on CUDA.
    """
    backend = get_shmem_backend_module()
    if torch.version.hip is not None:
        return backend.requires_rocshmem(jit_func)
    return backend.requires_nvshmem(jit_func)
