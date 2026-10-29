# [MODULE DOC] Helpers to page-lock (pin) and unlock existing host memory ranges via the CUDA runtime.
# Import has no side effects beyond importing torch.
import torch


def pin_memory(data_ptr: int, size: int) -> None:
    # [FUNCTION DOC] Registers an existing host memory range (address data_ptr, length size bytes) with CUDA as
    # portable page-locked memory (flag value 1). Returns None. Raises RuntimeError with the CUDA error code
    # if registration returns nonzero (possibly an async error from an earlier CUDA call).
    cudart = torch.cuda.cudart()
    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(
            f"Registering memory failed with cudaError: {succ}."
            " It's possible that this is an asynchronous error raised from a previous cuda operation."
            " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
        )


def unpin_memory(data_ptr: int) -> None:
    # [FUNCTION DOC] Unregisters a host memory range previously pinned at data_ptr. Returns None.
    # Raises AssertionError carrying the CUDA error code if the unregister call returns nonzero.
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    if succ != 0:
        raise AssertionError(f"Unpinning shared memory failed with error-code: {succ}")
