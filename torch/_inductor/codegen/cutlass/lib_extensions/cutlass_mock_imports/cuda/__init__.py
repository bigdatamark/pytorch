# [PACKAGE DOC] HELPER: Mock replacement for the cuda-python "cuda" package, used so the CUTLASS python
# library can be imported without the real dependency. Sets a version string from the torch CUDA
# version and re-exports the stub names from the cuda submodule (CUdeviceptr, CUstream, CUresult, nvrtc,
# cuDeviceGetCount).
import torch


__version__ = torch.version.cuda

from .cuda import *  # noqa: F403
