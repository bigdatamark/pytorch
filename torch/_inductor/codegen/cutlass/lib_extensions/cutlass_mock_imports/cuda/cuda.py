# [MODULE DOC] HELPER: Stub definitions mimicking the cuda-python driver API surface that the CUTLASS
# python library touches at import time. Provides empty placeholder classes and a device count function
# backed by torch. No import-time side effects beyond importing torch.
# mypy: disable-error-code="no-untyped-def"
# flake8: noqa
import torch


class CUdeviceptr:
    # [CLASS DOC] HELPER: Empty placeholder for the driver device pointer type; holds no state.
    pass


class CUstream:
    # [CLASS DOC] HELPER: Placeholder stream type; constructor accepts and ignores a value.
    def __init__(self, v):
        # [FUNCTION DOC] HELPER: Accepts a stream handle and ignores it; no state stored.
        pass


class CUresult:
    # [CLASS DOC] HELPER: Placeholder result enum exposing only CUDA_SUCCESS (set to True).
    CUDA_SUCCESS = True


class nvrtc:
    # [CLASS DOC] HELPER: Empty placeholder for the nvrtc namespace.
    pass


def cuDeviceGetCount():
    # [FUNCTION DOC] HELPER: Returns a pair of (success marker, number of CUDA devices reported by torch).
    return (CUresult.CUDA_SUCCESS, torch.cuda.device_count())
