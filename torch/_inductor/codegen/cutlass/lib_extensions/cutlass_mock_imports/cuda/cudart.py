# [MODULE DOC] HELPER: Stub of the cuda-python runtime API used by the CUTLASS python library. Offers a
# success error code class, a no-op free, and a device-properties query backed by torch. No import-time
# side effects beyond importing torch.cuda.
# mypy: disable-error-code="no-untyped-def"
import torch.cuda


class cudaError_t:
    # [CLASS DOC] HELPER: Placeholder error enum exposing only cudaSuccess (set to True).
    cudaSuccess = True


def cudaFree(n):
    # [FUNCTION DOC] HELPER: No-op free; ignores the pointer and returns a one-element tuple holding the success code.
    return (cudaError_t.cudaSuccess,)


def cudaGetDeviceProperties(d):
    # [FUNCTION DOC] HELPER: Returns a pair of (dummy error object whose value is False, torch device
    # properties for device d). Does not call the real CUDA runtime.
    class DummyError:
        # [CLASS DOC] HELPER: Fake error object with a falsy value attribute, meaning no error.
        value = False

    return (DummyError(), torch.cuda.get_device_properties(d))
