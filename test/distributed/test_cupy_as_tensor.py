# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process CUDA test checking that torch.as_tensor can wrap a raw device
# pointer described through the __cuda_array_interface__ protocol (as cupy arrays do) without
# copying, using memory shared between processes via CUDA IPC. Defines a small pointer wrapper
# dataclass, a from_buffer helper, and the CupyAsTensorTest case. Import has no side effects
# beyond resolving the cuda device module; run_tests is invoked only when run as a script.

# To run:
# python test/distributed/test_cupy_as_tensor.py

from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@dataclass
class CupyWrapper:
    # [CLASS DOC] HELPER: Dataclass holding a raw device pointer and a byte size. It mimics
    # a cupy array by exposing the CUDA array interface, so torch.as_tensor can adopt the
    # memory as a flat uint8 buffer without copying. Fields: data_ptr, size_in_bytes.
    data_ptr: int
    size_in_bytes: int

    @property
    def __cuda_array_interface__(self):
        # [FUNCTION DOC] HELPER: Property returning the version 3 CUDA array interface
        # mapping: a one-dimensional shape of size_in_bytes, unsigned 8-bit element type,
        # the device pointer marked writable (read-only flag false). No side effects.
        return {
            "shape": (self.size_in_bytes,),
            "typestr": "|u1",
            "data": (self.data_ptr, False),
            "version": 3,
        }


def from_buffer(
    data_ptr: int, size_in_bytes: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Wrap a raw device pointer as a tensor through the wrapper
    # object above, then reinterpret the bytes with the requested dtype. Arguments are the
    # pointer, its size in bytes, target device and element dtype. Returns the tensor.
    # Raises an AssertionError if the tensor's data pointer differs from the given one
    # (i.e. a copy was made instead of zero-copy adoption).
    data = torch.as_tensor(CupyWrapper(data_ptr, size_in_bytes), device=device).view(
        dtype
    )
    if data.data_ptr() != data_ptr:
        raise AssertionError(
            f"Expected data_ptr to be {data_ptr}, got {data.data_ptr()}"
        )
    return data


@requires_cuda_p2p_access()
class CupyAsTensorTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test case (gloo process group, one process per rank) that
    # is skipped unless CUDA peer-to-peer access is available. Holds one test that shares
    # a CUDA allocation from rank 1 to the other ranks and adopts it via the array interface.
    # The class teardown restores the allocator setting changed by the test.
    @classmethod
    def backend_str(cls):
        # [FUNCTION DOC] HELPER: Selects the gloo backend for the test process group.
        return "gloo"

    def _init_device(self) -> None:
        # [FUNCTION DOC] HELPER: Per-process setup. Enables the expandable-segments CUDA
        # allocator (process-global side effect, needed so memory is VMM based and IPC
        # sharing behaves as the test expects), pins the process to this rank's device and
        # allocates a one-element tensor to force context creation.
        # need to use vmm api to test it,
        # see https://forums.developer.nvidia.com/t/inconsistent-behavior-of-cudapointergetattributes-between-cudamalloc-ipc-and-vmm-based-ipc/339025/5
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        # init and pin the process to the device
        device_module.set_device(self.device)
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property giving the cuda device whose index equals this
        # process's rank.
        return torch.device(device_type, self.rank)

    @skip_if_rocm_multiprocess  # RuntimeError: pidfd_getfd Operation not permitted"
    @skip_but_pass_in_sandcastle_if(
        SM100OrLater,
        "Fails if ran in docker environment without privileged access (https://github.com/pytorch/pytorch/issues/165170)",
    )
    def test_cupy_as_tensor(self) -> None:
        # [FUNCTION DOC] Rank 1 allocates a random tensor and broadcasts its IPC
        # reduction metadata; every other rank rebuilds the tensor on its own device
        # (patching the device index field of the arguments to its rank), then re-wraps the
        # pointer with from_buffer, which checks zero-copy. After a barrier rank 1 fills
        # its tensor with ones, devices are synchronized, and all ranks check the tensor
        # values against 1. Raises AssertionError on mismatch.
        # KNOWN ISSUE: the final check calls allclose on the tensor with the tensor itself as
        # the other operand and 1 as the relative tolerance, not as the expected value, so
        # it never verifies the data written by rank 1.
        """
        Test that torch.as_tensor works for cupy array interface
        with zero-copy when the pointer is p2p-shared across processes.
        """
        self._init_device()

        tensor: torch.Tensor
        if self.rank == 1:
            # it seems only error from rank non-zero will be caught by this test
            tensor = torch.randn(2333, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            torch.distributed.broadcast_object_list([tensor_meta], src=1)
        else:
            recv_list = [None]
            torch.distributed.broadcast_object_list(recv_list, src=1)
            tensor_meta = recv_list[0]
            func, args = tensor_meta
            args = list(args)
            args[6] = self.rank
            ipc_tensor = func(*args)
            tensor = from_buffer(
                ipc_tensor.data_ptr(),
                ipc_tensor.numel() * ipc_tensor.element_size(),
                self.device,
                ipc_tensor.dtype,
            )

        torch.distributed.barrier()
        if self.rank == 1:
            tensor.fill_(1)
        device_module.synchronize()
        torch.distributed.barrier()
        if not tensor.allclose(tensor, 1):
            raise AssertionError("Expected tensor to be close to 1")
        torch.distributed.barrier()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Turns the expandable-segments allocator setting back off in
        # this process, then runs the parent class teardown.
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        super().tearDownClass()


if __name__ == "__main__":
    run_tests()
