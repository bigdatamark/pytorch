# Owner(s): ["oncall: distributed"]

# [MODULE DOC] HELPER: Shared infrastructure for backend-parameterized c10d tests. Defines
# dtype groups, a frozen BackendConfig capability record, the C10D_BACKENDS table (gloo,
# nccl-legacy, nccl2, nccl-lazy) and CUDA_BACKENDS subset, a C10dBackendTest mixin that
# spawns two processes and initializes a process group, and instantiate_backend_tests,
# which generates one skip-guarded MultiProcessTestCase subclass per backend into a module
# namespace. Import-time side effects: builds the constant tables only.

import os
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_CUDA


STANDARD_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
)

FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

COMPLEX_DTYPES = (
    torch.complex64,
    torch.complex128,
)


@dataclass(frozen=True)
class BackendConfig:
    # [CLASS DOC] HELPER: Immutable description of one c10d backend under test: its name,
    # device type, boolean capability flags (coalescing, bitwise reductions, CUDA graph
    # barrier, dropped P2P work, sequence numbers, collective timing, work sequence number,
    # work result, single-tensor gather, uneven all-gather) and the dtype tuples it
    # supports (standard, float8, premul-sum, complex). Tests read these to gate cases.
    name: str
    device_type: str
    supports_coalescing: bool = False
    supports_bitwise_reductions: bool = False
    supports_cuda_graph_barrier: bool = False
    supports_dropped_p2p_work: bool = False
    supports_sequence_numbers: bool = True
    supports_collectives_timing: bool = False
    supports_work_sequence_number: bool = False
    supports_work_result: bool = False
    supports_gather_single: bool = False
    supports_uneven_all_gather: bool = False
    dtypes: tuple[torch.dtype, ...] = STANDARD_DTYPES
    float8_dtypes: tuple[torch.dtype, ...] = ()
    premul_sum_dtypes: tuple[torch.dtype, ...] = ()
    complex_dtypes: tuple[torch.dtype, ...] = COMPLEX_DTYPES


C10D_BACKENDS = (
    BackendConfig(
        "gloo",
        "cpu",
        supports_bitwise_reductions=True,
        supports_work_sequence_number=True,
    ),
    BackendConfig(
        "nccl-legacy",
        "cuda",
        supports_coalescing=True,
        supports_dropped_p2p_work=True,
        supports_collectives_timing=True,
        supports_work_sequence_number=True,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(torch.float16, torch.float32, torch.float64),
    ),
    BackendConfig(
        "nccl2",
        "cuda",
        supports_coalescing=True,
        supports_cuda_graph_barrier=True,
        supports_dropped_p2p_work=True,
        supports_collectives_timing=True,
        supports_work_sequence_number=True,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ),
    ),
    # nccl-lazy wraps a primary ProcessGroupNCCL (all collectives delegate to
    # it) plus lazily-built per-peer P2P comms. Timing and sequence APIs are not
    # forwarded.
    BackendConfig(
        "nccl-lazy",
        "cuda",
        supports_coalescing=True,
        supports_cuda_graph_barrier=True,
        supports_dropped_p2p_work=True,
        supports_sequence_numbers=False,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ),
    ),
)

CUDA_BACKENDS = tuple(
    backend for backend in C10D_BACKENDS if backend.device_type == "cuda"
)


class C10dBackendTest:
    # [CLASS DOC] HELPER: Mixin placed before MultiProcessTestCase by
    # instantiate_backend_tests. Expects the generated class attributes backend_name and
    # device_type plus the base class rank, file_name. Fixes world size at 2, spawns
    # worker processes in setUp, and provides _init_pg to join a file-store process group.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property; always 2 processes.
        return 2

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this rank for cuda
        # backends, otherwise a device of the configured type without an index.
        if self.device_type == "cuda":
            return torch.device(self.device_type, self.rank)
        return torch.device(self.device_type)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group if initialized, calls
        # the base teardown, then removes the store file, ignoring OS errors.
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        # [FUNCTION DOC] HELPER: For cuda backends sets the current CUDA device to this
        # rank, then initializes the default process group with the test's backend name
        # over a FileStore at the test's temp file, with a 60 second timeout. Global state.
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )


def instantiate_backend_tests(namespace, suite_name, base_class, backends):
    # [FUNCTION DOC] HELPER: For each BackendConfig creates a test class named from the
    # backend (words title-cased) plus suite_name and "Test", deriving from base_class and
    # MultiProcessTestCase with the config fields copied as class attributes and the
    # module set to the caller's. Classes are skipped if the backend is unavailable, and
    # for cuda backends also if fewer than 2 GPUs. Each is stored in the namespace mapping
    # (mutated in place) so test discovery finds it. Returns nothing.
    for backend in backends:
        backend_name = backend.name.replace("-", " ").title().replace(" ", "")
        class_name = f"{backend_name}{suite_name}Test"
        test_class = type(
            class_name,
            (base_class, MultiProcessTestCase),
            {
                "__module__": namespace["__name__"],
                "backend_name": backend.name,
                "device_type": backend.device_type,
                "supports_coalescing": backend.supports_coalescing,
                "supports_bitwise_reductions": backend.supports_bitwise_reductions,
                "supports_cuda_graph_barrier": backend.supports_cuda_graph_barrier,
                "supports_dropped_p2p_work": backend.supports_dropped_p2p_work,
                "supports_sequence_numbers": backend.supports_sequence_numbers,
                "supports_collectives_timing": backend.supports_collectives_timing,
                "supports_work_sequence_number": backend.supports_work_sequence_number,
                "supports_work_result": backend.supports_work_result,
                "supports_gather_single": backend.supports_gather_single,
                "supports_uneven_all_gather": backend.supports_uneven_all_gather,
                "dtypes": backend.dtypes,
                "float8_dtypes": backend.float8_dtypes,
                "premul_sum_dtypes": backend.premul_sum_dtypes,
                "complex_dtypes": backend.complex_dtypes,
            },
        )
        test_class = unittest.skipIf(
            not dist.is_backend_available(backend.name),
            f"{backend.name} backend is not available",
        )(test_class)
        if backend.device_type == "cuda":
            test_class = unittest.skipIf(
                not TEST_CUDA or torch.cuda.device_count() < 2,
                f"{backend.name} requires 2+ GPUs",
            )(test_class)
        namespace[class_name] = test_class
