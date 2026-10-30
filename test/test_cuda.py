# Owner(s): ["module: cuda"]
# ruff: noqa: F841
# [MODULE DOC] Test suite for the torch.cuda Python API and the CUDA/HIP runtime integration:
# caching allocator and host (pinned) allocator statistics, memory snapshots and history
# recording, streams and events, CUDA graphs, autocast, gradient scaling, device properties,
# NVML helpers and misc runtime knobs. Main contents: module-level skip flags
# (TEST_LARGE_TENSOR, TEST_BF16, ...), a lazily compiled spin-wait kernel helper, and the large
# TestCuda class plus several smaller TestCase classes that are device-type instantiated at the
# end. Import-time side effects: queries CUDA device 0 properties when CUDA is available, probes
# for torchvision, and rebinds load_tests for test sharding.

import contextlib
import ctypes
import functools
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
import weakref
from collections import defaultdict
from copy import deepcopy
from itertools import product
from random import randint
from unittest.mock import patch

import psutil

import torch
import torch.cuda
import torch.nn as nn
from torch import inf, nan
from torch.cuda._memory_viz import (
    _profile_to_snapshot,
    profile_plot,
    segment_plot,
    trace_plot,
)
from torch.testing._internal.autocast_test_lists import AutocastTestLists, TestAutocast
from torch.testing._internal.common_cuda import (
    _create_scaling_case,
    _get_torch_cuda_version,
    blas_library_context,
    PLATFORM_SUPPORTS_GREEN_CONTEXT,
    PLATFORM_SUPPORTS_WORKQUEUE_CONFIG,
    SM70OrLater,
    SM89OrLater,
    TEST_CUDNN,
    TEST_MULTIGPU,
    tf32_on_and_off,
    xfailCUDAIfSM89OrLaterOnWindows,
)
from torch.testing._internal.common_device_type import (
    has_cusolver,
    has_hipsolver,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIf,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    cuda_python_error_check,
    EXPANDABLE_SEGMENTS,
    freeze_rng_state,
    gcIfJetson,
    get_cycles_per_ms,
    getRocmVersion,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_FBCODE,
    IS_JETSON,
    IS_LINUX,
    IS_SANDCASTLE,
    IS_WINDOWS,
    IS_X86,
    load_tests,
    MI200_ARCH,
    MI350_ARCH,
    parametrize,
    recover_orig_fp32_precision,
    requires_cuda_python_bindings,
    run_tests,
    serialTest,
    setBlasBackendsToDefaultFinally,
    skipCUDAMemoryLeakCheckIf,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    skipIfRocmArch,
    skipIfRocmVersionAtLeast,
    skipIfRocmVersionLessThan,
    slowTest,
    subtest,
    TemporaryFileName,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TEST_NUMPY,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TestCase,
)
from torch.utils._triton import has_triton
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.viz._cycles import observe_tensor_cycles


requiresCppContext = unittest.skipUnless(
    (IS_X86 or IS_ARM64) and IS_LINUX, "cpp contexts are linux x86/aarch64 only"
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

try:
    import torchvision.models  # noqa: F401

    # from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_BF16 = False
TEST_PYNVML = (
    torch.cuda._HAS_PYNVML and not IS_JETSON
)  # nvml not fully supported on jetson
if TEST_CUDA:
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
    TEST_BF16 = torch.cuda.is_bf16_supported()

_cycles_per_ms = None

_wait_for_cpu_kernel = None


def skip_background_threads_on_windows(f):
    # [FUNCTION DOC] HELPER: Decorator for test methods that receive keyword-only parametrized
    # arguments. The wrapper skips the test (raises SkipTest) on Windows with an SM89-or-newer
    # GPU when the use_background_threads keyword is truthy, since the pinned allocator
    # background thread hangs there; otherwise it calls the wrapped test with self and all
    # keyword arguments and returns its result.
    @functools.wraps(f)
    def wrapped(self, **kwargs):
        # [FUNCTION DOC] HELPER: Inner wrapper produced by the decorator above. Takes the test
        # instance and keyword arguments only; raises unittest.SkipTest for the Windows plus
        # SM89+ plus background-threads case, else forwards to the original.
        if IS_WINDOWS and SM89OrLater and kwargs.get("use_background_threads"):
            raise unittest.SkipTest("using background threads fails on Windows")
        return f(self, **kwargs)

    return wrapped


def get_wait_for_cpu_kernel():
    # [FUNCTION DOC] Returns a cached compiled GPU kernel that spins until the host writes a
    # non-zero value into a pinned int32 flag, so a stream can be blocked deterministically.
    # Compiles lazily on first call via torch.cuda._compile_kernel (NVIDIA uses a relaxed
    # system-scope load, AMD uses a system-scope atomic load) and stores the result in the
    # module global; later calls return the same object.
    """Returns a compiled CUDA/HIP spin-wait kernel that blocks the GPU stream
    until the host sets a pinned int32 flag to non-zero. On NVIDIA, requires
    SM70+ for the relaxed-system-scope PTX load. On AMD, uses
    __hip_atomic_load with __HIP_MEMORY_SCOPE_SYSTEM (the AMDGCN equivalent).

    Usage::

        kernel = get_wait_for_cpu_kernel()
        flag = torch.zeros(1, dtype=torch.int32, device="cpu").pin_memory()
        with torch.cuda.stream(s):
            kernel(grid=(1, 1, 1), block=(1, 1, 1), args=[flag])
        # stream s is now blocked until:
        flag[0] = 1
    """
    global _wait_for_cpu_kernel
    if _wait_for_cpu_kernel is None:
        from torch.cuda import _compile_kernel

        _wait_for_cpu_kernel = _compile_kernel(
            r"""
            __global__ void wait_for_cpu(int *pinned_cpu_flag) {
                int flag = 0;
                do {
            #ifdef __HIP_PLATFORM_AMD__
                    flag = __hip_atomic_load(pinned_cpu_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            #else
                    asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(flag) : "l"(pinned_cpu_flag) : "memory");
            #endif
                } while (flag == 0);
            }
            """,
            "wait_for_cpu",
        )
    return _wait_for_cpu_kernel


def _check_allocator_settings_on_tear_down(test_case):
    # [FUNCTION DOC] HELPER: Regression check meant for a test class tearDown. Reads the
    # allocator settings from the current memory snapshot and asserts that the
    # expandable_segments setting equals the suite baseline derived from the environment
    # (EXPANDABLE_SEGMENTS), failing the test case if a test changed it.
    # Regression check: no test in `test_case` should leave the runtime
    # expandable_segments knob mismatched against the suite's env-derived
    # baseline. This should be called in the class's `tearDown` method.
    md = torch.cuda.memory._snapshot()["allocator_settings"]
    test_case.assertEqual(md["expandable_segments"], EXPANDABLE_SEGMENTS)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCuda(TestCase):
    # [CLASS DOC] Main CUDA runtime test case. Class flags enable the per-test CUDA memory leak
    # check and force running tests on a non-default stream. Holds a constant FIFTY_MIL_CYCLES
    # used by spin-wait helpers. Covers allocators, pinned memory, streams, events, graphs,
    # snapshots, device queries and more. The whole class is skipped when CUDA is unavailable
    # and is wrapped for strict dynamo mode.
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture hook; only delegates to the base class setUp.
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture hook; only delegates to the base class tearDown.
        super().tearDown()

    @property
    def expandable_segments(self):
        # [FUNCTION DOC] HELPER: Property returning the module-level EXPANDABLE_SEGMENTS flag,
        # i.e. whether the suite runs with the expandable segments allocator mode.
        return EXPANDABLE_SEGMENTS

    def test_embedding_renorm_negative_indices(self):
        # [FUNCTION DOC] Checks the embedding_renorm_ op with negative indices: builds an index
        # tensor holding values from -32 to 31 and a weight table whose rows near both ends of
        # the table are large, runs the op on a CPU clone and on CUDA copies with max norm 1.0,
        # and asserts the CUDA result (moved to CPU) matches the CPU one.
        indices = torch.arange(-32, 32, dtype=torch.int32).repeat(8).reshape(32, 16)
        weight = torch.full((1000, 64), 0.01)
        weight[torch.arange(968, 1000)] = 1.0
        weight[torch.arange(0, 32)] = 1.0

        expected = torch.ops.aten.embedding_renorm_.default(
            weight.clone(), indices, 1.0, 1.0
        )
        actual = torch.ops.aten.embedding_renorm_.default(
            weight.cuda(), indices.cuda(), 1.0, 1.0
        )

        self.assertEqual(actual.cpu(), expected, rtol=1e-4, atol=1e-6)

    def test_pinned_memory_with_cudaregister(self):
        # [FUNCTION DOC] Enables the pinned_use_cuda_host_register allocator option with 8
        # register threads, checks a small tensor is not pinned, then pins 2 MiB-element and 16
        # Mi-element tensors and asserts they report pinned. A RuntimeError during pinning is
        # swallowed since some GPUs lack a shared host/device address space. Always resets the
        # option to False in the finally block.
        try:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
            )
            t = torch.ones(20)
            self.assertFalse(t.is_pinned())
            try:
                pinned_t = torch.ones(1 << 21).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
                pinned_t = torch.ones(1 << 24).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
            except RuntimeError as e:
                # Some GPUs don't support same address space on host and device side
                pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    def test_pinned_memory_with_cudaregister_multithread(self):
        # [FUNCTION DOC] Runs the cudaregister pinned-memory test body concurrently from 4
        # threads (the test method is the thread target), starts them all and joins them all.
        # Exceptions inside the threads are not propagated to the main test.
        num_threads = 4
        threads = [
            threading.Thread(target=self.test_pinned_memory_with_cudaregister)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    @serialTest()
    def test_host_memory_stats(self):
        # [FUNCTION DOC] Verifies host (pinned) memory allocator statistics. Records a baseline
        # of host_memory_stats after gc, host cache emptying and stat resets, then allocates two
        # pinned tensors (10K and 20K floats, rounded up to 16K and 32K elements) and checks
        # that allocation counts, bytes, active counters and the host alloc counters grow by the
        # expected deltas relative to the baseline. Also checks stats are unchanged by emptying
        # the host cache, then resets peak/accumulated stats.
        # Helper functions
        def empty_stats():
            # [FUNCTION DOC] HELPER: Returns a fresh mapping from each tracked host stat key
            # (allocated/current/freed/peak for bytes, allocation counts and active
            # bytes/requests, plus host alloc and free time counts and counters) to zero.
            return {
                "allocated_bytes.allocated": 0,
                "allocated_bytes.current": 0,
                "allocated_bytes.freed": 0,
                "allocated_bytes.peak": 0,
                "allocations.allocated": 0,
                "allocations.current": 0,
                "allocations.freed": 0,
                "allocations.peak": 0,
                "host_alloc_time.count": 0,
                "host_free_time.count": 0,
                "num_host_alloc": 0,
                "num_host_free": 0,
                "active_bytes.allocated": 0,
                "active_bytes.current": 0,
                "active_bytes.freed": 0,
                "active_bytes.peak": 0,
                "active_requests.allocated": 0,
                "active_requests.current": 0,
                "active_requests.freed": 0,
                "active_requests.peak": 0,
            }

        # Deltas from baseline captured after cleanup (not absolute totals), so
        # ordering against other tests cannot leave host stats in a broken state.
        baseline = {}

        def check_stats(expected):
            # [FUNCTION DOC] HELPER: Reads the current host memory stats and for every key in
            # the expected delta mapping asserts the value equals the closure baseline plus the
            # delta. On mismatch it prints the key, baseline, delta, wanted and actual first.
            stats = torch.cuda.host_memory_stats()
            for k, v in expected.items():
                want = baseline[k] + v
                actual = stats[k]
                if actual != want:
                    print(
                        f"key: {k}, baseline: {baseline[k]}, delta: {v}, "
                        f"want: {want}, got: {actual}"
                    )
                self.assertEqual(actual, want)

        # Setup the test cleanly
        alloc1 = 10
        alloc1_aligned = 16
        alloc2 = 20
        alloc2_aligned = 32
        expected = empty_stats()

        gc.collect()
        torch._C._host_emptyCache()
        torch.cuda.reset_peak_host_memory_stats()
        torch.cuda.reset_accumulated_host_memory_stats()
        baseline.update(torch.cuda.host_memory_stats())

        check_stats(expected)

        # Make first allocation and check stats
        t1 = torch.ones(alloc1 * 1024, pin_memory=True)
        self.assertTrue(t1.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size1 = alloc1_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size1

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Make second allocation and check stats
        t2 = torch.ones(alloc2 * 1024, pin_memory=True)
        self.assertTrue(t2.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size2 = alloc2_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size2

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Empty cache and check stats
        torch._C._host_emptyCache()

        check_stats(expected)

        # Finally, check the reset of peak and accumulated stats
        torch.cuda.reset_peak_host_memory_stats()
        torch.cuda.reset_accumulated_host_memory_stats()

        expected = empty_stats()

    @serialTest()
    def test_host_memory_stats_active_after_free_no_events(self):
        # [FUNCTION DOC] Regression test for the host allocator free path when no CUDA events
        # were recorded: after initializing CUDA and resetting host stats, twice pins and drops
        # a 1M-float tensor and asserts active_bytes.current and active_requests.current return
        # to zero each time (the counters must decrement even though the block goes straight
        # back to the free list).
        """Regression test for free()'s no-events fast path.

        When a pinned tensor is freed without any CUDA stream having been
        recorded against its block (i.e. it was never used as the source of
        an async H2D copy), the allocator pushes the block back onto the
        free list directly. That path must decrement active_requests.current
        and active_bytes.current; otherwise the counters grow monotonically
        on every plain pin-and-drop cycle even though the underlying memory
        is correctly recycled.

        Two cycles exercise both alloc paths: the first triggers a new slab
        via add_allocated_block, the second reuses the cached block via
        get_free_block.
        """
        # host_memory_stats() returns an empty dict until the host caching
        # allocator has been initialized; force init so the stat keys exist
        # regardless of test ordering.
        torch.cuda.init()
        gc.collect()
        torch._C._host_emptyCache()
        torch.cuda.reset_peak_host_memory_stats()
        torch.cuda.reset_accumulated_host_memory_stats()

        for _ in range(2):
            t = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
            self.assertTrue(t.is_pinned())
            del t
            gc.collect()
            stats = torch.cuda.host_memory_stats()
            self.assertEqual(
                stats["active_bytes.current"],
                0,
                "active_bytes.current should be 0 after a pin-and-drop cycle "
                "with no async copy: the block is returned to the cache, so "
                "the active counter must drop back to zero.",
            )
            self.assertEqual(
                stats["active_requests.current"],
                0,
                "active_requests.current should be 0 after a pin-and-drop cycle.",
            )

    @serialTest()
    def test_host_memory_stats_active_after_event_drain(self):
        # [FUNCTION DOC] Regression test for the pinned-block event-drain path: twice allocates
        # a pinned tensor, does a non-blocking copy to the GPU, synchronizes, frees it and
        # empties the host cache. Asserts active bytes/requests current are zero and that the
        # freed counters are non-negative (a negative value would mean the drain used the wrong
        # block size).
        """Regression test for the event-drain path through empty_cache.

        When a pinned block is used as the source of a non_blocking H2D copy
        and then freed, the block sits on the events queue until
        process_events_for_specific_size drains it. That function takes a
        size parameter that is -1 when invoked from process_events (the
        all-sizes case, reached via _host_emptyCache). The decrement of
        active_bytes_bucket_stats must use the block's actual size, not the
        -1 parameter, otherwise active_bytes.current is incremented by 1
        byte per drained event and active_bytes.freed goes negative.
        """
        torch.cuda.init()
        gc.collect()
        torch._C._host_emptyCache()
        torch.cuda.reset_peak_host_memory_stats()
        torch.cuda.reset_accumulated_host_memory_stats()

        for _ in range(2):
            t = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
            t.cuda(non_blocking=True)
            torch.cuda.synchronize()
            del t
            gc.collect()
            # Drain via process_events_for_specific_size(-1, ...).
            torch._C._host_emptyCache()

            stats = torch.cuda.host_memory_stats()
            self.assertEqual(
                stats["active_bytes.current"],
                0,
                "active_bytes.current should be 0 after empty_cache drains "
                "the events queue.",
            )
            self.assertEqual(
                stats["active_requests.current"],
                0,
                "active_requests.current should be 0 after drain.",
            )
            self.assertGreaterEqual(
                stats["active_bytes.freed"],
                0,
                "active_bytes.freed must be non-negative; a negative value "
                "means the drain decremented by the wrong size.",
            )
            self.assertGreaterEqual(
                stats["active_requests.freed"],
                0,
                "active_requests.freed must be non-negative.",
            )

    def test_pinned_memory_empty_cache(self):
        # [FUNCTION DOC] For both settings of the host-register allocator option, pins a 1M
        # element tensor, checks it is pinned, deletes it and empties the host cache. A
        # RuntimeError (unsupported shared address space on some GPUs) is ignored. The option is
        # reset to False afterwards.
        try:
            for alloc_settings in (True, False):
                torch.cuda.memory._set_allocator_settings(
                    f"pinned_use_cuda_host_register:{alloc_settings}"
                )
                try:
                    t = torch.ones(1024 * 1024, pin_memory=True)
                    self.assertTrue(t.is_pinned())
                    del t
                    torch._C._host_emptyCache()
                except RuntimeError as e:
                    # Some GPUs don't support same address space on host and device side
                    pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    # Pinned allocator background thread does not shut down cleanly on Windows
    # Python process hangs
    @unittest.skipIf(IS_WINDOWS and SM89OrLater, "Fails on windows with SM89+")
    def test_pinned_memory_use_background_threads(self):
        # [FUNCTION DOC] Runs a child Python process that enables the pinned allocator
        # background threads option, allocates a pinned tensor and prints whether it is pinned;
        # asserts the child exits with return code zero. Skipped on Windows with SM89+ because
        # the background thread hangs process shutdown there.
        script = """
import torch

torch.cuda.memory._set_allocator_settings(
    f"pinned_use_background_threads:True"
)
t = torch.ones(1024 * 1024, pin_memory=True)
print(t.is_pinned())
"""
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True)
        self.assertEqual(proc.returncode, 0)

    def test_cudart_register(self):
        # [FUNCTION DOC] Uses the cudart binding to register a plain CPU tensor's memory with
        # cudaHostRegister and checks the call returns 0 and the tensor becomes pinned, then
        # unregisters it and checks the return value is 0 and it is no longer pinned.
        t = torch.ones(20)
        self.assertFalse(t.is_pinned())
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        self.assertEqual(r, 0)
        self.assertTrue(t.is_pinned())
        r = cudart.cudaHostUnregister(t.data_ptr())
        self.assertEqual(r, 0)
        self.assertFalse(t.is_pinned())

    def test_memory_allocation(self):
        # [FUNCTION DOC] Allocates one byte through the caching allocator API and checks
        # torch.cuda.memory_allocated increases; in the cleanup path frees the pointer via the
        # allocator delete API and asserts the allocated counter returns to its previous value.
        gc.collect()
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.cuda.memory_allocated()
            mem = torch.cuda.caching_allocator_alloc(size)
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                self.assertEqual(torch.cuda.memory_allocated(), prev)

    def test_caching_allocator_alloc_negative_size(self):
        # [FUNCTION DOC] Asserts that requesting a negative size from the caching allocator
        # alloc API raises ValueError with an "Invalid memory size" message.
        with self.assertRaisesRegex(ValueError, "Invalid memory size"):
            torch.cuda.memory.caching_allocator_alloc(-1024)

    def test_memory_metadata_supported(self):
        # [FUNCTION DOC] Asserts that the runtime reports user memory metadata support exactly
        # when the allocator backend is not cudaMallocAsync.
        # The native caching allocator records user metadata; cudaMallocAsync
        # (and pluggable allocators) do not.
        self.assertEqual(
            torch._C._cuda_memoryMetadataSupported(), not TEST_CUDAMALLOCASYNC
        )

    def test_memory_metadata_unsupported_backend_is_noop(self):
        # [FUNCTION DOC] Skips if the backend supports metadata. Otherwise checks that setting
        # memory metadata is a silent best-effort no-op (warnings suppressed) and that reading
        # it back always returns an empty string without raising.
        if torch._C._cuda_memoryMetadataSupported():
            self.skipTest("backend supports user metadata")
        # On backends without metadata support, set is a best-effort no-op
        # (warns once) and get always returns ""; neither raises.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.cuda.memory._set_memory_metadata("metadata test")
        self.assertEqual(torch.cuda.memory._get_memory_metadata(), "")

    def test_memory_stats(self):
        # [FUNCTION DOC] Tests memory statistics through the accelerator API. After gc, cache
        # emptying and stat resets, checks allocated equals max allocated and reserved equals
        # max reserved. Allocates a 1 KiB tensor and checks the all-pools active_bytes current
        # grows by 1024 and freed is 0; after deleting it and emptying the cache current returns
        # to the prior value and freed is 1024. Finally resets peak stats and checks the max
        # values equal the earlier ones.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        self.assertEqual(prev_allocated, prev_max_allocated)
        self.assertEqual(prev_reserved, prev_max_reserved)
        # Activate 1kB memory
        prev_active_current = torch.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        tmp = torch.randn(256, device="cuda")
        # Detect if the current active memory is 1kB
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(torch.accelerator.memory_stats()["active_bytes.all.freed"], 0)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.freed"], 1024
        )
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(torch.accelerator.max_memory_reserved(), prev_max_reserved)

    def test_check_error(self):
        # [FUNCTION DOC] Checks torch.cuda.check_error: error code 0 must not raise, and
        # error code 2 must raise torch.cuda.CudaError with an out-of-memory message (CUDA or
        # HIP wording).
        # Assert this call doesn't raise.
        torch.cuda.check_error(0)

        with self.assertRaisesRegex(
            torch.cuda.CudaError, "out of memory|hipErrorOutOfMemory"
        ):
            torch.cuda.check_error(2)

    def test_cuda_get_device_name(self):
        # [FUNCTION DOC] Asserts get_device_name returns the same string for an explicit current
        # device index, for None, and for no argument.
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    def test_cuda_get_device_capability(self):
        # [FUNCTION DOC] Asserts get_device_capability returns the same value for the explicit
        # current device index, for None, and for no argument.
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

    def test_cuda_get_device_properties(self):
        # [FUNCTION DOC] Asserts get_device_properties returns equal results for the explicit
        # current device index, for None, and for no argument.
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        device_properties_None = torch.cuda.get_device_properties(None)
        self.assertEqual(current_device_properties, device_properties_None)

        # Testing the behaviour for No argument
        device_properties_no_argument = torch.cuda.get_device_properties()
        self.assertEqual(current_device_properties, device_properties_no_argument)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_out_of_memory(self):
        # [FUNCTION DOC] Checks out-of-memory error reporting. Skipped on Jetson and on ROCm
        # 7.14+ with expandable segments. Asserts a gigantic allocation raises RuntimeError with
        # the expected message (allowed-memory wording for cudaMallocAsync, otherwise the
        # tried-to-allocate wording naming the GPU), an even larger one raises the "more than
        # 1EB" error, and that a following kernel on an earlier tensor still works.
        if TEST_WITH_ROCM and getRocmVersion() >= (7, 14) and EXPANDABLE_SEGMENTS:
            self.skipTest(
                "TestCuda.test_out_of_memory: OOM tensor flag is False on ROCm "
                "expandable segments (7.14+)"
            )
        tensor = torch.zeros(1024, device="cuda")

        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else f"Tried to allocate 800000000.00 GiB. GPU {tensor.device.index} has a total capacity of"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(
                1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="cuda"
            )

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    # CUDA memory allocations on windows do not OOM on rtx even when they cross allowed memory
    # Skip test until this is investigated
    @unittest.skipIf(IS_WINDOWS and SM89OrLater, "Fails on windows with SM89+")
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC or IS_JETSON, "Segmentation fault (core dumped)"
    )
    # On MI350 (gfx950) allocating ~1.02x of the reported free memory does not
    # raise an OOM RuntimeError, so the assertRaisesRegex below fails.
    @skipIfRocmArch(MI350_ARCH)
    @serialTest()
    def test_out_of_memory_retry(self):
        # [FUNCTION DOC] Checks OOM-then-retry behavior. Allocates about 51 percent of the free
        # device memory, expects a second identical allocation to raise RuntimeError with an OOM
        # message, frees the first and checks the allocation then succeeds. Cleans up by
        # emptying the cache and resetting peak stats. Skipped on several platforms (Windows
        # SM89+, async allocator, Jetson, MI350).
        torch.cuda.empty_cache()
        available_memory = torch.cuda.mem_get_info(0)[0]
        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else "Tried to allocate"
        )
        size = int(available_memory * 0.51)
        a = torch.empty(size, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            b = torch.empty(size, dtype=torch.int8, device="cuda")
        del a
        b = torch.empty(size, dtype=torch.int8, device="cuda")
        del b
        # We used a lot of memory here, clean up so we don't affect other tests too much
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @serialTest()
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_set_per_process_memory_fraction(self):
        # [FUNCTION DOC] Tests set_per_process_memory_fraction. Checks invalid inputs raise (an
        # integer gives a TypeError, values below 0 or above 1 give ValueError), then sets
        # fraction 0.5 on device 0, verifies an allocation just under 0.499 of total memory
        # succeeds and one of 0.5 raises an OOM RuntimeError, and that a previously created
        # tensor is still usable. Restores the original fraction in a finally block.
        if TEST_WITH_ROCM and getRocmVersion() >= (7, 14) and EXPANDABLE_SEGMENTS:
            self.skipTest(
                "ROCm 7.14+ expandable segments reports OOM below the expected "
                "per-process memory fraction limit"
            )

        torch.cuda.empty_cache()
        orig = torch.cuda.get_per_process_memory_fraction(0)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            # test invalid fraction value.
            with self.assertRaisesRegex(TypeError, "Invalid type"):
                torch.cuda.set_per_process_memory_fraction(1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(-0.1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(2.0)

            tensor = torch.zeros(1024, device="cuda")
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.5, 0)

            # test 0.499 allocation is ok.
            application = int(total_memory * 0.499) - torch.cuda.max_memory_reserved()
            tmp_tensor = torch.empty(application, dtype=torch.int8, device="cuda")
            del tmp_tensor
            torch.cuda.empty_cache()

            application = int(total_memory * 0.5)
            # it will get OOM when try to allocate more than half memory.
            oom_regex = (
                "would exceed allowed memory"
                if TEST_CUDAMALLOCASYNC
                else "out of memory"
            )
            with self.assertRaisesRegex(RuntimeError, oom_regex):
                torch.empty(application, dtype=torch.int8, device="cuda")

            # ensure out of memory error doesn't disturb subsequent kernel
            tensor.fill_(1)
            self.assertTrue((tensor == 1).all())
        finally:
            torch.cuda.set_per_process_memory_fraction(orig, 0)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    @serialTest()
    def test_get_per_process_memory_fraction(self):
        # [FUNCTION DOC] Tests get_per_process_memory_fraction round trips: sets and reads back
        # 1.0 and 0.0 and three random fractions, then restores the initial fraction.
        # get the initial memory fraction
        init_fraction = torch.cuda.get_per_process_memory_fraction()

        # set and get the limiting cases
        torch.cuda.set_per_process_memory_fraction(1.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 1.0)
        torch.cuda.set_per_process_memory_fraction(0.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 0.0)

        # test a few random cases
        for val in torch.rand(3):
            torch.cuda.set_per_process_memory_fraction(float(val))
            self.assertEqual(torch.cuda.get_per_process_memory_fraction(), float(val))

        # restore the initial memory fraction
        torch.cuda.set_per_process_memory_fraction(init_fraction)

    def test_uuid(self):
        # [FUNCTION DOC] Checks the device properties uuid object: its string form is 36
        # characters long and its bytes representation is 16 bytes.
        uuid = torch.cuda.get_device_properties(0).uuid
        self.assertEqual(len(str(uuid)), 36)  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        self.assertEqual(len(uuid.bytes), 16)

    def test_copy_non_blocking(self):
        # [FUNCTION DOC] Tests non-blocking copies between pinned host memory and the GPU in
        # both directions using 10 MB uint8 tensors, including a pinned slice whose data pointer
        # differs from its storage pointer. Each copy is followed by an event record and
        # synchronize, then the values are compared.
        def _test_copy_non_blocking(a, b):
            # [FUNCTION DOC] HELPER: Copies b into a with a non-blocking copy, records and
            # synchronizes a CUDA event, then asserts a equals b.
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

        # Test the case where the pinned data_ptr is not equal to the storage data_ptr.
        x_base = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        x = x_base[1:]
        self.assertTrue(x.is_pinned())
        self.assertTrue(x_base.is_pinned())
        self.assertNotEqual(x_base.data_ptr(), x.data_ptr())
        self.assertEqual(x_base.storage().data_ptr(), x.storage().data_ptr())
        y = torch.ones(10000000 - 1, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_copy_non_blocking_type_conversion(self):
        # [FUNCTION DOC] Checks non-blocking copy with dtype conversion: after a long GPU
        # sleep, copies a float GPU tensor into a pinned CPU tensor and then into a long GPU
        # tensor (both non-blocking, same stream) and asserts the result equals the original
        # ignoring dtype.
        a = torch.ones(1, device="cuda")
        b = torch.zeros(1, device="cpu", pin_memory=True)
        c = torch.empty(1, device="cuda", dtype=torch.long)
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        b.copy_(a, non_blocking=True)
        c.copy_(b, non_blocking=True)
        self.assertEqual(a, c, exact_dtype=False)

    @serialTest()
    def test_to_non_blocking(self):
        # [FUNCTION DOC] Verifies the to method's non_blocking semantics for CPU to GPU and GPU
        # to CPU conversion. For each destination and flag, pushes a long sleep onto the current
        # stream, converts, and asserts the stream is still busy exactly when non_blocking is
        # true, then checks the values and that the result is pinned only for a non-blocking GPU
        # to CPU copy.
        stream = torch.cuda.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            # [FUNCTION DOC] HELPER: Synchronizes, enqueues a 0.1 second GPU sleep, converts
            # tensor a to the given device with the given flag, asserts the stream query result
            # is the negation of the flag, synchronizes, compares values, and checks the result
            # is pinned only when non-blocking and destination is the CPU.
            torch.cuda.synchronize()
            # Pushes an 0.1 second spin to stream so if the copy is non blocking,
            # stream will almost surely be active when we query().
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            b = a.to(device=dst, non_blocking=non_blocking)
            self.assertEqual(stream.query(), not non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in product(("cuda", "cpu"), (True, False)):
            # Creates source on the opposite device from destination.
            src = torch.randn(
                1000000,
                device="cuda" if dst == "cpu" else "cpu",
                pin_memory=dst == "cuda",
            )
            _test_to_non_blocking(src, try_non_blocking, dst)

    def test_to_cpu_blocking_by_default(self):
        # [FUNCTION DOC] Checks that moving a GPU tensor to CPU without non_blocking blocks:
        # after a long sleep on the stream, the conversion returns with the stream already idle,
        # the values match, and the CPU result is not pinned.
        src = torch.randn(1000000, device="cuda")
        torch.cuda.synchronize()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        dst = src.to(device="cpu")
        self.assertEqual(torch.cuda.current_stream().query(), True)
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())

    def test_serialization_array_with_storage(self):
        # [FUNCTION DOC] Saves a list holding a float CUDA tensor twice, an int CUDA tensor and
        # the latter's storage to a temporary file and loads it back. Asserts equality, that
        # aliasing is preserved (editing one copy changes the duplicate), that tensor and
        # storage types are restored, and that filling the int copy updates the loaded storage.
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        q_copy[1].fill_(10)
        self.assertEqual(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    @unittest.skipIf(
        IS_WINDOWS and SM89OrLater, "preferred_blas_library not supported on Windows"
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Does not work in fbcode yet")
    @setBlasBackendsToDefaultFinally
    def test_preferred_blas_library_settings(self):
        # [FUNCTION DOC] Tests torch.backends.cuda.preferred_blas_library. Checks the default
        # backend (always cublas on CUDA; on ROCm depends on the arch list for hipblaslt
        # preference), that "default" resets to the real default, that cublas and hipblas map to
        # the cublas enum, that bad strings or non-string inputs raise RuntimeError, and that
        # env variables selecting cublaslt/hipblaslt are honored in child processes. Skipped on
        # Windows SM89+ and fbcode/sandcastle; restores backends afterward.
        def _check_default():
            # [FUNCTION DOC] HELPER: Asserts the current preferred BLAS backend equals the
            # platform default: cublas on CUDA builds; on ROCm builds cublaslt when the
            # device gcn arch matches the ROCm-version-dependent hipblaslt-preferred arch
            # list, else cublas.
            default = torch.backends.cuda.preferred_blas_library()
            if torch.version.cuda:
                # CUDA logic is easy, it's always cublas
                self.assertTrue(default == torch._C._BlasBackend.Cublas)
            else:
                # ROCm logic is less so, it's cublaslt for some Instinct, cublas for all else
                # Mirror CUDAHooks::getHipblasltPreferredArchs in CUDAHooks.cpp
                ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split(".")[:2])
                archs = ["gfx90a", "gfx942"]
                if ROCM_VERSION >= (6, 4):
                    archs.extend(["gfx1200", "gfx1201"])
                if ROCM_VERSION >= (7, 0):
                    archs.append("gfx950")
                if ROCM_VERSION >= (7, 13):
                    archs.extend(["gfx1100", "gfx1101", "gfx1151"])
                gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
                hipblaslt_preferred = any(arch in gcn_arch_name for arch in archs)
                if hipblaslt_preferred:
                    self.assertTrue(default == torch._C._BlasBackend.Cublaslt)
                else:
                    self.assertTrue(default == torch._C._BlasBackend.Cublas)

        _check_default()
        # "Default" can be set but is immediately reset internally to the actual default value.
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("default")
            != torch._C._BlasBackend.Default
        )
        _check_default()
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("cublas")
            == torch._C._BlasBackend.Cublas
        )
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("hipblas")
            == torch._C._BlasBackend.Cublas
        )
        # check bad strings
        with self.assertRaisesRegex(
            RuntimeError,
            "Unknown input value. Choose from: default, cublas, hipblas, cublaslt, hipblaslt, ck.",
        ):
            torch.backends.cuda.preferred_blas_library("unknown")
        # check bad input type
        with self.assertRaisesRegex(RuntimeError, "Unknown input value type."):
            torch.backends.cuda.preferred_blas_library(1.0)
        # check env var override
        custom_envs = [
            ({"TORCH_BLAS_PREFER_CUBLASLT": "1"}, "_BlasBackend.Cublaslt"),
            ({"TORCH_BLAS_PREFER_HIPBLASLT": "1"}, "_BlasBackend.Cublaslt"),
            ({"TORCH_BLAS_PREFER_CUBLASLT": "0"}, "_BlasBackend.Cublas"),
            ({"TORCH_BLAS_PREFER_HIPBLASLT": "0"}, "_BlasBackend.Cublas"),
        ]
        test_script = "import torch;print(torch.backends.cuda.preferred_blas_library())"
        for env_config, expected in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual(expected, r)

        # explicitly check default when no env vars are set
        if not any(
            os.environ.get(v)
            for v in ("TORCH_BLAS_PREFER_CUBLASLT", "TORCH_BLAS_PREFER_HIPBLASLT")
        ):
            torch.backends.cuda.preferred_blas_library("default")
            _check_default()

    def test_current_solver_handle(self):
        # [FUNCTION DOC] Skips if neither cuSOLVER nor hipSOLVER is available. Requests the
        # current solver handle; a RuntimeError naming a missing linalg library or the solver
        # handle skips the test, any other error is re-raised. Otherwise asserts the handle is a
        # non-zero int.
        if not has_cusolver() and not has_hipsolver():
            self.skipTest("cuSOLVER/hipSOLVER not available")
        try:
            handle = torch.cuda.current_solver_handle()
        except RuntimeError as err:
            msg = str(err)
            if (
                "CurrentCUDASolverDnHandle" in msg
                or "libtorch_cuda_linalg.so" in msg
                or "libtorch_hip_linalg.so" in msg
            ):
                self.skipTest(f"cuSOLVER handle is unavailable in this build: {err}")
            raise
        self.assertIsInstance(handle, int)
        self.assertNotEqual(handle, 0)

    @unittest.skipIf(
        IS_WINDOWS and SM89OrLater, "preferred_blas_library not supported on Windows"
    )
    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled for async")
    @serialTest()
    @blas_library_context("cublas")
    def test_cublas_workspace_explicit_allocation(self):
        # [FUNCTION DOC] Checks cuBLAS workspace allocation sizes using the cublas backend.
        # Measures bytes newly allocated by a small matmul after clearing workspaces and asserts
        # the size is within 512 KiB of the expected default (platform and arch dependent), then
        # of an explicit 3 MiB size set through the API, that a negative size raises
        # RuntimeError, and that resetting restores the default.
        a = torch.randn(7, 7, device="cuda", requires_grad=False)
        if torch.version.hip:
            default_workspace_size = 1024 * 32 * 1024  # :1024:32  32MiB
            # different size (128 MiB) expected on MI300 GPU
            gcn_arch = str(
                torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
            )
            if gcn_arch in ["gfx942", "gfx950", "gfx1250"]:
                default_workspace_size = 1024 * 128 * 1024  # :1024:128
        else:
            default_workspace_size = (
                4096 * 2 * 1024 + 16 * 8 * 1024
            )  # :4096:2:16:8  8MiB
            # different size (32 MiB) expected on Hopper/Blackwell GPU
            if torch.cuda.get_device_capability()[0] in (9, 10, 12):
                default_workspace_size = 4096 * 8 * 1024

        def check_workspace_size(inp):
            # [FUNCTION DOC] HELPER: Clears cuBLAS workspaces, runs a no-grad matmul of the
            # input with itself and returns the increase in all-pool allocated bytes.
            torch._C._cuda_clearCublasWorkspaces()
            start = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            with torch.no_grad():
                torch.matmul(inp, inp)
            finish = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            return finish - start

        # check default
        self.assertLess(abs(check_workspace_size(a) - default_workspace_size), 524288)

        # check explicit size via API
        explicit_size = 3072 * 1024
        torch.backends.cuda.cublas_workspace_size(explicit_size)
        self.assertLess(abs(check_workspace_size(a) - explicit_size), 524288)

        # check invalid size rejected
        with self.assertRaisesRegex(
            RuntimeError, "cublas workspace size must be non-negative"
        ):
            torch.backends.cuda.cublas_workspace_size(-1)

        # restore default
        torch._C._cuda_resetCublasWorkspaceSize()
        self.assertLess(abs(check_workspace_size(a) - default_workspace_size), 524288)

        torch._C._cuda_clearCublasWorkspaces()

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled for async")
    @unittest.skipIf(IS_FBCODE, "not enabled by default on fbcode")
    @unittest.skipIf(TEST_WITH_ROCM, "not enabled by default on rocm")
    @setBlasBackendsToDefaultFinally
    @serialTest()
    def test_cublas_unified_workspace(self):
        # [FUNCTION DOC] With the cublas backend a matmul creates a workspace; then switches to
        # cublaslt and runs another matmul. Asserts peak allocated memory is unchanged, showing
        # the two backends share a unified workspace. Skipped for async allocator, fbcode and
        # ROCm; restores backends afterward.
        torch.cuda.reset_peak_memory_stats()
        # We run a cuBLAS matmul, guaranteeing a workspace present on the current stream
        torch.backends.cuda.preferred_blas_library("cublas")
        x = torch.randn(128, 128, device="cuda")
        torch.matmul(x, x)
        # Stash the maximum memory allocated after running matmuls
        warmed_alloc = torch.cuda.max_memory_allocated()
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.matmul(x, x)
        lt_alloc = torch.cuda.max_memory_allocated()
        # With unified workspaces, the peak memory allocation should not increase after
        # switching to Lt, otherwise the temporary allocation would bump the peak
        self.assertEqual(warmed_alloc, lt_alloc)

    @setBlasBackendsToDefaultFinally
    def test_cublas_workspace_size_api(self):
        # [FUNCTION DOC] Tests the cublas and cublaslt workspace size getter/setter API: default
        # sizes are positive, setting a new size returns it and the getter reflects it, and
        # negative sizes raise RuntimeError with "must be non-negative" messages.
        # Test getter returns a positive default
        original_size = torch.backends.cuda.cublas_workspace_size()
        self.assertGreater(original_size, 0)

        original_lt_size = torch.backends.cuda.cublaslt_workspace_size()
        self.assertGreater(original_lt_size, 0)

        # Test setter changes the value and getter reflects it
        new_size = 64 * 1024 * 1024  # 64 MiB
        result = torch.backends.cuda.cublas_workspace_size(new_size)
        self.assertEqual(result, new_size)
        self.assertEqual(torch.backends.cuda.cublas_workspace_size(), new_size)

        new_lt_size = 2 * 1024 * 1024  # 2 MiB
        result_lt = torch.backends.cuda.cublaslt_workspace_size(new_lt_size)
        self.assertEqual(result_lt, new_lt_size)
        self.assertEqual(torch.backends.cuda.cublaslt_workspace_size(), new_lt_size)

        # Test validation rejects negative values
        with self.assertRaisesRegex(
            RuntimeError, "cublas workspace size must be non-negative"
        ):
            torch.backends.cuda.cublas_workspace_size(-1)
        with self.assertRaisesRegex(
            RuntimeError, "cublaslt workspace size must be non-negative"
        ):
            torch.backends.cuda.cublaslt_workspace_size(-1)

    @setBlasBackendsToDefaultFinally
    def test_blas_workspace_size_api(self):
        # [FUNCTION DOC] Tests the backend-dispatching blas_workspace_size API. With no backend
        # it must equal the cublaslt size when the preferred backend is cublaslt, else the
        # cublas size; explicit string or enum backends select the matching getter; setting
        # through the dispatcher updates the cublas and cublaslt sizes; the ck backend, an
        # unknown string and a non-string type each raise RuntimeError.
        # Dispatches to the correct backend based on preferred_blas_library()
        pref = torch.backends.cuda.preferred_blas_library()
        if pref == torch._C._BlasBackend.Cublaslt:
            expected = torch.backends.cuda.cublaslt_workspace_size()
        else:
            # Default and Cublas both map to cuBLAS
            expected = torch.backends.cuda.cublas_workspace_size()
        self.assertEqual(torch.backends.cuda.blas_workspace_size(), expected)

        # Explicit backend= parameter
        self.assertEqual(
            torch.backends.cuda.blas_workspace_size(backend="cublas"),
            torch.backends.cuda.cublas_workspace_size(),
        )
        self.assertEqual(
            torch.backends.cuda.blas_workspace_size(backend="cublaslt"),
            torch.backends.cuda.cublaslt_workspace_size(),
        )
        self.assertEqual(
            torch.backends.cuda.blas_workspace_size(
                backend=torch._C._BlasBackend.Cublas
            ),
            torch.backends.cuda.cublas_workspace_size(),
        )

        # Setting via the dispatcher updates the underlying function
        new_size = 16 * 1024 * 1024
        torch.backends.cuda.blas_workspace_size(new_size, backend="cublas")
        self.assertEqual(torch.backends.cuda.cublas_workspace_size(), new_size)

        torch.backends.cuda.blas_workspace_size(new_size, backend="cublaslt")
        self.assertEqual(torch.backends.cuda.cublaslt_workspace_size(), new_size)

        # CK backend has no workspace
        with self.assertRaisesRegex(
            RuntimeError, "CK backend does not use a workspace."
        ):
            torch.backends.cuda.blas_workspace_size(backend="ck")

        # Invalid string
        with self.assertRaisesRegex(
            RuntimeError,
            "Unknown backend string. Choose from: default, cublas, hipblas, cublaslt, hipblaslt, ck.",
        ):
            torch.backends.cuda.blas_workspace_size(backend="invalid")

        # Invalid type
        with self.assertRaisesRegex(RuntimeError, "Unknown backend type."):
            torch.backends.cuda.blas_workspace_size(backend=42)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled for async")
    @setBlasBackendsToDefaultFinally
    def test_cublas_workspace_lazy_reallocation(self):
        # [FUNCTION DOC] Checks that changing the cublas workspace size is applied lazily: after
        # a first matmul, raising the size by 32 MiB leaves allocated bytes unchanged until the
        # next matmul, which then allocates more. Uses the cublas backend, skipped under the
        # async allocator, and restores backends afterward.
        torch.backends.cuda.preferred_blas_library("cublas")

        original_size = torch.backends.cuda.cublas_workspace_size()
        torch._C._cuda_clearCublasWorkspaces()

        # Trigger initial allocation with matmul
        a = torch.randn(7, 7, device="cuda", requires_grad=False)
        with torch.no_grad():
            torch.matmul(a, a)

        mem_after_first = torch.cuda.memory_stats()["active_bytes.all.allocated"]

        # Increase workspace size
        bigger_size = original_size + 32 * 1024 * 1024  # +32 MiB
        torch.backends.cuda.cublas_workspace_size(bigger_size)

        # No immediate memory change (lazy reallocation)
        mem_after_set = torch.cuda.memory_stats()["active_bytes.all.allocated"]
        self.assertEqual(mem_after_first, mem_after_set)

        # Next matmul triggers reallocation
        with torch.no_grad():
            torch.matmul(a, a)

        mem_after_realloc = torch.cuda.memory_stats()["active_bytes.all.allocated"]
        self.assertGreater(mem_after_realloc, mem_after_first)

    @recover_orig_fp32_precision
    def test_cublas_allow_tf32_get_set(self):
        # [FUNCTION DOC] Checks the cuBLAS TF32 flag getter/setter agreement. If the
        # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE environment variable is a truthy integer, only
        # asserts allow_tf32 is on and returns. Otherwise the low-level getter must equal the
        # Python flag, and flipping the flag must be reflected by the getter; the flag is
        # then restored.
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        if skip_tf32_cublas:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            return

        orig = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        torch.backends.cuda.matmul.allow_tf32 = orig

    def test_cublaslt_prefer_grouped_gemm_get_set(self):
        # [FUNCTION DOC] Checks the cuBLASLt grouped-gemm preference flag: the low-level getter
        # equals the Python property, flipping the property is reflected by the getter, then the
        # original value is restored.
        orig = torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm
        self.assertEqual(torch._C._get_cublaslt_prefer_grouped_gemm(), orig)
        torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm = not orig
        self.assertEqual(torch._C._get_cublaslt_prefer_grouped_gemm(), not orig)
        torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm = orig

    @recover_orig_fp32_precision
    def test_float32_matmul_precision_get_set(self):
        # [FUNCTION DOC] Tests set/get of float32 matmul precision. First checks the environment
        # is respected: without the TF32 override variable the TF32 flag must be off and
        # precision "highest"; with it, the flag must be on. Then medium and high must turn the
        # TF32 flag on, highest must turn it off, and the original precision is restored.
        orig = torch.get_float32_matmul_precision()
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        # this is really just checking that the environment variable is respected during testing
        # and not overwritten by another function that doesn't revert it to the initial value
        if not skip_tf32_cublas:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        else:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision(orig)

    def test_cublas_allow_fp16_reduced_precision_reduction_get_set(self):
        # [FUNCTION DOC] Checks the fp16 reduced-precision-reduction setting. The low-level
        # getter returns a pair of reduction and split-k flags equal to the Python properties;
        # setting a bool flips the first and leaves split-k True; setting a pair of two False
        # values is accepted; a pair that enables reduction but disables split-k raises
        # RuntimeError mentioning allow_splitk. Original values are restored.
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        orig_splitk = (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k
        )
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (orig, orig_splitk),
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (not orig, True),
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            False,
            False,
        )
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (False, False),
        )
        with self.assertRaisesRegex(RuntimeError, "allow_splitk=False"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                True,
                False,
            )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            orig,
            orig_splitk,
        )

    def test_cublas_allow_bf16_reduced_precision_reduction_get_set(self):
        # [FUNCTION DOC] Same checks as the fp16 variant but for the bf16 reduced-precision
        # reduction setting and its split-k flag: getter pair matches the properties, a bool
        # flips the first flag, a pair of False values is accepted, enabling reduction without
        # split-k raises RuntimeError, and the original pair is restored.
        orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_splitk = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
        )
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (orig, orig_splitk),
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (not orig, True),
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False,
            False,
        )
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (False, False),
        )
        with self.assertRaisesRegex(RuntimeError, "allow_splitk=False"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                True,
                False,
            )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            orig,
            orig_splitk,
        )

    def test_cublas_allow_fp16_accumulation_get_set(self):
        # [FUNCTION DOC] Checks the fp16 accumulation flag: low-level getter equals the Python
        # property, flipping the property is reflected in the getter, and the original value
        # is restored.
        orig = torch.backends.cuda.matmul.allow_fp16_accumulation
        self.assertEqual(torch._C._get_cublas_allow_fp16_accumulation(), orig)
        torch.backends.cuda.matmul.allow_fp16_accumulation = not orig
        self.assertEqual(torch._C._get_cublas_allow_fp16_accumulation(), not orig)
        torch.backends.cuda.matmul.allow_fp16_accumulation = orig

    def test_cudnn_allow_tf32_get_set(self):
        # [FUNCTION DOC] Uses the cudnn flags context manager to set allow_tf32 to False and
        # then True and asserts the cudnn allow_tf32 property reflects each setting inside
        # the block.
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_cudnn_depthwise_kernel_get_set(self):
        # [FUNCTION DOC] Tests the cudnn depthwise kernel setting: the default is "auto"; each
        # of auto, cudnn and native can be set through the flags context manager and is visible
        # inside; an invalid string passed to the low-level setter raises RuntimeError; and the
        # default "auto" is restored after the context exits.
        self.assertEqual(torch.backends.cudnn.depthwise_kernel, "auto")

        # Test all valid values via the flags() context manager
        for mode in ("auto", "cudnn", "native"):
            with torch.backends.cudnn.flags(
                enabled=None,
                benchmark=None,
                deterministic=None,
                allow_tf32=None,
                depthwise_kernel=mode,
            ):
                self.assertEqual(torch.backends.cudnn.depthwise_kernel, mode)

        # Invalid value should raise
        with self.assertRaises(RuntimeError):
            torch._C._set_cudnn_depthwise_kernel("invalid")

        # Verify the flags() context manager restores the previous value
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            deterministic=None,
            allow_tf32=None,
            depthwise_kernel="native",
        ):
            self.assertEqual(torch.backends.cudnn.depthwise_kernel, "native")
        self.assertEqual(torch.backends.cudnn.depthwise_kernel, "auto")

    @recover_orig_fp32_precision
    def test_fp32_precision_with_tf32(self):
        # [FUNCTION DOC] Checks that the legacy cudnn allow_tf32 flag maps onto the newer
        # fp32_precision settings: with allow_tf32 True the conv and rnn precisions read "tf32",
        # and with it False they read "none". Precision state is recovered after.
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=True,
            fp32_precision="none",
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")

        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=False,
            fp32_precision="none",
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "none")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "none")

    @recover_orig_fp32_precision
    @serialTest()
    def test_fp32_precision_with_float32_matmul_precision(self):
        # [FUNCTION DOC] Checks that set_float32_matmul_precision maps onto the CUDA matmul
        # fp32_precision setting: highest gives ieee, high and medium both give tf32. Runs
        # serially and restores the original precision state.
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        torch.set_float32_matmul_precision("high")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        torch.set_float32_matmul_precision("medium")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")

    @recover_orig_fp32_precision
    @serialTest()
    def test_invalid_status_for_legacy_api(self):
        # [FUNCTION DOC] Checks that mixing the legacy and new fp32 precision APIs makes the
        # legacy getters raise RuntimeError saying legacy and new APIs were mixed: reading cudnn
        # allow_tf32 after setting conv and rnn precisions differently, reading the float32
        # matmul precision after setting the new matmul precision to tf32, and (not on ROCm)
        # reading the matmul allow_tf32 flag.
        torch.backends.cudnn.conv.fp32_precision = "none"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
            print(torch.backends.cudnn.allow_tf32)

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
            print(torch.get_float32_matmul_precision())

        if not TEST_WITH_ROCM:
            with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
                print(torch.backends.cuda.matmul.allow_tf32)

    def test_type_conversions(self):
        # [FUNCTION DOC] Checks tensor and storage type conversions between CPU and CUDA:
        # float, cuda double, cuda float, back to CPU and to int yield the expected legacy
        # tensor classes, and the same chain on the underlying storage yields the matching
        # storage classes.
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().double(), torch.cuda.DoubleTensor)
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().double(), torch.cuda.DoubleStorage)
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    @unittest.skip("was disabled due to not enough memory, but actually it always fail")
    def test_arithmetic_large_tensor(self):
        # [FUNCTION DOC] Skipped test (always fails and needs too much memory). Intended to
        # exercise in-place fill, add, subtract, multiply and divide on a tensor with 2 to the
        # 30 elements and check the sums.
        x = torch.empty(2**30, device="cuda")

        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        x += 1
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    def test_gather_bool(self):
        # [FUNCTION DOC] Checks gather on a boolean CUDA tensor along dimension 1 with an index
        # tensor and compares to the expected boolean result.
        t = torch.tensor([[False, True], [True, True]], device="cuda")
        self.assertEqual(
            torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device="cuda")),
            torch.tensor([[False, False], [True, True]], device="cuda"),
        )

    def test_torch_manual_seed_seeds_cuda_devices(self):
        # [FUNCTION DOC] Checks that the global manual seed also seeds CUDA: with the RNG state
        # frozen, seeding with 2 makes initial_seed report 2, uniform fill results are
        # reproducible after re-seeding with the same value, and the seed is unchanged.
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_manual_seed(self):
        # [FUNCTION DOC] Checks torch.cuda.manual_seed: after seeding with 2, initial_seed is 2;
        # both uniform fill and a Bernoulli draw are identical when re-seeded with the same
        # value. The RNG state is frozen around the test.
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_specify_improper_device_name(self):
        # [FUNCTION DOC] Saves a parameter to a temporary file, then calls load with a malformed
        # device string and expects RuntimeError mentioning an invalid device string.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "tempfile.pt")
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                torch.save(
                    [torch.nn.Parameter(torch.randn(10, 10))],
                    fname,
                    _use_new_zipfile_serialization=True,
                )
                torch.load(fname, "cuda0")

    def test_get_device_index(self):
        # [FUNCTION DOC] Tests the cuda device index helper: a malformed device string raises
        # RuntimeError (invalid device string) and a CPU device object raises ValueError
        # (expected a cuda device), both with the optional flag set.
        from torch.cuda._utils import _get_device_index

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index("cuda0", optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            cpu_device = torch.device("cpu")
            _get_device_index(cpu_device, optional=True)

    def test_serialization_array_with_empty(self):
        # [FUNCTION DOC] Round-trips a list holding a normal CUDA tensor and an empty legacy
        # CUDA float tensor through save and load, and checks each loaded item equals the
        # original, has the same type and lives on the same device.
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        # [FUNCTION DOC] Basic stream test (not run on non-default stream mode). Checks the
        # current stream is the default stream with handle 0, a new user stream differs and has
        # a non-zero handle, entering the stream context makes it current, an idle user stream
        # queries as done, and the default stream can be synchronized.
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    def test_stream_invalid_device_index(self):
        # [FUNCTION DOC] Regression test: constructing a Stream with out-of-range device indices
        # (negative values, the device count and larger) and reading its handle must raise
        # RuntimeError about the device index rather than reading outside the pool.
        # Regression test for #184034: constructing a CUDAStream with an
        # out-of-range device_index must not OOB-index the static stream pool.
        num_devices = torch.cuda.device_count()
        for di in (-2, -8, num_devices, num_devices + 16, 128):
            with self.assertRaisesRegex(RuntimeError, "Device index (must be|value)"):
                s = torch.cuda.Stream(stream_id=3, device_index=di, device_type=1)
                _ = s.cuda_stream

    def test_stream_event_repr(self):
        # [FUNCTION DOC] Checks string representations: the current stream repr contains the
        # torch.cuda.Stream name, and an event repr contains torch.cuda.Event both before and
        # after being recorded on a stream.
        s = torch.cuda.current_stream()
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.cuda.Event" in e.__repr__())

    def test_cuda_stream_protocol(self):
        # [FUNCTION DOC] Checks the stream export protocol: a Stream has the protocol method
        # returning a tuple of length 2, protocol version 0 and its stream handle; an
        # ExternalStream built from that handle returns version 0 and its own handle.
        stream = torch.cuda.Stream()

        self.assertTrue(hasattr(stream, "__cuda_stream__"))

        result = stream.__cuda_stream__()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 0)  # Protocol version
        self.assertEqual(result[1], stream.cuda_stream)  # Stream handle

        external_stream = torch.cuda.ExternalStream(stream.cuda_stream)
        external_result = external_stream.__cuda_stream__()

        self.assertEqual(external_result[0], 0)
        self.assertEqual(external_result[1], external_stream.cuda_stream)

    def test_external_stream_wraps_null_handle(self):
        # [FUNCTION DOC] Regression test: an ExternalStream created with handle 0 wraps the null
        # (default) stream, as does get_stream_from_external with handle 0. Both report handle
        # 0, match the default stream's handle, and the same device as the default stream.
        # Regression for https://github.com/pytorch/pytorch/issues/182960
        # ExternalStream(0) must wrap the NULL stream (handle 0x0), not return
        # a pooled stream, and must behave identically to
        # torch.cuda.get_stream_from_external(0).
        device = torch.cuda.current_device()
        external_stream = torch.cuda.ExternalStream(0, device=device)
        from_get_stream_from_external = torch.cuda.get_stream_from_external(
            0, device=device
        )
        default = torch.cuda.default_stream(device)

        self.assertEqual(external_stream.cuda_stream, 0)
        self.assertEqual(from_get_stream_from_external.cuda_stream, 0)
        self.assertEqual(external_stream.cuda_stream, default.cuda_stream)

        self.assertEqual(external_stream.device, from_get_stream_from_external.device)
        self.assertEqual(external_stream.device, default.device)

    def test_events(self):
        # [FUNCTION DOC] Tests event behavior: a fresh event queries as complete; after
        # recording start and end events around a GPU sleep, the end event is incomplete until
        # synchronized and the elapsed time is positive. A new event has zero cuda_event and
        # event_id, which both become non-zero and equal after recording.
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

        event = torch.cuda.Event(enable_timing=True)
        self.assertEqual(event.cuda_event, 0)
        self.assertEqual(event.event_id, 0)

        event.record()
        self.assertNotEqual(event.cuda_event, 0)
        self.assertNotEqual(event.event_id, 0)
        self.assertEqual(event.cuda_event, event.event_id)

    def test_events_elapsedtime(self):
        # [FUNCTION DOC] Checks elapsed_time errors: events created without timing raise
        # ValueError about enable_timing (also the default), and timing-enabled events that were
        # never recorded raise ValueError saying both must be recorded.
        event1 = torch.cuda.Event(enable_timing=False)
        event2 = torch.cuda.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError, "Both events must be recorded before calculating elapsed time"
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.cuda.Event()
        event2 = torch.cuda.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    def test_generic_stream_event(self):
        # [FUNCTION DOC] Tests the generic torch.Stream and torch.Event with the cuda device: a
        # generic stream's device index is the current device, a cuda Stream built from its ids
        # shares the stream id and is an instance of the generic class; events record,
        # synchronize and query, give distinct ids and positive elapsed time, copies on the
        # stream work, and cuda Event is a subclass of the generic Event.
        stream = torch.Stream("cuda")
        self.assertEqual(stream.device_index, torch.cuda.current_device())
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertIsInstance(cuda_stream, torch.Stream)
        self.assertTrue(issubclass(type(cuda_stream), torch.Stream))
        self.assertTrue(torch.Stream in type(cuda_stream).mro())
        self.assertEqual(stream.stream_id, cuda_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.cuda.current_stream().stream_id)

        event1 = torch.Event("cuda", enable_timing=True)
        event2 = torch.Event("cuda", enable_timing=True)
        self.assertEqual(event1.event_id, 0)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.cuda.stream(cuda_stream):
            a_cuda = a.to("cuda", non_blocking=True)
            b_cuda = b.to("cuda", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.cuda.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_cuda = a_cuda + b_cuda
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_cuda.cpu(), a + b)
        self.assertTrue(event1.elapsed_time(event2) > 0)
        cuda_event = torch.cuda.Event()
        self.assertIsInstance(cuda_event, torch.Event)
        self.assertTrue(issubclass(type(cuda_event), torch.Event))
        self.assertTrue(torch.Event in type(cuda_event).mro())

    @unittest.skip("Fails with Triton 3.7")
    def test_stream_event_compatibility(self):
        # [FUNCTION DOC] Skipped (fails with Triton 3.7). Intended to check interoperability
        # between the accelerator stream API and cuda streams/events: setting streams updates
        # current stream ids and native handles, an invalid device index raises RuntimeError,
        # and mixing cuda and generic events in elapsed_time, record or wait raises RuntimeError
        # with type-expectation messages.
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s1.stream_id)
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s1.cuda_stream
        )
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s1.native_handle
        )
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s2.stream_id)
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s2.cuda_stream
        )
        self.assertEqual(
            torch.accelerator.current_stream().native_handle, s2.native_handle
        )
        with self.assertRaisesRegex(
            RuntimeError, "Device index value .* is out of index range"
        ):
            torch.accelerator.current_stream(torch.accelerator.device_count())
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.Event(enable_timing=True)
        s3 = torch.Stream()
        with s3:
            self.assertEqual(torch.accelerator.current_stream(), s3)
            e1.record(s3)
            a = torch.randn(1000)
            a_cuda = a.to("cuda")
            del a_cuda
            e2.record(s3)
            e2.synchronize()
            self.assertGreater(e1.elapsed_time(e2), 0)
            e3.record(s3)
        with self.assertRaisesRegex(
            RuntimeError, "expected other to be a torch.cuda.Event object"
        ):
            e1.elapsed_time(e3)
        with self.assertRaisesRegex(
            RuntimeError, "expected other to be a torch.Event object"
        ):
            e3.elapsed_time(e1)
        with self.assertRaisesRegex(
            RuntimeError, "expected event to be a torch.Event object"
        ):
            s3.record_event(e1)
        with self.assertRaisesRegex(
            RuntimeError,
            "expected stream to be a torch.Stream or torch.cuda.Stream object",
        ):
            e2.record(e2)
        with self.assertRaisesRegex(
            RuntimeError,
            "expected stream to be a torch.Stream or torch.cuda.Stream object",
        ):
            e2.wait(e2)

    def test_record_stream(self):
        # [FUNCTION DOC] Tests record_stream semantics with the caching allocator. A pinned
        # tensor is copied to the GPU on a side stream, the main stream is told to wait and the
        # tensor is marked as used by it, then a delayed copy into a result runs. A new
        # allocation on the side stream must not reuse the pointer too early, the result must
        # hold the values, and (native allocator only) after synchronizing the block is reused.
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.cuda.FloatTensor(t.size())
        stream = torch.cuda.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            # [FUNCTION DOC] HELPER: Performs a non-blocking host-to-device copy on the side
            # stream, remembers the device pointer in the enclosing list, makes the current
            # stream wait for the side stream, records the current stream on the tensor, sleeps
            # 50 ms on the GPU and copies the tensor into the result.
            with torch.cuda.stream(stream):
                tmp = t.cuda(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.cuda.current_stream().wait_stream(stream)
            tmp.record_stream(torch.cuda.current_stream())
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with torch.cuda.stream(stream):
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation reused to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        if not TEST_CUDAMALLOCASYNC:
            # In the native allocator, we expect "tmp"'s side-stream-tagged block will be reused
            # in that side stream after result.copy_(tmp) in the main stream finishes.
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(stream):
                tmp3 = torch.cuda.FloatTensor(t.size())
                self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not reused")

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/120318")
    def test_record_stream_on_shifted_view(self):
        # [FUNCTION DOC] Regression test (issue 27366): allocates a base tensor on one stream,
        # records a second busy stream on a shifted view (nonzero storage offset), deletes both
        # and checks a new same-size allocation on the first stream does not reuse the original
        # block pointer. Skipped on ROCm.
        # See issue #27366

        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        stream_alloc = torch.cuda.Stream()
        with torch.cuda.stream(stream_alloc):
            base = torch.cuda.FloatTensor([10, 10])

        # Record another stream on a shifted view tensor.
        view = base[5:]
        self.assertTrue(view.storage_offset() > 0)

        stream_record = torch.cuda.Stream()
        with torch.cuda.stream(stream_record):
            torch.cuda._sleep(int(50 * get_cycles_per_ms()))

        view.record_stream(stream_record)

        # Delete those tensors to make the block free soon.
        data_ptr = base.data_ptr()
        del base, view

        # A new tensor should not be allocated to the block above.
        stream_alloc.synchronize()

        with torch.cuda.stream(stream_alloc):
            try_realloc = torch.cuda.FloatTensor([10, 10])

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)

    def test_device_context_manager(self):
        # [FUNCTION DOC] Tests the accelerator device_index context manager: with None the
        # current device is unchanged, with index 0 it is 0 inside, and the previous device is
        # restored after each block.
        prev_device = torch.cuda.current_device()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.cuda.current_device(), prev_device)
        self.assertEqual(torch.cuda.current_device(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_device(), prev_device)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_multi_device_context_manager(self):
        # [FUNCTION DOC] Multi-GPU test: sets device 0, enters the accelerator device_index
        # context for device 1 and checks current device is 1 inside and 0 afterwards.
        src_device = 0
        dst_device = 1
        torch.cuda.set_device(src_device)
        with torch.accelerator.device_index(dst_device):
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.current_device(), src_device)

    def test_stream_context_manager(self):
        # [FUNCTION DOC] Checks that using a Stream as a context manager makes it the current
        # stream and restores the previous current stream on exit.
        prev_stream = torch.cuda.current_stream()
        with torch.cuda.Stream() as stream:
            self.assertEqual(stream, torch.cuda.current_stream())
        self.assertEqual(prev_stream, torch.cuda.current_stream())

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_multi_device_stream_context_manager(self):
        # [FUNCTION DOC] Multi-GPU test: creates a stream on device 1 while device 0 is current;
        # inside the context checks current device and stream are the new ones and the device 0
        # stream is unchanged; after exit checks device 0 is current again and both previous
        # streams are restored.
        src_device = 0
        dst_device = 1
        torch.cuda.set_device(src_device)
        src_prev_stream = torch.cuda.current_stream(src_device)
        dst_prev_stream = torch.cuda.current_stream(dst_device)
        with torch.cuda.Stream(dst_device) as dst_stream:
            self.assertEqual(dst_device, torch.cuda.current_device())
            self.assertEqual(dst_stream, torch.cuda.current_stream())
            self.assertEqual(src_prev_stream, torch.cuda.current_stream(src_device))
        self.assertEqual(src_device, torch.cuda.current_device())
        self.assertEqual(src_prev_stream, torch.cuda.current_stream())
        self.assertEqual(dst_prev_stream, torch.cuda.current_stream(dst_device))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_event_elapsed_time_restores_device_on_error(self):
        # [FUNCTION DOC] Multi-GPU regression test: records generic events on streams of devices
        # 0 and 1, then calls elapsed_time across devices, which must raise RuntimeError, and
        # asserts the current device is still 0 afterwards (the device guard restored it on the
        # error path).
        # The device-generic event path (torch.Event -> CUDAGuardImpl::elapsedTime)
        # temporarily switches to the event's device to call cudaEventElapsedTime.
        # A cross-device elapsed_time raises inside that switched region; the
        # current device must still be restored on the throwing path. Regression
        # test for the RAII restore in c10::cuda::impl::CUDAGuardImpl.
        torch.cuda.set_device(0)
        e0 = torch.Event(enable_timing=True)
        e1 = torch.Event(enable_timing=True)
        e0.record(torch.Stream(0))
        e1.record(torch.Stream(1))
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        self.assertEqual(torch.cuda.current_device(), 0)

        # device_index of elapsedTime is e1's device (1); the cross-device call
        # switches to device 1 and then raises.
        with self.assertRaises(RuntimeError):
            e1.elapsed_time(e0)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_noncontiguous_pinned_memory(self):
        # [FUNCTION DOC] Regression for issue 3266: checks that a non-contiguous (transposed)
        # tensor equals its pinned copy.
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        # [FUNCTION DOC] Tests the pinned host allocator cache. A freed pinned allocation of one
        # element must be reused (same pointer) by the next one; but when a pending non-blocking
        # copy to the GPU, delayed by a 1 s sleep, still uses the block, a new pinned allocation
        # must get a different pointer, and the copy result is still 1.
        cycles_per_ms = get_cycles_per_ms()

        # check that allocations are reused after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, msg="allocation not reused")

        # check that the allocation is not reused if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg="allocation reused too soon")
        self.assertEqual(list(gpu_tensor), [1])

    def test_caching_allocator_record_stream_oom(self):
        # [FUNCTION DOC] Regression test (issue 19219): on a side stream, repeatedly allocates
        # 40M-element tensors on the main stream, accumulates them on the side stream and
        # marks them with record_stream before freeing, so reuse is delayed. Passes if
        # allocation retries free the delayed blocks and no out-of-memory occurs; finally
        # empties the cache.
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cuda_malloc_retry. see issue #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device="cuda")

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device="cuda")
            with torch.cuda.stream(stream):
                y += x
            # delays reuse of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.cuda.empty_cache()

    # Tests for historic illegal memory access, see #17040.
    def test_reduction_gpu_memory_accessing(self):
        # [FUNCTION DOC] Smoke test for a historic illegal memory access (issue 17040): sums a
        # 512 by 8 tensor of ones over dimension 0 on the GPU. Passes if no error occurs.
        x = torch.ones(512, 8, dtype=torch.float32, device="cuda")
        torch.sum(x, 0)

    def test_sum_fp16(self):
        # [FUNCTION DOC] Tests half-precision sum: zeros, 65504 and 65536 ones (the latter two
        # with float32 accumulate dtype where the float16 result would overflow), and random
        # sparse data of two shapes compared against the CPU sum, including a reduction over
        # dimensions 0 and 2.
        x = torch.zeros(10, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        x = torch.ones(65504, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 65504)
        self.assertEqual(x.sum(dtype=torch.float32), 65504)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    def test_mean_fp16(self):
        # [FUNCTION DOC] Checks that the mean of 65536 float16 ones on the GPU equals 1, both
        # with the default and with float32 dtype argument.
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(dtype=torch.float32), 1)

    def test_prod_large(self):
        # [FUNCTION DOC] Checks prod over 240000 ones equals 1 for float32 and for complex float
        # and double inputs (multiplied by imaginary unit, so the product is checked for the
        # global reduction path with a non-zero identity).
        # tests global reduction (should_global_reduce = true) in case of non-zero identity element
        x = torch.ones(240000, device="cuda", dtype=torch.float32)
        self.assertEqual(x.prod(), 1)

        # test for complex types. Note 240k is divisible by 4
        for dtype in [torch.cfloat, torch.cdouble]:
            x = torch.ones(240000, device="cuda", dtype=dtype) * (0 + 1j)
            self.assertEqual(x.prod(), 1)

    def test_multinomial_ext(self):
        # [FUNCTION DOC] Regression tests for multinomial corner cases (issues 4858, 13867):
        # sampling 1000 draws with replacement from a distribution with many zero entries must
        # never pick a zero-probability index; a two-column distribution with all mass in column
        # one must always sample index one; and a million tiny probabilities must never sample a
        # zero-probability index.
        # Test two corner cases from older PyTorch (Issue #4858)
        freqs = torch.cuda.FloatTensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.03178183361887932,
                0.027680952101945877,
                0.033176131546497345,
                0.046052902936935425,
                0.07742464542388916,
                0.11543981730937958,
                0.14148041605949402,
                0.15784293413162231,
                0.13180233538150787,
                0.08271478116512299,
                0.049702685326337814,
                0.027557924389839172,
                0.018125897273421288,
                0.011851548217236996,
                0.010252203792333603,
                0.007422595750540495,
                0.005372154992073774,
                0.0045109698548913,
                0.0036087757907807827,
                0.0035267581697553396,
                0.0018864056328311563,
                0.0024605290964245796,
                0.0022964938543736935,
                0.0018453967059031129,
                0.0010662291897460818,
                0.0009842115687206388,
                0.00045109697384759784,
                0.0007791675161570311,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00012302644609007984,
                0.0,
                0.00012302644609007984,
                4.100881778867915e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        torch.cuda.manual_seed(11042)
        sample = torch.multinomial(freqs, 1000, True)
        self.assertNotEqual(freqs[sample].min(), 0)

        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        p[:, 1] = 1
        torch.cuda.manual_seed(5214)
        r = torch.multinomial(p, 1)
        self.assertNotEqual(r.min().item(), 0)

        # test corner case from Issue #13867
        torch.cuda.manual_seed(33)
        probs = torch.randn(1000000, device="cuda").clamp(min=0) * 3e-5
        samples = probs.multinomial(1000000, replacement=True)
        self.assertGreater(probs[samples].min().item(), 0)

    def _spawn_test_multinomial_invalid_probs_cuda(self, probs):
        # [FUNCTION DOC] HELPER: Runs, in a subprocess with a 10 s timeout (killed on timeout),
        # a script that calls multinomial on the GPU with the given invalid probability list and
        # expects a device-side assertion failure. Then asserts the subprocess stdout or stderr
        # contains one of several CUDA or ROCm assertion messages.
        import subprocess

        try:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"""\
import sys
import torch
from torch import inf, nan
try:
    with torch.random.fork_rng(devices=[0]):
        torch.multinomial(torch.tensor({probs}).to('cuda'), 2, replacement=True)
        torch.cuda.synchronize()
    sys.exit(-1) # Should not be reached
except RuntimeError as e:
    sys.exit(-2)
""",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            out, err = p.communicate(timeout=10)
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        expected_messages = [
            "device-side assert triggered",  # CUDA
            "Assertion",  # CUDA
            "HSA_STATUS_ERROR_EXCEPTION",  # ROCm with TORCH_USE_HIP_DSA
            "Device-side assertion",  # ROCm with TORCH_USE_HIP_DSA
            # ROCm without TORCH_USE_HIP_DSA returns a launch failure instead
            # of a proper device-side assertion, but still catches the error
            "hipErrorLaunchFailure",
            "unspecified launch failure",
        ]
        self.assertTrue(any(msg in out or msg in err for msg in expected_messages))

    @unittest.skipIf(
        not TEST_CUDAMALLOCASYNC, "requires the cudaMallocAsync allocator backend"
    )
    @serialTest()
    def test_cudamallocasync_trim_on_oom_retry(self):
        # [FUNCTION DOC] Only for the cudaMallocAsync backend, needs at least 16 GiB free.
        # Resets the per-process memory fraction to 1.0, fills about 82 percent of free memory,
        # frees a 10 percent tensor on a busy side stream so the free is pending, then requests
        # 15 percent. Passes only if the allocator trims freed backing memory and retries
        # instead of raising OOM. Cleans up and restores the memory fraction.
        # The fix under test: when the cudaMallocAsync allocator cannot fit an
        # allocation it normally raises OutOfMemoryError; the fix first releases the
        # pool's freed-but-cached backing to the OS and tries once more. We build an
        # allocation that only succeeds if that retry runs -- it OOMs before the fix
        # and succeeds after.
        #
        # The trick is to make memory that is freed but not yet reusable at the
        # moment of the allocation:
        #   1. Fill the GPU with a "wall" we keep alive, so almost nothing is free.
        #   2. On a side stream, keep the stream busy, then allocate A and free it.
        #      A's free is queued behind the busy work and has not run, so A is
        #      released by us but not yet reusable by the allocator.
        #   3. Ask for B (larger than A). Little is free and A is unusable, so the
        #      first attempt fails; the fix then syncs the stream (running A's free),
        #      trims the pool, retries, and B fits (a recovery is logged). Without
        #      the fix the allocation raises OutOfMemoryError and the test fails.
        dev = torch.cuda.current_device()
        # Neutralize any inherited per-process memory fraction: a fractional cap
        # would raise a *preemptive* OOM before cudaMallocAsync ever fails, which is
        # a different path than the trim-and-retry this test targets.
        orig_fraction = torch.cuda.get_per_process_memory_fraction(dev)
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
        torch.cuda.empty_cache()  # start from a clean, predictable amount of free memory
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info(dev)
        if free < 16 * 1024**3:
            self.skipTest("insufficient free CUDA memory for the trim-on-OOM repro")

        # wall + A reserve 0.92; with A unusable only ~0.08 is free, so B (0.15)
        # fails, and reclaiming A's 0.10 lets the retry fit.
        wall_size = int(free * 0.82)
        a_size = int(free * 0.10)
        b_size = int(free * 0.15)

        wall = torch.empty(wall_size, dtype=torch.int8, device=dev)  # keep most memory
        s1 = torch.cuda.Stream()
        b = None
        try:
            with torch.cuda.stream(s1):
                a = torch.empty(a_size, dtype=torch.int8, device=dev)
                # Keep the stream busy so A's free stays queued (pending, not yet
                # reclaimable) when B is requested. The sleep must still be running
                # when the host reaches B; if it finished first, A's free would run
                # and B would simply reuse it. 1s comfortably covers the host's
                # sub-ms trip to B even on a loaded machine.
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                del a
                gc.collect()  # free A, but its release is queued behind the busy work
                # Pre-fix this raises OutOfMemoryError; the fix trims and retries.
                b = torch.empty(b_size, dtype=torch.int8, device=dev)
                torch.cuda.synchronize()
            # Reaching here without OutOfMemoryError means the trim-and-retry ran.
            self.assertIsNotNone(b)
        finally:
            del wall, b
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(orig_fraction, dev)

    @slowTest
    def test_multinomial_invalid_probs_cuda(self):
        # [FUNCTION DOC] Slow test that runs the multinomial invalid-probability subprocess
        # helper with negative, positive-infinite, negative-infinite and NaN entries.
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -1.0, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, 1.0, nan])

    @slowTest
    def test_index_out_of_bounds_exception_cuda(self):
        # [FUNCTION DOC] Slow test: runs a child test script (embedded text) that indexes a CUDA
        # tensor out of bounds and captures its stderr. Asserts that stderr contains a
        # device-side assert message (CUDA) or one of the ROCm launch-failure/assert messages.
        # Test that indexing out of bounds causes assert
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)

class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_index_bounds_cuda(self):
        x = torch.arange(10, device="cuda")
        y = x[torch.tensor([11])].cpu()

if __name__ == '__main__':
    run_tests()
""")
        # CUDA raises cudaErrorAssert (710) with "device-side assert triggered"
        # ROCm with TORCH_USE_HIP_DSA raises a proper device-side assertion
        # ROCm without TORCH_USE_HIP_DSA raises hipErrorLaunchFailure (719)
        # which still catches the error but with less specific messaging
        is_cuda_assert = "device-side assert triggered" in stderr
        is_hip_assert = "hipErrorLaunchFailure" in stderr
        is_hip_assert = is_hip_assert or "unspecified launch failure" in stderr
        is_hip_assert = is_hip_assert or "HSA_STATUS_ERROR_EXCEPTION" in stderr
        self.assertTrue(
            is_cuda_assert or is_hip_assert,
            lambda msg: f"{msg}\nExpected device assert error in stderr, got: {stderr}",
        )

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_huge_index(self):
        # [FUNCTION DOC] Slow, serial test needing large GPU memory: builds a 15M by 45 long
        # tensor, gathers rows with a random permutation index on the GPU and compares with the
        # same indexing done on the CPU.
        src = torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(
            0, 2**22
        )
        idx = torch.randperm(src.shape[0], device="cuda")
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_randint_randomness_for_large_range(self) -> None:
        # [FUNCTION DOC] Regression for issue 125224: random states must not be reused with
        # ranges above 2 to the 32. Draws two consecutive randint batches of one million values
        # from a seeded generator on CUDA and CPU, counts unique values, and asserts the counts
        # differ by less than 10000.
        # For large ranges, randint generation is slightly different. This lead to a subtle bug where some Philox
        # offsets were not calculated correctly, resulting in reused random states.
        # See https://github.com/pytorch/pytorch/issues/125224
        size = 1_000_000
        high = 6_000_000_000  # Keep this above 2**32

        def run(dev: torch.device) -> int:
            # [FUNCTION DOC] HELPER: Seeds a generator on the given device, draws two randint
            # tensors of one million int64 values with a high above 2 to the 32 and returns the
            # number of unique values across both.
            # Measure how many unique numbers are generated in 2 consecutive calls to randint. If random states are
            # reused, this will yield fewer unique numbers.
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            t2 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        self.assertTrue(
            abs(run(torch.device("cuda")) - run(torch.device("cpu"))) < 10_000
        )

    @largeTensorTest("20GB", "cuda")
    @serialTest()
    def test_randint_generation_for_large_numel(self) -> None:
        # [FUNCTION DOC] Needs 20 GB. Generates int8 random values in [0, 2) for 2 to the 31
        # plus 1 elements on the GPU and asserts the sum is positive (i.e. the tail beyond 2 to
        # the 31 does not break generation).
        numel = 2**31 + 1
        s = torch.randint(2, (numel,), device="cuda", dtype=torch.int8).sum()
        self.assertTrue(s > 0, "expected randint in [0, 1] to generate nonzero values")

    @parametrize("dtype", [torch.float32, torch.double])
    def test_random_no_reused_random_states(self, dtype: torch.dtype) -> None:
        # [FUNCTION DOC] Parametrized over float32 and double. For each of rand and randn,
        # generates two consecutive million-element batches from a seeded generator on CUDA and
        # on CPU, counts unique values across the two batches, and asserts the CUDA and CPU
        # unique counts differ by less than 50000 (random states must not overlap).
        # Test if random states do not overlap between consecutive rand/randn calls.
        # See https://github.com/pytorch/pytorch/issues/125224

        def run(func, dev: torch.device, dtype: torch.dtype) -> int:
            # [FUNCTION DOC] HELPER: Seeds a new generator on the given device with 0, calls the
            # given random factory twice for one million elements of the given dtype using that
            # generator, and returns the number of unique values in the two results combined.
            # Measure how many unique numbers are generated in 2 consecutive calls. If random states are
            # reused, this will yield fewer unique numbers.
            size = 1000000
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = func((size,), device=dev, generator=gen, dtype=dtype)
            t2 = func((size,), device=dev, generator=gen, dtype=dtype)
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        for func in [torch.rand, torch.randn]:
            deviation = abs(
                run(func, torch.device("cuda"), dtype)
                - run(func, torch.device("cpu"), dtype)
            )
            self.assertTrue(deviation < 50_000, deviation)

    def test_min_max_inits(self):
        # [FUNCTION DOC] Checks that max over dimension 0 of a uint8 zero and min over dimension
        # 0 of a uint8 255 both return the index 0, verifying the reduction index initialization
        # at extreme values.
        # Testing if THC_reduceAll received the correct index initialization.
        # This affects the result of THC_reduceAll operations at extreme values
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

    def test_nvtx(self):
        # [FUNCTION DOC] Smoke test that the nvtx bindings exist and can be called: range push
        # and pop, a mark, and range start/end with the returned handle. No assertions.
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)

    def test_bincount_ext(self):
        # [FUNCTION DOC] Compares GPU bincount with CPU bincount, unweighted and weighted
        # (double weights), for int8 inputs with 50 bins (shared memory path) and int64 inputs
        # with 50000 bins (global memory path); also checks that a minlength of 65536 with a
        # large value 35488 in an int32 input does not overflow (counts sum to 10).
        # ensure CUDA code coverage
        input_size = (100000,)
        w = torch.randn(input_size, dtype=torch.double, device="cuda")
        w_cpu = w.cpu()
        # test shared memory impl
        t = torch.randint(50, input_size, dtype=torch.int8, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test global memory impl
        #   see `CUDAHistogramMemoryType` in SummaryOps.cu
        #   50000 * sizeof(int64_t) == 390 KiB, which should exceed smem of any known GPU
        t = torch.randint(50000, input_size, dtype=torch.int64, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device="cuda")
        # 35488 * 65536 as int32 would cause overflow to negative value
        # giving negative bin offset
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_tiny_half_norm_(self):
        # [FUNCTION DOC] Checks that the norm of a half tensor of tiny values (about 1e-8 scale,
        # converted from float32) is strictly positive.
        a = torch.arange(25).cuda().float()
        a /= 100000000
        b = a.half()
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        # [FUNCTION DOC] Checks that the zero-norm of 65536 half ones computed with float32
        # dtype equals 65536 (no half overflow in the count).
        a = torch.ones(65536).cuda().half()
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    @serialTest()
    def test_cuda_opaque_type(self):
        # [FUNCTION DOC] Needs medium-size GPU memory and runs serially: builds a 600M-element
        # int32 tensor of ones, applies a where selecting the same tensor in both branches and
        # asserts the output equals the input.
        x = torch.ones(600_000_000, dtype=torch.int32, device="cuda")
        y = torch.where(x > 0, x, x)
        self.assertEqual(y, x)

    def test_cuda_memory_leak_detection_propagates_errors(self):
        # [FUNCTION DOC] Checks that the leak-detection context manager does not swallow errors:
        # a broadcasting-mismatch addition of two CUDA tensors inside it must still raise
        # RuntimeError about the tensor size mismatch.
        with self.assertRaisesRegex(
            RuntimeError, r"The size of tensor a \(3\) must match"
        ):
            with self.assertLeaksNoCudaTensors():
                x = torch.randn(3, 1, device="cuda")
                y = torch.randn(2, 1, device="cuda")
                x + y

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    @serialTest()
    def test_cuda_kernel_loop_overflow(self):
        # [FUNCTION DOC] Regression for issue 24309 (kernel loop index overflow): runs average
        # pooling with kernel size 1 over a half tensor having 2 to the 30 plus 1 elements and
        # asserts the last element is preserved. Needs medium GPU memory.
        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**30 + 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**30]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**30], expected)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @gcIfJetson
    @serialTest()
    def test_cuda_kernel_loop_overflow_large(self):
        # [FUNCTION DOC] Needs large GPU memory. Average pooling over a half tensor with 2 to
        # the 31 elements must raise RuntimeError (value cannot be converted); with 2 to the 31
        # minus 1 elements it must work and keep the last element unchanged.
        # Make sure input.numel() > INT_MAX is handled:
        x = torch.randn(1, 1, 1, 2**31, dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "value cannot be converted to type"):
            y = torch.nn.functional.avg_pool2d(x, kernel_size=1)

        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**31 - 2]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**31 - 2], expected)

    # this might create a reference cycle on self...
    def _make_multiply_in_stream(self):
        # [FUNCTION DOC] HELPER: Builds and returns a custom autograd Function class that
        # multiplies by a constant in forward and, in backward, asserts the current stream is
        # the stream captured in forward, sleeps on the GPU to delay the op, and returns the
        # scaled gradient. The class closes over the test case, which may create a cycle.
        class MultiplyInStream(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function that multiplies its input by a scalar.
            # Forward saves the scalar and the current stream on the context; backward checks it
            # runs on that same stream, sleeps to delay the background stream, and returns the
            # gradient times the scalar (and no gradient for the scalar).
            @staticmethod
            def forward(ctx, x, val):
                # [FUNCTION DOC] HELPER: Stores the multiplier and the current stream on the
                # context and returns the input times the multiplier.
                ctx.val = val
                ctx.stream = torch.cuda.current_stream()
                return x * val

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Asserts the backward runs on the stream recorded in
                # forward, sleeps about 5M cycles on the GPU to delay the stream, then returns
                # gradient times the multiplier together with None for the multiplier input.
                self.assertEqual(torch.cuda.current_stream(), ctx.stream)
                # delays the operation in the background stream
                torch.cuda._sleep(1000 * 5000)
                return grad * ctx.val, None

        return MultiplyInStream

    @skipCUDANonDefaultStreamIf(True)
    def test_streaming_backwards_sync(self):
        # [FUNCTION DOC] Tests stream synchronization of backward passes (default stream only).
        # Case one: forward and backward under a side stream, gradients read after the default
        # stream waits on it. Case two: forward on a side stream, backward invoked in another
        # ambient stream; checks the gradients can be used in that ambient stream without extra
        # syncs and the ambient stream remains current.
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()

        MultiplyInStream = self._make_multiply_in_stream()

        # Tests using grads outside the backward() stream context
        # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 2)
            output.sum().backward()
        # sync needed
        default_stream.wait_stream(stream)
        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        self.assertEqual(torch.cuda.current_stream(), default_stream)

        # Tests that using grads in the same stream context as backward()
        # is safe regardless what streams bwd ops ran on
        bwd_ambient_stream = torch.cuda.Stream()
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 3)
        with torch.cuda.stream(bwd_ambient_stream):
            bwd_ambient_stream.wait_stream(stream)
            output.sum().backward()
            # x was first used on "stream" so its AccumulateGrad leaf should run on "stream".
            # The end of backward() should have synced "bwd_ambient_stream" with "stream"
            # so it should be safe to use x.grad here without any syncs.
            self.assertEqual(x.grad, torch.ones_like(x) * 3)
            self.assertEqual(torch.cuda.current_stream(), bwd_ambient_stream)

    def test_streaming_backwards_multiple_streams(self):
        # [FUNCTION DOC] Tests backward through a model using two side streams that merge
        # results, for every combination of whether the input is first used on the ambient
        # stream and of out-of-place gradient vs in-place backward with 1 or 5 iterations.
        # Checks the gradient hook runs on the expected stream and the final gradient equals
        # five times the iteration count.
        MultiplyInStream = self._make_multiply_in_stream()

        class StreamModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module holding one event and two side streams. Its
            # forward runs two multiply functions on separate streams (one on a clone of the
            # input) and joins them with an event wait, returning the sum of both results.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the event and the two streams used by forward.
                super().__init__()
                self.event = torch.cuda.Event()
                self.stream0 = torch.cuda.Stream()
                self.stream1 = torch.cuda.Stream()

            def forward(self, x, x_first_use_on_ambient):
                # [FUNCTION DOC] HELPER: Both side streams wait for the current stream. Stream 0
                # computes a clone of x (unless done earlier on the ambient stream) multiplied
                # by 2 and records the event; stream 1 computes x times 3, waits for the event
                # and returns the sum of both results.
                if x_first_use_on_ambient:
                    x0 = x.clone()
                self.stream0.wait_stream(torch.cuda.current_stream())
                self.stream1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream0):
                    if not x_first_use_on_ambient:
                        x0 = x.clone()
                    y0 = MultiplyInStream.apply(x0, 2)
                    self.event.record(stream=torch.cuda.current_stream())

                with torch.cuda.stream(self.stream1):
                    y1 = MultiplyInStream.apply(x, 3)
                    self.stream1.wait_event(self.event)
                    return y0 + y1

        stream = torch.cuda.Stream()

        for x_first_use_on_ambient in (True, False):
            # the out_of_place=False, iters=1 case stresses if proper syncs are inserted
            # when grads are initially None and stolen by backward ops.
            for out_of_place, iters in ((True, 1), (False, 1), (False, 5)):
                with torch.cuda.stream(stream):
                    x = torch.randn(5, 5, device="cuda", requires_grad=True)
                    model = StreamModel().cuda()
                    x.register_hook(
                        lambda grad: self.assertEqual(
                            torch.cuda.current_stream(),
                            stream if x_first_use_on_ambient else model.stream0,
                        )
                    )
                    for p in model.parameters():
                        self.assertTrue(p.grad is None)
                    for _ in range(iters):
                        loss = model(x, x_first_use_on_ambient).sum()
                        if out_of_place:
                            x_grad = torch.autograd.grad((loss,), (x,))[0]
                        else:
                            loss.backward()
                # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
                torch.cuda.current_stream().wait_stream(stream)

                if out_of_place:
                    self.assertEqual(x_grad, torch.ones_like(x) * 5 * iters)
                else:
                    self.assertEqual(x.grad, torch.ones_like(x) * 5 * iters)

    def test_streaming_backwards_sync_graph_root(self):
        # [FUNCTION DOC] Race-condition test over five trials: forward on one stream, a long
        # dummy kernel and gradient fill on another ambient stream, then backward from there.
        # Checks backward ops sync with the ambient stream before consuming the gradient and
        # that gradients of both leaves equal the incoming gradient times the other input. May
        # spuriously pass but a failure means a real bug.
        # This function tests if bwd ops running on a side stream properly sync with the GraphRoot.
        # The potential bug it targets is a race condition. The test uses multiple trials and
        # torch.cuda._sleep such that if the race condition exists, the test will almost certainly fail,
        # but there's a chance it may spuriously pass. Passing does not guarantee the backend is bug-free,
        # but failure does guarantee there is a bug.
        fwd_bwd_op_stream = torch.cuda.Stream()
        bwd_ambient_stream = torch.cuda.Stream()
        # We need these streams to be different otherwise the test is meaningless.
        self.assertTrue(fwd_bwd_op_stream != bwd_ambient_stream)

        size = int(1e3)

        a = torch.full((size,), 2.0, device="cuda", requires_grad=True)
        b = torch.full((size,), 3.0, device="cuda", requires_grad=True)

        # I don't think we need any manual record_streams below.
        # a and b remain in scope for the entire test.
        # c and grad remain in scope for each iteration, and there's a full sync between iterations.
        for trial in range(5):
            torch.cuda.synchronize()
            a.grad = b.grad = None
            with torch.cuda.stream(fwd_bwd_op_stream):
                c = a * b

            with torch.cuda.stream(bwd_ambient_stream):
                torch.cuda.synchronize()
                # Long-running dummy kernel on bwd_ambient_stream delays filling of grad
                torch.cuda._sleep(int(50 * get_cycles_per_ms()))
                # Fills grad on bwd_ambient_stream
                grad = torch.full((size,), float(trial + 1), device="cuda")

                # Bwd ops still run on fwd_bwd_ops_stream, so the following will likely fail if
                # bwd ops don't sync with bwd_ambient_stream before consuming grad.
                torch.autograd.backward(tensors=c, grad_tensors=grad)

                # See https://github.com/pytorch/pytorch/issues/47028
                # assertEquals below run on bwd_ambient_stream, so this test may also fail
                # if backward() fails to sync with bwd_ambient_stream at the end.
                # Synchronizing here works around the issue until a proper fix can be made.
                torch.cuda.synchronize()
                with torch.no_grad():
                    self.assertEqual(a.grad, grad * b)
                    self.assertEqual(b.grad, grad * a)

    def test_streaming_backwards_callback(self):
        # [FUNCTION DOC] Tests that autograd final callbacks sync with the leaf streams and the
        # stream surrounding backward. Builds leaves on separate streams, registers a hook
        # queuing a callback that clones leaf gradients, runs backward on a third stream, then
        # checks the cloned gradients both equal six.
        # Tests if autograd callbacks sync properly with respect to leaf streams and
        # the user-facing stream surrounding backward(). If it fails, first suspect is
        # sync logic where  "final_callbacks_" are called in torch/csrc/autograd/engine.cpp
        MultiplyInStream = self._make_multiply_in_stream()

        size = int(1e3)
        a = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)
        b = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        stash = []

        # sets up a nontrivial structure of leaf streams
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            c = MultiplyInStream.apply(a, 2)

        s1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s1):
            d = MultiplyInStream.apply(b, 3)
            s1.wait_stream(s0)
            e = c * d

            def clone_leaf_grads():
                # [FUNCTION DOC] HELPER: Callback that appends clones of the gradients of both
                # leaf tensors to the enclosing stash list.
                stash.append(a.grad.clone())
                stash.append(b.grad.clone())

            # Use a hook on e to install the callback
            e.register_hook(
                lambda grad: torch.autograd.Variable._execution_engine.queue_callback(
                    clone_leaf_grads
                )
            )

        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            e.sum().backward()
            # The autograd engine should sync s2 with all leaf streams then run the callback clone_leaf_grads on s2.
            # If those things happened properly, checking the values of the cloned grads on s2 should be safe:
            self.assertEqual(stash[0], torch.full_like(a, 6))
            self.assertEqual(stash[1], torch.full_like(a, 6))

    def test_fixed_cuda_assert_async(self):
        # [FUNCTION DOC] Checks torch._assert_async: empty and multi-element tensors raise
        # RuntimeError with ambiguity messages; true-like non-zero scalars of several types
        # pass; and for four false-like scalars (integer zero, float zero, False, complex zero)
        # a child process running the call then synchronizing must exit non-zero.
        with self.assertRaisesRegex(
            RuntimeError, "Boolean value of Tensor with no values is ambiguous"
        ):
            torch._assert_async(torch.tensor([], device="cuda"))
        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            torch._assert_async(torch.tensor([0, 0], device="cuda"))

        torch._assert_async(torch.tensor(1, device="cuda"))
        torch._assert_async(torch.tensor(0.1, device="cuda"))
        torch._assert_async(torch.tensor(-0.1, device="cuda"))
        torch._assert_async(torch.tensor(True, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0.1j, device="cuda"))

        fail_stmts = [
            "torch._assert_async(torch.tensor(0, device='cuda'))",
            "torch._assert_async(torch.tensor(0.0, device='cuda'))",
            "torch._assert_async(torch.tensor(False, device='cuda'))",
            "torch._assert_async(torch.tensor(0 + 0j, device='cuda'))",
        ]

        import subprocess

        for stmt in fail_stmts:
            with self.subTest(stmt=stmt):
                r = subprocess.call(
                    [
                        sys.executable,
                        "-c",
                        f"""\
import torch

{stmt}
torch.cuda.synchronize()
""",
                    ]
                )
                self.assertTrue(r != 0)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "FAIL")
    def test_cublas_multiple_threads_same_device(self):
        # [FUNCTION DOC] Race test for cuBLAS handle usage from two threads on the same device
        # (skipped for the async allocator). Each thread uses its own stream and a barrier,
        # repeatedly multiplies a ones matrix by a ones weight and divides by the size; over 3
        # trials checks each thread's result still sums to size squared.
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device="cuda")
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            # [FUNCTION DOC] HELPER: Thread body. Makes a private stream, synchronizes and waits
            # at the barrier, then on that stream repeats a matrix multiply of this thread's
            # result with the shared weight followed by an in-place divide, storing back into
            # the results mapping, and finally synchronizes.
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for _ in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS in its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    # Test is flaky on Windows (https://github.com/pytorch/pytorch/issues/57401)
    @unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows (see issue 57401)")
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_cudnn_multiple_threads_same_device(self):
        # [FUNCTION DOC] Race test for per-thread cuDNN handles (skipped on Windows and without
        # cuDNN). Two threads each run 1000 iterations of a 2 by 2 ones convolution and divide
        # on private streams over a ones image of 2048 squared, for 3 trials, and the sum must
        # equal the expected shrunken image size squared.
        # This function is intended to test the lazy creation and reuse of per-thread
        # cudnn handles on each device in aten/src/ATen/cudnn/Handles.cpp.
        # Failure here likely indicates something wrong with that logic.
        weight = torch.ones((1, 1, 2, 2), device="cuda")

        results = {}

        num_threads = 2
        trials = 3
        test_iters = 1000
        barrier = threading.Barrier(num_threads)

        with torch.backends.cudnn.flags(enabled=True):

            def _worker(t):
                # [FUNCTION DOC] HELPER: Thread body. Creates a private stream, synchronizes,
                # waits at the barrier, then repeatedly applies the convolution to this thread's
                # image in the results mapping and divides by 4, finishing with a synchronize.
                my_stream = torch.cuda.Stream()
                # Hard sync so we don't need to worry about creating and using tensors
                # across streams or the fact that default streams are thread-local.
                # Those issues are not the target of this test.
                torch.cuda.synchronize()
                # Line up threads to increase likelihood of race conditions.
                barrier.wait()
                with torch.cuda.stream(my_stream):
                    for _ in range(test_iters):
                        # If all threads are sharing the same cudnn handle,
                        # the following sequence may occur:
                        # thread 0 calls setCuDNNStreamToCurrent()
                        # thread 1 calls setCuDNNStreamToCurrent()
                        # thread 0 launches its raw convolution, which it thinks is in
                        #          its own stream, but is actually in thread 1's stream.
                        # thread 0 enqueues its div_, which IS in its own stream,
                        #          but now races with its convolution.
                        results[t] = torch.nn.functional.conv2d(
                            results[t], weight, padding=0
                        )
                        results[t].div_(4.0)
                torch.cuda.synchronize()

            for _ in range(trials):
                for t in range(num_threads):
                    results[t] = torch.ones((1, 1, 2048, 2048), device="cuda")

                threads = [
                    threading.Thread(target=_worker, args=(t,))
                    for t in range(num_threads)
                ]

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for t in range(num_threads):
                    self.assertEqual(
                        results[t].sum().item(),
                        (2048 - test_iters) * (2048 - test_iters),
                    )

    def test_cusparse_multiple_threads_same_device(self):
        # [FUNCTION DOC] Race test for cuSPARSE handle use from two threads: a sparse all-ones
        # matrix of size 1024 is multiplied with each thread's dense ones matrix 500 times on
        # private streams with divides, for 3 trials; each result must sum to size squared.
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 500

        def ones_sparse(size):
            # [FUNCTION DOC] HELPER: Builds a sparse COO tensor of the given size squared
            # with every entry equal to one, using the cartesian product of the index range
            # for the indices.
            a = torch.arange(size, device="cuda")
            indices = torch.cartesian_prod(a, a).t()
            values = torch.ones(size * size, device="cuda")
            return torch.sparse_coo_tensor(indices, values)

        weight = ones_sparse(size)
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            # [FUNCTION DOC] HELPER: Thread body. Creates a private stream, synchronizes and
            # waits at the barrier, then on that stream repeatedly does a sparse-dense matmul
            # with this thread's result and divides by the size, storing into the results
            # mapping; ends with a synchronize.
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for _ in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS in its own stream,
                    #          but actually now races with its gemm.
                    results[t] = weight.mm(results[t])
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_max_large_axis(self):
        # [FUNCTION DOC] Slow serial test needing large GPU memory: sets the last element of a
        # zeros int8 tensor of 2 to the 32 elements to one and checks max over dimension 0
        # returns value 1 and the last index.
        x = torch.zeros(2**32, device="cuda", dtype=torch.int8)
        x[-1] = 1
        val, idx = x.max(0)
        self.assertEqual(val, 1)
        self.assertEqual(idx, x.shape[0] - 1)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_to_numpy(self):
        # [FUNCTION DOC] Checks that converting a CUDA tensor to numpy raises TypeError. Skipped
        # when numpy is missing.
        self.assertRaises(TypeError, lambda: torch.empty(1, device="cuda").numpy())

    def test_accelerator_default_generator(self):
        # [FUNCTION DOC] For each accelerator device, compares the cuda default generator
        # with the accelerator default generator: same device and same state; seeding one
        # with 42 makes the other report initial seed 42; after reseeding randomly the states
        # stay equal.
        torch.cuda.init()
        for index in range(torch.accelerator.device_count()):
            cuda_default_generator = torch.cuda.default_generators[index]
            acc_default_generator = torch._C._accelerator_getDefaultGenerator(index)
            self.assertEqual(
                cuda_default_generator.device, acc_default_generator.device
            )
            # Verify they share the same underlying GeneratorImpl
            self.assertEqual(
                cuda_default_generator.get_state(),
                acc_default_generator.get_state(),
            )
            cuda_default_generator.manual_seed(42)
            self.assertEqual(acc_default_generator.initial_seed(), 42)
            # Verify state stays in sync after reseeding
            cuda_default_generator.seed()
            self.assertEqual(
                cuda_default_generator.get_state(),
                acc_default_generator.get_state(),
            )

    def test_graph_is_current_stream_capturing(self):
        # [FUNCTION DOC] Checks capture query functions: outside capture, the current-stream
        # capturing query is False. On non-ROCm, within a side stream the query and the stream
        # method are False before capture_begin and True after it, then capture ends.
        self.assertFalse(torch.cuda.is_current_stream_capturing())

        if TEST_CUDA and (not TEST_WITH_ROCM):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g = torch.cuda.CUDAGraph()
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                self.assertFalse(s.is_capturing())
                g.capture_begin()
                self.assertTrue(torch.cuda.is_current_stream_capturing())
                self.assertTrue(s.is_capturing())
                g.capture_end()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_simple(self):
        # [FUNCTION DOC] Basic CUDA graph capture on a side stream: captures ten additions of 1
        # to a tensor of ones (1000 elements), waits for the stream, replays, and asserts the
        # sum is 11000.
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_stale_default_stream_error(self):
        # [FUNCTION DOC] Warmup on the default stream then capture on a side stream without the
        # stale-stream override: backward during relaxed-mode capture must raise RuntimeError
        # mentioning a stale reference to the default stream. The capture is always ended.
        """Default-stream warmup + side-stream capture raises a clear error
        (not an opaque CUDA crash) when override is not enabled."""
        default_stream = torch.cuda.default_stream()
        with torch.cuda.stream(default_stream):
            model = torch.nn.Linear(32, 32).cuda()
            x = torch.ones(4, 32, device="cuda")
            loss = model(x).sum()
            loss.backward()
            model.zero_grad()

        s = torch.cuda.Stream()
        static_x = torch.ones(4, 32, device="cuda")

        with torch.cuda.stream(s):
            static_loss = model(static_x).sum()
            static_loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            g.capture_begin(capture_error_mode="relaxed")
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "stale reference to the default stream"
                ):
                    static_loss = model(static_x).sum()
                    static_loss.backward()
            finally:
                g.capture_end()
        torch.cuda.synchronize()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_override_stale_stream_default_warmup(self):
        # [FUNCTION DOC] With the stale-capture-stream override enabled, warmup on the default
        # stream followed by capture on a side stream succeeds: forward and backward are
        # captured, the graph is replayed, and all parameters have gradients. The override is
        # turned off again in a finally block.
        """override_stale_capture_stream fixes default-stream warmup +
        side-stream capture."""
        default_stream = torch.cuda.default_stream()
        with torch.cuda.stream(default_stream):
            model = torch.nn.Linear(32, 32).cuda()
            x = torch.ones(4, 32, device="cuda")
            loss = model(x).sum()
            loss.backward()
            model.zero_grad()

        s = torch.cuda.Stream()
        static_x = torch.ones(4, 32, device="cuda")

        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.stream(s):
                static_loss = model(static_x).sum()
                static_loss.backward()
                model.zero_grad()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                g.capture_begin()
                static_loss = model(static_x).sum()
                static_loss.backward()
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            g.replay()
            torch.cuda.synchronize()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_override_stale_stream_side_warmup(self):
        # [FUNCTION DOC] With the override enabled, warmup on one side stream and capture on a
        # different side stream succeeds; after replay all parameters have gradients. The
        # override is turned off in a finally block.
        """override_stale_capture_stream fixes side-stream warmup +
        different side-stream capture."""
        warmup_stream = torch.cuda.Stream()
        capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(warmup_stream):
            model = torch.nn.Linear(32, 32).cuda()
            x = torch.ones(4, 32, device="cuda")
            loss = model(x).sum()
            loss.backward()
            model.zero_grad()
            static_x = torch.ones(4, 32, device="cuda")

        torch.cuda.synchronize()

        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.stream(capture_stream):
                static_loss = model(static_x).sum()
                static_loss.backward()
                model.zero_grad()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                g.capture_begin()
                static_loss = model(static_x).sum()
                static_loss.backward()
                g.capture_end()

            torch.cuda.current_stream().wait_stream(capture_stream)
            g.replay()
            torch.cuda.synchronize()

            for p in model.parameters():
                self.assertIsNotNone(p.grad)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)

    class _NoWgrad(torch.autograd.Function):
        # [CLASS DOC] HELPER: Autograd function for matmul with weight transposed that returns
        # an undefined gradient for the weight (delayed-wgrad pattern); forward saves the
        # weight, backward returns the input gradient times the weight and None for the weight.
        """Uses the weight in forward but returns an undefined grad for it
        (delayed-wgrad pattern: the real wgrad is computed out of band)."""

        @staticmethod
        def forward(ctx, x, w):
            # [FUNCTION DOC] HELPER: Saves the weight for backward and returns the input matrix
            # times the transposed weight.
            ctx.save_for_backward(w)
            return x @ w.t()

        @staticmethod
        def backward(ctx, gy):
            # [FUNCTION DOC] HELPER: Returns the incoming gradient matrix times the saved weight
            # as the input gradient, and None as the weight gradient.
            (w,) = ctx.saved_tensors
            return gy @ w, None

    def _warmup_no_wgrad(self):
        # [FUNCTION DOC] HELPER: Creates a 16 by 16 CUDA parameter and a random input, runs
        # three forward and backward passes with the custom function on the default stream
        # with retained graph so the weight's gradient accumulator caches the stale stream,
        # then synchronizes and returns the input, weight and last output (which keeps the
        # graph alive).
        """Warmup on the default stream so the weight's AccumulateGrad caches
        a stale (default) stream, then return (x, w, keepalive).

        AccumulateGrad nodes are cached weakly: `keepalive` retains the warmup
        graph so the stale node survives until capture (as DDP/FSDP do by
        stashing grad accumulators)."""
        w = torch.nn.Parameter(torch.randn(16, 16, device="cuda"))
        x = torch.randn(4, 16, device="cuda")
        for _ in range(3):
            y = self._NoWgrad.apply(x, w)
            y.sum().backward(retain_graph=True)
        torch.cuda.synchronize()
        return x, w, y

    @skipIfRocm(msg="hipBLASLt lazy handle initialization fails during graph capture")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_undefined_grad_stale_leaf_error(self):
        # [FUNCTION DOC] A leaf whose incoming gradients are all undefined skips stream
        # reconciliation, so without the override the end-of-backward leaf sync must raise
        # RuntimeError ("across the capture boundary") during capture. Capture is always ended.
        # Skipped on ROCm and without graph support.
        """A leaf whose incoming grads are all undefined short-circuits out
        of the input buffer before stream reconciliation, so the override
        cannot fix its stale stream; the end-of-backward leaf sync must raise
        a clear error instead of invalidating the capture."""
        x, w, keepalive = self._warmup_no_wgrad()

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            g.capture_begin(capture_error_mode="relaxed")
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "across the capture boundary"
                ):
                    self._NoWgrad.apply(x, w).sum().backward()
            finally:
                g.capture_end()
        torch.cuda.synchronize()

    @skipIfRocm(msg="hipBLASLt lazy handle initialization fails during graph capture")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_undefined_grad_stale_leaf_override(self):
        # [FUNCTION DOC] With the override enabled, the sync with the stale undefined-gradient
        # leaf is skipped: capture, replay and synchronize succeed and the weight gradient stays
        # None. The override is reset afterward. Skipped on ROCm and without graphs.
        """With the override enabled, the meaningless sync with the stale
        undefined-grad leaf is skipped and capture + replay succeed."""
        x, w, keepalive = self._warmup_no_wgrad()

        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g = torch.cuda.CUDAGraph()
                g.capture_begin()
                self._NoWgrad.apply(x, w).sum().backward()
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            g.replay()
            torch.cuda.synchronize()
            self.assertIsNone(w.grad)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)

    class _CpuComputeCudaWeight(torch.autograd.Function):
        # [CLASS DOC] HELPER: Autograd function doing CPU compute (times two) on a CPU input
        # while taking a CUDA weight argument that gets no gradient; used so the weight's
        # gradient accumulator is created where apply runs.
        """CPU compute that references a CUDA weight: its AccumulateGrad is
        created where apply() runs, and the weight's grad is undefined."""

        @staticmethod
        def forward(ctx, x_cpu, w_cuda):
            # [FUNCTION DOC] HELPER: Returns the CPU input multiplied by two; the CUDA weight
            # is unused.
            return x_cpu * 2

        @staticmethod
        def backward(ctx, g):
            # [FUNCTION DOC] HELPER: Returns the gradient times two and None for the weight.
            return g * 2, None

    def _capture_reverse_leaf(self):
        # [FUNCTION DOC] HELPER: Builds the reverse stream-crossing scenario. Creates a CUDA
        # parameter and a CPU input, begins relaxed-mode graph capture on a side stream, adds a
        # trivial kernel so the capture is non-empty, applies the CPU function inside capture
        # and sums. Returns the graph, the stream and the loss; capture is left open.
        """Build the reverse crossing: the weight's AccumulateGrad is created
        inside the capture (its metadata stream is the capturing stream), and
        backward() runs with the non-capturing default stream current. The
        CPU root and CPU intermediate nodes produce no other stream syncs, so
        the end-of-backward leaf sync is the only capture crossing."""
        w = torch.nn.Parameter(torch.randn(8, device="cuda"))
        x = torch.randn(8)  # CPU

        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin(capture_error_mode="relaxed")
            torch.ones(4, device="cuda").mul_(2)  # non-empty capture
            loss = self._CpuComputeCudaWeight.apply(x, w).sum()
        return g, s, loss

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_capturing_leaf_noncapturing_caller_error(self):
        # [FUNCTION DOC] In the reverse direction, running backward from the non-capturing
        # default stream while the leaf's stream is capturing must raise RuntimeError ("across
        # the capture boundary"). The capture is always ended afterward on the side stream.
        """Reverse direction of the leaf-sync guard: leaf stream capturing,
        caller stream not."""
        g, s, loss = self._capture_reverse_leaf()
        try:
            with self.assertRaisesRegex(RuntimeError, "across the capture boundary"):
                loss.backward()
        finally:
            with torch.cuda.stream(s):
                g.capture_end()
        torch.cuda.synchronize()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_capturing_leaf_noncapturing_caller_override(self):
        # [FUNCTION DOC] Same reverse scenario with the override enabled: backward completes,
        # capture ends, the graph replays after synchronization; the override is reset in a
        # finally block.
        """Reverse direction with the override: the sync is skipped and the
        capture completes and replays."""
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            g, s, loss = self._capture_reverse_leaf()
            loss.backward()
            with torch.cuda.stream(s):
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            g.replay()
            torch.cuda.synchronize()
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graphsafe_set_get_rng_state(self):
        # [FUNCTION DOC] Tests graph-safe RNG state set/get. Builds a new generator and the
        # default generator with an old state and a cloned state each, captures the pattern of
        # random draws (new state once, old state twice) in a CUDA graph on the default
        # generator, then three times runs the same pattern eagerly on the new generator and
        # replays the graph, asserting final offsets and drawn values agree.
        # Define a function to create generator states, with optional graph registration
        def create_states(generator):
            # [FUNCTION DOC] HELPER: Initializes the CUDA generator by drawing once, seeds it
            # with 0, saves its current graph-safe state and a cloned state, and returns the
            # generator with both states.
            """Initializes generator states and registers them with a CUDA graph if provided."""
            # Ensure the CUDA generator is initialized
            torch.rand(1, device="cuda")
            generator.manual_seed(0)

            # Save the current state of the generator
            old_state = generator.graphsafe_get_state()
            # Create and save a cloned state of the generator
            new_state = generator.clone_state()
            # Return the original generator and its two states
            return generator, old_state, new_state

        # Define a function to perform specific RNG actions using the generator's states
        def perform_random_generation_steps(generator_state):
            # [FUNCTION DOC] HELPER: Given a tuple of generator and two states, sets the new
            # state and draws five random values, then sets the old state and draws five values
            # twice; returns the list of three tensors.
            generator, old_state, new_state = generator_state
            random_values = []

            # Generate random numbers with the new generator state
            generator.graphsafe_set_state(new_state)
            random_values.append(torch.rand(5, device="cuda", generator=generator))

            # Generate random numbers twice with the old generator state
            generator.graphsafe_set_state(old_state)
            random_values.extend(
                [torch.rand(5, device="cuda", generator=generator) for _ in range(2)]
            )

            return random_values

        # Define a function to retrieve the final offsets of the original and new generator states
        def get_final_offsets_of_states(generator_state):
            # [FUNCTION DOC] HELPER: Returns the pair of offsets of the old and new states of a
            # generator-state tuple.
            _, old_state, new_state = generator_state
            old_state_offset = old_state.get_offset()
            new_state_offset = new_state.get_offset()
            return old_state_offset, new_state_offset

        # Set up and test a new CUDA generator
        generator = torch.Generator(device="cuda")
        generator_state = create_states(generator)

        # Set up and test the default CUDA generator with a CUDA Graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        default_generator = torch.cuda.default_generators[0]
        default_generator_state = create_states(default_generator)

        # Perform random number generation within a CUDA graph
        with torch.cuda.stream(s):
            g.capture_begin()
            graphed_random_values = perform_random_generation_steps(
                default_generator_state
            )
            g.capture_end()

        # Synchronize the streams and replay the graph
        torch.cuda.current_stream().wait_stream(s)
        for _ in range(3):
            random_values = perform_random_generation_steps(generator_state)
            g.replay()
            offset = get_final_offsets_of_states(generator_state)
            graph_offset = get_final_offsets_of_states(default_generator_state)

            # Compare the final offsets of states for both generators to ensure consistency
            self.assertEqual(offset, graph_offset)
            # Compare the states generated outside and inside the graph
            self.assertEqual(random_values, graphed_random_values)

    def test_philox_state_eager(self):
        # [FUNCTION DOC] Checks philox_state in eager mode on a seeded generator: returns seed,
        # offset and intra-offset tensors of int64 shape one on CPU with expected values (seed
        # equal to the initial seed, zeros), the generator offset advances to 8 after reserving
        # 8, then to 16 on the next reservation, with the offset tensor returning 8.
        g = torch.Generator(device="cuda")
        g.manual_seed(123)
        seed_t, off_t, intra_t = g.philox_state(8)
        self.assertEqual(seed_t.item(), g.initial_seed())
        self.assertEqual(off_t.item(), 0)
        self.assertEqual(intra_t.item(), 0)
        # Outside capture all three are CPU tensors (PhiloxCudaState's
        # HostState); during capture seed/offset are CUDA (DevState).
        for t in (seed_t, off_t, intra_t):
            self.assertEqual(t.dtype, torch.int64)
            self.assertEqual(t.shape, (1,))
            self.assertEqual(t.device.type, "cpu")
        self.assertEqual(g.get_offset(), 8)
        _, off_t, _ = g.philox_state(8)
        self.assertEqual(off_t.item(), 8)
        self.assertEqual(g.get_offset(), 16)

    def test_philox_state_rounds_increment_to_multiple_of_4(self):
        # [FUNCTION DOC] Checks that reserved Philox increments are rounded up to multiples of
        # 4: reserving 1 moves the offset to 4, then reserving 5 moves it to 12.
        g = torch.Generator(device="cuda")
        g.manual_seed(0)
        g.philox_state(1)
        self.assertEqual(g.get_offset(), 4)
        g.philox_state(5)
        self.assertEqual(g.get_offset(), 12)

    def test_philox_state_large_seed_reinterpreted_as_int64(self):
        # [FUNCTION DOC] Checks that a seed above 2 to the 63 is returned reinterpreted as a
        # signed int64, recovering the original seed after masking to 64 bits.
        # Seeds >= 2**63 come back reinterpreted as negative int64.
        g = torch.Generator(device="cuda")
        seed = (1 << 64) - 3
        g.manual_seed(seed)
        seed_t, _, _ = g.philox_state(4)
        self.assertEqual(seed_t.item() & ((1 << 64) - 1), seed)

    def test_philox_state_offset_overflows_int64(self):
        # [FUNCTION DOC] Checks offset overflow handling: with the offset set to 2 to the 64
        # minus 4, the returned offset tensor carries the exact bits (recovered by masking to 64
        # bits) and the generator offset wraps around to 0 after reserving 4.
        # The C++ offset is uint64. The returned tensor carries its exact
        # bits in int64, so offsets >= 2**63 read back negative via .item()
        # and are recovered with `& (2**64 - 1)`. Advancement wraps modulo
        # 2**64 in uint64 space: from 2**64 - 4, reserving 4 lands on 0.
        g = torch.Generator(device="cuda")
        g.manual_seed(0)
        big = (1 << 64) - 4
        g.set_offset(big)
        _, off_t, _ = g.philox_state(4)
        self.assertEqual(off_t.item() & ((1 << 64) - 1), big)
        self.assertEqual(g.get_offset(), 0)

    def test_philox_state_composes_with_eager_ops(self):
        # [FUNCTION DOC] Checks that a Philox reservation consumes the same stream positions an
        # eager kernel would: after reserving 4 following a draw, the next random tensor equals
        # the one regenerated by reseeding and setting the offset to the earlier offset plus 4.
        # A reservation must consume the same Philox stream positions an
        # eager kernel would, so subsequent eager ops are reproducible via
        # set_offset.
        g = torch.Generator(device="cuda")
        g.manual_seed(0)
        torch.rand(1000, device="cuda", generator=g)
        off = g.get_offset()
        g.philox_state(4)
        b = torch.rand(1000, device="cuda", generator=g)
        g.manual_seed(0)
        g.set_offset(off + 4)
        self.assertEqual(torch.rand(1000, device="cuda", generator=g), b)

    def test_philox_state_advances_default_generator(self):
        # [FUNCTION DOC] Checks that reserving through the default CUDA generator advances the
        # global state used by generator-less operations by exactly the reserved amount: draws
        # after a reserve differ from the reference and equal the reference draw regenerated
        # with offset 4.
        # Reserving through the default generator advances the global state
        # that generator-less ops consume, by exactly the reserved amount.
        g = torch.cuda.default_generators[0]
        g.manual_seed(0)
        ref = torch.rand(1000, device="cuda")
        g.manual_seed(0)
        _, off_t, _ = g.philox_state(4)
        self.assertEqual(off_t.item(), 0)
        shifted = torch.rand(1000, device="cuda")
        self.assertNotEqual(shifted, ref)
        g.manual_seed(0)
        g.set_offset(4)
        self.assertEqual(torch.rand(1000, device="cuda"), shifted, atol=0, rtol=0)

    def test_philox_state_errors(self):
        # [FUNCTION DOC] Checks philox_state errors: a CPU generator raises NotImplementedError,
        # a negative increment raises OverflowError, and a non-integer increment raises
        # RuntimeError (expected an int).
        cpu_gen = torch.Generator()
        with self.assertRaisesRegex(NotImplementedError, "philox_state"):
            cpu_gen.philox_state(4)
        g = torch.Generator(device="cuda")
        with self.assertRaisesRegex(OverflowError, "negative"):
            g.philox_state(-1)
        with self.assertRaisesRegex(RuntimeError, "expected an int"):
            g.philox_state(1.5)

    def test_philox_state_graphsafe_shared_state(self):
        # [FUNCTION DOC] Checks that after installing a cloned graph-safe state in a generator,
        # reserving 8 Philox values advances the shared state object, so its offset is 8.
        g = torch.Generator(device="cuda")
        g.manual_seed(0)
        new_state = g.clone_state()
        g.graphsafe_set_state(new_state)
        g.philox_state(8)
        self.assertEqual(new_state.get_offset(), 8)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_philox_state_graph_capture(self):
        # [FUNCTION DOC] Captures two philox reservations in a CUDA graph on the default
        # generator. Checks seed and offset tensors are CUDA and the intra offsets are 0 and 4;
        # over two replays the seed tensor equals the initial seed, the offset tensor equals the
        # pre-replay generator offset and the generator offset advances by 8 per replay. Also
        # checks the returned alias tensors cannot be resized (RuntimeError).
        torch.rand(1, device="cuda")
        g = torch.cuda.default_generators[0]
        g.manual_seed(0)
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        x = torch.zeros(1, device="cuda")
        with torch.cuda.stream(s):
            graph.capture_begin()
            seed_t, off_t, intra_t0 = g.philox_state(4)
            x += 1
            _, _, intra_t1 = g.philox_state(4)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        self.assertEqual(seed_t.device.type, "cuda")
        self.assertEqual(off_t.device.type, "cuda")
        self.assertEqual(intra_t0.item(), 0)
        self.assertEqual(intra_t1.item(), 4)

        # Each replay refills the extragraph tensors with the generator's
        # current (seed, offset) and advances the offset by the whole-graph
        # increment.
        for _ in range(2):
            offset_before = g.get_offset()
            graph.replay()
            self.assertEqual(seed_t.item(), g.initial_seed())
            self.assertEqual(off_t.item(), offset_before)
            self.assertEqual(g.get_offset(), offset_before + 8)

        # The returned aliases must not allow reallocating the capture
        # state's storage out from under the captured graph.
        data_ptr = seed_t.data_ptr()
        with self.assertRaisesRegex(RuntimeError, "not resizable"):
            seed_t.resize_(64)
        self.assertEqual(seed_t.data_ptr(), data_ptr)

    @skipIfRocmVersionLessThan((7, 14))
    @xfailCUDAIfSM89OrLaterOnWindows
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_after_failed_capture(self):
        # [FUNCTION DOC] Regression test for RNG after a failed capture. Begins capture on a
        # side stream, performs a forbidden sync (RuntimeError) and a failing capture_end
        # (RuntimeError), then checks normal random generation works and a new capture on the
        # same stream can be replayed and fills a buffer; finally checks the count of active
        # allocations returns to the baseline. Skipped on ROCm with expandable segments and for
        # old ROCm versions.
        """Test that a stream can be captured again for RNG after a failed capture."""
        if TEST_WITH_ROCM and self.expandable_segments:
            self.skipTest(
                "ROCm expandable segments has known issue with graph capture recovery - #179911"
            )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_stats()["active.all.current"]

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        x = torch.ones(1, device="cuda")

        with torch.cuda.stream(stream):
            graph.capture_begin()
            with self.assertRaises(RuntimeError):
                (x + 1).item()
            with self.assertRaises(RuntimeError):
                graph.capture_end()

        torch.cuda.current_stream().wait_stream(stream)

        result = torch.randn(4, device="cuda")
        self.assertEqual(result.shape, (4,))

        # Capture again on the same stream to verify recovery
        new_graph = torch.cuda.CUDAGraph()
        buf = torch.empty(4, device="cuda")
        with torch.cuda.stream(stream):
            new_graph.capture_begin()
            buf.copy_(torch.randn_like(buf))
            new_graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)
        buf.zero_()
        new_graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(torch.allclose(buf, torch.zeros_like(buf)))

        # Cleanup: verify all allocations are freed
        del graph, new_graph, buf, x, result
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_stats()["active.all.current"]
        self.assertEqual(
            after, baseline, "Leaked CUDA/RNG allocations after failed capture test"
        )

    @skipIfRocmVersionLessThan((7, 14))
    @xfailCUDAIfSM89OrLaterOnWindows
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_error_releases_reserved_segments(self):
        # [FUNCTION DOC] Regression test: a failed graph capture must release the private memory
        # pool reserved segments when the graph is reset. Starts capture on a side stream,
        # allocates 32 MiB, triggers a forbidden synchronize (RuntimeError) and a failing
        # capture_end (RuntimeError), deletes the tensor and graph, empties the cache and
        # asserts memory_reserved equals the value measured before.
        # A failed capture (capture_end raises) must still release the graph's
        # private mempool on reset(); otherwise its reserved segments leak for
        # the rest of the process. Unlike test_graph_rng_after_failed_capture,
        # which asserts on active.all.current (no leaked live allocations), this
        # checks memory_reserved(): a leaked pool can show zero active bytes yet
        # still hold reserved segments, so only this assertion catches the leak.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        reserved_before = torch.cuda.memory_reserved()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            x = torch.empty(8 * 1024 * 1024, device="cuda")
            # A synchronizing op during capture invalidates it, but we must
            # still call capture_end() so the stream's capture is ended (with an
            # error); otherwise the stream is left mid-capture and teardown fails.
            with self.assertRaises(RuntimeError):
                torch.cuda.synchronize()
            with self.assertRaises(RuntimeError):
                g.capture_end()

        del x
        g.reset()
        del g
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.assertEqual(torch.cuda.memory_reserved(), reserved_before)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_concurrent_replay_on_different_streams(self):
        # [FUNCTION DOC] Checks two graphs sharing the default generator replayed concurrently
        # on different streams do not race on RNG state. Reference random tensors are drawn
        # after seeding; the same draws are captured into two graphs (copy into buffers),
        # buffers are zeroed, the graphs are replayed on separate streams and each buffer must
        # equal the corresponding reference.
        """Concurrent replay of two graphs sharing a generator on different streams.

        With per-generator state (old code), this would race on the shared
        rng_state_offset_extragraph_ tensor. With per-(generator, capture_id)
        state, each graph has its own tensors.
        """
        seed = 1234
        shape = (64,)

        torch.manual_seed(seed)
        ref0 = torch.randn(shape, device="cuda")
        ref1 = torch.randn(shape, device="cuda")

        torch.manual_seed(seed)
        g0 = torch.cuda.CUDAGraph()
        g1 = torch.cuda.CUDAGraph()
        s_cap = torch.cuda.Stream()
        buf0 = torch.empty(shape, device="cuda")
        buf1 = torch.empty(shape, device="cuda")

        with torch.cuda.stream(s_cap):
            g0.capture_begin()
            buf0.copy_(torch.randn_like(buf0))
            g0.capture_end()

        torch.cuda.current_stream().wait_stream(s_cap)

        with torch.cuda.stream(s_cap):
            g1.capture_begin()
            buf1.copy_(torch.randn_like(buf1))
            g1.capture_end()

        torch.cuda.current_stream().wait_stream(s_cap)

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()

        buf0.zero_()
        buf1.zero_()

        s0.wait_stream(torch.cuda.current_stream())
        s1.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s0):
            g0.replay()
        with torch.cuda.stream(s1):
            g1.replay()

        torch.cuda.synchronize()

        self.assertEqual(buf0, ref0)
        self.assertEqual(buf1, ref1)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/177001")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "ROCM does not support nvrtc or external cuda graph events"
    )
    @unittest.skipIf(not SM70OrLater, "SM70+ required for inline ptx")
    def test_graph_rng_replay_record_stream(self):
        # [FUNCTION DOC] Verifies RNG state tensors stay alive while a replay is in flight.
        # Captures a graph that spins on a pinned host flag then fills a buffer with uniform
        # random values; after capture, small-pool active blocks rise by 3 (buffer plus two RNG
        # tensors). Replays with the spin held, resets and deletes the graph, and asserts 2
        # blocks remain (kept by stream recording); after releasing the flag, syncing and
        # emptying the cache the count returns to baseline. Skipped on Linux (tracked issue),
        # ROCm and pre-SM70.
        """Verify RNG state tensors survive while a replay is in flight.

        See Note [RNG state tensor lifetime and recordStream] in
        CUDAGeneratorImpl.cpp.
        """
        spin_wait_kernel = get_wait_for_cpu_kernel()
        flag_cpu = torch.zeros(1, dtype=torch.int32, device="cpu").pin_memory()

        # Use small_pool stats to avoid interference from cuBLAS workspace
        # blocks (large pool) that preceding tests may create and that get
        # freed during g.reset() → clearCublasWorkspacesForStream.
        _stat = "active.small_pool.current"

        # Clean baseline
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_stats()[_stat]

        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        buf = torch.empty(32, device="cuda")

        with torch.cuda.stream(s):
            g.capture_begin(capture_error_mode="relaxed")
            spin_wait_kernel(grid=(1, 1, 1), block=(1, 1, 1), args=[flag_cpu])
            buf.uniform_()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        # After capture: buf + 2 RNG tensors = 3 new small-pool blocks
        after_capture = torch.cuda.memory_stats()[_stat]
        self.assertEqual(after_capture - baseline, 3)

        # Reference replay
        flag_cpu[0] = 1
        buf.zero_()
        g.replay()
        torch.cuda.synchronize()

        # Race: replay with spin held, then delete graph
        flag_cpu[0] = 0
        buf.zero_()
        with torch.cuda.stream(s):
            g.replay()  # GPU spinning

        g.reset()
        del g, buf

        # After reset: buf is freed (active decreases by 1).
        # The 2 RNG tensors should still be active because recordStream
        # defers their recycling until the replay stream finishes.
        after_reset = torch.cuda.memory_stats()[_stat]
        self.assertEqual(
            after_reset - baseline,
            2,
            "Expected 2 RNG state tensors kept alive by recordStream",
        )

        # Release GPU and clean up
        flag_cpu[0] = 1
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()

        after_cleanup = torch.cuda.memory_stats()[_stat]
        self.assertEqual(
            after_cleanup,
            baseline,
            "RNG state tensors should be freed after sync + empty_cache",
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_memory_stats_of_multiple_generators_and_graphs(self):
        # [FUNCTION DOC] Checks active block and byte accounting for graphs sharing multiple
        # generator states. For three configurations of graph and generator counts it captures
        # random draws per generator state in each graph, then asserts active blocks grew by two
        # per generator-graph pair and active bytes by 1024 bytes per pair, and cleans up.
        # Function to clear CUDA cache and collect garbage
        def clear_cuda_cache():
            # [FUNCTION DOC] HELPER: Synchronizes the device, runs garbage collection and
            # empties the CUDA cache.
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Executes a simple graph task which includes capturing and executing a random number generation within a CUDA graph.
        def simple_graph_task(graph, default_generator, generator_states):
            # [FUNCTION DOC] HELPER: On a fresh side stream, captures into the given graph
            # one random draw per supplied generator state (installing each state on the
            # default generator before drawing), then makes the current stream wait and
            # replays the graph.
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                graph.capture_begin()
                for generator_state in generator_states:
                    default_generator.graphsafe_set_state(generator_state)
                    torch.rand(1, device="cuda")
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            graph.replay()

        def get_memory_stats():
            # [FUNCTION DOC] HELPER: Returns a pair of the number of active allocation blocks
            # and total active bytes from the current memory statistics.
            stats = torch.cuda.memory_stats()
            num_blocks = stats["active.all.current"]
            total_size = stats["active_bytes.all.current"]
            return num_blocks, total_size

        def test(num_graphs, num_generators):
            # [FUNCTION DOC] HELPER: Core of the test for a graph count and generator-state
            # count. Clears the cache, records baseline stats, creates the graphs and generator
            # states (the first is the default's current state, the rest clones), captures and
            # replays each graph, then asserts block and byte growth match the expected per-pair
            # values and releases the graphs.
            clear_cuda_cache()
            baseline = get_memory_stats()
            baseline_num_blocks, baseline_total_size = baseline

            # Allocate CUDA graphs
            graphs = [torch.cuda.CUDAGraph() for _ in range(num_graphs)]

            # Allocate and manage generator states
            default_generator = torch.cuda.default_generators[0]
            generator_states = [default_generator.graphsafe_get_state()]

            for _ in range(1, num_generators):
                generator_states.append(default_generator.clone_state())

            for graph in graphs:
                simple_graph_task(graph, default_generator, generator_states)

            # Assert conditions after graph tasks
            num_blocks, total_size = get_memory_stats()
            # Each (generator, graph) pair gets 2 tensors (seed + offset)
            expected_captured_states = num_generators * num_graphs
            expected_blocks_diff = 2 * expected_captured_states
            expected_size_diff = 2 * 512 * expected_captured_states

            self.assertEqual(
                (num_blocks - baseline_num_blocks),
                expected_blocks_diff,
                "Unexpected number of active blocks.",
            )
            self.assertEqual(
                (total_size - baseline_total_size),
                expected_size_diff,
                "Unexpected total memory size.",
            )

            # Cleanup graphs and clear CUDA cache
            while graphs:
                graph = graphs.pop()
                del graph
            clear_cuda_cache()

        # Running the test function with different parameters
        test(1, 1)
        test(3, 2)
        test(10, 20)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reset_recapture(self):
        # [FUNCTION DOC] Captures ten additions in a graph on a side stream, replays it and
        # checks the sum (11000); then resets the graph, recaptures ten additions of 2 after
        # filling the buffer with 2, replays and checks the sum (22000); resets and deletes it.
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

        g.reset()

        with torch.cuda.stream(s):
            g.capture_begin()
            b.fill_(2.0)
            for _ in range(10):
                b = b + 2
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 22000.0)

        g.reset()
        del g

    @unittest.skipUnless(
        TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs; "
    )
    @requires_cuda_python_bindings  # required for debug_dump
    def test_graph_debugdump(self):
        # [FUNCTION DOC] Tests graph debug dumps (requires the cuda python bindings). Writes a
        # dot file from a graph in debug mode, from a graph built with keep_graph, and, for a
        # default graph, from an export hook run at capture end; each file must be non-empty. A
        # direct dump on the default graph after capture must raise RuntimeError saying no graph
        # handle is available.
        torch.cuda.empty_cache()
        x = torch.randn(1024, device="cuda")
        y = torch.rand_like(x)

        def capture(g):
            # [FUNCTION DOC] HELPER: Captures a small two-stream fork/join computation (an add
            # and a dependent add on a second stream) into the given graph, then synchronizes.
            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                g.capture_begin()
                z = x + y
                with torch.cuda.stream(s1):
                    s1.wait_stream(s0)
                    z + y
                s0.wait_stream(s1)
                g.capture_end()
            s0.synchronize()
            torch.cuda.synchronize()

        with tempfile.TemporaryDirectory() as tempdir:
            # enable_debug_mode retains the template (== keep_graph), so dump works.
            g = torch.cuda.CUDAGraph()
            g.enable_debug_mode()
            capture(g)
            p = os.path.join(tempdir, "debug_mode.dot")
            g.debug_dump(p)
            self.assertGreater(os.path.getsize(p), 0)

            # keep_graph=True: template persists, dump works.
            g2 = torch.cuda.CUDAGraph(keep_graph=True)
            capture(g2)
            p2 = os.path.join(tempdir, "keep_graph.dot")
            g2.debug_dump(p2)
            self.assertGreater(os.path.getsize(p2), 0)

            # keep_graph=False: the template is destroyed at finalize, so a plain
            # debug_dump after capture raises -- but export_dot as a capture-end
            # hook runs while the template is still live.
            g3 = torch.cuda.CUDAGraph()
            p3 = os.path.join(tempdir, "hook.dot")
            g3.register_capture_end_hook(torch.cuda.graphs.export_dot(p3))
            capture(g3)
            self.assertGreater(os.path.getsize(p3), 0)
            with self.assertRaisesRegex(
                RuntimeError, r"No (cuda|hip)Graph_t is available"
            ):
                g3.debug_dump(os.path.join(tempdir, "nope.dot"))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_and_instantiate_hooks(self):
        # [FUNCTION DOC] Tests graph hooks. Capture-end hooks fire in registration order and a
        # removed hook does not fire (expected "a" then "c"). Post-instantiate hooks fire on
        # each instantiate call, including re-instantiation, so two calls give two entries.
        x = torch.randn(8, device="cuda")

        # capture-end hooks fire in registration order; removable.
        order = []
        g = torch.cuda.CUDAGraph(keep_graph=True)
        g.register_capture_end_hook(lambda _g: order.append("a"))
        h = g.register_capture_end_hook(lambda _g: order.append("b"))
        h.remove()
        g.register_capture_end_hook(lambda _g: order.append("c"))
        with torch.cuda.graph(g):
            x + 1
        self.assertEqual(order, ["a", "c"])

        # post-instantiate hooks fire on each instantiate, including re-instantiate.
        instantiated = []
        g.register_post_instantiate_hook(lambda _g: instantiated.append(1))
        g.instantiate()
        g.instantiate()
        self.assertEqual(len(instantiated), 2)

    def _capture_trivial_graph(self):
        # [FUNCTION DOC] HELPER: Warms up a small tensor on a side stream with three additions,
        # then captures one addition into a new CUDA graph through the graph context manager and
        # returns the graph.
        g = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                x = x + 1
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            x + 1
        return g

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_register_destroy_callback(self):
        # [FUNCTION DOC] Checks a destroy callback registered on a graph fires exactly once when
        # the graph is deleted and collected.
        # Fires once on destruction; a graph-free sentinel observes it.
        g = self._capture_trivial_graph()
        calls = []
        g.register_destroy_callback(lambda: calls.append(1))
        del g
        gc.collect()
        self.assertEqual(calls, [1])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_fires_on_reset_and_rearms(self):
        # [FUNCTION DOC] Checks a destroy callback fires when the graph is reset, then the
        # holder is re-armed: a callback registered after a fresh capture on a new graph fires
        # at its destruction, giving the order first then second.
        g = self._capture_trivial_graph()
        calls = []
        g.register_destroy_callback(lambda: calls.append("first"))
        g.reset()
        # also-fire-on-reset: fires once, then the holder re-arms empty.
        self.assertEqual(calls, ["first"])
        # A fresh capture cycle can register on the re-armed holder.
        g = self._capture_trivial_graph()
        g.register_destroy_callback(lambda: calls.append("second"))
        del g
        gc.collect()
        self.assertEqual(calls, ["first", "second"])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_no_double_fire(self):
        # [FUNCTION DOC] Checks that after a reset fires the destroy callback, deleting the
        # graph does not fire it a second time.
        g = self._capture_trivial_graph()
        calls = []
        g.register_destroy_callback(lambda: calls.append(1))
        g.reset()
        del g
        gc.collect()
        self.assertEqual(calls, [1])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_object_released_on_destroy(self):
        # [FUNCTION DOC] Checks retain_object keeps an object alive while the graph lives and
        # releases it once the graph is destroyed, observed via a weak reference.
        g = self._capture_trivial_graph()

        class _Obj:
            # [CLASS DOC] HELPER: Empty class whose instance is retained by the graph and
            # tracked through a weak reference.
            pass

        obj = [_Obj()]
        ref = weakref.ref(obj[0])
        g.retain_object(obj[0])
        obj.clear()
        gc.collect()
        self.assertIsNotNone(ref(), "retained object stays alive while graph is alive")
        del g
        gc.collect()
        self.assertIsNone(ref(), "retained object released when graph is destroyed")

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_callback_exception_isolated(self):
        # [FUNCTION DOC] Checks that an exception raised by one destroy callback during
        # finalization does not propagate or stop later callbacks: a second callback still runs.
        g = self._capture_trivial_graph()
        calls = []
        g.register_destroy_callback(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        g.register_destroy_callback(lambda: calls.append("after"))
        del g
        gc.collect()  # must not raise out of finalization
        self.assertEqual(calls, ["after"])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_removable_handle(self):
        # [FUNCTION DOC] Checks that the handle returned from registering a destroy callback can
        # be used to remove it, so deleting the graph afterward calls nothing.
        g = self._capture_trivial_graph()
        calls = []
        handle = g.register_destroy_callback(lambda: calls.append(1))
        handle.remove()
        del g
        gc.collect()
        self.assertEqual(calls, [])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_sync_before_release_records_stream(self):
        # [FUNCTION DOC] Registers a destroy callback with synchronize-before-release, checks
        # the internal retained state flags it, that four replays on one stream record exactly
        # one replay stream, and that the callback fires once after deletion.
        g = self._capture_trivial_graph()
        freed = []
        g.register_destroy_callback(
            lambda: freed.append(1), synchronize_before_release=True
        )
        self.assertTrue(g._retained.sync_before_fire)
        # Same-stream replays dedup to a single recorded stream.
        for _ in range(4):
            g.replay()
        self.assertEqual(len(g._retained.replay_streams), 1)
        del g
        gc.collect()
        self.assertEqual(freed, [1])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_sync_records_all_replay_streams(self):
        # [FUNCTION DOC] With synchronize-before-release enabled, replays the graph on three
        # different streams and asserts the retained state recorded all three streams as a set;
        # the callback fires once after the graph is deleted.
        g = self._capture_trivial_graph()
        freed = []
        g.register_destroy_callback(
            lambda: freed.append(1), synchronize_before_release=True
        )
        # Concurrent replays on distinct streams must all be recorded so fire()
        # drains every stream that may still have work in flight.
        streams = [torch.cuda.Stream() for _ in range(3)]
        for s in streams:
            with torch.cuda.stream(s):
                g.replay()
        self.assertEqual(g._retained.replay_streams, set(streams))
        del g
        gc.collect()
        self.assertEqual(freed, [1])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_retain_graph_reference_pins_graph(self):
        # [FUNCTION DOC] Documents a footgun: retaining a list containing the graph itself pins
        # the graph because the global finalizer can reach it. The weak reference stays alive
        # after deletion and collection; the test then detaches the finalizer and clears the
        # retained objects so the graph is finally collected.
        # Documented footgun: a retained object reachable to the graph makes the
        # globally-held finalizer reach the graph, so it is never collected.
        g = self._capture_trivial_graph()
        g.retain_object([g])
        ref = weakref.ref(g)
        del g
        gc.collect()
        pinned = ref()
        self.assertIsNotNone(pinned, "graph reachable from a retained object is pinned")
        # Break the deliberate leak so the test does not strand a graph.
        pinned._retained_finalizer.detach()
        pinned._retained.objects.clear()
        del pinned
        gc.collect()
        self.assertIsNone(ref())

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_replay_hooks(self):
        # [FUNCTION DOC] Tests replay hooks. Start and end hooks fire in registration order on
        # every replay (two replays give start, end, start, end); removing the end hook leaves
        # only start; removing both gives nothing. Then, with replay patched to raise
        # RuntimeError, the end hook still fires after start and the error propagates.
        x = torch.randn(8, device="cuda")

        # replay hooks fire start-then-end, in registration order, on every
        # replay; removable, and a no-op (hot path) once removed.
        events: list[str] = []
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x + 1
        start_handle = g.register_replay_start_hook(lambda _g: events.append("start"))
        end_handle = g.register_replay_end_hook(lambda _g: events.append("end"))
        g.replay()
        g.replay()
        self.assertEqual(events, ["start", "end", "start", "end"])
        end_handle.remove()
        events.clear()
        g.replay()
        self.assertEqual(events, ["start"])
        start_handle.remove()
        events.clear()
        g.replay()
        self.assertEqual(events, [])

        # end hook fires even if the launch raises (start balanced by end), and
        # the launch error still propagates.
        events.clear()
        g.register_replay_start_hook(lambda _g: events.append("start"))
        g.register_replay_end_hook(lambda _g: events.append("end"))
        with unittest.mock.patch.object(
            torch._C._CUDAGraph, "replay", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                g.replay()
        self.assertEqual(events, ["start", "end"])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH,
        "CUDA >= 11.0 / ROCm >= 7.0 required for external events in cuda graphs",
    )
    @skipIfRocmVersionLessThan((7, 0))
    def test_graph_timing(self):
        # [FUNCTION DOC] Captures a graph with events marked external recording start and end
        # around an addition of two 10M-element tensors, replays it and asserts the elapsed time
        # between the two events is positive. Requires graph support and ROCm 7.0+.
        torch.cuda.empty_cache()
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        start_event = torch.cuda.Event(enable_timing=True, external=True)
        end_event = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(g):
            start_event.record()
            z = x + y
            end_event.record()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(start_event.elapsed_time(end_event) > 0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_error(self):
        # [FUNCTION DOC] Runs a child process that begins capture on the default stream and
        # exits with code 0 if the expected "non-default stream" RuntimeError arises, 1 for a
        # different error, 2 if no error. The parent fails the test for exit code 1 or 2.
        # We need to run this test in a separate thread as the error we trigger
        # puts the cuda context in a bad state
        script = """
import torch

g = torch.cuda.CUDAGraph()
try:
    g.capture_begin()
except RuntimeError as e:
    if "CUDA graphs must be captured on a non-default stream." in str(e):
        exit(0)
    else:
        exit(1)
exit(2)
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(
                    False,
                    "Error raise by starting capture without a stream is not the expected one",
                )
            elif e.returncode == 2:
                self.assertTrue(
                    False,
                    "Error raised by starting capture without a stream was not caught",
                )

    @unittest.skipIf(
        (not TEST_CUDA),
        "CUDA >= 11.0 required for graphs",
    )
    def test_graph_warn_if_has_zero_nodes(self):
        # [FUNCTION DOC] Captures an empty graph on a side stream while recording warnings and
        # asserts a warning saying the CUDA graph is empty was issued.
        with warnings.catch_warnings(record=True) as caught:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g.capture_begin()
                g.capture_end()
        self.assertTrue(
            any("The CUDA Graph is empty" in str(w.message) for w in caught)
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_graph_capture_oom(self):
        # [FUNCTION DOC] Asserts that allocating an enormous tensor (2 to the 40 elements)
        # during graph capture raises RuntimeError with an out-of-memory message (allowed memory
        # wording under cudaMallocAsync). Skipped on Jetson.
        oom_regex = (
            "would exceed allowed memory" if TEST_CUDAMALLOCASYNC else "out of memory"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                torch.zeros(2**40, device="cuda")

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/144922")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    @blas_library_context("cublas")
    def test_repeat_graph_capture_cublas_workspace_memory(self):
        # [FUNCTION DOC] Regression test for cuBLAS workspace growth over repeated captures:
        # after a warmup matmul, captures and replays a matmul 100 times, recording used GPU
        # memory after the first iteration and at the end; asserts the growth is under 5 MB.
        # Uses the cublas backend, serial, skipped on ROCm.
        (x, y, z) = 1024, 512, 64
        a = torch.rand((x, y), device="cuda")
        b = torch.rand((y, z), device="cuda")

        # warmup
        torch.mm(a, b)

        for i in range(100):
            torch_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(torch_graph):
                torch.mm(a, b)
            torch_graph.replay()
            if i == 0:
                free_bytes_before, total_bytes = torch.cuda.mem_get_info()
                used_gb_before = (total_bytes - free_bytes_before) / 1e9

        free_bytes_after, _ = torch.cuda.mem_get_info()
        used_gb_after = (total_bytes - free_bytes_after) / 1e9

        self.assertGreater(0.005 + used_gb_before, used_gb_after)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_functional(self):
        # [FUNCTION DOC] Checks RNG state handling of graphs for functional ops dropout and
        # rrelu (training). For each op, compares a graphed, eager, graphed sequence of six
        # invocations against an eager control, then for several seeds checks that replay output
        # changes with the seed and equals the same ops run eagerly. Errors are wrapped in
        # RuntimeError naming the op.
        ops_with_kwargs = (
            (torch.nn.functional.dropout, {"p": 0.1}),
            (torch.nn.functional.rrelu, {"training": True}),
        )
        size = 10000

        def run(op, kwargs):
            # [FUNCTION DOC] HELPER: Runs the control/graph comparison described above for one
            # operation and its keyword arguments, and synchronizes at the end.
            a = torch.randn((size,), device="cuda", dtype=torch.float)

            # Control
            torch.cuda.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # Do the same operations varying seeds
            seeds = [6, 128, 9999]

            for seed in seeds:
                torch.cuda.manual_seed(seed)
                graph_in.copy_(a)
                for _ in range(3):
                    g.replay()

                # If the random seed was not updated then the graph would
                # generate the same output as in previous check.
                try:
                    self.assertNotEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

                # Now repeat the same operations in non-graphed mode.
                torch.cuda.manual_seed(seed)
                for _ in range(3):
                    eager_out.copy_(a)
                    eager_out = op(eager_out, **kwargs)
                    eager_out = op(eager_out, **kwargs)

                # In the end, graph_out and eager_out must be equal
                # as they went under the same set of operations.
                try:
                    self.assertEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_distributions(self):
        # [FUNCTION DOC] Checks RNG state handling in graphs for torch functions (bernoulli,
        # normal variants, poisson, rand, randint, randn) and tensor methods (bernoulli, cauchy,
        # exponential, geometric, log_normal, normal, random, uniform). Each op is captured
        # twice in a graph; replays after seeding with several seeds must reproduce the eager
        # control outputs. Multinomial is not covered because it is not capturable.
        size = 10000
        input = torch.rand((size,), device="cuda", dtype=torch.float)
        alloc = torch.empty((size,), device="cuda", dtype=torch.float)

        # Torch ops to test with sample args (tuple) and kwargs (dict)
        torch_with_args = (
            ("bernoulli", (input.clone(),), {}),
            # multinomial uses some uncapturable CUDA calls.
            # TODO: reenable multinomial tests if/when the implementation is capturable.
            # ("multinomial", (input.clone(), size, True), {}),
            # ("multinomial", (input.clone(), size // 2, False), {}),
            ("normal", (input.clone() + 1, input.clone()), {}),
            ("normal", (input.clone() + 1, 1.0), {}),
            ("poisson", (input.clone(),), {}),
            ("rand", (size,), {"device": "cuda", "dtype": torch.float}),
            ("randint", (0, 3, (size,)), {"device": "cuda", "dtype": torch.float}),
            ("randn", (size,), {"device": "cuda", "dtype": torch.float}),
        )

        # Tensor methods to test with sample args (tuple)
        tensor_with_args = (
            ("bernoulli_", (input.clone(),)),
            ("cauchy_", ()),
            ("exponential_", ()),
            ("geometric_", (0.3,)),
            ("log_normal_", ()),
            ("normal_", ()),
            ("random_", ()),
            ("uniform_", ()),
        )

        def run(module, op, args, kwargs):
            # [FUNCTION DOC] HELPER: For one module label (torch or Tensor), operation, args and
            # kwargs: computes eager controls after a dummy op, captures two invocations in a
            # graph, optionally checks values were not populated at capture (native allocator
            # only), and for three seeds replays and compares with controls. Wraps assertion
            # failures in RuntimeError naming the op.
            torch.cuda.manual_seed(5)

            # Each path runs a dummy op to increment the state a bit before creating controls.
            if module == "torch":
                dummy = getattr(torch, op)(*args, **kwargs)
                control1 = getattr(torch, op)(*args, **kwargs)
                control2 = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                control1 = alloc.clone()
                control2 = alloc.clone()
                getattr(dummy, op)(*args)
                getattr(control1, op)(*args)
                getattr(control2, op)(*args)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                if module == "torch":
                    g.capture_begin()
                    t1 = getattr(torch, op)(*args, **kwargs)
                    t2 = getattr(torch, op)(*args, **kwargs)
                    g.capture_end()
                else:
                    t1 = alloc.clone()
                    t2 = alloc.clone()
                    g.capture_begin()
                    getattr(t1, op)(*args)
                    getattr(t2, op)(*args)
                    g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            if not TEST_CUDAMALLOCASYNC:
                # Makes sure values haven't been populated yet
                # (in other words, makes sure capture didn't actually run ops).
                # We can only try this with the native allocator, for which captured
                # addresses are already backed by cudaMalloced memory.
                # If we try it with cudaMallocAsync, CUDA won't event consider
                # the captured addresses allocated until replay(), and if we
                # access them before replay() we get IMAs.
                try:
                    self.assertNotEqual(control1, t1)
                    self.assertNotEqual(control2, t2)
                except Exception as e:
                    raise RuntimeError("Failed on " + module + "." + op) from e

            # Set a new seed to check if graph would use it
            for seed in [6, 314, 271]:
                torch.cuda.manual_seed(seed)
                # Runs a dummy op prelude, as for controls, to make sure replay()
                # picks up the dummy op's state increment.
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                    control1 = getattr(torch, op)(*args, **kwargs)
                    control2 = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)
                    getattr(control1, op)(*args)
                    getattr(control2, op)(*args)

                torch.cuda.manual_seed(seed)
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)

                # see above comment on TEST_CUDAMALLOCASYNC
                if not TEST_CUDAMALLOCASYNC:
                    t1.copy_(alloc)
                    t2.copy_(alloc)

                # Runs RNG ops that fill t1 and t2.
                g.replay()

                try:
                    self.assertEqual(control1, t1)
                    self.assertEqual(control2, t2)
                except Exception as e:
                    raise RuntimeError("Failed on " + module + "." + op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op_with_args in torch_with_args:
            run("torch", *op_with_args)

        for meth_with_args in tensor_with_args:
            # Adds an empty dict for kwargs, which none of the Tensor methods use
            run("Tensor", *(meth_with_args + ({},)))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_two_successive(self):
        # [FUNCTION DOC] Captures two graphs in succession on a stream under three pool modes
        # (no sharing, sharing via the first graph's pool, sharing via a pool handle),
        # interleaves eager work with replays and checks the sums (3070 and 442 per element).
        # For the native allocator, checks that sharing a pool lowers reserved bytes by one
        # small buffer (2 MiB) compared with no sharing.
        torch.cuda.empty_cache()

        size = 1000
        kSmallBuffer = 2097152

        def func_with_temps(t, val):
            # [FUNCTION DOC] HELPER: Clones the input twice, adds the value to each and returns
            # the sum of the two clones (creates temporary allocations).
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g1.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # mixes unrelated eager ops with replays
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            if not TEST_CUDAMALLOCASYNC:
                # These stat checks are specific to the native allocator.
                if share_mem != "Don't share":
                    self.assertEqual(
                        reserved_no_sharing  # noqa: F821
                        - torch.cuda.memory_stats()["reserved_bytes.all.current"],
                        kSmallBuffer,
                    )
                else:
                    reserved_no_sharing = torch.cuda.memory_stats()[
                        "reserved_bytes.all.current"
                    ]

            del a, b, c, g0, g1
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/104055")
    @unittest.skipIf(
        (not TEST_CUDA_GRAPH) or IS_WINDOWS,
        "Graph bindings disallow concurrent replay; "
        + "still appears broken on Windows. See "
        + "https://github.com/pytorch/pytorch/pull/57556",
    )
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_concurrent_replay(self):
        # [FUNCTION DOC] Replays two graphs concurrently on different streams after capturing
        # them in a stream under three pool modes. With the native allocator and shared pools
        # the results are expected to be corrupted (sums differ from 94 and 156 per element);
        # otherwise (no sharing, or cudaMallocAsync) the sums must be correct. Skipped on ROCm,
        # Windows or without graphs.
        torch.cuda.empty_cache()

        size = 1000000  # largeish to help expose race conditions

        def func_with_temps(t, val):
            # [FUNCTION DOC] HELPER: Clones the input twice, adds the value to each and returns
            # the sum of the two clones (creates temporary allocations).
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                c = a.clone()
                for _ in range(5):
                    c = func_with_temps(c, 2)
                g1.capture_end()

            # To reproduce data corruption, I need g0 and g1's kernels to run concurrently.
            # But replay() (especially cudaGraphLaunch) can incur significant CPU overhead.
            # The following pattern helps align device-side execution of g0 and g1's kernels.
            torch.cuda.synchronize()
            with torch.cuda.stream(s0):
                torch.cuda._sleep(1000000)
                s1.wait_stream(s0)
                g0.replay()
            with torch.cuda.stream(s1):
                g1.replay()
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.current_stream().wait_stream(s1)

            if (not TEST_CUDAMALLOCASYNC) and (share_mem != "Don't share"):
                # If we used the native allocator and shared mempools,
                # we expect the concurrent replays corrupted each other.
                self.assertNotEqual(b.sum().item(), size * 94)
                self.assertNotEqual(c.sum().item(), size * 156)
            else:
                # If we EITHER
                #   - used the native allocator without sharing mempools, OR
                #   - used cudaMallocAsync, which ignores graph pool-sharing hints and should always be safe
                # we don't expect memory corruption.
                self.assertEqual(b.sum().item(), size * 94)
                self.assertEqual(c.sum().item(), size * 156)

            del a, b, c, g0, g1
            # Tensors used across streams (a, b, c) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_three_successive(self):
        # [FUNCTION DOC] Captures three graphs in sequence (the second and third reuse the pool
        # or pool handle according to the three modes): the first makes two derived tensors, the
        # second adds 3 to one and drops its intermediate, the third adds 4 to the other.
        # Replays them in capture order and checks the sums are 5 and 7 per element.
        torch.cuda.empty_cache()

        size = 1000

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            a = torch.ones((size,), device="cuda")

            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()
            g2 = torch.cuda.CUDAGraph()

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                c = b + 1
                d = b + 2
                g0.capture_end()

                args = (g0.pool(),) if share_mem == "via pool()" else g0_args

                g1.capture_begin(*args)
                e = c + 3
                del c
                g1.capture_end()

                g2.capture_begin(*args)
                f = d + 4
                g2.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # Tests that replaying in capture order is valid
            g0.replay()
            g1.replay()
            g2.replay()

            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            # Tests that replaying as g0, g2, g1 is only valid if they don't share a pool
            g0.replay()
            g2.replay()
            g1.replay()

            expect_corruption = (not TEST_CUDAMALLOCASYNC) and (
                share_mem != "Don't share"
            )
            # If we used the native allocator and shared mempools, g2's capture should have reused c's memory for f.
            # We replayed g2 then g1, so we expect g1's captured "e = c + 3" mistakenly filled e with "f's vals + 3".
            self.assertEqual(
                e.sum().item(), size * (7 + 3) if expect_corruption else size * 5
            )
            self.assertEqual(f.sum().item(), size * 7)

            del a, b, d, e, f, g0, g1, g2
            # Tensors used across streams (a, e, f) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAGraph MemPool retention is specific to the native allocator",
    )
    def test_graph_mem_pool_arg_and_nested_use_mem_pool(self):
        # [FUNCTION DOC] Checks memory pool bookkeeping for graph capture with an explicit pool
        # argument and a nested use_mem_pool block (native allocator only). Captures three
        # additions with the default pool for the graph and a side pool for two of them; asserts
        # the graph pool id is the default pool id, the graph reports both pool ids, each pool
        # use count is 2, the graph still replays correctly after the side pool object is
        # deleted (result 8), and the default pool use count returns to 1 after reset.
        torch.cuda.empty_cache()

        g_default_pool = torch.cuda.MemPool()
        g_side_pool = torch.cuda.MemPool()

        x = torch.ones((4,), device="cuda")
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()

        with torch.cuda.graph(g, stream=stream, pool=g_default_pool):
            y = x + 1
            with torch.cuda.use_mem_pool(g_side_pool):
                tmp = y + 2
                tmp = tmp + 1
            z = tmp + 3
        torch.cuda.current_stream().wait_stream(stream)

        self.assertEqual(g.pool(), g_default_pool.id)
        self.assertEqual(set(g.pools()), {g_default_pool.id, g_side_pool.id})
        self.assertEqual(g_default_pool.use_count(), 2)
        self.assertEqual(g_side_pool.use_count(), 2)

        del g_side_pool
        gc.collect()
        g.replay()
        self.assertEqual(z, torch.full_like(z, 8))

        g.reset()
        self.assertEqual(g_default_pool.use_count(), 1)

    @unittest.skipIf(
        (not TEST_CUDA_GRAPH) or TEST_CUDAMALLOCASYNC,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs",
    )
    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        # [FUNCTION DOC] Validates caching allocator statistics around graph capture for five
        # tensor sizes (small pool, large pool boundary cases). For each case, after capture the
        # deltas of segment count, reserved bytes, active blocks and active bytes versus the
        # pre-capture stats must equal the expected values (adjusted for expandable segments),
        # also after an empty_cache and replay; after deleting the graph the private pool
        # shrinks to the single tensor still alive, and that tensor is still usable. Native
        # allocator only.
        kSmallSize = 1048576
        kSmallBuffer = 2097152
        kLargeBuffer = 20971520
        kMinLargeAlloc = 10485760
        kRoundLarge = 2097152

        elem = 4

        # this was annoying to write but stresses the expectations pretty rigorously
        cases = (
            (512 // elem, 1, 1 * kSmallBuffer, kSmallBuffer, "small_pool"),
            (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "small_pool"),
            ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
            (
                (kMinLargeAlloc - 512) // elem,
                2,
                2 * kLargeBuffer,
                kLargeBuffer,
                "large_pool",
            ),
            (
                (kMinLargeAlloc + 512) // elem,
                3,
                3
                * (
                    kRoundLarge
                    * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)
                ),
                kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                "large_pool",
            ),
        )

        stats_to_check = ("segment.", "reserved_bytes.", "active.", "active_bytes.")

        gc.collect()
        torch.cuda.empty_cache()

        s = torch.cuda.Stream()

        for (
            numel,
            delta_cudaMallocs,
            delta_cudaMalloc_bytes,
            delta_cudaMalloc_bytes_post_del_g,
            pool_string,
        ) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 1
                delta_active_bytes = numel * elem
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.cuda.CUDAGraph()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                a = torch.ones((numel,), device="cuda")

                precapture_stats = torch.cuda.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.cuda.memory_stats()

            expecteds = (
                delta_cudaMallocs,
                delta_cudaMalloc_bytes,
                delta_active_blocks,
                delta_active_bytes,
            )
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]

                    # There will only ever be one expandable segment in each of the small and large pools. The way the
                    # bookkeeping is done in the allocator means that we never increment the number of segments.
                    if self.expandable_segments and "segment" in stat:
                        expected = 0
                    # These two cases hit an edge case where the PyTorch allocator won't immediately unmap part of an
                    # expandable segment (and as a result reduce the number of reserved bytes) if the block to unmap is
                    # smaller than the page size
                    if (
                        self.expandable_segments
                        and "reserved" in stat
                        and (numel == cases[3][0] or numel == cases[4][0])
                    ):
                        expected = 2 * kLargeBuffer

                    self.assertEqual(
                        current,
                        expected,
                        "Pre to post capture delta of "
                        + stat
                        + f" = {current}, expected = {expected}, numel = {numel}",
                    )

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.cuda.empty_cache()

            del g
            gc.collect()
            torch.cuda.empty_cache()
            postdel_stats = torch.cuda.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_cudaMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]

                # There will only ever be one expandable segment in each of the small and large pools. The way the
                # bookkeeping is done in the allocator means that we never increment the number of segments.
                if self.expandable_segments and "segment" in stat:
                    expected = 0
                # These two cases hit an edge case where the PyTorch allocator won't immediately unmap part of an
                # expandable segment (and as a result reduce the number of reserved bytes) if the block to unmap is
                # smaller than the page size
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[3][0]
                ):
                    expected = 2 * kLargeBuffer
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[4][0]
                ):
                    expected = kLargeBuffer

                self.assertEqual(
                    current,
                    expected,
                    "Pre capture to post graph delete delta of "
                    + stat
                    + f" = {current}, expected = {expected}, numel = {numel}",
                )

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        (not TEST_CUDA_GRAPH) or (not TEST_CUDAMALLOCASYNC),
        "graph-mem-pool stat undercount is specific to the cudaMallocAsync backend",
    )
    def test_graph_mem_counted_in_reserved_async(self):
        # [FUNCTION DOC] For the cudaMallocAsync backend only: checks that memory_reserved
        # counts the per-device graph memory pool. Captures four 256 MiB buffers in a graph,
        # replays it (memory is committed at first replay), and asserts reserved memory grew by
        # at least 1 GiB; a second graph must grow it again. Resets graphs, frees buffers and
        # clears peak stats at the end.
        # cudaMallocAsync graph captures reserve backing in the per-device graph-mem
        # pool, not the default mempool; memory_reserved() must include it. That pool
        # is shared by all graphs, so a second graph must grow reserved again.
        device = torch.cuda.current_device()

        CHUNK = 256 * 1024 * 1024  # 256 MiB
        NCHUNK = 4  # 1 GiB per graph
        expected = CHUNK * NCHUNK

        def capture_and_replay():
            # [FUNCTION DOC] HELPER: Warms up a 256 MiB allocation on a side stream and frees
            # it, then captures four such allocations in a new graph, replays it and
            # synchronizes. Returns the graph and the list of captured buffers.
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # warm up workspace allocs outside capture
                _ = torch.empty(CHUNK, dtype=torch.uint8, device=device)
            torch.cuda.current_stream().wait_stream(s)
            del _
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            buffers = []
            with torch.cuda.graph(g):
                for _ in range(NCHUNK):
                    buffers.append(torch.empty(CHUNK, dtype=torch.uint8, device=device))
            g.replay()  # graph-mem commits at first replay, not at capture
            torch.cuda.synchronize()
            return g, buffers

        def assert_grew(before, after):
            # [FUNCTION DOC] HELPER: Asserts the reserved byte count increased by at least the
            # expected 1 GiB between the two readings, with a message about undercounting.
            # The NCHUNK live tensors can't be reused within the graph and reserved
            # only rounds up, so growth is exactly `expected`.
            self.assertGreaterEqual(
                after - before,
                expected,
                f"memory_reserved() undercounts the graph-mem pool: grew "
                f"{after - before}, expected >= {expected}",
            )

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        r0 = torch.cuda.memory_reserved(device)
        g1, b1 = capture_and_replay()
        r1 = torch.cuda.memory_reserved(device)
        assert_grew(r0, r1)

        g2, b2 = capture_and_replay()
        r2 = torch.cuda.memory_reserved(device)
        assert_grew(r1, r2)

        # Reset the graphs before freeing the tensors that alias their alloc nodes,
        # then clear stats so this test's ~2 GiB doesn't bleed into siblings.
        g1.reset()
        g2.reset()
        del b1, b2, g1, g2
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_record_stream(self):
        # [FUNCTION DOC] Checks that graph capture defers reclaiming allocations used across
        # streams. A tensor recorded on a busy side stream is freed, then a capture on another
        # stream clones a tensor, records a third stream on the clone and deletes it; the
        # capture must succeed. A later plain allocation then processes the pending
        # end-of-life events.
        # Makes sure graph capture defers attempting to reclaim allocations used across streams. See
        # "Q. Why skip process_events if a capture might be underway?" in c10/cuda/CUDACachingAllocator.cpp
        torch.cuda.empty_cache()

        potential_problem = torch.zeros((3,), device="cuda")
        a = torch.zeros((3,), device="cuda")
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()
        with torch.cuda.stream(s0):
            potential_problem.record_stream(s0)
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            potential_problem.fill_(1.0)
        del potential_problem

        with torch.cuda.stream(s1):
            g.capture_begin()
            # potential_problem's allocation should still be outstanding. if DeviceCachingAllocator::malloc
            # mistakenly calls process_events, it will trigger cudaEventQueries on potential_problem's end-of-life
            # event, which will cause the capture to error.
            b = a.clone()

            # Let's also see what happens if we record_stream on a tensor during capture.
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                b.fill_(1.0)
                b.record_stream(s2)  # dummy record_stream
                del b
            s1.wait_stream(s2)
            g.capture_end()
        torch.cuda.synchronize()

        # dummy allocation triggers process_events, Hopefully successfully processes b's end-of-life event.
        torch.zeros((3,), device="cuda")

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # If this test is the first in the process to try cudnn rnns with dropout, it'll initialize
    # DropoutState's long-lived internal buffer. Calling code perceives this (correct) behavior
    # as a memory leak unless we skip the leak check.
    @skipCUDAMemoryLeakCheckIf(True)
    @serialTest()
    def test_graph_cudnn_dropout(self):
        # [FUNCTION DOC] Tests cuDNN RNN dropout state with graph capture: runs a two-layer LSTM
        # with dropout once eagerly, captures one forward in a graph on a side stream, replays
        # it, then runs eagerly again; passes if no stream sync error arises. Memory leak check
        # is skipped because the first dropout use initializes a long-lived buffer. Serial.
        # Tests the interaction of cuda graph capture with DropoutState's syncs in ATen/native/cudnn/RNN.cpp.
        # In particular, if user runs a sequence of captured and noncaptured cudnn rnns, DropoutState should
        # avoid syncing noncapturing streams with captured events or vice versa.
        torch.cuda.empty_cache()

        model = torch.nn.LSTM(512, 512, 2, dropout=0.5).cuda()
        x = torch.ones(100, 192, 512, device="cuda")

        model(x)

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g.capture_begin()
            model(x)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        model(x)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    def test_graph_checkpoint_preserve_rng_state(self):
        # [FUNCTION DOC] Checks that activation checkpointing with RNG state preservation
        # gives the same gradient when captured in a graph as eagerly. Seeds with 42, computes
        # the eager gradient of a function using a random draw through non-reentrant
        # checkpoint, captures the same computation in a graph, reseeds with 42, replays, and
        # asserts exact equality.
        torch.cuda.manual_seed(42)

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by the sigmoid of a random normal draw
            # (one element on CUDA).
            return x * torch.sigmoid(torch.randn(1, device="cuda"))

        fn(torch.ones(1, device="cuda"))

        torch.cuda.manual_seed(42)
        eager_in = torch.ones(1, device="cuda", requires_grad=True)
        eager_out = torch.utils.checkpoint.checkpoint(
            fn, eager_in, use_reentrant=False, preserve_rng_state=True
        )
        (eager_in_grad,) = torch.autograd.grad(eager_out, eager_in)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_in = torch.ones(1, device="cuda", requires_grad=True)
            graph_out = torch.utils.checkpoint.checkpoint(
                fn, graph_in, use_reentrant=False, preserve_rng_state=True
            )
            (graph_in_grad,) = torch.autograd.grad(graph_out, graph_in)

        torch.cuda.manual_seed(42)
        g.replay()

        self.assertEqual(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    def test_graph_manual_seed_mismatch_raises(self):
        # [FUNCTION DOC] Asserts that calling manual_seed with a different seed during graph
        # capture raises RuntimeError saying the seed can only be set during capture if it
        # equals the original seed.
        torch.cuda.manual_seed(0)
        g = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(
            RuntimeError,
            "CUDAGeneratorImpl::set_current_seed can be called during stream capture only if new seed is the same as the original seed.",
        ):
            with torch.cuda.graph(g):
                torch.cuda.manual_seed(1)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[unittest.expectedFailure]),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        # [FUNCTION DOC] Parametrized by autocast usage, cache setting and allow_unused_input
        # (the listed combinations are marked expected failures). Builds two identical
        # three-part models plus a relu and a loss function, graphs four callables (including
        # two with dictionary inputs) with make_graphed_callables under autocast, trains graphed
        # and control versions on ten random batches with the same seeds and SGD, then asserts
        # parameters are equal. Also checks that eval mode still works on the graphed model.
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)

        N, D_in, H, D_out = 640, 4096, 2048, 1024

        class MLP1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two sequential blocks of a linear layer and
            # dropout (probabilities 0.1 and 0.2) moved to CUDA; takes a dictionary input and
            # uses only its x entry.
            def __init__(self, D_in: int, H: int, D_out: int):
                # [FUNCTION DOC] HELPER: Builds two sequential blocks of linear plus dropout
                # (0.1 then 0.2) from the given sizes and moves them to CUDA.
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).cuda()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).cuda()

            def forward(self, input_dict: dict):
                # [FUNCTION DOC] HELPER: Takes the x entry of the input dictionary and returns
                # it passed through both blocks.
                x = input_dict["x"]
                return self.net_2(self.net_1(x))

        class MLP2(torch.nn.Module):
            # [CLASS DOC] HELPER: Same structure as the previous module (two linear plus dropout
            # blocks on CUDA) but takes a plain tensor input.
            def __init__(self, D_in: int, H: int, D_out: int):
                # [FUNCTION DOC] HELPER: Builds two sequential blocks of linear plus dropout
                # (0.1 then 0.2) from the given sizes and moves them to CUDA.
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).cuda()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).cuda()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes the tensor through both blocks and returns it.
                return self.net_2(self.net_1(x))

        class ParameterlessModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module without parameters whose forward gathers rows of the
            # input along dimension 0 using an index grid, returning a dictionary with the
            # key output.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds an index grid of row numbers repeated across
                # columns, gathers rows of x with it along dimension 0 and returns a dictionary
                # holding it under the key output.
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = MLP1(D_in, H, H).cuda()
            model_section2 = MLP2(H, H, D_out).cuda()
            model_section3 = ParameterlessModule().cuda()
            models.append(
                torch.nn.Sequential(model_section1, model_section2, model_section3)
            )

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        opt_graphed = torch.optim.SGD(model_graphed.parameters(), lr=0.1)
        opt_control = torch.optim.SGD(model_control.parameters(), lr=0.1)

        x = torch.randn(N, D_in, device="cuda")
        h = torch.randn(N, H, device="cuda", requires_grad=True)
        h2 = torch.randn(N, D_out, device="cuda", requires_grad=True)
        unused_input = torch.randn(N, H, device="cuda", requires_grad=True)
        y_pred = torch.randn(N, D_out, device="cuda", requires_grad=True)
        y = torch.randn(N, D_out, device="cuda")

        loss_fn_control = torch.nn.functional.mse_loss
        relu_control = torch.nn.functional.relu

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
        ):
            (
                model_graphed[0],
                model_graphed[1],
                model_graphed[2],
                relu_graphed,
                loss_fn_graphed,
            ) = torch.cuda.make_graphed_callables(
                (
                    model_graphed[0],
                    model_graphed[1],
                    model_graphed[2],
                    relu_control,
                    loss_fn_control,
                ),
                (
                    ({"x": x, "unused_input": unused_input},),
                    (h,),
                    (h2,),
                    (y_pred,),
                    (y_pred, y),
                ),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m, opt, relu, loss_fn in zip(
            (model_graphed, model_control),
            (opt_graphed, opt_control),
            (relu_graphed, relu_control),
            (loss_fn_graphed, loss_fn_control),
        ):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.cuda.manual_seed(5)
            for data, target in zip(real_inputs, real_targets):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    y_pred = m({"x": data, "unused_input": unused_input})["output"]
                    y_pred = relu(y_pred)
                    loss = loss_fn(y_pred, target)
                    loss.backward()
                opt.step()

        for p, pc in zip(model_graphed.parameters(), model_control.parameters()):
            self.assertEqual(p, pc)

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((False, False, True)),
            subtest((True, False, True)),
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False)),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables_parameterless_nograd_module(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        # [FUNCTION DOC] Parametrized over autocast settings and allow_unused_input. Graphs a
        # single parameterless module that takes a dictionary input and an extra unused input
        # (none requiring gradients) with make_graphed_callables under autocast, runs ten
        # iterations on both graphed and control models with identical seeds, and checks eval
        # mode outputs of both models still match. One combination is an expected failure.
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)

        N, D_in, H, _ = 640, 4096, 2048, 1024

        class ParameterlessModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parameterless module taking a dictionary input; gathers rows
            # of the x entry with an index grid and returns a dictionary under the key output.
            def forward(self, input_dict: dict):
                # [FUNCTION DOC] HELPER: Reads the x entry, gathers rows with a repeated
                # row-index grid along dimension 0 and returns a dictionary with the key output.
                x = input_dict["x"]
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = ParameterlessModule().cuda()
            models.append(torch.nn.Sequential(model_section1))

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        x = torch.randn(N, D_in, device="cuda", requires_grad=False)
        unused_input = torch.randn(N, H, device="cuda", requires_grad=False)
        y = torch.randn(N, D_in, device="cuda")

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
        ):
            model_graphed[0] = torch.cuda.make_graphed_callables(
                model_graphed[0],
                ({"x": x, "unused_input": unused_input},),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x, requires_grad=True) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m in (model_graphed, model_control):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.cuda.manual_seed(5)
            for data, _ in zip(real_inputs, real_targets):
                with torch.amp.autocast(
                    device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    m({"x": data, "unused_input": unused_input})["output"]

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_make_graphed_callables_same_pool(self):
        # [FUNCTION DOC] Builds three small sequential models, makes a graphed copy of each
        # using one shared memory pool handle, then compares each model with its graphed copy on
        # random input: outputs, loss, parameters and parameter gradients are equal while the
        # parameter and gradient storages are distinct.
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)
        models = []
        num_models = 3
        for _ in range(num_models):
            models.append(
                torch.nn.Sequential(
                    torch.nn.Linear(32, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                ).cuda()
            )
        # we will reuse the same pool for all graph captures
        mempool = torch.cuda.graph_pool_handle()
        graphed_models = []
        for model in models:
            x = torch.randn([64, 32], device="cuda")
            graphed_model = deepcopy(model)
            graphed_model = torch.cuda.make_graphed_callables(
                graphed_model, (x,), pool=mempool
            )
            graphed_models.append(graphed_model)

        for model, graphed_model in zip(models, graphed_models):
            x = torch.randn([64, 32], device="cuda")
            y = model(x)
            yg = graphed_model(x)
            l = y.norm()
            lg = yg.norm()
            l.backward()
            lg.backward()

            self.assertEqual(y, yg)
            self.assertEqual(l, lg)
            for p, pg in zip(model.parameters(), graphed_model.parameters()):
                self.assertEqual(p, pg)
                self.assertEqual(p.grad, pg.grad)
                self.assertNotEqual(p.data_ptr(), pg.data_ptr())
                self.assertNotEqual(p.grad.data_ptr(), pg.grad.data_ptr())

    def test_cuda_graph_inference_mode(self):
        # [FUNCTION DOC] Checks that making a graphed callable inside inference mode does not
        # make RNG state tensors inference tensors: the callable captured in inference mode
        # returns the right result, and a later normal-mode capture and plain call still work.
        # This test is to verify that capturing a CUDAGraph in inference mode
        # doesn't create RNG State tensors as inference tensors which can't
        # be inplace modified later.

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        x = torch.randn(10, 10, device="cuda")

        with torch.inference_mode():
            captured_fn = torch.cuda.make_graphed_callables(fn, (x,))
            inp = torch.ones(10, 10, device="cuda")
            self.assertEqual(captured_fn(inp), inp + 1)

        torch.cuda.make_graphed_callables(fn, (x,))
        inp = torch.ones(10, 10, device="cuda") * 10
        self.assertEqual(fn(inp), inp + 1)

    def _test_graphed_optimizer(
        self, steps_warmup, steps_train, optimizer_ctor, kwargs
    ):
        # [FUNCTION DOC] HELPER: Shared check for optimizers. For both graphed and ungraphed
        # modes, creates identical parameters, runs a control optimizer with capturable off for
        # warmup plus train steps using the same pre-generated gradients, then an optimizer with
        # capturable on: warmup steps eager, then (if graphing) captures one step and replays it
        # for training, copying new gradients in before each replay, else steps eagerly. Asserts
        # parameters equal.
        for actually_do_graphs in (True, False):
            params = [torch.randn((i + 5, i + 5), device="cuda") for i in range(2)] + [
                torch.randn((), device="cuda")
            ]
            params_control = [p.clone().requires_grad_() for p in params]
            params_graphed = [p.clone().requires_grad_() for p in params]

            grads = [
                [torch.randn_like(p) for p in params]
                for _ in range(steps_warmup + steps_train)
            ]

            # Control (capturable=False)

            opt = optimizer_ctor(params_control, capturable=False, **kwargs)

            for i in range(steps_warmup + steps_train):
                for j, p in enumerate(params_control):
                    p.grad = grads[i][j]
                opt.step()

            # capturable=True

            opt = optimizer_ctor(params_graphed, capturable=True, **kwargs)

            for i in range(steps_warmup):
                for j, p in enumerate(params_graphed):
                    p.grad = grads[i][j]
                opt.step()

            if actually_do_graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    opt.step()

            for i in range(steps_train):
                if actually_do_graphs:
                    for j, p in enumerate(params_graphed):
                        p.grad.copy_(grads[i + steps_warmup][j])
                    g.replay()
                else:
                    # Passing capturable=True to the constructor and running without graphs should still be
                    # numerically correct, even if it's not ideal for performance.
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i + steps_warmup][j]
                    opt.step()

            for p_control, p_graphed in zip(params_control, params_graphed):
                self.assertEqual(p_control, p_graphed)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_optims_with_explicitly_capturable_param_groups(self):
        # [FUNCTION DOC] For several optimizers (Adam, AdamW, ASGD, Adamax, NAdam, RAdam,
        # Adadelta, RMSprop, Rprop) and a flag for whether the second param group is capturable,
        # compares a reference non-capturable optimizer to one with two param groups. If the
        # second group is not capturable, graph capture of the step must raise RuntimeError;
        # otherwise captured replays must match the reference parameters.
        # mimicking `_test_graphed_optimizer` maladroitly to pass two param_groups to optimizer.__init__
        n_warmup, n_replay = 3, 2
        for optimizer, second_param_group_capturable in product(
            (
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.ASGD,
                torch.optim.Adamax,
                torch.optim.NAdam,
                torch.optim.RAdam,
                torch.optim.Adadelta,
                torch.optim.RMSprop,
                torch.optim.Rprop,
            ),
            (True, False),
        ):
            ref_p1, param1 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            ref_p2, param2 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            grads1, grads2 = (
                [torch.randn_like(param1) for _ in range(n_warmup + n_replay)]
                for _ in range(2)
            )
            ref_grads1, ref_grads2 = (
                [t.clone() for t in tensors] for tensors in (grads1, grads2)
            )
            params = [
                {"params": [param1], "capturable": True},
                {"params": [param2], "capturable": second_param_group_capturable},
            ]
            opt = optimizer(params)
            opt_ = optimizer(
                [
                    {"params": [ref_p1], "capturable": False},
                    {"params": [ref_p2], "capturable": False},
                ]
            )

            for i in range(n_warmup + n_replay):
                ref_p1.grad = ref_grads1[i]
                ref_p2.grad = ref_grads2[i]
                opt_.step()

            for i in range(n_warmup):
                param1.grad = grads1[i]
                param2.grad = grads2[i]
                opt.step()

            g = torch.cuda.CUDAGraph()
            if not second_param_group_capturable:
                with self.assertRaisesRegex(RuntimeError, "Attempting CUDA graph"):
                    with torch.cuda.graph(g):
                        opt.step()
            else:
                with torch.cuda.graph(g):
                    opt.step()

                for i in range(n_replay):
                    param1.grad.copy_(grads1[n_warmup + i])
                    param2.grad.copy_(grads2[n_warmup + i])
                    g.replay()
                self.assertEqual(ref_p1, param1)
                self.assertEqual(ref_p2, param2)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_error_options(self):
        # [FUNCTION DOC] Checks graph capture error modes with a raw allocator call from a
        # second thread during capture. For the thread-local and relaxed modes the capture must
        # not raise (the helper returns False); the global mode case is left out because the
        # exception would corrupt the process.
        def fn():
            # [FUNCTION DOC] HELPER: Allocates a 2000-element zero tensor and returns the sum of
            # three copies of it.
            x = torch.zeros([2000], device="cuda")
            y = x + x + x
            return y

        mem = None

        def raw_malloc():
            # [FUNCTION DOC] HELPER: Thread body that allocates 1024 bytes from the caching
            # allocator on a fresh stream, then frees it. Uses a module-level global named mem
            # to hold the pointer.
            # KNOWN ISSUE: the global declaration makes this write a module global, while the
            # caller reads a closure variable of the same name defined in the test, so the
            # cleanup check in the caller never sees the pointer assigned here.
            global mem
            mem = None
            stream = torch.cuda.Stream()
            try:
                with torch.cuda.stream(stream):
                    mem = torch.cuda.caching_allocator_alloc(1024)
            except BaseException:
                if mem is None:
                    return
            try:
                torch.cuda.caching_allocator_delete(mem)
                mem = None
                return None
            except BaseException:
                pass

        def throws_on_cuda_event(capture_error_mode):
            # [FUNCTION DOC] HELPER: Warms up the helper function on a side stream, then
            # captures it with the given capture error mode while another thread performs a raw
            # allocation; returns True if the capture raised any exception (freeing the leaked
            # pointer if set), else False.
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            stream.synchronize()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            try:
                with torch.cuda.graph(
                    graph, stream=stream, capture_error_mode=capture_error_mode
                ):
                    out = fn()
                    thread = threading.Thread(target=raw_malloc)
                    thread.start()
                    thread.join()
            except Exception:
                if mem is not None:
                    torch.cuda.caching_allocator_delete(mem)
                return True

            return False

        self.assertFalse(throws_on_cuda_event("thread_local"))
        self.assertFalse(throws_on_cuda_event("relaxed"))

        # Exception would Corrupt Process and make other tests fail
        # self.assertTrue(throws_on_cuda_event("global"))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs, cuda-python must be installed",
    )
    def test_cuda_graph_raw_graph_keep_graph_false(self):
        # [FUNCTION DOC] Captures an addition with a graph created with keep_graph disabled.
        # Calling instantiate must raise RuntimeError (only allowed when keep_graph is
        # enabled) and asking for the raw graph handle must raise RuntimeError saying no graph
        # handle is available.
        graph = torch.cuda.CUDAGraph(keep_graph=False)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            z = x + y

        with self.assertRaisesRegex(
            RuntimeError,
            r"instantiate\(\) is intended to be called by the user only when keep_graph=true",
        ):
            raw_pointer = graph.instantiate()

        with self.assertRaisesRegex(
            RuntimeError,
            r"No (cuda|hip)Graph_t is available",
        ):
            raw_pointer = graph.raw_cuda_graph()

    @unittest.skipUnless(
        TEST_CUDA_GRAPH,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs",
    )
    @requires_cuda_python_bindings
    def test_cuda_graph_raw_graph(self):
        # [FUNCTION DOC] With keep_graph enabled, captures an addition, obtains the raw graph
        # handle, wraps it for the cuda python runtime bindings, queries its node count and
        # nodes, and node types with error checking, then replays the graph. Needs the cuda
        # python bindings.
        import cuda.bindings.runtime as cudart

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            z = x + y

        raw_pointer = graph.raw_cuda_graph()

        cudart_cuda_graph = cudart.cudaGraph_t(init_value=raw_pointer)
        _, num_nodes = cuda_python_error_check(
            cudart.cudaGraphGetNodes(cudart_cuda_graph)
        )
        nodes, _ = cuda_python_error_check(
            cudart.cudaGraphGetNodes(cudart_cuda_graph, num_nodes)
        )
        for node in nodes:
            cuda_python_error_check(cudart.cudaGraphNodeGetType(node))

        graph.replay()

    @unittest.skipUnless(
        TEST_CUDA_GRAPH,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs",
    )
    @requires_cuda_python_bindings
    @parametrize("keep_graph", [True, False])
    def test_cuda_graph_raw_graph_exec(self, keep_graph):
        # [FUNCTION DOC] Parametrized by keep_graph. Captures an addition; when keep_graph is
        # true, asking for the raw executable graph before instantiate must raise RuntimeError,
        # then instantiates. Wraps the raw exec handle for the cuda python bindings, queries its
        # flags with error checking, and replays the graph. Needs the cuda python bindings.
        import cuda.bindings.runtime as cudart

        graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        x = torch.zeros([2000], device="cuda")
        y = torch.ones([2000], device="cuda")
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            z = x + y

        if keep_graph:
            with self.assertRaisesRegex(
                RuntimeError,
                r"You cannot access the raw (cuda|hip)GraphExec_t instance until instantiate\(\) has been called",
            ):
                graph.raw_cuda_graph_exec()

            graph.instantiate()
        raw_pointer = graph.raw_cuda_graph_exec()

        cudart_cuda_graph_exec = cudart.cudaGraphExec_t(init_value=raw_pointer)
        cuda_python_error_check(cudart.cudaGraphExecGetFlags(cudart_cuda_graph_exec))

        graph.replay()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_raw_graph_reset_and_recapture(self):
        # [FUNCTION DOC] Tests instantiate, replay, reset and recapture on a graph with
        # keep_graph: adds 1.0 in place, instantiates twice with replays checking values grow to
        # 1, 2 and 3; after reset, captures an addition of 2.0 on a new tensor, and checks
        # values after instantiate, re-instantiate and replays are 2, 4 and 6.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        x = torch.zeros([2000], device="cuda")
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            x += 1.0

        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 1.0))
        # Exercise the code path where you reinstantiate the cuda graph twice.
        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 2.0))
        graph.replay()
        self.assertTrue(torch.all(x == 3.0))

        # Check that graph capture can succeed after resetting.
        graph.reset()

        # Don't do x[:] = 0.0 because we want to capture a new address
        # in the next cuda graph, to make sure we are running a new
        # cuda graph.
        x = torch.zeros([2000], device="cuda")
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            x += 2.0

        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 2.0))
        # Exercise the code path where you reinstantiate the cuda graph twice.
        graph.instantiate()
        graph.replay()
        self.assertTrue(torch.all(x == 4.0))
        graph.replay()
        self.assertTrue(torch.all(x == 6.0))

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/122413",
    )
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_allocator_propagates_stream(self):
        # [FUNCTION DOC] Checks that the allocator propagates a capture to private pool
        # consistently across streams: captures an addition in one stream, a dependent op in a
        # second stream, then ends the capture. Two new segments must exist and share the same
        # pool id. Skipped on Linux, ROCm and slow runs (known issue) and without graph support.
        segments = torch.cuda.memory_snapshot()
        existing_pools = {s["segment_pool_id"] for s in segments}
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            g.capture_begin()
            z = x + y
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            z + y
        s0.wait_stream(s1)
        with torch.cuda.stream(s0):
            g.capture_end()
        segments = torch.cuda.memory_snapshot()
        x = [
            s["segment_pool_id"]
            for s in segments
            if s["segment_pool_id"] not in existing_pools
        ]
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0], x[1])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM
        or not torch.version.cuda
        or tuple(int(x) for x in torch.version.cuda.split(".")) < (12, 4),
        "CUDA >= 12.4 required for conditional graph nodes",
    )
    def test_cuda_graph_many_if_nodes_across_graphs(self):
        # [FUNCTION DOC] Creates 40 separate graphs, each capturing a conditional if node that
        # adds 1 to a scalar based on a boolean predicate. Passes if no error occurs. Needs
        # CUDA 12.4+.
        pred = torch.zeros((), device="cuda", dtype=torch.bool)
        x = torch.zeros((), device="cuda")
        for _ in range(40):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                g.begin_capture_to_if_node(pred)
                x.add_(1.0)
                g.end_capture_to_conditional_node()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM
        or not torch.version.cuda
        or tuple(int(x) for x in torch.version.cuda.split(".")) < (12, 4),
        "CUDA >= 12.4 required for conditional graph nodes",
    )
    def test_cuda_graph_many_if_nodes_in_one_graph(self):
        # [FUNCTION DOC] Captures 40 conditional if nodes in a single graph, each adding 1 to a
        # scalar under a boolean predicate. Passes if no error occurs. Needs CUDA 12.4+.
        pred = torch.zeros((), device="cuda", dtype=torch.bool)
        x = torch.zeros((), device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(40):
                g.begin_capture_to_if_node(pred)
                x.add_(1.0)
                g.end_capture_to_conditional_node()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_tensor_item_not_allowed(self):
        # [FUNCTION DOC] Runs a child Python script that captures a graph where a function
        # indexes a tensor with a value taken from another CUDA tensor (an item-style sync) and
        # expects the process to fail; the parent asserts a CalledProcessError whose output says
        # a synchronize is not allowed in a cudagraph.
        test_script = """\
import torch
import sys
# Tensor.item() calls a synchronize which is not allowed in a cudagraph
# Valid for CUDA and ROCm
def my_func(a: torch.Tensor, b: torch.Tensor, perm: torch.Tensor):
    idx = perm[0]
    a[0] *= b[idx]  # should raise an error during capture
    return a

a = torch.rand(500, 500, device="cuda")
b = torch.rand(500, 500, device="cuda")
perm = torch.randint(0, 500, (500,), device="cuda")

g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g):
    output = my_func(a, b, perm)
"""
        with self.assertRaisesRegex(
            subprocess.CalledProcessError,
            "calls a synchronize which is not allowed in a cudagraph",
        ):
            r = (
                subprocess.check_output([sys.executable, "-c", test_script])
                .decode("ascii")
                .strip()
            )

    def test_batch_norm_gather_stats(self):
        # [FUNCTION DOC] Calls batch_norm_gather_stats on CUDA with per-device mean and invstd
        # of ones (two rows), no running stats, and count 2; asserts both returned mean and
        # invstd equal ones of size 3.
        input = torch.randn(1, 3, 3, 3, device="cuda")
        mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean=torch.ones(2, 3, device="cuda"),
            invstd=torch.ones(2, 3, device="cuda"),
            running_mean=None,
            running_var=None,
            momentum=0.1,
            eps=1e-5,
            count=2,
        )
        self.assertEqual(mean, torch.ones(3, device="cuda"))
        self.assertEqual(invstd, torch.ones(3, device="cuda"))

    def test_matmul_memory_use(self):
        # [FUNCTION DOC] Checks matmul memory use does not depend on expansion: measures peak
        # allocated memory for a batched matmul with a broadcast operand, for the same with the
        # operand explicitly expanded, and for bmm; the expanded matmul and bmm peaks must equal
        # the first.
        def get_max_used():
            # [FUNCTION DOC] HELPER: Synchronizes, reads the max allocated memory, resets peak
            # stats and returns the value.
            torch.cuda.synchronize()
            val = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return val

        a = torch.rand(1, 32, 32, device="cuda")
        b = torch.rand(24, 32, 1, device="cuda")

        get_max_used()

        torch.matmul(a, b)

        matmul_mem = get_max_used()

        a = a.expand(24, 32, 32)
        torch.matmul(a, b)

        matmul_expand_mem = get_max_used()

        torch.bmm(a, b)

        bmm_mem = get_max_used()

        self.assertEqual(matmul_expand_mem, matmul_mem)
        self.assertEqual(bmm_mem, matmul_mem)

    @unittest.skipIf(not TEST_WITH_ROCM, "ROCm-only test")
    def test_rocm_backward_pass_guard(self):
        # [FUNCTION DOC] ROCm-only test. Trains one step of a model whose custom autograd
        # function asserts that the ROCm backward-pass flag is false in forward and true in
        # backward, so the guard flag is exercised during an ordinary forward, backward and
        # optimizer step.
        # The test exercises a ROCm-specific feature.

        class MyFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function multiplying a tensor by a constant; forward
            # asserts the ROCm backward-pass flag is off and backward asserts it is on,
            # returning the gradient scaled by the constant (and None for the constant).
            @staticmethod
            def forward(ctx, tensor, constant):
                # [FUNCTION DOC] HELPER: Asserts the backward-pass flag is false, stores the
                # constant on the context and returns the tensor times the constant.
                self.assertFalse(torch._C._rocm_is_backward_pass())
                ctx.constant = constant
                return tensor * constant

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Asserts the backward-pass flag is true and returns the
                # gradient times the stored constant, with None for the constant.
                self.assertTrue(torch._C._rocm_is_backward_pass())
                return grad_output * ctx.constant, None

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding one scalar parameter and applying the custom
            # function with it in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a scalar random parameter named a.
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(()))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the custom multiply function to x and
                # the parameter.
                return MyFunction.apply(x, self.a)

        model = MyModule()
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

        x = torch.randn(5, 5)
        result = model(x)
        loss = criterion(result, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_matmul_device_mismatch(self):
        # [FUNCTION DOC] Checks matmul and addmm with mixed CPU and CUDA operands: matrix
        # product of CPU and CUDA tensors in either order raises RuntimeError (all tensors must
        # be on the same device); for every combination of three tensors addmm succeeds only
        # when all share a device, else raises the same error.
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            cpu @ cuda
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            cuda @ cpu

        for s, m1, m2 in product((cpu, cuda), repeat=3):
            if s.device == m1.device == m2.device:
                torch.addmm(s, m1, m2)
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Expected all tensors to be on the same device"
                ):
                    torch.addmm(s, m1, m2)

    @unittest.skipIf(TEST_MULTIGPU, "Testing on one GPU is sufficient")
    def test_lazy_init(self):
        # [FUNCTION DOC] Validates that importing torch does not initialize CUDA. Runs a
        # child process hiding all devices by environment variable and expects a device count
        # of 0; on non-ROCm also runs a child that imports torch and calls the driver device
        # count function via ctypes, expecting error code 3 (not initialized). Skipped with
        # multiple GPUs.
        """Validate that no CUDA calls are made during `import torch` call"""

        def check_output(script: str) -> str:
            # [FUNCTION DOC] HELPER: Runs the given script text in a Python subprocess and
            # returns its stripped, ascii-decoded standard output.
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        VISIBLE_DEVICES = (
            "HIP_VISIBLE_DEVICES" if TEST_WITH_ROCM else "CUDA_VISIBLE_DEVICES"
        )
        test_script = f"import os; import torch;os.environ['{VISIBLE_DEVICES}']='32';torch.get_device_module().device_count();print(torch.cuda.device_count())"
        rc = check_output(test_script)
        self.assertEqual(rc, "0")
        if not TEST_WITH_ROCM:
            # Check that `cuInit` was not called during the import
            # By using ctypes and calling cuDeviceCountGet() and expect CUDA_ERROR_NOT_INITIALIZED == 3
            # See https://github.com/pytorch/pytorch/issues/116276 for more details
            libcuda_name = "libcuda.so.1" if not IS_WINDOWS else "nvcuda.dll"
            cuda_driver_api_call = (
                f"ctypes.CDLL('{libcuda_name}').cuDeviceGetCount(ctypes.byref(x))"
            )
            rc = check_output(
                f"import torch; import ctypes;x=ctypes.c_int(-1);print({cuda_driver_api_call})"
            )
            self.assertEqual(rc, "3")

    @unittest.skipIf(not TEST_WITH_ROCM, "not relevant for CUDA testing")
    @skipIfRocmVersionAtLeast([7, 14])
    def test_hip_device_count(self):
        # [FUNCTION DOC] ROCm-only test (skipped on ROCm 7.14+): for several environment
        # settings of CUDA, HIP and ROCR visible device variables (None removes the variable),
        # runs a child process printing the device count and asserts it is 1 each time.
        """Validate device_count works with both CUDA/HIP visible devices"""
        test_script = """\
import torch
import os
print(f"{torch.cuda.device_count()}")
"""
        custom_envs = [
            {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": None},
            {"CUDA_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": "0"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "0"},
            {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": None},
        ]

        if torch.cuda.device_count() >= 2:
            custom_envs.extend(
                [
                    {"ROCR_VISIBLE_DEVICES": "1,2,3", "HIP_VISIBLE_DEVICES": "0"},
                ]
            )

        for env_config in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual("1", r)

    @unittest.skipIf(not TEST_MULTIGPU, "requires multiple devices")
    def test_device_count_not_cached_pre_init(self):
        # [FUNCTION DOC] Multi-GPU test: a child process reads the device count, changes the
        # visible devices environment variable to device 0 before initializing, reads the count
        # again and allocates on CUDA; the parent asserts the printed pair equals the parent's
        # device count followed by 1, i.e. the count was not cached before initialization.
        visible_devices = (
            "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
        )
        test_script = f"""\
import torch
import os
r1 = torch.cuda.device_count()
os.environ['{visible_devices}'] = '0'
r2 = torch.cuda.device_count()
torch.empty(10, device='cuda')
print(f"{{r1}}, {{r2}}")
"""

        r = (
            subprocess.check_output([sys.executable, "-c", test_script])
            .decode("ascii")
            .strip()
        )

        x = torch.cuda.device_count()
        self.assertEqual(f"{x}, 1", r)

    @unittest.skipUnless("CI" in os.environ, "Only run on CI")
    @unittest.skipIf(
        _get_torch_cuda_version() >= (13, 1),
        "This test does not fail on CUDA 13.1 or newer",
    )
    @unittest.skipIf(
        TEST_WITH_ROCM and torch.cuda.gds.is_available(),
        "hipFile uses a POSIX compat fallback on ROCm, so GdsFile does not fail here",
    )
    def test_gds_fails_in_ci(self):
        # [FUNCTION DOC] Runs only in CI on CUDA older than 13.1 (and not on ROCm with gds
        # available). Opens a temporary file with the GDS file wrapper and expects RuntimeError:
        # a file handle registration failure if GDS is available, else a
        # not-supported-on-platform message.
        if torch.cuda.gds.is_available():
            error_msg = "cuFileHandleRegister failed"
        else:
            error_msg = "is not supported on this platform"
        with TemporaryFileName() as f:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)

    @unittest.skipIf(
        IS_WINDOWS, "test relies on fork; Windows multiprocessing uses spawn"
    )
    def test_is_pinned_no_context(self):
        # [FUNCTION DOC] Runs a child script that calls is_pinned on a CPU tensor, then forks a
        # child process (explicit fork context) which uses the accelerator and sends back a
        # value over a pipe; asserts the printed result is 1.0, i.e. is_pinned does not poison
        # later forks. Skipped on Windows.
        test_script = """\
import torch
import multiprocessing


def fork_and_check_is_pinned():
    # Explicitly use the fork context: this test relies on fork semantics
    # (a forked child inherits the parent's CUDA context state), and Python
    # 3.14 changed the default start method on non-macOS POSIX to forkserver,
    # which would try to pickle the local `worker` function below and fail.
    mp_ctx = multiprocessing.get_context("fork")
    # Create a pipe to communicate between parent and child processes
    parent_conn, child_conn = mp_ctx.Pipe()

    def worker(conn):
        try:
            x = torch.randn(10)
            x.is_pinned()
            dev = torch.accelerator.current_accelerator()
            x = torch.ones(10, device=dev)[0].item()
            conn.send(x)
        except Exception as e:
            conn.send(str(e))
        finally:
            conn.close()
    # Fork a new process
    p = mp_ctx.Process(target=worker, args=(child_conn,))
    p.start()
    # Receive the result from the child process
    result = parent_conn.recv()
    parent_conn.close()
    # Wait for the child process to finish
    p.join()
    if isinstance(result, str) and result.startswith("Error"):
        raise RuntimeError(result)
    return result

x = torch.randn(10)
# check that is_pinned won't poison future fork
x.is_pinned()
ret = fork_and_check_is_pinned()
print(ret)

"""
        r = (
            subprocess.check_output([sys.executable, "-c", test_script])
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "1.0")


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@unittest.skipIf(
    TEST_WITH_ROCM and EXPANDABLE_SEGMENTS,
    "expandable_segments mode is not supported on ROCm",
)
class TestResizeStorageWithAddr(TestCase):
    # [CLASS DOC] Tests for resizing a CUDA storage back to a given address inside a memory pool
    # (exact-address allocation). Flags enable the memory leak check and non-default stream
    # mode; the class is skipped without CUDA and for ROCm expandable segments. Includes a
    # helper that finds a block in a pool snapshot.
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _find_pool_block(self, pool, addr):
        # [FUNCTION DOC] HELPER: Searches the pool snapshot segments for a block at the given
        # address and returns the list of blocks of that segment with the index of the match;
        # fails the test if no block has that address.
        for segment in pool.snapshot(include_traces=False):
            for idx, block in enumerate(segment["blocks"]):
                if block["address"] == addr:
                    return segment["blocks"], idx
        self.fail(f"Could not find block at address {addr}")

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_exact_allocation(self):
        # [FUNCTION DOC] In a memory pool, allocates a small and a 4096-byte tensor, shrinks
        # both storages to size zero, then within the pool resizes the second back to its
        # original size at its original address and allocates a new small tensor. Asserts the
        # size and address are restored and the new tensor got a different address. Skipped
        # for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            other = torch.empty(294, dtype=torch.uint8, device="cuda")
            t = torch.empty(4096, dtype=torch.uint8, device="cuda")

        original_ptr = t.untyped_storage().data_ptr()
        original_size = t.untyped_storage().nbytes()
        t.untyped_storage().resize_(0)
        other.untyped_storage().resize_(0)

        # `resize_(0)` does not record mempool information. We still need `use_mem_pool`
        # when resizing it back.
        with torch.cuda.use_mem_pool(pool):
            t.untyped_storage()._resize_with_addr_(original_size, original_ptr)
            other = torch.empty(294, dtype=torch.uint8, device="cuda")

        self.assertEqual(t.untyped_storage().nbytes(), original_size)
        self.assertEqual(t.untyped_storage().data_ptr(), original_ptr)
        self.assertNotEqual(other.untyped_storage().data_ptr(), original_ptr)

    def test_resize_storage_negative_size_raises(self):
        # [FUNCTION DOC] Asserts resizing a storage to a negative size raises RuntimeError
        # saying the value cannot be represented as a size_t.
        # A negative requested storage size must be rejected, not silently
        # wrapped to a huge size_t (c10::overflows strict_unsigned path).
        t = torch.empty(8, dtype=torch.uint8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "cannot be represented as a size_t"):
            t.untyped_storage().resize_(-1)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_occupied_address_errors(self):
        # [FUNCTION DOC] Shrinks a pool tensor to zero, then tries to resize it to the address
        # of another live tensor of 4096 bytes; asserts RuntimeError mentioning the exact
        # address could not be obtained. Skipped for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            occupied = torch.empty(4096, dtype=torch.uint8, device="cuda")
            t = torch.empty(2048, dtype=torch.uint8, device="cuda")

        occupied_ptr = occupied.untyped_storage().data_ptr()
        occupied_size = occupied.untyped_storage().nbytes()
        t.untyped_storage().resize_(0)

        with self.assertRaisesRegex(RuntimeError, "exact address"):
            with torch.cuda.use_mem_pool(pool):
                t.untyped_storage()._resize_with_addr_(occupied_size, occupied_ptr)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_middle_of_block(self):
        # [FUNCTION DOC] Allocates three adjacent tensors in a pool, frees all three storages,
        # then resizes the middle one back at its address. Asserts the address matches, and via
        # the pool snapshot that its block is active, the neighboring blocks before and after
        # are inactive and contiguous with it. Skipped for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            prefix = torch.empty(3729, dtype=torch.uint8, device="cuda")
            t = torch.empty(4017, dtype=torch.uint8, device="cuda")
            suffix = torch.empty(2738, dtype=torch.uint8, device="cuda")

        original_ptr = t.untyped_storage().data_ptr()
        original_size = t.untyped_storage().nbytes()
        prefix.untyped_storage().resize_(0)
        t.untyped_storage().resize_(0)
        suffix.untyped_storage().resize_(0)

        with torch.cuda.use_mem_pool(pool):
            t.untyped_storage()._resize_with_addr_(original_size, original_ptr)
            self.assertEqual(t.untyped_storage().data_ptr(), original_ptr)

        # when `original_ptr` is in the middle of an empty block, `_resize_with_addr_`
        # should still allocate a tensor with exactly `original_size` at `original_ptr`.
        # It should also keep prefix and suffix blocks inactive. These assertions
        # check that the block state matches expectations.
        blocks, idx = self._find_pool_block(pool, original_ptr)
        self.assertGreater(idx, 0)
        self.assertLess(idx + 1, len(blocks))
        self.assertEqual(blocks[idx]["state"], "active_allocated")
        self.assertEqual(blocks[idx - 1]["state"], "inactive")
        self.assertEqual(
            blocks[idx - 1]["address"] + blocks[idx - 1]["size"],
            original_ptr,
        )
        self.assertEqual(blocks[idx + 1]["state"], "inactive")
        self.assertEqual(
            blocks[idx]["address"] + blocks[idx]["size"],
            blocks[idx + 1]["address"],
        )

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_misaligned_address_errors(self):
        # [FUNCTION DOC] Shrinks a pool tensor to zero and asks to resize it to an address one
        # byte past the original; asserts RuntimeError saying the address must be aligned to the
        # minimum block size. Skipped for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            t = torch.empty(4096, dtype=torch.uint8, device="cuda")

        original_ptr = t.untyped_storage().data_ptr()
        t.untyped_storage().resize_(0)

        with self.assertRaisesRegex(RuntimeError, "aligned to kMinBlockSize"):
            with torch.cuda.use_mem_pool(pool):
                t.untyped_storage()._resize_with_addr_(2048, original_ptr + 1)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_recovers_viewed_slices(self):
        # [FUNCTION DOC] Mimics state offloading: a base tensor and two views are in a pool; the
        # storage is shrunk to zero (all views see size zero), then resized back at the original
        # address and size; asserts base and both views report the original data pointer and
        # byte size again. Skipped for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            base = torch.empty(4096, dtype=torch.uint8, device="cuda")
            view1 = base[512:1024]
            view2 = base[1024:1536]

        storage = base.untyped_storage()
        original_ptr = storage.data_ptr()
        original_size = storage.nbytes()
        storage.resize_(0)

        self.assertEqual(base.untyped_storage().nbytes(), 0)
        self.assertEqual(view1.untyped_storage().nbytes(), 0)
        self.assertEqual(view2.untyped_storage().nbytes(), 0)

        with torch.cuda.use_mem_pool(pool):
            storage._resize_with_addr_(original_size, original_ptr)

        # This test mimics a state offloading pattern where we offload a large
        # tensor and keep many small tensors as slices of the large tensor.
        self.assertEqual(base.untyped_storage().data_ptr(), original_ptr)
        self.assertEqual(view1.untyped_storage().data_ptr(), original_ptr)
        self.assertEqual(view2.untyped_storage().data_ptr(), original_ptr)
        self.assertEqual(base.untyped_storage().nbytes(), original_size)
        self.assertEqual(view1.untyped_storage().nbytes(), original_size)
        self.assertEqual(view2.untyped_storage().nbytes(), original_size)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_addr_requires_zero_sized_source(self):
        # [FUNCTION DOC] Asserts that resizing a non-empty storage to an address raises
        # RuntimeError requiring a zero-sized CUDA storage. Skipped for cudaMallocAsync.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            t = torch.empty(4096, dtype=torch.uint8, device="cuda")

        original_ptr = t.untyped_storage().data_ptr()
        original_size = t.untyped_storage().nbytes()

        with self.assertRaisesRegex(RuntimeError, "zero-sized CUDA storage"):
            with torch.cuda.use_mem_pool(pool):
                t.untyped_storage()._resize_with_addr_(original_size, original_ptr)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_size_zero_ignores_addr(self):
        # [FUNCTION DOC] Checks the address is ignored when the requested size is zero: resizing
        # a 4096-byte storage to size 0 with address 0 succeeds and leaves zero bytes. Skipped
        # for cudaMallocAsync.
        t = torch.empty(4096, dtype=torch.uint8, device="cuda")
        t.untyped_storage()._resize_with_addr_(0, 0)
        self.assertEqual(t.untyped_storage().nbytes(), 0)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_multiple_blocks_in_the_middle(self):
        # [FUNCTION DOC] Allocates ten interleaved dummy and target tensors in a pool, shrinks
        # all to zero, then resizes each target back at its original address and size within
        # the pool, and asserts each target data pointer and byte size are restored. Skipped
        # for cudaMallocAsync.
        # Test allocating multiple target tensors in the middle of an empty memory segment
        pool = torch.cuda.MemPool()

        targets = []
        dummys = []
        with torch.cuda.use_mem_pool(pool):
            for _ in range(10):
                dummy = torch.empty(3729, dtype=torch.uint8, device="cuda")
                target = torch.empty(4017, dtype=torch.uint8, device="cuda")
                targets.append(target)
                dummys.append(dummy)

        original_ptrs = [t.untyped_storage().data_ptr() for t in targets]
        original_sizes = [t.untyped_storage().nbytes() for t in targets]

        for t, d in zip(targets, dummys):
            t.untyped_storage().resize_(0)
            d.untyped_storage().resize_(0)

        with torch.cuda.use_mem_pool(pool):
            for i in range(10):
                ptr = original_ptrs[i]
                size = original_sizes[i]
                targets[i].untyped_storage()._resize_with_addr_(size, ptr)

        for i in range(10):
            ptr = original_ptrs[i]
            size = original_sizes[i]
            t = targets[i]
            self.assertEqual(t.untyped_storage().data_ptr(), ptr)
            self.assertEqual(t.untyped_storage().nbytes(), size)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC,
        "CUDAMallocAsync does not support exact-address allocation",
    )
    def test_resize_storage_with_coalesced_small_prefixes(self):
        # [FUNCTION DOC] Allocates 800 small prefix tensors, two target tensors with another
        # tensor between and a suffix in a pool, frees them all, then resizes both targets back
        # at their original addresses (which requires allocating a large combined prefix from
        # the small pool). Asserts both targets get their original pointers and sizes. Skipped
        # for cudaMallocAsync.
        # Suppose we have allocated many small prefix tensors and a target tensor, where
        # all tensors falls into the small block pool. When resize the target tensor later,
        # we allocate 1 large prefix tensors for all small prefix tensors, then allocate
        # the target tensor, and finally free the large prefix tensor. This large prefix
        # tensor should come from the small block pool, even if its size is beyond the threshold.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            small_prefixes = []
            for _ in range(800):
                prefix = torch.empty(3729, dtype=torch.uint8, device="cuda")
                small_prefixes.append(prefix)
            t1 = torch.empty(4017, dtype=torch.uint8, device="cuda")
            mid = torch.empty(2738, dtype=torch.uint8, device="cuda")
            t2 = torch.empty(3897, dtype=torch.uint8, device="cuda")
            suffix = torch.empty(9124, dtype=torch.uint8, device="cuda")

        original_ptr1 = t1.untyped_storage().data_ptr()
        original_size1 = t1.untyped_storage().nbytes()
        original_ptr2 = t2.untyped_storage().data_ptr()
        original_size2 = t2.untyped_storage().nbytes()

        for prefix in small_prefixes:
            prefix.untyped_storage().resize_(0)
        mid.untyped_storage().resize_(0)
        suffix.untyped_storage().resize_(0)
        t1.untyped_storage().resize_(0)
        t2.untyped_storage().resize_(0)

        with torch.cuda.use_mem_pool(pool):
            t1.untyped_storage()._resize_with_addr_(original_size1, original_ptr1)
            t2.untyped_storage()._resize_with_addr_(original_size2, original_ptr2)

        self.assertEqual(t1.untyped_storage().data_ptr(), original_ptr1)
        self.assertEqual(t1.untyped_storage().nbytes(), original_size1)
        self.assertEqual(t2.untyped_storage().data_ptr(), original_ptr2)
        self.assertEqual(t2.untyped_storage().nbytes(), original_size2)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaAllocator(TestCase):
    # [CLASS DOC] Tests for the CUDA caching allocator's memory history, snapshots, traces and
    # plots. Skipped without CUDA; wrapped for strict dynamo mode. tearDown checks that no test
    # changed the expandable_segments allocator setting.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base class tearDown, then runs the allocator-settings
        # regression check on this test case.
        super().tearDown()
        _check_allocator_settings_on_tear_down(self)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot(self):
        # [FUNCTION DOC] Records memory history with state and python stacks, makes a mixed set
        # of allocations (including 1000 small tensors freed in random order), takes a snapshot,
        # and checks every segment has frames, the block of requested size 311 by 411 floats has
        # a frame from this test file at the top and its address matches the tensor storage.
        # Outside Windows, saves segment usage to a file containing the test file name; after
        # emptying the cache the last device trace action is a segment free or unmap. History
        # recording is turned off at the end. Not for cudaMallocAsync.
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="python")
            # make x the second block in a segment
            torch.rand(2 * 311, 411, device="cuda")
            unused = torch.rand(310, 410, device="cuda")
            x = torch.rand(311, 411, device="cuda")

            # create a bunch of tensors that all will tile into the
            # same segment to  exercise the history merging code
            # 512B is the minimum block size,
            # so we allocate all the tensors to this size to make sure
            # they tile evenly
            tensors = [torch.rand(128, device="cuda") for _ in range(1000)]
            while tensors:
                del tensors[randint(0, len(tensors) - 1)]

            # exercise the history trimming code
            torch.rand(128 * 5, device="cuda")

            ss = torch.cuda.memory._snapshot()
            found_it = False
            for seg in ss["segments"]:
                self.assertTrue("frames" in seg)
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("test_cuda" in b["frames"][0]["filename"])
                        found_it = True
                        self.assertEqual(x.untyped_storage().data_ptr(), b["address"])
            self.assertTrue(found_it)

            if not IS_WINDOWS:
                with tempfile.NamedTemporaryFile() as f:
                    torch.cuda.memory._save_segment_usage(f.name)
                    with open(f.name) as f2:
                        self.assertTrue("test_cuda.py" in f2.read())
            del unused
            del x
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            ss = torch.cuda.memory._snapshot()
            self.assertTrue(
                ss["device_traces"][0][-1]["action"]
                in ("segment_free", "segment_unmap")
            )

        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_allocation_traceback(self):
        # [FUNCTION DOC] Records full history with python stacks, allocates through a named
        # helper function, and asserts the allocation traceback for its pointer is a non-empty
        # list of frames including that function name; after freeing the tensor and emptying the
        # cache the traceback is None. Not for cudaMallocAsync.
        def this_is_a_funky_allocator():
            # [FUNCTION DOC] HELPER: Returns a random CUDA tensor of 311 by 411 elements; used
            # so the function name appears in the allocation traceback.
            return torch.rand(311, 411, device="cuda")

        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("all", stacks="python")

            x = this_is_a_funky_allocator()
            frames = torch.cuda.memory._allocation_traceback(x.data_ptr())
            self.assertIsInstance(frames, list)
            self.assertGreater(len(frames), 0)
            func_names = [f["name"] for f in frames]
            self.assertIn("this_is_a_funky_allocator", func_names)

            # freed pointer should return None
            ptr = x.data_ptr()
            del x
            torch.cuda.empty_cache()
            self.assertIsNone(torch.cuda.memory._allocation_traceback(ptr))
        finally:
            torch.cuda.memory._record_memory_history(None)

    def test_allocation_traceback_no_recording(self):
        # [FUNCTION DOC] With recording disabled, asserts that the allocation traceback of a
        # live tensor pointer is None.
        torch.cuda.memory._record_memory_history(None)
        x = torch.rand(64, device="cuda")
        self.assertIsNone(torch.cuda.memory._allocation_traceback(x.data_ptr()))

    @unittest.skipUnless(
        (IS_X86 or IS_ARM64) and IS_LINUX, "linux x86/aarch64 only cpp unwinding"
    )
    def test_direct_traceback(self):
        # [FUNCTION DOC] Gathers a traceback including C++ frames directly with the profiler
        # binding, symbolizes it, and asserts the text mentions this test file and an unwind
        # frame. Linux x86 or aarch64 only.
        from torch._C._profiler import gather_traceback, symbolize_tracebacks  # @manual

        c = gather_traceback(True, True, True)
        (r,) = symbolize_tracebacks([c])
        r = str(r)
        self.assertTrue("test_cuda.py" in r)
        self.assertTrue("unwind" in r)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_snapshot_with_cpp(self):
        # [FUNCTION DOC] Records memory history with all stacks (C++ included), allocates a
        # random tensor and checks the block of that size in the snapshot has frames
        # containing the C++ random function name. Skipped for cudaMallocAsync and platforms
        # without C++ context.
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="all")
            x = torch.rand(311, 411, device="cuda")

            ss = torch.cuda.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("::rand" in str(b["frames"]))
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot_forward_trace(self):
        # [FUNCTION DOC] Records full history, runs a forward and backward pass under anomaly
        # detection without NaN checking, then verifies the snapshot device traces include
        # entries with a non-empty list of string forward frames. Not for cudaMallocAsync.
        """Test that forward traces from anomaly mode are captured in memory snapshots."""
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("all", stacks="python")

            # Create a simple model with a backward pass
            # Use detect_anomaly(check_nan=False) to capture forward traces
            x = torch.randn(10, 10, device="cuda", requires_grad=True)

            with torch.autograd.detect_anomaly(check_nan=False):
                y = x * 2  # Forward pass - creates autograd node
                z = y.sum()
                z.backward()  # Backward pass - allocations here should have forward traces

            ss = torch.cuda.memory._snapshot()

            # Check that we have device traces with forward_frames
            found_forward_frames = False
            if "device_traces" in ss:
                for device_trace in ss["device_traces"]:
                    for entry in device_trace:
                        if isinstance(entry, dict) and "forward_frames" in entry:
                            # forward_frames should be a list of strings
                            self.assertIsInstance(entry["forward_frames"], list)
                            if len(entry["forward_frames"]) > 0:
                                found_forward_frames = True
                                # The forward frames should be strings
                                self.assertIsInstance(entry["forward_frames"][0], str)

            # forward_frames must be present when using detect_anomaly
            self.assertTrue(
                found_forward_frames,
                "forward_frames should be present in memory snapshot when using detect_anomaly",
            )

        finally:
            torch.cuda.memory._record_memory_history(None)

    @skipIfRocm(msg="ROCTracer does not capture Python stack frames in profiler output")
    def test_memory_profiler_viz(self):
        # [FUNCTION DOC] Profiles a small CUDA computation with stacks and memory profiling,
        # builds the profile plot and the snapshot JSON, and asserts the JSON mentions this test
        # file, this test name and a category field. Skipped on ROCm.
        with torch.profiler.profile(
            with_stack=True, profile_memory=True, record_shapes=True
        ) as prof:
            x = torch.rand(128, 128, device="cuda")
            x * x + x * x
        plot = profile_plot(prof)
        plot = json.dumps(_profile_to_snapshot(prof))
        self.assertTrue("test_cuda.py" in plot)
        self.assertTrue("test_memory_profiler_viz" in plot)
        self.assertTrue("category" in plot)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_cycles(self):
        # [FUNCTION DOC] Checks the tensor reference cycle observer: registers an observer and
        # creates a closure that references a CUDA tensor and itself; after garbage collection
        # and a following call (the callback runs outside the collect), the observer must have
        # fired and the HTML contained the tensor type, this test name and a cell contents
        # mention. The observer is disarmed at the end. Not for cudaMallocAsync or platforms
        # without C++ context.
        fired = False

        def observer(html):
            # [FUNCTION DOC] HELPER: Callback receiving the report HTML; sets the enclosing
            # fired flag and asserts the report mentions a tensor, this test module and closure
            # cell contents.
            nonlocal fired
            fired = True
            self.assertTrue("torch.Tensor" in html)
            self.assertTrue("test_cuda" in html)
            self.assertTrue("cell_contents" in html)

        disarm = observe_tensor_cycles(observer)

        def noop():
            # [FUNCTION DOC] HELPER: Does nothing; its call gives the cycle observer a chance
            # to run.
            pass

        try:

            def create():
                # [FUNCTION DOC] HELPER: Creates a CUDA tensor and returns a recursive closure
                # over it, producing a reference cycle that holds the tensor.
                x = torch.empty(3, 4, device="cuda")

                def foo(p):
                    # [FUNCTION DOC] HELPER: Recursive function that calls itself once with
                    # the opposite flag when given a truthy argument, else returns the
                    # captured tensor.
                    if p:
                        return foo(not p)
                    else:
                        return x

                return foo

            create()
            gc.collect()
            # the callback has to run outside of the collect
            # call so it doesn't actual fire until the next
            # method call after a gc.collect
            noop()
            self.assertTrue(fired)
        finally:
            disarm()

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots(self):
        # [FUNCTION DOC] For three combinations of context and stacks settings, records history,
        # runs a small computation, builds trace plots (also with freed filtered) and a segment
        # plot from the snapshot, and checks the JSON text: the test name appears exactly when
        # context was recorded, the C++ random function name appears exactly when C++ stacks
        # were requested, and the byte size of the tensor appears.
        for context, stacks in (
            ("all", "all" if IS_LINUX else "python"),
            ("all", "python"),
            (None, "python"),
        ):
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(
                    "all", context=context, stacks=stacks
                )

                def run():
                    # [FUNCTION DOC] HELPER: Allocates a 128 by 128 CUDA tensor and computes a
                    # sum of squares expression to generate allocations.
                    x = torch.rand(128, 128, device="cuda")
                    x * x + x * x

                run()
                cpp = stacks == "all"
                record_context = context is not None
                ss = torch.cuda.memory._snapshot()

                trace_plot(ss)
                trace_plot(ss, filter_freed=True)
                segment_plot(ss)
                text = json.dumps(ss)

                self.assertTrue(record_context == ("test_memory_plots" in text))
                self.assertTrue(cpp == ("::rand" in text))
                self.assertTrue(str(128 * 128 * 4) in text)

            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots_free_stack(self):
        # [FUNCTION DOC] For each context mode (alloc, all, state), allocates a tensor in one
        # helper and frees it in another, snapshots the history as JSON text and asserts the
        # free helper name appears only for mode all and the allocate helper name appears for
        # all modes except state.
        for context in ["alloc", "all", "state"]:
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = None

                def thealloc():
                    # [FUNCTION DOC] HELPER: Allocates a random CUDA tensor into the
                    # enclosing variable.
                    nonlocal x
                    x = torch.rand(3, 4, device="cuda")

                def thefree():
                    # [FUNCTION DOC] HELPER: Deletes the enclosing variable holding the tensor.
                    nonlocal x
                    del x

                thealloc()
                thefree()
                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("thefree" in ss), (context == "all"))
                self.assertEqual(("thealloc" in ss), (context != "state"))
            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(not has_triton(), "test needs triton")
    @requiresCppContext
    @serialTest()
    def test_memory_compile_regions(self):
        # [FUNCTION DOC] Records memory history with compile context for a small compiled model
        # with a data-dependent branch and checks the sequence of distinct compile region labels
        # in the device traces equals the expected three regions (0/0, 1/0, 0/0). RuntimeError
        # during the body is swallowed; cleanup resets dynamo, deletes objects and collects
        # garbage. Needs triton, the C++ context and not cudaMallocAsync; serial.
        expected_allocation_sequence = [
            "Torch-Compiled Region: 0/0",
            "Torch-Compiled Region: 1/0",
            "Torch-Compiled Region: 0/0",
        ]

        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Two linear layers of size 10 with a data dependent branch in
            # between (adds or subtracts one depending on the sign of the sum), causing a graph
            # break when compiled.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first linear layer, adds or subtracts one
                # based on whether the sum is positive, then applies the second layer and
                # returns the result.
                x = self.linear1(x)

                if x.sum() > 0:
                    x = x + 1
                else:
                    x = x - 1

                x = self.linear2(x)

                return x

        try:
            torch.cuda.memory.empty_cache()
            # Region ids come from Dynamo's process-global frame counter, which
            # reset() zeroes; do it before compiling so the region strings are
            # deterministic.
            torch._dynamo.reset()
            input_tensor = torch.randn(1, 10, device="cuda")
            # Create an instance of the model
            model = MyModel()
            model.to("cuda")
            # Compile the model using torch.compile
            compiled_model = torch.compile(model)
            # Create a sample input tensor
            torch.cuda.memory._record_memory_history(
                context="all", compile_context=True
            )
            compiled_model(input_tensor)
            ss = torch.cuda.memory._snapshot()["device_traces"]
            device_idx = 0
            allocation_sequence = []
            while len(ss[device_idx]) == 0:
                device_idx = device_idx + 1
            for s in ss[device_idx]:
                context = s["compile_context"]
                if context == "N/A":
                    continue
                if len(allocation_sequence) > 0 and allocation_sequence[-1] == context:
                    continue
                allocation_sequence.append(context)
            self.assertEqual(allocation_sequence, expected_allocation_sequence)
        except RuntimeError as e:
            pass
        finally:
            torch.cuda.memory._record_memory_history(None)
            # Clean up the model and input tensor to fix memory leaks in other tests
            del compiled_model, model, input_tensor
            ss = None
            torch._dynamo.reset()
            # This test requires to run gc.collec() to fix other memory tests
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots_history_context(self):
        # [FUNCTION DOC] Checks that switching history recording settings on the fly
        # controls whether call stacks are captured: allocations made while recording with
        # context and python stacks (first and third helpers) appear in the snapshot text,
        # the one made while context was None does not. Not for cudaMallocAsync or platforms
        # without C++ context.
        try:
            torch.cuda.memory.empty_cache()
            x = None

            def should_capture1():
                # [FUNCTION DOC] HELPER: Allocates a 4 by 4 random CUDA tensor into the
                # enclosing variable.
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            def should_not_capture():
                # [FUNCTION DOC] HELPER: Allocates a 3 by 4 random CUDA tensor into the
                # enclosing variable.
                nonlocal x
                x = torch.rand(3, 4, device="cuda")

            def should_capture2():
                # [FUNCTION DOC] HELPER: Allocates a 4 by 4 random CUDA tensor into the
                # enclosing variable.
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            # Recording with context and python call stacks should capture the call stack.
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture1()
            # Recording with context=None should not capture the call stack.
            torch.cuda.memory._record_memory_history(context=None)
            should_not_capture()
            # Recording with context and python call stacks should capture the call stack.
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture2()

            ss = json.dumps(torch.cuda.memory._snapshot())
            self.assertTrue("should_capture1" in ss)
            self.assertTrue("should_not_capture" not in ss)
            self.assertTrue("should_capture2" in ss)
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots_free_segment_stack(self):
        # [FUNCTION DOC] For each context mode, records history, allocates and frees a tensor,
        # empties the cache, and asserts the cache-emptying function name shows up in the
        # snapshot text only for mode all (segment free stack recorded).
        for context in ["alloc", "all", "state"]:
            try:
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = torch.rand(3, 4, device="cuda")
                del x
                torch.cuda.memory.empty_cache()

                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("empty_cache" in ss), (context == "all"))
            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots_metadata(self):
        # [FUNCTION DOC] For each context mode, sets memory metadata, records history, allocates
        # and frees a tensor and empties the cache, clears the metadata, then asserts every
        # event in the first device trace carries the metadata string.
        for context in ["alloc", "all", "state"]:
            try:
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._set_memory_metadata("metadata test")
                torch.cuda.memory._record_memory_history(context=context)
                x = torch.rand(3, 4, device="cuda")
                del x
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._set_memory_metadata("")

                ss = torch.cuda.memory._snapshot()
                for event in ss["device_traces"][0]:
                    self.assertTrue(event["user_metadata"] == "metadata test")
            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @requiresCppContext
    def test_memory_plots_annotate(self):
        # [FUNCTION DOC] Tests tensor annotation in memory history: sets allocation-time
        # metadata, allocates, annotates the tensor twice and via a view; checks two annotate
        # events with the storage base address and texts in order, that a view annotation
        # resolves to the base address, that the allocation event keeps the allocation-time
        # metadata, and that annotating an address after the tensor is gone raises RuntimeError
        # (no live allocation).
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._set_memory_metadata("alloc-time metadata")
            torch.cuda.memory._record_memory_history(context="all")
            x = torch.rand(3, 4, device="cuda")
            torch.cuda.memory._set_memory_metadata("")
            ptr = x.untyped_storage().data_ptr()
            torch.cuda.memory._annotate_tensor(x, "retained by autograd")
            torch.cuda.memory._annotate_tensor(x, "second annotation")

            ss = torch.cuda.memory._snapshot()
            notes = [e for e in ss["device_traces"][0] if e["action"] == "annotate"]
            self.assertEqual(len(notes), 2)
            for e in notes:
                self.assertEqual(e["addr"], ptr)
            self.assertEqual(notes[0]["user_metadata"], "retained by autograd")
            self.assertEqual(notes[1]["user_metadata"], "second annotation")
            # views resolve to the storage base address
            view = x[1:]
            self.assertNotEqual(view.data_ptr(), ptr)
            torch.cuda.memory._annotate_tensor(view, "via view")
            ss = torch.cuda.memory._snapshot()
            notes = [e for e in ss["device_traces"][0] if e["action"] == "annotate"]
            self.assertEqual(notes[-1]["addr"], ptr)
            self.assertEqual(notes[-1]["user_metadata"], "via view")
            # allocation-time metadata is not clobbered
            alloc_events = [
                e
                for e in ss["device_traces"][0]
                if e["action"] == "alloc" and e["addr"] == ptr
            ]
            self.assertEqual(len(alloc_events), 1)
            self.assertEqual(alloc_events[0]["user_metadata"], "alloc-time metadata")
            # annotating a dead pointer raises
            del x, view
            torch.cuda.synchronize()
            with self.assertRaisesRegex(RuntimeError, "no live allocation"):
                torch._C._cuda_annotateMemory(ptr, "should fail")
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot_script(self):
        # [FUNCTION DOC] Records state history with python stacks, creates a tensor inside a
        # scripted function, and asserts the snapshot block of that size has the top frame named
        # foo (script frame names are recorded). Not for cudaMallocAsync.
        try:
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="python")

            @torch.jit.script
            def foo():
                # [FUNCTION DOC] HELPER: Scripted function returning a 311 by 411 random
                # CUDA tensor.
                return torch.rand(311, 411, device="cuda")

            x = foo()

            ss = torch.cuda.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertEqual(b["frames"][0]["name"], "foo")
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot_skip_actions(self):
        # [FUNCTION DOC] For several skip_actions settings (single action, multiple actions,
        # alloc, segment actions, an empty list and None), records history with the skip list,
        # does an allocation and free, takes a snapshot, finds the first device trace with
        # entries and asserts it is non-empty. When a skip list is given, none of its action
        # names may appear in the recorded actions and at least one other action must remain.
        # Recording is turned off in the finally block. Not for cudaMallocAsync.
        """Test that skip_actions parameter correctly filters out specified action types."""
        # Test cases: (skip_actions, description)
        test_cases = [
            (["free_requested"], "single action"),
            (["free_requested", "free_completed"], "multiple actions"),
            (["alloc"], "alloc action"),
            (["segment_alloc", "segment_free"], "segment actions"),
            ([], "empty list"),
            (None, "None default"),
        ]

        for skip_actions, description in test_cases:
            with self.subTest(skip_actions=skip_actions, description=description):
                try:
                    torch.cuda.memory.empty_cache()
                    torch.cuda.memory._record_memory_history(
                        context="all", stacks="python", skip_actions=skip_actions
                    )

                    # Perform allocations and frees to generate various actions
                    x = torch.rand(128, 128, device="cuda")
                    del x
                    torch.cuda.synchronize()

                    # Take snapshot to potentially generate snapshot event
                    ss = torch.cuda.memory._snapshot()
                    device_idx = 0
                    while len(ss["device_traces"][device_idx]) == 0:
                        device_idx = device_idx + 1
                    device_trace = ss["device_traces"][device_idx]

                    # Verify traces are not empty (other actions are still recorded)
                    self.assertGreater(
                        len(device_trace),
                        0,
                        lambda msg: f"{msg}\nTrace should not be empty for skip_actions={skip_actions}",
                    )

                    if skip_actions:
                        all_actions = {event["action"] for event in device_trace}

                        # Verify skipped actions are not in traces
                        for sa in skip_actions:
                            self.assertNotIn(
                                sa,
                                all_actions,
                                lambda msg: f"{msg}\nAction {sa} should be skipped with skip_actions={skip_actions}",
                            )
                        # Check that we have actions that are NOT in the skip list
                        non_skipped_actions = all_actions - set(skip_actions)
                        self.assertGreater(
                            len(non_skipped_actions),
                            0,
                            lambda msg: f"{msg}\nShould have non-skipped actions for skip_actions={skip_actions}",
                        )

                finally:
                    torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_host_memory_snapshot(self):
        # [FUNCTION DOC] Records history including pinned host memory, allocates a pinned tensor
        # via a named helper and checks the snapshot: host segments have device -1 and exactly
        # one block each, the active block has a frame from this test file, the helper name
        # appears in the JSON, and host traces contain segment_alloc and alloc. After freeing,
        # free_requested and free_completed appear; after emptying the host cache the last trace
        # is segment_free.
        try:
            torch._C._host_emptyCache()
            torch.cuda.memory._record_memory_history(
                "all",
                context="alloc",
                stacks="python",
                record_pinned_host_memory=True,
                clear_history=True,
            )

            def host_alloc():
                # [FUNCTION DOC] HELPER: Allocates a pinned 311 by 411 CPU tensor so that its
                # name shows up in the recorded host allocation stack.
                return torch.empty(311, 411, pin_memory=True)

            x = host_alloc()

            ss = torch.cuda.memory._snapshot()

            # Find our block by size in host_segments
            found_it = False
            for seg in ss["host_segments"]:
                self.assertEqual(seg["device"], -1)
                self.assertEqual(len(seg["blocks"]), 1)
                b = seg["blocks"][0]
                if b["state"] == "active_allocated":
                    self.assertTrue("test_cuda" in b["frames"][0]["filename"])
                    found_it = True
            self.assertTrue(found_it)

            # Verify alloc stack trace contains the named function
            text = json.dumps(ss)
            self.assertIn("host_alloc", text)

            # Verify trace actions: segment_alloc then alloc
            actions = [te["action"] for te in ss["host_traces"]]
            self.assertIn("segment_alloc", actions)
            self.assertIn("alloc", actions)

            # Free and verify free traces
            del x
            ss = torch.cuda.memory._snapshot()
            actions = [te["action"] for te in ss["host_traces"]]
            self.assertIn("free_requested", actions)
            self.assertIn("free_completed", actions)

            # Empty cache and verify segment_free is the last action
            torch._C._host_emptyCache()
            ss = torch.cuda.memory._snapshot()
            self.assertEqual(ss["host_traces"][-1]["action"], "segment_free")

        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_host_memory_snapshot_disabled(self):
        # [FUNCTION DOC] Records history without enabling pinned host recording, allocates a
        # pinned tensor and asserts the snapshot has no host segments and no host traces. Not
        # for cudaMallocAsync.
        try:
            torch._C._host_emptyCache()
            torch.cuda.memory._record_memory_history("all")
            x = torch.empty(1024, pin_memory=True)
            ss = torch.cuda.memory._snapshot()
            self.assertEqual(len(ss["host_segments"]), 0)
            self.assertEqual(len(ss["host_traces"]), 0)
            del x
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_host_memory_snapshot_max_entries(self):
        # [FUNCTION DOC] Records history with a maximum of 4 entries and pinned host
        # recording, then allocates, frees and empties the host cache 20 times; asserts the
        # number of host traces is exactly 4, showing the ring buffer is full and bounded. Not
        # for cudaMallocAsync.
        try:
            torch._C._host_emptyCache()
            max_entries = 4
            torch.cuda.memory._record_memory_history(
                "all",
                max_entries=max_entries,
                record_pinned_host_memory=True,
                clear_history=True,
            )

            # Allocate and free enough pinned tensors to overflow the ring
            # buffer many times over (each alloc/free produces multiple trace
            # entries: segment_alloc, alloc, free_requested, free_completed).
            for i in range(20):
                x = torch.empty(1024 + i, pin_memory=True)
                del x
                torch._C._host_emptyCache()

            ss = torch.cuda.memory._snapshot()
            self.assertLessEqual(len(ss["host_traces"]), max_entries)
            # We did enough work that the buffer should be full.
            self.assertEqual(len(ss["host_traces"]), max_entries)
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_host_memory_snapshot_none_disables(self):
        # [FUNCTION DOC] Enables host recording and checks some host traces appear; then turns
        # the history off with None and checks that a later pinned allocation leaves no host
        # segments or traces in the snapshot. Not for cudaMallocAsync.
        try:
            torch._C._host_emptyCache()
            torch.cuda.memory._record_memory_history(
                "all",
                record_pinned_host_memory=True,
                clear_history=True,
            )
            x = torch.empty(1024, pin_memory=True)
            ss = torch.cuda.memory._snapshot()
            self.assertGreater(len(ss["host_traces"]), 0)
            del x

            # None should turn off host recording without record_pinned_host_memory=True
            torch.cuda.memory._record_memory_history(None)

            torch._C._host_emptyCache()
            y = torch.empty(2048, pin_memory=True)
            ss = torch.cuda.memory._snapshot()
            self.assertEqual(len(ss["host_segments"]), 0)
            self.assertEqual(len(ss["host_traces"]), 0)
            del y
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_host_memory_snapshot_host_only(self):
        # [FUNCTION DOC] Records history with pinned host recording on and CUDA recording off;
        # allocates pinned and CUDA tensors; asserts host traces and host segments exist while
        # every device trace is empty. Not for cudaMallocAsync.
        try:
            torch._C._host_emptyCache()
            torch.cuda.memory._record_memory_history(
                "all",
                record_pinned_host_memory=True,
                record_cuda=False,
                clear_history=True,
            )

            x = torch.empty(311, 411, pin_memory=True)
            d = torch.empty(311, 411, device="cuda")

            ss = torch.cuda.memory._snapshot()

            # Host traces should be recorded.
            self.assertGreater(len(ss["host_traces"]), 0)
            self.assertGreater(len(ss["host_segments"]), 0)

            # CUDA traces should be empty since record_cuda=False, even though
            # we allocated a CUDA tensor.
            for device_trace in ss["device_traces"]:
                self.assertEqual(len(device_trace), 0)

            del x
            del d
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/180263"
    )
    @serialTest()
    def test_max_split_expandable(self):
        # [FUNCTION DOC] Sets a process memory limit of about 120 MB above current reserved
        # memory, then allocates three 40 MB tensors with expandable segments toggled off, on,
        # off (with a 40 MB max split size), and expects a fourth 40 MB allocation to raise
        # OutOfMemoryError. Then frees them and allocates 120 MB to force cached block release
        # with expandable segments in the free list. Restores the memory fraction and expandable
        # setting in the finally block. Serial; skipped on Linux and slow runs (tracked issue).
        try:
            orig = torch.cuda.get_per_process_memory_fraction()
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual(int(round(fraction_allowed * all_memory)), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                # [FUNCTION DOC] HELPER: Returns an int8 tensor of ones on CUDA with the given
                # number of mebibytes.
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,max_split_size_mb:40"
            )
            b = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            c = alloc(40)
            with self.assertRaises(torch.OutOfMemoryError):
                alloc(40)
            del a, b, c
            # force release_cached_blocks to run with some expandable segments in the free list
            alloc(120)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(orig)
            # Test toggles expandable_segments internally; restore the
            # suite's baseline so subsequent tests see consistent state.
            torch.cuda.memory._set_allocator_settings(
                f"expandable_segments:{EXPANDABLE_SEGMENTS}"
            )

    @serialTest()
    def test_garbage_collect_expandable(self):
        # [FUNCTION DOC] Sets a process memory limit of 120 MB above reserved memory and checks
        # it exactly, then allocates 40 MB with expandable segments off and 40 MB with them on
        # (both with a 0.5 garbage collection threshold), frees them and allocates 80 MB so
        # garbage collection runs with an expandable segment block in the free list. Restores
        # the memory fraction and expandable segments setting at the end. Serial.
        try:
            orig = torch.cuda.get_per_process_memory_fraction(0)
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual((fraction_allowed * all_memory), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                # [FUNCTION DOC] HELPER: Returns an int8 tensor of ones on CUDA with the given
                # number of mebibytes.
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,garbage_collection_threshold:0.5"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,garbage_collection_threshold:0.5"
            )
            b = alloc(40)
            del a, b
            # causes GC to run. The expandable segment block will be split
            # so GC would not attempt to free it anyway, but this at least makes sure
            # expandable_segment blocks can be in the free list when this is called.
            alloc(80)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(orig)
            # Test toggles expandable_segments internally; restore the
            # suite's baseline so subsequent tests see consistent state.
            torch.cuda.memory._set_allocator_settings(
                f"expandable_segments:{EXPANDABLE_SEGMENTS}"
            )

    def test_allocator_settings(self):
        # [FUNCTION DOC] Tests allocator settings strings through the memory API. Checks default
        # rounding, a single-value round-up-power-of-two-divisions setting (allocated delta
        # equals the divided size; native allocator), that setting other options resets the
        # divisions, an array-form divisions setting for small and big sizes, the lock release
        # setting, and that invalid settings (unknown keys, threshold above 1, tiny max split
        # size, invalid booleans and thread counts) raise ValueError, plus valid and invalid
        # throw-on-malloc-oom values.
        def power2_div(size, div_factor):
            # [FUNCTION DOC] HELPER: Computes the expected rounded allocation size for a size
            # and a division factor: finds the next power of two at or above the size and, if
            # not equal, steps upward from half of it in increments of half that power divided
            # by the factor until the size is reached; returns the result.
            pow2 = 1
            while pow2 < size:
                pow2 = pow2 * 2
            if pow2 == size:
                return pow2
            step = pow2 / 2 / div_factor
            ret = pow2 / 2
            while ret < size:
                ret = ret + step
            return ret

        torch.cuda.memory.empty_cache()
        key_allocated = (
            "active_bytes.all.allocated"
            if not TEST_CUDAMALLOCASYNC
            else "allocated_bytes.all.current"
        )
        key_requested = "requested_bytes.all.allocated"

        nelems = 21 * 1024 * 1024
        nbytes = 4 * nelems  # floats are 4 bytes

        nelems_big = 100 * 1024 * 1024
        nbytes_big = 4 * nelems_big  # floats are 4 bytes

        start_mem = torch.cuda.memory_stats()[key_allocated]
        torch.cuda.memory._set_allocator_settings("")
        x = torch.rand(nelems, device="cuda")

        # test roundup_power2_divisions single value syntax
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        start_requested = torch.cuda.memory_stats()[key_requested]
        torch.cuda.memory._set_allocator_settings("roundup_power2_divisions:4")
        y = torch.rand(nelems, device="cuda")

        pow2_div4_mem = torch.cuda.memory_stats()[key_allocated]
        current_requested = torch.cuda.memory_stats()[key_requested]

        self.assertEqual(reg_mem - start_mem, nbytes)
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div4_mem - reg_mem, power2_div(nbytes, 4))
            self.assertEqual(current_requested - start_requested, nbytes)

        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.5")
        torch.cuda.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,max_split_size_mb:40"
        )

        # should have reset the power2 divisions now
        torch.cuda.memory.empty_cache()
        start_mem = torch.cuda.memory_stats()[key_allocated]
        z = torch.rand(nelems, device="cuda")
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        self.assertEqual(reg_mem - start_mem, nbytes)

        # roundup_power2_divisions knob array syntax
        torch.cuda.memory.empty_cache()
        torch.cuda.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,roundup_power2_divisions:[64:8,128:2,256:2,512:2,1024:1,>:1]"
        )
        start_mem = torch.cuda.memory_stats()[key_allocated]
        w = torch.rand(nelems, device="cuda")

        pow2_div8_mem = torch.cuda.memory_stats()[key_allocated]
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div8_mem - start_mem, power2_div(nbytes, 8))

        torch.cuda.memory.empty_cache()
        start_mem = torch.cuda.memory_stats()[key_allocated]
        v = torch.rand(nelems_big, device="cuda")

        pow2_div2_mem = torch.cuda.memory_stats()[key_allocated]
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div2_mem - start_mem, power2_div(nbytes_big, 2))

        torch.cuda.memory.empty_cache()
        torch.cuda.memory._set_allocator_settings("release_lock_on_cudamalloc:True")
        start_mem = torch.cuda.memory_stats()[key_allocated]
        w = torch.rand(nelems, device="cuda")
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        self.assertEqual(reg_mem - start_mem, nbytes)

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings("foo:1,bar:2")

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings(
                "garbage_collection_threshold:1.2"
            )

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings("max_split_size_mb:2")

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings(
                "release_lock_on_cudamalloc:none"
            )

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings(
                "pinned_use_cuda_host_register:none"
            )

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings(
                "pinned_num_register_threads:none"
            )

        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings(
                "pinned_num_register_threads:1024"
            )

        # Test throw_on_cudamalloc_oom config parsing - valid formats
        torch.cuda.memory._set_allocator_settings("throw_on_cudamalloc_oom:True")
        torch.cuda.memory._set_allocator_settings("throw_on_cudamalloc_oom:False")

        # Test throw_on_cudamalloc_oom config parsing - invalid formats
        with self.assertRaises(ValueError):
            torch._C._accelerator_setAllocatorSettings("throw_on_cudamalloc_oom:maybe")

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "throw_on_cudamalloc_oom not supported")
    @serialTest()
    def test_throw_on_cudamalloc_oom(self):
        # [FUNCTION DOC] Tests throw_on_cudamalloc_oom with a per-process fraction. With the
        # option off, a 10 MiB allocation works. With it on and fraction 0.005, an allocation
        # slightly above the fraction limit raises torch.cuda.OutOfMemoryError and the
        # num_oom_rejections statistic is positive. Settings are reset to off and fraction 1.0
        # afterward. Serial; not for cudaMallocAsync.
        """Test that throw_on_cudamalloc_oom + per_process_memory_fraction works correctly."""
        torch.cuda.empty_cache()
        device = torch._C._cuda_getDevice()
        torch._C._cuda_resetAccumulatedMemoryStats(device)

        try:
            # Test 1: With rejection disabled (default), allocations should succeed
            torch._C._accelerator_setAllocatorSettings("throw_on_cudamalloc_oom:False")
            x = torch.empty(10 * 1024 * 1024, dtype=torch.int8, device="cuda")
            del x
            torch.cuda.empty_cache()

            # Test 2: With throw_on_cudamalloc_oom enabled and a tight memory
            # fraction, allocations that exceed the fraction limit should be
            # preemptively rejected with OutOfMemoryError.
            # Both settings must go through _accelerator_setAllocatorSettings so
            # they are read from CUDAAllocatorConfig.
            fraction = 0.005
            torch._C._accelerator_setAllocatorSettings(
                f"throw_on_cudamalloc_oom:True,per_process_memory_fraction:{fraction}"
            )

            total_mem = torch.cuda.get_device_properties(0).total_memory
            # Allocate the allowed threshold + 1 MiB to guarantee rejection
            alloc_bytes = int(total_mem * fraction) + 1024 * 1024
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                torch.empty(alloc_bytes, dtype=torch.int8, device="cuda")

            # Check that rejection counter was incremented
            stats = torch.cuda.memory_stats()
            self.assertGreater(stats["num_oom_rejections"], 0)

        finally:
            torch.cuda.empty_cache()
            torch._C._accelerator_setAllocatorSettings(
                "throw_on_cudamalloc_oom:False,per_process_memory_fraction:1.0"
            )

    @skipIfRocmVersionAtLeast([7, 14])
    def test_allocator_backend(self):
        # [FUNCTION DOC] Runs a child process (with a minimal environment, keeping a few
        # system variables on Windows) that sets the allocator config environment variable to
        # select the cudaMallocAsync backend with other options before importing torch,
        # initializes CUDA and prints the backend name; asserts it is cudaMallocAsync. Skipped
        # on ROCm 7.14+.
        def subprocess_env():
            # [FUNCTION DOC] HELPER: Returns the environment mapping for the subprocess: on
            # Windows only a few system variables copied from the current environment (an empty
            # one breaks loading), otherwise an empty mapping.
            if IS_WINDOWS:
                # An empty env breaks Windows stdlib/CUDA extension loading (no PATH).
                return {
                    k: os.environ[k]
                    for k in (
                        "SYSTEMROOT",
                        "SYSTEMDRIVE",
                        "WINDIR",
                        "COMSPEC",
                        "PATH",
                    )
                    if k in os.environ
                }
            return {}

        def check_output(script: str) -> str:
            # [FUNCTION DOC] HELPER: Runs a script in a Python subprocess with the minimal
            # environment as text mode (using the temp directory as working directory on
            # Windows) and returns stripped output.
            kwargs = {"env": subprocess_env(), "text": True}
            if IS_WINDOWS:
                kwargs["cwd"] = tempfile.gettempdir()
            return subprocess.check_output(
                [sys.executable, "-c", script], **kwargs
            ).strip()

        test_script = """\
import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:20,backend:cudaMallocAsync,release_lock_on_cudamalloc:none"
import torch
torch.cuda.init()
print(torch.cuda.get_allocator_backend())
"""
        rc = check_output(test_script)
        self.assertEqual(rc, "cudaMallocAsync")

    def test_allocator_memory_fraction_setting(self):
        # [FUNCTION DOC] Tests the per_process_memory_fraction allocator config via environment
        # variable. For fractions 0.0, 0.5 and 1.0, a child process prints the fraction read
        # back from the memory API and must match the value; for -0.1 and 1.1 the child must
        # fail with an error saying the fraction is invalid.
        def make_env(fraction):
            # [FUNCTION DOC] HELPER: Copies the environment and rewrites the allocator config
            # variable, dropping any existing fraction entry and appending the given fraction.
            env = os.environ.copy()
            var = "PYTORCH_CUDA_ALLOC_CONF"
            key = "per_process_memory_fraction"
            value = [
                x
                for x in env.get(var, "").split(",")
                if len(x) > 0 and not x.startswith(f"{key}:")
            ]
            value.append(f"{key}:{fraction}")
            env[var] = ",".join(value)
            return env

        def run_test(value):
            # [FUNCTION DOC] HELPER: Runs a child Python process using the environment for the
            # given fraction; the child prints the per-process memory fraction of the current
            # device. Returns the completed process, raising on a non-zero exit.
            test_script = """\
import os
import torch
device = torch._C._cuda_getDevice()
value = torch.cuda.memory.get_per_process_memory_fraction(device)
print(value, end="")
            """
            return subprocess.run(
                [sys.executable, "-c", test_script],
                env=make_env(value),
                text=True,
                check=True,
                capture_output=True,
            )

        self.assertEqual(run_test(0.0).stdout, "0.0")
        self.assertEqual(run_test(0.5).stdout, "0.5")
        self.assertEqual(run_test(1.0).stdout, "1.0")

        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_test(-0.1)
        if "per_process_memory_fraction is invalid" not in e.exception.stderr:
            raise AssertionError(e.exception.stderr)

        with self.assertRaises(subprocess.CalledProcessError) as e:
            run_test(1.1)
        if "per_process_memory_fraction is invalid" not in e.exception.stderr:
            raise AssertionError(e.exception.stderr)

    def test_cachingAllocator_raw_alloc(self):
        # [FUNCTION DOC] Tests that the raw allocation entry points respect the caching
        # allocator enable switch. With the caching allocator disabled, a raw allocation of 1
        # MiB changes the requested bytes statistic by 0; after re-enabling (in the finally
        # block) the delta equals 1 MiB.
        # Test that raw_alloc respects the setting that
        # activates/deactivates the caching allocator

        # Helper function that calls raw_alloc and returns
        # relevant field in data structure
        def requested_bytes_alloc_stats(raw_alloc_size, stream):
            # [FUNCTION DOC] HELPER: Reads the requested-bytes allocated stat, raw-allocates the
            # given size on the given stream, reads the stat again, frees the pointer and
            # returns the increase.
            start = torch.cuda.memory_stats()["requested_bytes.all.allocated"]
            mem_ptr = torch._C._cuda_cudaCachingAllocator_raw_alloc(
                raw_alloc_size, stream
            )
            finish = torch.cuda.memory_stats()["requested_bytes.all.allocated"]
            torch._C._cuda_cudaCachingAllocator_raw_delete(mem_ptr)
            return finish - start

        torch.cuda.empty_cache()
        device = torch._C._cuda_getDevice()
        stream = torch._C._cuda_getCurrentRawStream(device)
        torch._C._cuda_resetAccumulatedMemoryStats(device)

        # size of allocation
        raw_alloc_size = 1024 * 1024  # 1 MB

        try:
            # Deactivate the caching allocator
            torch.cuda.caching_allocator_enable(False)

            # For a deactivated caching allocator, result is zero
            cuda_alloc_size = requested_bytes_alloc_stats(raw_alloc_size, stream)
            self.assertEqual(cuda_alloc_size, 0)

        finally:
            # Make sure we get back to the default state that is
            # an activated caching allocator
            torch.cuda.caching_allocator_enable(True)

            # For an active caching allocator, result matches raw_alloc_size
            cuda_alloc_size = requested_bytes_alloc_stats(raw_alloc_size, stream)
            self.assertEqual(cuda_alloc_size, raw_alloc_size)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    @parametrize("max_split_size_mb_setting", [False, True])
    def test_raises_oom(self, max_split_size_mb_setting):
        # [FUNCTION DOC] Parametrized by whether a max split size is set. Optionally sets a
        # 1024 MB max split size and empties the cache (to exercise more allocator code),
        # then asserts an absurdly large allocation raises torch.cuda.OutOfMemoryError.
        # Skipped on Jetson.
        if max_split_size_mb_setting:
            # CudaCachingAllocator does early return when searching available blocks
            # if max_split_size_mb is not set
            # Setting this triggers more parts of the code
            torch.cuda.memory._set_allocator_settings("max_split_size_mb:1024")
            torch.cuda.memory.empty_cache()
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")

    @unittest.skipIf(
        not ((IS_X86 or IS_ARM64) and IS_LINUX), "cpp traces are linux x86/aarch64 only"
    )
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_cpp_memory_snapshot_pickle(self):
        # [FUNCTION DOC] Builds a C++ extension exposing the pickled memory snapshot and the
        # history recording switch. For recording without and with context, allocates from a
        # scripted function and unpickles the snapshot: the active block of the expected
        # size must exist, with context its frames mention the C++ random function, the
        # script function name and python test case file; the last device trace is an alloc
        # of that size. Recording is turned off in the finally block. Linux x86/aarch64
        # only, not for cudaMallocAsync.
        from torch.utils.cpp_extension import load_inline

        source = """
        #include <torch/csrc/cuda/memory_snapshot.h>
        py::object do_snapshot() {
            std::string data = torch::cuda::_memory_snapshot_pickled();
            return py::bytes(data);
        }
        void record(bool e, bool ctx) {
            torch::cuda::_record_memory_history(e, ctx, 10, ctx, ctx);
        }
        """
        m = load_inline(
            name="snapshot", cpp_sources=[source], functions=["do_snapshot", "record"]
        )
        for ctx in (False, True):
            try:
                m.record(True, ctx)

                @torch.jit.script
                def the_script_fn():
                    # [FUNCTION DOC] HELPER: Scripted function returning a 311 by 411 random
                    # CUDA tensor.
                    return torch.rand(311, 411, device="cuda")

                def run():
                    # [FUNCTION DOC] HELPER: Calls the scripted function and returns the
                    # unpickled memory snapshot from the extension.
                    t = the_script_fn()
                    return pickle.loads(m.do_snapshot())

                mem = run()
                found = False
                for s in mem["segments"]:
                    for b in s["blocks"]:
                        if b["state"] == "active_allocated":
                            if b["requested_size"] == 311 * 411 * 4:
                                if ctx:
                                    frame_text = str(b["frames"])
                                    # C++ frame
                                    self.assertTrue("::rand" in frame_text)
                                    # script frame
                                    self.assertTrue("the_script_fn" in frame_text)
                                    # python frame
                                    self.assertTrue("case.py" in frame_text)
                                found = True
                last_action = mem["device_traces"][0][-1]
                self.assertEqual(last_action["action"], "alloc")
                self.assertEqual(last_action["size"], 311 * 411 * 4)
                self.assertTrue(found)
            finally:
                m.record(False, False)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_notifies_oom(self):
        # [FUNCTION DOC] Attaches an out-of-memory observer, triggers an out-of-memory error
        # with a huge allocation (expecting OutOfMemoryError), and asserts the observer was
        # called. Skipped for cudaMallocAsync and on Jetson.
        x = False

        def cb(device, alloc, device_alloc, device_free):
            # [FUNCTION DOC] HELPER: Observer callback taking device, allocation size and the
            # device allocated and free amounts; only sets the enclosing flag.
            nonlocal x
            x = True

        torch._C._cuda_attach_out_of_memory_observer(cb)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")
        self.assertTrue(x)

    def test_allocator_fuzz(self):
        # [FUNCTION DOC] Fuzz test for the allocator (skipped on ROCm with expandable segments).
        # With a fixed random seed, performs 10000 random actions (allocate int32 tensors of 2
        # to 20 MB filled with their id, free a random one verifying contents, or empty the
        # cache), keeping the total under about 100 MB, and restores the random state afterward.
        # fuzz
        if (
            torch.version.hip
            and "expandable_segments:True"
            in torch._C._accelerator_getAllocatorSettings()
        ):
            raise unittest.SkipTest(
                "ROCm needs https://github.com/ROCm/rocm-systems/pull/3023"
            )
        state = random.getstate()
        random.seed(123)
        N = 10000
        try:
            mem = []
            total = 0
            c = 0

            def alloc():
                # [FUNCTION DOC] HELPER: Allocates a CUDA int32 tensor of random size, filled
                # with a counter value, appends it with the counter to the list, advances the
                # counter and adds its size to the running total.
                nonlocal total, c
                b = random.randrange(2 * 1024 * 1024 // 4, 20 * 1024 * 1024 // 4)
                mem.append((c, torch.full((b,), c, dtype=torch.int32, device="cuda")))
                c += 1
                total += b

            def free():
                # [FUNCTION DOC] HELPER: Removes a random entry from the list, asserts its
                # tensor is entirely the recorded counter value, and subtracts its size from the
                # running total.
                nonlocal total
                idx = random.randrange(0, len(mem))
                v, x = mem.pop(idx)
                self.assertTrue(torch.all(v == x))
                total -= x.numel()

            choices = [alloc, free, torch.cuda.memory.empty_cache]
            for _ in range(N):
                while total >= 1024 * 1024 * 1024 / (4 * 10):
                    free()
                (action,) = random.choices(choices, weights=[1, 1 if mem else 0, 0.1])
                action()
        finally:
            random.setstate(state)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_nvml_get_handler(self):
        # [FUNCTION DOC] Checks the NVML (or AMD SMI on ROCm) handler getter returns a non-None
        # handler. Skipped without pynvml/amdsmi and on ROCm 7.14+.
        if not torch.version.hip:
            self.assertTrue(torch.cuda._get_pynvml_handler() is not None)
        else:
            self.assertTrue(torch.cuda._get_amdsmi_handler() is not None)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_temperature(self):
        # [FUNCTION DOC] Checks the reported GPU temperature is between 0 and 150. Skipped
        # without pynvml/amdsmi and on ROCm 7.14+.
        self.assertTrue(0 <= torch.cuda.temperature() <= 150)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_device_memory_used(self):
        # [FUNCTION DOC] Checks device_memory_used: records usage, allocates 512 MiB, syncs,
        # empties the cache and reads again. On Windows only sanity is checked (non-negative,
        # not decreasing, not above total memory); elsewhere the increase must be within a
        # factor of 32 of the allocation.
        """
        Verify used device memory in bytes.
        On Windows the NVML used value has been observed not to increase after
        a CUDA allocation (delta 0); we only assert API sanity there (non-negative,
        non-decreasing after alloc, <= total memory). Need to investigate expected behavior
        with Windows WDDM
        """
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        a = torch.cuda.device_memory_used()
        num_bytes = 512 * 1024**2
        _ = torch.empty(num_bytes, dtype=torch.int8, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        b = torch.cuda.device_memory_used()
        if IS_WINDOWS:
            # NVML used memory does not reflect CUDA allocations on WDDM; only check API sanity
            self.assertGreaterEqual(a, 0, "device_memory_used should be non-negative")
            self.assertGreaterEqual(b, 0, "device_memory_used should be non-negative")
            self.assertGreaterEqual(
                b, a, "used memory should not decrease after allocation"
            )
            total = torch.cuda.get_device_properties(0).total_memory
            self.assertLessEqual(a, total, "used should not exceed total device memory")
            self.assertLessEqual(b, total, "used should not exceed total device memory")
        else:
            mem_bytes = b - a
            # test the order of magnitude
            self.assertTrue(num_bytes // 32 <= mem_bytes <= num_bytes * 32)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_power_draw(self):
        # [FUNCTION DOC] Checks the reported power draw is non-negative. Skipped without
        # pynvml/amdsmi and on ROCm 7.14+.
        self.assertTrue(torch.cuda.power_draw() >= 0)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_clock_speed(self):
        # [FUNCTION DOC] Checks the clock rate is non-negative. Skipped without pynvml/amdsmi
        # and on ROCm 7.14+.
        self.assertTrue(torch.cuda.clock_rate() >= 0)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_raw_amdsmi_device_count(self):
        # [FUNCTION DOC] ROCm AMD SMI test: counts GPUs listed by the amd-smi command line tool
        # and asserts it equals the raw AMD SMI device count function.
        """
        This unit test will verify if the number of GPUs shown in `amd-smi
        list` is equivalent to the count returned by `_raw_device_count_amdsmi`.
        This should be unaffected by visible device settings.
        """
        raw_device_cnt = int(
            subprocess.check_output(
                "amd-smi list | grep 'GPU' | wc -l", shell=True
            ).strip()
        )
        self.assertEqual(torch.cuda._raw_device_count_amdsmi(), raw_device_cnt)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180227")
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_raw_amdsmi_device_uuids(self):
        # [FUNCTION DOC] ROCm AMD SMI test: reads GPU UUIDs from rocminfo output and asks for
        # the raw AMD SMI UUID list.
        # KNOWN ISSUE: the matching flag is reassigned on every loop iteration, so only the last
        # UUID decides the final assertion; earlier missing UUIDs are not detected.
        """
        This unit test will extract a list of UUIDs for each GPU using
        rocminfo information, and check whether each UUID is present in
        the output from `_raw_device_uuid_amdsmi` this allows us to test
        that the pytorch call is returning a correct list of UUIDs.
        """
        cmd = "rocminfo | grep -o 'Uuid:.*GPU-.*' | sed 's/Uuid:.*GPU-//'"
        uuids = subprocess.check_output(cmd, shell=True, text=True).strip().split("\n")
        uuids = [s.strip() for s in uuids]
        raw_uuids = torch.cuda._raw_device_uuid_amdsmi()
        for uuid in uuids:
            matching = True
            if not any(uuid in raw_id for raw_id in raw_uuids):
                matching = False
        self.assertEqual(True, matching)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180123")
    @unittest.skipIf(not TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_uuid_visible_devices(self):
        # [FUNCTION DOC] ROCm AMD SMI test: for each UUID from rocminfo, runs a child process
        # with that UUID in the CUDA or HIP visible devices variable (the other removed) and
        # asserts it prints a device count of 1.
        """
        This unit test will simulate an environment where a UUID is passed
        via CUDA/HIP_VISIBLE_DEVICES and ensure that the correct device count
        is returned. This allows us to test that the visible device functionality
        is operating as expected.
        """
        test_script = """\
import torch
import os
print(f"{torch.cuda.device_count()}")
        """
        cmd = "rocminfo | grep -o 'Uuid:.*GPU-.*' | sed 's/Uuid://'"
        uuids = subprocess.check_output(cmd, shell=True, text=True).strip().split("\n")
        uuids = [s.strip() for s in uuids]

        custom_envs = []
        for uuid in uuids:
            custom_envs.append(
                {"CUDA_VISIBLE_DEVICES": f"{uuid}", "HIP_VISIBLE_DEVICES": None}
            )
            custom_envs.append(
                {"HIP_VISIBLE_DEVICES": f"{uuid}", "CUDA_VISIBLE_DEVICES": None}
            )

        for env_config in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual("1", r)


MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1048576
SMALL_BUFFER = 2097152
LARGE_BUFFER = 20971520


def get_cudagraph_segments(pool_id):
    # [FUNCTION DOC] HELPER: Returns the list of memory snapshot segments whose pool id equals
    # the given pool id.
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


def get_all_cudagraph_segments():
    # [FUNCTION DOC] HELPER: Returns the list of memory snapshot segments that belong to any
    # graph pool (pool id different from the zero pair).
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]


def cudagraphify(fn, inputs, pool=None, stream=None):
    # [FUNCTION DOC] HELPER: Warms up the function on a side stream with the inputs, then
    # captures it in a CUDA graph (optionally in the given pool and stream) and returns the
    # graph with the static outputs. Raises SkipTest when graphs are unsupported.
    if not TEST_CUDA_GRAPH:
        raise unittest.SkipTest("cuda graph test is skipped")

    torch.cuda.synchronize()
    if stream is None:
        stream = torch.cuda.Stream()

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        static_outputs = fn(*inputs)

    return graph, static_outputs


def int8_cuda(size):
    # [FUNCTION DOC] HELPER: Returns a CUDA uint8 tensor of ones with the given number
    # of elements.
    return torch.ones([size], device="cuda", dtype=torch.uint8)


def live_blocks(pool_id):
    # [FUNCTION DOC] HELPER: Counts blocks in the active allocated state across all segments of
    # the given graph pool and returns the count.
    blocks = 0
    seg = get_cudagraph_segments(pool_id)
    for segment in get_cudagraph_segments(pool_id):
        for block in segment["blocks"]:
            blocks += block["state"] == "active_allocated"
    return blocks


def tensor_metadata(x):
    # [FUNCTION DOC] HELPER: Returns a mapping describing a tensor for later reconstruction:
    # storage byte size and data pointer, shape, strides, dtype, device and storage offset.
    return {
        "nbytes": x.untyped_storage().nbytes(),
        "data_ptr": x.untyped_storage().data_ptr(),
        "size": x.shape,
        "stride": x.stride(),
        "dtype": x.dtype,
        "device": x.device,
        "storage_offset": x.storage_offset(),
    }


def reconstruct_from_tensor_metadata(metadata):
    # [FUNCTION DOC] HELPER: Rebuilds a tensor from such a metadata mapping by constructing a
    # storage wrapper around the recorded data pointer, creating an empty tensor and pointing it
    # to that storage with the recorded offset, shape and strides; returns the tensor.
    s = torch._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    t = torch.empty([0], device=metadata["device"], dtype=metadata["dtype"])
    t.set_(
        source=s,
        storage_offset=metadata["storage_offset"],
        size=metadata["size"],
        stride=metadata["stride"],
    )
    return t


@unittest.skipIf(
    not TEST_CUDA or TEST_CUDAMALLOCASYNC,
    "CUDA required, not supported with CUDAMallocAsync",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestBlockStateAbsorption(TestCase):
    # [CLASS DOC] Tests for checkpointing and restoring graph private pool state in the
    # caching allocator. Skipped without CUDA or with cudaMallocAsync; wrapped for strict
    # dynamo mode. setUp records the number of graph segments and tearDown cleans up and
    # asserts the number is unchanged. Helper methods compare block and segment states and run
    # checkpoint round trips.
    @property
    def expandable_segments(self):
        # [FUNCTION DOC] HELPER: Property returning the suite expandable-segments flag.
        return EXPANDABLE_SEGMENTS

    def checkCheckpointedBlock(self, before_block, after_block):
        # [FUNCTION DOC] HELPER: Asserts the size and state fields of two blocks are equal.
        for field in ("size", "state"):
            self.assertEqual(before_block[field], after_block[field])

    def checkCheckpointedState(self, before_segments, after_segments):
        # [FUNCTION DOC] HELPER: Checks every segment captured before exists in the after list
        # (matched by address) with equal device, total, allocated and active sizes, segment
        # type and pool id, the same number of blocks, and equal blocks. The after list may have
        # extra segments.
        # after may contain additional segments, but all of the segments in before
        # should be exactly equivalent to after
        after_ptr_to_segment = {
            segment["address"]: segment for segment in after_segments
        }

        for before_segment in before_segments:
            self.assertTrue(before_segment["address"] in after_ptr_to_segment)
            after_segment = after_ptr_to_segment[before_segment["address"]]

            for field in (
                "device",
                "total_size",
                "allocated_size",
                "active_size",
                "segment_type",
                "segment_pool_id",
            ):
                self.assertEqual(before_segment[field], after_segment[field])

            self.assertEqual(
                len(before_segment["blocks"]), len(after_segment["blocks"])
            )
            for before_block, after_block in zip(
                before_segment["blocks"], after_segment["blocks"]
            ):
                self.checkCheckpointedBlock(before_block, after_block)

    @staticmethod
    def setCheckpointPoolState(
        device, state, stale_storages_ptr, storages_deleters=None
    ):
        # [FUNCTION DOC] HELPER: Static method that converts the stale tensors and deleter
        # tensors to their underlying storage handles (deleters default to an empty list) and
        # calls the internal function that sets the checkpointed pool state for the device.
        stale_storages_ptr = [t.untyped_storage()._cdata for t in stale_storages_ptr]
        storages_deleters = (
            []
            if not storages_deleters
            else [t.untyped_storage()._cdata for t in storages_deleters]
        )
        torch._C._cuda_setCheckpointPoolState(
            device, state, stale_storages_ptr, storages_deleters
        )

    def checkFunction(self, fn, inputs, pool=None):
        # [FUNCTION DOC] HELPER: Captures the function in a graph (optionally in a pool),
        # records segments of the graph pool, gets the checkpoint state, sets that state back
        # with no stale or deleter storages, and checks the segments are unchanged.
        graph, outputs = cudagraphify(fn, inputs, pool=pool)

        pool_id = graph.pool()
        device = outputs[0].device.index

        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        state = torch._C._cuda_getCheckpointState(device, pool_id)
        self.setCheckpointPoolState(device, state, [], [])

        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and stores how many graph segments
        # exist now.
        super().setUp()
        self.segment_length = len(get_all_cudagraph_segments())

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Synchronizes, collects garbage and empties the cache,
        # asserts the number of graph segments equals the count recorded at setUp, then calls
        # the base tearDown.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self.assertEqual(len(get_all_cudagraph_segments()), self.segment_length)

        super().tearDown()

    def test_simple(self):
        # [FUNCTION DOC] Round-trips checkpoint state for a graph that makes a large tensor and
        # several small ones of mixed lifetimes.
        def foo():
            # [FUNCTION DOC] HELPER: Creates a large uint8 tensor doubled with an
            # addition, several one-MiB intermediates summed and a final tensor, returning
            # three live tensors.
            x = torch.zeros([SMALL_SIZE * 8], device="cuda", dtype=torch.uint8)
            x = x + x
            x1 = int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE)
            y = int8_cuda(SMALL_SIZE) + x1
            z = int8_cuda(SMALL_SIZE)
            return x, y, z

        self.checkFunction(foo, [])

    def test_allocated_in_middle_of_segment(self):
        # [FUNCTION DOC] Round-trips checkpoint state for a graph where a live buffer sits in
        # the middle of a segment of eleven minimum-size buffers.
        def foo():
            # [FUNCTION DOC] HELPER: Creates eleven minimum-size buffers and returns the sixth
            # after adding 2.
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5].add_(2)

        self.checkFunction(foo, [])

    def test_multiple_middle_allocations(self):
        # [FUNCTION DOC] Round-trips checkpoint state when two non-adjacent middle buffers
        # stay alive.
        def foo():
            # [FUNCTION DOC] HELPER: Creates eleven minimum-size buffers and returns the sixth
            # and ninth.
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5], small_buffers[8]

        self.checkFunction(foo, [])

    def test_middle_allocations_contiguous(self):
        # [FUNCTION DOC] Round-trips checkpoint state when two adjacent middle buffers
        # stay alive.
        def foo():
            # [FUNCTION DOC] HELPER: Creates eleven minimum-size buffers and returns the sixth
            # and seventh.
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5], small_buffers[6]

        self.checkFunction(foo, [])

    def test_additional_free_following_checkpoint(self):
        # [FUNCTION DOC] Captures a first graph, takes its pool checkpoint, captures a
        # second graph in the same pool, restores the checkpoint treating the second graph
        # outputs as stale, deletes them and checks the first graph segments match the
        # pre-checkpoint snapshot.
        def foo():
            # [FUNCTION DOC] HELPER: Returns a tuple with one minimum-size buffer.
            return (int8_cuda(MIN_BLOCK_SIZE),)

        def foo2():
            # [FUNCTION DOC] HELPER: Returns a tuple with one minimum-size buffer.
            return (int8_cuda(MIN_BLOCK_SIZE),)

        graph, outputs = cudagraphify(foo, [])
        pool_id = graph.pool()

        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())

        self.setCheckpointPoolState(outputs[0].device.index, state, outputs2, [])

        del outputs2

        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    # TODO: re-enable
    # def test_additional_free_error(self):
    #     def foo():
    #         return int8_cuda(MIN_BLOCK_SIZE),

    #     def foo2():
    #         return int8_cuda(MIN_BLOCK_SIZE),

    #     graph, outputs = cudagraphify(foo, [])
    #     pool_id = graph.pool()

    #     segments_before_checkpoint = get_cudagraph_segments(pool_id)

    #     state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

    # graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())
    # with self.assertRaisesRegex(Exception, "being manually freed must be passed"):
    #     self.setCheckpointPoolState(outputs[0].device.index, state, [], [])

    def test_tensor_dies_after_checkpoint(self):
        # [FUNCTION DOC] Captures a graph returning two buffers, takes a checkpoint, remembers
        # the data pointers and deletes the outputs, restores the checkpoint with nothing stale
        # (which revives both blocks), then checks two live blocks; raw-deleting the pointers
        # one at a time reduces live blocks to 1 and then 0.
        def foo():
            # [FUNCTION DOC] HELPER: Returns two minimum-size buffers.
            return int8_cuda(MIN_BLOCK_SIZE), int8_cuda(MIN_BLOCK_SIZE)

        graph, outputs = cudagraphify(foo, [])
        pool_id = graph.pool()
        device = outputs[0].device.index

        segments_before_checkpoint = get_cudagraph_segments(pool_id)
        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        output_data_ptrs = [output.data_ptr() for output in outputs]

        del outputs

        self.setCheckpointPoolState(device, state, [], [])

        self.assertEqual(live_blocks(pool_id), 2)
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[0])
        self.assertEqual(live_blocks(pool_id), 1)
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[1])
        self.assertEqual(live_blocks(pool_id), 0)

    def test_assigning_back_deleter_fns_to_tensor(self):
        # [FUNCTION DOC] Captures a graph with three outputs (two small buffer and one large
        # buffer sized) from an input, replays it, takes a checkpoint and tensor metadata,
        # deletes the outputs (live blocks drop from 3 to 0), reconstructs tensors from the
        # metadata and checks their values; then adds to the input and replays again to confirm
        # the graph still works through the reconstructed tensors.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns three tensors of small, small and large buffer size
            # each added to the input.
            return (
                int8_cuda(SMALL_BUFFER) + x,
                int8_cuda(SMALL_BUFFER) + x,
                int8_cuda(LARGE_BUFFER) + x,
            )

        inp = torch.tensor([1], device="cuda")
        graph, outputs = cudagraphify(foo, [inp])
        pool_id = graph.pool()
        graph.replay()

        device = outputs[0].device.index

        for i in range(len(outputs)):
            self.assertEqual(outputs[i].mean(dtype=torch.float), 2)

        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        output_ptrs = [output.untyped_storage().data_ptr() for output in outputs]
        ten_metadata = [tensor_metadata(t) for t in outputs]

        self.assertEqual(live_blocks(pool_id), 3)

        del outputs

        self.assertEqual(live_blocks(pool_id), 0)

        reconstructed_tensors = [
            reconstruct_from_tensor_metadata(metadata) for metadata in ten_metadata
        ]

        for i in range(len(reconstructed_tensors)):
            self.assertEqual(reconstructed_tensors[i].mean(dtype=torch.float), 2)

        inp.add_(1)
        graph.replay()

        for i in range(len(reconstructed_tensors)):
            self.assertEqual(reconstructed_tensors[i].mean(dtype=torch.float), 3)

        self.setCheckpointPoolState(
            device, state, [], [reconstructed_tensors[0], reconstructed_tensors[1]]
        )

        self.assertEqual(live_blocks(pool_id), 3)

        reconstructed_tensors[0] = None
        self.assertEqual(live_blocks(pool_id), 2)

        reconstructed_tensors[1] = None
        self.assertEqual(live_blocks(pool_id), 1)

        # should not change, we did not pass it in to swap data ptrs
        reconstructed_tensors[2] = None
        self.assertEqual(live_blocks(pool_id), 1)

        torch._C._cuda_cudaCachingAllocator_raw_delete(output_ptrs[2])

        self.assertEqual(live_blocks(pool_id), 0)

    @skipIfNoTorchVision
    def test_resnet(self):
        # [FUNCTION DOC] Builds a torchvision resnet50 in eval mode on CUDA and runs the
        # checkpoint round trip check with a random 255 by 255 image input. Skipped
        # without torchvision.
        import torchvision

        m = torchvision.models.resnet50()
        m.eval()
        m = m.cuda()

        inp = torch.rand([1, 3, 255, 255], device="cuda")
        self.checkFunction(m, [inp])

    def test_check_pool_live_allocations(self):
        # [FUNCTION DOC] Captures a graph in a graph pool handle returning a small tensor, then
        # checks the pool live allocation query: it is true when given the output pointer, false
        # when given the pointer plus an extra zero or an empty set, and true for an empty set
        # after the output is deleted.
        def foo():
            # [FUNCTION DOC] HELPER: Returns a CUDA tensor of four ones.
            return torch.ones([4], device="cuda")

        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)

        index = outputs[0].device.index

        def check(live_dps):
            # [FUNCTION DOC] HELPER: Asks the allocator whether exactly the given set of data
            # pointers are the live allocations of the pool on the device.
            return torch._C._cuda_checkPoolLiveAllocations(index, pool, live_dps)

        self.assertTrue(check({outputs[0].data_ptr()}))

        self.assertFalse(check({outputs[0].data_ptr(), 0}))
        self.assertFalse(check(set()))

        del outputs
        self.assertTrue(check(set()))

    def test_allocate_in_thread_to_pool(self):
        # [FUNCTION DOC] Captures a graph into a pool, then runs two threads at once: one
        # allocates tensors routed into that pool through a context manager (the thread-to-pool
        # routing) and also a few outside it, the other allocates without the pool. Checks the
        # pool has 1 segment before, then 2 (expandable) or 4 segments after, and 0 after the
        # graph is deleted and caches cleared.
        def foo():
            # [FUNCTION DOC] HELPER: Returns a CUDA tensor of four random values.
            return torch.rand([4], device="cuda")

        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)
        device = outputs[0].device.index
        del outputs

        @contextlib.contextmanager
        def _use_cuda_memory_pool_manager(device, mem_pool):
            # [FUNCTION DOC] HELPER: Context manager that synchronizes, creates a stream, and
            # inside that stream and device begins routing the current thread allocations to the
            # given pool; on exit it ends the routing and releases the pool, then makes the
            # original stream wait for the stream.
            """
            Context manager to use cuda graph pool for new allocations. If you use this manager
            all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
            existing_graph should already have been used in a capture, and the mem_pool must already exist.
            """
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream), torch.device(device):
                torch._C._cuda_beginAllocateCurrentThreadToPool(device, mem_pool)
                try:
                    yield
                finally:
                    torch._C._cuda_endAllocateToPool(device, mem_pool)
                    torch._C._cuda_releasePool(device, mem_pool)

            torch.cuda.current_stream().wait_stream(stream)

        segments = get_cudagraph_segments(pool)
        self.assertEqual(len(get_cudagraph_segments(pool)), 1)

        def use_pool():
            # [FUNCTION DOC] HELPER: Thread body: inside the pool routing context, runs the
            # three-allocation helper ten times, then once more outside it.
            def alloc_three():
                # [FUNCTION DOC] HELPER: Allocates two large uint8 buffers and their sum.
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                c = a + b

            with _use_cuda_memory_pool_manager(device, pool):
                # three allocations
                for _ in range(10):
                    alloc_three()

            # three more allocations not in pool
            alloc_three()

        def no_pool():
            # [FUNCTION DOC] HELPER: Thread body: ten times allocates two large uint8 buffers
            # and deletes them.
            # two allocations
            for _ in range(10):
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                del a, b

        graph_thread = threading.Thread(target=use_pool)
        no_graph_thread = threading.Thread(target=no_pool)
        graph_thread.start()
        no_graph_thread.start()

        graph_thread.join()
        no_graph_thread.join()

        self.assertEqual(
            len(get_cudagraph_segments(pool)), 2 if self.expandable_segments else 4
        )

        del graph

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self.assertEqual(len(get_cudagraph_segments(pool)), 0)

    @unittest.skipIf(not EXPANDABLE_SEGMENTS, "requires expandable_segments")
    def test_expandable_segment_checkpoint_growth(self):
        # [FUNCTION DOC] Tests restoring a pool checkpoint after an expandable segment has grown
        # (needs expandable segments). Captures an empty graph to get the original state, then a
        # 2 MB graph (checked via blocks: 2 MB active plus 18 MB inactive), restores the
        # original state, captures a 32 MB graph (32 MB active plus 8 MB inactive), restores
        # again, replays the first graph, reconstructs its tensors and restores the small state
        # with them as deleters. The mapped blocks must be a 2 MB active block and a merged 38
        # MB inactive block.
        # Test restoring checkpoint when expandable segment has grown.
        # Captures graph1 (small), checkpoints, then captures graph2 (larger)
        # which grows the segment. Restoring to the smaller checkpoint should
        # handle the extra mapped memory correctly.
        pool_id = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        device = torch.cuda.current_device()

        def get_large_mapped_blocks():
            # [FUNCTION DOC] HELPER: Returns a list of size and state pairs for blocks in
            # segments of type large from the memory snapshot (which lists mapped blocks only).
            # memory_snapshot only includes mapped blocks
            snapshot_segments = torch.cuda.memory_snapshot()
            res = []
            for segment in snapshot_segments:
                if segment["segment_type"] != "large":
                    continue
                blocks = segment["blocks"]
                for block in blocks:
                    res.append((block["size"], block["state"]))
            return res

        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            g.capture_begin(pool=pool_id)
            g.capture_end()
        original_state = torch._C._cuda_getCheckpointState(device, pool_id)

        # 2MB allocation
        def small_fn():
            # [FUNCTION DOC] HELPER: Returns a tuple holding one 2 MiB int8 CUDA tensor.
            return (torch.empty(2 * 1024 * 1024, device="cuda", dtype=torch.int8),)

        graph1, out1 = cudagraphify(small_fn, [], pool=pool_id, stream=stream)

        large_mapped_blocks = get_large_mapped_blocks()
        # small_fn allocates 2 MB, which leads to a 20 MB segment in large blocks. This becomes a 2 MB (or 2097152 bytes)
        # active allocated block and a 18 MB (or 18874368 bytes) inactive block.
        self.assertEqual(
            large_mapped_blocks, [(2097152, "active_allocated"), (18874368, "inactive")]
        )

        small_state = torch._C._cuda_getCheckpointState(device, pool_id)
        out1_metadata = [tensor_metadata(t) for t in out1]
        del out1

        self.setCheckpointPoolState(device, original_state, [], [])

        # 32MB allocation
        def large_fn():
            # [FUNCTION DOC] HELPER: Returns a tuple holding one 32 MiB int8 CUDA tensor.
            return (torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.int8),)

        graph2, out2 = cudagraphify(large_fn, [], pool=pool_id, stream=stream)

        large_mapped_blocks = get_large_mapped_blocks()
        # large_fn allocates 32 MB, which leads to a 40 MB segment in large blocks. This becomes a 32 MB (or 33554432 bytes)
        # active allocated block and a 8 MB (or 8388608 bytes) inactive block.
        self.assertEqual(
            large_mapped_blocks, [(33554432, "active_allocated"), (8388608, "inactive")]
        )

        self.setCheckpointPoolState(device, original_state, out2, [])

        graph1.replay()
        reconstructed = [reconstruct_from_tensor_metadata(m) for m in out1_metadata]
        out1_storage = [o.untyped_storage()._cdata for o in reconstructed]
        torch._C._cuda_setCheckpointPoolState(device, small_state, [], out1_storage)

        large_mapped_blocks = get_large_mapped_blocks()
        # private memory pool never decreases size so we still have 40 MB segment. When we set checkpoint pool state for
        # small_state, we allocate a 2 MB (or 2097152 bytes) active block and a 18 MB (or 18874368 bytes) inactive block.
        # Since the total segment size is 40 MB, we still have a 20 MB inactive block. These two inactive blocks
        # automatically merge into a single 38 MB (or 39845888 bytes) inactive block. As a result, we have a 2 MB active
        # block and a 38 MB inactive block in the memory snapshot.
        self.assertEqual(
            large_mapped_blocks, [(2097152, "active_allocated"), (39845888, "inactive")]
        )

    def test_no_triton_on_import(self):
        # [FUNCTION DOC] Runs a child Python process that imports torch, uses a CUDA tensor and
        # prints whether triton is in the loaded modules; asserts the output is False, i.e.
        # using the GPU does not import triton.
        """Test that Triton is not imported on first GPU use"""
        script = "import sys; import torch; torch.rand(2, device='cuda'); print('triton' in sys.modules)"

        rc = (
            subprocess.check_output(
                [sys.executable, "-c", script],
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
            .strip()
            .decode("ascii")
        )
        self.assertEqual(rc, "False", "Triton was imported when importing torch!")


@contextlib.contextmanager
def caching_host_allocator_use_host_register(use_cuda_host_register: bool):
    # [FUNCTION DOC] HELPER: Context manager. If the flag is true, enables the pinned allocator
    # host register option with 8 register threads before yielding, and in all cases, if the
    # flag is true, switches the option off again after the block.
    if use_cuda_host_register:
        torch.cuda.memory._set_allocator_settings(
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
        )
    try:
        yield
    finally:
        if use_cuda_host_register:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )


@contextlib.contextmanager
def caching_host_allocator_use_background_threads(use_background_threads: bool):
    # [FUNCTION DOC] HELPER: Context manager. If the flag is true, enables the pinned allocator
    # background threads option for the duration of the block and disables it afterward (even
    # on error).
    if use_background_threads:
        torch.cuda.memory._set_allocator_settings("pinned_use_background_threads:True")
    try:
        yield
    finally:
        if use_background_threads:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_background_threads:False"
            )


@contextlib.contextmanager
def caching_host_allocator_max_round_threshold_and_max_cached_size(
    max_round_threshold_mb: int | None = None, max_cached_size_mb: int | None = None
):
    # [FUNCTION DOC] HELPER: Context manager that sets the pinned allocator maximum round
    # threshold and/or maximum cached size (in MB, only those given) through the accelerator
    # allocator settings, reading current values from the memory snapshot first, and restores
    # both previous values afterward.
    allocator_settings = torch._C._cuda_memorySnapshot(None)["allocator_settings"]
    cur_max_round_threshold_mb = (
        allocator_settings["max_round_threshold"] // 1024 // 1024
    )
    cur_max_cached_size_mb = allocator_settings["max_cached_size"] // 1024 // 1024

    parts = []
    if max_round_threshold_mb is not None:
        parts.append(f"pinned_max_round_threshold_mb:{max_round_threshold_mb}")
    if max_cached_size_mb is not None:
        parts.append(f"pinned_max_cached_size_mb:{max_cached_size_mb}")
    torch._C._accelerator_setAllocatorSettings(",".join(parts))
    try:
        yield
    finally:
        torch._C._accelerator_setAllocatorSettings(
            f"pinned_max_round_threshold_mb:{cur_max_round_threshold_mb},pinned_max_cached_size_mb:{cur_max_cached_size_mb}"
        )


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestCachingHostAllocatorConfig(TestCase):
    # [CLASS DOC] Tests for pinned (host) caching allocator size-threshold settings. Skipped
    # without CUDA. setUp and tearDown collect garbage (setUp) and empty the host cache so stats
    # start clean.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls base setUp, collects garbage and empties the host
        # allocator cache.
        super().setUp()
        gc.collect()
        torch._C._host_emptyCache()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Empties the host allocator cache, then calls base tearDown.
        torch._C._host_emptyCache()
        super().tearDown()

    def test_max_round_threshold(self):
        # [FUNCTION DOC] With the round threshold at 128 MB, allocates a 129 MiB pinned tensor
        # and checks allocated bytes are at least the request but below 256 MiB (not rounded up
        # to the next power of two).
        size_bytes = 129 * 1024 * 1024

        with caching_host_allocator_max_round_threshold_and_max_cached_size(128, None):
            t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            stats = torch.cuda.host_memory_stats()

            allocated = stats["allocated_bytes.current"]
            self.assertLess(allocated, 256 * 1024 * 1024)
            self.assertGreaterEqual(allocated, size_bytes)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/182111"
    )
    def test_max_cached_size(self):
        # [FUNCTION DOC] With maximum cached size 32 MB, allocates and frees a 64 MiB pinned
        # tensor and asserts the allocation count is 0 afterward (block not cached). Skipped on
        # Linux and slow runs (tracked issue).
        size_bytes = 64 * 1024 * 1024

        with caching_host_allocator_max_round_threshold_and_max_cached_size(None, 32):
            t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            del t
            gc.collect()

            stats = torch.cuda.host_memory_stats()
            self.assertEqual(stats["allocations.current"], 0)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/182087"
    )
    def test_both_thresholds(self):
        # [FUNCTION DOC] With thresholds 128 MB (round) and 256 MB (cached), allocates a 150
        # MiB pinned tensor and checks allocated bytes stay below 256 MiB, then after freeing
        # the allocation count remains 1 (block cached). Skipped on Linux and slow runs
        # (tracked issue).
        size_bytes = 150 * 1024 * 1024

        with caching_host_allocator_max_round_threshold_and_max_cached_size(128, 256):
            t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            stats = torch.cuda.host_memory_stats()

            self.assertLess(stats["allocated_bytes.current"], 256 * 1024 * 1024)
            del t
            gc.collect()

            stats = torch.cuda.host_memory_stats()
            self.assertEqual(stats["allocations.current"], 1)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/182261"
    )
    def test_round_threshold_larger_than_cached_size(self):
        # [FUNCTION DOC] With round threshold 256 MB above the cache limit of 64 MB, a 100 MiB
        # pinned tensor is neither rounded up (allocated bytes within 128 MiB) nor cached after
        # being freed (allocation count 0). Skipped on Linux and slow runs (tracked issue).
        # When round_threshold > cached_size, allocations above the cache
        # limit should be neither rounded up nor cached.
        size_bytes = 100 * 1024 * 1024

        with caching_host_allocator_max_round_threshold_and_max_cached_size(256, 64):
            t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            stats = torch.cuda.host_memory_stats()

            # 100 MB is within the 256 MB round threshold but above the
            # 64 MB cache limit — should NOT be rounded to 128 MB.
            allocated = stats["allocated_bytes.current"]
            self.assertGreaterEqual(allocated, size_bytes)
            self.assertLess(allocated, 128 * 1024 * 1024)

            del t
            gc.collect()

            # Block should have been freed, not cached.
            stats = torch.cuda.host_memory_stats()
            self.assertEqual(stats["allocations.current"], 0)

    def test_unrounded_bucket_no_undersized_reuse(self):
        # [FUNCTION DOC] Regression check: with round threshold 128 MB and cache limit 1024 MB,
        # frees a 150 MiB pinned tensor then allocates a 200 MiB one; the new tensor must not
        # reuse the smaller cached block's address (both fall in the same size bucket).
        # Unrounded requests (above round_threshold) that are still cached
        # (below max_cached_size) share a Log2_64_Ceil bucket despite differing
        # sizes. A later, larger request in the same bucket must not be handed
        # back a smaller cached block, which would overflow the buffer.
        MB = 1024 * 1024
        # 150 MB and 200 MB both map to Log2_64_Ceil bucket 28.
        small = 150 * MB
        large = 200 * MB

        with caching_host_allocator_max_round_threshold_and_max_cached_size(128, 1024):
            a = torch.empty(small, dtype=torch.uint8, pin_memory=True)
            small_ptr = a.data_ptr()
            del a
            gc.collect()

            b = torch.empty(large, dtype=torch.uint8, pin_memory=True)
            # Must not reuse the smaller cached block.
            self.assertNotEqual(b.data_ptr(), small_ptr)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestCachingHostAllocatorCudaGraph(TestCase):
    # [CLASS DOC] Tests how pinned host memory allocated during CUDA graph capture is pooled and
    # recycled. Skipped without CUDA. Tests are parametrized by whether the host-register option
    # is used (and background threads in the two-graph test).
    # As soon as pinned host memory allocated by a private pool is
    # used (by copy_ in this case) during stream capture, it can never
    # be recycled.
    @parametrize("use_cuda_host_register", [True, False])
    def test_pin_memory_no_use(self, use_cuda_host_register):
        # [FUNCTION DOC] Parametrized by host register use. Within a graph capture allocates
        # two pinned tensors and asserts they have different addresses (a live block cannot
        # be reused).
        # A pinned host memory block cannot be reused if it is not deleted
        with caching_host_allocator_use_host_register(use_cuda_host_register):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                data = torch.empty(8, pin_memory=True)
                data2 = torch.empty(8, pin_memory=True)
            if data.data_ptr() == data2.data_ptr():
                raise AssertionError("data and data2 should have different data_ptr")
            del data2

    @parametrize("use_cuda_host_register", [True, False])
    def test_pin_memory_no_use2(self, use_cuda_host_register):
        # [FUNCTION DOC] Parametrized by host register use. Within a capture, allocates a pinned
        # tensor, deletes it and allocates another; asserts the second reuses the same address
        # since the block was never used by a copy.
        # A pinned host memory block can be reused if it is deleted
        # and has never been used by copy_
        with caching_host_allocator_use_host_register(use_cuda_host_register):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                data = torch.randn(8).pin_memory()
                data_ptr = data.data_ptr()
                del data
                data2 = torch.randn(8).pin_memory()
                if data2.data_ptr() != data_ptr:
                    raise AssertionError(
                        "data2 should have same data_ptr as deleted data"
                    )

    @parametrize("use_cuda_host_register", [True, False])
    def test_pin_memory_use(self, use_cuda_host_register):
        # [FUNCTION DOC] Parametrized by host register use. Within a capture, pins a tensor,
        # copies it to a GPU tensor non-blocking, deletes it and pins another; asserts the new
        # address differs from the old one because a block used by a copy cannot be recycled.
        # A pinned host memory block cannot be reused if it has been used by copy_
        with caching_host_allocator_use_host_register(use_cuda_host_register):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                data = torch.randn(8).pin_memory()
                data_gpu = torch.randn(8, device="cuda")
                data_gpu.copy_(data, non_blocking=True)
                old_data_ptr = data.data_ptr()
                del data
                data2 = torch.randn(8).pin_memory()
            if data2.data_ptr() == old_data_ptr:
                raise AssertionError(
                    "data2 should have different data_ptr than old_data_ptr"
                )

    @parametrize("use_cuda_host_register", [True, False])
    @parametrize("use_background_threads", [True, False])
    @parametrize(
        "use_memory, delete_memory",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    @skip_background_threads_on_windows
    def test_two_graphs(
        self, use_background_threads, use_cuda_host_register, use_memory, delete_memory
    ):
        # [FUNCTION DOC] Parametrized over host register use, background threads, whether memory
        # is used by a copy and whether it is deleted. Captures two graphs sharing one pool,
        # each allocating a pinned tensor; asserts the second graph reuses the first address
        # only when the memory was deleted and never used, otherwise it must differ. Skipped for
        # background threads on Windows SM89+.
        # Pinned host memory belongs to private pools, not to
        # graphs. Therefore, using two graphs that share a pool
        # instead of one graph does not change any of our invariants.
        with (
            caching_host_allocator_use_background_threads(use_background_threads),
            caching_host_allocator_use_host_register(use_cuda_host_register),
        ):
            shared_pool = torch.cuda.graph_pool_handle()
            graph1 = torch.cuda.CUDAGraph()
            graph2 = torch.cuda.CUDAGraph()

            with torch.cuda.graph(
                graph1, pool=shared_pool, capture_error_mode="thread_local"
            ):
                data = torch.randn(8).pin_memory()
                if use_memory:
                    data_gpu = torch.randn(8, device="cuda")
                    data_gpu.copy_(data, non_blocking=True)

                old_data_ptr = data.data_ptr()
                if delete_memory:
                    del data

            with torch.cuda.graph(
                graph2, pool=shared_pool, capture_error_mode="thread_local"
            ):
                data2 = torch.randn(8).pin_memory()
                if use_memory:
                    data_gpu = torch.randn(8, device="cuda")
                    data_gpu.copy_(data2, non_blocking=True)

                new_data_ptr = data2.data_ptr()
                if delete_memory:
                    del data2

            if delete_memory and not use_memory:
                if new_data_ptr != old_data_ptr:
                    raise AssertionError("new_data_ptr should equal old_data_ptr")
            else:
                if new_data_ptr == old_data_ptr:
                    raise AssertionError("new_data_ptr should differ from old_data_ptr")

    def test_unpinned_memory_use(self):
        # [FUNCTION DOC] Asserts that, during graph capture, a non-blocking copy from an
        # unpinned CPU tensor into a CUDA tensor raises RuntimeError (cannot copy between CPU
        # and CUDA tensors).
        # Copying between CPU and CUDA tensors during graph capture
        # with unpinned CPU memory should raise an error.
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot copy between CPU and CUDA tensors during CUDA graph capture",
        ):
            with torch.cuda.graph(graph):
                data = torch.empty(8)
                data_gpu = torch.randn(8, device="cuda")
                data_gpu.copy_(data, non_blocking=True)

    def test_unpinned_memory_use_device_to_host(self):
        # [FUNCTION DOC] Asserts that, during graph capture, a non-blocking copy from a CUDA
        # tensor into an unpinned CPU tensor raises the same RuntimeError.
        # Copying from CUDA to unpinned CPU during graph capture should also error.
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot copy between CPU and CUDA tensors during CUDA graph capture",
        ):
            with torch.cuda.graph(graph):
                data_gpu = torch.randn(8, device="cuda")
                data = torch.empty(8)
                data.copy_(data_gpu, non_blocking=True)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestMemPool(TestCase):
    # [CLASS DOC] Tests for torch.cuda.MemPool and use_mem_pool: pool ids, snapshots, custom
    # pluggable allocators (compiled dummy allocator), use_on_oom fallback, no_split pools and
    # multithreading. Skipped without CUDA. tearDown checks the expandable segments setting is
    # unchanged. Helpers set up and tear down a per-process memory limit.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls base tearDown, then runs the allocator-settings
        # regression check.
        super().tearDown()
        _check_allocator_settings_on_tear_down(self)

    def _setup_mempool_limited_memory_test(self, additional_allowed_memory_in_mb):
        # [FUNCTION DOC] HELPER: Saves the current per-process memory fraction, empties the
        # cache and sets a fraction so the device may use only the given extra number of MB
        # beyond what is already reserved. Returns the device (cuda:0) and the int8 dtype.
        device = torch.device("cuda:0")

        self.init_fraction = torch.cuda.get_per_process_memory_fraction()
        torch.cuda.memory.empty_cache()
        mb = 1024 * 1024
        _, all_memory = torch.cuda.memory.mem_get_info(device)
        pre_reserved = torch.cuda.memory_reserved(device)
        total_allowed = additional_allowed_memory_in_mb * mb + pre_reserved
        fraction_allowed = total_allowed / all_memory
        torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed, device)

        dtype = torch.int8
        return device, dtype

    def _teardown_mempool_limited_memory_test(self):
        # [FUNCTION DOC] HELPER: Empties the cache and restores the memory fraction saved by the
        # setup helper.
        torch.cuda.memory.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(self.init_fraction)

    def test_mempool_id(self):
        # [FUNCTION DOC] Compares pool ids from a graph pool handle and from MemPool: both
        # ids begin with zero in the first element for a user pool, and their second
        # elements differ.
        pool1 = torch.cuda.graph_pool_handle()
        pool2 = torch.cuda.MemPool().id

        # first value of id in a user created pool is always zero
        self.assertEqual(pool1[0] == 0, pool2[0] == 0)

        # each call to torch.cuda.graph_pool_handle() or torch.cuda.MemPool()
        # increments the id
        self.assertTrue(abs(pool2[1] - pool1[1]) > 0)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_pool_id_in_snapshot(self):
        # [FUNCTION DOC] Records history, allocates within a MemPool and checks the snapshot:
        # some segment has the pool id and some alloc trace entry carries the pool id. Not
        # for cudaMallocAsync.
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("all")

            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool):
                x = torch.rand(64, device="cuda")

            ss = torch.cuda.memory._snapshot()

            # segment_pool_id should match the MemPool id
            found_segment = False
            for seg in ss["segments"]:
                if seg["segment_pool_id"] == pool.id:
                    found_segment = True
                    break
            self.assertTrue(found_segment)

            # trace entries for this allocation should carry pool_id
            found_trace = False
            for trace in ss["device_traces"]:
                for te in trace:
                    if "pool_id" not in te:
                        continue
                    if te["pool_id"] == pool.id and te["action"] == "alloc":
                        found_trace = True
                        break
            self.assertTrue(found_trace)

            del x
        finally:
            torch.cuda.memory._record_memory_history(None)

    def get_dummy_allocator(self, check_vars):
        # [FUNCTION DOC] HELPER: Compiles inline a small C++ extension implementing a pluggable
        # allocator (managed-memory allocate and free functions); when check_vars is true the
        # library also exports counters that record that allocate (123) and free (321) were
        # called. Returns the pluggable allocator object and the library path.
        dummy_allocator_source_vars = """
        #include <torch/extension.h>
        #include <ATen/cuda/Exceptions.h>
        #include <cuda_runtime_api.h>

        extern "C" {
            C10_EXPORT int called_dummy_alloc = 0;
            C10_EXPORT int called_dummy_free = 0;

            // Note that windows needs __declspec(dllexport): https://stackoverflow.com/a/24575865
            C10_EXPORT void* dummy_alloc(size_t size, int device, void* stream) {
            called_dummy_alloc = 123;
            void* ptr;
            C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
            return ptr;
            }

            C10_EXPORT void dummy_free(void* ptr, size_t size, int device, void* stream) {
            called_dummy_free = 321;
            C10_CUDA_CHECK(cudaFree(ptr));
            }
        }
        """
        dummy_allocator_source_no_vars = """
        #include <torch/extension.h>
        #include <ATen/cuda/Exceptions.h>
        #include <cuda_runtime_api.h>

        extern "C" {
          // Note that windows needs __declspec(dllexport): https://stackoverflow.com/a/24575865
          C10_EXPORT void* dummy_alloc(size_t size, int device, void* stream) {
            void* ptr;
            C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
            return ptr;
          }

          C10_EXPORT void dummy_free(void* ptr, size_t size, int device, void* stream) {
            C10_CUDA_CHECK(cudaFree(ptr));
          }
        }
        """

        from torch.utils.cpp_extension import load_inline

        dummy_allocator_libname = "dummy_allocator"
        dummy_allocator = load_inline(
            name=dummy_allocator_libname,
            cpp_sources=dummy_allocator_source_vars
            if check_vars
            else dummy_allocator_source_no_vars,
            is_python_module=False,
            keep_intermediates=False,
            verbose=True,
            with_cuda=True,
        )
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            dummy_allocator,
            "dummy_alloc",
            "dummy_free",
        )
        return allocator, dummy_allocator

    def test_mempool_empty_cache(self):
        # [FUNCTION DOC] Allocates tensors in the default pool and in a MemPool, deletes both
        # and the pool, and asserts the snapshot still has segments (empty_cache is not expected
        # to free pool memory still cached).
        torch.cuda.empty_cache()
        pool = torch.cuda.MemPool()
        x = torch.empty(1024, 1024, device="cuda")

        with torch.cuda.use_mem_pool(pool):
            y = torch.empty(1024, 1024, device="cuda")

        del y
        del x
        del pool
        segments = torch.cuda.memory._snapshot()["segments"]
        self.assertTrue(len(segments) > 0, "expected more than one segment")

    @serialTest()
    def test_mempool_empty_cache_inactive(self):
        # [FUNCTION DOC] Uses the dummy allocator: allocates a tensor from a pool created inside
        # a function that returns (the pool then dies), checks alloc counter 123 and free 0,
        # then after deleting the tensor and emptying the cache the free counter is 321.
        # Counters are reset. Serial.
        torch.cuda.empty_cache()
        allocator, dummy_allocator = self.get_dummy_allocator(check_vars=True)
        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")
        self.assertEqual(called_dummy_alloc.value, 0)
        self.assertEqual(called_dummy_free.value, 0)

        def f():
            # [FUNCTION DOC] HELPER: Creates a MemPool with the dummy allocator, allocates a
            # tensor within it and returns the tensor (the pool is destroyed on return).
            pool = torch.cuda.MemPool(allocator.allocator())

            # allocate memory with ncclMemAlloc
            with torch.cuda.use_mem_pool(pool):
                x = torch.arange(1024 * 1024 * 2, device="cuda")
            # Note: pool will be destroyed upon function return, but x, which
            # was allocated via the pool is still alive.
            return x

        x = f()
        self.assertEqual(called_dummy_alloc.value, 123)
        self.assertEqual(called_dummy_free.value, 0)

        del x
        torch.cuda.empty_cache()
        self.assertEqual(called_dummy_free.value, 321)

        # reset dummy allocator indicator variables because they are global
        called_dummy_free.value = 0
        called_dummy_alloc.value = 0

    @serialTest()
    def test_mempool_with_allocator(self):
        # [FUNCTION DOC] Uses the dummy allocator with a MemPool: use count is 1; inside
        # use_mem_pool it is 2 and back to 1 after; the custom alloc counter becomes 123;
        # snapshot segment counts go from 1 to 2 as three 1 MB tensors are allocated; deleting
        # tensors and pool triggers the custom free (counter 321). Counters are reset at the
        # end. Serial.
        pool = torch.cuda.MemPool()

        allocator, dummy_allocator = self.get_dummy_allocator(check_vars=True)

        pool = torch.cuda.MemPool(allocator.allocator())

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)

        # no allocations happened yet, so called_dummy_alloc and
        # called_dummy_free should be 0
        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")
        self.assertEqual(called_dummy_alloc.value, 0)
        self.assertEqual(called_dummy_free.value, 0)

        nelem_1mb = 1024 * 1024 // 4

        with torch.cuda.use_mem_pool(pool):
            out_0 = torch.randn(nelem_1mb, device="cuda")

            # pool's use count should be 2 at this point as use_mem_pool
            # holds a reference
            self.assertEqual(pool.use_count(), 2)

        # pool's use count should be back to 1 at this point as use_mem_pool
        # released its reference
        self.assertEqual(pool.use_count(), 1)

        # called_dummy_alloc should be 123 if dummy_alloc was used to allocate
        # out tensor
        self.assertEqual(called_dummy_alloc.value, 123)

        out_non_pool = torch.empty(nelem_1mb, device="cuda")

        with torch.cuda.use_mem_pool(pool):
            # pool should have 1 segment since we made a small allocation (1 MB)
            # above and so the CUDACachingAllocator packed it into a 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_1 = torch.randn(nelem_1mb, device="cuda")

            # pool should still have 1 segment since we made another small allocation
            # (1 MB) that got packed into the existing 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_2 = torch.randn(nelem_1mb, device="cuda")

            # pool now should have 2 segments since the CUDACachingAllocator had
            # to make a new 2 MB buffer to accommodate out_2
            self.assertEqual(len(pool.snapshot()), 2)

        self.assertEqual(len(pool.snapshot()), 2)

        del out_0, out_1, out_2

        # pool's destructor calls emptyCache()
        del pool

        # called_dummy_free should be 321 if dummy_free was used to deallocate
        # out tensor
        self.assertEqual(called_dummy_free.value, 321)

        # reset dummy allocator indicator variables because they are global
        called_dummy_free.value = 0
        called_dummy_alloc.value = 0

    @serialTest()
    def test_tensor_delete_after_allocator_delete(self):
        # [FUNCTION DOC] Checks that a tensor allocated through a custom allocator can be
        # freed after the Python allocator and pool objects are deleted: free counter stays 0
        # until the tensor is deleted and the cache is emptied, then it is 321. Counters
        # reset. Serial.
        allocator, dummy_allocator = self.get_dummy_allocator(check_vars=True)
        pool = torch.cuda.MemPool(allocator.allocator())

        with torch.cuda.use_mem_pool(pool):
            data = torch.empty(4, device="cuda")

        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")

        self.assertEqual(called_dummy_alloc.value, 123)
        self.assertEqual(called_dummy_free.value, 0)

        # Explicitly drop the allocator before releasing the
        # tensor. CUDACachingAllocator's PrivatePool instance should
        # keep the allocator live until all tensors die.

        # N.B.: Deleting the pool doesn't actually do anything, since
        # it doesn't own the allocator object. But we do it anyway
        # because it is the situation that users are likely to face.
        del pool
        del allocator

        self.assertEqual(called_dummy_free.value, 0)

        del data
        torch.cuda.empty_cache()

        self.assertEqual(called_dummy_free.value, 321)

        # reset dummy allocator indicator variables because they are global
        called_dummy_free.value = 0
        called_dummy_alloc.value = 0

    @serialTest()
    def test_mempool_limited_memory_with_allocator(self):
        # [FUNCTION DOC] Tests the use_on_oom fallback with a limit of 80 MB: allocates 40 MB in
        # each of two custom pools (one with fallback enabled), a further 40 MB allocation
        # raises OutOfMemoryError; after freeing, a 30 MB tensor lands in the fallback pool
        # (same address as an earlier one) but a second 30 MB tensor fails; the fallback pool
        # remains usable. Serial.
        allocator, _ = self.get_dummy_allocator(check_vars=False)
        pool_do_not_use = torch.cuda.MemPool(allocator.allocator())
        pool_use = torch.cuda.MemPool(allocator.allocator(), use_on_oom=True)

        nelem_1mb = 1024 * 1024 // 4

        self._setup_mempool_limited_memory_test(80)
        try:
            # remaining free mem: 80 mb
            # mempool_use [] 0 mb
            # mempool_do_not_use [] 0 mb
            # default pool [] 0 mb
            with torch.cuda.use_mem_pool(pool_do_not_use):
                a = torch.randn(40 * nelem_1mb, device="cuda")
            with torch.cuda.use_mem_pool(pool_use):
                b = torch.randn(40 * nelem_1mb, device="cuda")
            a_dataptr = a.data_ptr()
            b_dataptr = b.data_ptr()
            # remaining free mem: 0 mb
            # mempool_do_not_use [aaaa] 40 mb
            # mempool_use [bbbb] 40 mb
            # default pool [] 0 mb
            with self.assertRaises(torch.OutOfMemoryError):
                # out of memory
                c = torch.randn(40 * nelem_1mb, device="cuda")

            del a, b
            # remaining free mem: 0 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [____] 40 mb
            # default pool [] 0 mb

            # c should not oom and instead can use mempool_use as fallback
            c = torch.randn(30 * nelem_1mb, device="cuda")
            c_dataptr = c.data_ptr()
            # remaining free mem: 0 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [ccc_] 40 mb
            # default pool [] 0 mb
            with self.assertRaises(torch.OutOfMemoryError):
                # out of memory since can't use mempool_do_not_use
                d = torch.randn(30 * nelem_1mb, device="cuda")

            del c
            # remaining free mem: 0 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [____] 40 mb
            # default pool [] 0 mb

            # expect that we used same memory address for both a and c
            self.assertEqual(b_dataptr, c_dataptr)

            # make sure we can still use mempool_use as intended after c is deleted
            with torch.cuda.use_mem_pool(pool_use):
                e = torch.randn(20 * nelem_1mb, device="cuda")
            # remaining free mem: 0 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [ee__] 40 mb
            # default pool [] 0 mb

            e_dataptr = e.data_ptr()
            del e

            self.assertEqual(e_dataptr, c_dataptr)

            # pool's destructor calls emptyCache()
            del pool_use, pool_do_not_use
        finally:
            self._teardown_mempool_limited_memory_test()

    @serialTest()
    def test_mempool_release_cached_blocks_during_diversion(self):
        # [FUNCTION DOC] Regression test: with an 80 MB limit, caches 60 MB in the default pool,
        # then allocates 60 MB inside a MemPool; this requires releasing cached default-pool
        # blocks during the OOM retry even though no real capture is running. Passes if the
        # allocation succeeds with the right element count. Serial.
        # Regression test for the bug where allocations failing inside
        # `torch.cuda.use_mem_pool(...)` would skip the OOM-time
        # release_cached_blocks() retry, because the gate read
        # `captures_underway.empty()` — which is non-empty during any
        # private-pool diversion, even when no real cudaStreamBeginCapture
        # is active. The fix uses a separate `num_active_captures_` counter
        # tracked by CUDAGraph::capture_begin/end, so default-pool cached
        # blocks can still be reclaimed during plain mempool diversion.
        nelem_1mb = 1024 * 1024
        device, dtype = self._setup_mempool_limited_memory_test(80)
        try:
            # Reserve 60 MB on the default pool, then free → cached but not
            # returned to the driver. memory_reserved stays at 60 MB.
            a = torch.empty(60 * nelem_1mb, device=device, dtype=dtype)
            del a

            pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(pool):
                # 60 MB into the private pool. Reservation budget remaining
                # is only 20 MB (80 cap − 60 cached), so cudaMalloc must
                # fail. The OOM retry needs to call release_cached_blocks()
                # on the default pool to free the 60 MB and retry. Pre-fix,
                # that path was gated off and this would raise.
                b = torch.empty(60 * nelem_1mb, device=device, dtype=dtype)
            self.assertEqual(b.numel(), 60 * nelem_1mb)
            del b
            del pool
        finally:
            self._teardown_mempool_limited_memory_test()

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "not supported by CUDAMallocAsync")
    @unittest.skipIf(EXPANDABLE_SEGMENTS, "not supported by expandable segments")
    @serialTest()
    def test_mempool_no_split(self):
        # [FUNCTION DOC] Compares a normal MemPool with a no_split one: allocating 4 MB twice in
        # each pool; the no_split pool must have more segments, exactly one block per segment
        # and fewer blocks overall than the splitting pool. Skipped for cudaMallocAsync and
        # expandable segments; serial.
        torch.cuda.empty_cache()

        # Create two pools - one with no_split=True, one without
        pool_split = torch.cuda.MemPool()  # default: no_split=False
        pool_no_split = torch.cuda.MemPool(no_split=True)

        # 1 MB in number of float32 elements
        nelem_1mb = 1024 * 1024 // 4

        # Allocate 4 MB in each pool - this should allocate a 20 MB segment
        with torch.cuda.use_mem_pool(pool_split):
            a_split = torch.randn(4 * nelem_1mb, device="cuda")
        with torch.cuda.use_mem_pool(pool_no_split):
            a_no_split = torch.randn(4 * nelem_1mb, device="cuda")

        # Now allocate another 4 MB - this should cause splitting in the normal
        # pool but not in the no_split pool. Instead, the no_split pool should
        # have a new segment.
        with torch.cuda.use_mem_pool(pool_split):
            b_split = torch.randn(4 * nelem_1mb, device="cuda")
        with torch.cuda.use_mem_pool(pool_no_split):
            b_no_split = torch.randn(4 * nelem_1mb, device="cuda")

        # assert the number of segments in the no_split pool is larger than that
        # of the split pool
        if len(pool_no_split.snapshot()) <= len(pool_split.snapshot()):
            raise AssertionError(
                f"Expected no_split pool to have more segments, "
                f"but got {len(pool_no_split.snapshot())} vs {len(pool_split.snapshot())}"
            )

        # Specifically, the no_split pool should have exactly 1 block per segment
        for seg in pool_no_split.snapshot():
            if len(seg["blocks"]) != 1:
                raise AssertionError(
                    f"Expected 1 block in no_split segment, got {len(seg['blocks'])}"
                )

        # Count blocks in each pool
        def count_blocks(pool):
            # [FUNCTION DOC] HELPER: Returns the total number of blocks across all segments of a
            # pool snapshot.
            total = 0
            for seg in pool.snapshot():
                total += len(seg["blocks"])
            return total

        blocks_split = count_blocks(pool_split)
        blocks_no_split = count_blocks(pool_no_split)

        # The pool with no_split should have fewer blocks because it doesn't
        # have a remaining block representing the remaining memory in the segment.
        self.assertLess(
            blocks_no_split,
            blocks_split,
            lambda msg: f"{msg}\nExpected no_split pool to have fewer blocks, "
            f"but got {blocks_no_split} vs {blocks_split}",
        )

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "not supported by CUDAMallocAsync")
    @unittest.skipIf(EXPANDABLE_SEGMENTS, "not supported by expandable segments")
    @serialTest()
    def test_mempool_no_split_erased_on_release(self):
        # [FUNCTION DOC] Checks that destroying a no_split pool clears its no_split marking: two
        # 4 MB allocations get two segments, after deleting everything the snapshot for that id
        # is empty, and re-registering the same pool id as a normal pool lets two 4 MB
        # allocations share one segment. Pool is released at the end. Skipped for
        # cudaMallocAsync and expandable segments; serial.
        # Destroying a no_split MemPool must drop its id from the allocator's
        # no_split_pools set. Otherwise a pool later registered under the same
        # id silently inherits no_split and stops sharing segments.
        gc.collect()
        torch.cuda.empty_cache()
        device = torch.cuda.current_device()
        nelem_1mb = 1024 * 1024 // 4

        pool = torch.cuda.MemPool(no_split=True)
        pool_id = pool.id
        with torch.cuda.use_mem_pool(pool):
            a = torch.randn(4 * nelem_1mb, device="cuda")
            b = torch.randn(4 * nelem_1mb, device="cuda")
        # no_split: each 4 MB allocation gets its own 20 MB segment
        self.assertEqual(len(pool.snapshot()), 2)

        del a, b, pool
        torch.cuda.empty_cache()
        # the pool is fully gone -- nothing left under its id
        self.assertEqual(torch.cuda.memory_snapshot(pool_id), [])

        # Re-register the same id without no_split. The two 4 MB allocations
        # should now share one 20 MB segment (split). Pre-fix, the stale
        # no_split_pools entry would prevent splitting, giving 2 segments.
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, pool_id)
        try:
            c = torch.randn(4 * nelem_1mb, device="cuda")
            d = torch.randn(4 * nelem_1mb, device="cuda")
            self.assertEqual(len(torch.cuda.memory_snapshot(pool_id)), 1)
        finally:
            del c, d
            torch._C._cuda_endAllocateToPool(device, pool_id)
            torch._C._cuda_releasePool(device, pool_id)
            torch.cuda.empty_cache()

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/176145")
    @skipIfRocmArch(MI200_ARCH)
    @serialTest()
    def test_deleted_mempool_not_used_on_oom(self):
        # [FUNCTION DOC] Checks that deleted pools with use_on_oom are removed from the fallback
        # list: with a 40 MB limit, creates and deletes ten fallback pools holding allocations,
        # creates a new fallback pool using all 40 MB, then allocates two 20 MB tensors that
        # must be served from the valid pool without error. Skipped on Linux (tracked issue) and
        # MI200 ROCm; serial.
        """
        Test that a deleted mempool with use_on_oom=True is properly removed from use_on_oom_pools.
        """
        allocator, _ = self.get_dummy_allocator(check_vars=False)

        nelem_1mb = 1024 * 1024 // 4

        # set 40 mb total available memory
        self._setup_mempool_limited_memory_test(40)
        try:
            # Create many pools with use_on_oom=True, allocate memory, then delete the pools
            for _ in range(10):
                pool_use_on_oom = torch.cuda.MemPool(
                    allocator.allocator(), use_on_oom=True
                )
                with torch.cuda.use_mem_pool(pool_use_on_oom):
                    a = torch.randn(40 * nelem_1mb, device="cuda")
                del a
                del pool_use_on_oom

            # create new pool that we want to use_on_oom, all other pools should be deleted
            # all available 40mb in use by mempool
            new_pool_use_on_oom = torch.cuda.MemPool(
                allocator.allocator(), use_on_oom=True
            )
            with torch.cuda.use_mem_pool(new_pool_use_on_oom):
                a = torch.randn(40 * nelem_1mb, device="cuda")
            del a

            # allocate tensors that will fallback to use_on_oom pool since all available 40mb in use by mempool
            # tensors should only use valid pool and not deleted pools
            b = torch.randn(20 * nelem_1mb, device="cuda")
            c = torch.randn(20 * nelem_1mb, device="cuda")

            del b
            del c
            del new_pool_use_on_oom
        finally:
            self._teardown_mempool_limited_memory_test()

    def test_mempool_multithread(self):
        # [FUNCTION DOC] Creates four pools from four threads and asserts all four ids are
        # unique (id creation is atomic).
        pool_ids = []

        def create_mempool_and_make_active():
            # [FUNCTION DOC] HELPER: Thread body creating a MemPool and appending its id to the
            # shared list.
            pool = torch.cuda.MemPool()
            pool_ids.extend([pool.id])

        num_threads = 4
        threads = [
            threading.Thread(target=create_mempool_and_make_active)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # each thread should create a unique mempool, since
        # mempool id creation is atomic
        self.assertEqual(len(set(pool_ids)), 4)

    def test_mempool_emptycache_multithread(self):
        # [FUNCTION DOC] Four threads each allocate and free a small tensor inside their own mem
        # pool and call empty_cache; afterward each pool snapshot still has exactly one segment
        # (empty_cache does nothing under a mempool context).
        num_threads = 4

        def my_function(pool):
            # [FUNCTION DOC] HELPER: Thread body: inside the pool context allocates and deletes
            # a tensor, then empties the cache.
            with torch.cuda.use_mem_pool(pool):
                x = torch.randn(4, device="cuda")
                del x
                torch.cuda.empty_cache()

        pools = [torch.cuda.MemPool() for _ in range(num_threads)]
        threads = [
            threading.Thread(target=my_function, args=(pools[i],))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # empty_cache should have done nothing under mempool context
        for p in pools:
            s = p.snapshot()
            self.assertEqual(len(s), 1, "Expected to have a single segment")

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/180367")
    @serialTest()
    def test_nested_mempool(self):
        # [FUNCTION DOC] Skipped on Linux (tracked issue); serial. Nests three mem pool
        # contexts and allocates 20 MB tensors at each level (one in pool 1, one in pool 2, one
        # in pool 3, then one more in pool 2 and pool 1 after leaving inner scopes). Asserts
        # per-pool segment counts of 2, 2 and 1, showing the innermost active pool receives
        # each allocation.
        torch.cuda.empty_cache()
        pool1 = torch.cuda.MemPool()
        pool2 = torch.cuda.MemPool()
        pool3 = torch.cuda.MemPool()

        data = []
        nelem_1mb = 1024 * 1024 // 4

        def allocate_data():
            # [FUNCTION DOC] HELPER: Allocates a 20 MB float tensor and appends it to the shared
            # list so it stays alive.
            x = torch.empty(nelem_1mb * 20, device="cuda")
            data.append(x)

        with torch.cuda.use_mem_pool(pool1):
            allocate_data()
            with torch.cuda.use_mem_pool(pool2):
                allocate_data()
                with torch.cuda.use_mem_pool(pool3):
                    allocate_data()
                allocate_data()
            allocate_data()

        pool1_segments = torch.cuda.memory.memory_snapshot(pool1.id)
        pool2_segments = torch.cuda.memory.memory_snapshot(pool2.id)
        pool3_segments = torch.cuda.memory.memory_snapshot(pool3.id)

        self.assertEqual(len(pool1_segments), 2)
        self.assertEqual(len(pool2_segments), 2)
        self.assertEqual(len(pool3_segments), 1)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reclaim_2_streams(self):
        # [FUNCTION DOC] Enables the graph-capture record-stream-reuse allocator option and
        # captures a graph across two streams: a block read on a forked stream is freed before
        # the join and must not be reused by the next allocation, but is reused after the
        # streams are joined. Option is reset at the end.
        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )
        torch.cuda.empty_cache()

        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        g = torch.cuda.CUDAGraph(keep_graph=True)

        torch.cuda.synchronize()

        with torch.cuda.stream(s1):
            g.capture_begin()

            # A sink node allocated up-front so it doesn't steal data1's block later.
            sink1 = torch.empty(8, device="cuda")

            # Source tensor on s1; this block is the reuse candidate.
            data1 = torch.empty(8, device="cuda")
            data1_ptr = data1.data_ptr()

            # Fork: do real work on s2 that READS data1 and writes to its own buffer.
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                buf2 = torch.empty_like(data1)
                torch.add(data1, 2.0, out=buf2)
                data1.record_stream(s2)

            del data1

            # BEFORE JOIN: must NOT reuse
            data2 = torch.empty(8, device="cuda")
            data2_ptr = data2.data_ptr()

            # Join s2 -> s1 and add a sink node on s1.
            s1.wait_stream(s2)
            sink1.fill_(1.0)

            # AFTER JOIN: now reuse is allowed
            data3 = torch.empty(8, device="cuda")
            data3_ptr = data3.data_ptr()

            g.capture_end()

        torch.cuda.synchronize()

        # No reuse before join; reuse after join.
        self.assertNotEqual(data1_ptr, data2_ptr)
        self.assertEqual(data1_ptr, data3_ptr)

        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:False"
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reclaim_4_streams(self):
        # [FUNCTION DOC] Same reuse check with four streams: a block read on three forked
        # streams is freed; after partial joins (some streams joined, others not) a new
        # allocation must not reuse it, and after the final join of all branches the next
        # allocation reuses it. Option is reset at the end.
        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )

        torch.cuda.empty_cache()
        s1, s2, s3, s4 = (
            torch.cuda.Stream(),
            torch.cuda.Stream(),
            torch.cuda.Stream(),
            torch.cuda.Stream(),
        )
        g = torch.cuda.CUDAGraph(keep_graph=True)

        torch.cuda.synchronize()

        with torch.cuda.stream(s1):
            g.capture_begin()

            # Source tensor allocated on s1. This block is the candidate for reuse.
            data1 = torch.ones(8, device="cuda")
            data1_ptr = data1.data_ptr()
            sink1 = torch.empty_like(data1)
            sink3 = torch.empty_like(data1)

            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                buf2 = torch.empty_like(data1)
                torch.add(data1, 2.0, out=buf2)
                data1.record_stream(s2)

            s3.wait_stream(s1)
            with torch.cuda.stream(s3):
                buf3 = torch.empty_like(data1)
                torch.add(data1, 3.0, out=buf3)
                data1.record_stream(s3)

            s4.wait_stream(s1)
            with torch.cuda.stream(s4):
                buf4 = torch.empty_like(data1)
                torch.add(data1, 4.0, out=buf4)
                data1.record_stream(s4)

            # Free data1 inside capture; allocator may reuse later when it's safe.
            del data1

            # PARTIAL JOINS: should NOT allow reuse yet
            # Join s2 -> s1 and add a sink node on s1.
            s1.wait_stream(s2)
            sink1.fill_(1.0)

            # Join s4 -> s3 and add a sink node on s3.
            s3.wait_stream(s4)
            with torch.cuda.stream(s3):
                sink3.fill_(3.0)
                sink3.record_stream(s3)

            # At this point, s1 and s3 subgraphs are NOT yet joined together.
            # Allocating data2 here must NOT reuse data1's block.
            data2 = torch.empty(8, device="cuda")
            data2_ptr = data2.data_ptr()

            # FINAL JOIN: now reuse is allowed
            # Join s3 -> s1 and add a sink node on s1.
            s1.wait_stream(s3)
            sink1.add_(sink3)

            # Now allocator should safely reuse data1's block.
            data3 = torch.empty(8, device="cuda")
            data3_ptr = data3.data_ptr()

            g.capture_end()

        torch.cuda.synchronize()

        # No reuse before full join; reuse after full join.
        self.assertNotEqual(data1_ptr, data2_ptr)
        self.assertEqual(data1_ptr, data3_ptr)

        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:False"
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reclaim_shared_pool(self):
        # [FUNCTION DOC] With the record-stream-reuse option, captures a first graph in a shared
        # pool that allocates, uses on a side stream (joined back) and frees an 8 MiB block,
        # then a second graph in the same pool; the second graph allocation must reuse the first
        # block address. Option reset at end.
        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        shared_pool = torch.cuda.graph_pool_handle()
        cap_stream = torch.cuda.Stream()
        side_stream = torch.cuda.Stream()

        g1 = torch.cuda.CUDAGraph()
        g2 = torch.cuda.CUDAGraph()

        numel = (8 * 1024 * 1024) // 4

        with torch.cuda.stream(cap_stream):
            g1.capture_begin(pool=shared_pool)
            data = torch.empty(numel, device="cuda")
            data_ptr = data.data_ptr()

            side_stream.wait_stream(cap_stream)
            with torch.cuda.stream(side_stream):
                data.add_(1.0)
                data.record_stream(side_stream)

            cap_stream.wait_stream(side_stream)

            del data
            g1.capture_end()

        torch.cuda.current_stream().wait_stream(cap_stream)
        torch.cuda.synchronize()

        with torch.cuda.stream(cap_stream):
            g2.capture_begin(pool=shared_pool)
            data2 = torch.empty(numel, device="cuda")
            data2.fill_(42.0)
            data2_ptr = data2.data_ptr()
            g2.capture_end()

        torch.cuda.current_stream().wait_stream(cap_stream)
        torch.cuda.synchronize()

        self.assertEqual(data_ptr, data2_ptr)

        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:False"
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(TEST_WITH_ROCM, "ROCM does not support nvrtc")
    @unittest.skipIf(
        not SM70OrLater, "Compute capability >= SM70 required for relaxed ptx flag"
    )
    def test_graph_capture_pre_capture_stream_use(self):
        # [FUNCTION DOC] Exercises a block with pre-capture stream usage in a shared pool. A
        # block is allocated in the first capture, recorded on a side stream held busy by a
        # spin-wait kernel, and freed in a second capture. While the kernel spins, a third
        # capture allocation must not reuse the address; after releasing the kernel, a fourth
        # capture allocation reuses it. Needs SM70+ and not ROCm.
        # Tests that a block with pre-capture stream uses is correctly handled
        # when freed during a subsequent capture on the same pool.
        # Exercises the insert_events path in endAllocateToPool.
        spin_wait_kernel = get_wait_for_cpu_kernel()

        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        shared_pool = torch.cuda.graph_pool_handle()
        cap_stream = torch.cuda.Stream()
        side_stream = torch.cuda.Stream()
        flag_cpu = torch.zeros(1, dtype=torch.int32, device="cpu").pin_memory()

        g1 = torch.cuda.CUDAGraph()
        g2 = torch.cuda.CUDAGraph()
        g3 = torch.cuda.CUDAGraph()
        g4 = torch.cuda.CUDAGraph()

        numel = (8 * 1024 * 1024) // 4

        # First capture: allocate data in the shared pool, keep it alive.
        with torch.cuda.stream(cap_stream):
            g1.capture_begin(pool=shared_pool)
            data = torch.empty(numel, device="cuda")
            data_ptr = data.data_ptr()
            g1.capture_end()

        torch.cuda.synchronize()

        # Between captures: block side_stream with a spin-wait kernel
        # (pre-capture stream use). The kernel holds the stream busy until
        # we explicitly set the flag from the host.
        with torch.cuda.stream(side_stream):
            spin_wait_kernel(grid=(1, 1, 1), block=(1, 1, 1), args=[flag_cpu])
            data.record_stream(side_stream)

        # Second capture: free data during capture.
        with torch.cuda.stream(cap_stream):
            g2.capture_begin(pool=shared_pool)
            del data
            g2.capture_end()

        # Trigger process_events. The spin kernel is still holding side_stream,
        # so cudaEventQuery returns cudaErrorNotReady and the block stays pending.
        torch.empty(1, device="cuda")

        # Allocate from the same pool: block must NOT be reused yet.
        with torch.cuda.stream(cap_stream):
            g3.capture_begin(pool=shared_pool)
            not_reused = torch.empty(numel, device="cuda")
            not_reused_ptr = not_reused.data_ptr()
            g3.capture_end()

        self.assertNotEqual(data_ptr, not_reused_ptr)

        # Release the spin kernel so side_stream can finish.
        flag_cpu[0] = 1
        torch.cuda.synchronize()

        # Trigger process_events to reclaim the block.
        torch.empty(1, device="cuda")

        # Fourth capture: the block should now be reusable.
        with torch.cuda.stream(cap_stream):
            g4.capture_begin(pool=shared_pool)
            reused = torch.empty(numel, device="cuda")
            reused_ptr = reused.data_ptr()
            g4.capture_end()

        self.assertEqual(data_ptr, reused_ptr)

        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:False"
        )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/177000"
    )
    @unittest.skipIf(
        not EXPANDABLE_SEGMENTS,
        "requires expandable_segments mode (run via test_cuda_expandable_segments.py)",
    )
    @skipIfRocm(msg="expandable_segments mode is not supported on ROCm")
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Load_inline doesn't work in fbcode")
    def test_mempool_expandable(self):
        # [FUNCTION DOC] Needs expandable segments mode (not ROCm, not fbcode; skipped on Linux
        # and slow runs for a tracked issue). Allocates one tensor from a MemPool with a custom
        # allocator and one from the default pool; asserts the snapshot has two segments of
        # which exactly one is expandable.
        torch.cuda.empty_cache()
        allocator, _ = self.get_dummy_allocator(check_vars=False)
        pool = torch.cuda.MemPool(allocator.allocator())

        data = []
        nelem = 1024 * 1024 // 4
        with torch.cuda.use_mem_pool(pool):
            data.append(torch.empty(nelem, device="cuda"))

        # the second allocation should be in expandable segment
        data.append(torch.empty(nelem, device="cuda"))

        segments = torch.cuda.memory.memory_snapshot()

        num_expandable_segments = 0
        for segment in segments:
            if segment["is_expandable"]:
                num_expandable_segments += 1

        self.assertEqual(len(segments), 2, "Expected to have 2 segment")
        self.assertEqual(
            num_expandable_segments, 1, "Expected to have 1 expandable segment only"
        )

    @serialTest()
    def test_mempool_ctx_multithread(self):
        # [FUNCTION DOC] Verifies mem pool context is thread local. After clearing caches and
        # checking an empty snapshot, the main thread enters a pool context and signals; a
        # second thread then allocates (default pool) and takes a snapshot. The snapshot must
        # have two segments: the default pool holds 2 MB active and the custom pool holds none.
        # Times out after 10 s with failure; serial.
        # Collect first: a tensor leaked by a prior test that is only reachable
        # through a reference cycle keeps its segment active, so empty_cache
        # can't reclaim it and the "empty pool" assertion below flakes.
        gc.collect()
        torch._C._cudnn_clear_dropout_state()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        segments = torch.cuda.memory._snapshot()["segments"]
        self.assertEqual(len(segments), 0, "Expected empty pool in the beginning")

        nelem = 1024 * 1024
        trigger_alloc = threading.Event()
        done_allocation = threading.Event()

        def main_thread_fn():
            # [FUNCTION DOC] HELPER: Thread body that creates a pool, allocates outside it, and
            # inside the pool context allocates and frees a tensor, signals the other thread and
            # waits until it is done.
            pool = torch.cuda.MemPool()
            out1 = torch.empty(nelem, dtype=torch.int8, device="cuda")
            with torch.cuda.use_mem_pool(pool):
                out = torch.empty(nelem, dtype=torch.int8, device="cuda")
                del out
                trigger_alloc.set()
                done_allocation.wait()

        def side_thread_fn(segments):
            # [FUNCTION DOC] HELPER: Thread body that waits for the signal, allocates a
            # tensor on the default pool, appends a memory snapshot to the given list and
            # signals completion.
            trigger_alloc.wait()
            out = torch.empty(nelem, dtype=torch.int8, device="cuda")
            s = torch.cuda.memory._snapshot()["segments"]
            segments.append(s)
            done_allocation.set()

        segments = []
        main_thread = threading.Thread(target=main_thread_fn)
        side_thread = threading.Thread(target=side_thread_fn, args=(segments,))

        main_thread.start()
        side_thread.start()
        main_thread.join(timeout=10)
        side_thread.join(timeout=10)

        if main_thread.is_alive() or side_thread.is_alive():
            # release threads so that they don't hang forever
            trigger_alloc.set()
            done_allocation.set()
            self.fail(
                "Test timed out - threads did not complete within the allowed time"
            )

        self.assertEqual(len(segments), 1, "Expected to have memory snapshot")
        self.assertEqual(len(segments[0]), 2, "Expected to have 2 segments allocated")
        active = defaultdict(int)
        for s in segments[0]:
            active[s["segment_pool_id"]] += s["active_size"]
        for k, v in active.items():
            if k == (0, 0):
                self.assertEqual(
                    v, 2097152, "Expected to have 2MB allocated in the default pool"
                )
            else:
                self.assertEqual(
                    v, 0, "Expected to have 0 bytes allocated in the custom pool"
                )

    def test_snapshot_include_traces(self):
        # [FUNCTION DOC] Benchmark-style test of the include_traces option. Records history,
        # allocates 1000 tensors in a MemPool, then times (10 runs after warmup) pool and global
        # snapshots with and without traces, asserts the last snapshot is non-empty, prints the
        # averages and speedups, and turns history recording off.
        """Test that snapshot() include_traces parameter works correctly"""
        import time

        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history()

        pool = torch.cuda.MemPool()

        # Generate trace entries
        with torch.cuda.use_mem_pool(pool):
            tensors = []
            for i in range(1000):
                tensors.append(torch.randn(1024, device="cuda"))
            del tensors

        NUM_RUNS = 10
        times_pool_full = []
        times_pool_notrace = []
        times_global_full = []
        times_global_notrace = []

        # Measure mempool snapshot without traces
        for _ in range(NUM_RUNS):
            # warmup
            snapshot = pool.snapshot(include_traces=False)
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            snapshot = pool.snapshot(include_traces=False)
            times_pool_notrace.append((time.perf_counter() - start) * 1000)

        # Measure global snapshot without traces
        for _ in range(NUM_RUNS):
            # warmup
            snapshot = torch.cuda.memory_snapshot(include_traces=False)
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            snapshot = torch.cuda.memory_snapshot(include_traces=False)
            times_global_notrace.append((time.perf_counter() - start) * 1000)

        # Measure mempool snapshot with traces
        for _ in range(NUM_RUNS):
            # warmup
            snapshot = pool.snapshot()
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            snapshot = pool.snapshot()
            times_pool_full.append((time.perf_counter() - start) * 1000)

        # Measure global snapshot with traces
        for _ in range(NUM_RUNS):
            # warmup
            snapshot = torch.cuda.memory_snapshot()
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            snapshot = torch.cuda.memory_snapshot()
            times_global_full.append((time.perf_counter() - start) * 1000)

        self.assertTrue(len(snapshot) > 0)

        print(f"Mempool with traces:    {sum(times_pool_full) / NUM_RUNS:.1f} ms")
        print(f"Mempool without traces: {sum(times_pool_notrace) / NUM_RUNS:.1f} ms")
        print(f"Global with traces:     {sum(times_global_full) / NUM_RUNS:.1f} ms")
        print(f"Global without traces:  {sum(times_global_notrace) / NUM_RUNS:.1f} ms")
        print()
        print(f"Mempool speedup: {sum(times_pool_full) / sum(times_pool_notrace):.1f}x")
        print(
            f"Global speedup: {sum(times_global_full) / sum(times_global_notrace):.1f}x"
        )

        torch.cuda.memory._record_memory_history(enabled=None)

    def test_snapshot_include_traces_correctness(self):
        # [FUNCTION DOC] Allocates three 1 MiB tensors, takes global snapshots with and without
        # traces and asserts both are non-empty and report the same total bytes in active
        # allocated blocks.
        """Test that snapshot with include_traces=False still returns correct memory state"""
        torch.cuda.empty_cache()

        tensor_size = 1024 * 1024
        dtype = torch.uint8
        t1 = torch.ones(tensor_size, device="cuda", dtype=dtype)
        t2 = torch.ones(tensor_size, device="cuda", dtype=dtype)
        t3 = torch.ones(tensor_size, device="cuda", dtype=dtype)

        snapshot_with_traces = torch.cuda.memory_snapshot(include_traces=True)
        snapshot_without_traces = torch.cuda.memory_snapshot(include_traces=False)

        # Both should have segment information
        self.assertTrue(
            len(snapshot_with_traces) > 0, "Snapshot with traces should have segments"
        )
        self.assertTrue(
            len(snapshot_without_traces) > 0,
            "Snapshot without traces should have segments",
        )

        # Count total allocated memory in each snapshot
        def get_total_allocated(segments):
            # [FUNCTION DOC] HELPER: Sums the size of blocks in the active allocated state
            # across all segments in a snapshot.
            total = 0
            for segment in segments:
                for block in segment.get("blocks", []):
                    if block.get("state") == "active_allocated":
                        total += block.get("size", 0)
            return total

        allocated_with_traces = get_total_allocated(snapshot_with_traces)
        allocated_without_traces = get_total_allocated(snapshot_without_traces)

        # Allocated memory should match between both snapshots
        self.assertEqual(
            allocated_with_traces,
            allocated_without_traces,
            "Allocated memory should be same with or without traces",
        )

    @unittest.skipIf(TEST_WITH_ROCM, "not enabled by default on rocm")
    @serialTest()
    def test_multi_threads_alloc_in_same_order(self):
        # [FUNCTION DOC] Checks allocation addresses are reproducible with multi-threaded
        # allocation. Four threads pop sizes under a lock and allocate int8 tensors; the
        # addresses of the first round are compared with a second round done after a split-merge
        # cycle of smaller tensors; all addresses must match. Skipped on ROCm; serial.
        def alloc_tensors(
            stream: torch.cuda.Stream,
            sizes: [int],
            device: str,
            alloc_results: [],
            lock,
            thread_id=0,
        ):
            # [FUNCTION DOC] HELPER: Thread body: repeatedly, while holding the lock, pops a
            # size from the shared list and appends a tuple of the allocated int8 tensor, thread
            # id and size to the results; returns when the list is empty.
            while len(sizes) > 0:
                with lock:
                    if len(sizes) == 0:
                        return
                    sz = sizes.pop()
                    alloc_results.append(
                        (
                            torch.empty(sz, dtype=torch.int8, device=device),
                            thread_id,
                            sz,
                        )
                    )

        def alloc_with_threads(
            thread_count: int, stream: torch.cuda.Stream, mem_sizes: [int], device: str
        ):
            # [FUNCTION DOC] HELPER: Starts the given number of threads each running the
            # allocation loop over a copy of the size list, joins them, and returns the list of
            # data pointer, thread id and size for every allocated tensor (the tensors are freed
            # afterward when the list goes away).
            tensors_list = []
            threads = []
            mem_sizes = mem_sizes[:]
            lock = threading.Lock()
            for i in range(thread_count):
                t = threading.Thread(
                    target=alloc_tensors,
                    args=(stream, mem_sizes, "cuda", tensors_list, lock, i),
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            tensors_ptrs = [
                (t.data_ptr(), thread_id, sz) for (t, thread_id, sz) in tensors_list
            ]
            return tensors_ptrs

        stream = torch.cuda.current_stream()
        thread_count = 4
        mem_sizes = [
            s * 2 * 1024 * 1024 for s in range(1, 5) for _ in range(thread_count)
        ]
        tensor_ptrs_round_1 = alloc_with_threads(
            thread_count, stream, mem_sizes, "cuda"
        )
        # Force a split-merge cycle: allocate half-sized tensors (splits the
        # freed blocks), then free them (triggers merges back). With per-block
        # counters this changes the free-set ordering; with segment-level
        # counters it stays stable.
        splitters = [
            torch.empty(s, dtype=torch.int8, device="cuda")
            for s in [s * 1024 * 1024 for s in range(1, 5) for _ in range(thread_count)]
        ]
        del splitters
        tensor_ptrs_round_2 = alloc_with_threads(
            thread_count, stream, mem_sizes, "cuda"
        )
        ptrs_round_1 = torch.tensor([ptr for ptr, _, _ in tensor_ptrs_round_1])
        ptrs_round_2 = torch.tensor([ptr for ptr, _, _ in tensor_ptrs_round_2])
        self.assertTrue((ptrs_round_1 == ptrs_round_2).all().item())

    @unittest.skipIf(TEST_WITH_ROCM, "not enabled by default on rocm")
    @requires_cuda_python_bindings
    @serialTest()
    def test_nccl_mem_alloc_addresses_in_random_order(self):
        # [FUNCTION DOC] Simulates a symmetric-memory allocator (like ncclMemAlloc) whose
        # addresses come from a pre-allocated 1 GB buffer handed out from the head for the first
        # stream and from the tail for the second. Allocates three rounds of tensors on two
        # streams in a MemPool with that allocator, frees them, does a split-merge cycle,
        # allocates again and asserts the second round got the same addresses as the first and
        # as the allocator record, with one segment per tensor. Needs the cuda python bindings;
        # skipped on ROCm; serial.
        """
        Test NCCL mem allocator with non-increasing allocation addresses.

        This test uses a custom allocator to simulate ncclMemAlloc scenario where
        allocated memory addresses may not be monotonically increasing across
        different ranks due to fragmentation, OS allocation policies, or prior
        allocations.

        The registration counter-based comparator ensures consistent block ordering
        regardless of actual memory addresses, which is critical for NCCL
        symmetric memory alignment.
        """

        from cuda.bindings import runtime

        ALLOC_FN = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )
        FREE_FN = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )

        class AllocState:
            # [CLASS DOC] HELPER: State of the custom allocator: remembers the first and second
            # stream seen, the list of handed-out addresses, and a 1 GB buffer allocated with
            # the cuda runtime (head and tail cursors); frees the buffer when destroyed.
            def __init__(self, test_instance):
                # [FUNCTION DOC] HELPER: Allocates the 1 GB device buffer via the runtime
                # bindings, asserting success through the test instance, and initializes cursors
                # at both ends.
                self.first_stream = None
                self.second_stream = None
                self.allocated_addrs = []
                self.buffer_size = 1 * 1024 * 1024 * 1024  # 1GB
                self.base_ptr = -1
                err, base_ptr = runtime.cudaMalloc(self.buffer_size)
                test_instance.assertEqual(
                    err,
                    runtime.cudaError_t.cudaSuccess,
                    "init allocation for test_nccl_mem_alloc_addresses_in_random_order should be successful",
                )
                self.base_ptr = base_ptr
                self.head = base_ptr
                self.tail = base_ptr + self.buffer_size

            def __del__(self):
                # [FUNCTION DOC] HELPER: Frees the backing buffer if it was allocated.
                if self.base_ptr > 0:
                    runtime.cudaFree(self.base_ptr)

        state = AllocState(self)

        def my_alloc(size, device, stream, _runtime=runtime):
            # [FUNCTION DOC] HELPER: Custom allocation callback. Remembers the first two streams
            # seen; for the first stream returns the next address from the head, otherwise from
            # the tail of the buffer, records and returns the address.
            nonlocal state
            if state.first_stream is None:
                state.first_stream = stream
            elif state.second_stream is None:
                state.second_stream = stream

            if state.first_stream == stream:
                ptr = state.head
                state.head += size
            else:
                state.tail -= size
                ptr = state.tail
            state.allocated_addrs.append(ptr)
            return ptr

        def my_free(ptr, size, device, stream, _runtime=runtime):
            # [FUNCTION DOC] HELPER: Custom free callback that does nothing (memory is owned by
            # the state buffer).
            pass

        # Must keep these alive for the lifetime of the allocator
        c_alloc = ALLOC_FN(my_alloc)
        c_free = FREE_FN(my_free)
        alloc_ptr = ctypes.cast(c_alloc, ctypes.c_void_p).value
        free_ptr = ctypes.cast(c_free, ctypes.c_void_p).value
        allocator = torch._C._cuda_customAllocator(alloc_ptr, free_ptr)
        pool = torch.cuda.MemPool(allocator)

        first_stream = torch.cuda.Stream()
        second_stream = torch.cuda.Stream()

        tensor_sizes = [24 * 1024 * 1024, 32 * 1024 * 1024]
        alloc_cases = []
        case_no = 0
        for idx in range(3):
            for stream_idx, stream in enumerate([first_stream, second_stream]):
                # for second stream in reverse order
                reverse_order = stream_idx % 2 == 1
                tensor_sizes.sort(reverse=reverse_order)
                for size in tensor_sizes:
                    alloc_cases.append(
                        {"stream": stream, "size": size, "case_no": case_no}
                    )
                    case_no += 1

        # Test allocation and deallocation patterns
        def alloc_tensors():
            # [FUNCTION DOC] HELPER: Allocates a uint8 tensor for each recorded case on the
            # case's stream and returns the list of tensors.
            all_tensors = []
            for case in alloc_cases:
                with torch.cuda.stream(case["stream"]):
                    all_tensors.append(
                        torch.empty(case["size"], dtype=torch.uint8, device="cuda")
                    )
            return all_tensors

        with torch.cuda.use_mem_pool(pool):
            first_round_tensors = alloc_tensors()
        tensor_ptrs = [t.data_ptr() for t in first_round_tensors]
        del first_round_tensors

        # Split-merge cycle on ONE block per stream. Each stream has
        # three 24 MB blocks with counters [c0, c1, c2]. Splitting only
        # the first (lowest-counter) block gives it a new, higher
        # counter after the merge, reversing its position among the
        # three. With per-block counters the reallocation order changes
        # (wrong); with segment-level counters it stays stable.
        with torch.cuda.use_mem_pool(pool):
            with torch.cuda.stream(first_stream):
                s1 = torch.empty(tensor_sizes[0] // 2, dtype=torch.uint8, device="cuda")
            with torch.cuda.stream(second_stream):
                s2 = torch.empty(tensor_sizes[0] // 2, dtype=torch.uint8, device="cuda")
            del s1, s2

        with torch.cuda.use_mem_pool(pool):
            second_round_tensors = alloc_tensors()
        second_round_tensor_ptrs = [t.data_ptr() for t in second_round_tensors]
        del second_round_tensors

        mem_snapshot = pool.snapshot()
        self.assertEqual(
            len(mem_snapshot),
            len(tensor_ptrs),
            lambda msg: f"{msg}\nexpected to have {len(tensor_ptrs)} segments, but actually got {len(mem_snapshot)}",
        )

        for idx, first_addr in enumerate(tensor_ptrs):
            second_addr = second_round_tensor_ptrs[idx]
            self.assertEqual(
                second_addr,
                first_addr,
                "mem addr allocated for second round should be same with first round",
            )
            self.assertEqual(
                second_addr,
                state.allocated_addrs[idx],
                lambda msg: f"{msg}\n{second_round_tensor_ptrs[idx]=} != {state.allocated_addrs[idx]=}",  # noqa: F821
            )
        del pool
        del state
        torch.cuda.empty_cache()

    def _alloc_record_free_sync(self, pool_ctx, s2, nbytes):
        # [FUNCTION DOC] HELPER: Within the given context manager, allocates a block, records it
        # on a second stream and uses it there, frees it, synchronizes and allocates a block of
        # the same size again; returns the original and the new data pointers.
        """Allocate a block, record it on a second stream, free, synchronize,
        then try to reallocate.  Returns (original_ptr, new_ptr)."""
        with pool_ctx:
            a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            original_ptr = a.data_ptr()

            with torch.cuda.stream(s2):
                a.record_stream(s2)
                _ = a + 1

            del a
            torch.cuda.synchronize()

            b = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            new_ptr = b.data_ptr()
            del b
        return original_ptr, new_ptr

    @serialTest()
    def test_mempool_oom_recovery_releases_cached_blocks(self):
        # [FUNCTION DOC] Checks that OOM recovery releases cached blocks of the default pool
        # even when allocating inside a user mem pool. For both the default pool and a user
        # pool, fills about half of free memory and frees it (cached), then allocates around 7/8
        # of the free memory; the allocation must not raise OutOfMemoryError. Skips if not
        # enough memory or on ROCm 7.14+ with expandable segments; serial.
        """Test that the allocator can release default-pool cached blocks to
        satisfy a new allocation when a user mempool is active.

        We fill the default pool's free list with cached blocks (by allocating
        and freeing outside use_mem_pool), then try to allocate inside
        use_mem_pool.  Two paths:
          1. Default pool -- OOM recovery releases cached blocks, succeeds.
          2. use_mem_pool -- same recovery should work (the fix).
        """
        if TEST_WITH_ROCM and getRocmVersion() >= (7, 14) and EXPANDABLE_SEGMENTS:
            self.skipTest(
                "ROCm 7.14+ expandable segments OOMs before mempool cached "
                "blocks can be recovered"
            )

        MB = 1024 * 1024
        device = torch.device("cuda:0")

        def align_down_2mb(n):
            # [FUNCTION DOC] HELPER: Rounds a byte count down to a multiple of 2 MiB.
            return n & ~(2 * MB - 1)

        for label, make_ctx in [
            ("default pool", lambda: contextlib.nullcontext()),
            (
                "user mempool",
                lambda: torch.cuda.use_mem_pool(torch.cuda.MemPool()),
            ),
        ]:
            with self.subTest(label=label):
                torch.cuda.empty_cache()
                free_before = torch.cuda.mem_get_info(device)[0]

                fill_size = align_down_2mb(free_before // 2)
                if fill_size < 64 * MB:
                    self.skipTest("Not enough GPU memory for this test")

                filler = torch.empty(fill_size, dtype=torch.uint8, device=device)
                del filler

                alloc_size = align_down_2mb(free_before - free_before // 8)
                oom = False
                try:
                    with make_ctx():
                        big = torch.empty(alloc_size, dtype=torch.uint8, device=device)
                        del big
                except torch.cuda.OutOfMemoryError:
                    oom = True

                self.assertFalse(
                    oom,
                    lambda msg: f"{msg}\n[{label}] OOM even though the default pool had "
                    f"{fill_size // MB} MiB of freeable cached blocks "
                    "-- release_cached_blocks was likely skipped",
                )

    @serialTest()
    def test_mempool_block_free_not_deferred(self):
        # [FUNCTION DOC] Checks that multi-stream block frees are not deferred when only a user
        # mem pool is active (no capture): for the default pool and a user pool, a block
        # recorded on a second stream, freed and synchronized is reused by the next same-sized
        # allocation. Serial.
        """Test that multi-stream block frees are NOT deferred when only a
        user mempool is active (no graph capture).

        During graph capture, block frees must be deferred (cudaEventRecord
        is illegal).  For user mempools, insert_events should be called
        immediately.  Two paths:
          1. Default pool -- block recycled (baseline).
          2. use_mem_pool -- block recycled (the fix).
        """
        torch.cuda.empty_cache()
        s2 = torch.cuda.Stream()
        nbytes = 1024 * 1024

        for label, pool_ctx in [
            ("default pool", contextlib.nullcontext()),
            ("user mempool", torch.cuda.use_mem_pool(torch.cuda.MemPool())),
        ]:
            with self.subTest(label=label):
                torch.cuda.empty_cache()
                orig, new = self._alloc_record_free_sync(pool_ctx, s2, nbytes)
                self.assertEqual(
                    new,
                    orig,
                    lambda msg: f"{msg}\n[{label}] Block not reused after multi-stream free "
                    "-- free was likely deferred as if under graph capture",
                )

    def _reserved_bytes_by_private_pool(self, pool_id):
        # [FUNCTION DOC] HELPER: Returns the nested stats mapping for the reserved bytes of the
        # given private pool id from the nested memory stats.
        return torch.cuda.memory_stats_as_nested_dict()[
            "reserved_bytes_by_private_pools"
        ][pool_id]

    def _check_reserved_bytes_by_private_pools(self):
        # [FUNCTION DOC] HELPER: Shared checks for private pool reserved byte statistics using
        # 20 MB tensors: current bytes grow as tensors are allocated and match the flattened
        # stats key; empty_cache and stat resets do not reduce them; deleting the pool sets
        # current to zero while keeping peak and the freed total; two pools keep separate
        # entries and deleting one only zeroes its own.
        # 5M float32 elements = 20MB per tensor, above kMinLargeAlloc (10MB)
        NELEMS = 5 * 1024 * 1024
        TENSOR_SIZE = NELEMS * 4

        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()

        # Case 1: Allocating multiple tensors increases reserved bytes for one pool.
        pool = torch.cuda.MemPool()
        pool_id = pool.id
        tensors = []
        prev = 0
        with torch.cuda.use_mem_pool(pool):
            for i in range(4):
                tensors.append(torch.randn(NELEMS, device="cuda", dtype=torch.float32))
                cur = self._reserved_bytes_by_private_pool(pool_id)["all"]["current"]
                self.assertGreaterEqual(cur, (i + 1) * TENSOR_SIZE)
                self.assertGreaterEqual(cur, prev)
                prev = cur
        reserved_after_alloc = self._reserved_bytes_by_private_pool(pool_id)["all"][
            "current"
        ]
        flat_pool_id = "_".join(str(part) for part in pool_id)
        stats = torch.cuda.memory_stats()
        self.assertEqual(
            stats[f"reserved_bytes_by_private_pools.{flat_pool_id}.all.current"],
            reserved_after_alloc,
        )

        # Case 2: empty_cache does not reduce private pool reserved bytes.
        del tensors
        torch.cuda.empty_cache()
        self.assertEqual(
            self._reserved_bytes_by_private_pool(pool_id)["all"]["current"],
            reserved_after_alloc,
        )

        # Case 3: Resetting stats does not reduce private pool reserved bytes.
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        reserved_bytes = self._reserved_bytes_by_private_pool(pool_id)["all"]
        self.assertEqual(reserved_bytes["current"], reserved_after_alloc)
        self.assertEqual(reserved_bytes["peak"], reserved_after_alloc)
        self.assertEqual(reserved_bytes["allocated"], 0)
        self.assertEqual(reserved_bytes["freed"], 0)

        # Case 4: Deleting the pool zeros current bytes but preserves history.
        del pool
        torch.cuda.empty_cache()
        reserved_bytes = self._reserved_bytes_by_private_pool(pool_id)["all"]
        self.assertEqual(reserved_bytes["current"], 0)
        self.assertEqual(reserved_bytes["peak"], reserved_after_alloc)
        self.assertEqual(reserved_bytes["allocated"], 0)
        self.assertEqual(reserved_bytes["freed"], reserved_after_alloc)

        # Case 5: Two private pools maintain separate entries.
        pool1 = torch.cuda.MemPool()
        pool2 = torch.cuda.MemPool()
        pool1_id = pool1.id
        pool2_id = pool2.id
        with torch.cuda.use_mem_pool(pool1):
            t1 = torch.randn(NELEMS, device="cuda", dtype=torch.float32)
        reserved1 = self._reserved_bytes_by_private_pool(pool1_id)["all"]["current"]
        self.assertEqual(reserved1, TENSOR_SIZE)

        with torch.cuda.use_mem_pool(pool2):
            t2 = torch.randn(NELEMS, device="cuda", dtype=torch.float32)
        reserved2 = self._reserved_bytes_by_private_pool(pool2_id)["all"]["current"]
        self.assertEqual(reserved2, TENSOR_SIZE)

        # Deleting one pool zeros only that pool's current bytes.
        del t1
        del pool1
        torch.cuda.empty_cache()
        self.assertEqual(
            self._reserved_bytes_by_private_pool(pool1_id)["all"]["current"], 0
        )
        self.assertEqual(
            self._reserved_bytes_by_private_pool(pool2_id)["all"]["current"],
            TENSOR_SIZE,
        )

        # Deleting the second pool also zeros it, while the entry remains.
        del t2
        del pool2
        torch.cuda.empty_cache()
        self.assertEqual(
            self._reserved_bytes_by_private_pool(pool2_id)["all"]["current"], 0
        )

    @serialTest()
    def test_reserved_bytes_by_private_pools(self):
        # [FUNCTION DOC] Runs the private pool reserved-bytes checks with the current allocator
        # settings (serial).
        self._check_reserved_bytes_by_private_pools()

    @skipIfRocm(msg="expandable_segments mode is not supported on ROCm")
    @serialTest()
    def test_reserved_bytes_by_private_pools_expandable(self):
        # [FUNCTION DOC] Runs the private pool reserved-bytes checks after enabling
        # expandable segments, then restores the suite baseline expandable setting. Skipped
        # on ROCm; serial.
        torch.cuda.empty_cache()
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        try:
            self._check_reserved_bytes_by_private_pools()
        finally:
            torch.cuda.empty_cache()
            # Test toggles expandable_segments internally; restore the
            # suite's baseline so subsequent tests see consistent state.
            torch.cuda.memory._set_allocator_settings(
                f"expandable_segments:{EXPANDABLE_SEGMENTS}"
            )

    @skipIfRocm(msg="cudaMallocManaged (UVM) is not supported on ROCm")
    @requires_cuda_python_bindings
    def test_use_uvm(self):
        # [FUNCTION DOC] Inside the unified virtual memory context, creates two CUDA matrices
        # and multiplies them; asserts the result shape and that it is on CUDA. Needs cuda
        # python bindings; skipped on ROCm.
        with torch.cuda._use_uvm():
            x = torch.randn(256, 256, device="cuda")
            y = torch.randn(256, 256, device="cuda")
            z = x @ y
        self.assertEqual(z.shape, (256, 256))
        self.assertTrue(z.is_cuda)

    @skipIfRocm(msg="cudaMallocManaged (UVM) is not supported on ROCm")
    @requires_cuda_python_bindings
    def test_use_uvm_numerics(self):
        # [FUNCTION DOC] Seeds, generates a random tensor inside the UVM context and again
        # outside with the same seed; asserts the values are equal. Needs cuda python bindings;
        # skipped on ROCm.
        torch.manual_seed(42)
        with torch.cuda._use_uvm():
            a_uvm = torch.randn(128, 128, device="cuda")
        torch.manual_seed(42)
        a_reg = torch.randn(128, 128, device="cuda")
        self.assertEqual(a_uvm, a_reg)

    @skipIfRocm(msg="cudaMallocManaged (UVM) is not supported on ROCm")
    @requires_cuda_python_bindings
    def test_use_uvm_backward(self):
        # [FUNCTION DOC] Runs a forward and backward pass of a linear layer inside the UVM
        # context with autograd multithreading disabled; asserts the weight gradient exists with
        # the right shape. Needs cuda python bindings; skipped on ROCm.
        with torch.cuda._use_uvm():
            with torch.autograd.set_multithreading_enabled(False):
                model = torch.nn.Linear(512, 512, device="cuda")
                out = model(torch.randn(16, 512, device="cuda"))
                out.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertEqual(model.weight.grad.shape, (512, 512))

    @skipIfRocm(msg="cudaMallocManaged (UVM) is not supported on ROCm")
    @requires_cuda_python_bindings
    def test_use_uvm_tensor_outlives_context(self):
        # [FUNCTION DOC] Regression test: a tensor allocated in the UVM context may outlive it
        # and be freed later by empty_cache. Loops twice over the default device and, with more
        # than one GPU, device 1, allocating inside the context, checking it is CUDA, deleting
        # it, collecting garbage and emptying the cache; must not crash. Needs cuda python
        # bindings; skipped on ROCm.
        # Regression test: a tensor allocated inside _use_uvm() can outlive the
        # context, leaving its block cached in the PrivatePool until a later global
        # empty_cache() frees it. Before the fix the alloc/free ctypes closures
        # were GC'd by then, so the free dangled and segfaulted. Loop twice and
        # (where available) over a second device to exercise the now-shared
        # allocator across calls/devices.
        devices = [None]
        if torch.cuda.device_count() > 1:
            devices.append(1)
        for _ in range(2):
            for device in devices:
                with torch.cuda._use_uvm(device=device):
                    x = torch.randn(256, 256, device=device or "cuda")
                self.assertTrue(x.is_cuda)
                del x
                gc.collect()
                torch.cuda.empty_cache()


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaOptims(TestCase):
    # [CLASS DOC] Optimizer tests using CUDA graphs and fused implementations. The methods are
    # instantiated for device types with the optimizer database decorator. Skipped without CUDA;
    # wrapped for strict dynamo mode.
    # These tests will be instantiate with instantiate_device_type_tests
    # to apply the new OptimizerInfo structure.

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >=5.3 required for graphs"
    )
    @optims(
        [optim for optim in optim_db if optim.has_capturable_arg],
        dtypes=[torch.float32],
    )
    def test_graph_optims(self, device, dtype, optim_info):
        # [FUNCTION DOC] Device-generic test over optimizers that have a capturable argument
        # (float32). For each input set, trains a control optimizer with capturable off and
        # one with capturable on for 3 warmup and 2 train steps, with and without actual
        # graph capture of the step (replaying with copied gradients), and asserts the
        # parameters match the control. Sets lr 0.1 and betas for Adam types. CUDA only;
        # needs graph support.
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        steps_warmup = 3
        steps_train = 2

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            # lr and betas as a Tensor is not supported when capturable=False and foreach=True for torch.optim.adam
            # and torch.optim.adamw
            kwargs["lr"] = 0.1
            if optim_cls in (torch.optim.Adam, torch.optim.AdamW):
                kwargs["betas"] = (0.9, 0.99)

            for actually_do_graphs in (True, False):
                params = [
                    torch.randn((i + 5, i + 5), device=device) for i in range(2)
                ] + [torch.randn((), device=device)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]

                # Control (capturable=False)
                kwargs["capturable"] = False

                opt = optim_cls(params_control, **kwargs)
                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads[i][j]
                    opt.step()

                # capturable=True
                kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i][j]
                    opt.step()

                if actually_do_graphs:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        opt.step()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads[i + steps_warmup][j])
                        g.replay()
                    else:
                        # Passing capturable=True to the constructor and running without graphs should still be
                        # numerically correct, even if it's not ideal for performance.
                        for j, p in enumerate(params_graphed):
                            p.grad = grads[i + steps_warmup][j]
                        opt.step()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @optims(
        [
            optim
            for optim in optim_db
            if "fused" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_scaling_fused_optimizers(self, device, dtype, optim_info):
        # [FUNCTION DOC] Device-generic test over fused optimizers on CUDA with graphs and
        # gradient scaling. For each input set, with and without graph capture (only without
        # if the optimizer has no capturable option), trains a control optimizer (capturable
        # off) and a graphed one (capturable on) through gradient scaler steps and updates for
        # 3 warmup and 2 train steps using cloned gradients, and asserts the final parameters
        # are equal.
        optim_cls = optim_info.optim_cls

        steps_warmup = 3
        steps_train = 2

        optim_inputs = optim_info.optim_inputs_func(device=device)

        for optim_input in optim_inputs:
            kwargs = optim_input.kwargs
            kwargs["fused"] = True

            for actually_do_graphs in (
                (True, False) if optim_info.has_capturable_arg else (True,)
            ):
                params = [torch.randn((i + 5, i + 5), device=device) for i in range(2)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                # `GradScaler` in-place updates gradients thus it's necessary to duplicate gradients.
                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]
                with torch.no_grad():
                    grads_control = [[g.clone() for g in gs] for gs in grads]
                    grads_graphed = [[g.clone() for g in gs] for gs in grads]

                # Gradient Scaler
                scaler_for_control = torch.cuda.amp.GradScaler(init_scale=128.0)
                with torch.no_grad():
                    scaler_for_control._lazy_init_scale_growth_tracker(device)

                scaler_for_graphed = torch.cuda.amp.GradScaler()
                scaler_for_graphed.load_state_dict(scaler_for_control.state_dict())
                with torch.no_grad():
                    scaler_for_graphed._lazy_init_scale_growth_tracker(device)

                # Control (capturable=False)
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = False
                opt = optim_cls(params_control, **kwargs)

                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads_control[i][j]
                    scaler_for_control.step(opt)
                    scaler_for_control.update()

                # capturable=True
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads_graphed[i][j]
                    scaler_for_graphed.step(opt)
                    scaler_for_graphed.update()

                if actually_do_graphs:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads_graphed[i + steps_warmup][j])
                        g.replay()
                    else:
                        # Passing capturable=True to the constructor and running without graphs should still be
                        # numerically correct, even if it's not ideal for performance.
                        for j, p in enumerate(params_graphed):
                            p.grad = grads_graphed[i + steps_warmup][j]
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    # This test cases compares autocast scaling on a fused optim vs autocast scaling
    # on a forloop optim.
    @onlyNativeDeviceTypes
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_grad_scaling_autocast_fused_optimizers(self, device, dtype, optim_info):
        # [FUNCTION DOC] Compares autocast plus gradient scaling on a fused optimizer against
        # the for-loop implementation, for each optimizer input and with or without a separate
        # unscale step. Trains matching models on shared data, and at every step uses a tensor
        # tracker to check the loss, gradients, parameters and optimizer state agree (looser
        # tolerances for second-moment state and parameters when betas are floats). Skips if the
        # device does not support fused for that optimizer.
        device = device.split(":")[0]
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            for _separate_unscale in (True, False):
                kwargs = optim_input.kwargs
                kwargs["fused"] = True
                torch.manual_seed(20)
                (
                    mod_control,
                    mod_scaling,
                    opt_control,
                    opt_scaling,
                    data,
                    loss_fn,
                    _,
                ) = _create_scaling_case(
                    optimizer_ctor=optim_cls, optimizer_kwargs=kwargs, device=device
                )
                optimizer_kwargs = deepcopy(kwargs)
                optimizer_kwargs["fused"] = False
                if "lr" not in kwargs:
                    # _create_scaling_case will set lr = 1.0 if optimizer_kwargs do not set lr
                    optimizer_kwargs["lr"] = 1.0
                opt_control = optim_cls(mod_control.parameters(), **optimizer_kwargs)
                scaler_scaling = torch.amp.GradScaler(device, init_scale=128.0)
                scaler_control = torch.amp.GradScaler(device, init_scale=128.0)

                tracker = TensorTracker()
                # Increase the tolerance for param and (max_)exp_avg_sq when betas aren't tensors
                # cuz the discrepancy between double vs float betas becomes too big. When Tensor
                # betas are used, fused and forloop both would have float betas and the default
                # tolerances are fine.
                assert_eq_kwargs = {}
                if "betas" in opt_control.param_groups[0] and not torch.is_tensor(
                    opt_control.param_groups[0]["betas"][0]
                ):
                    assert_eq_kwargs = {"atol": 2e-5, "rtol": 2e-5}
                for input, target in data:
                    opt_control.zero_grad()
                    with torch.autocast(device_type=device, dtype=torch.half):
                        output_control = mod_control(input)
                        loss_control = loss_fn(output_control, target)
                    scaler_control.scale(loss_control).backward()
                    scaler_control.step(opt_control)
                    scaler_control.update()

                    opt_scaling.zero_grad()
                    with torch.autocast(device_type=device, dtype=torch.half):
                        output_scaling = mod_scaling(input)
                        loss_scaling = loss_fn(output_scaling, target)
                    scaler_scaling.scale(loss_scaling).backward()
                    if _separate_unscale:
                        scaler_scaling.unscale_(opt_scaling)
                    scaler_scaling.step(opt_scaling)
                    scaler_scaling.update()

                    tracker.add(loss_control)
                    tracker.pop_check_set(loss_scaling, self)
                    for param_control, param_scaling in zip(
                        mod_control.parameters(), mod_scaling.parameters()
                    ):
                        tracker.add(param_control.grad)
                        tracker.pop_check_set(param_scaling.grad, self)
                        tracker.add(param_control)
                        tracker.pop_check_set(param_scaling, self, assert_eq_kwargs)

                        state_control, state_scaling = (
                            opt_control.state[param_control],
                            opt_scaling.state[param_scaling],
                        )

                        for k in state_control:
                            actual = state_scaling[k]
                            if k == "step":
                                actual = actual.squeeze()
                            tracker.add(state_control[k])
                            tracker.pop_check_set(
                                actual,
                                self,
                                assert_eq_kwargs
                                if k == "exp_avg_sq" or k == "max_exp_avg_sq"
                                else {},
                            )

    @onlyCUDA
    @parametrize("in_place_unscale", [False, True])
    @optims(
        [optim for optim in optim_db if "cuda" in optim.supports_fused_on],
        dtypes=[torch.float32],
    )
    def test_grad_scaler_with_preset_grad_scale(
        self, device, dtype, optim_info, in_place_unscale
    ):
        # [FUNCTION DOC] Parametrized by whether to unscale in place. With a fused optimizer,
        # sets a gradient of 15 on a weight, scaler scale 5, optionally unscales (gradient
        # becomes 3), sets a user grad_scale of 3 on the optimizer and steps; the gradient must
        # then equal 1 (divided by 5 times 3 in total). CUDA only.
        weight = torch.ones((5, 5), device="cuda", requires_grad=True)
        weight.grad = torch.full_like(weight, fill_value=15)
        opt = optim_info.optim_cls([weight], lr=0.1, fused=True)
        scaler = torch.amp.GradScaler(init_scale=5)

        # simulate scaling a loss
        scaler.scale(torch.ones(5))

        if in_place_unscale:
            scaler.unscale_(opt)
            # the gradient should have been divided in-place
            self.assertEqual(weight.grad, torch.full_like(weight, fill_value=3))

        # the user sets a `grad_scale` value which should be fused with the optimizer step
        opt.grad_scale = torch.Tensor([3]).cuda()
        scaler.step(opt)

        # check that the user's grad_scale was respected (i.e. the gradient was divided by 5 * 3)
        self.assertEqual(weight.grad, torch.full_like(weight, fill_value=1))

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize("foreach, fused", [(False, False), (True, False), (False, True)])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_grad_scaling(self, device, dtype, optim_info, foreach, fused):
        # [FUNCTION DOC] Parametrized by foreach and fused flags. Captures a half-precision
        # forward and scaled backward in a CUDA graph, then replays with four inputs; checks
        # after each replay the weight gradient (scaled value or infinity), steps the scaler and
        # optimizer, and asserts the scaler scale and growth tracker follow the expected
        # sequence (scale 4, 2, 2, 1; tracker 1, 0, 1, 0).
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler(device="cuda", init_scale=4.0)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        weight = torch.ones((100,), device="cuda", requires_grad=True)
        opt = optim_info.optim_cls([weight], lr=0.1, foreach=foreach, fused=fused)
        static_input = torch.ones_like(weight)
        static_grad = torch.ones_like(weight)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
        torch.cuda.current_stream().wait_stream(s)

        opt.zero_grad(set_to_none=True)

        # capture
        with torch.cuda.stream(s):
            g.capture_begin()
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
            g.capture_end()

        input_vals = [5, 20000, 5, 40000]
        # If the scale gets updated properly, these are the scale, growth tracker,
        # and grad values we expect.
        expected_scales = [4, 2, 2, 1]
        expected_growth_trackers = [1, 0, 1, 0]
        expected_grad_vals = [5 * 4, float("inf"), 5 * 2, float("inf")]

        for data, scale, growth_tracker, grad_val in zip(
            input_vals, expected_scales, expected_growth_trackers, expected_grad_vals
        ):
            static_input.fill_(data)
            g.replay()
            self.assertEqual(weight.grad, torch.full_like(weight.grad, grad_val))
            scaler.step(opt)
            scaler.update()
            self.assertEqual(scaler._scale, scale)
            self.assertEqual(scaler._growth_tracker, growth_tracker)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestGDS(TestCase):
    # [CLASS DOC] Tests for GPUDirect Storage file transfers through torch.cuda.gds. Skipped
    # without CUDA; individual tests skip when GDS is not built in or the temp directory is
    # not on a local ext4 or xfs filesystem. Helpers detect the filesystem type and enforce
    # the requirements.
    def _get_tmp_dir_fs_type(self):
        # [FUNCTION DOC] HELPER: Finds the filesystem type of the partition mounted at the temp
        # directory real path; if it is not found returns the type of the root mount (empty
        # string when unknown).
        my_path = os.path.realpath(tempfile.gettempdir())
        root_type = ""
        for part in psutil.disk_partitions():
            if part.mountpoint == "/":
                root_type = part.fstype
                continue
            if part.mountpoint == my_path:
                return part.fstype
        return root_type

    def _require_gds(self):
        # [FUNCTION DOC] HELPER: Skips the calling test unless GDS is available and the temp
        # directory filesystem is ext4 or xfs.
        # GDS needs the cuFile/hipFile bindings built in, plus an O_DIRECT-capable
        # local filesystem (ext4/xfs) for the temp file the transfer targets.
        if not torch.cuda.gds.is_available():
            self.skipTest("GDS (cuFile/hipFile) not built into this install")
        if self._get_tmp_dir_fs_type() not in ("ext4", "xfs"):
            self.skipTest("GPUDirect Storage requires ext4/xfs for local filesystem")

    def test_gds_is_available(self):
        # [FUNCTION DOC] Asserts the GDS availability check can be called and returns a bool.
        # is_available() must be callable and return a bool even when GDS is off.
        self.assertIsInstance(torch.cuda.gds.is_available(), bool)

    def test_gds_read_write_tensors(self):
        # [FUNCTION DOC] Registers two CUDA tensors as GDS buffers, writes both to one temporary
        # file at different offsets through a GdsFile, reads them back into new tensors and
        # asserts equality; deregisters buffers in finally. Skips when GDS unavailable.
        # Register buffers, pack two tensors at different offsets, read them back.
        self._require_gds()
        src1 = torch.randn(1024, device="cuda")
        src2 = torch.randn(2, 1024, device="cuda")
        torch.cuda.gds.gds_register_buffer(src1.untyped_storage())
        torch.cuda.gds.gds_register_buffer(src2.untyped_storage())
        dest1 = torch.empty(1024, device="cuda")
        dest2 = torch.empty(2, 1024, device="cuda")
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
                file.save_storage(src1.untyped_storage(), offset=0)
                file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
                file.load_storage(dest1.untyped_storage(), offset=0)
                file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
            self.assertEqual(src1, dest1)
            self.assertEqual(src2, dest2)
        finally:
            torch.cuda.gds.gds_deregister_buffer(src1.untyped_storage())
            torch.cuda.gds.gds_deregister_buffer(src2.untyped_storage())

    def test_gds_read_write_without_register_buffer(self):
        # [FUNCTION DOC] Round-trips a CUDA tensor through a GdsFile without registering
        # buffers (registering is only an optimization) and asserts equality. Skips when
        # GDS unavailable.
        # gds_register_buffer is a performance optimization, not required for I/O.
        self._require_gds()
        src = torch.arange(4096, device="cuda", dtype=torch.float32)
        dest = torch.empty_like(src)
        with TemporaryFileName() as f:
            file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
            file.save_storage(src.untyped_storage())
            file.load_storage(dest.untyped_storage())
        self.assertEqual(src, dest)

    @parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.uint8])
    def test_gds_dtypes(self, dtype):
        # [FUNCTION DOC] Parametrized over float32, float64, int64 and uint8: registers the
        # source buffer, saves it to a temporary GDS file and loads it back into an empty
        # tensor, asserting the values round-trip (raw byte copy). Deregisters in finally. Skips
        # when GDS unavailable.
        # The transfer is a raw-byte copy, so it must round-trip any dtype.
        self._require_gds()
        if dtype.is_floating_point:
            src = torch.randn(4096, device="cuda", dtype=dtype)
        else:
            src = torch.randint(0, 128, (4096,), device="cuda", dtype=dtype)
        dest = torch.empty_like(src)
        torch.cuda.gds.gds_register_buffer(src.untyped_storage())
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
                file.save_storage(src.untyped_storage())
                file.load_storage(dest.untyped_storage())
            self.assertEqual(src, dest)
        finally:
            torch.cuda.gds.gds_deregister_buffer(src.untyped_storage())

    def test_gds_handle_register_deregister(self):
        # [FUNCTION DOC] Tests the GdsFile handle lifecycle: after construction the handle is
        # set; registering again, deregistering twice, saving or loading without a registered
        # handle raise AssertionError with specific messages; deregistering sets the handle to
        # None and re-registering restores the file for use. Skips when GDS unavailable.
        # Explicit handle lifecycle plus the guard assertions in GdsFile.
        self._require_gds()
        storage = torch.randn(1024, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
            # The constructor already registered the handle.
            self.assertIsNotNone(file.handle)
            with self.assertRaisesRegex(
                AssertionError, "Cannot register a handle that is already registered"
            ):
                file.register_handle()
            file.deregister_handle()
            self.assertIsNone(file.handle)
            # Operations without a registered handle are rejected.
            with self.assertRaisesRegex(
                AssertionError, "Cannot deregister a handle that is not registered"
            ):
                file.deregister_handle()
            with self.assertRaisesRegex(
                AssertionError, "Cannot save data to a file that is not registered"
            ):
                file.save_storage(storage)
            with self.assertRaisesRegex(
                AssertionError, "Cannot load data from a file that is not registered"
            ):
                file.load_storage(storage)
            # Re-registering restores usability.
            file.register_handle()
            self.assertIsNotNone(file.handle)
            file.save_storage(storage)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestCudaAutocast(TestAutocast):
    # [CLASS DOC] CUDA specialization of the shared autocast test base class. setUp builds the
    # autocast op lists for device cuda:0 and tearDown drops them. Test methods run lists of ops
    # under autocast (fp16, bf16, fp32 policy, promotion, banned ops), custom function
    # decorators, RNNs, caching and checkpointing. Skipped without CUDA.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp then stores the autocast test op lists
        # for cuda:0.
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device("cuda:0"))

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Deletes the stored op lists and calls the base tearDown.
        del self.autocast_lists
        super().tearDown()

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_fp16(self):
        # [FUNCTION DOC] With deterministic cudnn, runs every op of the fp16 torch list
        # (skipping entries flagged for skipping on ROCm) through the out-of-place autocast
        # check expecting half precision results. Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                if not skip_test:
                    self._run_autocast_outofplace(
                        op, args, torch.float16, device="cuda", amp_dtype=torch.float16
                    )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_bf16(self):
        # [FUNCTION DOC] With deterministic cudnn, runs the fp16 op list under bfloat16
        # autocast. Ops with cudnn in their name expect a RuntimeError when the cudnn v8 API is
        # disabled or compute capability is below 8.0; otherwise they run if bf16 is supported,
        # else a RuntimeError saying bfloat16 is unsupported is expected. Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                should_error_from_cudnn = "cudnn" in op and (
                    "TORCH_CUDNN_V8_API_DISABLED" in os.environ
                    and int(os.environ["TORCH_CUDNN_V8_API_DISABLED"])
                    or torch.cuda.get_device_capability() < (8, 0)
                )
                should_error_from_not_implemented = should_error_from_cudnn
                if not skip_test:
                    if should_error_from_not_implemented:
                        with self.assertRaises(
                            RuntimeError,
                            msg=str(op) + " should not be supported for bfloat16!",
                        ):
                            self._run_autocast_outofplace(
                                op, args, torch.bfloat16, device="cuda"
                            )
                    else:
                        if torch.cuda.is_bf16_supported():
                            self._run_autocast_outofplace(
                                op, args, torch.bfloat16, device="cuda"
                            )
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "Device does not support bfloat16"
                            ):
                                self._run_autocast_outofplace(
                                    op, args, torch.bfloat16, device="cuda"
                                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_fp32(self):
        # [FUNCTION DOC] Runs each op of the fp32 policy list (with optional keyword arguments)
        # through the out-of-place autocast check expecting float32 results under half precision
        # autocast. Needs cudnn.
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_need_autocast_promote(self):
        # [FUNCTION DOC] Runs the ops that need autocast promotion through the out-of-place
        # check expecting float32 under half precision autocast. Needs cudnn.
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(
                op, args, torch.float32, device="cuda", amp_dtype=torch.float16
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_expect_builtin_promote(self):
        # [FUNCTION DOC] Runs ops expected to follow builtin type promotion, with their expected
        # output type, through the out-of-place autocast check. Needs cudnn.
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_fp16(self):
        # [FUNCTION DOC] With deterministic cudnn, runs the nn functional fp16 ops from
        # the internal nn module through the out-of-place check expecting half precision.
        # Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.nn_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=torch._C._nn,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_bf16(self):
        # [FUNCTION DOC] Runs the nn fp16 op list under bfloat16 autocast: if bf16 is
        # supported checks results, else expects RuntimeError (device does not support
        # bfloat16). Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.nn_fp16:
                if torch.cuda.is_bf16_supported():
                    self._run_autocast_outofplace(
                        op, args, torch.bfloat16, device="cuda", module=torch._C._nn
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "Device does not support bfloat16"
                    ):
                        self._run_autocast_outofplace(
                            op, args, torch.bfloat16, device="cuda", module=torch._C._nn
                        )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_fp32(self):
        # [FUNCTION DOC] Runs the nn fp32 op list through the out-of-place check expecting
        # float32 outputs under half precision autocast. Needs cudnn.
        for op, args in self.autocast_lists.nn_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=torch._C._nn,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_linalg_fp16(self):
        # [FUNCTION DOC] With deterministic cudnn, runs the linalg fp16 ops through the
        # out-of-place check expecting half precision. Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.linalg_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=torch._C._linalg,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_fp16(self):
        # [FUNCTION DOC] With deterministic cudnn, runs the tensor-method fp16 ops through the
        # out-of-place check expecting half precision. Needs cudnn.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.methods_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=None,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_fp32(self):
        # [FUNCTION DOC] Runs the tensor-method fp32 ops through the out-of-place check
        # expecting float32 under half precision autocast. Needs cudnn.
        for op, args in self.autocast_lists.methods_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=None,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_expect_builtin_promote(self):
        # [FUNCTION DOC] Runs the tensor-method ops expected to follow builtin promotion, with
        # their expected output types, through the out-of-place check. Needs cudnn.
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    def test_autocast_banned(self):
        # [FUNCTION DOC] Under CUDA autocast, each op in the banned list (with its module and
        # arguments) must raise RuntimeError.
        with torch.autocast("cuda"):
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    def test_autocast_ignored_types(self):
        # [FUNCTION DOC] Under CUDA autocast, checks double and int32 inputs are ignored by cast
        # policies: a double matrix product with half raises RuntimeError, and the result dtype
        # of mm, pow, sum and norm (where applicable) with autocast on equals the dtype with
        # autocast disabled.
        with torch.autocast("cuda"):
            for ignore_type in (torch.double, torch.int32):
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                c_16 = torch.ones((8, 8), dtype=torch.float16, device="cuda:0")

                # Tests if CastPolicy::fp16 ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with self.assertRaises(RuntimeError):
                        torch.mm(a_ignore, c_16)
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    self.assertTrue(
                        torch.mm(a_ignore, b_ignore).dtype is type_no_autocast
                    )

                # Tests if CastPolicy::fp32 ops ignore double and int
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.pow(a_ignore, 2.0).dtype
                self.assertTrue(torch.pow(a_ignore, 2.0).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_set_opt_dtype ops ignore double and int
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_append_dtype ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)

    def test_autocast_custom_enabled(self):
        # [FUNCTION DOC] Tests custom autograd function with the custom forward and backward
        # decorators and no input casting. Under autocast for half and (if supported) bfloat16,
        # forward receives float32 inputs with autocast enabled, the output has the autocast
        # dtype, and backward runs with autocast enabled and produces gradients in that dtype.
        class MyMM(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function doing a matrix product; forward asserts
            # float32 inputs and autocast enabled and saves inputs; backward asserts autocast
            # enabled and gradients have the loop dtype of the enclosing test.
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Asserts both inputs are float32 and autocast is
                # enabled, saves them and returns their matrix product.
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Asserts autocast is enabled, computes the two input
                # gradients with matrix products and asserts their dtype equals the enclosing
                # loop dtype; returns them.
                self.assertTrue(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                a_grad, b_grad = grad.mm(b.t()), a.t().mm(grad)
                self.assertTrue(a_grad.dtype is dtype and b_grad.dtype is dtype)
                return a_grad, b_grad

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)

        dtypes = (torch.float16, torch.bfloat16) if TEST_BF16 else (torch.float16,)
        for dtype in dtypes:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = mymm(x, y)
                self.assertTrue(output.dtype is dtype)
                loss = output.sum()
            loss.backward()

    def test_autocast_custom_cast_inputs(self):
        # [FUNCTION DOC] Tests custom forward decorator with input casting to float32 (including
        # a tensor inside a nested container): under autocast the inputs are cast to float32 and
        # autocast is disabled inside, the output is float32 and backward runs; outside autocast
        # the decorator is a no-op so half inputs stay half.
        class MyMM(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function doing a matrix product where the second
            # operand comes from a nested container; returns no gradients for the
            # non-tensor arguments.
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                # [FUNCTION DOC] HELPER: Takes the second operand from the nested container,
                # asserts both operands have the expected type and autocast is disabled, saves
                # them and returns their matrix product.
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Asserts autocast is disabled and returns the gradient
                # times the transposed saved operand plus None for the two other arguments.
                self.assertFalse(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float16, requires_grad=True)
        # Puts one input tensor in a nested container.  y's contained Tensor won't receive a gradient,
        # because torch.autograd.Function can't hand gradients back to non-Tensor forward arguments.
        # Sets requires_grad=False explicitly so we don't lie about expecting a gradient.
        y = (
            0,
            {
                0: torch.randn(
                    (8, 8), device="cuda", dtype=torch.float16, requires_grad=False
                )
            },
        )

        with torch.autocast("cuda"):
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()

    def test_autocast_custom_deprecated_warning(self):
        # [FUNCTION DOC] Defines a custom function with the deprecated cuda amp decorators while
        # recording warnings and asserts the two deprecation messages (forward then backward
        # decorators) were issued; then runs forward and backward under autocast to make sure it
        # still works.
        with warnings.catch_warnings(record=True) as w:

            class MyMM(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd function adding two tensors, decorated
                # with the deprecated cuda amp forward (casting inputs to float32) and
                # backward decorators.
                @staticmethod
                @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
                def forward(ctx, x, y):
                    # [FUNCTION DOC] HELPER: Saves inputs, asserts autocast is disabled inside
                    # and returns their sum.
                    ctx.save_for_backward(x, y)
                    self.assertFalse(torch.is_autocast_enabled())
                    return x + y

                @staticmethod
                @torch.cuda.amp.custom_bwd
                def backward(ctx, grad):
                    # [FUNCTION DOC] HELPER: Asserts autocast is disabled and returns the
                    # gradient for both inputs.
                    _, _ = ctx.saved_tensors
                    self.assertFalse(torch.is_autocast_enabled())
                    return grad, grad

        self.assertRegex(
            str(w[0].message), r"`torch.cuda.amp.custom_fwd\(args...\)` is deprecated."
        )
        self.assertRegex(
            str(w[1].message), r"`torch.cuda.amp.custom_bwd\(args...\)` is deprecated."
        )

        mymm = MyMM.apply
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        with torch.amp.autocast("cuda"):
            output = mymm(x, y)
            loss = output.sum()
        loss.backward()

    def test_autocast_cat_jit(self):
        # [FUNCTION DOC] Regression test (issue 38958): runs a module calling cat and stack on
        # CPU tensors both eagerly and as a scripted module under CUDA autocast, to exercise the
        # boxed call path. Passes if no error is raised.
        # Reported at https://github.com/pytorch/pytorch/issues/38958

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs whose forward creates two random
            # one-element tensors, concatenates them and stacks the result twice.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the stack of two copies of the concatenation of
                # two random one-element tensors.
                a = torch.randn(1)
                b = torch.randn(1)
                c = torch.cat((a, b), 0)
                d = torch.stack([c, c], 0)
                return d

        # The JIT here doesn't really matter, we just need to call
        # cat via the boxed API
        model = Model()
        model_jit_script = torch.jit.script(model)

        with torch.autocast("cuda", enabled=True):
            model()
            model_jit_script()

    # cudnn RNNs require special backend handling (weights are cast to FP16 and reflattened)
    # so they get a dedicated test.
    # Despite the large number of RNN cases it tries, the test takes < 15 seconds on a Titan V (similar to V100).
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_rnn(self):
        # [FUNCTION DOC] Exhaustively tests cudnn RNNs under autocast (needs cudnn): for RNN,
        # GRU and LSTM, layer counts, bias, layouts (seq-first, batch-first, packed),
        # bidirectional, pre-flattened or not, and input, hidden and weight dtypes. Autocast
        # output must be half, have the cudnn backward node, and equal (outputs, final hidden
        # state and gradients) a control run done by converting the module and inputs to half.
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # seq, batch, features, hidden size
            clses = ("RNN", "GRU", "LSTM")
            T, B, F, H = 3, 4, 5, 6
            dtypes = (torch.float16, torch.float32)
            input_layouts = ("seq_first", "batch_first", "packed")

            for (
                cls,
                num_layers,
                bias,
                input_layout,
                bidirectional,
                try_nonpreflattened_weights,
                input_dtype,
                hidden_dtype,
                weight_dtype,
            ) in product(
                clses,
                (1, 2),
                (True, False),
                input_layouts,
                (True, False),
                (True, False),
                dtypes,
                dtypes,
                dtypes,
            ):
                if input_layout == "seq_first":
                    batch_first = False
                    x = torch.randn((T, B, F), device="cuda", dtype=input_dtype)
                elif input_layout == "batch_first":
                    batch_first = True
                    x = torch.randn((B, T, F), device="cuda", dtype=input_dtype)
                elif input_layout == "packed":
                    batch_first = False
                    x = torch.nn.utils.rnn.pack_padded_sequence(
                        torch.randn((T, B, F), device="cuda", dtype=input_dtype),
                        lengths=(3, 2, 1, 3),
                        enforce_sorted=False,
                    )

                rnn = (
                    getattr(torch.nn, cls)(
                        F,
                        H,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        bias=bias,
                        batch_first=batch_first,
                    )
                    .cuda()
                    .to(dtype=weight_dtype)
                )

                if try_nonpreflattened_weights:
                    for p in rnn.parameters():
                        with torch.no_grad():
                            p.set_(p.clone())

                h = torch.randn(
                    (num_layers * (2 if bidirectional else 1), B, H),
                    device="cuda",
                    dtype=hidden_dtype,
                )
                if cls == "LSTM":
                    c = torch.randn(
                        (num_layers * (2 if bidirectional else 1), B, H),
                        device="cuda",
                        dtype=hidden_dtype,
                    )
                    h = (h, c)

                with torch.autocast("cuda"):
                    out, h_out = rnn(x, h)
                out = out.data if input_layout == "packed" else out
                self.assertEqual(out.dtype, torch.float16)
                # Autocast wrapper requires at::_cudnn_rnn is autograd-exposed.  This check can't guarantee
                # at::_cudnn_rnn is autograd-exposed, but if it fires, it indicates some funny business has
                # occurred and we should double check that at::_cudnn_rnn remains autograd-exposed.
                self.assertEqual(
                    out.grad_fn.name(),
                    "MiopenRnnBackward0" if torch.version.hip else "CudnnRnnBackward0",
                )
                out.sum().backward()
                grads = [p.grad.clone() for p in rnn.parameters()]

                rnn.zero_grad()

                if cls == "LSTM":
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(
                        x.half(), (h[0].half(), h[1].half())
                    )
                else:
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(
                        x.half(), h.half()
                    )
                out_control = (
                    out_control.data if input_layout == "packed" else out_control
                )
                out_control.sum().backward()
                grads_control = [p.grad.clone() for p in rnn.parameters()]

                # Compares with default tolerances, even for FP16 execution.  Barring nondeterminism,
                # autocast and control results should be bitwise identical.
                self.assertEqual(out, out_control)

                if cls == "LSTM":
                    self.assertTrue(
                        h_out[0].dtype is torch.float16
                        and h_out[1].dtype is torch.float16
                    )
                    self.assertEqual(h_out[0], h_out_control[0])
                    self.assertEqual(h_out[1], h_out_control[1])
                else:
                    self.assertEqual(h_out.dtype, torch.float16)
                    self.assertEqual(h_out, h_out_control)
                for grad, grad_control in zip(grads, grads_control):
                    self.assertEqual(grad.half(), grad_control)

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_rnn_packed_sequence_batch_sizes_must_be_cpu(self):
        # [FUNCTION DOC] Verifies calling the relu RNN op with a CUDA batch-sizes tensor raises
        # RuntimeError saying the batch sizes tensor must be on CPU (instead of crashing).
        # Needs cudnn.
        # Verify that passing batch_sizes on CUDA raises an error instead of segfaulting
        data = torch.randn([18, 16], dtype=torch.float32, device="cuda")
        batch_sizes = torch.tensor([8, 6, 4], dtype=torch.int64, device="cuda")
        hx = torch.randn([1, 8, 32], dtype=torch.float32, device="cuda")
        params = [
            torch.randn([32, 16], dtype=torch.float32, device="cuda"),
            torch.randn([32, 32], dtype=torch.float32, device="cuda"),
        ]

        with self.assertRaisesRegex(
            RuntimeError, "batch_sizes tensor should be on CPU"
        ):
            torch.ops.aten.rnn_relu(
                data,
                batch_sizes=batch_sizes,
                hx=hx,
                params=params,
                has_biases=False,
                num_layers=1,
                dropout=0.5,
                train=False,
                bidirectional=False,
            )

    @serialTest()
    def test_autocast_cache_leak(self):
        # [FUNCTION DOC] Regression test (issue 48049) that autocast does not recache weights in
        # a no-grad block: after the first forward under autocast and no_grad the allocated
        # memory must be the same after three more forwards. Serial.
        # Reported at https://github.com/pytorch/pytorch/issues/48049
        # Test is used to check, if autocast recaches the same parameters
        # when executed in a `torch.no_grad()` block.

        linear = torch.nn.Linear(10, 10).to("cuda")
        data = torch.randn(1, 10, device="cuda")

        with torch.autocast("cuda"):
            with torch.no_grad():
                out = linear(data)
                first_iter_mem = torch.cuda.memory_allocated()
                for _ in range(3):
                    out = linear(data)
                self.assertEqual(first_iter_mem, torch.cuda.memory_allocated())

    def test_autocast_checkpointing(self):
        # [FUNCTION DOC] Runs checkpoint_sequential on a three-linear-layer half precision input
        # under autocast for reentrant and non-reentrant checkpointing; the output must require
        # grad and be half; runs backward.
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).cuda()
        input = torch.rand(
            (8, 8), device="cuda", dtype=torch.float16, requires_grad=True
        )
        for reentrant in (True, False):
            with torch.autocast("cuda"):
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            self.assertTrue(output.requires_grad)
            self.assertTrue(output.dtype is torch.float16)
            output.sum().backward()

    def test_cuda_autocast_deprecated_warning(self):
        # [FUNCTION DOC] Asserts constructing the deprecated cuda amp autocast raises
        # FutureWarning telling the user to use the generic autocast with cuda.
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cuda.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cuda', args...\)` instead.",
        ):
            with torch.cuda.amp.autocast():
                _ = torch.ones(10)


class TestCompileKernel(TestCase):
    # [CLASS DOC] Tests for the runtime compilation of CUDA kernels via
    # torch.cuda._compile_kernel. Test methods skip when CUDA is unavailable; no
    # class-level state.
    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel(self):
        # [FUNCTION DOC] Compiles a vector addition kernel from source with the runtime compile
        # helper and launches it on 1024 float tensors, checking the sum; does the same for an
        # int32 version; compiles a kernel using a macro and a device helper function (scale by
        # 2) and checks the result; finally checks that compiling invalid source raises
        # RuntimeError. Skipped without CUDA.
        # Simple vector addition kernel
        kernel_source = """
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        """

        # Compile the kernel
        from torch.cuda import _compile_kernel

        add_kernel = _compile_kernel(kernel_source, "add_tensors")

        # Prepare data
        N = 1024
        a = torch.rand(N, device="cuda")
        b = torch.rand(N, device="cuda")
        c = torch.empty_like(a)

        # Calculate grid and block dimensions
        threads_per_block = 256
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

        # Launch kernel
        add_kernel(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[a, b, c, N],
        )

        # Verify results
        expected = a + b
        self.assertEqual(c, expected)

        # Test with different tensor types
        a_int = torch.randint(0, 100, (N,), device="cuda", dtype=torch.int32)
        b_int = torch.randint(0, 100, (N,), device="cuda", dtype=torch.int32)
        c_int = torch.empty_like(a_int)

        # Integer addition kernel
        int_kernel_source = """
        __global__ void add_int_tensors(const int* a, const int* b, int* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        """
        from torch.cuda import _compile_kernel

        add_int_kernel = _compile_kernel(int_kernel_source, "add_int_tensors")

        # Launch kernel
        add_int_kernel(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[a_int, b_int, c_int, N],
        )

        # Verify results
        expected_int = a_int + b_int
        self.assertEqual(c_int, expected_int)

        # Test with header code
        scale_kernel_source = """
        #define SCALE_FACTOR 2.0f

        __device__ float scale_value(float val) {
            return val * SCALE_FACTOR;
        }

        __global__ void scale_tensors(const float* input, float* output, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                output[i] = scale_value(input[i]);
        }
        """

        scale_kernel = _compile_kernel(scale_kernel_source, "scale_tensors")

        input_tensor = torch.rand(N, device="cuda")
        output_tensor = torch.empty_like(input_tensor)

        scale_kernel(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[input_tensor, output_tensor, N],
        )

        # Verify scaling
        expected_scaled = input_tensor * 2.0
        self.assertEqual(output_tensor, expected_scaled)

        # Test error handling with invalid kernel
        invalid_kernel_source = """
        __global__ void invalid_kernel(float* a) {
            undeclared_variable = 10; // This will cause a compilation error
        }
        """

        with self.assertRaises(RuntimeError):
            _compile_kernel(invalid_kernel_source, "invalid_kernel")

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_large_shared_memory(self):
        # [FUNCTION DOC] Compiles a shared-memory reduction kernel, requests 64 KiB of dynamic
        # shared memory with the kernel config setter, launches 4 blocks of 1024 threads over
        # 4096 ones and checks each block sum is 1024. Then requests twice the device maximum
        # shared memory (AMD limits per arch, NVIDIA opt-in limit) and expects RuntimeError.
        # Skipped without CUDA.
        kernel_source = """
        __global__ void large_shared_memory_kernel(const float* input, float* output, int n) {
            extern __shared__ float shared_data[];

            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Load data into shared memory
            if (idx < n) {
                shared_data[tid] = input[idx];
            } else {
                shared_data[tid] = 0.0f;
            }
            __syncthreads();

            // Perform reduction in shared memory
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    shared_data[tid] += shared_data[tid + stride];
                }
                __syncthreads();
            }

            // Write result
            if (tid == 0) {
                output[blockIdx.x] = shared_data[0];
            }
        }
        """

        from torch.cuda import _compile_kernel, get_device_properties

        kernel = _compile_kernel(kernel_source, "large_shared_memory_kernel")

        threads_per_block = 1024  # 1024 threads * 4 bytes = 4KB, but we'll request 64KB
        shared_mem_size = 64 * 1024  # 64KB

        kernel.set_shared_memory_config(shared_mem_size)

        N = 4096
        input_data = torch.ones(N, device="cuda", dtype=torch.float32)
        output_data = torch.zeros(4, device="cuda", dtype=torch.float32)  # 4 blocks

        kernel(
            grid=(4, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[input_data, output_data, N],
            shared_mem=shared_mem_size,
        )

        # Each block should sum 1024 ones = 1024
        expected = torch.full((4,), 1024.0, dtype=torch.float32)
        self.assertEqual(output_data.cpu(), expected)

        # Test error handling with more than supported shared memory size
        if torch.version.hip:
            gcn_arch = get_device_properties().gcnArchName.split(":", 1)[0]
            if gcn_arch == "gfx1250":
                max_smem = 320 * 1024
            elif gcn_arch == "gfx950":
                max_smem = 160 * 1024
            else:
                max_smem = 65536
        else:
            max_smem = get_device_properties().shared_memory_per_block_optin
        excessive_shared_mem = max_smem * 2

        with self.assertRaises(RuntimeError):
            kernel.set_shared_memory_config(excessive_shared_mem)

    # AMD CDNA3 XF32 (v_mfma_f32_*_xf32) uses round-down accumulation, which
    # produces a negative-biased error ~-5e-3 with max_abs ~8e-3 at K=32 for
    # uniform[0,1] operands — exceeding the 0.005 tolerance that NVIDIA TF32
    # comfortably meets (~2e-3). This is the documented behavior of CDNA3's
    # matrix instructions (arXiv:2511.10909 §IV-F), not a hipBLASLt bug.
    # See https://github.com/jeffdaily/tf32_analysis.
    @tf32_on_and_off(0.01 if TEST_WITH_ROCM else 0.005)
    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_advanced(self):
        # [FUNCTION DOC] Compiles a naive matrix multiplication kernel and checks the output of
        # a 64 by 32 times 32 by 48 product against the framework result, with TF32 tolerance
        # wrapper; then recompiles with an explicit compute capability (major/minor digits on
        # CUDA, the gcn arch name on ROCm) and checks again. Skipped without CUDA.
        # Test matrix multiplication
        matmul_kernel_source = """
        __global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N) {
                float sum = 0.0f;
                for (int i = 0; i < K; i++) {
                    sum += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
        """
        from torch.cuda import _compile_kernel

        matmul_kernel = _compile_kernel(matmul_kernel_source, "matrix_multiply")

        # Matrix dimensions
        M, K, N = 64, 32, 48

        # Create matrices
        A = torch.rand((M, K), device="cuda")
        B = torch.rand((K, N), device="cuda")
        C = torch.zeros((M, N), device="cuda")

        # Calculate grid and block dimensions
        block_dim = (16, 16, 1)
        grid_dim = (
            (N + block_dim[0] - 1) // block_dim[0],
            (M + block_dim[1] - 1) // block_dim[1],
            1,
        )

        # Launch kernel
        matmul_kernel(
            grid=grid_dim,
            block=block_dim,
            args=[A.contiguous(), B.contiguous(), C, M, N, K],
        )

        # Verify results
        expected = torch.matmul(A, B)
        self.assertEqual(C, expected)

        # Test with different compute capability if specified
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if not torch.version.hip:
            compute_cap = f"{device_props.major}{device_props.minor}"
        else:
            compute_cap = f"{device_props.gcnArchName}"

        # Recompile with explicit compute capability
        matmul_kernel_explicit = _compile_kernel(
            matmul_kernel_source, "matrix_multiply", compute_capability=compute_cap
        )

        C_explicit = torch.zeros((M, N), device="cuda")

        # Launch kernel
        matmul_kernel_explicit(
            grid=grid_dim,
            block=block_dim,
            args=[A.contiguous(), B.contiguous(), C_explicit, M, N, K],
        )

        # Verify results
        self.assertEqual(C_explicit, expected)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_as_custom_op(self):
        # [FUNCTION DOC] Wraps a runtime compiled vector addition kernel in a registered custom
        # op (with input dimension and size checks and a fake implementation returning an empty
        # tensor) and checks the op result on 1024 random floats against plain addition. Skipped
        # without CUDA.
        # Define a simple vector addition kernel
        kernel_source = """
        __global__ void vector_add(const float* a, const float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        @torch.library.custom_op("test_compile_kernel::vector_add", mutates_args=())
        def vector_add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation. Validates both inputs are
            # one-dimensional with equal size (raising through the check helper), compiles the
            # addition kernel, launches it and returns the output.
            from torch.cuda import _compile_kernel

            # Validate that tensors are 1-dimensional and have the same size
            torch._check(
                a.dim() == 1,
                lambda: f"Expected tensor 'a' to be 1-dimensional, but got {a.dim()} dimensions",
            )
            torch._check(
                b.dim() == 1,
                lambda: f"Expected tensor 'b' to be 1-dimensional, but got {b.dim()} dimensions",
            )
            torch._check(
                a.size() == b.size(),
                lambda: f"Expected tensors to have the same size, but got a.size()={a.size()} and b.size()={b.size()}",
            )
            compiled_kernel = _compile_kernel(kernel_source, "vector_add")

            c = torch.empty_like(a)
            n = a.numel()

            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            compiled_kernel(
                grid=(blocks_per_grid, 1, 1),
                block=(threads_per_block, 1, 1),
                args=[a, b, c, n],
            )

            return c

        @vector_add_op.register_fake
        def _(a, b):
            # [FUNCTION DOC] HELPER: Fake (shape-only) implementation returning an empty tensor
            # like the first input.
            return torch.empty_like(a)

        device = torch.device("cuda:0")
        size = (1024,)

        a = torch.randn(size, device=device, dtype=torch.float32)
        b = torch.randn(size, device=device, dtype=torch.float32)

        result = vector_add_op(a, b)

        expected = a + b
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_custom_op_validation(self):
        # [FUNCTION DOC] Registers a custom op that adds a double scalar to a tensor using a
        # runtime compiled kernel, with a fake implementation; runs the library opcheck
        # validation on it and checks the actual result equals the input plus the scalar.
        # Skipped without CUDA.
        kernel_source = """
        __global__ void add_scalar(const float* input, float* output, double scalar, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input[idx] + scalar;
            }
        }
        """

        @torch.library.custom_op("test_compile_kernel::add_scalar", mutates_args=())
        def add_scalar_op(input_tensor: torch.Tensor, scalar: float) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: compiles the add-scalar kernel,
            # launches it over the input with the scalar and returns the output tensor.
            from torch.cuda import _compile_kernel

            compiled_kernel = _compile_kernel(kernel_source, "add_scalar")

            output = torch.empty_like(input_tensor)
            n = input_tensor.numel()

            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            compiled_kernel(
                grid=(blocks_per_grid, 1, 1),
                block=(threads_per_block, 1, 1),
                args=[input_tensor, output, scalar, n],
            )

            return output

        @add_scalar_op.register_fake
        def _(input_tensor, scalar):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like
            # the input.
            return torch.empty_like(input_tensor)

        # Test with opcheck
        device = torch.device("cuda:0")
        input_data = torch.randn((64,), device=device, dtype=torch.float32)
        scalar_val = 3.14

        # Run opcheck validation
        torch.library.opcheck(add_scalar_op, (input_data, scalar_val), {})

        # Also test the actual functionality
        result = add_scalar_op(input_data, scalar_val)
        expected = input_data + scalar_val
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_double_precision(self):
        # [FUNCTION DOC] Checks that a Python float argument reaches a compiled kernel as a
        # double: a kernel fills a float64 tensor of 10 elements with a 15-digit value and the
        # output must equal it within 1e-14. Skipped without CUDA.
        """Test that Python floats are correctly handled as doubles in kernels."""
        kernel_source = """
        __global__ void test_double_precision(double* output, double value, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = value;
            }
        }
        """

        from torch.cuda import _compile_kernel

        compiled_kernel = _compile_kernel(kernel_source, "test_double_precision")

        # Test with high precision value that would lose precision if cast to float32
        # float32 has 7 digits of precision, so we use a value with 15 digits
        high_precision_value = 1.23456789012345
        n = 10

        output = torch.zeros(n, device="cuda", dtype=torch.float64)
        compiled_kernel(
            grid=(1, 1, 1),
            block=(256, 1, 1),
            args=[output, high_precision_value, n],
        )

        # Verify high precision is preserved (would fail with old float32 casting)
        expected = torch.full(
            (n,), high_precision_value, device="cuda", dtype=torch.float64
        )
        torch.testing.assert_close(output, expected, rtol=1e-14, atol=1e-14)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_cuda_headers(self):
        # [FUNCTION DOC] Checks that a runtime compiled kernel can include the half precision
        # header (except under HIP) and write half values: a kernel converts a double argument
        # to half and fills 100 elements; output compared with a float16 filled tensor. Skipped
        # without CUDA.
        """Test that kernels can include and use CUDA headers like cuda_fp16.h."""
        kernel_source = """
        #ifndef __HIPCC__
        #include <cuda_fp16.h>
        #endif

        extern "C"
        __global__ void half_precision_kernel(__half* output, double input_value, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = __float2half((float)input_value);
            }
        }
        """

        from torch.cuda import _compile_kernel

        compiled_kernel = _compile_kernel(kernel_source, "half_precision_kernel")

        n = 100
        test_value = 3.14159
        output = torch.zeros(n, device="cuda", dtype=torch.float16)

        compiled_kernel(
            grid=(1, 1, 1),
            block=(256, 1, 1),
            args=[output, test_value, n],
        )

        expected = torch.full((n,), test_value, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_template(self):
        # [FUNCTION DOC] Compiles a templated addition kernel by instantiated names for float
        # and int, launches each on 1024 elements and asserts the outputs equal the sums.
        # Skipped without CUDA.
        kernel_source = """
        template<typename T>
        __global__ void add_tensors(const T* a, const T* b, T* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        """

        # Compile the kernel
        from torch.cuda import _compile_kernel

        add_kernel_float = _compile_kernel(kernel_source, "add_tensors<float>")

        # Prepare data
        N = 1024
        a = torch.rand(N, device="cuda")
        b = torch.rand(N, device="cuda")
        c = torch.empty_like(a)

        # Calculate grid and block dimensions
        threads_per_block = 256
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

        # Launch kernel
        add_kernel_float(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[a, b, c, N],
        )

        # Verify results
        expected = a + b
        self.assertEqual(c, expected)

        # do again with different dtype
        add_kernel_int = _compile_kernel(kernel_source, "add_tensors<int>")

        # Prepare data
        N = 1024
        a = torch.randint(-1000, 1000, size=(N,), dtype=torch.int, device="cuda")
        b = torch.randint(-1000, 1000, size=(N,), dtype=torch.int, device="cuda")
        c = torch.empty_like(a)

        # Calculate grid and block dimensions
        threads_per_block = 256
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

        # Launch kernel
        add_kernel_int(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[a, b, c, N],
        )

        # Verify results
        expected = a + b
        self.assertEqual(c, expected)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_compile_kernel_dlpack(self):
        # [FUNCTION DOC] Checks compiled kernels accept tensors created through DLPack round
        # trip: adds two DLPack-imported tensors into an output and compares; then writes into
        # the imported tensor and checks the original tensor changed (shared memory). Skipped
        # without CUDA.
        """Test that compile_kernel works with tensors created via DLPack."""
        kernel_source = """
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        """

        from torch.cuda import _compile_kernel

        add_kernel = _compile_kernel(kernel_source, "add_tensors")

        N = 512
        a = torch.rand(N, device="cuda", dtype=torch.float32)
        b = torch.rand(N, device="cuda", dtype=torch.float32)

        a_dlpack = torch.utils.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(a))
        b_dlpack = torch.utils.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(b))
        c = torch.empty_like(a)

        threads_per_block = 256
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

        add_kernel(
            grid=(blocks_per_grid, 1, 1),
            block=(threads_per_block, 1, 1),
            args=[a_dlpack, b_dlpack, c, N],
        )

        self.assertEqual(c, a + b)
        a_dlpack[0] = 42.0
        self.assertEqual(a[0].item(), 42.0, "DLPack tensors should share memory")


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestCudaDeviceParametrized(TestCase):
    # [CLASS DOC] Device-type-instantiated test class for CUDA graph and event interplay.
    # Skipped without CUDA. Holds a single test using external events and a host-controlled
    # spin-wait kernel.
    @skipIfRocmVersionLessThan((7, 0))
    @skipCUDAIf(
        not SM70OrLater, "Compute capability >= SM70 required for relaxed ptx flag"
    )
    def test_graph_external_wait_and_record(self):
        # [FUNCTION DOC] Tests external events in CUDA graphs. A signalling graph spins until a
        # host flag is set then records an external start event; a work graph waits on that
        # event, copies to pinned memory and records an end event. Replaying both on separate
        # streams must not copy or finish the end event until the host sets the flag; afterwards
        # the copy is complete, the end event synchronizes and the work stream is idle (polled
        # briefly). The flag is always set in a finally block. Requires ROCm 7.0+ and SM70+.
        torch.cuda.empty_cache()

        spin_wait_kernel = get_wait_for_cpu_kernel()

        x = torch.ones(4, device="cuda")
        x_cpu = torch.zeros(x.shape, device="cpu").pin_memory()
        flag_cpu = torch.zeros(1, dtype=torch.int32, device="cpu").pin_memory()
        start_event = torch.cuda.Event(external=True)
        end_event = torch.cuda.Event(external=True)

        signalling_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(signalling_graph, capture_error_mode="relaxed"):
            spin_wait_kernel(grid=(1, 1, 1), block=(1, 1, 1), args=[flag_cpu])
            start_event.record()

        # This is counter-intuitive, but a cudaEventRecord() during
        # stream capture does not count as the first call to
        # cudaEventRecord(). Rather, cudaGraphLaunch() counts as the
        # first call to cudaEventRecord(). Therefore, all calls to
        # cudaEventQuery() will succeed before that happens.

        # See:
        # "Before the first call to cudaEventRecord(), an event represents an empty set of work, so for example cudaEventQuery() would return cudaSuccess."
        # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        self.assertTrue(start_event.query(), "Start event's work should be empty")

        work_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(work_graph, capture_error_mode="relaxed"):
            start_event.wait()
            x_cpu.copy_(x, non_blocking=True)
            end_event.record()

        self.assertTrue(
            torch.all(x_cpu == 0.0), "Copy cannot occur until start_event is recorded"
        )

        try:
            signalling_stream = torch.cuda.Stream()
            with torch.cuda.stream(signalling_stream):
                signalling_graph.replay()

            work_stream = torch.cuda.Stream()
            with torch.cuda.stream(work_stream):
                work_graph.replay()

            self.assertFalse(
                end_event.query(), "Event record node cannot run until flag_cpu[0]=1"
            )

            # Sleep for a little to make sure that work doesn't proceed until we set flag_cpu[0]=1
            time.sleep(1)
            self.assertTrue(
                torch.all(x_cpu == 0.0),
                "Copy cannot occur until start_event is recorded",
            )
        finally:
            # In case an assertion fails, we still need to empty out
            # the GPU queue of work. Therefore, we do this write
            # unconditionally, even if an exception is thrown.

            # This writes allows wait_for_cpu to proceed
            # This is an atomic store at system scope according to this rule:
            # "the scope is thread_scope_system and it is a load or store that affects a naturally-aligned object of sizes 1, 2, 4, 8, or 16 bytes on mapped memory"
            # https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/memory_model.rst

            # Note that every CPU store is implicitly system scope,
            # even if we don't use C++ atomics like this:
            # std::atomic_ref<int32_t>::store(1);
            flag_cpu[0] = 1

        end_event.synchronize()
        self.assertTrue(
            torch.all(x_cpu == 1.0),
            "Copy should be done once end_event is synchronized",
        )
        # On HIP graphs, the event-record node can become visible to
        # event.synchronize() a few microseconds before the host-side
        # stream state is updated; poll briefly so we exercise the
        # invariant ("after sync the stream is drained") without flaking.
        deadline = time.monotonic() + 0.1
        while not work_stream.query() and time.monotonic() < deadline:
            pass
        self.assertTrue(
            work_stream.query(),
            "end_event.synchronize() completing should imply that work_stream is done",
        )


class TestFXMemoryProfiler(TestCase):
    # [CLASS DOC] Tests that memory snapshots can be augmented with compiler graph debug
    # information: frames of allocations during compiled execution carry node op, name, target
    # and original stack trace text. Defines two small MLP modules whose source differs only by
    # variable names, and a helper that collects annotated frames from a snapshot.
    """Tests for memory profiler augmentation with original stack traces."""

    class MLPModule(nn.Module):
        # [CLASS DOC] HELPER: Two-linear-layer MLP with a relu in between, seeded with a fixed
        # seed at creation; forward names its intermediate results a, b and c.
        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Seeds the generator with 5 and creates linear layer 10 to
            # 16, relu and linear layer 16 to 10 on the given device.
            super().__init__()
            torch.manual_seed(5)
            self.net1 = nn.Linear(10, 16, bias=True, device=device)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 10, bias=True, device=device)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the first linear layer, the relu and the second
            # linear layer in order and returns the result.
            a = self.net1(x)
            b = self.relu(a)
            c = self.net2(b)
            return c

    class MLPModule2(nn.Module):
        # [CLASS DOC] HELPER: Same architecture as the first MLP but with intermediate variable
        # names d, e and f, so its generated graph source differs only in those names.
        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Seeds the generator with 5 and creates linear layer 10 to
            # 16, relu and linear layer 16 to 10 on the given device.
            super().__init__()
            torch.manual_seed(5)
            self.net1 = nn.Linear(10, 16, bias=True, device=device)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 10, bias=True, device=device)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the first linear layer, the relu and the second
            # linear layer in order and returns the result.
            d = self.net1(x)
            e = self.relu(d)
            f = self.net2(e)
            return f

    def collect_frames(
        self, augmented_snapshot, collect_device_traces=True, collect_segments=True
    ):
        # [FUNCTION DOC] HELPER: Walks a snapshot and returns a list of frames (mappings) that
        # carry graph node op or name fields, looking in device trace entries and, optionally,
        # in segment blocks. Flags select which sources to search.
        """Collects all frames that has node metadata from a memory snapshot."""
        # Collect all frames with FX metadata
        fx_frames = []

        # Check device traces for FX debug fields
        if collect_device_traces and "device_traces" in augmented_snapshot:
            for trace_list in augmented_snapshot["device_traces"]:
                for trace_entry in trace_list:
                    if isinstance(trace_entry, dict) and "frames" in trace_entry:
                        for frame in trace_entry["frames"]:
                            if isinstance(frame, dict):
                                # Check for FX debug fields
                                if "fx_node_op" in frame or "fx_node_name" in frame:
                                    fx_frames.append(frame)

        # Check segments/blocks for FX debug fields
        if collect_segments and "segments" in augmented_snapshot:
            for segment in augmented_snapshot["segments"]:
                if "blocks" in segment:
                    for block in segment["blocks"]:
                        if "frames" in block:
                            for frame in block["frames"]:
                                if isinstance(frame, dict):
                                    if "fx_node_op" in frame or "fx_node_name" in frame:
                                        fx_frames.append(frame)
        return fx_frames

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
    def test_fx_memory_profiler_augmentation(self):
        # [FUNCTION DOC] Compiles a small MLP with the eager-aot backend while recording memory
        # history, takes a snapshot augmented with graph traces and checks there are more than
        # two annotated frames each having node op, name, target and original trace, names
        # restricted to addmm, relu and addmm_1, and the original trace text matching the model
        # source line. Repeats with a second module that has identical generated code but
        # different source text to verify the metadata is not shared. Skipped without CUDA.
        # KNOWN ISSUE: asserts on quoted source text of the forward methods of the helper
        # modules in this class (the original trace lines); edits to those methods must keep the
        # three statements in each unchanged.
        """Test that memory snapshots are augmented with FX debug information."""

        device = "cuda"
        mod = self.MLPModule(device)
        # reset cache to start fresh
        torch.cuda.memory.empty_cache()
        torch.cuda.memory._record_memory_history()
        compiled = torch.compile(mod, backend="aot_eager", fullgraph=True)
        result = compiled(torch.randn(10, 10, device=device))
        augmented_snapshot = torch.cuda.memory._snapshot(augment_with_fx_traces=True)
        torch.cuda.memory._record_memory_history(enabled=None, clear_history=True)
        torch.cuda.empty_cache()

        fx_frames = self.collect_frames(augmented_snapshot)
        self.assertGreater(len(fx_frames), 2)

        for frame in fx_frames:
            # Every FX frame should have both node_op and node_name
            self.assertIn("fx_node_op", frame)
            self.assertIn("fx_node_name", frame)
            self.assertIn("fx_node_target", frame)
            self.assertIn("fx_original_trace", frame)

            self.assertIn(frame["fx_node_name"], ["addmm", "relu", "addmm_1"])
            fx_node_name = frame["fx_node_name"]
            if fx_node_name == "addmm":
                self.assertIn("a = self.net1(x)", frame["fx_original_trace"])
            elif fx_node_name == "addmm_1":
                self.assertIn("c = self.net2(b)", frame["fx_original_trace"])
            elif fx_node_name == "relu":
                self.assertIn("b = self.relu(a)", frame["fx_original_trace"])

        # Test that when we have two graphs with the same src_code, they're not hashed
        # to the same metadata
        mod = self.MLPModule2(device)
        torch.cuda.memory._record_memory_history()
        compiled = torch.compile(mod, backend="aot_eager", fullgraph=True)
        result = compiled(torch.randn(10, 10, device=device))
        augmented_snapshot = torch.cuda.memory._snapshot(augment_with_fx_traces=True)
        torch.cuda.memory._record_memory_history(enabled=None, clear_history=True)

        # avoid collecting segments from previous run for unit test purpose
        fx_frames = self.collect_frames(augmented_snapshot, collect_segments=False)
        self.assertGreater(len(fx_frames), 0)

        for frame in fx_frames:
            # Every FX frame should have both node_op and node_name
            self.assertIn("fx_node_op", frame)
            self.assertIn("fx_node_name", frame)
            self.assertIn("fx_node_target", frame)
            self.assertIn("fx_original_trace", frame)

            self.assertIn(frame["fx_node_name"], ["addmm", "relu", "addmm_1"])
            fx_node_name = frame["fx_node_name"]
            if fx_node_name == "addmm":
                self.assertIn("d = self.net1(x)", frame["fx_original_trace"])
            elif fx_node_name == "addmm_1":
                self.assertIn("f = self.net2(e)", frame["fx_original_trace"])
            elif fx_node_name == "relu":
                self.assertIn("e = self.relu(d)", frame["fx_original_trace"])


@unittest.skipIf(
    not PLATFORM_SUPPORTS_GREEN_CONTEXT, "Green contexts are not supported"
)
class TestCudaGreenContexts(TestCase):
    # [CLASS DOC] Tests for green contexts (SM partitioning) in torch.cuda.green_contexts. The
    # whole class is skipped when the platform lacks green context support. Tests cover
    # deprecated set/pop methods, stream context restoration, SM carveout slowdown, graph
    # capture on a green stream and workqueue limits.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture hook; only delegates to the base class setUp.
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture hook; only delegates to the base class tearDown.
        super().tearDown()

    def test_greencontext_set_pop_context_deprecation(self):
        # [FUNCTION DOC] On a side stream, creates a one-SM green context; asserts that the
        # deprecated set and pop context methods emit FutureWarning naming themselves, that the
        # context stream differs from the starting stream, and that popping restores the
        # starting stream (compared by raw handle).
        # need to start on a side stream as we are comparing pointers and want to avoid
        # two NULL streams...
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            start_stream = torch.cuda.current_stream()
            ctx = torch.cuda.green_contexts.GreenContext(num_sms=1)
            with self.assertWarnsRegex(FutureWarning, "GreenContext.set_context"):
                ctx.set_context()
            context_stream = torch.cuda.current_stream()
            with self.assertWarnsRegex(FutureWarning, "GreenContext.pop_context"):
                ctx.pop_context()
            end_stream = torch.cuda.current_stream()
            self.assertEqual(start_stream.cuda_stream, end_stream.cuda_stream)
            self.assertNotEqual(start_stream.cuda_stream, context_stream.cuda_stream)

    def test_greencontext_stream_context_restores_stream(self):
        # [FUNCTION DOC] On a side stream, creates a green context stream and enters it as a
        # stream context manager; the entered value is None, the current stream inside has the
        # green stream handle, and the starting stream is restored on exit.
        # need to start on a side stream as we are comparing pointers and
        # want to avoid the NULL stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            start_stream = torch.cuda.current_stream()
            ctx = torch.cuda.green_contexts.GreenContext(num_sms=1)
            ctx_stream = ctx.Stream()
            stream_context = torch.cuda.stream(ctx_stream)
            with stream_context as context_value:
                self.assertIsNone(context_value)
                context_stream = torch.cuda.current_stream()
            end_stream = torch.cuda.current_stream()
            self.assertIsInstance(stream_context, torch.cuda.StreamContext)
            self.assertEqual(ctx_stream.cuda_stream, context_stream.cuda_stream)
            self.assertEqual(start_stream.cuda_stream, end_stream.cuda_stream)
            self.assertNotEqual(start_stream.cuda_stream, context_stream.cuda_stream)

    @serialTest()
    def test_greencontext_stream_carveout(self):
        # [FUNCTION DOC] Measures that a matmul run on a one-SM green stream takes longer
        # than the same matmul on the full device while giving equal results (carveout
        # effect). Serial.
        # By default, everything is performed on the current device
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        ctx = torch.cuda.green_contexts.GreenContext(num_sms=1)
        ctx_stream = ctx.Stream()
        with torch.cuda.stream(ctx_stream):
            torch.matmul(a, a)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            partial_res = torch.matmul(a, a)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        torch.matmul(a, a)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        full_res = torch.matmul(a, a)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        self.assertEqual(partial_res, full_res)
        self.assertGreater(t1 - t0, t3 - t2)

    @serialTest()
    def test_greencontext_graphs(self):
        # [FUNCTION DOC] Computes a matmul on a one-SM green stream, then captures the same
        # matmul into a CUDA graph on that stream, replays it and checks it equals the full
        # device result. Serial.
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        ctx = torch.cuda.green_contexts.GreenContext(num_sms=1)
        ctx_stream = ctx.Stream()
        with torch.cuda.stream(ctx_stream):
            partial_res = torch.matmul(a, a)
        full_res = torch.matmul(a, a)
        full_res.zero_()
        partial_res.zero_()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=ctx_stream):
            partial_res = torch.matmul(a, a)
        g.replay()
        full_res = torch.matmul(a, a)
        self.assertEqual(partial_res, full_res)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_WORKQUEUE_CONFIG, "Workqueue config is not supported"
    )
    @serialTest()
    def test_greencontext_workqueue_concurrency_limit(self):
        # [FUNCTION DOC] Requires workqueue config support and at least 4 workqueues. Times four
        # concurrent sleeping streams in the default context as a baseline, then four streams of
        # a green context whose workqueue concurrency is limited to 1, and asserts the limited
        # run takes longer. Serial.
        # By default, everything is performed on the current device
        n_streams = 4
        GreenContext = torch.cuda.green_contexts.GreenContext
        max_wq = GreenContext.max_workqueue_concurrency()
        if max_wq < n_streams:
            self.skipTest(
                f"Device has {max_wq} workqueue(s), need >{n_streams} to test concurrency limiting"
            )

        wq_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_streams * 2)]

        def run_multi_stream_sleep(streams):
            # [FUNCTION DOC] HELPER: For each given stream, records a start timing event,
            # sleeps 100M GPU cycles and records an end event on that stream, then synchronizes
            # the device.
            for i, s in enumerate(streams):
                with torch.cuda.stream(s):
                    # note we need to record timing events to ensure that the
                    # workqueue is actually used
                    wq_events[i * 2].record(s)
                    # 100M cycles is enough on all currently supported GPUs
                    # to ensure that the test runs significantly longer than
                    # host overhead to 1) activate the different streams,
                    # 2) record timing events, and 3) synchronize with the GPU.
                    torch.cuda._sleep(100_000_000)
                    wq_events[i * 2 + 1].record(s)
            torch.cuda.synchronize()

        # Note: in case we have lazy module loading, ensure that we called the
        # sleep kernel at least once s.t. it is loaded in memory.
        torch.cuda._sleep(1_000)

        # Baseline: n_streams streams from default context, full workqueue concurrency
        baseline_streams = [torch.cuda.Stream() for _ in range(n_streams)]
        # note: torch.cuda.synchronize() will wait for all kernels on all streams
        # so we can safely use CPU based timing here.
        t0 = time.perf_counter()
        run_multi_stream_sleep(baseline_streams)
        t1 = time.perf_counter()
        baseline_time = t1 - t0

        # Green context with workqueue concurrency limited to 1
        ctx = GreenContext(
            workqueue_scope="balanced",
            workqueue_concurrency_limit=1,
        )
        green_streams = [ctx.Stream() for _ in range(n_streams)]
        t2 = time.perf_counter()
        run_multi_stream_sleep(green_streams)
        t3 = time.perf_counter()
        limited_time = t3 - t2

        self.assertGreater(limited_time, baseline_time)


class TestCudaArchList(TestCase):
    # [CLASS DOC] Tests for torch.cuda.get_arch_list that do not need a GPU device.
    def test_get_arch_list_empty_when_cuda_not_compiled(self):
        # [FUNCTION DOC] Skips if CUDA is compiled in; otherwise asserts get_arch_list returns
        # an empty list.
        if torch.cuda._is_compiled():
            self.skipTest("CUDA is compiled")
        self.assertEqual(torch.cuda.get_arch_list(), [])

    @unittest.skipIf(not torch.cuda._is_compiled(), "CUDA not compiled")
    def test_get_arch_list_does_not_require_cuda_is_available(self):
        # [FUNCTION DOC] With CUDA compiled in, patches availability to False and asserts
        # get_arch_list still returns the list of architectures from the compile-time
        # flag string.
        expected = (torch._C._cuda_getArchFlags() or "").split()
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(torch.cuda.get_arch_list(), expected)


instantiate_parametrized_tests(TestCuda)
instantiate_parametrized_tests(TestCudaAllocator)
instantiate_parametrized_tests(TestCompileKernel)
instantiate_parametrized_tests(TestCachingHostAllocatorCudaGraph)
instantiate_parametrized_tests(TestCachingHostAllocatorConfig)
instantiate_parametrized_tests(TestGDS)
instantiate_device_type_tests(TestCudaOptims, globals())
instantiate_device_type_tests(TestCudaDeviceParametrized, globals())
instantiate_device_type_tests(TestCudaGreenContexts, globals(), except_for="cpu")


# Tests for fp32_precision flag propagation that don't require an actual CUDA
# device — they only exercise C++ context state management.
class TestFP32PrecisionFlags(TestCase):
    # [CLASS DOC] Tests for fp32 precision flag propagation that only exercise C++ context state
    # and need no CUDA device.
    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/182021"
    )
    @recover_orig_fp32_precision
    @serialTest()
    def test_generic_fp32_precision_propagates_to_cudnn_conv_rnn(self):
        # [FUNCTION DOC] Regression test: checks the defaults of cudnn conv and rnn precisions
        # are tf32, and that setting the global backend fp32 precision to ieee inside the flags
        # context manager changes cudnn conv, cudnn rnn, cudnn overall and cuda matmul
        # precisions to ieee. Skipped on Linux and slow runs (tracked issue); serial.
        # Regression test: setting torch.backends.fp32_precision = "ieee"
        # must propagate to cudnn.conv and cudnn.rnn.
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")
        with torch.backends.flags(fp32_precision="ieee"):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "ieee")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "ieee")
            self.assertEqual(torch.backends.cudnn.fp32_precision, "ieee")
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")


class TestMemoryViz(TestCase):
    # [CLASS DOC] Tests for the torch.cuda memory visualization helpers.
    def test_format_flamegraph_download_moves_temp_file(self):
        # [FUNCTION DOC] Regression test for flamegraph formatting: with the download function
        # and the subprocess launcher patched (the download writes a small script file, the
        # launcher yields a mock process printing fixed output), formatting a flamegraph into an
        # absent cached script path must return the mocked output and leave the downloaded
        # script in place without raising on cleanup.
        # Regression test: format_flamegraph downloads flamegraph.pl into a temp
        # file and moves it into place. Previously the temp file was created by a
        # delete=True NamedTemporaryFile, so after os.rename moved it away the
        # context manager raised FileNotFoundError on exit (a CI flake, since the
        # download only happens when the cached script is absent).
        from torch.cuda import _memory_viz

        def fake_urlretrieve(url, filename):
            # [FUNCTION DOC] HELPER: Replacement for the URL retrieve function that writes a
            # perl shebang line to the destination file and returns the filename with None.
            with open(filename, "wb") as f:
                f.write(b"#!/usr/bin/perl\n")
            return filename, None

        @contextlib.contextmanager
        def fake_popen(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for the subprocess launcher: a context manager
            # yielding a mock process whose stdout read returns a fixed text and whose wait
            # returns 0.
            proc = unittest.mock.MagicMock()
            proc.stdout.read.return_value = "FLAMEGRAPH_OUTPUT"
            proc.wait.return_value = 0
            yield proc

        with tempfile.TemporaryDirectory() as cache_dir:
            script = os.path.join(cache_dir, "flamegraph.pl")
            with (
                patch("urllib.request.urlretrieve", fake_urlretrieve),
                patch.object(_memory_viz.subprocess, "Popen", fake_popen),
            ):
                result = _memory_viz.format_flamegraph("stack 1\n", script)
            self.assertEqual(result, "FLAMEGRAPH_OUTPUT")
            self.assertTrue(os.path.exists(script))


if __name__ == "__main__":
    run_tests()
