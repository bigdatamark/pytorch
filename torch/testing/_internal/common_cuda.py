# mypy: ignore-errors

# [MODULE DOC] Shared CUDA/ROCm/XPU test support. Defines lazily evaluated module-level flags
# describing the hardware and build (multi-GPU, cuDNN and CUPTI availability and versions, SM
# generation predicates such as SM80OrLater, exact SM and Jetson/Thor checks, ROCm version,
# numba CUDA, and PLATFORM_SUPPORTS_* feature gates for attention backends, bf16, atomics,
# fp8, MX gemm, green contexts and workqueue), decorators and context managers to control TF32
# and to mark expected failures by architecture, and small helpers to create
# autocast/grad-scaler test models. Many flags are LazyVal objects evaluated on first use so
# CUDA is not initialized. Import-time side effects: records whether CUDA was already
# initialized, builds the flags, imports numba.cuda if available, and raises an error if
# importing this module initialized CUDA.

r"""This file is allowed to initialize CUDA context when imported."""

import functools
import threading
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS, IS_MACOS, TEST_XPU
from torch.utils._import_utils import _check_module_exists
import inspect
import contextlib
import os
import unittest


CUDA_ALREADY_INITIALIZED_ON_IMPORT = torch.cuda.is_initialized()


TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = torch.device("cuda:0") if TEST_CUDA else None
# note: if ROCm is targeted, TEST_CUDNN is code for TEST_MIOPEN
if TEST_WITH_ROCM:
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA)
else:
    TEST_CUDNN = LazyVal(lambda: TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=CUDA_DEVICE)))

TEST_CUDNN_VERSION = LazyVal(lambda: torch.backends.cudnn.version() if TEST_CUDNN else 0)
ROCM_VERSION = LazyVal(lambda : tuple(int(v) for v in torch.version.hip.split('.')[:2]) if torch.version.hip else (0, 0))

# The CUPTI monitor needs both the cupti-python bindings and the build-generated
# _cupti_stubs catalogs (emitted only on CUDA >= 13.3 builds where the field-id codegen
# ran); the module hard-imports the latter, so guard on both to skip (not error) where
# the stubs were not generated.
TEST_CUPTI = (
    _check_module_exists("cupti")
    and _check_module_exists("torch.profiler._cupti._cupti_stubs")
    and not TEST_WITH_ROCM
)

def _cupti_version():
    # [FUNCTION DOC] HELPER: Returns the CUPTI library version as an integer, or 0 if CUPTI is
    # unavailable or querying the version fails for any reason.
    if not TEST_CUPTI:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti
        return pylibcupti().get_version()
    except Exception:
        return 0


TEST_CUPTI_V13_3 = LazyVal(lambda: TEST_CUPTI and _cupti_version() >= 130300)


def _cuda_graph_tools_id_available():
    # [FUNCTION DOC] HELPER: Returns True if the CUDA graph node tools-id API is usable (checked
    # through the graph annotations module, which probes the driver), and False on any error.
    # cudaGraphNodeGetToolsId needs cuda-bindings >= 13.1 *and* a CUDA driver
    # >= 13.1 (or cuda-compat), which is independent of the libcupti version:
    # a CUDA 13.0 runner can ship libcupti >= 13.3 and still lack the API.
    # is_available() probes the driver and caches the result.
    try:
        from torch.cuda.graph_annotations import is_available
        return is_available()
    except Exception:
        return False


TEST_CUDA_GRAPH_TOOLS_ID = LazyVal(_cuda_graph_tools_id_available)

SM53OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3))
SM60OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (6, 0))
SM70OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0))
SM75OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5))
SM80OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0))
SM89OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
SM90OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0))
SM100OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0))
SM120OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0))

IS_THOR = LazyVal(lambda: torch.cuda.is_available() and torch.version.cuda is not None and
                  ((torch.cuda.get_device_capability() == (11, 0) and int(torch.version.cuda[:2]) >= 13) or
                   (torch.cuda.get_device_capability() == (10, 1) and int(torch.version.cuda[:2]) < 13)))
IS_JETSON = LazyVal(lambda: torch.cuda.is_available() and (torch.cuda.get_device_capability() in [(7, 2), (8, 7)] or IS_THOR))
# These exact-match SM predicates identify specific NVIDIA architectures and must
# be False on ROCm, where torch.cuda.get_device_capability() returns the gfx-arch
# version and collides with NVIDIA capability tuples (e.g. gfx90a reads as (9, 0)).
IS_SM89 = LazyVal(lambda: torch.version.hip is None and torch.cuda.is_available() and
                  torch.cuda.get_device_capability() == (8, 9))
IS_SM90 = LazyVal(lambda: torch.version.hip is None and torch.cuda.is_available() and
                  torch.cuda.get_device_capability() == (9, 0))
IS_SM100 = LazyVal(lambda: torch.version.hip is None and torch.cuda.is_available() and
                   torch.cuda.get_device_capability() == (10, 0))
IS_SM10X = LazyVal(lambda: torch.version.hip is None and torch.cuda.is_available() and
                   torch.cuda.get_device_capability()[0] == 10)
IS_SM12X = LazyVal(lambda: torch.version.hip is None and torch.cuda.is_available() and
                   torch.cuda.get_device_capability()[0] == 12)

@contextlib.contextmanager
def blas_library_context(backend):
    # [FUNCTION DOC] Context manager that sets the preferred BLAS library backend for CUDA
    # matmuls to the given backend and restores the previous preference on exit, even if the
    # body raises.
    prev_backend = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(backend)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(prev_backend)

def evaluate_gfx_arch_within(arch_list):
    # [FUNCTION DOC] Returns True if the current GPU's architecture name contains any of the
    # given gfx target prefixes. Uses the environment variable that overrides the arch name for
    # flash attention tests when set. Returns False if CUDA is unavailable.
    if not torch.cuda.is_available():
        return False
    gcn_arch_name = torch.cuda.get_device_properties('cuda').gcnArchName
    effective_arch = os.environ.get('PYTORCH_DEBUG_FLASH_ATTENTION_GCN_ARCH_OVERRIDE', gcn_arch_name)
    # gcnArchName can be complicated strings like gfx90a:sramecc+:xnack-
    # Hence the matching should be done reversely
    return any(arch in effective_arch for arch in arch_list)

# Per-generation gfx targets, ordered oldest -> newest. Each "OrLater" helper
# below unions its own generation with every newer one, so the predicates are
# nested by construction: CDNA5OrLater => CDNA3OrLater => CDNA2OrLater.
_CDNA2_ARCHS = ["gfx90a"]
_CDNA3_ARCHS = ["gfx942", "gfx950"]
# GFX1250 (CDNA 5)
_CDNA5_ARCHS = ["gfx1250"]

def CDNA5OrLater():
    # [FUNCTION DOC] HELPER: Returns True on gfx1250 (CDNA5) devices.
    return evaluate_gfx_arch_within(_CDNA5_ARCHS)

def CDNA3OrLater():
    # [FUNCTION DOC] HELPER: Returns True on CDNA3 or newer devices (gfx942, gfx950, gfx1250).
    return evaluate_gfx_arch_within(_CDNA3_ARCHS + _CDNA5_ARCHS)

def CDNA2OrLater():
    # [FUNCTION DOC] HELPER: Returns True on CDNA2 or newer devices (gfx90a and all
    # newer targets).
    return evaluate_gfx_arch_within(_CDNA2_ARCHS + _CDNA3_ARCHS + _CDNA5_ARCHS)

# Archs that take the opportunistic_fastAtomicAdd path (packed 2x16 atomics + DPP
# lane coalescing) in ScatterGatherKernel.cu. Keep in sync with that kernel's arch
# gate; this is intentionally not CDNA3OrLater (gfx1250 uses plain fastAtomicAdd).
def gfx_arch_supports_opportunistic_fastatomics():
    # [FUNCTION DOC] Returns True if the device arch is one of gfx942 or gfx950, which take the
    # opportunistic fast atomic add path in the scatter/gather kernel (must stay in sync with
    # the kernel's gate).
    return evaluate_gfx_arch_within(["gfx942", "gfx950"])

def evaluate_platform_supports_flash_attention():
    # [FUNCTION DOC] Returns whether flash attention is supported on this platform: on ROCm, a
    # device in a fixed gfx list (extended with more archs if the experimental AOTriton
    # environment variable is set); on CUDA, not Windows and SM80 or newer; on XPU, True;
    # otherwise False.
    if TEST_WITH_ROCM:
        # NOTE: gfx1250 is omitted until flash-attention artifacts ship for it.
        # The AOTriton path needs the gfx1250 GPU image from AOTriton 0.12.1b,
        # which is wired up by PR #188242 (which also adds gfx1250 to this list);
        # this gate-only PR leaves it out so the two changes do not conflict
        # (see cmake/External/aotriton.cmake). The CK FAv3/AITER codegen is
        # separately not yet wired for gfx1250
        # (see aten/src/ATen/native/transformers/hip/flash_attn/ck/fav_v3/CMakeLists.txt).
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return not IS_WINDOWS and SM80OrLater
    if TEST_XPU:
        return True
    return False

def evaluate_platform_supports_ck_sdpa():
    # [FUNCTION DOC] Returns True only on ROCm builds where the CK scaled-dot-product-attention
    # backend is available; False otherwise.
    if TEST_WITH_ROCM:
        return torch.backends.cuda.is_ck_sdpa_available()
    else:
        return False

def evaluate_platform_supports_efficient_attention():
    # [FUNCTION DOC] Returns whether memory-efficient attention is supported: on ROCm, a device
    # in a fixed gfx list (extended when the experimental AOTriton environment variable is set);
    # True on CUDA and XPU; False otherwise.
    if TEST_WITH_ROCM:
        # NOTE: gfx1250 is omitted until mem-efficient-attention artifacts ship
        # for it. The AOTriton gfx1250 image (from AOTriton 0.12.1b) is wired up
        # by PR #188242, which also adds gfx1250 here; this gate-only PR leaves
        # it out to avoid conflicting with that change. The CK FAv3/AITER codegen
        # is separately not yet wired for gfx1250.
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return True
    if TEST_XPU:
        return True
    return False

def evaluate_platform_supports_cudnn_attention():
    # [FUNCTION DOC] Returns True for non-ROCm devices with SM80 or newer and cuDNN version at
    # least 90000.
    return (not TEST_WITH_ROCM) and SM80OrLater and (TEST_CUDNN_VERSION >= 90000)

def evaluate_platform_supports_green_context():
    # [FUNCTION DOC] Returns True if green contexts are supported on this machine, determined by
    # calling the support check and catching its RuntimeError.
    from torch.cuda.green_contexts import _ensure_supported
    try:
        _ensure_supported()
        return True
    except RuntimeError:
        return False

PLATFORM_SUPPORTS_FLASH_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_flash_attention())
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_efficient_attention())
PLATFORM_SUPPORTS_CUDNN_ATTENTION: bool = LazyVal(lambda: evaluate_platform_supports_cudnn_attention())
# This condition always evaluates to PLATFORM_SUPPORTS_MEM_EFF_ATTENTION but for logical clarity we keep it separate
PLATFORM_SUPPORTS_FUSED_ATTENTION: bool = LazyVal(lambda: PLATFORM_SUPPORTS_FLASH_ATTENTION or
                                                  PLATFORM_SUPPORTS_CUDNN_ATTENTION or
                                                  PLATFORM_SUPPORTS_MEM_EFF_ATTENTION)

PLATFORM_SUPPORTS_FUSED_SDPA: bool = TEST_CUDA and not TEST_WITH_ROCM

PLATFORM_SUPPORTS_CK_SDPA: bool = LazyVal(lambda: evaluate_platform_supports_ck_sdpa())


def evaluate_platform_supports_bf16():
    # [FUNCTION DOC] Returns whether bfloat16 is supported: SM80 or newer on CUDA, True on ROCm
    # and XPU, False otherwise.
    if torch.version.cuda:
        return SM80OrLater
    elif torch.version.hip:
        return True
    elif TEST_XPU:
        return True
    return False


def evaluate_platform_supports_bf16_atomics():
    # [FUNCTION DOC] Returns whether bfloat16 atomics are supported: SM80 or newer on CUDA, ROCm
    # version 8.0 or newer on ROCm, False otherwise.
    if torch.version.cuda:
        return SM80OrLater
    elif torch.version.hip:
        return ROCM_VERSION >= (8, 0)
    return False


def evaluate_platform_supports_half_atomics():
    # [FUNCTION DOC] Returns whether half-precision atomics are supported: on ROCm only for
    # version 8.0 or newer, True otherwise.
    if torch.version.hip:
        return ROCM_VERSION >= (8, 0)
    return True


PLATFORM_SUPPORTS_BF16: bool = LazyVal(lambda: evaluate_platform_supports_bf16())
PLATFORM_SUPPORTS_BF16_ATOMICS: bool = LazyVal(lambda: evaluate_platform_supports_bf16_atomics())
PLATFORM_SUPPORTS_HALF_ATOMICS: bool = LazyVal(lambda: evaluate_platform_supports_half_atomics())

PLATFORM_SUPPORTS_GREEN_CONTEXT: bool = LazyVal(lambda: evaluate_platform_supports_green_context())

def evaluate_platform_supports_workqueue_config():
    # [FUNCTION DOC] Returns True if green contexts and workqueue configuration are both
    # supported, catching the RuntimeError raised by the support checks.
    from torch.cuda.green_contexts import _ensure_supported, _ensure_workqueue_supported
    try:
        _ensure_supported()
        _ensure_workqueue_supported()
        return True
    except RuntimeError:
        return False

PLATFORM_SUPPORTS_WORKQUEUE_CONFIG: bool = LazyVal(lambda: evaluate_platform_supports_workqueue_config())

def evaluate_platform_supports_fp8():
    # [FUNCTION DOC] Returns whether fp8 is supported. On ROCm: gfx94, gfx120 (ROCm 6.3+), gfx95
    # (ROCm 6.5+) or gfx1250 (ROCm 7.14+) devices. On CUDA: SM90 or newer or capability 8.9. On
    # XPU: True. If no GPU is present, returns True because CPU supports fp8.
    if torch.cuda.is_available():
        if torch.version.hip:
            archs = ['gfx94']
            if ROCM_VERSION >= (6, 3):
                archs.extend(['gfx120'])
            if ROCM_VERSION >= (6, 5):
                archs.append('gfx95')
            if ROCM_VERSION >= (7, 14):
                archs.append('gfx1250')
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
            return False
        else:
            return SM90OrLater or torch.cuda.get_device_capability() == (8, 9)
    if torch.xpu.is_available():
        return True
    # As CPU supports FP8 and is always available, return True.
    return True

def evaluate_platform_supports_fp8_grouped_gemm():
    # [FUNCTION DOC] Returns whether fp8 grouped gemm is supported: on ROCm, requires the MSLK
    # build option and a gfx942 or gfx950 device; on CUDA, SM90 or newer but below SM100; False
    # without CUDA.
    if torch.cuda.is_available():
        if torch.version.hip:
            if "USE_MSLK" not in torch.__config__.show():
                return False
            # gfx1250 omitted: MSLK only builds gfx942/gfx950 kernels (see the arch
            # filter in aten/src/ATen/CMakeLists.txt). Add gfx1250 here once MSLK does.
            archs = ['gfx942', 'gfx950']
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return SM90OrLater and not SM100OrLater
    return False

def evaluate_platform_supports_mx_gemm():
    # [FUNCTION DOC] Returns whether MX-format gemm is supported: on ROCm 7.0+ for gfx950 (or
    # gfx1250 with ROCm 7.14+), on CUDA SM100 or newer, True on XPU, otherwise False.
    if torch.cuda.is_available():
        if torch.version.hip:
            if ROCM_VERSION >= (7, 0):
                gcn_name = torch.cuda.get_device_properties(0).gcnArchName
                return 'gfx950' in gcn_name or ('gfx1250' in gcn_name and ROCM_VERSION >= (7, 14))
        else:
            return SM100OrLater
    if torch.xpu.is_available():
        return True
    return False

def evaluate_platform_supports_mxfp8_grouped_gemm():
    # [FUNCTION DOC] Returns True only for NVIDIA CUDA builds with MSLK on SM100 devices (exact
    # match); False otherwise.
    if torch.cuda.is_available() and not torch.version.hip:
        built_with_mslk = "USE_MSLK" in torch.__config__.show()
        return built_with_mslk and IS_SM100
    return False

def evaluate_platform_supports_fp8_sparse():
    # [FUNCTION DOC] Returns whether fp8 sparse gemm is supported: on ROCm, gfx950 devices; on
    # CUDA, SM90 or newer or capability 8.9 with cuSPARSELt available at version 602 or newer;
    # False without a GPU.
    if torch.cuda.is_available():
        if torch.version.hip:
            return 'gfx950' in torch.cuda.get_device_properties(0).gcnArchName
        else:
            return (
                (SM90OrLater or torch.cuda.get_device_capability() == (8, 9))
                and torch.backends.cusparselt.is_available()
                and torch.backends.cusparselt.version() >= 602
            )
    return False

PLATFORM_SUPPORTS_MX_GEMM: bool = LazyVal(lambda: evaluate_platform_supports_mx_gemm())
PLATFORM_SUPPORTS_FP8: bool = LazyVal(lambda: evaluate_platform_supports_fp8())
PLATFORM_SUPPORTS_FP8_SPARSE: bool = LazyVal(lambda: evaluate_platform_supports_fp8_sparse())
PLATFORM_SUPPORTS_FP8_GROUPED_GEMM: bool = LazyVal(lambda: evaluate_platform_supports_fp8_grouped_gemm())
PLATFORM_SUPPORTS_MXFP8_GROUPED_GEMM: bool = LazyVal(lambda: evaluate_platform_supports_mxfp8_grouped_gemm())

if TEST_NUMBA:
    try:
        import numba.cuda
        TEST_NUMBA_CUDA = numba.cuda.is_available()
    except (ImportError, RuntimeError, OSError):
        TEST_NUMBA_CUDA = False
        TEST_NUMBA = False
else:
    TEST_NUMBA_CUDA = False

# Used below in `initialize_cuda_context_rng` to ensure that CUDA context and
# RNG have been initialized.
__cuda_ctx_rng_initialized = False


# after this call, CUDA context and RNG must have been initialized on each GPU
def initialize_cuda_context_rng():
    # [FUNCTION DOC] Ensures a CUDA context and RNG exist on every visible GPU by drawing a
    # random number on each, once per process (tracked by a module-level flag). Raises an error
    # if CUDA is not available.
    global __cuda_ctx_rng_initialized
    if not TEST_CUDA:
        raise AssertionError('CUDA must be available when calling initialize_cuda_context_rng')
    if not __cuda_ctx_rng_initialized:
        # initialize cuda context and rng for memory tests
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device=f"cuda:{i}")
        __cuda_ctx_rng_initialized = True


_tf32_off_lock = threading.Lock()
_tf32_off_depth = 0
_tf32_off_saved_precision = None
_tf32_off_cudnn_ctx = None


@contextlib.contextmanager
def tf32_off():
    # [FUNCTION DOC] Context manager that disables TF32 for CUDA matmul and cuDNN. It is
    # reentrant and thread safe: the first entry (depth counter under a lock) saves the matmul
    # fp32 precision and disables TF32, the last exit restores the saved state. Mutates
    # process-global backend flags.
    # First-in saves state and disables tf32; last-out restores. Multithreaded
    # test runners (e.g. MultiThreadedTestCase) enter this context once per
    # rank thread over the same process-global flags, so per-entry
    # save/restore can interleave and leak a modified state past the last
    # exit, which the TestCase fp32 precision leak detector then flags.
    global _tf32_off_depth, _tf32_off_saved_precision, _tf32_off_cudnn_ctx
    with _tf32_off_lock:
        if _tf32_off_depth == 0:
            # Snapshot fp32_precision (a string), not allow_tf32 (a bool):
            # writing allow_tf32 back can't reproduce the "none" default (it
            # yields "ieee"), which the leak detector would flag on ROCm.
            _tf32_off_saved_precision = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.allow_tf32 = False
            _tf32_off_cudnn_ctx = torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False)
            _tf32_off_cudnn_ctx.__enter__()
        _tf32_off_depth += 1
    try:
        yield
    finally:
        with _tf32_off_lock:
            _tf32_off_depth -= 1
            if _tf32_off_depth == 0:
                _tf32_off_cudnn_ctx.__exit__(None, None, None)
                _tf32_off_cudnn_ctx = None
                torch.backends.cuda.matmul.fp32_precision = _tf32_off_saved_precision
                _tf32_off_saved_precision = None


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    # [FUNCTION DOC] Context manager (takes the TestCase self) that enables TF32 for matmul and
    # cuDNN and temporarily sets self.precision to tf32_precision. On exit restores the matmul
    # fp32 precision and self.precision.
    old_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_fp32_precision
        self.precision = old_precision


@contextlib.contextmanager
def tf32_enabled():
    # [FUNCTION DOC] Context manager that enables TF32 for CUDA matmul and cuDNN and restores
    # the previous matmul fp32 precision on exit.
    """
    Context manager to temporarily enable TF32 for CUDA operations.
    Restores the previous TF32 state after exiting the context.
    """
    old_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_fp32_precision


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as specified by the
# argument. For example:
#    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
#    @tf32_on_and_off(0.005)
#    def test_matmul(self, device, dtype):
#        a = ...; b = ...;
#        c = torch.matmul(a, b)
#        self.assertEqual(c, expected)
# In the above example, when testing torch.float32 and torch.complex64 on CUDA
# on a CUDA >= 11 build on an >=Ampere architecture, the matmul will be running at
# TF32 mode and TF32 mode off, and on TF32 mode, the assertEqual will use reduced
# precision to check values.
#
# This decorator can be used for function with or without device/dtype, such as
# @tf32_on_and_off(0.005)
# def test_my_op(self)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device, dtype)
# @tf32_on_and_off(0.005)
# def test_my_op(self, dtype)
# if neither device nor dtype is specified, it will check if the system has ampere device
# if device is specified, it will check if device is cuda
# if dtype is specified, it will check if dtype is float32 or complex64
# tf32 and fp32 are different only when all the three checks pass
def tf32_on_and_off(tf32_precision=1e-5, *, only_if=True):
    # [FUNCTION DOC] Decorator factory for tests: the returned decorator wraps a test so it runs
    # twice (TF32 disabled, then TF32 enabled with relaxed tolerance tf32_precision) when TF32
    # is supported, only_if is true, the device argument (if present) is CUDA, and the dtype
    # argument (if present) is float32 or complex64; otherwise it runs once normally. The
    # wrapped test is called with all arguments as keywords, so the test needs a self parameter.
    def with_tf32_disabled(self, function_call):
        # [FUNCTION DOC] HELPER: Runs function_call with TF32 disabled.
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        # [FUNCTION DOC] HELPER: Runs function_call with TF32 enabled, using tf32_precision as
        # the test precision of self.
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        # [FUNCTION DOC] HELPER: Decorator applied to the test function f: records f's parameter
        # names and returns the wrapped test.
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Merges positional arguments into keyword arguments by
            # parameter name, checks whether the TF32 comparison applies, and runs f under TF32
            # off and on, or once normally.
            kwargs.update(zip(arg_names, args, strict=False))
            cond = torch.cuda.is_tf32_supported() and only_if
            if 'device' in kwargs:
                cond = cond and (torch.device(kwargs['device']).type == 'cuda')
            if 'dtype' in kwargs:
                cond = cond and (kwargs['dtype'] in {torch.float32, torch.complex64})
            if cond:
                with_tf32_disabled(kwargs['self'], lambda: f(**kwargs))
                with_tf32_enabled(kwargs['self'], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped
    return wrapper

# This is a wrapper that wraps a test to run it with TF32 turned off.
# This wrapper is designed to be used when a test uses matmul or convolutions
# but the purpose of that test is not testing matmul or convolutions.
# Disabling TF32 will enforce torch.float tensors to be always computed
# at full precision.
def with_tf32_off(f):
    # [FUNCTION DOC] Decorator that runs the test function f with TF32 disabled for its whole
    # body, returning the test's result.
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls f with all arguments inside the TF32-off context and
        # returns its result.
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped

def _get_magma_version():
    # [FUNCTION DOC] HELPER: Returns the MAGMA version as a tuple of integers parsed from the
    # build configuration text, or (0, 0) if MAGMA is not part of the build. Can raise
    # ValueError if the version text is not purely numeric.
    if 'Magma' not in torch.__config__.show():
        return (0, 0)
    position = torch.__config__.show().find('Magma ')
    version_str = torch.__config__.show()[position + len('Magma '):].split('\n')[0]
    return tuple(int(x) for x in version_str.split("."))

def _get_torch_cuda_version():
    # [FUNCTION DOC] HELPER: Returns torch.version.cuda parsed into a tuple of integers, or (0,
    # 0) if this is not a CUDA build.
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))

def _get_torch_rocm_version():
    # [FUNCTION DOC] HELPER: Returns the ROCm (HIP) version as a tuple of integers with any git
    # suffix removed, or (0, 0) if not running with ROCm.
    if not TEST_WITH_ROCM or torch.version.hip is None:
        return (0, 0)
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split("-", maxsplit=1)[0]    # ignore git sha
    return tuple(int(x) for x in rocm_version.split("."))

def _get_torch_hipblaslt_version():
    # [FUNCTION DOC] HELPER: Returns the hipBLASLt version as a (major, minor, patch) tuple
    # decoded from the packed integer reported by the C binding, or None if not on ROCm, the
    # value is zero/None, or the binding is missing or fails.
    if not TEST_WITH_ROCM:
        return None
    try:
        # Access through direct C binding
        # versionHipBLASLt returns: MAJOR * 10000 + MINOR * 100 + PATCH
        version_int = torch._C._cuda_getHipblasltVersion()
        if version_int is None or version_int == 0:
            return None
        major = version_int // 10000
        minor = (version_int % 10000) // 100
        patch = version_int % 100
        return (major, minor, patch)
    except (AttributeError, RuntimeError):
        return None

def _check_cusparse_generic_available():
    # [FUNCTION DOC] HELPER: Returns True when cuSPARSE generic API tests apply, i.e. when not
    # on ROCm.
    return not TEST_WITH_ROCM

def _check_hipsparse_generic_available():
    # [FUNCTION DOC] HELPER: Returns True when hipSPARSE generic API tests apply, i.e. on ROCm
    # with HIP version information present.
    return TEST_WITH_ROCM and torch.version.hip is not None


TEST_CUSPARSE_GENERIC = _check_cusparse_generic_available()
TEST_HIPSPARSE_GENERIC = _check_hipsparse_generic_available()

# Shared by test_torch.py and test_multigpu.py
def _create_scaling_models_optimizers(device="cuda", optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    # [FUNCTION DOC] HELPER: Builds two identical two-layer linear models on device (one for a
    # control run, one for gradient scaling) with matching initial weights, and an optimizer for
    # each created with optimizer_ctor, learning rate 1.0 and optimizer_kwargs overrides.
    # Returns the tuple (control model, scaling model, control optimizer, scaling optimizer).
    # Consumes the global RNG.
    # Create a module+optimizer that will use scaling, and a control module+optimizer
    # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
    mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    with torch.no_grad():
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters(), strict=True):
            s.copy_(c)

    kwargs = {"lr": 1.0}
    if optimizer_kwargs is not None:
        kwargs.update(optimizer_kwargs)
    opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
    opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)

    return mod_control, mod_scaling, opt_control, opt_scaling

# Shared by test_torch.py, test_cuda.py and test_multigpu.py
def _create_scaling_case(device="cuda", dtype=torch.float, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    # [FUNCTION DOC] HELPER: Builds a gradient-scaling test case: returns the models and
    # optimizers from the models/optimizers helper plus a list of four random (input, target)
    # batches, an MSE loss function, and the iteration index (2) at which the test should skip
    # a step.
    data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

    loss_fn = torch.nn.MSELoss().to(device)

    skip_iter = 2

    return _create_scaling_models_optimizers(
        device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
    ) + (data, loss_fn, skip_iter)


def xfailIfSM89(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure when running on an
    # SM89 GPU, otherwise returns func unchanged.
    return func if not IS_SM89 else unittest.expectedFailure(func)

def xfailIfSM90(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on SM90 GPUs,
    # otherwise returns func unchanged.
    return func if not IS_SM90 else unittest.expectedFailure(func)

def xfailIfSM89PreCUDA13(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on SM89 GPUs only when
    # the CUDA version is below 13; otherwise returns func unchanged.
    """xfail on SM89 only for CUDA < 13. On CUDA 13+, test should pass on all architectures."""
    if IS_SM89 and _get_torch_cuda_version() < (13, 0):
        return unittest.expectedFailure(func)
    return func

def xfailIfSM100OrLater(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on NVIDIA GPUs with
    # SM100 or newer; never applies on ROCm, where it returns func unchanged.
    # SMxxx LazyVals are derived from torch.cuda.get_device_capability(), which
    # on ROCm reports the gfx-arch major version (e.g. (11, 0) for gfx1100). That
    # makes SM100OrLater spuriously true on AMD gfx10/11/12, so guard with
    # TEST_WITH_ROCM to keep the xfail NVIDIA-only.
    if TEST_WITH_ROCM:
        return func
    return func if not SM100OrLater else unittest.expectedFailure(func)

def xfailIfSM120OrLater(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on NVIDIA GPUs with
    # SM120 or newer; never applies on ROCm.
    if TEST_WITH_ROCM:
        return func
    return func if not SM120OrLater else unittest.expectedFailure(func)

def xfailIfSM12X(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on SM12X GPUs.
    return func if not IS_SM12X else unittest.expectedFailure(func)

def xfailIfDistributedNotSupported(func):
    # [FUNCTION DOC] HELPER: Decorator: marks func as an expected failure on macOS or Jetson,
    # where distributed is not supported.
    return func if not (IS_MACOS or IS_JETSON) else unittest.expectedFailure(func)

def _xfail_cuda_on_windows_wrapper(test_fn):
    # [FUNCTION DOC] HELPER: Wraps test_fn so that running on a CUDA device (or a plain TestCase
    # with no device type) is treated as an expected failure: a raised exception becomes a
    # pytest xfail, and an unexpected success raises an error. Tests on non-CUDA device types
    # run normally.
    """Wraps a test to xfail when running on a CUDA device.

    Works for both device-parameterized tests (device_type == "cuda") and plain
    TestCase CUDA tests (device_type is None). Only passes through for tests
    explicitly running on a non-CUDA device type (e.g. "cpu", "xpu").
    """
    @functools.wraps(test_fn)
    def wrapper(slf, *args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper described by the enclosing function; imports pytest
        # at call time and returns the test's result for non-CUDA device types.
        import pytest
        device_type = getattr(slf, "device_type", None)
        if device_type is not None and device_type != "cuda":
            return test_fn(slf, *args, **kwargs)
        try:
            test_fn(slf, *args, **kwargs)
        except Exception as e:
            pytest.xfail(f"Expected failure for {test_fn.__name__}: {e}")

        raise AssertionError(f"Test {test_fn.__name__} was expected to fail but succeeded.")
    return wrapper


def xfailCUDAIfSM89OrLaterOnWindows(test_fn):
    # [FUNCTION DOC] Decorator: on Windows with SM89 or newer GPUs, marks a CUDA test as an
    # expected failure (via the helper); otherwise returns test_fn unchanged.
    """Mark a CUDA test as expected failure on Windows with SM >= 8.9.

    Works for both device-parameterized tests and plain TestCase CUDA tests.
    CPU/XPU variants of device-parameterized tests are unaffected.
    """
    return _xfail_cuda_on_windows_wrapper(test_fn) if IS_WINDOWS and SM89OrLater else test_fn


# When using nvcc from the CUDA toolkit its versuib must be at least the one from ptxas bundled with Triton
TRITON_PTXAS_VERSION = (12, 8)
requires_triton_ptxas_compat = unittest.skipIf(not torch.version.xpu
                                               and torch.version.hip is None
                                               and _get_torch_cuda_version() < TRITON_PTXAS_VERSION,
                                               "Requires CUDA {}.{} to match Tritons ptxas version".format(*TRITON_PTXAS_VERSION))

# Importing this module should NOT eagerly initialize CUDA
if not CUDA_ALREADY_INITIALIZED_ON_IMPORT:
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA should not be initialized on import")
