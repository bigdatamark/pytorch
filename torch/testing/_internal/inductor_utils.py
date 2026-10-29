# mypy: ignore-errors

# [MODULE DOC]
# Helpers for Inductor tests: feature-detection flags (HAS_CPU, HAS_TRITON, HAS_PALLAS,
# HAS_HELION, HAS_GPU, GPU_TYPE, RUN_CPU, RUN_GPU, RUN_TPU, IS_A100, IS_H100, IS_BIG_GPU,
# many of them lazily evaluated), skip decorators (requires_gpu, requires_triton,
# skipCUDAIf and similar), float8 quantization helpers, a mock graph handler, and context
# managers to patch the Inductor backend registration or add a custom fallback pass.
# Import-time side effects: probes for Triton, GPUs and device test bases, may import triton,
# and creates the module logger.

import contextlib
import functools
import logging
import os
import re
import sys
import unittest
from subprocess import CalledProcessError

import torch
import torch._inductor.config as config

from torch._inductor import compile_fx  # noqa: F401
from torch._inductor.utils import (
    get_gpu_shared_memory,
    get_gpu_type,
    GPU_TYPES,
    is_big_gpu,
    is_gpu,
    OrderedSet,
)
from torch.utils._helion import has_helion
from torch.utils._pallas import has_pallas_package, has_tpu_pallas
from torch.utils._triton import has_triton
from torch.utils._config_module import ConfigModule
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    LazyVal,
    TestCase,
)

from collections.abc import Callable

log: logging.Logger = logging.getLogger(__name__)


def test_cpu():
    # [FUNCTION DOC] HELPER: Detects whether a C++ compiler works for Inductor by loading an
    # empty C++ code cache entry. Returns True on success and False if the compile or
    # compiler lookup raises a process, OS, invalid-compiler or compile error. May compile.
    try:
        from torch._inductor.codecache import CppCodeCache

        CppCodeCache.load("")
        return True
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False


HAS_CPU = LazyVal(test_cpu)

HAS_TRITON = has_triton()

HAS_PALLAS = LazyVal(has_pallas_package)

HAS_HELION = has_helion()

if HAS_TRITON:
    import triton

    TRITON_HAS_CPU = "cpu" in triton.backends.backends
else:
    TRITON_HAS_CPU = False


HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON

HAS_MTIA_AND_TRITON = torch.mtia.is_available() and HAS_TRITON

HAS_XPU_AND_TRITON = torch.xpu.is_available() and HAS_TRITON

HAS_MPS = torch.mps.is_available()

HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON or HAS_MTIA_AND_TRITON
HAS_GPU_AND_TRITON = HAS_GPU

GPU_TYPE = get_gpu_type()

HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPU_TYPES
)

_desired_test_bases = get_desired_device_type_test_bases(allow_xpu=True)
RUN_GPU = HAS_GPU and any(
    is_gpu(getattr(x, "device_type", "")) for x in _desired_test_bases
)

RUN_CPU = LazyVal(
    lambda: HAS_CPU
    and any(getattr(x, "device_type", "") == "cpu" for x in _desired_test_bases)
)

HAS_TPU = LazyVal(has_tpu_pallas)
# TPU is a privateuse1 backend that isn't in _desired_test_bases (it requires
# runtime initialization before the test base is registered). Check the env var
# directly, matching the same semantics as RUN_CPU/RUN_GPU: when the env var is
# unset, run if the hardware is available; when set, only run if "tpu" is listed.
RUN_TPU = LazyVal(
    lambda: HAS_TPU
    and (
        "tpu" in os.environ.get("PYTORCH_TESTING_DEVICE_ONLY_FOR", "").split(",")
        if os.environ.get("PYTORCH_TESTING_DEVICE_ONLY_FOR", "")
        else True
    )
)


def _check_has_dynamic_shape(
    self: TestCase,
    code,
):
    # [FUNCTION DOC] HELPER: Test assertion on generated C++ source text. Scans lines for a
    # for loop, and for one whose header has a symbolic size variable (name containing ks)
    # between semicolons. Fails the test via assertTrue if no dynamic loop or no loop is found.
    for_loop_found = False
    has_dynamic = False
    lines = code.split("\n")
    for line in lines:
        if "for(" in line:
            for_loop_found = True
            if re.search(r";.*ks.*;", line) is not None:
                has_dynamic = True
                break
    self.assertTrue(
        has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
    )
    self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")


def skipDeviceIf(cond, msg, *, device):
    # [FUNCTION DOC]
    # Builds a test-method decorator that skips the test when cond is true and the test
    # instance's device attribute equals the given device string (raising SkipTest with msg).
    # If cond is false the decorator returns the function unchanged. Returns the decorator.
    if cond:

        def decorate_fn(fn):
            # [FUNCTION DOC] HELPER: Wraps fn so the device check happens at call time.
            @functools.wraps(fn)
            def inner(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Logs a warning if the test instance has no device
                # attribute, skips when the device matches, else calls the wrapped test.
                # KNOWN ISSUE: after warning about a missing device attribute it still reads
                # self.device, which raises AttributeError.
                if not hasattr(self, "device"):
                    warn_msg = (
                        "Expect the test class to have attribute device but not found. "
                    )
                    if hasattr(self, "device_type"):
                        warn_msg += "Consider using the skip device decorators in common_device_type.py"
                    log.warning(warn_msg)
                if self.device == device:
                    raise unittest.SkipTest(msg)
                return fn(self, *args, **kwargs)

            return inner

    else:

        def decorate_fn(fn):
            # [FUNCTION DOC] HELPER: No-op decorator; returns fn unchanged.
            return fn

    return decorate_fn


def skip_windows_ci(name: str, file: str) -> None:
    # [FUNCTION DOC]
    # On Windows CI only: writes a message to stderr naming the test module, then exits the
    # process with status 0 if name is the main module, else raises SkipTest. No-op elsewhere.
    # KNOWN ISSUE: the module name is computed by stripping any of the characters '.', 'p', 'y'
    # from both ends of the basename, not by removing the .py suffix, so names are mangled.
    if IS_WINDOWS and IS_CI:
        module = os.path.basename(file).strip(".py")
        sys.stderr.write(
            f"Windows CI does not have necessary dependencies for {module} tests yet\n"
        )
        if name == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("requires sympy/functorch/filelock")


# TODO: Remove HAS_MPS condition  when `HAS_GPU` includes HAS_MPS
requires_gpu = functools.partial(
    unittest.skipIf, not (HAS_GPU or HAS_MPS), "requires gpu"
)
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, "requires triton")
requires_helion = functools.partial(unittest.skipIf, not HAS_HELION, "requires helion")


def requires_gpu_with_enough_memory(min_mem_required):
    # [FUNCTION DOC]
    # Builds a decorator that skips the decorated test unless an XPU or CUDA device with at
    # least min_mem_required bytes of memory is available. Returns the decorator.
    def inner(fn):
        # [FUNCTION DOC] HELPER: Queries total memory of the current XPU (preferred) or CUDA
        # device at decoration time and returns either a skipped version of fn or fn itself.
        total_memory = sys.maxsize
        if torch.xpu.is_available():
            total_memory = torch.xpu.get_device_properties().total_memory
        elif torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties().total_memory
        if (
            not (torch.cuda.is_available() or torch.xpu.is_available())
            or total_memory < min_mem_required
        ):
            return unittest.skip(
                f"Only if the GPU device has at least {min_mem_required / 1e9:.3f}GB memory to be safe"
            )(fn)
        else:
            return fn

    return inner


skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")

IS_A100 = LazyVal(lambda: HAS_CUDA_AND_TRITON and get_gpu_shared_memory() == 166912)

IS_H100 = LazyVal(lambda: HAS_CUDA_AND_TRITON and get_gpu_shared_memory() == 232448)

IS_BIG_GPU = LazyVal(lambda: HAS_GPU_AND_TRITON and is_big_gpu())


def dummy_graph():
    # [FUNCTION DOC]
    # Builds a GraphLowering from a fake-traced addition of two random tensors with a shape
    # environment derived from the example inputs. Returns the graph object, for tests that
    # need a graph to access size variables. Side effects: none beyond random tensor creation.
    """
    Create a graph. This is useful for unit testing code which accesses
    V.graph.sizevars.
    """
    from torch._inductor.compile_fx import shape_env_from_inputs
    from torch._inductor.graph import GraphLowering
    from torch.fx.experimental.proxy_tensor import make_fx

    example_inputs = [torch.randn(10) for _ in range(2)]
    gm = make_fx(torch.add, tracing_mode="fake")(*example_inputs)
    shape_env = shape_env_from_inputs(example_inputs)
    graph = GraphLowering(
        gm,
        shape_env=shape_env,
    )

    return graph


def maybe_skip_size_asserts(op):
    # [FUNCTION DOC]
    # Returns a context manager. For FFT and a few linalg ops whose meta and eager strides
    # differ, and when the TORCHINDUCTOR_SIZE_ASSERTS environment variable is unset, it is the
    # Inductor config patch disabling size asserts; otherwise a null context.
    """
    For certain ops, there meta and eager implementation returns different
    strides. This cause size/strides assert fail. Skip adding those
    asserts for now.
    """
    if (
        op.aten_name
        in (
            "fft_hfftn",
            "fft_hfft",
            "fft_hfft2",
            "fft_ihfftn",
            "fft_fft",
            "fft_fft2",
            "fft_fftn",
            "fft_ifft",
            "fft_ifft2",
            "fft_ifftn",
            "fft_irfft",
            "fft_irfft2",
            "fft_irfftn",
            "fft_ihfft",
            "fft_ihfft2",
            "fft_rfft",
            "fft_rfft2",
            "fft_rfftn",
            "linalg_eig",
            "linalg_eigvals",
        )
        and "TORCHINDUCTOR_SIZE_ASSERTS" not in os.environ
    ):
        return torch._inductor.config.patch(size_asserts=False)
    else:
        return contextlib.nullcontext()


def get_func_call() -> str:
    # [FUNCTION DOC]
    # Returns the text that begins the generated entry function: the C++ entry signature
    # when cpp_wrapper is enabled in the Inductor config, else the Python call definition.
    return (
        "void inductor_entry_impl("
        if torch._inductor.config.cpp_wrapper
        else "def call("
    )


def get_kernel_launch() -> str:
    # [FUNCTION DOC]
    # Returns the text that identifies a kernel launch in generated code: the C++ triton call
    # prefix when cpp_wrapper is enabled, else the Python run call.
    return "call_triton_" if torch._inductor.config.cpp_wrapper else ".run("


def clone_preserve_strides_offset(x, device=None):
    # [FUNCTION DOC]
    # Copies a tensor while preserving its sizes, strides and storage offset (including
    # unused storage outside the view). Non-tensors are returned unchanged. The whole underlying
    # storage is viewed flat, cloned (or copied to device if given), and re-viewed with the
    # original geometry. Returns the new tensor.
    if not isinstance(x, torch.Tensor):
        return x
    buffer = torch.as_strided(
        x, (x.untyped_storage().size() // x.element_size(),), (1,), 0
    )
    if not device:
        buffer = buffer.clone()
    else:
        buffer = buffer.to(device, copy=True)
    out = torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    return out


# define the e4m3/e5m2 constants
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
E4M3FNUZ_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
E5M2FNUZ_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max

FP16_MAX_POS: float = torch.finfo(torch.float16).max
EPS: float = 1e-12

Tensor = torch.Tensor


def _to_fp8_saturated(x: Tensor, float8_dtype: torch.dtype) -> Tensor:
    # [FUNCTION DOC] HELPER: Clamps x to the representable range of the given float8 dtype
    # (e4m3fn, e5m2, e4m3fnuz or e5m2fnuz) then casts to it, i.e. saturating instead of
    # overflowing. Raises TypeError for any other dtype.
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif float8_dtype == torch.float8_e5m2:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        x = x.clamp(min=-1 * E4M3FNUZ_MAX_POS, max=E4M3FNUZ_MAX_POS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        x = x.clamp(min=-1 * E5M2FNUZ_MAX_POS, max=E5M2FNUZ_MAX_POS)
    else:
        raise TypeError(f"Unsupported float8_dtype: {float8_dtype}")
    return x.to(float8_dtype)


@torch.no_grad()
def _amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts absolute-max values to float8 scales (max representable
    # value divided by amax floored at a tiny epsilon), in float32; e4m3fn uses its maximum,
    # every other dtype uses the e5m2 maximum. For float16 originals the scale is capped at the
    # float16 maximum. Runs without gradient tracking.
    # To make scale dtype to be fp32 for accuracy
    amax = amax.float()
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    return res


def _quantize_tensorwise(x: Tensor, float8_dtype: torch.dtype):
    # [FUNCTION DOC] HELPER: Quantizes x to float8 with one scale for the whole tensor.
    # Returns the saturated float8 tensor and the reciprocal scale.
    amax = torch.max(torch.abs(x))
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


def _quantize_rowwise(x: Tensor, float8_dtype: torch.dtype):
    # [FUNCTION DOC] HELPER: Quantizes a 2D tensor to float8 with one scale per row (amax
    # along dim 1, kept as a column). Returns the float8 tensor and the reciprocal scales.
    amax = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


def _quantize_blockwise(
    x: Tensor, float8_dtype: torch.dtype, block_outer: int, block_inner: int
):
    # [FUNCTION DOC] HELPER: Quantizes a 2D tensor to float8 with one scale per block of
    # (block_outer x block_inner) elements (block sizes capped at the tensor dims). Scales are
    # expanded back to element granularity for the division. Returns the float8 tensor
    # and the per-block reciprocal scales.
    min_outer = min(block_outer, x.shape[0])
    min_inner = min(block_inner, x.shape[1])
    x = x.unflatten(1, (-1, min_inner)).unflatten(0, (-1, min_outer))
    amax = x.abs().amax(dim=[1, 3], keepdim=True).float()
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x = x.flatten(2, 3).flatten(0, 1)
    scale = scale.flatten(2, 3).flatten(0, 1)
    scale_expanded = scale.repeat_interleave(min_outer, dim=0).repeat_interleave(
        min_inner, dim=1
    )
    x_fp8 = _to_fp8_saturated(
        x / scale_expanded,  # Ensures that scaling doesn't cause inf/nan values
        float8_dtype,
    )
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


class MockGraphHandler:
    # [CLASS DOC] HELPER: Minimal stand-in for a graph lowering object, to install as the
    # virtualized graph in tests. State: a size-variable allocator, a mapping from buffer name
    # to buffer, and empty containers for graph inputs, mutated and removed buffers, constants,
    # plus a scheduler of None. Provides buffer lookup and a fixed float32 dtype.
    """Minimal mock graph handler for testing virtualized context."""

    def __init__(self, name_to_buffer=None):
        # [FUNCTION DOC] HELPER: Creates a fresh size-variable allocator and stores the
        # optional buffer mapping (empty mapping if not given) and empty bookkeeping containers.
        import torch._inductor.sizevars

        self.sizevars = torch._inductor.sizevars.SizeVarAllocator()
        self.name_to_buffer = name_to_buffer or {}
        self.graph_inputs = {}
        self.mutated_buffers = OrderedSet()
        self.removed_buffers = OrderedSet()
        self.constants = {}
        self.scheduler = None

    def try_get_buffer(self, buffer_name: str):
        # [FUNCTION DOC] HELPER: Returns the buffer registered under the name, or None.
        return self.name_to_buffer.get(buffer_name)

    def get_buffer(self, buffer_name: str):
        # [FUNCTION DOC] HELPER: Returns the buffer registered under the name; raises KeyError
        # if absent.
        return self.name_to_buffer[buffer_name]

    def get_dtype(self, buffer_name: str) -> torch.dtype:
        # [FUNCTION DOC] HELPER: Ignores the name and always returns float32.
        """Return default dtype for any buffer (for testing)."""
        return torch.float32


def has_cpp_wrapper_for_device(device: str) -> bool:
    # [FUNCTION DOC]
    # Initializes Inductor backend registration (global side effect) and returns whether a C++
    # wrapper code generator is registered for the given device string.
    from torch._inductor.codegen.common import (
        get_wrapper_codegen_for_device,
        init_backend_registration,
    )

    init_backend_registration()
    return get_wrapper_codegen_for_device(device, cpp_wrapper=True) is not None


@contextlib.contextmanager
def patch_inductor_backend(
    device: str,
    python_wrapper_codegen=None,
    custom_pass=None,
    custom_backend_config: ConfigModule = None,
):
    # [FUNCTION DOC]
    # Context manager that temporarily re-registers the Inductor backend of a device with an
    # overridden Python wrapper code generator, custom graph pass and/or custom backend config
    # (each falls back to the existing registration when None), keeping the scheduling and the
    # C++ and FX wrappers. The original registration is restored on exit, even on error.
    # Mutates the global backend registry.
    """
    Patch the inductor backend for a specific device.
    """
    from torch._inductor.codegen.common import (
        get_custom_backend_config_for_device,
        get_custom_backend_pass_for_device,
        get_scheduling_for_device,
        get_wrapper_codegen_for_device,
        init_backend_registration,
        register_backend_for_device,
    )

    # Make sure the backend is already registered
    init_backend_registration()

    # Get the original registration parameters
    original_scheduling = get_scheduling_for_device(device)
    original_python_wrapper = get_wrapper_codegen_for_device(device, False)
    original_cpp_wrapper = get_wrapper_codegen_for_device(device, True)
    original_fx_wrapper = get_wrapper_codegen_for_device(device, fx_wrapper=True)
    original_custom_pass = get_custom_backend_pass_for_device(device)
    original_custom_backend_config = get_custom_backend_config_for_device(device)

    try:
        # Register modified backend for the device
        register_backend_for_device(
            device,
            original_scheduling,
            (
                python_wrapper_codegen
                if python_wrapper_codegen is not None
                else original_python_wrapper
            ),
            original_cpp_wrapper,
            original_fx_wrapper,
            custom_pass if custom_pass is not None else original_custom_pass,
            (
                custom_backend_config
                if custom_backend_config is not None
                else original_custom_backend_config
            ),
        )
        yield
    finally:
        # Restore the original backend
        register_backend_for_device(
            device,
            original_scheduling,
            original_python_wrapper,
            original_cpp_wrapper,
            original_fx_wrapper,
            original_custom_pass,
            original_custom_backend_config,
        )

def patch_custom_fallback_pass(predicate: Callable[[torch.fx.Node], bool]) -> contextlib.ContextDecorator:
    # [FUNCTION DOC]
    # Returns a config patch (usable as a context manager or decorator) that installs a custom
    # post-grad pre-pass which sets a should-fallback flag in the metadata of every graph node
    # for which predicate returns true.
    """
    Create a custom pass which falls back based on the provided predicate. For example,
    we could provide a predicate which returns True for all aten.add.default nodes.
    Returns a context activating the pass.
    """
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class Pass(CustomGraphPass):
        # [CLASS DOC] HELPER: Custom graph pass that marks nodes selected by the enclosing
        # predicate for fallback. Stateless; reports no cache uuid.
        def __call__(self, graph: torch.fx.Graph):
            # [FUNCTION DOC] HELPER: Mutates the graph in place: sets the should-fallback
            # metadata flag on each node where the predicate is true. Returns nothing.
            for node in graph.nodes:
                if predicate(node):
                    node.meta["should_fallback"] = True

        def uuid(self):
            # [FUNCTION DOC] HELPER: Returns None, meaning the pass is not cache-keyable.
            return None


    return config.patch(post_grad_custom_pre_pass=Pass())
