# Owner(s): ["module: inductor"]
# [MODULE DOC] Test suite for AOTInductor (ahead-of-time compiled exported models). It defines a
# large template class of device-generic tests (compile an exported model to a shared library or
# package, load it, compare against eager, and inspect generated C++ code). The template is copied
# into concrete TestCase classes for CPU, GPU, GPU with lazy autotuning (dual wrapper) and MPS,
# with per-device expected-failure/skip tables. It also holds logging, standalone-config and
# lower-bound-check tests and a main entry that runs the tests when GPU or macOS is available.
# Import-time effects: exits or skips early on Windows CI or if helper test modules cannot be
# imported; if a GPU with triton exists, triton test kernels are imported; parametrized tests are
# instantiated and the template tests are copied onto the concrete classes.
import contextlib
import functools
import itertools
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile
from unittest import skip
from unittest.mock import patch

import torch
import torch._export
import torch._inductor
import torch._inductor.config
import torch.nn as nn
from torch._dynamo import config as dynamo_config
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch._inductor import config
from torch._inductor.codecache import WritableTempFile
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.package import package_aoti
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch._inductor.utils import (
    is_big_gpu,
    maybe_aoti_standalone_config,
    run_and_get_cpp_code,
)
from torch._library import capture_triton
from torch._utils_internal import full_aoti_runtime_assert
from torch.export import Dim, export
from torch.export.pt2_archive._package import load_pt2
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import (
    CDNA2OrLater,
    CDNA5OrLater,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_FP8_GROUPED_GEMM,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    requires_triton_ptxas_compat,
    SM80OrLater,
    SM90OrLater,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    _has_sufficient_memory,
    e4m3_type,
    e5m2_type,
    skipCUDAIf,
)
from torch.testing._internal.common_dtype import (
    highest_precision_complex,
    highest_precision_float,
)
from torch.testing._internal.common_quantization import (
    _group_quantize_tensor,
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    MACOS_VERSION,
    NAVI_ARCH,
    parametrize,
    random_matrix_with_scaled_reduction_dim,
    runOnRocm,
    skipIfRocm,
    skipIfRocmArch,
    skipIfWindows,
    skipIfWindowsXPU,
    skipIfXpu,
    TEST_MPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.custom_tensor import CustomTensorPlainOut
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_XPU_AND_TRITON,
    IS_BIG_GPU,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import requires_gpu
from torch.utils import _pytree as pytree
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_tensor_descriptor_host_tma,
)


f8_msg = "FP8 is only supported on H100+, SM 8.9 and MI300+, XPU and CPU devices"


@contextlib.contextmanager
def caching_allocator_disabled():
    # [FUNCTION DOC] HELPER: Context manager. On CUDA it enters the CUDA
    # caching-allocator-disabled context for the body so allocations hit the driver directly; on
    # other devices it does nothing.
    if GPU_TYPE == "cuda":
        from torch.cuda import (
            caching_allocator_disabled as _cuda_caching_allocator_disabled,
        )

        with _cuda_caching_allocator_disabled():
            yield
    else:
        yield


def requires_autotune_at_compile_time(fn):
    # [FUNCTION DOC] HELPER: Decorator for test methods. The wrapped test raises a SkipTest when
    # the inductor config option triton.autotune_at_compile_time is explicitly False; otherwise the
    # test is called unchanged with its arguments.
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper produced by the decorator; checks the config flag at call
        # time, skips the test when it is False, otherwise forwards to the original test and
        # returns its result.
        if config.triton.autotune_at_compile_time is False:
            raise unittest.SkipTest("requires triton.autotune_at_compile_time=True")
        return fn(self, *args, **kwargs)

    return wrapper


@contextlib.contextmanager
def use_fa3():
    # [FUNCTION DOC] HELPER: Context manager that activates the FA3 flash attention backend for the
    # body and restores the previous implementation on exit. Raises SkipTest if activating FA3
    # fails with a missing module or runtime error (flash_attn_interface not available).
    try:
        activate_flash_attention_impl("FA3")
    except (ModuleNotFoundError, RuntimeError) as err:
        raise unittest.SkipTest(
            "FA3 backend not available (flash_attn_interface missing)"
        ) from err
    try:
        yield
    finally:
        restore_flash_attention_impl()


if HAS_GPU:
    import triton  # @manual
    from triton import language as tl

    from torch.testing._internal.triton_utils import (
        add_kernel,
        add_kernel_2d_autotuned,
        add_kernel_autotuned,
        add_kernel_autotuned_weird_param_order,
        add_kernel_on_device_tma_new_api,
        add_kernel_on_device_tma_old_api,
        add_kernel_with_boolean_param,
        add_kernel_with_none_param_and_equal_to_1_arg,
        add_kernel_with_optional_param,
        add_kernel_with_scaling,
        add_kernel_with_tma_1d_new_api,
        add_kernel_with_tma_1d_old_api,
        add_kernel_with_tma_2d_new_api,
        add_kernel_with_tma_2d_old_api,
        create_tensor_descriptor_shim,
        masked_add_kernel_with_bool_tensor,
        mul2_inplace_kernel,
        strange_config_matmul_kernel,
        sub_kernel_autotuned,
    )

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_control_flow import (
            CondModels,
            prepend_counters,
            prepend_predicates,
            SwitchModels,
            WhileLoopModels,
        )
        from .test_torchinductor import copy_tests, requires_multigpu, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_control_flow import (  # @manual=fbcode//caffe2/test/inductor:control_flow-library
            CondModels,
            prepend_counters,
            prepend_predicates,
            SwitchModels,
            WhileLoopModels,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            requires_multigpu,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


def get_module_ext_type():
    # [FUNCTION DOC] HELPER: Returns the extension of the compiled model library: "pyd" on Windows,
    # "so" elsewhere.
    if IS_WINDOWS:
        return "pyd"
    else:
        return "so"


def get_triton_grid_info(kernel, total_elements, src_code):
    # [FUNCTION DOC] HELPER: Computes the expected triton launch grid sizes for a kernel with block
    # configs, as the ceiling division of total_elements by the product of the BLOCK_SIZE_M and
    # BLOCK_SIZE_N of each autotune config. Also searches the generated source code for the C++
    # declaration of the first grid dimension. Returns a tuple of (grid found in the code as int,
    # or None when absent; list of expected grids per config).
    expected_grids = [
        triton.cdiv(
            total_elements, cfg.kwargs["BLOCK_SIZE_M"] * cfg.kwargs["BLOCK_SIZE_N"]
        )
        for cfg in kernel.configs
    ]
    grid_match = re.search(r"uint32_t grid_0 = (\d+)L;", src_code)
    if grid_match:
        return int(grid_match.group(1)), expected_grids
    else:
        return None, expected_grids


class AOTInductorTestsTemplate:
    # [CLASS DOC] Template of AOTInductor tests, not a TestCase itself. It is mixed into concrete
    # TestCase classes (per device and configuration) which supply self.device, self.check_model,
    # self.code_check_count, self.use_minimal_arrayref_interface and similar attributes. Each test
    # builds a small model, compiles it with AOTInductor under some config patch, and checks
    # numerics and/or properties of the generated C++ code.
    @common_utils.parametrize("embed_kernel_binary", [False, True])
    @common_utils.parametrize("max_autotune", [False, True])
    def test_simple(self, embed_kernel_binary, max_autotune):
        # [FUNCTION DOC] Compiles a linear-plus-add model with every combination of
        # embed_kernel_binary and max_autotune, checks numerics, and checks the generated C++ code
        # for the device-specific kernel launch (MPS function lookup, or a GPU launchKernel that
        # must not name a binary file when the kernel binary is embedded). Skips max_autotune on
        # CPU/macOS. With the minimal arrayref interface it also checks the run entry point appears
        # exactly once.
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding one 10x10 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "max_autotune": max_autotune,
            }
        ):
            self.check_model(model, example_inputs)

            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            if self.device == "mps":
                FileCheck().check("aoti_torch_mps_get_kernel_function(").run(code)
            elif self.device == GPU_TYPE:
                FileCheck().check("launchKernel(").run(code)
                if config.aot_inductor.embed_kernel_binary:
                    # Not expect to see launchKernel("CUBIN_FILE_NAME"
                    FileCheck().check_not('launchKernel("').run(code)

        if self.use_minimal_arrayref_interface:
            self.code_check_count(
                model, example_inputs, "AOTInductorModelRunMinimalArrayrefInterface(", 1
            )

    @common_utils.parametrize("embed_kernel_binary", [False, True])
    def test_loaded_modules_tracking(self, embed_kernel_binary):
        # [FUNCTION DOC] Checks that on CUDA/HIP the generated code passes the loaded-modules
        # container to the kernel loader so module handles are tracked and unloaded on
        # destruction (avoids GPU code object leaks). Parametrized on embed_kernel_binary.
        # Skipped on non-CUDA devices.
        # Verify that AOTI codegen on CUDA/HIP passes &kernels_.loaded_modules_
        # to loadKernel so CUmodule handles are tracked and unloaded on
        # destruction, preventing GPU code object leaks.
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA/HIP")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a 10x10 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model().to(self.device)
        with config.patch({"aot_inductor.embed_kernel_binary": embed_kernel_binary}):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            FileCheck().check("&kernels_.loaded_modules_").run(code)

    def test_triton_kernel_bool_param(self):
        # [FUNCTION DOC] Verifies a user triton kernel that takes a boolean scalar parameter
        # compiles and runs correctly through AOTInductor. Skipped unless the device is a
        # non-MPS GPU.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that launches the add kernel with a boolean add_xy flag
            # over a zeros output and returns the output.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Allocates a zeros output and launches the triton kernel on
                # one block, then returns the output.
                out = torch.zeros_like(x)
                add_kernel_with_boolean_param[1,](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=x.numel(),
                    add_xy=True,
                    BLOCK_SIZE=1,
                )
                return out

        inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), inputs)

    def test_triton_kernel_bool_tensor_arg(self):
        # [FUNCTION DOC] Verifies a user triton kernel taking a boolean tensor mask argument works
        # through AOTInductor, using a mask that selects the first half of 128 elements. Skipped
        # without a GPU.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that runs a masked-add triton kernel with a bool
            # tensor mask.
            def forward(self, x, y, mask):
                # [FUNCTION DOC] HELPER: Launches the masked add kernel with one program per
                # element over a zeros output, returns that output.
                out = torch.zeros_like(x)
                n = x.numel()
                masked_add_kernel_with_bool_tensor[(n,)](
                    in_ptr0=x,
                    in_ptr1=y,
                    mask_ptr=mask,
                    out_ptr=out,
                    n_elements=n,
                    BLOCK_SIZE=1024,
                )
                return out

        n = 128
        x = torch.randn(n, device=self.device)
        y = torch.randn(n, device=self.device)
        mask = torch.arange(n, device=self.device) < n // 2
        inputs = (x, y, mask)
        self.check_model(Model(), inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "toolchain doesn't support ptx to fatbin",
    )
    # Skip embed_kernel_binary == True for now as it shows random
    # failure on CI
    @common_utils.parametrize("embed_kernel_binary", [False])
    @requires_triton_ptxas_compat
    def test_simple_multi_arch(self, embed_kernel_binary):
        # [FUNCTION DOC] Compiles with multi-arch kernel emission enabled and checks numerics; when
        # the kernel binary is not embedded, checks the generated code mentions the
        # arch-independent kernel file extension (spv on XPU, hsaco on ROCm, fatbin otherwise).
        # Skipped in fbcode and without a GPU; requires triton ptxas compat. Only
        # embed_kernel_binary False is parametrized.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the output of a 10x16 linear layer applied to y
            # to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 16)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 16, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                "aot_inductor.emit_multi_arch_kernel": True,
            }
        ):
            self.check_model(model, example_inputs)
            if not embed_kernel_binary:
                _, code = run_and_get_cpp_code(
                    AOTIRunnerUtil.compile, model, example_inputs
                )
                file_extension = (
                    ".spv"
                    if self.device == "xpu"
                    else (".hsaco" if torch.version.hip else ".fatbin")
                )
                FileCheck().check(file_extension).run(code)

    def test_small_constant(self):
        # [FUNCTION DOC] Checks numerics of a 4x4 linear model compiled with small tensor constants
        # always kept (not inlined).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a 4x4 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.linear(x)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    @unittest.skipIf(
        not HAS_GPU or GPU_TYPE != "cuda" or TEST_WITH_ROCM,
        "Pinned async constant copy is CUDA-only",
    )
    @patch.dict(
        os.environ,
        {
            "AOTI_COPY_USE_PINNED_ASYNC": "1",
            "AOTI_COPY_STAGE_BUFFER_BYTES": "1024",
        },
    )
    def test_small_constant_pinned_async_copy(self):
        # [FUNCTION DOC] Same small-constant check as test_small_constant but with environment
        # variables enabling pinned async constant copy and a 1024 byte staging buffer. CUDA-only
        # (skipped for ROCm and non-CUDA GPUs); the environment is patched for the duration of
        # the test.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a 4x4 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.linear(x)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"always_keep_tensor_constants": True}):
            self.check_model(Model().to(self.device), example_inputs)

    def test_output_path_1(self):
        # [FUNCTION DOC] Checks numerics when the AOTInductor output_path option is set to a
        # relative prefix name.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a 10x10 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "tmp_output_"):
            self.check_model(Model(), example_inputs)

    def test_output_path_2(self):
        # [FUNCTION DOC] Compiles with the legacy compile helper while passing an explicit
        # output_path in a fresh temp dir under the inductor cache directory, and asserts the
        # returned library path equals the requested one.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a 10x10 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        model = Model().to(device=self.device)
        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        expected_path = normalize_path_separator(
            os.path.join(
                tempfile.mkdtemp(dir=cache_dir()), f"model.{get_module_ext_type()}"
            )
        )
        actual_path = AOTIRunnerUtil.legacy_compile(
            model, example_inputs, options={"aot_inductor.output_path": expected_path}
        )
        self.assertTrue(actual_path == expected_path)

    def test_empty_constant_folding(self):
        # [FUNCTION DOC] Compiles with runtime constant folding enabled and checks the generated
        # model constructor header reports 1 input, 1 output and 2 constants (no folded constants
        # appear for this model).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two random constant tensors, a 4x4 weight and a bias,
            # computing a matmul plus bias.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the weight and bias as plain tensor attributes on
                # the device.
                super().__init__()
                self.w = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matmul of x with the weight plus the bias.
                return torch.matmul(x, self.w) + self.b

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            # We should have 1 input, 1 output, 2 constants for the model.
            FileCheck().check_count("AOTInductorModelBase(1,", 1).check_next(
                "1,"
            ).check_next("2,").run(code)

    def test_constant_folding(self):
        # [FUNCTION DOC] Checks numerics with runtime constant folding on a model whose weight is
        # transposed, relu'd and biased from constants only.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias; the weight is transposed,
            # relu'd and added to the bias before a matmul.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the constant weight and bias tensors on the device.
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds the folded weight from constants and returns x
                # matmul it.
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_const_graph_no_autotune_at_compile_time(self):
        # [FUNCTION DOC] Checks numerics with runtime constant folding enabled while triton
        # autotune at compile time is disabled, so the constant graph must compile without
        # compile-time autotuning.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with 16x16 constant weight and 16 bias; folds
            # relu(transpose) plus bias then matmuls with x.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the constant weight and bias tensors on the device.
                super().__init__()
                self.w_pre = torch.randn(16, 16, device=device)
                self.b = torch.randn(16, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds the weight from constants and returns x matmul it.
                w_t = torch.transpose(self.w_pre, 0, 1)
                w = torch.nn.functional.relu(w_t) + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(16, 16, device=self.device),)
        with config.patch(
            {
                "triton.autotune_at_compile_time": False,
                "aot_inductor.use_runtime_constant_folding": True,
            }
        ):
            self.check_model(Model(self.device), example_inputs)

    def test_constant_folding_with_update(self):
        # [FUNCTION DOC] Compiles a constant-folding model under no_grad with tensor constants
        # kept, loads the legacy runner and checks that update_constant_buffer works: first
        # updating the active buffer changes outputs; then updating the inactive buffer leaves
        # outputs unchanged until swap_constant_buffer is called, after which outputs match the
        # updated eager model.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias; folds relu(transpose) plus
            # bias, then matmuls with x. Its attributes are reassigned by the test to mirror
            # runner updates.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the constant weight and bias tensors on the device.
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds the weight from constants and returns x matmul it.
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.use_runtime_constant_folding": True,
                }
            ),
        ):
            model = Model(self.device)
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: reads the call spec,
            # flattens args and kwargs with the input tree spec, keeps only tensor leaves, runs the
            # runner and unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(4, 4, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on active buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, False, False)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Update with new weights on inactive buffer
        new_weights = {
            "L__self___b": torch.randn(4, device=self.device),
            "L__self___w_pre": torch.randn(4, 4, device=self.device),
        }
        model.w_pre = new_weights["L__self___w_pre"]
        model.b = new_weights["L__self___b"]
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, True, False)
        new_output = runner_call(test_inputs)
        # We have not yet swapped the buffer, new_output should be the same as the old one.
        self.assertEqual(output, new_output)
        # Swap the buffer, should get the correct result now.
        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        self.assertEqual(expected, new_output)

    def test_update_inactive_constant_buffer_with_interleaved_folded_constants(self):
        # [FUNCTION DOC] Checks that updating the inactive constant buffer works when constants are
        # interleaved with runtime-folded constants. Compiles a model with parameters, buffers and
        # folded buffers, asserts folded constants exist, then perturbs all state, builds an update
        # map from constant names to original FQNs and compares runner outputs against eager.
        # Skipped on MPS.
        if self.device == "mps":
            raise unittest.SkipTest("MPS baseline mismatch")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two linear layers, three persistent buffers, some used
            # directly and some via a permute plus relu that gets folded.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Creates the two linear layers and registers three
                # persistent buffers.
                super().__init__()
                self.fc1 = nn.Linear(2, 2, bias=True, device=device)
                self.post = nn.Linear(2, 2, bias=False, device=device)
                self.register_buffer(
                    "uie_user_memory_network",
                    torch.randn(2, 2, device=device),
                    persistent=True,
                )
                self.register_buffer(
                    "uie_item_memory_network",
                    torch.randn(2, 2, device=device),
                    persistent=True,
                )
                self.register_buffer(
                    "late_bias",
                    torch.randn(2, device=device),
                    persistent=True,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the first linear layer, matmuls with the buffers
                # directly and with their relu of transposes, sums these, adds the late bias and
                # applies the second layer.
                x = self.fc1(x)
                direct_user = torch.matmul(x, self.uie_user_memory_network)
                direct_item = torch.matmul(x, self.uie_item_memory_network)
                folded_user = torch.relu(self.uie_user_memory_network.permute(1, 0))
                folded_item = torch.relu(self.uie_item_memory_network.permute(1, 0))
                out = direct_user + direct_item
                out = out + torch.matmul(x, folded_user) + torch.matmul(x, folded_item)
                return self.post(out + self.late_bias)

        example_inputs = (torch.randn(2, 2, device=self.device),)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.use_runtime_constant_folding": True,
                }
            ),
        ):
            model = Model(self.device)
            so_path, _ = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)
        name_to_fqn = runner.get_constant_names_to_original_fqns()
        self.assertTrue(
            [name for name in name_to_fqn if name.startswith("_FOLDED_CONST_")],
            msg="Expected runtime-folded constants in generated model",
        )

        def runner_call(x):
            # [FUNCTION DOC] HELPER: Runs the loaded runner on a single input tensor and returns
            # its first output.
            return runner.run([x])[0]

        test_inputs = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device=self.device)
        atol = 1e-3
        rtol = 1e-3
        expected = model(test_inputs)
        self.assertEqual(expected, runner_call(test_inputs), atol=atol, rtol=rtol)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
            for b in model.buffers():
                b.add_(2.0)

        state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        new_weights = {
            const_name: state[fqn].detach().clone()
            for const_name, fqn in name_to_fqn.items()
            if fqn in state
        }
        self.assertTrue(new_weights, msg="Expected non-empty constant update map")

        new_expected = model(test_inputs)

        runner.update_constant_buffer(new_weights, True, True)
        self.assertEqual(expected, runner_call(test_inputs), atol=atol, rtol=rtol)

        runner.swap_constant_buffer()
        self.assertEqual(new_expected, runner_call(test_inputs), atol=atol, rtol=rtol)

    @requires_gpu
    def test_update_constant_buffer_mixed_device(self):
        # [FUNCTION DOC] Tests update_constant_buffer on a mixed-device model (GPU parameter plus
        # CPU buffer) compiled with constants kept. Builds new weights from the runner's constants
        # map across both devices, updates the active buffer, then the inactive buffer followed by
        # a swap, checking outputs against eager. Also asserts a RuntimeError when a CPU constant
        # is updated with a GPU tensor and when a user-managed update supplies a CPU tensor for a
        # GPU constant. Needs a GPU.
        # Mixed-device model: GPU parameter + CPU buffer. Exercises
        # update_constant_buffer with a constants_map containing tensors on
        # different devices, which was previously rejected outright.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a device constant weight and a CPU constant bias; the
            # bias is moved to the input device inside forward.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a random weight on the given device and a random
                # bias on CPU.
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.cpu_bias = torch.randn(4, device="cpu")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds relu of transposed weight plus the bias moved to x
                # device, and returns x matmul that.
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.cpu_bias.to(x.device)
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with (
            torch.no_grad(),
            config.patch({"always_keep_tensor_constants": True}),
        ):
            model = Model(self.device)
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(4, 4, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # The compiler preserves the CPU constant (always_keep_tensor_constants)
        # and also promotes a GPU copy used in forward. To keep inference
        # correct, build new_weights from the runner's actual constants (which
        # span both devices) and write the new bias data into both the CPU
        # original and the GPU-promoted copy.
        cmap = runner.extract_constants_map(False)

        def build_new_weights():
            # [FUNCTION DOC] HELPER: Creates fresh random weight (on test device) and bias (on CPU)
            # tensors and a mapping from every constant name in the runner constants map to the
            # matching new tensor, moved to that constant's device. Returns (mapping, new weight,
            # new bias). Calls self.fail for any constant whose name matches neither.
            new_w_pre = torch.randn(4, 4, device=self.device)
            new_cpu_bias = torch.randn(4, device="cpu")
            nw = {}
            for name, t in cmap.items():
                if "w_pre" in name:
                    nw[name] = new_w_pre
                elif "cpu_bias" in name:
                    nw[name] = new_cpu_bias.to(t.device)
                else:
                    self.fail(f"unexpected constant {name}")
            return nw, new_w_pre, new_cpu_bias

        # Sanity: the runner actually holds constants on more than one device.
        devices_seen = {t.device.type for t in cmap.values()}
        self.assertIn("cpu", devices_seen)
        self.assertIn(self.device, devices_seen)
        cpu_constant_name = next(
            name for name, t in cmap.items() if t.device.type == "cpu"
        )
        device_constant_name = next(
            name for name, t in cmap.items() if t.device.type == self.device
        )

        # Mixed-device update on the active buffer.
        new_weights, new_w_pre, new_cpu_bias = build_new_weights()
        model.w_pre = new_w_pre
        model.cpu_bias = new_cpu_bias
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, False, False)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        # Mixed-device update on the inactive buffer, then swap.
        new_weights, new_w_pre, new_cpu_bias = build_new_weights()
        model.w_pre = new_w_pre
        model.cpu_bias = new_cpu_bias
        expected = model(test_inputs)
        runner.update_constant_buffer(new_weights, True, False)
        # Not yet swapped: output should still match the previous update.
        new_output = runner_call(test_inputs)
        self.assertEqual(output, new_output)
        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        self.assertEqual(expected, new_output)

        bad_cpu_update = {
            cpu_constant_name: torch.randn(
                cmap[cpu_constant_name].shape,
                device=self.device,
                dtype=cmap[cpu_constant_name].dtype,
            )
        }
        with self.assertRaises(RuntimeError):
            runner.update_constant_buffer(bad_cpu_update, False, False)

        bad_user_managed_update = {
            device_constant_name: torch.randn(
                cmap[device_constant_name].shape,
                device="cpu",
                dtype=cmap[device_constant_name].dtype,
            )
        }
        with self.assertRaises(RuntimeError):
            runner.update_constant_buffer(
                bad_user_managed_update,
                True,
                False,
                True,
            )

    @requires_gpu
    def test_duplicate_constant_folding(self):
        # [FUNCTION DOC] Checks numerics with runtime constant folding for a model concatenating
        # four identical-shaped constant weights and then concatenating that with the input. Needs
        # a GPU.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with four constant 4x4 weights concatenated in forward.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores four random constant weights on the device.
                super().__init__()
                self.w1 = torch.randn(4, 4, device=device)
                self.w2 = torch.randn(4, 4, device=device)
                self.w3 = torch.randn(4, 4, device=device)
                self.w4 = torch.randn(4, 4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Concatenates the weights, then concatenates x with
                # that result.
                w_concat = torch.cat((self.w1, self.w2, self.w3, self.w4))
                return torch.cat((x, w_concat))

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_autotune_with_constant_folding(self):
        # [FUNCTION DOC] Smoke test: exports a model with a large half precision constant and
        # compiles and packages it with runtime constant folding on. Only checks that compilation
        # completes; no numerics comparison.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a large half precision constant; outputs abs of the
            # constant and abs of the input.
            def __init__(self, device) -> None:
                # [FUNCTION DOC] HELPER: Stores a random 2048x2048 half precision constant on
                # the device.
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                # [FUNCTION DOC] HELPER: Returns the elementwise absolute value of input.
                return torch.abs(input)

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns a tuple of the absolute values of the constant and
                # of y.
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep, inductor_configs={"aot_inductor.use_runtime_constant_folding": True}
        )

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "Compilation error",
    )
    def test_aot_inductor_consts_cpp_build(self):
        # [FUNCTION DOC] Same model as the autotune/constant-folding smoke test, compiled and
        # packaged with runtime constant folding on and the consts assembly build disabled, so the
        # constants are built via C++. Only checks compilation succeeds. Skipped on MPS before
        # macOS 14.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a large half precision constant; outputs abs of the
            # constant and abs of the input.
            def __init__(self, device) -> None:
                # [FUNCTION DOC] HELPER: Stores a random 2048x2048 half precision constant on
                # the device.
                super().__init__()
                self.x = torch.randn(2048, 2048, dtype=torch.float16, device=device)

            def _quantize(self, input):
                # [FUNCTION DOC] HELPER: Returns the elementwise absolute value of input.
                return torch.abs(input)

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns a tuple of the absolute values of the constant and
                # of y.
                abs_weight = self._quantize(self.x)
                abs_y = self._quantize(y)

                return abs_weight, abs_y

        input1 = (torch.rand(2048, 2048, dtype=torch.float16, device=self.device),)
        model = Model(self.device).to(self.device)

        _ = model(*input1)

        ep = torch.export.export(model, input1, dynamic_shapes=None, strict=False)
        torch._inductor.aoti_compile_and_package(
            ep,
            inductor_configs={
                "aot_inductor.use_runtime_constant_folding": True,
                "aot_inductor.use_consts_asm_build": False,
            },
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_on_device_tma(self, dynamic, tma_version):
        # [FUNCTION DOC] Parametrized on dynamic shapes and TMA API version (new tensor descriptor
        # or old experimental descriptor). Runs a triton kernel that builds TMA descriptors on
        # device, with a scratch workspace in the old API, over tiled 2D inputs, after installing a
        # triton allocator. Compares through check_model, optionally with dynamic first dimensions.
        # Skipped without a GPU or without the matching triton TMA support.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_on_device_tma_new_api
            if tma_version == "new"
            else add_kernel_on_device_tma_old_api
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that zero-allocates an output and launches the selected
            # on-device TMA add kernel over a grid of 32x32 tiles, returning the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Allocates a zeros output, a workspace buffer for the old
                # API (none for the new one), computes the tile grid, launches the kernel and
                # returns the output.
                BLOCK_SIZE = 32
                out = torch.zeros_like(a)
                m, n = out.size()

                # Allocate workspace for on-device TMA descriptors
                # Need 128 bytes per descriptor, 3 descriptors total
                if tma_version == "old":
                    workspace = torch.zeros(3 * 128, dtype=torch.uint8, device=a.device)
                else:
                    workspace = None

                grid = (triton.cdiv(m, BLOCK_SIZE), triton.cdiv(n, BLOCK_SIZE))

                kernel[grid](
                    a,
                    b,
                    out,
                    m,
                    n,
                    workspace,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return out

        a = torch.randn((32 * 4, 32 * 8), device=self.device)
        b = torch.randn((32 * 4, 32 * 8), device=self.device)
        example_inputs = (a, b)

        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=GPU_TYPE
            )
        )

        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dim1 = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0, 1: None},
                "b": {0: dim1, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @requires_gpu
    def test_multi_device(self):
        # [FUNCTION DOC] Checks a model that moves a tensor GPU to CPU and back with arithmetic in
        # between. Skipped when the test device is CPU but the GPU is XPU (issue 140805).
        if self.device == "cpu" and GPU_TYPE == "xpu":
            raise unittest.SkipTest(
                "In this scenario, the test case will run XPU code in "
                "AOTIModelContainerRunnerCpu, which is not reasonable,"
                "See issue #140805"
            )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding 1, moving to CPU, adding 2, moving back to the GPU.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Performs the add, CPU move, add, GPU move sequence and
                # returns it.
                x = x + 1
                x = x.cpu()
                x = x + 2
                x = x.to(GPU_TYPE)
                return x

        example_inputs = (torch.randn(32, 64, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip(
        "install_free_tensors leads to OOM - https://github.com/pytorch/pytorch/issues/164062"
    )
    def test_large_weight(self):
        # [FUNCTION DOC] Compile-only test of a model with a very large linear weight (2048 by
        # 262144); does not run it, to avoid CI out-of-memory. Currently unconditionally skipped
        # (issue 164062).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a large linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the large linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2048, 262144)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 262144, device=self.device),
            torch.randn(1, 2048, device=self.device),
        )

        # We only test compilation since we often get OOM running in CI.
        model = Model()
        model = model.to(self.device)
        AOTIRunnerUtil.compile(model, example_inputs)

    def test_constant_type_propagation(self):
        # [FUNCTION DOC] Compiles a constant folding model and asserts the generated C++ never
        # labels a constant with the Unknown constant type.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias; relu of the transposed
            # weight plus bias, then matmul with x.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores constant weight and bias on the device.
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes the folded weight and returns x matmul it.
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        model = Model(self.device)
        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            so_path, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            FileCheck().check_not("torch::aot_inductor::ConstantType::Unknown").run(
                code
            )

    @requires_gpu
    def test_device_moved_constant(self):
        # [FUNCTION DOC] For both directions (GPU buffers moved to CPU and CPU buffers moved to
        # GPU), compiles a model whose buffer and parameter are moved to the input device, with
        # runtime constant folding off. Asserts the generated code has no Unknown constant type,
        # and exactly two Buffer and two Parameter constant type mentions. Needs a GPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Mixed-device test requires GPU")

        # testing both directions
        device_movements = [
            (torch.device(type=GPU_TYPE, index=0), torch.device("cpu")),
            (torch.device("cpu"), torch.device(type=GPU_TYPE, index=0)),
        ]

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a registered buffer and a non-trainable parameter
            # created on the source device; forward returns both moved to the input device.
            def __init__(self, from_device):
                # [FUNCTION DOC] HELPER: Registers a random buffer and a non-trainable random
                # parameter on from_device.
                super().__init__()
                self.register_buffer("_buf", torch.randn(6, 7, device=from_device))
                self._param = torch.nn.Parameter(
                    torch.rand(6, 7, device=from_device), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Moves the buffer and the parameter to the device of x and
                # returns both.
                to_device = x.device
                moved_buf = self._buf.to(to_device)
                moved_param = self._param.to(to_device)
                return moved_buf, moved_param

        with config.patch(
            {
                "aot_inductor.use_runtime_constant_folding": False,
            }
        ):
            for from_device, to_device in device_movements:
                model = Model(from_device)
                example_inputs = (torch.randn(6, 7, device=to_device),)
                _, code = run_and_get_cpp_code(
                    AOTIRunnerUtil.compile, model, example_inputs
                )
                FileCheck().check_not("torch::aot_inductor::ConstantType::Unknown").run(
                    code
                )
                FileCheck().check_count(
                    "torch::aot_inductor::ConstantType::Buffer", 2, exactly=True
                ).run(code)
                FileCheck().check_count(
                    "torch::aot_inductor::ConstantType::Parameter", 2, exactly=True
                ).run(code)

    def test_subclasses(self):
        # [FUNCTION DOC] Runs a module with a plain parameter and a parameter that is a custom
        # tensor subclass through AOTIRunnerUtil.run under no_grad, and asserts the result
        # matches eager.
        device_to_init = self.device

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones parameter and a custom tensor subclass
            # parameter; forward adds the sum of twice the first plus the second to x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a plain ones parameter and a parameter wrapping a
                # custom plain output tensor subclass, all on the test device.
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4, device=device_to_init))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        torch.ones(3, 4, device=device_to_init),
                        torch.ones(3, 4, device=device_to_init),
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums twice p1 plus p2 to a scalar and adds it to x.
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(3, 4, device=device_to_init)

        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                m,
                (ref_x,),
            )
        actual = m(ref_x)
        self.assertTrue(same(result, actual))

    def test_large_mmaped_weights(self):
        # [FUNCTION DOC] Checks numerics of a model with a large linear weight when the weights are
        # forced to be memory mapped.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a 512 to 250112 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the large linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch({"aot_inductor.force_mmap_weights": True}):
            self.check_model(Model(), example_inputs)

    def test_large_mmaped_weights_on_disk(self):
        # [FUNCTION DOC] Checks numerics of a model with a large linear weight when constants are
        # packaged on disk in the binary blob format.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a 512 to 250112 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the large linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(512, 250112)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear layer applied to y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(1, 250112, device=self.device),
            torch.randn(1, 512, device=self.device),
        )
        with config.patch(
            {"aot_inductor.package_constants_on_disk_format": "binary_blob"}
        ):
            self.check_model(Model(), example_inputs)

    def test_with_offset(self):
        # [FUNCTION DOC] Checks a model whose constants are views of one tensor with nonzero
        # storage offsets (a row of a batch, and a slice of that), used in a linear and an add.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a base constant and a sliced view of it.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Takes the first batch entry of a random tensor as the base
                # constant and a view of its rows from index five on.
                super().__init__()
                self.orig_tensor = torch.randn(2, 15, 10, device=device)[0]
                self.tensor = self.orig_tensor[5:, :]

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus a linear of y with the first ten rows of
                # the base, plus the sliced view.
                return (
                    x
                    + torch.nn.functional.linear(y, self.orig_tensor[:10, :])
                    + self.tensor
                )

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_freezing(self):
        # [FUNCTION DOC] Checks numerics with freezing enabled on a model whose weight is
        # concatenated with a padding row at runtime. Skipped in fbcode.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a weight constant and a padding row constant that are
            # concatenated and used as a linear weight.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a 9x10 weight and a 1x10 padding constant on
                # the device.
                super().__init__()
                self.weight = torch.randn(9, 10, device=device)
                self.padding = torch.randn(1, 10, device=device)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Concatenates weight and padding and returns x plus y
                # linear with it.
                padded_weight = torch.cat((self.weight, self.padding), dim=0)
                return x + torch.nn.functional.linear(y, padded_weight)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        with config.patch({"freezing": True}):
            self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_conv_freezing(self):
        # [FUNCTION DOC] With freezing enabled, loops over dtypes (bfloat16 too on SM80+) and group
        # counts one and two, building a conv model with a constant weight and checking numerics.
        # Skipped in fbcode.
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype, groups in itertools.product(dtypes, [1, 2]):
            iC = 2
            oC = 3

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Conv model with a constant weight cast to the loop dtype,
                # using the loop's group count and channel sizes through closure.
                def __init__(self, device):
                    # [FUNCTION DOC] HELPER: Stores a random conv weight on the device cast to the
                    # loop dtype.
                    super().__init__()
                    self.weight = torch.randn(oC * groups, iC, 3, 3, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    # [FUNCTION DOC] HELPER: Applies a grouped 2D convolution of y with the
                    # stored weight.
                    return torch.nn.functional.conv2d(y, self.weight, groups=groups)

            example_inputs = (
                torch.randn(2, iC * groups, 10, 10, device=self.device).to(dtype),
            )

            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    @tf32_on_and_off(0.005)
    def test_deconv_freezing(self):
        # [FUNCTION DOC] With freezing enabled, loops over dtypes (bfloat16 only if mkldnn bf16 is
        # supported) and groups two and one, checking numerics of a transposed convolution with a
        # constant weight. Runs with TF32 on and off with tolerance 0.005. Skipped in fbcode.
        dtypes = [torch.float]
        if torch._C._has_mkldnn and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype, groups in itertools.product(dtypes, [2, 1]):
            iC = 4
            oC = 2

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Transposed conv model with a constant weight cast to the loop
                # dtype, using the loop's group count through closure.
                def __init__(self, device):
                    # [FUNCTION DOC] HELPER: Stores a random transposed conv weight on the device
                    # cast to the dtype.
                    super().__init__()
                    self.weight = torch.randn(iC, oC * groups, 2, 2, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    # [FUNCTION DOC] HELPER: Applies a grouped 2D transposed convolution of y with
                    # the weight.
                    return torch.nn.functional.conv_transpose2d(
                        y, self.weight, groups=groups
                    )

            example_inputs = (torch.randn(1, iC, 3, 3, device=self.device).to(dtype),)
            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @unittest.skipIf(
        IS_FBCODE,
        "Not yet runnable in fbcode when the model.so is newly generated while older PyTorch is used",
    )
    def test_linear_freezing(self):
        # [FUNCTION DOC] With freezing enabled, loops over dtypes (bfloat16 too on SM80+) and
        # checks numerics of a linear with constant weight and bias. Skipped in fbcode.
        dtypes = [torch.bfloat16, torch.float] if SM80OrLater else [torch.float]
        for dtype in dtypes:

            class LinearModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Model applying a linear with constant 10x10 weight and bias
                # cast to the loop dtype.
                def __init__(self, device):
                    # [FUNCTION DOC] HELPER: Stores the weight and bias constants cast to the
                    # loop dtype.
                    super().__init__()
                    self.weight = torch.randn(10, 10, device=device).to(dtype)
                    self.bias = torch.randn(10, device=device).to(dtype)

                def forward(self, y):
                    # [FUNCTION DOC] HELPER: Returns the linear of y with the stored weight
                    # and bias.
                    return torch.nn.functional.linear(y, self.weight, self.bias)

            example_inputs = (torch.randn(10, 10, device=self.device).to(dtype),)

            with config.patch({"freezing": True}):
                model = LinearModel(device=self.device)
                self.check_model(model, example_inputs)

    def test_same_backing(self):
        # [FUNCTION DOC] Defines a scoped custom operator taking two tensors and returning their
        # sum, and checks a model that applies it to a concatenation of each input with itself (so
        # both operands alias the same buffer) and scales by the dimension size.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo2",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo2", "CompositeExplicitAutograd", lib=lib)
            def foo_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Implementation of the custom op: returns the elementwise
                # sum a plus b.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Model applying the custom op to duplicated concatenations of
                # both inputs, scaled by the original lengths of the first dimension.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: For each input, concatenates it with itself, adds it
                    # to itself via the custom op, and multiplies by its original length; returns
                    # both results.
                    x = a.shape[0]
                    y = b.shape[0]
                    a = torch.cat([a, a])
                    a = torch.ops.mylib.foo2(a, a)
                    a = a * x
                    b = torch.cat([b, b])
                    b = torch.ops.mylib.foo2(b, b)
                    b = b * y
                    return a, b

            inp = (torch.ones(3, device=self.device), torch.ones(3, device=self.device))
            self.check_model(M(), inp)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "MPS BFloat16 is only supported on MacOS 14+",
    )
    def test_empty_cat_dtype_promotion(self):
        # [FUNCTION DOC] Checks concatenating a bfloat16 tensor with an empty float tensor then
        # casting to bfloat16 and scaling. Skipped on MPS before macOS 14.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that concatenates two inputs along dim 1, casts to
            # bfloat16, doubles.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Concatenates x and y, casts to bfloat16 and multiplies
                # by two.
                z = torch.cat([x, y], dim=1)
                z = z.to(dtype=torch.bfloat16)
                return z * 2

        model = Foo()
        inps = (torch.randn(4, 10, dtype=torch.bfloat16), torch.randn(4, 0))
        self.check_model(model, inps)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_linear_dynamic_maxautotune(self):
        # [FUNCTION DOC] Exports a 1x1 linear model with a dynamic batch dimension, compiles and
        # packages it with max-autotune restricted to the Triton gemm backend, loads it and
        # compares against eager for several batch sizes. Skipped on CPU and on non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a 1 to 1 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.linear(x)

        model = Model().to(device=self.device)
        compile_inputs = (torch.randn(2048, 1, device=self.device),)
        dim0_x = Dim("dim0_x", min=2, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}}
        ep = torch.export.export(
            model, compile_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )
        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                ep,
                inductor_configs={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                },
            )
        )
        runtime_input = torch.randn(10, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(16, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))
        runtime_input = torch.randn(100, 1, device=self.device)
        self.assertTrue(same(optimized(runtime_input), model(runtime_input)))

    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_stack_tahn_unbind_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_simple_split(self):
        # [FUNCTION DOC] With split/cat fusion passes enabled via config patch, runs a model that
        # splits the input and concatenates the parts along another dim through the legacy runner.
        # Asserts numerics and that the inductor counters record one removed split, one removed cat
        # and one removed split-sections.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that splits x into chunks of 4 along dim 1 and concatenates
            # them along dim -2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Splits then concatenates the pieces as described in
                # the class.
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        example_inputs = (torch.randn(2, 8, device=self.device),)
        counters.clear()
        model = Model().to(device=self.device)
        actual = AOTIRunnerUtil.legacy_run(self.device, model, example_inputs)
        self.assertTrue(same(model(*example_inputs), actual))
        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 1)
        self.assertEqual(counters["inductor"]["scmerge_split_sections_removed"], 1)

    def test_amp_fallback_random(self):
        # [FUNCTION DOC] Checks a plain function applying a linear under autocast with
        # fallback_random enabled.
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Returns the linear of x with weight w.
            return torch.functional.F.linear(x, w)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"fallback_random": True}):
            with torch.amp.autocast(device_type=self.device):
                self.check_model(fn, example_inputs)

    def test_missing_output(self):
        # [FUNCTION DOC] Checks a model whose intermediate results are computed and consumed only
        # internally (sin, matmul, cos).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing cos of the matmul of sin of x with y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns cos of the matmul of sin of x and y.
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.cos(b)
                return c

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_output_misaligned(self):
        # [FUNCTION DOC] Checks a model returning a sigmoid result and a slice of a stacked tensor,
        # with runtime constant folding on, where an output may be a view with an unaligned offset.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that stacks two inputs, then returns sigmoid of the first
            # slice and the second slice as-is.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Unsqueezes and concatenates the inputs, then returns
                # sigmoid of the first slice and the second slice.
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                x_getitem = cat[0]
                y_getitem = cat[1]
                x_sigmoid = torch.sigmoid(x_getitem)
                return x_sigmoid, y_getitem

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @skip("Test was marked as expected failure, but does not fail always anymore.")
    def test_dynamic_smem_above_default_limit(self):
        # [FUNCTION DOC] Checks a matmul that needs more dynamic shared memory than the default
        # limit, using max-autotune with the Triton gemm backend. Always skipped (marked as no
        # longer reliably failing). Skipped on CPU and non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model doing a matrix multiplication of x and y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and y.
                return x @ y

        model = Model().to(self.device)
        # on A100, the generated Triton kernel for this MM
        # requires 55296 bytes of dynamic SMEM which is above
        # the A100's default dynamic SMEM limit of 49152 bytes.
        example_inputs = (
            torch.randn(10285, 96, device=self.device),
            torch.randn(96, 1, device=self.device),
        )
        self.check_model(
            model,
            example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    def test_seq(self):
        # [FUNCTION DOC] Checks a sequential network alternating a shared layer norm and ReLU in
        # eval mode on a 1D input. Skipped in fbcode.
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(net.eval(), example_inputs)

    def test_addmm(self):
        # [FUNCTION DOC] Checks a linear (addmm) with constant weight and bias on a batched input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random weight of shape n by k and bias of size n on
                # the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a with the stored weight and bias.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        # We should be able to call self.check_model here, but torch.export.export
        # constants (non-parameter, non-buffer) doesn't work today.
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    def test_aliased_buffer_reuse(self):
        # [FUNCTION DOC] Checks a model where intermediate buffers alias slices of other buffers,
        # to test buffer reuse correctness.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model concatenating scaled inputs, adding one, squaring by matmul
            # and adding a slice of the result to the scaled first input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes as described in the class.
                x = 2 * x
                y = 2 * y
                c = torch.cat([x, y], dim=-1)
                d = 1 + c
                m = torch.mm(d, d)
                return m[:, :2] + x

        example_inputs = (
            torch.randn(4, 2, device=self.device),
            torch.randn(4, 2, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_buffer_reuse(self):
        # [FUNCTION DOC] Checks a chain of pointwise ops and matmuls whose temporaries can
        # reuse buffers.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with sin, cos, matmul, relu, sigmoid and a second matmul
            # combined by addition.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes as described in the class.
                a = torch.sin(x)
                b = torch.cos(y)
                c = torch.mm(a, b)
                d = torch.relu(c)
                e = torch.sigmoid(d)
                f = torch.mm(x, y)
                g = e + f
                return g

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_duplicated_params(self):
        # [FUNCTION DOC] Checks a model where two attributes refer to the same parameter object.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that stores one parameter under two attribute names.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random parameter and aliases it as a
                # second attribute.
                super().__init__()
                self.p = torch.nn.Parameter(torch.rand(6))
                self.q = self.p

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter times x plus the aliased attribute.
                return self.p * x + self.q

        example_inputs = (torch.rand(6, device=self.device),)
        self.check_model(Model(), example_inputs)

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_inf(self):
        # [FUNCTION DOC] Local-only check with an infinite value in the input and the inf/nan debug
        # check option on. Always skipped because it aborts with SIGABRT.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 10x10 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y.
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("Inf")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @unittest.skip("Skip this test, only for local test. SIGABRT is produced.")
    def test_nan(self):
        # [FUNCTION DOC] Local-only check with a NaN value in the input and the inf/nan debug check
        # option on. Always skipped because it aborts with SIGABRT.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 10x10 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y.
                return x + self.linear(y)

        x = torch.randn(10, 10, device=self.device)
        x[0][0] = float("nan")
        example_inputs = (
            x,
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(
            Model().to(self.device),
            example_inputs,
            options={"debug_check_inf_and_nan": True},
        )

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_assert_async(self):
        # [FUNCTION DOC] Checks a model that extracts a scalar from a tensor, asserts it exceeds 3
        # and uses it as a size, exercising the runtime assert-async path. Skipped unless the
        # device is the GPU and on Windows XPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU_TYPE")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that reads a scalar from x, runtime-checks it is greater
            # than 3 and returns the first element of a ones tensor of that length.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Extracts the scalar, adds the check and returns the first
                # element of a ones tensor of that size.
                u0 = x.item()
                torch._check(u0 > 3)
                return torch.ones(u0)[0]

        x = torch.tensor(23, device=self.device)
        example_inputs = (x,)
        self.check_model(Model(), example_inputs)

    def test_simple_dynamic(self):
        # [FUNCTION DOC] Checks relu of a sum with a dynamic first dimension shared by both inputs.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning relu of x plus y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns relu of x plus y.
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_large_dynamic_dim(self):
        # [FUNCTION DOC] Same as the simple dynamic test but with the dynamic dimension upper bound
        # exceeding the 64-bit integer range. Skipped on Windows (crash).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning relu of x plus y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns relu of x plus y.
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        # Use a dimension that exceeds the maximum value of a C long long (2^63 - 1)
        dim0_x = Dim("dim0_x", min=1, max=1171368248680556527362)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        example_inputs = (x, y)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_fp8(self):
        # [FUNCTION DOC] Checks a scaled matmul in float8 with a transposed half precision weight
        # cast to e4m3, bias and unit inverse scales, with a dynamic first dimension of x. GPU
        # only; skipped without FP8 support.
        # cuda only
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model doing a scaled matrix multiply of x with an
            # e4m3-cast weight.
            def __init__(self, dtype):
                # [FUNCTION DOC] HELPER: Stores the requested output dtype.
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                # [FUNCTION DOC] HELPER: Casts weight to e4m3 and runs the scaled matmul with the
                # given scales and the stored output dtype; returns the result.
                # KNOWN ISSUE: the bias argument is ignored; the call uses the enclosing test's
                # input_bias via closure instead (same tensor is passed as argument in this test,
                # so behavior is unaffected).
                weight = weight.to(e4m3_type)
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        b_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        input_bias = torch.rand(32, device=GPU_TYPE, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=GPU_TYPE, dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None, None)
        self.check_model(
            Model(dtype),
            (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8_GROUPED_GEMM,
        "scaled_grouped_mm is only supported on SM90 and MI300+ devices",
    )
    def test_scaled_grouped_mm(self):
        # [FUNCTION DOC] Checks the scaled grouped matmul op on e4m3 inputs: three groups of 64 by
        # 128 inputs against a concatenated transposed weight, with unit scales and cumulative
        # int32 offsets of the per-group output sizes. Runs through check_model. CUDA only.
        # Test torch._scaled_grouped_mm AOTI lowering
        # cuda only
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model calling the scaled grouped matmul with the transposed
            # weight, fast accumulation on, and returning the result cast to half.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, weight, scale_a, scale_b, offsets):
                # [FUNCTION DOC] HELPER: Runs the scaled grouped matmul of x and the
                # transposed weight with the given scales and offsets, and returns the output
                # as half precision.
                # x: [num_groups, batch, in_features] - FP8 inputs
                # weight: [total_out_features, in_features] - FP8 weights (transposed)
                # scale_a: [num_groups] - input scales
                # scale_b: [num_groups] - weight scales
                # offsets: [num_groups] - cumulative output sizes
                output = torch._scaled_grouped_mm(
                    x,
                    weight.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    offs=offsets,
                    use_fast_accum=True,
                )
                return output.half()

        dtype = torch.float16
        num_groups = 3
        batch_size = 64
        in_features = 128
        out_features_list = [64, 128, 256]  # Different output sizes for each group

        device = GPU_TYPE

        # Calculate offsets (cumulative output sizes)
        offsets = torch.cumsum(torch.tensor(out_features_list), dim=0).to(
            device, dtype=torch.int32
        )
        total_out_features = sum(out_features_list)

        # Create FP8 input tensors - stacked for all groups
        x_fp16 = torch.randn(
            num_groups, batch_size, in_features, dtype=dtype, device=device
        )
        x_fp8 = x_fp16.to(e4m3_type)

        # Create FP8 weight tensor - concatenated and transposed
        weight_fp16 = torch.randn(
            total_out_features, in_features, dtype=dtype, device=device
        )
        weight_fp8 = weight_fp16.to(e4m3_type)

        # Create scales
        scale_a = torch.ones(num_groups, batch_size, device=device, dtype=torch.float32)
        scale_b = torch.ones(total_out_features, device=device, dtype=torch.float32)

        self.check_model(
            Model(),
            (x_fp8, weight_fp8, scale_a, scale_b, offsets),
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_fp8_view_of_param(self):
        # [FUNCTION DOC] Checks a scaled matmul where the weight is a constant that is transposed
        # inside the graph, so AOTI must materialize the view before calling the extern kernel.
        # Dynamic first dim for x. GPU only; requires FP8 platform support.
        # cuda only
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a constant e4m3 weight and an output dtype; forward
            # transposes the weight and calls the scaled matmul.
            def __init__(self, dtype, weight):
                # [FUNCTION DOC] HELPER: Stores the output dtype and the constant weight.
                super().__init__()
                self.out_dtype = dtype
                self.weight = weight

            def forward(self, x, bias, scale_a, scale_b):
                # [FUNCTION DOC] HELPER: Transposes the stored weight and runs the scaled matmul
                # with the given scales and out dtype.
                # KNOWN ISSUE: the bias argument is ignored; the enclosing test's input_bias is
                # used via closure (the test passes the same tensor, so results are unaffected).
                # test: do the view inside of the graph,
                # AOTI needs to materialize this view before passing
                # it into the scaled_mm extern kernel
                weight = self.weight.T
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=self.device)
        b_scale = torch.Tensor([1.0]).to(device=self.device)
        input_bias = torch.rand(32, device=self.device, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=self.device, dtype=dtype).to(
            e4m3_type
        )
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=self.device, dtype=dtype).to(e4m3_type)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None, None)
        self.check_model(
            Model(dtype, weight),
            (x, input_bias, a_inverse_scale, b_inverse_scale),
            dynamic_shapes=dynamic_shapes,
        )

    def test_poi_multiple_dynamic(self):
        # [FUNCTION DOC] Checks relu of a sum with a shared dynamic first dim, running check_model
        # with three input sets of batch sizes 128, 64 and 211.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning relu of x plus y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns relu of x plus y.
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            Model(), list_example_inputs, dynamic_shapes=dynamic_shapes
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_addmm_multiple_dynamic(self):
        # [FUNCTION DOC] Checks a linear on a 3D input with a dynamic batch dimension, compiled
        # with max-autotune and the Triton gemm backend, over batch sizes 2, 2048 and 128. Skipped
        # on CPU and non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores a random weight of shape n by k and bias of size n
                # on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a with the stored weight and bias.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}}
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_bmm_multiple_dynamic(self):
        # [FUNCTION DOC] Checks batched matmul with dynamic shared batch dimension under
        # max-autotune with the Triton backend, over batch sizes 1024, 2048 and 128. Skipped on CPU
        # and non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model doing a batched matrix multiplication of a and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matrix product of a and b.
                return torch.bmm(a, b)

        M = 8
        N = 6
        K = 16
        model = Model()
        batch = 1024
        a = torch.randn(batch, M, K, device=self.device)
        b = torch.randn(batch, K, N, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_a}}
        list_example_inputs = [(a, b)]
        batch = 2048
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        batch = 128
        list_example_inputs.append(
            (
                torch.randn(batch, M, K, device=self.device),
                torch.randn(batch, K, N, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/179958")
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_bmm_large_batch_dynamic(self):
        # [FUNCTION DOC] Compiles a half precision batched matmul with dynamic batch (bound up to 2
        # to the 17th) at batch 100 under max-autotune with Triton, then also runs batch 70000,
        # beyond the CUDA grid limit of 65535 on one axis. Inputs are scaled to reduce half
        # precision noise. Skipped on ROCm, CPU and non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model doing a batched matrix multiplication of a and b.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matrix product of a and b.
                return torch.bmm(a, b)

        M, K, N = 64, 64, 64
        dtype = torch.float16
        model = Model()

        # Scale inputs so this grid-size test is not sensitive to FP16 BMM noise.
        def make_inputs(batch):
            # [FUNCTION DOC] HELPER: Builds a pair of half precision batched random matrices (64x64
            # each) with the given batch size, using the scaled reduction dimension helper and
            # scaled by 0.5.
            return (
                0.5
                * random_matrix_with_scaled_reduction_dim(
                    M, K, batch, device=self.device, dtype=dtype, reduction_dim=-1
                ),
                0.5
                * random_matrix_with_scaled_reduction_dim(
                    K, N, batch, device=self.device, dtype=dtype, reduction_dim=-2
                ),
            )

        # Compile with small batch, then run with batch > 65535 (CUDA grid.y limit)
        compile_batch = 100
        a, b = make_inputs(compile_batch)
        dim0_a = Dim("dim0_a", min=1, max=2**17)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_a}}
        list_example_inputs = [(a, b)]

        # Large batch exceeding CUDA grid.y limit of 65535
        large_batch = 70000
        list_example_inputs.append(make_inputs(large_batch))
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_foreach_multiple_dynamic(self):
        # [FUNCTION DOC] Checks a model stacking two inputs through unsqueeze and concat, with a
        # dynamic shared first dim, over three input sets (128, 64, 211 rows). Skipped on Windows.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that unsqueezes both inputs and concatenates them along a
            # new leading dim.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the stack of x and y built by unsqueeze
                # and concat.
                x_unsqueeze = torch.unsqueeze(x, dim=0)
                y_unsqueeze = torch.unsqueeze(y, dim=0)
                cat = torch.cat([x_unsqueeze, y_unsqueeze], dim=0)
                return cat

        model = Model()
        x = torch.randn(128, 2048, device=self.device)
        y = torch.randn(128, 2048, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}
        list_example_inputs = [(x, y)]
        list_example_inputs.append(
            (
                torch.randn(64, 2048, device=self.device),
                torch.randn(64, 2048, device=self.device),
            ),
        )
        list_example_inputs.append(
            (
                torch.randn(211, 2048, device=self.device),
                torch.randn(211, 2048, device=self.device),
            ),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # scaled_dot_product_flash_attention
    @unittest.skipIf(
        not SM80OrLater and not HAS_XPU_AND_TRITON,
        "bfloat16 only supported in sm80+ or XPU",
    )
    def test_sdpa(self):
        # [FUNCTION DOC] Checks scaled dot product attention on bfloat16 1x48x64x64 inputs,
        # returning the first output. Skipped without SM80 or XPU with Triton.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the first output of scaled dot product attention
            # of q, k, v.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Returns the attention output (first element) for q, k, v.
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        not SM80OrLater and not HAS_XPU_AND_TRITON,
        "bfloat16 only supported in sm80+ or XPU",
    )
    @unittest.skipIf(
        # for archs where this isn't lowered to flash attention, the math
        # backend will be used and it doesn't work for bfloat16
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Some archs don't support SDPA with bfloat16",
    )
    def test_sdpa_2(self):
        # [FUNCTION DOC] Checks causal scaled dot product attention on bfloat16 inputs with the
        # result added to a fourth input. Skipped without SM80 or XPU, and on platforms without
        # flash attention (where the math backend lacks bfloat16 support).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding x to the causal attention output of q, k, v.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, q, k, v, x):
                # [FUNCTION DOC] HELPER: Computes causal attention of q, k, v, takes the first
                # output and adds x.
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                return x + t

        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear(self):
        # [FUNCTION DOC] Checks the dynamic fp16 quantized linear op with unpacked weight and bias,
        # with runtime constant folding. Skipped without FBGEMM.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with constant weight and bias applying the dynamic fp16
            # quantized linear op.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the fp16 dynamic quantized linear op to x.
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, self.bias
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quantized_linear_bias_none(self):
        # [FUNCTION DOC] Same as the quantized linear test but passing no bias (None) to the op.
        # Skipped without FBGEMM.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a constant weight applying the dynamic fp16 quantized
            # linear op with no bias.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a random constant weight on the device.
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the op to x with no bias.
                return torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(
                    x, self.weight, None
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @skipIfNoFBGEMM
    def test_quanatized_int8_linear(self):
        # [FUNCTION DOC] Checks the wrapped quantized linear op with scale and zero point tensors,
        # with runtime constant folding. Skipped without FBGEMM.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding constant weight, bias, input/weight/output scales
            # and zero points (CPU scalar tensors) and the output channel count.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the constants listed in the class doc.
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)
                self.input_scale = torch.tensor(0.1)
                self.input_zero_point = torch.tensor(0)
                self.weight_scale = torch.tensor(0.1)
                self.weight_zero_point = torch.tensor(0)
                self.output_scale = torch.tensor(0.1)
                self.output_zero_point = torch.tensor(0)
                self.out_channel = 10

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the wrapped quantized linear op with the stored
                # quantization parameters, weight, bias and out channels.
                return torch.ops._quantized.wrapped_quantized_linear(
                    x,
                    self.input_scale,
                    self.input_zero_point,
                    self.weight,
                    self.weight_scale,
                    self.weight_zero_point,
                    self.bias,
                    self.output_scale,
                    self.output_zero_point,
                    self.out_channel,
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    def test_zero_grid_with_unbacked_symbols(self):
        # [FUNCTION DOC] Checks a model whose nonzero result gives an unbacked size feeding
        # elementwise ops of possibly zero grid size, a matmul and a sum.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that takes the nonzero indices of x, builds ones and
            # zeros half precision tensors of that shape, sums them, matmuls with y and returns
            # the total sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes the half precision sum of ones and zeros shaped
                # by the nonzero result, matmuls with y and returns the scalar sum.
                nz = torch.nonzero(x)
                b = torch.ones_like(nz, dtype=torch.float16)
                c = torch.zeros_like(nz, dtype=torch.float16)
                d = (b + c) @ y
                return d.sum()

        example_inputs = (
            torch.tensor([1, 1, 1], device=self.device),
            torch.randn((1, 32), dtype=torch.float16, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @config.patch({"unbacked_symint_fallback": 12})
    @parametrize("shift_k", [0, 1, 2, 3])
    @parametrize("use_static_size", [True, False])
    def test_unbacked_expr_replacements(self, shift_k, use_static_size):
        # [FUNCTION DOC] Parametrized on shift_k (rotation of the tensors before concatenation)
        # and use_static_size (whether the fourth size is static or unbacked). Builds tensors
        # with sizes that are expressions of backed and unbacked symbols realized by a triton
        # kernel, concatenates them and squares, with the CUDA caching allocator disabled to
        # catch illegal memory access. Skipped unless a non-MPS GPU. The unbacked fallback value
        # is patched to 12.
        """
        Test parameters
        - shift_k: Validates that torch._check assertion order doesn't affect
        results by shifting the order of torch._checks
        - use_static_size: Tests torch._check compatibility between unbacked
        symbolic expressions and static shapes
        """

        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("Need triton for user-defined triton kernel")

        def realize_out_tensor_with_size(size):
            # [FUNCTION DOC] HELPER: Creates a ones tensor of the given number of rows and 256
            # columns, runs the add kernel in place to force it to be realized as a buffer, and
            # returns it.
            STATIC_DIM = 256  # large enough to hit IMA w/o compute-sanitizer
            tensor = torch.ones((size, STATIC_DIM), device=self.device)
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )
            return tensor

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing four sizes from shapes of x and y and from
            # unbacked values of the list input, realizing tensors of those sizes, rotating them
            # and concatenating.
            def forward(self, x, y, lst):
                # [FUNCTION DOC] HELPER: Reads the shapes of x and y and five integers from lst,
                # builds the four sizes (the last is static or unbacked per the outer flag),
                # realizes the tensors, rotates them by the outer shift, concatenates along dim 1
                # and returns the square of the result.
                STATIC_SIZE = 300
                s0, s1 = x.shape
                s2, s3 = y.shape
                u0, u1, u2, u3, u100 = lst.tolist()

                expr1 = s0 + u0
                expr2 = s1 + u1
                expr3 = (s2 * s3) + (u2 // u3)  # make this one a lil complicated
                expr4 = STATIC_SIZE if use_static_size else u100

                t1 = realize_out_tensor_with_size(expr1)
                t2 = realize_out_tensor_with_size(expr2)
                t3 = realize_out_tensor_with_size(expr3)
                t4 = realize_out_tensor_with_size(expr4)

                # shift tensors to change up the torch._check order
                tensors = [t1, t2, t3, t4]
                shifted_tensors = tensors[shift_k:] + tensors[:shift_k]

                # torch.cat implicitly runs torch._check(lhs == rhs)
                cat = torch.cat(shifted_tensors, dim=1)

                return cat * cat

        # Disable cuda caching allocator to check for IMA
        with caching_allocator_disabled():
            model = Repro()
            example_inputs = (
                # s0, s1
                torch.randn((100, 200), device=self.device),
                # s2, s3
                torch.randn((100, 3), device=self.device),
                # u0, u1, u2, u3, u100
                torch.tensor(
                    [200, 100, 0, 1, 300], device=self.device, dtype=torch.int
                ),
            )
            spec = {
                "x": (Dim.DYNAMIC, Dim.DYNAMIC),
                "y": (Dim.DYNAMIC, Dim.DYNAMIC),
                "lst": (Dim.STATIC,),
            }
            self.check_model(model, example_inputs, dynamic_shapes=spec)

    @config.patch({"unbacked_symint_fallback": 12})
    @config.patch({"triton.autotune_at_compile_time": None})
    def test_replace_unbacked_symbol_with_backed_expr(self):
        # [FUNCTION DOC] Checks that sample inputs generated for compile-time autotuning are sized
        # correctly when a runtime check ties a backed size sum to an unbacked size; wrong sizes
        # would cause illegal memory access. Disables the caching allocator. Skipped unless a
        # non-MPS GPU.
        # This will test how autotune_at_compile_time generates sample inputs
        # when the user torch._checks(s0 + s1 == u0).
        # We may fail with IMA if the generated input sizes aren't correct.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires triton")

        def force_realize(tensor):
            # [FUNCTION DOC] HELPER: Runs the add kernel in place over the given 2D tensor so it is
            # realized as a buffer.
            # Realize the tensor as an intermediate buffer
            nrows, ncols = tensor.shape
            numel = tensor.numel()
            add_kernel[nrows,](
                in_ptr0=tensor,
                in_ptr1=tensor,
                out_ptr=tensor,
                n_elements=numel,
                BLOCK_SIZE=ncols,
            )

        INNER_DIM = 256

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that concatenates x and y, realizes the result, realizes a
            # ones tensor whose rows come from the nonzero count of lengths, asserts the row counts
            # match, adds and squares.
            def forward(self, x, y, lengths):
                # [FUNCTION DOC] HELPER: Performs the sequence described in the class and returns
                # the squared sum.
                # Realize an intermediate buffer with backed shape: s0 + s1
                relevant_embeddings = torch.cat([x, y], dim=0)
                force_realize(relevant_embeddings)

                # Realize an intermediate buffer with unbacked shape: u0
                num_relevant_embeddings = lengths.nonzero().size(0)
                ones = torch.ones((num_relevant_embeddings, INNER_DIM), device=x.device)
                force_realize(ones)

                # Add deferred runtime assertion: s0 + s1 == u0
                torch._check(relevant_embeddings.size(0) == ones.size(0))
                relevant_embeddings += ones
                return relevant_embeddings * relevant_embeddings

        with caching_allocator_disabled():
            model = Repro()
            example_inputs = (
                torch.randn((1000, INNER_DIM), device=self.device),
                torch.randn((2000, INNER_DIM), device=self.device),
                torch.ones(3000),
            )
            spec = {
                "x": (Dim.DYNAMIC, Dim.STATIC),
                "y": (Dim.DYNAMIC, Dim.STATIC),
                "lengths": (Dim.DYNAMIC,),
            }
            self.check_model(model, example_inputs, dynamic_shapes=spec)

    @config.patch({"triton.autotune_at_compile_time": None})
    @torch.fx.experimental._config.patch("backed_size_oblivious", True)
    def test_slice_independent_backed_symints_no_unbacked(self):
        # [FUNCTION DOC] Checks that a slice with an end bound from another tensor's size, with
        # backed size oblivious mode on, produces a bounded backed size rather than an unbacked
        # symbol with a bad fallback. Skipped unless a non-MPS GPU.
        # x[0:s1] where x.size(0) = s0-1 should produce Min(s1, s0-1),
        # not an unbacked symint with a bad fallback value.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires triton")

        INNER_DIM = 4224

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that drops the last row of x, slices the first rows by the
            # size of y, reshapes, expands and does a shift and mask by 8-bit lanes.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Performs the slicing, reshape, expand and per-lane shift
                # and mask.
                x_trimmed = x[:-1]
                sliced = x_trimmed[: y.size(0)]
                reshaped = sliced.reshape(-1, 128, 33)
                expanded = reshaped.unsqueeze(3).expand(-1, 128, 33, 8)
                shifts = torch.arange(0, 64, 8, device=x.device, dtype=torch.int64)
                return (expanded >> shifts) & 255

        with caching_allocator_disabled():
            model = Repro()
            example_inputs = (
                torch.randint(
                    0, 256, (200, INNER_DIM), device=self.device, dtype=torch.int64
                ),
                torch.randn(50, 8, device=self.device),
            )
            spec = {
                "x": (Dim.DYNAMIC, Dim.STATIC),
                "y": (Dim.DYNAMIC, Dim.STATIC),
            }
            self.check_model(model, example_inputs, dynamic_shapes=spec)

    @config.patch({"triton.autotune_at_compile_time": None})
    @torch.fx.experimental._config.patch("backed_size_oblivious", True)
    def test_slice_negative_index_backed_symints_no_unbacked(self):
        # [FUNCTION DOC] Like the previous test but with a slice from a negative index based on the
        # size of y; checks the size is a bounded backed expression. Skipped unless a non-MPS GPU.
        # x[-s1:] where x.size(0) = s0-1 should produce Max(s0-1 - s1, 0),
        # not an unbacked symint with a bad fallback value.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires triton")

        INNER_DIM = 4224

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that drops the last row of x, slices the last rows by the
            # size of y, reshapes, expands and does a shift and mask by 8-bit lanes.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Performs the slicing, reshape, expand and per-lane shift
                # and mask.
                x_trimmed = x[:-1]
                sliced = x_trimmed[-y.size(0) :]
                reshaped = sliced.reshape(-1, 128, 33)
                expanded = reshaped.unsqueeze(3).expand(-1, 128, 33, 8)
                shifts = torch.arange(0, 64, 8, device=x.device, dtype=torch.int64)
                return (expanded >> shifts) & 255

        with caching_allocator_disabled():
            model = Repro()
            example_inputs = (
                torch.randint(
                    0, 256, (200, INNER_DIM), device=self.device, dtype=torch.int64
                ),
                torch.randn(50, 8, device=self.device),
            )
            spec = {
                "x": (Dim.DYNAMIC, Dim.STATIC),
                "y": (Dim.DYNAMIC, Dim.STATIC),
            }
            self.check_model(model, example_inputs, dynamic_shapes=spec)

    @config.patch({"triton.autotune_at_compile_time": None})
    def test_stride_with_unbacked_expr(self):
        # [FUNCTION DOC] Checks a strided empty tensor whose size and strides depend on an unbacked
        # scalar times a backed size, filled with 7 and tripled.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model creating an empty strided tensor sized by an unbacked
            # scalar times the size of y, filling it with 7 and multiplying by 3.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Reads the scalar from x, checks it is at least 1 and
                # builds, fills and scales the strided tensor.
                u0 = x.item()
                torch._check(u0 >= 1)
                s0 = y.size(0)
                expr = u0 * s0
                sevens = torch.empty_strided(
                    size=(10, expr, 32), stride=(expr * 32, 32, 1), device=x.device
                ).fill_(7)
                return sevens * 3

        example_inputs = (
            torch.scalar_tensor(2, dtype=torch.int, device=self.device),
            torch.ones(8, device=self.device),
        )
        self.check_model(Repro(), example_inputs)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "bfloat16 is only supported on MacOS 14+",
    )
    def test_size_with_unbacked_add_expr(self):
        # [FUNCTION DOC] Checks that autotune sample inputs get correct sizes for a size that is
        # the sum of a backed size and an unbacked scalar. Skipped on MPS before macOS 14.
        # Tests AOTI autotuning to make sure the correct input tensor sizes
        # are generated for sizes that include an expr such as s0 + u0.

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model combining a repeat-interleave and index select with a
            # repeat whose count is a backed plus unbacked size and concatenates them.
            def forward(self, values, repeats, mask, embeddings, x, z, scalar):
                # [FUNCTION DOC] HELPER: Selects embeddings by repeat-interleaved clamped indexes,
                # repeats x by the backed size of z plus the unbacked scalar, and concatenates both
                # along dim 1.
                repeat_interleave = torch.repeat_interleave(values, repeats)
                index = torch.clamp(repeat_interleave, min=0, max=400).int()
                index_select = torch.index_select(embeddings, 0, index)

                backed = z.size(0)
                unbacked = scalar.item()

                unbacked_add_expr = backed + unbacked
                repeated = x.repeat(unbacked_add_expr, 1)
                return torch.cat([repeated, index_select], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 12,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((1, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 127, dtype=torch.int64, device=self.device),
            torch.scalar_tensor(10, dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.STATIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.STATIC),
            "scalar": (),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    @config.patch({"unbacked_symint_fallback": 128})
    def test_size_with_unbacked_add_expr_transitive(self):
        # [FUNCTION DOC] Edge case where one size expression is checked equal to another and that
        # one is implicitly equal to an unbacked size; autotune input sizes must coalesce them into
        # one size. Skipped unless a GPU and on Windows XPU; fallback value 128.
        # Edge case with torch._check(expr1, expr2) + torch._check(expr2, unbacked).
        # When generating example input sizes for autotuning, it should coalesce
        # expr1, expr2, unbacked into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with repeat counts that mix backed sizes and unbacked
            # values from the list input, a triton add on one repeated tensor and an explicit size
            # equality check.
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                # [FUNCTION DOC] HELPER: Builds two repeated tensors with sizes from backed plus
                # unbacked values, runs the add kernel on one, concatenates it with the
                # index-selected embeddings (implicitly tying sizes), adds the repeated tensors,
                # checks size equality and returns the concatenation and sum.
                index = torch.repeat_interleave(values, repeats)
                index_select = torch.index_select(embeddings, 0, index)

                u0, u1 = lst.tolist()
                backed0, backed1 = z.size(0), z.size(1)

                repeated0 = y.repeat(backed0 + u0, 1)
                repeated1 = x.repeat(backed1 + u1, 1)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                # Implicitly add torch._check(expr2, unbacked)
                cat = torch.cat([out1, index_select], dim=1)
                add = repeated0 + repeated1

                # Explicitly add torch._check(expr1, expr2)
                torch._check(repeated0.size(0) == out1.size(0))
                return cat, add

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 10], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    @config.patch({"unbacked_symint_fallback": 128})
    def test_size_with_unbacked_add_and_mul_expr(self):
        # [FUNCTION DOC] Edge case with sizes given by an add expression and a mix of add and
        # multiply expressions that are constrained equal; autotune input sizes must coalesce. GPU
        # only; fallback value 128.
        # Edge case with torch._check(add_expr, mul_expr). When generating example
        # input sizes for autotuning, make sure they coalesce into a single size.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that repeats x and y by an add expression and an
            # add-and-multiply expression of backed and unbacked values, runs triton add on each
            # and concatenates.
            def forward(self, values, repeats, mask, embeddings, x, y, z, lst):
                # [FUNCTION DOC] HELPER: Reads three unbacked values from lst, builds the repeated
                # tensors with the two size expressions, applies the add kernel to each and returns
                # their concatenation.
                u0, u1, u2 = lst.tolist()
                backed = z.size(0)
                backed1 = z.size(1)

                unbacked_add_expr = backed + u0
                unbacked_mul_expr = backed1 + (u1 * u2)
                repeated0 = x.repeat(unbacked_add_expr, 1)
                repeated1 = y.repeat(unbacked_mul_expr, 1)
                out0 = torch.empty_like(repeated0)
                out1 = torch.empty_like(repeated1)
                add_kernel[(out0.numel(),)](
                    repeated0, repeated0, out0, out0.numel(), BLOCK_SIZE=2
                )
                add_kernel[(out1.numel(),)](
                    repeated1, repeated1, out1, out1.numel(), BLOCK_SIZE=2
                )

                return torch.cat([out1, out0], dim=1)

        example_inputs = (
            torch.ones(64, dtype=torch.int64, device=self.device),
            torch.ones(64, dtype=torch.int64, device=self.device) * 24,
            torch.ones((768,), dtype=torch.int64, device=self.device).bool(),
            torch.randn((401, 8), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.randn((2, 256), dtype=torch.bfloat16, device=self.device),
            torch.ones(758, 758, dtype=torch.int64, device=self.device),
            torch.tensor([10, 5, 2], dtype=torch.int32, device=self.device),
        )
        spec = {
            "values": (Dim.DYNAMIC,),
            "repeats": (Dim.DYNAMIC,),
            "mask": (Dim.DYNAMIC,),
            "embeddings": (Dim.DYNAMIC, Dim.STATIC),
            "x": (Dim.DYNAMIC, Dim.STATIC),
            "y": (Dim.DYNAMIC, Dim.STATIC),
            "z": (Dim.DYNAMIC, Dim.DYNAMIC),
            "lst": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=spec)

    def test_wide_tolist_sum_precomputed_for_triton_autotune(self):
        # [FUNCTION DOC] Checks that a long list of size values (a thousand entries) converted to a
        # list and summed is precomputed so triton autotune sees it. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Model multiplying x by the sum of the entries of sizes.
            def forward(self, x, sizes):
                # [FUNCTION DOC] HELPER: Converts sizes to a list, sums it and scales x by that.
                return x * sum(sizes.tolist())

        n = 1000
        example_inputs = (
            torch.randn(4, device=self.device),
            torch.ones(n, dtype=torch.int64, device=self.device),
        )
        dynamic_shapes = {
            "x": (Dim.STATIC,),
            "sizes": (Dim.STATIC,),
        }
        self.check_model(Repro(), example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Some archs don't support flash SDPA"
    )
    def test_fallback_kernel_with_symexpr_output(self):
        # [FUNCTION DOC] Exports a flash attention model whose reshape sizes are symbolic
        # expressions of dynamic dims, compiles with the AOT compile API, loads it and asserts
        # closeness to eager. GPU only; requires flash attention platform support.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping q, k and v with symbolic sizes and calling the
            # flash attention op, returning its first output.
            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Reshapes each of q, k, v to four dims derived from their
                # shapes, calls the flash attention op and returns the attention output.
                q = q.reshape(
                    q.shape[0],
                    2,
                    q.shape[2] * q.shape[3],
                    q.shape[1] // 2,
                )
                k = k.reshape(
                    k.shape[0],
                    2,
                    k.shape[2] * k.shape[3],
                    k.shape[1] // 2,
                )
                v = v.reshape(
                    v.shape[0],
                    2,
                    v.shape[2] * v.shape[3],
                    v.shape[1] // 2,
                )

                res = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    k,
                    v,
                )
                return res[0]

        m = Module().to(device=self.device)
        # Use head_dim = q.shape[1] // 2 = 64, which is supported by both CUDA
        # and XPU flash attention (XPU supports: 64, 96, 128, 192).
        tensor_shape = (4, 128, 4, 4)
        inputs = (
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
            torch.randn(tensor_shape, dtype=torch.float16, device=self.device),
        )

        dynamic_shapes = {
            "q": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "k": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
            "v": {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        }
        ep = torch.export.export(m, inputs, dynamic_shapes=dynamic_shapes, strict=False)
        path = torch._inductor.aot_compile(ep.module(), inputs)
        aot_model = torch._export.aot_load(path, device=self.device)
        torch.testing.assert_close(m(*inputs), aot_model(*inputs))

    @unittest.skipIf(IS_MACOS, "fp8 is not supported on Mac")
    def test_aoti_fp8(self):
        # [FUNCTION DOC] Checks conversion of e4m3 and e5m2 float8 inputs to float32 and addition.
        # Skipped on macOS, and on non-CPU devices without FP8 support.
        if self.device != "cpu" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(
                "FP8 is only supported on H100+, SM 8.9 and MI300+ devices"
            )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its two inputs after converting both to float32.
            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Converts x1 and x2 to float32 and returns the sum.
                return x1.to(torch.float32) + x2.to(torch.float32)

        m = M().eval().to(self.device)
        x = torch.randn(16, 16, device=self.device)
        x1 = x.to(torch.float8_e4m3fn)
        x2 = x.to(torch.float8_e5m2)

        self.check_model(m, (x1, x2))

    def test_aoti_constant_tensor(self):
        # [FUNCTION DOC] Checks a model using two constant tensors as weight and bias for a linear.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two constant 4x4 ones tensors used as linear weight
            # and bias.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores two constant ones tensors on the device.
                super().__init__()
                self.a = torch.ones(4, 4, device=device)
                self.b = torch.ones(4, 4, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear op with the two constants to x.
                return torch.ops.aten.linear.default(x, self.a, self.b)

        example_inputs = (torch.ones(4, 4, device=self.device),)
        self.check_model(Foo(self.device), example_inputs)

    def test_aoti_constant_tensor_name_collision(self):
        # [FUNCTION DOC] Exports a model with buffers named like generated tensor constants (a
        # top-level and a submodule buffer sharing a name), after running decompositions to trigger
        # the name collision, then checks the module with check_model.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a persistent ones buffer named like a
            # generated constant.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Registers a persistent size-one buffer.
                super().__init__()
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls a linear attribute that this module does not define.
                # KNOWN ISSUE: this class has no linear attribute, so calling it would fail; it is
                # never called by the test.
                return self.linear(x)

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two persistent buffers, a submodule with a same-named
            # buffer, and a forward that mutates a slice of one buffer and selects columns of x.
            def __init__(self, user_float_feature_idx, device):
                # [FUNCTION DOC] HELPER: Stores the feature index list, registers two persistent
                # buffers and creates the submodule.
                super().__init__()
                self.user_float_feature_idx = user_float_feature_idx
                self.register_buffer(
                    "_tensor_constant0",
                    torch.ones(5, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.register_buffer(
                    "_tensor_constant1",
                    torch.ones(1, device=device, dtype=torch.float32),
                    persistent=True,
                )
                self.sub_mod = SubModule(device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets a slice of the first buffer to one in place, selects
                # columns of x by the stored index, and returns that with the buffers and the
                # submodule buffer.
                self._tensor_constant0[1:2] = 1
                return (
                    torch.index_select(
                        x, 1, torch.tensor(self.user_float_feature_idx, device=x.device)
                    ),
                    self._tensor_constant0,
                    self._tensor_constant1,
                    self.sub_mod._tensor_constant1,
                )

        example_inputs = (torch.ones(4, 4, device=self.device),)
        user_float_feature_idx = [1]
        # we have to have run_decomposition first to trigger the name collision
        ep = torch.export.export(
            Foo(user_float_feature_idx, self.device), example_inputs, strict=False
        ).run_decompositions()
        gm = ep.module()
        self.check_model(gm.to(self.device), example_inputs)

    def test_large_grid(self):
        # [FUNCTION DOC] Checks a reshape plus permute plus clone on a large tensor whose generated
        # grid exceeds 65536 along its second axis. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that reshapes to rows of 2 by 4, permutes the last two dims
            # and clones with contiguous layout.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, primals_5):
                # [FUNCTION DOC] HELPER: Reshapes, permutes and clones contiguously, returning
                # the result.
                view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
                primals_5 = None
                permute = torch.ops.aten.permute.default(view, [0, 2, 1])
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                return clone

        # let y_grid = 65537
        s0 = 16777472
        s1 = 8
        example_inputs = (torch.rand(s0, s1, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_cond_simple(self):
        # [FUNCTION DOC] Runs the simple conditional model from the control flow test helpers
        # with predicates prepended to two inputs and a shared dynamic first dim, via check on
        # multiple inputs.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Simple(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_nested(self):
        # [FUNCTION DOC] Runs the nested conditional model with three predicates prepended to three
        # inputs sharing a dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p0": {},
            "p1": {},
            "p2": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Nested(),
            prepend_predicates(inputs, num_predicates=3),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_parameters(self):
        # [FUNCTION DOC] Runs the conditional model that uses module parameters (built on the test
        # device) with a predicate prepended to one input with a dynamic first dim.
        inputs = (torch.randn((10, 20), device=self.device),)
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.Parameters(self.device),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_reinterpret_view_inputs_outputs(self):
        # [FUNCTION DOC] Runs the conditional model whose branches take and return reinterpreted
        # views. The dynamic dim minimum is 5 because slicing in the branch and size-oblivious
        # metadata extraction assume the sliced size is not 0 or 1 (see the TODO in the code).
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # TODO: the min value need to be 5 because in the body_fn, we're slicing over z1[2:],
        # since the output size is [dim0_ab-3], when we extract tensor metadata out of the output
        # we call guard_size_oblivious, which assumes the dim0_ab-3 != 0 or 1. So we have to set
        # the minimum to 5 for now. We need to relax this restriction either by writing a less
        # constrained shape checking in fake impl of cond.
        dim0_ab = Dim("s0", min=5, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.ReinterpretView(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @requires_gpu
    def test_cond_with_replace_view_ops(self):
        # [FUNCTION DOC] Exports a model with a conditional on a cache plus linear, decomposes
        # it, applies the pass replacing view ops with view-copy ops, and compiles and packages
        # it with max-autotune using Triton and ATen gemm backends. Compile-only; no numerics
        # check. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class CondModelWithViewAndLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward branches on whether the
            # cache has any nonzero values.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 4 by 4 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, cache, x):
                # [FUNCTION DOC] HELPER: Defines two branch functions and conditionally applies one
                # to the cache and x, returning the result.
                # KNOWN ISSUE: the call passes the false branch function for both branches, so the
                # true branch function is never used.
                def true_fn(cache, x):
                    # [FUNCTION DOC] HELPER: True branch: returns the cache plus one (unused, see
                    # the enclosing note).
                    return cache + 1.0

                def false_fn(cache, x):
                    # [FUNCTION DOC] HELPER: False branch: applies the linear layer to x and
                    # reshapes to 1 by 2 by 4 by 4.
                    return self.linear(x).view(1, 2, 4, 4)

                cache_is_initialized = (cache != 0).any()
                return torch.cond(cache_is_initialized, false_fn, false_fn, [cache, x])

        example_input = (
            torch.zeros(1, 2, 4, 4, dtype=torch.float32, device=self.device),
            torch.randn(8, 4, dtype=torch.float32, device=self.device),
        )
        model = CondModelWithViewAndLinear().to(device=self.device)
        exported_program = torch.export.export(model, example_input)
        program = exported_program.run_decompositions()
        program = program._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON,ATEN"}
        ):
            torch._inductor.aoti_compile_and_package(program)

    def test_cond_with_multiple_outputs(self):
        # [FUNCTION DOC] Runs the conditional model with multiple outputs, with a predicate and
        # three inputs; the third input has its own dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((30, 40), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dim0_c = Dim("s1", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
            "c": {0: dim0_c, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.MultipleOutputs(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_with_outer_code_before_after(self):
        # [FUNCTION DOC] Runs the conditional model that has computation before and after the cond,
        # with a predicate prepended to two inputs with dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterCode(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_use_buffers_from_outer_scope(self):
        # [FUNCTION DOC] Runs the conditional model whose branches use buffers computed in the
        # outer scope, over a predicate and three inputs with a shared dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_abc = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "p": {},
            "a": {0: dim0_abc, 1: None},
            "b": {0: dim0_abc, 1: None},
            "c": {0: dim0_abc, 1: None},
        }
        self.check_model_with_multiple_inputs(
            CondModels.OuterBuffers(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the conditional model whose predicate is not
        # a tensor (derived from shapes) over two inputs; when dynamic, runs two input sets with
        # different sizes and independent dynamic first dims.
        inputs1 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
        )
        inputs2 = (
            torch.randn((10, 20), device=self.device),
            torch.randn((5, 20), device=self.device),
        )
        inputs = (inputs1,)
        dynamic_shapes = None
        if dynamic:
            inputs = (inputs1, inputs2)
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_a, 1: None},
                "b": {0: dim0_b, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.WithNonTensorPredicate(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_unbacked_symint_closure(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the conditional model whose branch closes
        # over an unbacked integer, with a prepended predicate and three inputs; dynamic adds
        # dynamic first dims.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((15, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            dim0_a = Dim("s0", min=2, max=1024)
            dim0_b = Dim("s1", min=2, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.UnbackedSymIntClosure(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    @common_utils.parametrize("dynamic", [False, True])
    def test_cond_mismatched_branch_output(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the conditional model whose branches return
        # outputs of different sizes. When dynamic, dynamic dims have minimum 4 because slicing
        # past the first two rows would otherwise specialize. Skipped on Windows (crash).
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dynamic_shapes = None
        if dynamic:
            # Note the minimum has to be 4 because the model
            # is slicing over the first dim with [2:], if first
            # dim is 2 or 3, the slicing will be 0/1 specialized,
            # causing a constraint violation error.
            dim0_a = Dim("s0", min=4, max=1024)
            dim0_b = Dim("s1", min=4, max=1024)
            dynamic_shapes = {
                "p": {},
                "x": {0: dim0_a, 1: None},
                "y": {0: dim0_b, 1: None},
                "z": {0: dim0_a, 1: None},
            }
        self.check_model_with_multiple_inputs(
            CondModels.MismatchedOutputSize(),
            prepend_predicates(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input(self):
        # [FUNCTION DOC] Checks a conditional on the first dimension of x being large, whose
        # branches close over sizes taken from other inputs, using two input sets and dynamic
        # first dims.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conditional on the size of x, with branches that
            # use the sizes of y and z.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Reads sizes of y and z, defines two branches and applies
                # the cond on whether the first dim of x exceeds five.
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: adds the size of y to x.
                    return x + a

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: adds z scaled by its size to x.
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        self.check_model_with_multiple_inputs(
            M(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_cond_symint_input_disable_one_pass(self):
        # [FUNCTION DOC] Same conditional with symbolic integer inputs as the previous test, but
        # with compile-time triton autotuning disabled via config.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conditional on the size of x, with branches that
            # use the sizes of y and z.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Reads sizes of y and z, defines two branches and applies
                # the cond on whether the first dim of x exceeds five.
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: adds the size of y to x.
                    return x + a

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: adds z scaled by its size to x.
                    return x + b * z

                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        input1 = (
            torch.ones(3, 3, device=self.device),
            torch.ones(5, device=self.device),
            torch.ones(3, 3, device=self.device),
        )
        input2 = (
            torch.ones(10, 3, device=self.device),
            torch.ones(6, device=self.device),
            torch.ones(10, 3, device=self.device),
        )
        inputs = (input1, input2)
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
            self.check_model_with_multiple_inputs(
                M(),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )

    @common_utils.parametrize("max_autotune", [False, True])
    def test_cond_cpu_predicate_cuda_operands(self, max_autotune):
        # [FUNCTION DOC] Parametrized on max_autotune. Regression test for a conditional with a CPU
        # predicate and GPU operands: the device must come from the operands, not the predicate,
        # during autotuning. Skipped unless CUDA or XPU. Its docstring describes the same.
        """
        Test torch.cond with CPU predicate and CUDA operands.
        This is a regression test for the bug where inductor incorrectly
        determined device from [predicate] + operands, causing CPU predicates
        to force CUDA outputs onto CPU during autotuning.
        """
        if self.device != "cuda" and self.device != "xpu":
            raise unittest.SkipTest("requires CUDA or XPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three linear layers; branches apply different layers
            # to x and a final linear is applied to the cond result.
            def __init__(self, input_dim=4, hidden_dim=8):
                # [FUNCTION DOC] HELPER: Creates three biased linear layers with the given input
                # and hidden dims.
                super().__init__()
                self.true_linear = torch.nn.Linear(input_dim, hidden_dim, bias=True)
                self.false_linear = torch.nn.Linear(input_dim, hidden_dim, bias=True)
                self.another_linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

            def forward(self, predicate: torch.Tensor, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Applies the conditional on the predicate and then the
                # final linear layer.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: linear scaled by two.
                    return self.true_linear(x) * 2.0

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: linear plus one.
                    return self.false_linear(x) + 1.0

                res = torch.cond(predicate, true_fn, false_fn, (x,))
                return self.another_linear(res)

        # Predicate on CPU, data on CUDA
        predicate = torch.tensor(True, dtype=torch.bool, device="cpu")
        x = torch.randn(4, 4, device=self.device)
        example_inputs = (predicate, x)

        with config.patch({"max_autotune": max_autotune}):
            self.check_model(
                Model().to(self.device),
                example_inputs=example_inputs,
            )

    def test_switch_simple(self):
        # [FUNCTION DOC] Runs the simple switch model with an index counter prepended and taking
        # values zero, one and two, with a dynamic first dim for the input.
        inputs = (torch.randn((10, 20), device=self.device),)
        dim0_x = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "idx": {},
            "x": {0: dim0_x, 1: None},
        }
        self.check_model_with_multiple_inputs(
            SwitchModels.Simple(),
            prepend_counters(inputs, counter_values=(0, 1, 2)),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_simple(self):
        # [FUNCTION DOC] Runs the simple while-loop model with a loop counter prepended to two
        # inputs with a shared dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Simple(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_nested(self):
        # [FUNCTION DOC] Runs the nested while-loop model with two counters prepended to two inputs
        # with a shared dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "cj": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Nested(),
            prepend_counters(inputs, num_counters=2),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_code(self):
        # [FUNCTION DOC] Runs the while-loop model with code outside the loop, with a counter and
        # two inputs with dynamic first dim.
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterCode(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_parameters(self):
        # [FUNCTION DOC] Runs the while-loop model with parameters on a double precision input with
        # a dynamic first dim. GPU only.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        inputs = (
            torch.randn(
                (
                    10,
                    20,
                ),
                dtype=torch.float64,
                device=self.device,
            ),
        )
        dim0_a = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "c": {},
            "a": {0: dim0_a, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Parameters(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_outer_buffers(self):
        # [FUNCTION DOC] Runs the while-loop model that uses outer-scope buffers with static shapes
        # (dynamic shapes are disabled because of issue 123596).
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        # dynamic shapes don't work now due to
        # https://github.com/pytorch/pytorch/issues/123596
        # dim0_ab = Dim("s0", min=2, max=1024)
        # dynamic_shapes = {
        #     "c": {},
        #     "a": {0: dim0_ab, 1: None},
        #     "b": {0: dim0_ab, 1: None},
        # }
        dynamic_shapes = None
        self.check_model_with_multiple_inputs(
            WhileLoopModels.OuterBuffers(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    def test_while_loop_with_pytree_inputs(self):
        # [FUNCTION DOC] Runs the while-loop model carrying nested list and dictionary tensors in
        # the loop state, with static shapes and one input set.
        inputs = (
            torch.tensor(0, device=self.device),
            (
                [torch.randn(10, 20, device=self.device)],
                {
                    "x": torch.randn(10, 20, device=self.device),
                    "y": torch.randn(10, 20, device=self.device),
                },
            ),
        )
        self.check_model_with_multiple_inputs(
            WhileLoopModels.PytreeCarry(),
            [inputs],
            dynamic_shapes=None,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_unbacked_symint_closure(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the while-loop model with a closure over an
        # unbacked integer, with optional dynamic first dims.
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.UnbackedSymIntClosure(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_mixed_device(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the while-loop model that mixes devices,
        # with optional dynamic first dims.
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.MixedDevice(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_sym_expr_cond(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the while-loop model with a symbolic
        # expression condition, with optional dynamic first dims.
        inputs = (
            torch.randn(10, 20, device=self.device),
            torch.randn(10, 20, device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.SymExprCond(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    def test_while_loop_with_conv(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Runs the while-loop model with a convolution on a
        # four dimensional double precision input, with optional dynamic first dim.
        inputs = (torch.randn(2, 4, 4, 4, device=self.device, dtype=torch.float64),)
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {
                "c": {},
                "x": {0: dim0_ab, 1: None},
            }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Conv(self.device),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @config.patch({"is_predispatch": True})
    def test_constant(self):
        # [FUNCTION DOC] Checks, with predispatch export enabled, a model creating a constant
        # tensor from the last dimension size, applying sqrt of three times it and scaling x.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that scales x by sqrt of three times its last dimension
            # size, built as a new tensor on the stored device.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the device (overriding the module attribute with
                # the device name).
                super().__init__()
                self.device = device

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a float tensor holding the last dim, takes sqrt of
                # three times it and multiplies x.
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        self.check_model(M(self.device), (torch.randn(5, 5, device=self.device),))

    @unittest.skipIf(IS_MACOS, "no CUDA on Mac")
    def test_zero_grid_with_backed_symbols(self):
        # [FUNCTION DOC] Compiles an add with a dynamic first dim and broadcasting, loads it as a
        # package, runs with nonzero size, then reruns with first dim zero (an empty grid) and
        # compares with eager. GPU only; skipped on macOS.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding x and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, b):
                # [FUNCTION DOC] HELPER: Returns x plus b.
                return x + b

        example_inputs = (
            torch.randn((3, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        dynamic_shapes = {
            "x": {0: Dim("dx"), 1: Dim.STATIC},
            "b": None,
        }

        # Compile & run model where dynamic dim size > 0.
        package_path: str = AOTIRunnerUtil.compile(
            Repro(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        aot_inductor_module(*example_inputs)

        # Re-run where dynamic dim size is 0.
        example_inputs = (
            torch.randn((0, 2), device=self.device),
            torch.randn((1, 2), device=self.device),
        )
        actual = aot_inductor_module(*example_inputs)
        expected = Repro()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_repeat_interleave(self):
        # [FUNCTION DOC] Checks the repeat-interleave op with a tensor of repeats and a given
        # output size.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying repeat-interleave of x with an output size
            # of twelve.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the repeat-interleaved tensor.
                return torch.ops.aten.repeat_interleave.Tensor(x, output_size=12)

        example_inputs = (torch.ones((1,), dtype=torch.int32, device=self.device) * 12,)
        self.check_model(Repro(), example_inputs)

    def test_dynamic_cat(self):
        # [FUNCTION DOC] Checks concatenation of two inputs along dim 0 where each has its own
        # dynamic first dim.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating a and b along dim 0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the concatenation of a and b.
                return torch.cat([a, b], dim=0)

        a = torch.randn(2, 4, device=self.device)
        b = torch.randn(3, 4, device=self.device)
        dim0_a = Dim("dim0_a", min=1, max=10)
        dim0_b = Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        example_inputs = (a, b)
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_buffer_mutation_1(self):
        # [FUNCTION DOC] Checks a model that mutates a buffer in place by adding one then adds it
        # to x.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one random buffer mutated in forward.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Registers a random 4x4 buffer.
                super().__init__()
                self.foo = torch.nn.Buffer(torch.randn(4, 4, device=device))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to the buffer in place and returns buffer plus x.
                self.foo.add_(1)
                return self.foo + x

        example_inputs = (torch.rand(4, 4, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    def test_non_tensor_input(self):
        # [FUNCTION DOC] Compiles the add op directly with a non-tensor keyword argument alpha of
        # two for two cpp simdlen settings, loads the legacy runner, runs on both tensors, and
        # checks the result is a one-element list equal to the eager result.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module used to compute the eager reference of add with alpha.
            def forward(self, a, b, alpha=1.0):
                # [FUNCTION DOC] HELPER: Returns the add of a and b with the given alpha.
                return torch.add(a, b, alpha=alpha)

        a = torch.randn(10, device=self.device)
        b = torch.randn(10, device=self.device)

        for simdlen in [0, None]:
            with torch._inductor.config.patch({"cpp.simdlen": simdlen}):
                so_path = torch._export.aot_compile(
                    torch.ops.aten.add,
                    args=(a, b),
                    kwargs={"alpha": 2.0},
                )
                kernel_runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)
                res = kernel_runner.run([a, b])
                self.assertTrue(isinstance(res, list))
                self.assertTrue(len(res) == 1)
                self.assertEqual(Model()(a, b, alpha=2.0), res[0])

    def test_buffer_mutation_2(self):
        # [FUNCTION DOC] Checks a model that scales one buffer in place and assigns an element of
        # it into another buffer, returning results that depend on both.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two arange buffers that are mutated in forward.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Registers two buffers each holding zero through nine.
                super().__init__()
                self.foo = torch.nn.Buffer(torch.arange(10, device=device))
                self.bar = torch.nn.Buffer(torch.arange(10, device=device))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Doubles the second buffer in place, copies its first
                # element into slot five of the first buffer, and returns x plus the second and x
                # times the first.
                self.bar.mul_(2)
                self.foo[5] = self.bar[0]
                return x + self.bar, x * self.foo

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    @skipIfWindows(
        msg="OpenMP crashed application on windows"
    )  # TODO: (xuhancn) need to root cause and fix.
    def test_buffer_mutation_3(self):
        # [FUNCTION DOC] Checks index assignment into cache buffers via a position tensor, and
        # asserts the generated code allocates strided empty buffers exactly twice. Skipped on
        # Windows (OpenMP crash).
        class KVCache(torch.nn.Module):
            # [CLASS DOC] HELPER: Key/value cache module with two zero buffers of shape batch,
            # heads, sequence length, head dim.
            def __init__(
                self,
                max_batch_size,
                max_seq_length,
                n_heads,
                head_dim,
                dtype=torch.float,
            ):
                # [FUNCTION DOC] HELPER: Builds the cache shape and registers zero key and value
                # buffers on CPU.
                super().__init__()
                cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
                self.k_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))
                self.v_cache = torch.nn.Buffer(torch.zeros(cache_shape, dtype=dtype))

            def update(self, input_pos, k_val, v_val):
                # [FUNCTION DOC] HELPER: Writes k_val and v_val into the caches at the given
                # positions along the sequence axis in place and returns both caches.
                # input_pos: [S], k_val: [B, H, S, D]
                k_out = self.k_cache
                v_out = self.v_cache
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val

                return k_out, v_out

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a key/value cache that updates in forward and returns
            # the keys plus one and values divided by two.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Creates the cache with batch 1, sequence 256, 6 heads,
                # head dim 48. The device argument is not used.
                super().__init__()
                self.kv_cache = KVCache(1, 256, 6, 48)

            def forward(self, inp_pos, k, v):
                # [FUNCTION DOC] HELPER: Updates the cache at the given position and returns
                # derived tensors.
                self.kv_cache.update(inp_pos, k, v)
                return self.kv_cache.k_cache + 1, self.kv_cache.v_cache / 2

        example_inputs = (
            torch.tensor([0], device=self.device),
            torch.randn(1, 6, 1, 48, device=self.device),
            torch.randn(1, 6, 1, 48, device=self.device),
        )
        model = Model(self.device)
        self.check_model(model, example_inputs)
        self.code_check_count(model, example_inputs, "empty_strided", 2)

    def test_buffer_mutation_4(self):
        # [FUNCTION DOC] Compile-only check of a model with a CPU int64 buffer moved to the input
        # device. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a CPU integer buffer added to x after moving to
            # its device.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a CPU int64 buffer of size 38.
                super().__init__()
                self.register_buffer(
                    "_tensor_constant0",
                    torch.randint(1, size=[38], dtype=torch.int64, device="cpu"),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the buffer, moved to x device, to x.
                return x + self._tensor_constant0.to(x.device)

        example_inputs = (
            torch.randint(1, size=[38], dtype=torch.int64, device=GPU_TYPE),
        )
        torch._export.aot_compile(Model(), example_inputs)

    @skipCUDAIf(True, "Test for x86 backend")
    @skipIfXpu(msg="Test for x86 backend")
    @unittest.skipUnless(IS_X86, "Test for x86 backend")
    @unittest.skipIf(sys.platform == "darwin", "Skip MacOS")
    @unittest.skipIf(IS_FBCODE, "Need newer ideep")
    def test_buffer_mutation_and_force_mmap_weights(self):
        # [FUNCTION DOC] Regression test for forced mmap weights together with a static quantized
        # linear whose lowering creates an int32 zero point constant; the constants size must be
        # computed correctly. Uses freezing, no_grad, and x86 CPU only (skipped for CUDA, XPU,
        # macOS, fbcode).
        """
        This issue occurs when weight zero point is int64 and aot_inductor.force_mmap_weights is ON.
        The lowering path of qlinear will create a new constant buffer of int32 type for weight zero point,
        and the CodeCache computes the constant buffer size wrong.
        Original PR: https://github.com/pytorch/pytorch/pull/139054
        """
        from torch.testing._internal.common_quantization import (
            _static_quantized_linear_module,
        )

        example_inputs = (torch.randn(32, 16),)
        model = _static_quantized_linear_module(
            N=15, K=16, bias=True, example_input=example_inputs[0]
        )
        model = torch.export.export(model, example_inputs, strict=True).module()
        with (
            config.patch({"freezing": True, "aot_inductor.force_mmap_weights": True}),
            torch.no_grad(),
        ):
            self.check_model(model, example_inputs)

    def test_fallback_mem_leak_fix(self):
        # [FUNCTION DOC] Compiles a model with an as_strided view and boolean-index fallback
        # kernel, loads the package, checks that device allocated memory is unchanged across a run,
        # and compares outputs with eager. GPU only.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding x and y, taking a strided view of it and indexing
            # by a mask.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y, idx):
                # [FUNCTION DOC] HELPER: Returns the strided view of x plus y and the result of
                # masking it.
                tmp = x + y
                w = torch.ops.aten.as_strided(tmp, x.shape, x.stride())
                out = torch.ops.aten.index.Tensor(w, [idx])
                return w, out

        example_inputs = (
            torch.randn(4, 1, 4, device=GPU_TYPE),
            torch.randn(4, 1, 4, device=GPU_TYPE),
            torch.randn(4, device=GPU_TYPE) > 0,
        )

        dim0 = Dim("dim0", min=1, max=2048)
        dynamic_shapes = {
            "x": {0: dim0},
            "y": {0: dim0},
            "idx": {0: dim0},
        }
        package_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        device: int = device_interface.current_device()
        mem_before = device_interface.memory_allocated(device)
        aot_inductor_module(*example_inputs)
        mem_after = device_interface.memory_allocated(device)
        self.assertEqual(mem_before, mem_after)

        actual = aot_inductor_module(*example_inputs)
        expected = Model()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @requires_multigpu()
    def test_replicate_on_devices(self):
        # [FUNCTION DOC] Compiles on device zero, then loads the package and runs it on each GPU,
        # comparing with a CPU reference. Needs multiple GPUs.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two constant weights scaling x and y, returning
            # their sum.
            def __init__(self, w1, w2):
                # [FUNCTION DOC] HELPER: Stores the two weights.
                super().__init__()
                self.w1 = w1
                self.w2 = w2

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x times w1 plus y times w2.
                a = x * self.w1
                b = y * self.w2
                return a + b

        w1 = torch.randn(10, 10)
        w2 = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(w1, w2)(*inputs)

        # Compile model with AOTInductor
        device_interface = get_interface_for_device(GPU_TYPE)
        with device_interface.device(0):
            package_path = AOTIRunnerUtil.compile(
                model=Model(
                    w1.to(torch.device(GPU_TYPE, 0)), w2.to(torch.device(GPU_TYPE, 0))
                ),
                example_inputs=tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        # Run model on gpu:N
        for i in range(device_interface.device_count()):
            with device_interface.device(i):
                example_inputs = tuple(t.to(torch.device(GPU_TYPE, i)) for t in inputs)
                optimized = torch._inductor.aoti_load_package(package_path)
                result_gpu = optimized(*example_inputs)
            self.assertTrue(same(result_cpu, result_gpu.cpu()))

    @requires_multigpu()
    def test_on_gpu_device1(self):
        # [FUNCTION DOC] Compiles and runs a small MLP on device index one with the legacy API and
        # checks against eager. Skipped if device one is unavailable. Needs multiple GPUs.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        device_interface = get_interface_for_device(GPU_TYPE)
        try:
            device_interface.get_device_properties(1)
        except AssertionError:
            raise unittest.SkipTest("GPU device 1 is not available") from None

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Small network of linear, relu, linear and sigmoid.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the four layers.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the layers in sequence.
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x

        device = f"{GPU_TYPE}:1"
        model = Model().to(device)
        example_inputs = (torch.randn(8, 10, device=device),)
        expected = model(*example_inputs)

        so_path = AOTIRunnerUtil.legacy_compile(model, example_inputs)
        optimized = AOTIRunnerUtil.legacy_load(device, so_path)
        actual = optimized(*example_inputs)
        torch.testing.assert_close(actual, expected)

    def test_pytree_inputs(self):
        # [FUNCTION DOC] Checks a model taking a dictionary of tensors input and returning a list
        # with the sum and product of its values.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing and multiplying all tensors in the
            # input dictionary.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x: dict[str, torch.Tensor]):
                # [FUNCTION DOC] HELPER: Accumulates a sum into zeros and a product into ones over
                # all values and returns the pair as a list.
                device = next(iter(x.values())).device
                add_ = torch.zeros(5, device=device)
                mul_ = torch.ones(5, device=device)
                for v in x.values():
                    add_ += v
                    mul_ *= v

                return [add_, mul_]

        self.check_model(
            M(),
            (
                {
                    "x": torch.ones(5, device=self.device),
                    "y": torch.ones(5, device=self.device),
                },
            ),
        )

    @requires_multigpu()
    def test_non_default_gpu_device(self):
        # [FUNCTION DOC] Compiles and runs the model separately under device zero and device one
        # contexts with weights moved to each, comparing both with a CPU reference. Needs
        # multiple GPUs.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding x to a linear of y with a stored weight.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the weight tensor.
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y with the weight.
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10)
        inputs = (torch.randn(10, 10), torch.randn(10, 10))
        result_cpu = Model(weight)(*inputs)

        device_interface = get_interface_for_device(GPU_TYPE)
        with device_interface.device(0), torch.no_grad():
            result_gpu_0 = AOTIRunnerUtil.run(
                Model(weight.to(torch.device(GPU_TYPE, 0))),
                tuple(t.to(torch.device(GPU_TYPE, 0)) for t in inputs),
            )

        with device_interface.device(1), torch.no_grad():
            result_gpu_1 = AOTIRunnerUtil.run(
                Model(weight.to(torch.device(GPU_TYPE, 1))),
                tuple(t.to(torch.device(GPU_TYPE, 1)) for t in inputs),
            )

        self.assertTrue(same(result_cpu, result_gpu_0.cpu()))
        self.assertTrue(same(result_cpu, result_gpu_1.cpu()))

    @requires_multigpu()
    def test_load_package_multiple_gpus(self):
        # [FUNCTION DOC] Compiles a package once and loads it on each GPU by device index, running
        # with inputs moved to that device and comparing with a reference. Needs multiple GPUs.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding x to a linear of y with a stored weight.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the weight tensor.
                super().__init__()
                self.weight = weight

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y with the weight.
                return x + torch.nn.functional.linear(y, self.weight)

        weight = torch.randn(10, 10, device=self.device)
        inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model(weight).to(device=self.device)
        result_ref = model(*inputs)

        package_path = AOTIRunnerUtil.compile(model, inputs)

        # Load AOT package on gpu:N
        device_interface = get_interface_for_device(GPU_TYPE)
        for i in range(device_interface.device_count()):
            device = torch.device(GPU_TYPE, i)
            with device_interface.device(i), torch.no_grad():
                model_package = torch._inductor.aoti_load_package(
                    package_path, device_index=i
                )
                inputs_on_device = [input.to(device=device) for input in inputs]
                result_package = model_package(*inputs_on_device)
            self.assertTrue(same(result_ref.cpu(), result_package.cpu()))

    def test_reuse_kernel(self):
        # [FUNCTION DOC] Checks numerics of a model applying tanh twice with tolerance 1e-4, then
        # checks the generated code loads the tanh kernel once (reused): the MPS kernel lookup
        # appears once, or for GPUs either the fused matmul-tanh reduction kernel (native matmul)
        # or the pointwise tanh kernel load appears exactly once.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying tanh, matmul, tanh and another matmul.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes tanh of x, matmul with y, tanh of that, then
                # matmul of the two.
                a = torch.tanh(x)
                b = torch.mm(a, y)
                c = torch.tanh(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device=self.device),
            torch.randn(87, 87, device=self.device),
        )
        model = Model()

        # 1e-4 is the tol value used in pytorch/torch/_dynamo/utils.py
        self.check_model(model, example_inputs, atol=1e-4, rtol=1e-4)

        if self.device == "mps":
            self.code_check_count(
                model, example_inputs, "aoti_torch_mps_get_kernel_function(", 1
            )
        elif self.device == GPU_TYPE:
            if config.triton.native_matmul:
                self.code_check_count(
                    model, example_inputs, "triton_red_fused_mm_tanh_0(in_ptr0", 1
                )
            else:
                self.code_check_count(
                    model, example_inputs, "triton_poi_fused_tanh_0 = loadKernel(", 1
                )

    def test_reuse_kernel_dynamic(self):
        # [FUNCTION DOC] Checks a model with two structurally identical subgraphs (with
        # separate constants) over dynamic shapes, so that the generated kernels can be reused.
        # Tolerance 1e-5.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two copies of constant weights and biases; forward
            # computes two identical chains of add, expand, multiply, permute, batched matmul, bias
            # add, reshape, scale and a second batched matmul with y, and adds the two results.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores two sets of random float constants (bias of 48 and
                # weights of 6 by 48 by 48) on the device.
                super().__init__()
                self.cst = torch.randn(48, device=device, dtype=torch.float)
                self.weights = torch.randn(6, 48, 48, device=device, dtype=torch.float)
                self.cst_1 = torch.randn(48, device=device, dtype=torch.float)
                self.weights_1 = torch.randn(
                    6, 48, 48, device=device, dtype=torch.float
                )

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Runs the two chains (using different constants but the
                # same inputs) whose shapes depend on the second dimension of x, and returns the
                # sum of their results.
                dim0 = x.size(1)
                add_0 = z + z
                expand_2 = add_0.expand(-1, -1, 48)
                # [s0, 6, 48]
                mul_3 = add_0 * expand_2
                # [6, s0, 48]
                permute_4 = torch.permute(mul_3, (1, 0, 2))
                # [6, s0, 48]
                bmm_5 = torch.bmm(permute_4, self.weights)
                add_6 = bmm_5 + self.cst
                reshape_7 = torch.reshape(add_6, [6, dim0 * 6, 8])
                # [6*s0, 6, 8]
                permute_8 = torch.permute(reshape_7, (1, 0, 2))
                mul_9 = permute_8 * 0.123
                reshape_10 = torch.reshape(y, [8, dim0 * 6, 4])
                # [6*s0, 8, 4]
                permute_11 = torch.permute(reshape_10, (1, 0, 2))
                bmm_12 = torch.bmm(mul_9, permute_11)

                add_0_1 = z + z
                expand_2_1 = add_0_1.expand(-1, -1, 48)
                # [s0, 6, 48]
                mul_3_1 = add_0_1 * expand_2_1
                # [6, s0, 48]
                permute_4_1 = torch.permute(mul_3_1, (1, 0, 2))
                # [6, s0, 48]
                bmm_5_1 = torch.bmm(permute_4_1, self.weights_1)
                add_6_1 = bmm_5_1 + self.cst_1
                reshape_7_1 = torch.reshape(add_6_1, [6, dim0 * 6, 8])
                # [6*s0, 6, 8]
                permute_8_1 = torch.permute(reshape_7_1, (1, 0, 2))
                mul_9_1 = permute_8_1 * 0.123
                reshape_10_1 = torch.reshape(y, [8, dim0 * 6, 4])
                # [6*s0, 8, 4]
                permute_11_1 = torch.permute(reshape_10_1, (1, 0, 2))
                bmm_12_1 = torch.bmm(mul_9_1, permute_11_1)
                return bmm_12 + bmm_12_1

        x = torch.randn(6, 2, 48, device=self.device, dtype=torch.float)
        y = torch.randn(48, 2, 4, device=self.device, dtype=torch.float)
        z = torch.randn(2, 6, 1, device=self.device, dtype=torch.float)
        dim0 = Dim("dim0", min=1, max=2048)
        dynamic_shapes = {
            "x": {1: dim0},
            "y": {1: dim0},
            "z": {0: dim0},
        }

        example_inputs = (x, y, z)
        model = Model(self.device).to(dtype=torch.float)
        self.check_model(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_fake_tensor_device_validation(self):
        # [FUNCTION DOC] Exports on CPU, moves the graph module and inputs to the GPU and
        # asserts that aot_compile raises a ValueError about a device mismatch with the fake
        # inputs. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        # Export on CPU
        exported_program = export(Model(), example_inputs, strict=True)

        # Compile exported model on GPU
        gm = exported_program.graph_module.to(self.device)
        with self.assertRaisesRegex(ValueError, "Device mismatch between fake input"):
            torch._inductor.aot_compile(
                gm, tuple(i.to(self.device) for i in example_inputs)
            )

    def test_fx_gm_return_tuple_validation(self):
        # [FUNCTION DOC] Traces a module with symbolic make_fx whose graph returns a single tensor,
        # and asserts aot_compile raises an AssertionError saying the graph output must be a tuple.
        from torch.fx.experimental.proxy_tensor import make_fx

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus y.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        example_inputs = (torch.randn(10, 10), torch.randn(10, 10))

        gm = make_fx(Model(), tracing_mode="symbolic")(*example_inputs)
        with self.assertRaisesRegex(
            AssertionError,
            r"Graph output must be a tuple\(\). This is so that we can avoid "
            "pytree processing of the outputs.",
        ):
            torch._inductor.aot_compile(gm, example_inputs)

    def test_consecutive_compiles(self):
        # [FUNCTION DOC] Compiles a symbolically traced module twice with the export aot_compile
        # and raises AssertionError if either call returns None; the second compile exercises a
        # cache hit.
        """Test that compilation behaves correctly with cache hits"""

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        mod = TestModule()
        inp = torch.rand(1)
        mod(inp)
        mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])
        so = torch._export.aot_compile(mod2, (inp,))
        if so is None:
            raise AssertionError("Expected aot_compile to return non-None")
        # compile the 2nd time with cache hit
        so = torch._export.aot_compile(mod2, (inp,))
        if so is None:
            raise AssertionError("Expected aot_compile to return non-None (cache hit)")

    def test_normal_functional(self):
        # [FUNCTION DOC] Checks the functional normal random op under fallback_random.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the functional normal op to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the functional normal op applied to x.
                return torch.ops.aten.normal_functional.default(x)

        with config.patch({"fallback_random": True}):
            self.check_model(Model(), (torch.empty(4, 1, 4, 4, device=self.device),))

    def test_empty_graph(self):
        # [FUNCTION DOC] Checks a graph that just returns its input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        example_inputs = (torch.randn(8, 4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)

    @patch("torch._dynamo.utils.CompileEventLogger.log_instant_event")
    def test_backward_no_op_logging(self, mock_log_instant_event):
        # [FUNCTION DOC] With the instant event logger mocked, compiles an identity graph through
        # the inner compile function and asserts a single instant event named backward no-op was
        # logged with no compile id at the PT2 compile level.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        model = Model()
        dummy_input = torch.randn(1, 5)

        from torch._dynamo.utils import CompileEventLogLevel
        from torch._inductor import compile_fx

        graph_module = torch.fx.symbolic_trace(model)
        compile_fx._compile_fx_inner(graph_module, (dummy_input,))
        mock_log_instant_event.assert_called_once_with(
            "backward no-op",
            metadata={"compile_id": None},
            log_level=CompileEventLogLevel.PT2_COMPILE,
        )

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_dup_unbacked_sym_decl(self):
        # [FUNCTION DOC] Checks a model that indexes with the same boolean mask twice (two
        # identical unbacked size declarations). Skipped in fbcode.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a division of sin of a masked selection by a
            # second identical masked selection of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a mask of near-nonzero values, selects twice with
                # it, and divides sin of the first selection by the second.
                abs_1 = torch.ops.aten.abs.default(x)
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                sin = torch.ops.aten.sin.default(index_1)
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                return div_3

        example_inputs = (torch.randn(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    # This exercises _eliminate_unbacked path in ShapeEnv
    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_dup_unbacked_sym_decl_with_refinement(self):
        # [FUNCTION DOC] Like the duplicate unbacked declaration test but with an extra size check
        # on the first selection, exercising the unbacked elimination path in the shape environment
        # (as the comment above the test says). Input is all ones. Skipped in fbcode.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a division of sin of a masked selection by a
            # second identical masked selection of x, after checking the selection size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds the mask, selects, asserts the selection has 256
                # elements, and returns sin of the first selection divided by the second.
                abs_1 = torch.ops.aten.abs.default(x)
                lt = torch.ops.aten.lt.Scalar(abs_1, 0.001)
                eq = torch.ops.aten.eq.Scalar(lt, 0)
                index_1 = torch.ops.aten.index.Tensor(x, [eq])
                torch._check(index_1.size(0) == 4**4)
                sin = torch.ops.aten.sin.default(index_1)
                index_2 = torch.ops.aten.index.Tensor(x, [eq])
                div_3 = torch.ops.aten.div.Tensor(sin, index_2)
                return div_3

        example_inputs = (torch.ones(4, 4, 4, 4).to(self.device),)
        self.check_model(Model(), example_inputs)

    def test_run_with_grad_enabled(self):
        # [FUNCTION DOC] Compiles under no_grad with inputs that require grad, then loads the
        # package and runs it with grad enabled, asserting grad mode is still on and outputs
        # match eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the addmm op.
            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Returns addmm with bias, weight and x.
                return torch.ops.aten.addmm(bias, weight, x)

        m = Model().to(device=self.device)
        x = torch.rand(8, 8, device=self.device, requires_grad=True)
        weight = torch.rand(8, 8, device=self.device, requires_grad=True)
        bias = torch.rand(8, device=self.device, requires_grad=True)
        example_inputs = (x, weight, bias)

        expected = m(*example_inputs)
        expected = pytree.tree_leaves(expected)

        # compiler under no_grad
        with torch.no_grad():
            package_path = AOTIRunnerUtil.compile(m, example_inputs)

        # run under grad enabled
        self.assertTrue(torch.is_grad_enabled())

        optimized = torch._inductor.aoti_load_package(package_path)
        actual = optimized(*example_inputs)
        actual = pytree.tree_leaves(actual)

        self.assertTrue(same(actual, expected))

    def test_return_constant(self):
        # [FUNCTION DOC] Checks a model returning its input and a clone of a stored constant.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x and a clone of a constant tensor.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a random 5 by 5 constant.
                super().__init__()
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x and a clone of the constant.
                a = self.cst.clone()
                return (x, a)

        x = torch.randn(5, device=self.device)
        self.check_model(Model(self.device), (x,))

    def test_return_view_constant(self):
        # [FUNCTION DOC] Checks a model returning its input and a transposed view of a
        # stored constant.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x and a transposed view of a constant tensor.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a random 5 by 5 constant.
                super().__init__()
                self.cst = torch.randn(5, 5, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x and the transposed constant view.
                a = torch.transpose(self.cst, 0, 1)
                return (x, a)

        x = torch.randn(5, device=self.device)
        self.check_model(Model(self.device), (x,))

    def test_profile_benchmark_harness(self):
        # [FUNCTION DOC] Smoke test: compiles a function with torch.compile using the bandwidth
        # profile output option and benchmark harness off, then runs it once. No result assertion
        # beyond running.
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def create_test_fn():
            # [FUNCTION DOC] HELPER: Builds and returns the function to compile.
            def test_fn():
                # [FUNCTION DOC] HELPER: Allocates random batch input and weight, matmuls them,
                # applies a layer norm to the product and returns True.
                inp = torch.randn(
                    batch_size, seq_length, hidden_size, device=self.device
                )
                weight = torch.randn(hidden_size, hidden_size, device=self.device)
                matmul_output = inp @ weight
                torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
                return True

            return test_fn

        fn = torch.compile(
            options={"profile_bandwidth_output": "foo", "benchmark_harness": False}
        )(create_test_fn())
        fn()

    def test_with_profiler(self):
        # [FUNCTION DOC] Checks numerics of a linear model with the bandwidth profiling
        # config enabled.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 10x10 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    def test_with_no_triton_profiler(self):
        # [FUNCTION DOC] Checks numerics of a permute-only model (no triton kernel) with bandwidth
        # profiling enabled.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module transposing a 2D input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x permuted to swap its two dims.
                return torch.permute(x, (1, 0))

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"profile_bandwidth": "1", "profile_bandwidth_regex": ""}):
            self.check_model(Model(), example_inputs)

    def test_repeat_output(self):
        # [FUNCTION DOC] Checks a model that returns the same tensor twice.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sin of x twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pair of sin of x and itself.
                y = torch.sin(x)
                return y, y

        example_inputs = (torch.randn(3, 10, device=self.device),)
        self.check_model(Model(), example_inputs)

    @skipIfRocmArch(NAVI_ARCH)  # regression on ROCm 7.2
    def test_repeated_calling(self):
        # [FUNCTION DOC] Exports, compiles and loads a sin model, warms it, then runs it ten times
        # and asserts result equality with eager and that CUDA allocated memory is unchanged across
        # the loop (no leaks from repeated calls). CUDA only; skipped for Navi ROCm.
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sin of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sin of x.
                return torch.sin(x)

        example_inputs = (torch.randn(10, 10, device=self.device),)
        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                torch.export.export(Model(), example_inputs, strict=True)
            )
        )
        expected = torch.sin(example_inputs[0])

        # Warm up to trigger any one-time allocations
        result = optimized(*example_inputs)
        torch.cuda.synchronize()

        mem_before = torch.cuda.memory_allocated()
        for _ in range(10):
            result = optimized(*example_inputs)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        self.assertEqual(result, expected)
        self.assertEqual(mem_before, mem_after)

    def test_view_outputs(self):
        # [FUNCTION DOC] Checks a model returning a tensor and two views of it, one of the same
        # shape and one with a leading dim added.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sin of x and two view outputs of it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sin result, a view with the same shape and a
                # view with an extra leading dim.
                y = torch.sin(x)
                y_same_size = y.view(*y.shape)
                y_diff_size = y.view(1, *y.shape)
                return y, y_same_size, y_diff_size

        example_inputs = (torch.randn(3, 10, device=self.device),)
        self.check_model(Model(), example_inputs)

    @skip_if_no_torchvision
    def test_missing_cubin(self):
        # [FUNCTION DOC] Checks a dilated ResNet-50-like network returning four feature maps, in
        # eval mode at the highest available float precision, to test that missing cubin handling
        # works. Skipped without torchvision.
        from torchvision.models.resnet import Bottleneck, ResNet

        class Model(ResNet):
            # [CLASS DOC] HELPER: ResNet subclass built from bottleneck blocks with dilation in the
            # last stage, returning intermediate features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a four stage bottleneck ResNet with layers 3, 4, 6,
                # 3 and dilation in the final stage.
                super().__init__(
                    block=Bottleneck,
                    layers=[3, 4, 6, 3],
                    replace_stride_with_dilation=[False, False, True],
                    norm_layer=None,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the stem and four stages and returns the stem output
                # and the first two stage outputs and the last stage output as a list of four.
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                f1 = x
                x = self.maxpool(x)
                x = self.layer1(x)
                f2 = x
                x = self.layer2(x)
                f3 = x
                x = self.layer3(x)
                x = self.layer4(x)
                f4 = x
                return [f1, f2, f3, f4]

        # Call eval() here so that batch_norm won't update the running stats
        # Use float64 to avoid numeric difference failure
        dtype = highest_precision_float(self.device)
        model = Model().to(device=self.device, dtype=dtype).eval()
        example_inputs = (torch.randn(4, 3, 64, 64, device=self.device, dtype=dtype),)
        self.check_model(model, example_inputs)

    def test_triton_next_power_of_2(self):
        # [FUNCTION DOC] Checks a triton kernel taking a scaling factor computed with the next
        # power of two of the max of an input. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching a scaled add kernel using next power of two of
            # the max length.
            def forward(self, a, b, lengths):
                # [FUNCTION DOC] HELPER: Allocates an output, computes the scaling factor from the
                # maximum of lengths, launches the kernel over one program per element and returns
                # the output.
                n_elements = a.numel()
                out = torch.empty_like(a)
                max_len = int(lengths.max())
                scaling_factor = triton.next_power_of_2(max_len)
                add_kernel_with_scaling[(n_elements,)](
                    a,
                    b,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (
            torch.randn(2, device=self.device),
            torch.randn(2, device=self.device),
            torch.arange(end=4, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("minmax", [min, max])
    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_sympy_cpp_printer_min_max(self, minmax):
        # [FUNCTION DOC] Parametrized on min versus max. Checks a triton kernel whose scaling
        # argument is min or max of a backed dimension, an unbacked value and a constant, so the
        # symbolic printer must emit min/max for C++. Dynamic first dims on two inputs. GPU only;
        # skipped on Windows XPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching a scaled add kernel with a scaling factor that
            # is min or max (selected by the test parameter) of a size, an unbacked value and 100.
            def forward(self, a, b, ranks):
                # [FUNCTION DOC] HELPER: Computes the scaling factor from a size and the max of
                # ranks, launches the kernel over one program per element and returns the output.
                n_elements = a.numel()
                out = torch.empty_like(a)
                backed = a.size(0)
                unbacked = int(ranks.max())
                scaling_factor = minmax(backed, unbacked, 100)
                add_kernel_with_scaling[(n_elements,)](
                    a,
                    b,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (
            torch.randn(16, device=self.device),
            torch.randn(16, device=self.device),
            torch.arange(end=4, device=self.device, dtype=torch.int16),
        )
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        torch._dynamo.mark_dynamic(example_inputs[1], 0)
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("grid_type", [1, 2, 3])
    @common_utils.parametrize("num_dims", [1, 2])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotune", [False, True])
    def test_triton_kernel(self, grid_type, num_dims, dynamic, autotune):
        # [FUNCTION DOC] Parametrized over grid style (tuple, lambda, or named function), one or
        # two dimensions, static or dynamic first dim, and an autotuned or fixed-config kernel.
        # Builds a model launching an add kernel with the selected grid and kernel, and checks it.
        # GPU only.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching one of several add kernels with a grid selected
            # by the enclosing test parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Allocates a zeros output, picks sizes, selects the grid
                # and kernel from the test parameters, launches and returns the output.
                output = torch.zeros_like(x)
                if autotune and num_dims == 2:
                    x_elements = output.size()[0]
                    y_elements = output.size()[1]
                else:
                    n_elements = output.numel()

                # Select grid
                if autotune and num_dims == 2:
                    if grid_type == 1:
                        grid = (x_elements, y_elements)
                    elif grid_type == 2:
                        grid = lambda meta: (  # noqa: E731
                            triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                            triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                        )
                    else:

                        def grid_fn(meta):
                            # [FUNCTION DOC] HELPER: Grid function for the 2D autotuned case:
                            # returns the number of tiles in each dimension for the chosen
                            # block sizes.
                            return (
                                triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                                triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                            )

                        grid = grid_fn
                else:
                    if grid_type == 1:
                        grid = (n_elements,)
                    elif grid_type == 2:
                        grid = lambda meta: (  # noqa: E731
                            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                        )
                    else:

                        def grid_fn(meta):
                            # [FUNCTION DOC] HELPER: Grid function for the 1D case: returns the
                            # number of blocks needed to cover the elements.
                            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                        grid = grid_fn

                # Select kernel
                if autotune:
                    if num_dims == 1:
                        add_kernel_autotuned[grid](x, y, output, n_elements)
                    else:
                        add_kernel_2d_autotuned[grid](
                            x, y, output, x_elements, y_elements
                        )
                else:
                    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

        dims = [10] * num_dims
        x = torch.randn(*dims, device=self.device)
        y = torch.randn(*dims, device=self.device)
        dynamic_shapes = []
        if dynamic:
            dim0_x = Dim("dim0_x", min=1, max=10)
            dim0_y = Dim("dim0_y", min=1, max=10)
            dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(Model(), (x, y), dynamic_shapes=dynamic_shapes)

    def test_triton_kernel_dynamic_shape_with_div(self):
        # [FUNCTION DOC] Checks a triton kernel launched with a grid computed from a size divided
        # by four under a dynamic first dim. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Triton kernel with no body, used only to test argument handling.
        @triton.jit
        def pass_kernel(x, num):
            pass

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the empty kernel with a computed element count.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Launches the empty kernel with grid from a quarter of the
                # number of elements, and returns x.
                num = x.numel() // 4

                grid = lambda meta: (triton.cdiv(num, 16),)  # noqa: E731
                pass_kernel[grid](x, num)
                return x

        x = torch.randn(10, device=self.device)
        dim0_x = Dim("dim0_x", min=1, max=10)
        dynamic_shapes = {"x": {0: dim0_x}}
        self.check_model(Model(), (x,), dynamic_shapes=dynamic_shapes)

    def test_triton_kernel_reinterpret_view(self):
        # [FUNCTION DOC] Checks launching a triton add kernel on two differently offset sliced
        # views of a tensor (reinterpret view arguments). GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Triton kernel with no body; defined but not used by the test.
        @triton.jit
        def pass_kernel(x, y):
            pass

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running the add kernel on two column-offset slices of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Allocates a zeros output sized as a slice and launches the
                # add kernel on offset slices with 160 elements; returns the output.
                out = torch.zeros_like(x[:, 4:])
                # the slicing below creates two ReinterpretView
                # instances: with offset=3 and offset=4
                add_kernel[(10,)](
                    in_ptr0=x[:, 3:-1],
                    in_ptr1=x[:, 4:],
                    out_ptr=out,
                    n_elements=160,
                    BLOCK_SIZE=16,
                )
                return out

        example_inputs = (torch.randn(10, 20, device=self.device),)
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_tma_descriptor_1d(self, dynamic, tma_version):
        # [FUNCTION DOC] Parametrized on dynamic and TMA API version. Builds host-side tensor
        # descriptors for three tensors and launches a 1D TMA add kernel, then checks the model.
        # Skipped when the requested triton TMA support is missing or without a GPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating 1D tensor descriptors for a, b and a zeros output
            # and launching the selected TMA kernel.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Creates descriptors with block size 256, launches the
                # kernel over a grid sized by element count and returns the output.
                BLOCK_SIZE = 256
                out = torch.zeros_like(a)
                n_elements = out.numel()

                desc_a, desc_b, desc_out = (
                    create_tensor_descriptor_shim(
                        t, [BLOCK_SIZE], new_api=(tma_version == "new")
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                kernel[grid](
                    desc_a,
                    desc_b,
                    desc_out,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return out

        a = torch.randn(301, device=self.device)
        b = torch.randn(301, device=self.device)
        example_inputs = (a, b)

        dynamic_shapes = None
        if dynamic:
            dim0_ab = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_triton_kernel_tma_descriptor_2d(self, dynamic, tma_version):
        # [FUNCTION DOC] Parametrized on dynamic and TMA API version. Like the 1D test but with
        # 2D descriptors and blocks of 16 by 32 over a 25 by 16 input. Skipped when TMA support
        # is missing.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_2d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_2d_old_api
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating 2D tensor descriptors for a, b and a zeros output
            # and launching the selected 2D TMA kernel.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Creates descriptors with 16 by 32 blocks, launches over a
                # two dimensional grid, and returns the output.
                BLOCK_SIZE_X = 16
                BLOCK_SIZE_Y = 32
                out = torch.zeros_like(a)
                x_size, y_size = out.size()

                desc_a, desc_b, desc_out = (
                    create_tensor_descriptor_shim(
                        t,
                        [BLOCK_SIZE_X, BLOCK_SIZE_Y],
                        new_api=(tma_version == "new"),
                    )
                    for t in (a, b, out)
                )

                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
                )
                kernel[grid](
                    desc_a,
                    desc_b,
                    desc_out,
                    BLOCK_SIZE_X=BLOCK_SIZE_X,
                    BLOCK_SIZE_Y=BLOCK_SIZE_Y,
                )

                return out

        a = torch.randn((25, 16), device=self.device)
        b = torch.randn((25, 16), device=self.device)
        example_inputs = (a, b)

        dynamic_shapes = None
        if dynamic:
            dim0_ab = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "a": {0: dim0_ab, 1: None},
                "b": {0: dim0_ab, 1: None},
            }

        self.check_model(
            Model(),
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    def test_triton_kernel_sympy_expr_arg(self):
        # [FUNCTION DOC] Checks a triton kernel whose element count argument is a max expression of
        # an unbacked scalar item. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching an add kernel with a symbolic element count
            # from e.
            def forward(self, x, e):
                # [FUNCTION DOC] HELPER: Computes the larger of one and the scalar from e, launches
                # the add kernel on one program and returns the zeros-initialized output.
                sympy_expr = max(1, e.item())
                out = torch.zeros_like(x)
                add_kernel[(1,)](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=sympy_expr,
                    BLOCK_SIZE=1,
                )
                return out

        NUMEL = 64
        inputs = (
            torch.randn(NUMEL, device=self.device),
            torch.tensor(NUMEL, device=self.device),
        )
        self.check_model(Model(), inputs)

    def test_triton_kernel_sympy_fn_like_arg(self):
        # [FUNCTION DOC] Regression test for passing a string equal to a math function name as a
        # constexpr argument to a triton kernel, which previously crashed in symbolic expansion.
        # GPU only.
        # This test should hit sympy.expand("sqrt") which crashes with
        # AttributeError: 'function' object has no attribute 'expand'.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching a kernel with a string constexpr argument.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Launches the add kernel with an optional-param string
                # argument and returns the zeros-initialized output.
                out = torch.zeros_like(x)
                add_kernel_with_optional_param[1,](
                    in_ptr0=x,
                    in_ptr1=x,
                    out_ptr=out,
                    n_elements=x.numel(),
                    BLOCK_SIZE=1,
                    ARGS_PASSED="sqrt",  # sqrt is a valid sympy fn
                )
                return out

        inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), inputs)

    def test_triton_kernel_with_none_input(self):
        # [FUNCTION DOC] Checks launching an optional-parameter add kernel with a None tensor
        # argument and again with a real tensor, combining weighted results. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the optional-parameter kernel with and without y
            # and combining the outputs with fixed weights.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs the kernel with None and with y, and returns 2.71
                # times the first plus 3.14 times the second.
                n_elements = x.size()[0]
                BLOCK_SIZE = 1024

                output_wo_y = torch.empty_like(x)
                output_with_y = torch.empty_like(x)

                add_kernel_with_optional_param[(1,)](
                    x,
                    None,
                    output_wo_y,
                    n_elements,
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                add_kernel_with_optional_param[(1,)](
                    x,
                    y,
                    output_with_y,
                    n_elements,
                    ARGS_PASSED="two",
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                return 2.71 * output_wo_y + 3.14 * output_with_y

        example_inputs = (
            torch.randn(1023, device=self.device),
            torch.randn(1023, device=self.device),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_equal_to_1_arg(self):
        # [FUNCTION DOC] Checks an add kernel on single-element inputs, where the element count
        # equals one and is specialized. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching an add kernel with one program per element.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel and returns the output.
                out = torch.empty_like(x)
                n_elements = x.numel()
                add_kernel[(n_elements,)](x, y, out, n_elements, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(1, device=self.device),
            torch.randn(1, device=self.device),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_with_none_inputs_and_equal_to_1_arg(self):
        # [FUNCTION DOC] Checks a kernel launched twice with a None argument before arguments
        # equal to one, to test removal of None args followed by index updates for equal-to-one
        # args. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the none-param kernel twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the kernel on x and then on a scaled copy of its
                # output with a None argument and unit stride; returns the second output.
                n_elements = x.size()[0]
                BLOCK_SIZE = 1024
                out1 = torch.empty_like(x)
                out2 = torch.empty_like(x)
                # Run the same kernel multiple times to test the optimization
                # of removing None arguments and then update the indices of
                # equal_to_1 arguments. The None arguments need to be before
                # the equal_to_1 arguments
                add_kernel_with_none_param_and_equal_to_1_arg[(1,)](
                    x,
                    None,
                    out1,
                    n_elements,
                    x.stride(0),  # equal to 1
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                add_kernel_with_none_param_and_equal_to_1_arg[(1,)](
                    2.71 * out1,
                    None,
                    out2,
                    n_elements,
                    x.stride(0),  # equal to 1
                    ARGS_PASSED="one",
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return out2

        example_inputs = (torch.randn(1023, device=self.device),)
        self.check_model(Model(), example_inputs)

    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Checks a scaled add kernel where the float
        # scaling argument evaluates to exactly one. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching a scaled add kernel with scaling factor one
            # computed from the element count.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes the float factor as the element count to the
                # power zero over one, launches the kernel and returns the output.
                out = torch.empty_like(x)
                n_elements = x.numel()
                scaling_factor = (n_elements**0) / 1.0
                add_kernel_with_scaling[(n_elements,)](
                    x,
                    y,
                    out,
                    n_elements,
                    scaling_factor,
                    BLOCK_SIZE=16,
                )
                return out

        dynamic_shapes = None
        if dynamic:
            dim0_xy = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "x": {0: dim0_xy},
                "y": {0: dim0_xy},
            }
        example_inputs = (
            torch.randn(2, device=self.device),
            torch.randn(2, device=self.device),
        )
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @common_utils.parametrize("threshold", [float("inf"), float("-inf"), float("nan")])
    def test_triton_kernel_inf_float_arg(self, threshold):
        # [FUNCTION DOC] Parametrized on positive infinity, negative infinity and NaN thresholds.
        # Checks a clamp kernel taking the threshold as a float scalar argument stored on the
        # module. GPU only, not MPS.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Triton kernel: for each block, loads inputs within bounds,
        # replaces values above the threshold with the threshold, and stores the result.
        @triton.jit
        def clamp_kernel(
            in_ptr, out_ptr, threshold, n_elements, BLOCK_SIZE: "tl.constexpr"
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            out = tl.where(x > threshold, threshold, x)
            tl.store(out_ptr + offsets, out, mask=mask)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a float threshold and applying the clamp kernel.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Stores the threshold.
                super().__init__()
                self.t = t

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Launches the clamp kernel with the stored threshold and
                # returns the output.
                out = torch.empty_like(x)
                n = x.numel()
                clamp_kernel[(n,)](x, out, self.t, n, BLOCK_SIZE=16)
                return out

        example_inputs = (torch.randn(16, device=self.device),)
        self.check_model(Model(threshold), example_inputs)

    def test_triton_kernel_weird_param_order(self):
        # [FUNCTION DOC] Checks a triton kernel whose parameters are declared in an unusual order.
        # GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the autotuned kernel with unusual
            # parameter order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Launches the kernel with keyword arguments and returns
                # the output.
                out = torch.empty_like(x)
                add_kernel_autotuned_weird_param_order[16,](
                    in_ptr0=x,
                    in_ptr1=x,
                    n_elements=x.numel(),
                    out_ptr=out,
                )
                return out

        x = torch.randn(16, 16, device=self.device)
        self.check_model(Model(), (x,))

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_triton_kernel_dynamic_grid(self):
        # [FUNCTION DOC] Checks an autotuned add kernel whose launch grid is computed from an
        # unbacked scalar item. GPU only; skipped on Windows XPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        import math

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching an autotuned add kernel with a grid depending on
            # an unbacked scalar.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y, n_elements_tensor):
                # [FUNCTION DOC] HELPER: Reads the scalar, defines the grid function, launches
                # the kernel (see the grid function and the remaining code below) and returns
                # the output.
                output = torch.zeros_like(x)
                n_elements_symint = n_elements_tensor.item()
                n_elements = x.numel()

                def grid(meta):
                    # [FUNCTION DOC] HELPER: Grid function: divides the scalar by one as float,
                    # divides by the block size from the autotune meta and truncates, returning a
                    # one-element grid.
                    n_elements_complicated = n_elements_symint // 1.0
                    return (math.trunc(n_elements_complicated / meta["BLOCK_SIZE"]),)

                add_kernel_autotuned[grid](
                    x,
                    y,
                    output,
                    n_elements,
                )

                return output

        x = torch.randn(128, device=self.device)
        y = torch.randn(128, device=self.device)
        n_elem = torch.tensor(128)
        dim0_x = Dim("dim0_x", min=8, max=256)
        dim0_y = Dim("dim0_y", min=8, max=256)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}, "n_elements_tensor": {}}
        self.check_model(Model(), (x, y, n_elem), dynamic_shapes=dynamic_shapes)

    def test_shifted_constraint_ranges(self):
        # [FUNCTION DOC] Checks a model with a runtime check that the second input's first dim is
        # one more than the first input's, with a dynamic dim for x and its shift by one for y.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module checking the size relation and returning the sum of both
            # inputs reduced over the first dim.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Asserts y has one more row than x, and returns the sum
                # over dim 0 of x plus that of y.
                torch._check(y.size(0) == x.size(0) + 1)
                return x.sum(0) + y.sum(0)

        a = torch.randn((4, 5), device=self.device)
        b = torch.randn((5, 5), device=self.device)
        dim0_x = Dim("dim0_x", min=2, max=1024)
        dim0_y = dim0_x + 1
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(
            Model(),
            (a, b),
            dynamic_shapes=dynamic_shapes,
        )

    def test_scatter_fallback(self):
        # [FUNCTION DOC] Checks the scatter op on int64 inputs of mismatched shapes, which goes
        # through a fallback kernel.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying scatter along dim 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
                src: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Returns the scatter of src into inp along dim 1 at index.
                return torch.scatter(inp, 1, index, src)

        inputs = (
            torch.ones((3, 5), device=self.device, dtype=torch.int64),
            torch.tensor([[0, 1, 2, 0]], device=self.device, dtype=torch.int64),
            torch.zeros((2, 5), device=self.device, dtype=torch.int64),
        )

        self.check_model(Model(), inputs)

    def test_scatter_reduce_fallback(self):
        # [FUNCTION DOC] Checks the scatter-reduce op with sum reduction on int64 inputs,
        # via fallback.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying scatter-reduce with sum along dim 0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
                src: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Returns the sum scatter-reduce of src into inp along
                # dim 0.
                return torch.scatter_reduce(inp, 0, index, src, reduce="sum")

        inputs = (
            torch.tensor([1, 10, 100, 1000], device=self.device, dtype=torch.int64),
            torch.tensor([0, 1, 0, 1, 2, 1], device=self.device, dtype=torch.int64),
            torch.tensor([1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.int64),
        )

        self.check_model(Model(), inputs)

    def test_index_put_fallback(self):
        # [FUNCTION DOC] Under deterministic algorithms mode (index_put falls back there), checks
        # an accumulating index-put with a boolean index tensor on int64 inputs.
        # index_put falls back in the deterministic mode
        with DeterministicGuard(True):

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying an accumulating index-put.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Only calls the base initializer.
                    super().__init__()

                def forward(
                    self,
                    self_tensor: torch.Tensor,
                    indices: tuple[torch.Tensor],
                    values: torch.Tensor,
                ):
                    # [FUNCTION DOC] HELPER: Returns the accumulating index-put of values into
                    # self_tensor at indices.
                    return torch.index_put(
                        self_tensor, indices, values, accumulate=True
                    )

            inputs = (
                torch.ones(4, device=self.device, dtype=torch.int64),
                (torch.tensor([1, 1, 2, 2], device=self.device, dtype=torch.bool),),
                torch.ones(4, device=self.device, dtype=torch.int64),
            )

            self.check_model(Model(), inputs)

    def test_scatter_fallback_error_code_check(self):
        # [FUNCTION DOC] Compiles a scatter model and, on GPU, asserts via a regular expression
        # that the generated code wraps the scatter shim call in the AOTI error code check (so
        # shim failures are not swallowed). On CPU nothing is asserted, since scatter is lowered
        # to C++ there.
        # On GPU, scatter falls back to the aoti_torch_*_scatter c-shim; that
        # call must be wrapped in AOTI_TORCH_ERROR_CODE_CHECK, otherwise a shim
        # failure (e.g. setStorage on an empty self) is silently swallowed and
        # the model returns a wrong result instead of raising. (On CPU, scatter
        # is lowered to a C++ kernel, so there is no shim call to check.)
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying scatter along dim 1.
            def forward(self, inp, index, src):
                # [FUNCTION DOC] HELPER: Returns the scatter of src into inp along dim 1 at index.
                return torch.scatter(inp, 1, index, src)

        inputs = (
            torch.ones((3, 5), device=self.device, dtype=torch.int64),
            torch.tensor([[0, 1, 2, 0]], device=self.device, dtype=torch.int64),
            torch.zeros((2, 5), device=self.device, dtype=torch.int64),
        )
        _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, Model(), inputs)
        if self.device == GPU_TYPE:
            FileCheck().check_regex(
                r"AOTI_TORCH_ERROR_CODE_CHECK\(aoti_torch_\w+_scatter"
            ).run(code)

    def test_index_put_fallback_error_code_check(self):
        # [FUNCTION DOC] Under deterministic mode, compiles an accumulating index-put model and
        # asserts the generated code wraps the index-put shim call in the AOTI error code check.
        # Same invariant as test_scatter_fallback_error_code_check, for the
        # index_put fallback shim call.
        with DeterministicGuard(True):

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying an accumulating index-put.
                def forward(self, self_tensor, indices, values):
                    # [FUNCTION DOC] HELPER: Returns the accumulating index-put of values into
                    # self_tensor at indices.
                    return torch.index_put(
                        self_tensor, indices, values, accumulate=True
                    )

            inputs = (
                torch.ones(4, device=self.device, dtype=torch.int64),
                (torch.tensor([1, 1, 2, 2], device=self.device, dtype=torch.bool),),
                torch.ones(4, device=self.device, dtype=torch.int64),
            )
            _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, Model(), inputs)
            FileCheck().check_regex(
                r"AOTI_TORCH_ERROR_CODE_CHECK\(aoti_torch_\w*index_put"
            ).run(code)

    def test_narrow_fallback(self):
        # [FUNCTION DOC] Checks the narrow op with non-tensor integer dim, start and length inputs.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying narrow to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, inp: torch.Tensor, dim: int, start: int, length: int):
                # [FUNCTION DOC] HELPER: Returns the narrowed tensor.
                return torch.ops.aten.narrow(inp, dim, start, length)

        inputs = (torch.rand((3, 4), device=self.device), 0, 0, 2)

        self.check_model(Model(), inputs)

    def test_pad_fallback(self):
        # [FUNCTION DOC] Checks the pad op with a tuple of integer padding sizes as a
        # non-tensor input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying padding to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                inp: torch.Tensor,
                pad: tuple[int, ...],
            ):
                # [FUNCTION DOC] HELPER: Returns the padded tensor.
                return torch.ops.aten.pad(inp, pad)

        inputs = (torch.rand((3, 3, 4, 2), device=self.device), (0, 1, 2, 1, 3, 3))

        self.check_model(Model(), inputs)

    def test_fill__fallback(self):
        # [FUNCTION DOC] Checks an in-place fill with a float scalar input, returning the
        # mutated input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling its input in place with a scalar.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, inp: torch.Tensor, scalar: float):
                # [FUNCTION DOC] HELPER: Fills inp with the scalar in place and returns inp.
                torch.ops.aten.fill_(inp, scalar)
                return inp

        inputs = (torch.rand((3, 3, 4, 2), device=self.device), 0.5)
        self.check_model(Model(), inputs)

    @common_utils.parametrize("embed_kernel_binary", [False, True])
    def test_repeated_user_defined_triton_kernel(self, embed_kernel_binary):
        # [FUNCTION DOC] Parametrized on embed_kernel_binary. Calls an in-place triton kernel three
        # times in a model, checks numerics, and checks the generated code launches a kernel (and,
        # when the binary is embedded, does not name a binary file in the launch). GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that runs an in-place doubling kernel three times and
            # returns x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Launches the kernel in a loop of three iterations on x and
                # returns x.
                for _ in range(3):
                    mul2_inplace_kernel[4,](x, n_elements=4, BLOCK_SIZE=16)
                return x

        inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.embed_kernel_binary": embed_kernel_binary}):
            model = Model()
            self.check_model(model, inputs)
            _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, model, inputs)
            FileCheck().check("launchKernel(").run(code)
            if config.aot_inductor.embed_kernel_binary:
                # Not expect to see launchKernel("CUBIN_FILE_NAME"
                FileCheck().check_not('launchKernel("').run(code)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_convolution(self):
        # [FUNCTION DOC] Checks a transposed strided convolution under max-autotune with the Triton
        # backend. Skipped on CPU and non big GPUs.
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the convolution op with stride 4, transposed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, w, b):
                # [FUNCTION DOC] HELPER: Returns the transposed convolution of x with weight w and
                # bias b.
                return torch.ops.aten.convolution(x, w, b, [4], [0], [1], True, [0], 1)

        example_inputs = (
            torch.randn([2, 32, 90], device=self.device),
            torch.randn([32, 16, 8], device=self.device),
            torch.randn([16], device=self.device),
        )
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            self.check_model(Model(), example_inputs)

    def test_zero_size_weight(self):
        # [FUNCTION DOC] Checks a squeeze-and-excite style block with 4 channels where the reduced
        # channel count is zero-sized.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with adaptive average pool and a small bottleneck network
            # ending in a sigmoid gate multiplied with the input.
            def __init__(self, channel, r=8):
                # [FUNCTION DOC] HELPER: Builds the pool and a sequential of linear layers (channel
                # to channel divided by r, relu, back to channel, sigmoid) without bias.
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(channel, channel // r, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(channel // r, channel, bias=False),
                    torch.nn.Sigmoid(),
                )

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Pools to a per-channel vector, passes through the network,
                # reshapes as a gate and multiplies the input.
                b, c, _, _ = inp.shape
                x = self.pool(inp).view(b, c)
                x = self.net(x).view(b, c, 1, 1)
                x = inp * x
                return x

        inputs = (torch.rand(4, 4, 4, 4, device=self.device),)
        self.check_model(Model(4), inputs)

    def test_zero_size_buffer(self):
        # [FUNCTION DOC] Checks a model returning x plus one and a zero-size buffer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty buffer returned as an output.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Registers an empty buffer on the device.
                super().__init__()
                self.foo = torch.nn.Buffer(torch.zeros((0, 0), device=device))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one and the buffer.
                return x + 1, self.foo

        example_inputs = (torch.rand(4, 4, device=self.device),)
        self.check_model(Model(self.device), example_inputs)

    def test_no_args(self):
        # [FUNCTION DOC] Checks a model with no inputs that multiplies two parameters.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters multiplied in forward.
            def __init__(self, m, n):
                # [FUNCTION DOC] HELPER: Creates two random parameters of the given shape.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(m, n),
                )
                self.alpha = torch.nn.Parameter(torch.randn(m, n))

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the elementwise product of the two parameters.
                return self.weight * self.alpha

        self.check_model(Model(6, 4), ())

    def test_dynamic_scalar(self):
        # [FUNCTION DOC] Checks a model that computes a cross entropy loss and also returns a
        # statistics dictionary containing the Python scalar of the loss.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a summed cross entropy loss and a dictionary
            # with its scalar value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the cross entropy loss with no reduction.
                super().__init__()
                self.criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")

            def forward(self, inputs, targets, split_index=None):
                # [FUNCTION DOC] HELPER: Computes the summed loss, stores its Python value in a
                # dictionary and returns both. The split_index argument is unused.
                statistics = {}
                total_loss = self.criterion_ce(inputs, targets).sum()
                statistics["dl"] = total_loss.item()
                return total_loss, statistics

        inputs = (
            torch.rand(4, 4, 4, 4, device=self.device),
            torch.rand(4, 4, 4, 4, device=self.device),
        )
        self.check_model(Model(), inputs)

    def test_symint_item(self):
        # [FUNCTION DOC] Checks returning a Python integer from a one-element int tensor.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the scalar value of its input.
            def forward(self, tensor):
                # [FUNCTION DOC] HELPER: Returns the Python scalar of the tensor.
                return tensor.item()

        inputs = (torch.tensor([1], dtype=torch.int, device=self.device),)
        self.check_model(Model(), inputs)

    def test_symbool_item(self):
        # [FUNCTION DOC] Checks returning a Python boolean from a one-element bool tensor.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the scalar value of its input.
            def forward(self, tensor):
                # [FUNCTION DOC] HELPER: Returns the Python scalar of the tensor.
                return tensor.item()

        inputs = (torch.tensor([0], dtype=torch.bool, device=self.device),)
        self.check_model(Model(), inputs)

    def test_symfloat_item(self):
        # [FUNCTION DOC] Checks returning a Python float from a one-element float tensor.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the scalar value of its input.
            def forward(self, tensor):
                # [FUNCTION DOC] HELPER: Returns the Python scalar of the tensor.
                return tensor.item()

        inputs = (torch.tensor([3.14], dtype=torch.float, device=self.device),)
        self.check_model(Model(), inputs)

    def test_constant_original_fqn_and_dtype(self):
        # [FUNCTION DOC] Compiles a nested module with a parameter named zero and other parameters
        # and buffers, then asserts the runner's constant-name-to-original-FQN mapping and
        # constant-name-to- dtype mapping match expected dictionaries (all dtype code 6, float32).
        class FooBarModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a parameter named "0", a buffer and a parameter,
            # combined arithmetically with the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the parameter named zero, a buffer and
                # another parameter.
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the buffer to x, multiplies by the zero-named
                # parameter, and divides by the other parameter.
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Top-level module containing the submodule plus its own parameter
            # and buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule, a parameter and a buffer.
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the submodule, adds the parameter and multiplies
                # by the buffer.
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        with torch.no_grad():
            so_path = AOTIRunnerUtil.legacy_compile(
                model=TestModule().to(device=self.device),
                example_inputs=(torch.rand(3, 4, device=self.device),),
            )
        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        expected_original_fqns = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        expected_dtypes = {
            "L__self___test_param": 6,
            "L__self___test_buf": 6,
            "L__self___foo_bar_0": 6,
            "L__self___foo_bar_test_param": 6,
            "L__self___foo_bar_test_buf": 6,
        }
        self.assertEqual(expected_dtypes, runner.get_constant_names_to_dtypes())

    def test_masked_select_dynamic(self):
        # [FUNCTION DOC] Checks masked select (data-dependent output size) with bounded dynamic
        # first and second dims.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting the elements of x that are at least 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Builds the mask and returns the masked selection.
                mask = x.ge(0.5)
                return torch.masked_select(x, mask)

        example_args = (torch.randn(3, 4, 5, device=self.device),)
        dim0_x_max, dim1_x_max = 100, 7
        dynamic_shapes = {
            "x": {
                0: Dim("dim0_x", max=dim0_x_max),
                1: Dim("dim1_x_max", max=dim1_x_max),
            }
        }
        m = M()
        self.check_model(m, example_args, dynamic_shapes=dynamic_shapes)

    def test_grid_sampler_3d(self):
        # [FUNCTION DOC] Checks the 3D grid sampler op with fixed interpolation settings on 5D
        # input and a 5D grid.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the 3D grid sampler.
            def forward(self, input, grid):
                # [FUNCTION DOC] HELPER: Returns grid_sampler_3d of input with grid, mode zero,
                # padding zero and align corners on.
                return torch.grid_sampler_3d(input, grid, 0, 0, True)

        # input: (N, C, D_in, H_in, W_in), grid: (N, D_out, H_out, W_out, 3)
        example_args = (
            torch.randn(1, 1, 4, 4, 4, device=self.device),
            torch.randn(1, 2, 2, 2, 3, device=self.device),
        )
        self.check_model(M(), example_args)

    def test_proxy_executor_permute(self):
        # [FUNCTION DOC] Checks permute of a complex tensor (exercises the proxy executor path).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module permuting dims one and two of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x with dims reordered to first, third, second.
                return torch.ops.aten.permute.default(x, [0, 2, 1])

        example_args = (torch.randn((1, 3001, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_abs(self):
        # [FUNCTION DOC] Checks absolute value of a complex tensor (exercises the proxy
        # executor path).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying abs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns abs of x.
                return torch.ops.aten.abs.default(x)

        example_args = (torch.randn((1, 3001, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_squeeze(self):
        # [FUNCTION DOC] Checks squeeze of dim zero of a complex tensor (proxy executor path).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squeezing dim 0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x squeezed along dim 0.
                return torch.ops.aten.squeeze.dim(x, 0)

        example_args = (torch.randn((1, 300, 201), dtype=torch.complex64),)
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_hann(self):
        # [FUNCTION DOC] Checks a model with no inputs that creates a Hann window of 400 samples.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a Hann window.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a Hann window of length 400.
                return torch.ops.aten.hann_window.default(400)

        example_args = ()
        m = M()
        self.check_model(m, example_args)

    def test_proxy_executor_error_message_preserved(self):
        # [FUNCTION DOC] Registers a custom op that raises a runtime error on NaN input, with a
        # fake implementation and fallback lowering. Compiles a model using it, checks normal
        # output, checks that a NaN input raises an error with the original message, and that the
        # loaded module still works afterwards.
        @torch.library.custom_op("aoti_test::validate_input", mutates_args=())
        def validate_input(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body: raises RuntimeError if x contains NaN, else
            # returns a clone of x.
            if x.isnan().any():
                raise RuntimeError("NaN detected in input tensor")
            return x.clone()

        @validate_input.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake (meta) implementation of the custom op: returns a clone
            # of x.
            return x.clone()

        from torch._inductor.lowering import make_fallback

        make_fallback(torch.ops.aoti_test.validate_input.default, warn=False)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that applies the custom op and doubles the result.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the custom op and multiplies by two.
                x = torch.ops.aoti_test.validate_input(x)
                return x * 2

        m = M()
        sample = torch.ones(3, device=self.device)
        so_path = AOTIRunnerUtil.compile(m, (sample,))
        aoti_module = torch._inductor.aoti_load_package(so_path)
        result = aoti_module(sample)
        self.assertTrue(torch.allclose(result, sample * 2))
        nan_input = torch.tensor([1.0, float("nan"), 3.0], device=self.device)
        with self.assertRaisesRegex(RuntimeError, "NaN detected in input tensor"):
            aoti_module(nan_input)
        result2 = aoti_module(sample)
        self.assertTrue(torch.allclose(result2, sample * 2))

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @config.patch({"size_asserts": True, "fx_graph_cache": False})
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_aoti_custom_op_bad_fake_dtype_fails_fast(self):
        # [FUNCTION DOC] Registers a custom op in a unique namespace whose fake implementation
        # returns bfloat16 while the real op returns the input dtype, with size asserts and runtime
        # input checks on. Checks the generated code contains the stride and dtype check for
        # bfloat16, and that running raises a RuntimeError describing the dtype mismatch from an
        # incorrect fake. Skipped on MPS and in fbcode.
        if self.device == "mps":
            raise unittest.SkipTest("bfloat16 custom op fallback not covered on MPS")

        namespace = f"aoti_test_bad_fake_dtype_{uuid.uuid4().hex}"

        @torch.library.custom_op(f"{namespace}::bad_meta_mul", mutates_args=())
        def bad_meta_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body: returns a times b.
            return a * b

        @bad_meta_mul.register_fake
        def _(a, b):
            # [FUNCTION DOC] HELPER: Wrong fake implementation: returns an empty tensor of bfloat16
            # dtype with the shape of a.
            return torch.empty_like(a, dtype=torch.bfloat16)

        from torch._inductor.lowering import make_fallback

        op = getattr(torch.ops, namespace).bad_meta_mul
        make_fallback(op.default, warn=False)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom op and adding one.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Calls the op and returns the result plus one.
                x = op(a, b)
                return x + 1

        sample = (
            torch.randn(16, device=self.device, dtype=torch.float32),
            torch.randn(16, device=self.device, dtype=torch.float32),
        )
        package_path, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, M(), sample)
        FileCheck().check("aoti_torch_proxy_executor_call_function").check(
            "if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride("
        ).check('"torch.bfloat16"').run(code)

        aoti_module = torch._inductor.aoti_load_package(package_path)
        with self.assertRaisesRegex(
            RuntimeError,
            r"(?s)expected dtype torch\.bfloat16 but got dtype code .*incorrect fake",
        ):
            aoti_module(*sample)

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_check_error_message_preserved(self):
        # [FUNCTION DOC] With runtime input checks enabled, compiles a plus-one model for float32,
        # runs it, then runs with float16 and asserts a RuntimeError containing the unmatched dtype
        # message (so the error from the C boundary reaches Python). Skipped in fbcode.
        # Exception thrown from CONVERT_EXCEPTION_TO_ERROR_CODE at the outer
        # AOTI C ABI boundary (i.e. not from an aoti_torch_* shim) must reach
        # Python via AOTInductorGetLastError, not be flattened to the generic
        # "run_func_(...) API call failed" message.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        m = M()
        good = torch.randn(4, dtype=torch.float32, device=self.device)
        bad_dtype = good.to(torch.float16)
        so_path = AOTIRunnerUtil.compile(m, (good,))
        aoti_module = torch._inductor.aoti_load_package(so_path)
        aoti_module(good)
        with self.assertRaisesRegex(RuntimeError, "unmatched dtype"):
            aoti_module(bad_dtype)

    def test_fqn(self):
        # [FUNCTION DOC] Checks a nested module hierarchy with parameters and buffers at several
        # levels (root, children and a grandchild) combined in a sequence of arithmetic operations.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Grandchild module with a buffer of threes that divides the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the buffer.
                super().__init__()
                self.nestedchild3buffer = torch.nn.Buffer(torch.ones(2, 3) * 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x divided by the buffer.
                return x / self.nestedchild3buffer

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the grandchild and a parameter of ones, adding the
            # parameter after applying the grandchild.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested child and registers the parameter.
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the nested child then adds the parameter.
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of twos subtracted from the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the buffer.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3) * 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus the buffer.
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module with two children and a root parameter of fours.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children and the root parameter.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3) * 4)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scales x by the root parameter then applies the first and
                # second children.
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        self.check_model(MyModule(), (torch.randn(2, 3, device=self.device),))

    def test_model_modified_weights(self):
        # [FUNCTION DOC] Checks a linear model, then increments the weight in place and checks it
        # again, so compiled weights stored in the library must be regenerated.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight of shape n by k and bias of
                # n on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a with the weight and bias.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 16
        N = 10
        K = 128
        example_inputs = (torch.randn(2, M, K, device=self.device),)
        model = Model(N, K, self.device)
        self.check_model(model, example_inputs)

        # Update model weights, after this AOTInductor should re-generate model.so
        # if weights are stored in the model.so
        model.weight += 1
        self.check_model(model, example_inputs)

    @skipIfWindowsXPU(msg="crash on Windows XPU.")
    def test_triton_kernel_extern_kernel_arg(self):
        # [FUNCTION DOC] Checks a triton kernel taking the output of a matrix multiply (an extern
        # kernel output) as an argument. GPU only; skipped on Windows XPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module passing the result of a matmul to an add kernel.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel with x, the matmul of x and y, and
                # a zeros output; returns the output.
                out = torch.zeros_like(x)
                # torch.mm is ExternKernelOut
                add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_multi_output_arg(self):
        # [FUNCTION DOC] Checks a triton kernel taking the sorted values (a multi-output fallback
        # result) as an argument. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module passing the sorted values of y to an add kernel.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel with x, the sorted values of y and
                # a zeros output; returns the output.
                out = torch.zeros_like(x)
                # torch.sort creates fallback kernel and hence MultiOutput
                add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        )

        self.check_model(Model(), example_inputs)

    def test_triton_kernel_reinterpret_view_mem_leak(self):
        # [FUNCTION DOC] Compiles a model passing a reshaped (reinterpreted view) intermediate into
        # a triton kernel, loads the package and asserts device allocated memory is unchanged
        # across two runs (no leak), then compares output with eager. GPU only.
        # Check for memory leak when using user-defined Triton Kernel + AOTI.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squaring y, reshaping it to x's shape, and passing it to
            # an add kernel.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes y squared, launches the add kernel with x and the
                # reshaped value, and returns the output.
                out = torch.zeros_like(x)
                yy = y * y
                # reshape creates a ReinterpretView
                add_kernel[(4,)](x, yy.reshape_as(x), out, 4, 16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(1, 16, device=GPU_TYPE),
        )

        package_path: str = AOTIRunnerUtil.compile(
            Model(),
            example_inputs,
        )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        # Don't assign outputs to a variable b/c it will allocate GPU memory.
        device_interface = get_interface_for_device(GPU_TYPE)
        device: int = device_interface.current_device()
        mem_before = device_interface.memory_allocated(device)
        aot_inductor_module(*example_inputs)
        aot_inductor_module(*example_inputs)
        mem_after = device_interface.memory_allocated(device)
        self.assertEqual(mem_before, mem_after)

        actual = aot_inductor_module(*example_inputs)
        expected = Model()(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotuning", [False, True])
    def test_triton_kernel_unbacked_symint_in_grid(self, dynamic, autotuning):
        # [FUNCTION DOC] Parametrized on dynamic and autotuning; scalar outputs are captured.
        # Checks a triton kernel whose grid depends on an unbacked integer item, with an autotuned
        # or fixed kernel. GPU only, not MPS.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching an add kernel with a grid from an
            # unbacked scalar.
            def forward(self, x, y, n_elements_tensor):
                # [FUNCTION DOC] HELPER: Reads the scalar from the tensor input, defines the
                # grid, launches the autotuned or fixed kernel per the test parameter and returns
                # the output.
                output = torch.zeros_like(x)
                n_elements_symint = n_elements_tensor.item()
                n_elements = x.numel()

                def grid(meta):
                    # [FUNCTION DOC] HELPER: Grid function: returns a one-element grid sized to
                    # cover the unbacked scalar with the block size from the meta mapping.
                    return (triton.cdiv(n_elements_symint, meta["BLOCK_SIZE"]),)

                if autotuning:
                    add_kernel_autotuned[grid](
                        x,
                        y,
                        output,
                        n_elements,
                    )
                else:
                    add_kernel[grid](
                        x,
                        y,
                        output,
                        n_elements,
                        BLOCK_SIZE=16,
                    )

                return output

        example_inputs = (
            torch.randn(123, device=GPU_TYPE),
            torch.randn(123, device=GPU_TYPE),
            torch.tensor(123),
        )

        dynamic_shapes = None
        if dynamic:
            dim0 = Dim("s0", min=2, max=1024)
            dynamic_shapes = {
                "x": {0: dim0},
                "y": {0: dim0},
                "n_elements_tensor": {},
            }

        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Some archs don't support mem eff SDPA"
    )
    def test_scaled_dot_product_efficient_attention(self):
        # [FUNCTION DOC] Checks the memory efficient attention op with an attention bias, returning
        # the first output. GPU only; skipped without mem-efficient attention support.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the first output of efficient attention.
            def forward(self, q, k, v, attn_bias):
                # [FUNCTION DOC] HELPER: Calls the efficient attention op with the bias and no
                # dropout flag and returns element zero.
                return torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, attn_bias, False
                )[0]

        example_inputs = (
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
            torch.randn(4, 4, 36, 36, device=GPU_TYPE),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(not SM90OrLater, "FA3 requires SM90+")
    def test_varlen_attn_paged_kv_cache(self):
        # [FUNCTION DOC] Checks variable length attention with a paged key-value cache and
        # sequence-used lengths, using the FA3 backend, two sequences with one new token each and
        # cached length 32. Skipped before SM90 or without FA3.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        from torch.nn.attention.varlen import varlen_attn

        batch_size, num_heads, head_dim = 2, 4, 64
        page_size, max_pages_per_seq = 64, 2
        cache_seqlen, new_seqlen = 32, 1

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling variable-length attention with a paged cache.
            def forward(self, q, k_pages, v_pages, cu_seq_q, seqused_k, page_table):
                # [FUNCTION DOC] HELPER: Calls variable-length attention with the query, paged keys
                # and values, query cumulative lengths, used key lengths and page table, using
                # maximum lengths from the enclosing test constants, and returns the result.
                return varlen_attn(
                    query=q,
                    key=k_pages,
                    value=v_pages,
                    cu_seq_q=cu_seq_q,
                    cu_seq_k=None,
                    max_q=new_seqlen,
                    max_k=max_pages_per_seq * page_size,
                    seqused_k=seqused_k,
                    page_table=page_table,
                )

        total_new = new_seqlen * batch_size
        total_pages = batch_size * max_pages_per_seq
        example_inputs = (
            torch.randn(
                total_new, num_heads, head_dim, dtype=torch.bfloat16, device=self.device
            ),
            torch.randn(
                total_pages,
                page_size,
                num_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.randn(
                total_pages,
                page_size,
                num_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.arange(
                0, total_new + 1, new_seqlen, dtype=torch.int32, device=self.device
            ),
            torch.full(
                (batch_size,), cache_seqlen, dtype=torch.int32, device=self.device
            ),
            torch.arange(total_pages, dtype=torch.int32, device=self.device).reshape(
                batch_size, max_pages_per_seq
            ),
        )

        with use_fa3():
            self.check_model(Model(), example_inputs)

    def test_aoti_runtime_asserts(self):
        # [FUNCTION DOC] Defines a custom op that slices by a data-dependent amount with a fake
        # that returns an unbacked size, draft-exports a model that asserts bounds on the result
        # size, compiles and packages it (patching the fake kernel generation flag if the export
        # report shows a mismatched fake kernel), compares with eager, and asserts running with a
        # violating input raises an error with the expected-but-received message.
        from torch.export._draft_export import draft_export, FailureType

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: CPU implementation: returns the leading slice of a whose
                # length is the scalar value of b.
                return a[: b.item()]

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_fake_impl(a, b):
                # [FUNCTION DOC] HELPER: Fake implementation: returns an empty tensor with a new
                # dynamic (unbacked) size.
                ctx = torch.library.get_ctx()
                u = ctx.new_dynamic_size()
                return torch.empty(u)

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the custom op, checking the result length is
                # between 4 and less than the input length, and returning one element of a.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Calls the op, adds two runtime checks on the length
                    # and returns the element at the length minus three.
                    res = torch.ops.mylib.foo(a, b)
                    s = res.shape[0]
                    torch._check(s > 3)
                    torch._check(s < a.shape[0])
                    return a[s - 3]

            example_inputs = (torch.randn(100), torch.tensor(10))
            ep = draft_export(M(), example_inputs)
            report = ep._report
            need_config_patch = any(
                not f.xfail and f.failure_type == FailureType.MISMATCHED_FAKE_KERNEL
                for f in report.failures
            )
            m = ep.module()

            # This should no longer be needed after #150093
            from torch._functorch import config as functorch_config

            with functorch_config.patch(
                {"generate_fake_kernels_from_real_mismatches": need_config_patch}
            ):
                pt2_file = torch._inductor.aoti_compile_and_package(ep)
            optimized = torch._inductor.aoti_load_package(pt2_file)

            self.assertTrue(same(optimized(*example_inputs), m(*example_inputs)))

            with self.assertRaisesRegex(Exception, "Expected .* but received"):
                optimized(torch.randn(100), torch.tensor(2))

    @patch.dict(os.environ, {"TORCHINDUCTOR_SCALAR_ASSERTS_FULL": "1"})
    def test_aoti_runtime_asserts_backed_symint(self):
        # [FUNCTION DOC] With scalar assert full mode, checks a reshape with a dynamic dimension:
        # the compiled model works for the compile-time input and raises an error when the size is
        # not a multiple of 100. Skipped unless full runtime asserts are on.
        if not full_aoti_runtime_assert():
            raise unittest.SkipTest("full runtime assert not turned on")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping to 100 rows, cloning and adding one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reshapes x into 100 rows, clones and adds one.
                y = x.reshape(100, -1).clone()
                y = y + 1
                return y

        model = Model().to(self.device)
        input1 = (torch.rand(100, device=self.device),)
        input2 = (torch.rand(2099, device=self.device),)
        dynamic_shapes = {
            "x": {0: torch.export.Dim.DYNAMIC},
        }
        package_path = AOTIRunnerUtil.compile(
            model,
            input1,
            dynamic_shapes=dynamic_shapes,
        )
        optimized = torch._inductor.aoti_load_package(package_path)
        self.assertEqual(model(*input1), optimized(*input1))
        with self.assertRaisesRegex(Exception, "Expected .* but received"):
            optimized(*input2)

    @skipIfWindows(msg="TODO: (xuhancn) confirm, Crash: access violation")
    def test_index_put_with_none_index(self):
        # [FUNCTION DOC] Under deterministic mode, checks an accumulating index-put with None
        # entries in the indices tuple. Skipped on Windows.
        # index_put falls back in the deterministic mode
        with DeterministicGuard(True):

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying an accumulating index-put with leading None
                # indices and two index tensors.
                def forward(self, x, i1, i2, y):
                    # [FUNCTION DOC] HELPER: Returns the accumulating index-put into x with indices
                    # None, None, i1 and the transposed i2, with values y.
                    return torch.ops.aten.index_put(
                        x,
                        (None, None, i1, i2.transpose(0, 1)),
                        y,
                        accumulate=True,
                    )

            example_inputs = (
                torch.rand(8, 192, 30, 30, device=self.device),
                torch.zeros(3, 14, 1, 1, dtype=torch.int64, device=self.device),
                torch.ones(14, 3, dtype=torch.int64, device=self.device),
                torch.randn(8, 192, 3, 14, 3, 14, device=self.device),
            )
            self.check_model(Model(), example_inputs)

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks(self):
        # [FUNCTION DOC] With runtime input checking enabled, compiles a model taking a dictionary
        # of tensors of many dtypes (dtype set depends on the platform: float64 unless MPS,
        # bfloat16 on SM80+) with dynamic dims on various axes. Reads the generated C++ source and
        # asserts exact counts of the dtype, dim value, dim bound and stride mismatch check
        # messages, with counts depending on the number of dynamic dims for the platform (10, 8 or
        # 9 dynamic dims). Then runs the check_model comparison.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the list of values of the input dictionary.
            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the list of the dictionary values.
                return list(inputs.values())

        inputs = {}
        dtypes = [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]

        if not TEST_MPS:
            dtypes.append(torch.float64)
        if SM80OrLater:
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            inputs[f"x_{str(dtype)}"] = torch.ones(
                4, 8, 10, dtype=dtype, device=self.device
            )

        dim0 = Dim("s0", min=2, max=1024)
        dim1 = Dim("s1", min=2, max=512)
        dim2 = Dim("s2", min=2, max=128)
        dynamic_shapes = {
            "x_torch.float16": {0: dim0},
            "x_torch.float32": {0: dim0},
            "x_torch.bool": {1: dim1},
            "x_torch.int8": {1: dim1},
            "x_torch.int16": {},
            "x_torch.int32": {2: dim2},
            "x_torch.int64": {2: dim2},
            "x_torch.uint8": {2: dim2},
        }
        if not TEST_MPS:
            dynamic_shapes["x_torch.float64"] = {0: dim0}
        if SM80OrLater:
            dynamic_shapes["x_torch.bfloat16"] = {1: dim1}

        m = Model()
        inputs = (inputs,)
        dynamic_shapes = (dynamic_shapes,)
        with torch.no_grad():
            so_path = AOTIRunnerUtil.legacy_compile(
                m, inputs, dynamic_shapes=dynamic_shapes
            )

        # Expected results for the following checks:
        # ("unmatched dtype", "unmatched dim value at", "dim value is too", "unmatched stride value at")
        if SM80OrLater:
            # 10 dynamic dims
            expected_results = (10, 21, 18, 21)
        elif TEST_MPS:
            # 8 dynamic dims
            expected_results = (8, 17, 14, 16)
        else:
            # 9 dynamic dims
            expected_results = (9, 19, 16, 19)

        with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                "unmatched dtype",
                expected_results[0],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched dim value at",
                expected_results[1],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "dim value is too",
                expected_results[2],
                exactly=True,
            ).run(src_code)
            FileCheck().check_count(
                "unmatched stride value at",
                expected_results[3],
                exactly=True,
            ).run(src_code)

        self.check_model(m, inputs)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_assert_size_stride_guarded_by_env_check(self):
        # [FUNCTION DOC] Compiles a QR model and asserts the generated code guards the size and
        # stride asserts behind the runtime environment variable check, so they fire only when
        # AOTI_RUNTIME_CHECK_INPUTS is set. Skipped without FP8 platform support.
        # Verify that assert_size_stride calls in AOTI-generated code are
        # guarded by _check_aoti_runtime_check_inputs_env() so they only
        # fire when AOTI_RUNTIME_CHECK_INPUTS is set.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the first output of a QR decomposition.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the Q factor of x.
                return torch.linalg.qr(x)[0]

        example_inputs = (torch.randn(4, 4, device=self.device),)
        _, code = run_and_get_cpp_code(AOTIRunnerUtil.compile, Model(), example_inputs)
        FileCheck().check(
            "if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride("
        ).run(code)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_fp8(self):
        # [FUNCTION DOC] With runtime checks enabled, compiles and checks a model summing two
        # float8 inputs (e4m3 and e5m2 variants) converted to float, with a dynamic first dim. CUDA
        # only: other devices return early without testing; skipped without FP8 support.
        # cuda only
        if self.device != "cuda":
            return

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing two inputs after converting both to float.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x0, x1):
                # [FUNCTION DOC] HELPER: Returns x0 plus x1 after converting both to float.
                t = x0.to(torch.float) + x1.to(torch.float)
                return t

        inputs = []
        for dtype in (
            e4m3_type,  # float8_e4m3fn (CUDA) or float8_e4m3fnuz (ROCm)
            e5m2_type,  # float8_e5m2 (CUDA) or float8_e5m2fnuz (ROCm)
        ):
            inputs.append(torch.ones(8, 8, 8, dtype=dtype, device=self.device))
        dim0 = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {0: dim0},
        }
        with torch.no_grad():
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    @skipIfXpu(msg="Total size of kernel arguments exceeds driver limit on XPU")
    def test_runtime_checks_large(self):
        # [FUNCTION DOC] Compile-only check with a thousand tensor inputs summed in a loop, to
        # exercise generated runtime checks for a very large number of arguments. Skipped on XPU
        # (kernel argument limit).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing a variable number of inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, *inputs):
                # [FUNCTION DOC] HELPER: Sums all positional inputs and returns the total.
                result = inputs[0]
                for i in range(1, len(inputs)):
                    result = result + inputs[i]
                return result

        inputs = []
        for _ in range(1000):
            inputs.append(torch.ones(8, 8, 8, dtype=torch.float16, device=self.device))
        inputs = tuple(inputs)
        model = Model()
        with torch.no_grad():
            AOTIRunnerUtil.compile(
                model,
                inputs,
            )

    def test_runtime_checks_complex(self):
        # [FUNCTION DOC] Checks a model returning complex inputs of three complex dtypes unchanged,
        # with a dynamic first dim on the first.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning its three inputs as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x0, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the tuple of the three inputs.
                return (x0, x1, x2)

        inputs = []
        x0 = torch.tensor([1, -1], dtype=torch.complex32, device=self.device)
        x1 = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1],
            dtype=torch.complex64,
            device=self.device,
        )
        x2 = torch.tensor(
            128,
            dtype=highest_precision_complex(self.device),
            device=self.device,
        )
        inputs.append(x0)
        inputs.append(x1)
        inputs.append(x2)
        dim0 = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "x0": {0: dim0},
            "x1": {},
            "x2": {},
        }
        with torch.no_grad():
            self.check_model(
                Model(),
                tuple(inputs),
                dynamic_shapes=dynamic_shapes,
            )

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_dtype_failed(self):
        # [FUNCTION DOC] With runtime checks on, compiles for a half precision input, then loads
        # the package and asserts that calling with a float32 input raises an exception. Skipped
        # in fbcode.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module casting x to float.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x converted to float.
                y = x.type(torch.float)
                return y

        x = torch.randn(1, 4, dtype=torch.float16, device=self.device)
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        x_casted = x.float()
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(x_casted)

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_device_type_failed(self):
        # [FUNCTION DOC] With runtime checks on, compiles for a CPU input, runs it, then asserts an
        # exception when the input is moved to the GPU. Skipped unless the device is the GPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        x = torch.randn(1, 4, dtype=torch.float16, device="cpu")
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                (x,),
            )

        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        aot_inductor_module(x)
        x_casted = x.to(GPU_TYPE)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(x_casted)

    def test_non_contiguous_output_alias(self):
        # [FUNCTION DOC] Checks returning a non-contiguous tensor and its contiguous copy: results
        # match eager and the two outputs have different data pointers.
        # Test return x, x.contiguous() where x is non-contiguous.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a transposed square and its contiguous copy.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Squares x, transposes it, makes a contiguous copy and
                # returns both.
                squared = x * x
                transposed = squared.t()  # non-contiguous
                contig = transposed.contiguous()
                return transposed, contig

        x = torch.randn(3, 4, dtype=torch.float16, device=self.device)
        model = Model()
        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                model,
                (x,),
            )
        actual = model(x)
        self.assertTrue(same(result, actual))

        # contiguous() should create a new tensor
        self.assertTrue(result[0].data_ptr() != result[1].data_ptr())

    def test_multiple_output_alias(self):
        # [FUNCTION DOC] Checks outputs that alias the same tensor: results match eager; the first
        # three outputs share a data pointer and the fourth does not.
        # Test when multiple outputs alias the same tensor
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tensor, two aliases of it and a
            # different tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the square, its contiguous alias, its reshaped
                # alias and the square times x.
                squared = x * x
                contig = squared.contiguous()  # alias
                reshaped = squared.reshape(squared.shape)  # alias
                cubed = squared * x
                return squared, contig, reshaped, cubed

        x = torch.randn(3, 4, dtype=torch.float32, device=self.device)
        model = Model()

        with torch.no_grad():
            result = AOTIRunnerUtil.run(
                model,
                (x,),
            )
        actual = model(x)
        self.assertTrue(same(result, actual))

        # squared, contig and reshaped alias the same tensor.
        self.assertTrue(result[0].data_ptr() == result[1].data_ptr())
        self.assertTrue(result[0].data_ptr() == result[2].data_ptr())
        # cubed shouldn't be an alias.
        self.assertTrue(result[0].data_ptr() != result[3].data_ptr())

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_checks_shape_failed(self):
        # [FUNCTION DOC] With runtime checks on, compiles an identity model with a dynamic first
        # dim (min 4, max 1024) and asserts that wrong second dim, wrong strides, and out-of-range
        # first dims raise exceptions while a valid first dim works.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        x = torch.randn(4, 4, 4, dtype=torch.float16, device=self.device)
        y0 = torch.randn(8, 4, 4, dtype=torch.float16, device=self.device)
        y1 = torch.randn(4, 8, 4, dtype=torch.float16, device=self.device)
        y2 = rand_strided(
            (4, 4, 4), (16, 1, 4), dtype=torch.float16, device=self.device
        )
        # batch size is outside of the range
        y3 = torch.randn(2048, 3, 4, dtype=torch.float16, device=self.device)
        y4 = torch.randn(2048, 4, 4, dtype=torch.float16, device=self.device)
        dim0 = Dim("s0", min=4, max=1024)
        dynamic_shapes = {
            "x": {0: dim0},
        }
        model = Model()
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model, (x,), dynamic_shapes=dynamic_shapes
            )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        # dynamic dim works fine
        _ = aot_inductor_module(y0)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y1)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y2)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y3)
        with self.assertRaisesRegex(Exception, ""):
            aot_inductor_module(y4)

    @patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
    def test_runtime_check_overbound_no_input_leak(self):
        # [FUNCTION DOC] With runtime checks on, compiles a linear model with a dynamic batch bound
        # of 64, then repeatedly calls it with 64 fresh oversized inputs, each expected to raise a
        # dim value too large error, and asserts the device allocated memory is unchanged (rejected
        # inputs are not leaked). GPU only, not MPS.
        # When AOTI_RUNTIME_CHECK_INPUTS rejects an over-bound input, the check
        # throws before the input handles are stolen into RAII. A correct runner
        # must free those un-stolen handles; otherwise every rejected call leaks
        # one at::Tensor (and its backing GPU storage) per input.
        # MPS has no memory_allocated accounting (needed below); skip CPU and MPS.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU with memory_allocated support")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a 2048 to 2048 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2048, 2048)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.linear(x)

        model = Model().to(self.device)
        x = torch.randn(8, 2048, device=self.device)  # example within bound
        dim0 = Dim("b", min=1, max=64)  # compiled upper bound = 64
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model, (x,), dynamic_shapes={"x": {0: dim0}}
            )
        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        # in-bound call works and warms up the allocator
        aot_inductor_module(torch.randn(8, 2048, device=self.device))

        device_interface = get_interface_for_device(GPU_TYPE)
        device = device_interface.current_device()
        device_interface.synchronize()
        mem_before = device_interface.memory_allocated(device)
        # Repeatedly reject FRESH over-bound inputs; storage must not be retained.
        for _ in range(64):
            over = torch.randn(128, 2048, device=self.device)  # 128 > 64 -> rejected
            with self.assertRaisesRegex(Exception, "dim value is too large"):
                aot_inductor_module(over)
            del over
        device_interface.synchronize()
        mem_after = device_interface.memory_allocated(device)
        self.assertEqual(mem_after, mem_before)

    def test_add_complex(self):
        # [FUNCTION DOC] Checks adding two complex tensors.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a plus b.
                return torch.add(a, b)

        x = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        y = torch.tensor(
            [1 + 1j, -1 + 1j, -2 + 2j, 3 - 3j, 0, 1j, 1, -1], device=self.device
        )
        self.check_model(Model(), (x, y))

    def test_embedding_bag(self):
        # [FUNCTION DOC] Checks the embedding bag op with weights, indices and offsets.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the embedding bag op with sum mode zero and no
            # per-sample weights.
            def forward(self, w, i, o):
                # [FUNCTION DOC] HELPER: Returns the output of the embedding bag op.
                return torch.ops.aten._embedding_bag(w, i, o, False, 0, False, None)

        example_inputs = (
            torch.randn([10, 4], device=self.device),
            torch.randint(10, [8], device=self.device),
            torch.tensor([0, 2, 6], device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "FFT operations are only supported on MacOS 14+",
    )
    def test_fft_c2c(self):
        # [FUNCTION DOC] Checks a complex-to-complex FFT over all dims returning the result and its
        # real part. Skipped on MPS before macOS 14.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning an n-dimensional FFT and its real part.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the FFT of x (computed twice) and the real part of
                # the second.
                return torch.fft.fftn(x), torch.fft.fftn(x).real

        example_inputs = (torch.randn(16, 16, 16, device=self.device),)
        self.check_model(Model(), example_inputs)

    def test_bool_input(self):
        # [FUNCTION DOC] Checks a model with a Python boolean input: the compiled model is
        # specialized on the branch taken for the example value (true).
        # Specialize on whichever branch the example input for b is
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the square or the double of x depending on
            # a boolean.
            def forward(self, x, b):
                # [FUNCTION DOC] HELPER: Returns x times x if b is true, else x plus x.
                if b:
                    return x * x
                else:
                    return x + x

        example_inputs = (torch.randn(3, 3, device=self.device), True)
        self.check_model(Model(), example_inputs)

    def test_int_list_input(self):
        # [FUNCTION DOC] Checks a model with a list of integers input used as multipliers.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying x by the first two entries of an integer list.
            def forward(self, x, i):
                # [FUNCTION DOC] HELPER: Returns x times the first entry times the second entry.
                return x * i[0] * i[1]

        example_inputs = (torch.randn(3, 3, device=self.device), [3, 4])
        self.check_model(Model(), example_inputs)

    def test_nested_tensor_from_jagged(self):
        # [FUNCTION DOC] Builds a nested tensor from jagged values and offsets, applies an MLP and
        # returns the values, and runs the legacy path for four input sets with varying data
        # amounts and batch sizes under dynamic shapes, comparing with eager.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with an MLP of two linear layers with ReLU and sigmoid
            # applied to a jagged nested tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the MLP.
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )

            def forward(self, values, offsets):
                # [FUNCTION DOC] HELPER: Wraps values and offsets in a jagged nested tensor,
                # applies the MLP and returns the values of the result.
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                res = self.mlp(nt)
                return res.values()

        model = Model().to(device=self.device)

        example_inputs_1 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 4, 10, 15], device=self.device),
        )

        # same "NT batch size", different actual amount of data
        example_inputs_2 = (
            torch.randn((31, 128), device=self.device),
            torch.tensor([0, 1, 20, 25, 31], device=self.device),
        )

        # same actual amount of data, different "NT batch size"
        example_inputs_3 = (
            torch.randn((15, 128), device=self.device),
            torch.tensor([0, 3, 10, 15], device=self.device),
        )

        # different "NT batch size"
        example_inputs_4 = (
            torch.randn((37, 128), device=self.device),
            torch.tensor([0, 5, 16, 25, 29, 37], device=self.device),
        )

        dim0_values = Dim("dim0_values", min=1, max=128)
        dim0_offsets = Dim("dim0_offsets", min=1, max=9)
        dynamic_shapes = {"values": {0: dim0_values}, "offsets": {0: dim0_offsets}}
        example_inputs_list = [
            example_inputs_1,
            example_inputs_2,
            example_inputs_3,
            example_inputs_4,
        ]
        for example_input in example_inputs_list:
            actual = AOTIRunnerUtil.legacy_run(
                self.device,
                model,
                example_input,
                dynamic_shapes=dynamic_shapes,
            )
            self.assertTrue(same(model(*example_input), actual))

    @common_utils.parametrize("max_autotune", [True, False])
    def test_misc_1(self, max_autotune):
        # [FUNCTION DOC] Parametrized on max_autotune. Checks a small recommendation-style model
        # (MLP, embedding bag, over-arch). Skipped for max_autotune on macOS CPU.
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Model with an MLP, an embedding bag and an over-arch network
            # combining their outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three components.
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )
                self.emb = nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)
                self.over_arch = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.Sigmoid()
                )

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs the MLP on x, the embedding bag on y, concatenates
                # along dim 1 and applies the over-arch.
                mlp_output = self.mlp(x)
                emb_output = self.emb(y)
                return self.over_arch(torch.concat([mlp_output, emb_output], dim=1))

        example_inputs = (
            torch.randn(16, 128, device=self.device),
            torch.randint(0, 128, (16, 10), device=self.device),
        )
        self.check_model(
            Model(), example_inputs, options=dict(max_autotune=max_autotune)
        )

    @skip_if_no_torchvision
    def test_torchvision_transforms_functional_tensor_resize(self):
        # [FUNCTION DOC] Checks the torchvision tensor resize function with the target size given
        # as a tensor converted to a list with bounds checks, with dynamic image height and width.
        # Skipped without torchvision.
        import torchvision

        # https://fb.workplace.com/groups/1075192433118967/permalink/1501860707118802/
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module resizing an image to a size read from a tensor.
            def forward(self, image: torch.Tensor, target_size: torch.Tensor):
                # [FUNCTION DOC] HELPER: Reads the target height and width, checks they are between
                # 1 and 4000, and resizes the image bilinearly without antialiasing.
                target_h, target_w = target_size.tolist()
                torch._check(target_h > 0)
                torch._check(target_w > 0)
                torch._check(target_h <= 4000)
                torch._check(target_w <= 4000)

                return torchvision.transforms._functional_tensor.resize(
                    image,
                    size=[target_h, target_w],
                    interpolation="bilinear",
                    antialias=False,
                )

        model = A()
        example_inputs = (
            torch.ones([3, 800, 600], device=self.device),
            torch.tensor([448, 336], device=self.device),
        )
        dynamic_shapes = {
            "image": {
                1: torch.export.Dim("height", min=1, max=4000),
                2: torch.export.Dim("width", min=1, max=4000),
            },
            "target_size": None,
        }
        self.check_model(model, example_inputs, dynamic_shapes=dynamic_shapes)

    @unittest.skipIf(config.triton.native_matmul, "matmul is generated")
    def test_aoti_debug_printer_codegen(self):
        # [FUNCTION DOC] Compiles a linear model with the intermediate value printer enabled
        # and checks that the generated code includes the tensor print shim call and before and
        # after launch markers around expected kernel calls with expected counts; then repeats
        # with a filter for one kernel name and checks only that kernel is wrapped. Skipped
        # with native matmul.
        # basic addmm model to test codegen for aoti intermediate debug printer
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)

        if self.device == "mps":
            kernel_calls = [("aoti_torch_mps_addmm_out", 2)]
        elif self.device == GPU_TYPE:
            kernel_calls = [
                ("triton_poi_fused_0", 1),
                (f"aoti_torch_{GPU_TYPE}_addmm_out", 2),
            ]
        else:
            kernel_calls = [("aoti_torch_cpu_addmm_out", 2)]

        # test default debug printing all tensor values codegen
        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )

            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check the codegen for debug printing around the actual kernel call is expected

            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

        # test printing selected kernel's tensor values codegen
        filtered_kernel_name = f"aoti_torch_{self.device}_addmm_out"
        with config.patch(
            {
                "aot_inductor.debug_intermediate_value_printer": "2",
                "aot_inductor.filtered_kernel_names": filtered_kernel_name,
            }
        ):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile, model, example_inputs
            )
            filtered_kernel_calls = [
                (filtered_kernel_name, 2),
            ]
            for kernel_call, count in filtered_kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

            kernel_calls_not_to_print = [
                kernel_call
                for kernel_call in kernel_calls
                if kernel_call[0] != filtered_kernel_name
            ]
            for kernel_name, _ in kernel_calls_not_to_print:
                FileCheck().check_not(f"before_launch - {kernel_name}").run(code)
                FileCheck().check_not(f"after_launch - {kernel_name}").run(code)

    @unittest.skipIf(
        config.triton.native_matmul, "different kernel name when native matmul"
    )
    @common_utils.parametrize("enable_kernel_profile", (True, False))
    @common_utils.parametrize("enable_kernel_context_guard", (True, False))
    def test_aoti_profiler(self, enable_kernel_context_guard, enable_kernel_profile):
        # [FUNCTION DOC] Parametrized on kernel profile and kernel context guard flags. Compiles
        # a linear model and checks the generated code: with profiling on it has the record
        # function handle for the addmm kernel and the context guard according to its flag; with
        # profiling off neither appears. Then checks numerics. Only on linux and win32; skipped
        # with native matmul.
        # basic addmm model
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        if sys.platform not in ["linux", "win32"]:
            raise unittest.SkipTest(
                "enable_kernel_profile only supported on linux and win32"
            )

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)
        kernel_calls = (
            f"aoti_torch_{GPU_TYPE}_addmm_out"
            if self.device == GPU_TYPE
            else "aoti_torch_cpu_addmm_out"
        )
        with config.patch(
            {
                "cpp.enable_kernel_profile": enable_kernel_profile,
                "cpp.enable_kernel_context_guard": enable_kernel_context_guard,
            }
        ):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            shim_fn_codes = f'RAIIAtenRecordFunctionHandle .*\\("{kernel_calls}"'
            if enable_kernel_profile:
                if enable_kernel_context_guard:
                    FileCheck().check("KernelContextGuard").run(code)
                else:
                    FileCheck().check_not("KernelContextGuard").run(code)
                FileCheck().check_regex(shim_fn_codes).run(code)
            else:
                FileCheck().check_not("RAIIAtenRecordFunctionHandle").check_not(
                    "KernelContextGuard"
                ).run(code)

            self.check_model(Model(N, K, self.device), example_inputs)

    @unittest.skipIf(
        sys.platform not in ["linux", "win32"],
        "enable_kernel_profile only supported on linux and win32",
    )
    def test_kernel_profile_repeated_kernel_numel(self):
        # [FUNCTION DOC] With kernel profiling enabled and a model calling a concat of three
        # dynamic inputs, asserts (on GPU) the generated code has no assignment to a kernel numel
        # variable without the 64-bit integer declaration, because each kernel call is in its own
        # scope. Then checks numerics. Only on linux and win32.
        # When enable_kernel_profile is on, each kernel call is wrapped in its
        # own {} scope block. If the same kernel is called multiple times with
        # symbolic numel, the int64_t declaration must be emitted each time
        # (not just on first use) since prior declarations go out of scope.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating three inputs along dim 0.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the concatenation of x, y, z.
                return torch.cat([x, y, z], dim=0)

        example_inputs = tuple(torch.randn(4, 8, device=self.device) for _ in range(3))
        dim0 = Dim("dim0", min=1, max=32)
        dynamic_shapes = {"x": {0: dim0}, "y": {0: dim0}, "z": {0: dim0}}

        with config.patch({"cpp.enable_kernel_profile": True}):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile,
                Model(),
                example_inputs,
                dynamic_shapes=dynamic_shapes,
            )
            # When profiling is enabled, every kernel numel variable must have
            # the int64_t type declaration since each kernel call lives in its
            # own scope block. Verify no bare assignment (without int64_t)
            # appears for numel variables.
            if self.device == GPU_TYPE:
                # Match bare numel assignments like "foo_xnumel = expr;"
                # but not declarations like "int64_t foo_xnumel = expr;"
                bare_numel_assign = re.compile(r"^\s*(\w+_[xr]numel)\s*=\s*.+;$")
                for line in code.splitlines():
                    m = bare_numel_assign.match(line)
                    if m:
                        self.fail(
                            f"Found numel assignment without int64_t declaration "
                            f"in profiling mode: {line.strip()}"
                        )

            self.check_model(
                Model(),
                example_inputs,
                dynamic_shapes=dynamic_shapes,
            )

    def test_aoti_user_defined_triton_kernel_profiling(self):
        # [FUNCTION DOC] Runs a user-defined triton kernel with kernel profiling and the torch
        # profiler active, exports the chrome trace and asserts exactly one event for the add
        # kernel exists whose input args list the four kernel argument names. GPU only, not MPS.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running an add kernel over a zeros output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel on 4 programs with four elements
                # and returns the output.
                out = torch.zeros_like(x)
                add_kernel[(4,)](x, y, out, n_elements=4, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )

        with (
            config.patch({"cpp.enable_kernel_profile": True}),
            torch.profiler.profile(
                record_shapes=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    getattr(torch.profiler.ProfilerActivity, GPU_TYPE.upper()),
                ],
            ) as prof,
        ):
            self.check_model(Model(), example_inputs)
        with common_utils.TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                import json

                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "kernels_.add_kernel_0"
                ]
                self.assertEqual(len(op_events), 1)
                self.assertEqual(
                    op_events[0]["args"].get("Input Args", ""),
                    ["in_ptr0", "in_ptr1", "out_ptr", "n_elements"],
                )

    def test_aoti_debug_printer_user_defined_triton_kernel(self):
        # [FUNCTION DOC] Compiles a model with a user-defined triton kernel with the value printer
        # on, and asserts the print shim is present and the kernel has the expected number of
        # before and after launch markers. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running an add kernel over a zeros output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel on 4 programs with four elements
                # and returns the output.
                out = torch.zeros_like(x)
                add_kernel[(4,)](x, y, out, n_elements=4, BLOCK_SIZE=16)
                return out

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )

        kernel_calls = [
            ("add_kernel_0", 3),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            # check the codegen for debug printing around the actual kernel call is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printer_cpp_kernel(self):
        # [FUNCTION DOC] Compiles a CPU model with a fused C++ kernel with the value printer
        # enabled and asserts the print shim call exists and the fused kernel has before and after
        # launch markers twice each. CPU only.
        if self.device != "cpu":
            raise unittest.SkipTest("cpu test case only")

        # a simple cpp kernel test case for testing the debug printer codegen
        # on cpp kernel cpu device.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling x by sqrt of three times its last dim as a new
            # CPU tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a float tensor with the last dim, takes sqrt of
                # three times it and multiplies x.
                t = torch.tensor(x.size(-1), device="cpu", dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        example_inputs = (torch.randn(4, 4, device="cpu"),)

        kernel_calls = [
            ("cpp_fused_mul_sqrt_0", 2),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            # check the codegen for debug printing around the actual kernel call is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printer_sym_inputs(self):
        # [FUNCTION DOC] Compiles a model whose grid and sizes depend on a maximum with an unbacked
        # scalar, with the value printer on, and asserts each expected scalar argument expression
        # (two numels and the max expression) appears twice in the generated code. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching an add kernel over ones tensors with a size from
            # the larger of a scalar from x and 512.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the scalar, takes the larger of it and 512, creates
                # ones tensors of that length, launches the add kernel with a grid depending on the
                # length, and returns the output.
                maxlen = max(x.item(), 512)
                a = torch.ones(maxlen, device=GPU_TYPE)
                b = torch.ones(maxlen, device=GPU_TYPE)
                out = torch.zeros_like(a)
                # unbacked symint in grid
                add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
                return out

        example_inputs = (torch.randint(high=1024, size=(1,), device=self.device),)

        expected_scalar_args = [
            "triton_poi_fused_zeros_like_0_xnumel",
            "triton_poi_fused_ones_1_xnumel",
            "std::max(static_cast<int64_t>(512L), static_cast<int64_t>(u0))",
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, Model(), example_inputs
            )
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)
            for scalar in expected_scalar_args:
                FileCheck().check_count(
                    f"{scalar}",
                    2,
                ).run(code)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_aoti_debug_printer_fp8_dtype(self):
        # [FUNCTION DOC] Compiles a scaled float8 matmul with the value printer on and checks the
        # print shim is present and the scaled matmul kernel has five before and after launch
        # markers each. GPU only, requires FP8 support.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doing a scaled matrix multiply with an e4m3-cast weight.
            def __init__(self, dtype):
                # [FUNCTION DOC] HELPER: Stores the requested output dtype.
                super().__init__()
                self.out_dtype = dtype

            def forward(self, x, weight, bias, scale_a, scale_b):
                # [FUNCTION DOC] HELPER: Casts weight to e4m3 and runs the scaled matmul with the
                # stored output dtype.
                # KNOWN ISSUE: the bias argument is ignored; the enclosing test's input_bias is
                # used via closure.
                weight = weight.to(e4m3_type)
                output = torch._scaled_mm(
                    x,
                    weight,
                    bias=input_bias,
                    out_dtype=self.out_dtype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                )
                return output

        dtype = torch.float16

        a_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        b_scale = torch.Tensor([1.0]).to(device=GPU_TYPE)
        input_bias = torch.rand(32, device=GPU_TYPE, dtype=dtype)
        weight_shape = (32, 16)
        weight = torch.rand(*weight_shape, device=GPU_TYPE, dtype=dtype).T
        a_inverse_scale = 1 / a_scale
        b_inverse_scale = 1 / b_scale

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=dtype).to(e4m3_type)

        kernel_calls = [
            (f"aoti_torch_{GPU_TYPE}__scaled_mm_out", 5),
        ]

        # test default debug printing all tensor values codegen
        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile,
                Model(dtype),
                (x, weight, input_bias, a_inverse_scale, b_inverse_scale),
            )

            # check the c shim print_tensor_handle call is triggered by the config and injected the cpp output code as expected
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check the codegen for debug printing around the actual kernel call is expected and float8 dtype is printed as expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"before_launch - {kernel_call}",
                    count,
                ).run(code)
                FileCheck().check_count(
                    f"after_launch - {kernel_call}",
                    count,
                ).run(code)

    def test_aoti_debug_printing_model_inputs_codegen(self):
        # [FUNCTION DOC] Compiles an addmm plus gelu model on CUDA or XPU with the value printer on
        # and checks the print shim is present, the generated triton source appears as a comment,
        # and the model inputs print marker appears three times.
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing gelu of an addmm of a scaled input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Scales a by a constant, computes addmm with c, x and b,
                # and applies gelu.
                x = a * 3.14
                y = torch.addmm(c, x, b)
                z = torch.nn.functional.gelu(y)
                return z

        example_inputs = (
            torch.randn(10, 20, device=GPU_TYPE),
            torch.randn(20, 30, device=GPU_TYPE),
            torch.randn(10, 30, device=GPU_TYPE),
        )
        model = Model()
        kernel_calls = [
            ("aoti_model_inputs", 3),
        ]

        with config.patch({"aot_inductor.debug_intermediate_value_printer": "2"}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )
            self.assertEqual("aoti_torch_print_tensor_handle" in code, True)

            # check if the triton kernel is printed as comment
            self.assertEqual("def triton_" in code, True)

            # check the codegen for debug printing around aoti model inputs is expected
            for kernel_call, count in kernel_calls:
                FileCheck().check_count(
                    f"{kernel_call}",
                    count,
                ).run(code)

    def test_size_from_multi_output(self):
        # [FUNCTION DOC] Checks a model that uses the unique op with inverse indices (multi-output
        # with data-dependent size), then relu on the cloned values.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying unique and a ReLU layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a ReLU module.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes sorted unique values and inverse indices, clones
                # the values and returns relu of them and the indices.
                _x, _i = torch.unique(x, sorted=True, return_inverse=True)
                _x = _x.detach().clone()
                return self.relu(_x), _i

        example_inputs = (torch.randn(8, device=self.device),)
        self.check_model(Model(), example_inputs)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_sym_i64_input_codegen(self):
        # [FUNCTION DOC] With scalar outputs captured, compiles a model with an int32 scalar input
        # used as an unbacked size and in the grid of an add kernel, and checks the generated code
        # declares the scalar arguments of the kernel calls with the expected variable lists
        # (shifted by one more assertion if full runtime asserts are on). GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating ones tensors sized by a scalar and adding them
            # via a triton kernel.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the scalar from x, builds two ones tensors of
                # that length, launches the add kernel with a grid depending on it, and returns
                # the output.
                x_symint = x.item()
                a = torch.ones(x_symint, device=GPU_TYPE)
                b = torch.ones(x_symint, device=GPU_TYPE)
                out = torch.zeros_like(a)
                # unbacked symint in grid
                add_kernel[(1, 1, x_symint)](a, b, out, x_symint, 32)
                return out

        example_inputs = (
            torch.randint(high=1024, size=(1,), device=self.device, dtype=torch.int32),
        )
        # This simple unit test case model generates two triton kernels:
        # 1. triton_poi_fused_ones_1:
        # triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i64'}
        # 2. add_kernel:
        # triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr': '*fp32', 'n_elements': 'i64'}
        # input u0 was defined as int32_t initially, verify for every kernel var args downstream,
        # it gets explicitly declared using its data types in the cpp wrapper codegen code.
        expected_scalar_args = [
            "buf3, u0",
            "buf4, u0",
            "buf4, buf5, buf3, u0",
        ]
        if full_aoti_runtime_assert():
            # we'll have one more assertion
            expected_scalar_args = [
                "buf4, u0",
                "buf5, u0",
                "buf5, buf6, buf4, u0",
            ]
        # check the new behavior of codegen is expected
        result, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, Model(), example_inputs
        )
        for scalar_line in expected_scalar_args:
            FileCheck().check_count(
                scalar_line,
                1,
            ).run(code)
        self.check_model(Model(), example_inputs)

    def test_input_codegen_with_sympy_expr(self):
        # [FUNCTION DOC] Checks a model with five dynamic inputs whose sizes are related by derived
        # dimension expressions (twice a base dim, and sums), exercising code generation for inputs
        # with symbolic expression sizes. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that ORs two byte tensors, concatenates two float tensors
            # and adds them, and adds a sliced tensor to offsets, returning both results.
            def forward(self, getitem_54, getitem_52, getitem_19, values_2, offsets):
                # [FUNCTION DOC] HELPER: Computes the bitwise or of the first two inputs, the
                # concatenation of the next two plus it, and a slice-plus-offsets result; returns
                # the sum and the slice result.
                bitwise_or = torch.bitwise_or(getitem_54, getitem_52)
                combined = torch.cat([getitem_19, values_2], dim=0)
                add = combined + bitwise_or

                sliced = values_2[:-1] + offsets
                return add, sliced

        inps = (
            torch.randint(0, 1, (240,), device=GPU_TYPE, dtype=torch.uint8),
            torch.randint(0, 1, (240,), device=GPU_TYPE, dtype=torch.uint8),
            torch.randn((192,), device=GPU_TYPE),
            torch.randn((48,), device=GPU_TYPE),
            torch.randint(0, 100, (47,), device=GPU_TYPE, dtype=torch.uint8),
        )

        dim = torch.export.Dim("dimensionality")
        derived_dim = 2 * dim
        spec = {
            "getitem_54": (Dim.AUTO,),  # [s33 + 2*s40 + 1]
            "getitem_52": (Dim.AUTO,),  # [s33 + 2*s40 + 1]
            "getitem_19": (derived_dim,),  # [2*s40]
            "values_2": (Dim.AUTO,),  # [s33 + 1]
            "offsets": (Dim.AUTO,),  # [s33]
        }

        self.check_model(MyModel(), inps, dynamic_shapes=spec)

    @common_utils.parametrize("mark_unbacked", (True, False))
    def test_unbacked_equals_input_size_runtime_assertion(self, mark_unbacked: bool):
        # [FUNCTION DOC] Parametrized on mark_unbacked. Checks generated runtime assertions
        # relating an unbacked size to another size (an unbacked one when marked unbacked, a backed
        # one when marked dynamic): the lower bound assertion text appears once in the code,
        # running with a zero-size input raises an exception, and with scalar asserts disabled the
        # same inputs run without error.
        # This test checks the unbacked symint runtime assertions, for the following cases:
        # (A) an unbacked symint equals an unbacked symint (mark_unbacked=True)
        # (B) an unbacked symint equals a backed symint    (mark_unbacked=False)
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating a ones matrix sized by the nonzero count of a and
            # the size of b, requiring at least one row, and adding c.
            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Counts the nonzero elements of a, builds the ones matrix,
                # checks it has at least one row and returns it plus c.
                nz = torch.nonzero(a)
                ones = a.new_ones([nz.size(0), b.size(0)])
                torch._check(ones.size(0) >= 1)
                equals = torch.add(ones, c)
                return equals

        model = Model()
        example_inputs = (
            torch.ones(64, device=self.device),
            b := torch.randn((32,), device=self.device),
            c := torch.randn((64, 32), device=self.device),
        )
        if mark_unbacked:
            torch._dynamo.decorators.mark_unbacked(c, 0)
        else:
            torch._dynamo.mark_dynamic(c, 0)

        # Check the runtime assertion is codegen'ed.
        so_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.legacy_compile, model, example_inputs
        )
        lowerbound_check = "u1 >= 1" if mark_unbacked else "u0 >= 2"
        FileCheck().check_count(lowerbound_check, 1).run(code)

        compiled = AOTIRunnerUtil.legacy_load(self.device, so_path)
        compiled(*example_inputs)

        # Check the runtime assertion.
        with self.assertRaisesRegex(Exception, ""):
            unexpected_inputs = (torch.ones(0, device=self.device), b, c)
            compiled(*unexpected_inputs)

        # Try it again without runtime assertions.
        with config.patch({"scalar_asserts": False}):
            AOTIRunnerUtil.run_multiple(model, [example_inputs, unexpected_inputs])

    def test_multi_input_nonzero_slice_shared_dim(self):
        # [FUNCTION DOC] Regression test: several inputs sharing a dynamic batch dim are sliced
        # with the same nonzero count; generated guard code used to reference symbols replaced
        # during constraint solving but never defined.
        # Regression: when multiple inputs share a dynamic batch dim and are
        # sliced with the same nonzero result, the generated C++ guard code
        # referenced symbolic variables that were replaced during constraint
        # solving but never defined in the wrapper.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer, slicing three inputs from an offset
            # equal to the nonzero count of a mask.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 4 by 4 linear layer.
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x, a, b, mask):
                # [FUNCTION DOC] HELPER: Counts nonzero entries in mask, slices x, a and b from
                # that offset, and returns the linear of the sliced x plus the sliced a and b.
                n = torch.nonzero(mask).shape[0]
                return self.linear(x[n:]) + a[n:] + b[n:]

        B = Dim("B", min=1, max=1024)
        dynamic_shapes = {
            "x": {0: B},
            "a": {0: B},
            "b": {0: B},
            "mask": {0: B},
        }
        example_inputs = (
            torch.randn(8, 4, device=self.device),
            torch.randn(8, 4, device=self.device),
            torch.randn(8, 4, device=self.device),
            torch.tensor(
                [True, True, False, True, False, False, True, True],
                device=self.device,
            ),
        )
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_none_args_aot_codegen(self):
        # [FUNCTION DOC] Checks code generation for a triton kernel called with a None input
        # pointer and an argument known to equal one, by running check_model both with a None input
        # and with a real tensor input (compilation specializes on each). GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Autotuned triton kernel (two block-size configs, keyed on element
        # count) computing the sine of the input block plus an argument, with zero used in place of
        # the input when the pointer is None; stores into the output pointer. The extra
        # equal-to-one argument is there to test index recomputation after None args are removed.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            # We want to include an arg known to be 1 at compile time
            # This is because we remove None args from the arg list; changing the eq_1/constexpr arg indices.
            # We want to make sure we recompute these correctly
            EQ_1_ARG,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            if in_ptr0 is not None:
                x = tl.load(in_ptr0 + offsets, mask=mask)
            else:
                x = 0.0
            output = tl.sin(x) + EQ_1_ARG
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            # [FUNCTION DOC] HELPER: Launches the sin kernel with one program per output element,
            # with argument one and the element count; returns the output.
            n_elements = out.numel()
            sin_kernel[(n_elements,)](x, out, 1, n_elements)
            return out

        x = torch.randn(65, device=self.device)
        out = torch.empty_like(x)

        not_none_inputs = (x, out)
        none_inputs = (None, out)

        # AOTI compilation specializes on either None or non-None inputs
        # So we have to check twice here

        self.check_model(sin_triton, none_inputs)
        self.check_model(sin_triton, not_none_inputs)

    @skipIfXpu(
        msg="SYCL work-item index overflow issue when block sizes are used in this test."
    )
    def test_autotune_int64_user_defined_triton_kernel(self):
        # [FUNCTION DOC] Compiles a model using a triton operator whose kernel computes with 64-bit
        # block ids over a two-dimensional grid, on a very large int8 input (1379584 by 2048) with
        # dynamic first dim and max-autotune; checks via check_model. Skips on out-of-memory. GPU
        # only; skipped on XPU.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Triton kernel: computes a 64-bit linear block id from a
        # two-dimensional program id, loads two blocks within bounds, adds them as int32, masks to
        # 8 bits and stores as int8.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid0 = tl.program_id(axis=0).to(tl.int64)
            pid1 = tl.program_id(axis=1).to(tl.int64)
            grid0 = tl.num_programs(axis=0).to(tl.int64)
            block_id = pid1 * grid0 + pid0
            block_start = block_id * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            s = x.to(tl.int32) + y.to(tl.int32)
            output = (s & 0xFF).to(tl.int8)
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args=())
        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom triton operator: allocates an output like x and runs
            # the add kernel over a two-dimensional grid that splits blocks to stay under the grid
            # x limit; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Grid function: splits the number of blocks over two grid
                # axes with ceiling division by 65535 so no axis exceeds the limit; returns the
                # two sizes.
                num_blocks = triton.cdiv(n_elements, meta["BLOCK_SIZE"])
                # 2D split to respect the 65535 grid-x limit. Use cdiv-by-constant
                # rather than min(num_blocks, 65535): a symbolic min introduces an
                # inequality guard that AOTInductor dynamic-shape export rejects.
                grid1 = triton.cdiv(num_blocks, 65535)
                grid0 = triton.cdiv(num_blocks, grid1)
                return (grid0, grid1)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 256)
            return output

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom add, splitting the result into four
            # column chunks and returning the first scaled by three.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the custom add of x with itself, splits into four
                # chunks along dim 1 and returns three times the first chunk.
                x = custom_add(x, x)
                split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
                    x, [512, 512, 512, 512], 1
                )
                getitem_29 = split_with_sizes_1[0]
                return getitem_29 * 3

        n = 1379584

        try:
            buf196 = torch.randint(
                0, 100, (n, 2048), dtype=torch.int8, device=self.device
            )
            example_inputs = (buf196,)

            self.check_model(
                Model(),
                example_inputs,
                dynamic_shapes={
                    "x": (Dim("x", max=1379584), Dim.STATIC),
                },
                options={"max_autotune": True},
            )
        except torch.OutOfMemoryError:
            # CI can OOM because this test uses too much memory
            raise unittest.SkipTest("OOM. Test is too large") from None

    @skipIfWindows(
        msg="OpenMP crashed application on windows"
    )  # TODO: (xuhancn) need to root cause and fix.
    def test_issue_140766(self):
        # [FUNCTION DOC] Regression test for issue 140766: a two block network of layer
        # norm, attention over reshaped windows and an MLP with dynamic batch up to 12.
        # Skipped on Windows.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with an MLP, a layer norm and a stored attention function
            # applied twice in sequence to a transposed input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the MLP (128 to 512 to 128), a layer norm over 128
                # and stores the scaled dot product attention function.
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(128, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128),
                )
                self.norm = torch.nn.LayerNorm(128)
                self.attn = torch.nn.functional.scaled_dot_product_attention

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Transposes the input and applies the block twice.
                # [2, 128, 4096]
                x = x.transpose(1, 2)
                # [2, 4096, 128]
                for _ in range(2):
                    x = self.forward_block(x)
                return x

            def forward_block(self, x):
                # [FUNCTION DOC] HELPER: One block on a batch of 64 by 64 positions with 128
                # channels: layer norm, reshape, attention, window reshuffle, residual add and
                # MLP residual.
                # x: B, H*W, C
                B = x.shape[0]
                H, W, C = 64, 64, 128
                shortcut = x
                x = self.norm(x)
                x = x.reshape(B, H, W, C)
                # B, H, W, C
                x = self.attn(x, x, x)
                x = x.reshape(B, H // 8, W // 8, 8, 8, -1)
                x = x.transpose(2, 3).reshape(B, H * W, -1)

                x = shortcut + x
                x = x + self.mlp(self.norm(x))
                return x

        bs = torch.export.Dim("bs", max=12)
        example_inputs = (torch.randn(2, 128, 4096, device=self.device),)
        self.check_model(Model(), example_inputs, dynamic_shapes={"x": {0: bs}})

    @requires_gpu
    def test_d2h_copy(self):
        # [FUNCTION DOC] Exports and packages a model that permutes, replaces NaN and moves to CPU
        # asynchronously, runs it under the profiler and asserts the result matches eager and that
        # no contiguous op appears in the profiled ops. CUDA/XPU only.
        # device to copy host should always have the same stride
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("This test is only for CUDA or XPU")

        class ToCpuModel(nn.Module):
            # [CLASS DOC] HELPER: Module that transposes, cleans NaN and infinities, copies to host
            # and returns a row and a ones tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Permutes x, replaces nan and infinities with zero, moves
                # to CPU without blocking, takes the first row and a ones tensor like it, and
                # returns both.
                predictions = x.permute(1, 0)
                predictions = torch.nan_to_num(
                    predictions, nan=0.0, posinf=0.0, neginf=0.0
                )
                predictions = predictions.to("cpu", non_blocking=True)
                p = predictions[0]
                ones = p.new_ones(1)
                return p, ones

        model = ToCpuModel().to(GPU_TYPE)
        input_tensor = torch.randn(5, 10, device=GPU_TYPE).to(dtype=torch.float16)
        ep = torch.export.export(model, (input_tensor,))
        package_path = torch._inductor.aoti_compile_and_package(ep)

        aoti_model = torch._inductor.aoti_load_package(package_path)

        expect_res = model(input_tensor)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            true_res = aoti_model(input_tensor)

        self.assertEqual(expect_res, true_res)
        all_ops = [event.key for event in prof.key_averages()]
        self.assertTrue(not any("aten::contiguous" in op for op in all_ops))

    @requires_multigpu()
    def test_cuda_to_cuda_device_copy(self):
        # [FUNCTION DOC] Compiles a model returning x and a copy of it on a second CUDA device,
        # loads the package on device zero and asserts the output devices and values. Needs
        # multiple CUDA GPUs; not ROCm.
        if self.device != GPU_TYPE or GPU_TYPE != "cuda" or TEST_WITH_ROCM:
            raise unittest.SkipTest("This test requires CUDA")

        device0 = torch.device(type=GPU_TYPE, index=0)
        device1 = torch.device(type=GPU_TYPE, index=1)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning its input and a copy on device one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x and x moved to the second device.
                return x, x.to(device1)

        model = Model().to(device0)
        x = torch.randn(10, device=device0)
        package_path = AOTIRunnerUtil.compile(model, (x,))
        aoti_model = torch._inductor.aoti_load_package(package_path, device_index=0)

        expected = model(x)
        actual = aoti_model(x)

        self.assertEqual(actual[0].device, device0)
        self.assertEqual(actual[1].device, device1)
        self.assertEqual(actual, expected)

    def test_so_without_weight(self):
        # [FUNCTION DOC] Compiles a linear model twice with constants packaged in the library and
        # not, and asserts the first library is over 10MB and the weightless one is under 10MB.
        # Loads the weightless runner, checks the constant name mappings, attaches the weights
        # through the constant update call and compares with eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight of shape n by k and bias of
                # n on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": True,
                }
            ),
        ):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": False,
                }
            ),
        ):
            so_path_weightless = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )
        self.assertTrue(os.path.getsize(so_path) > 10_000_000)
        self.assertTrue(os.path.getsize(so_path_weightless) < 10_000_000)

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path_weightless)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
            "L__self___bias": "L__self___bias",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        attach_weights = {
            "L__self___weight": model.weight,
            "L__self___bias": model.bias,
        }
        runner.update_constant_buffer(attach_weights, False, False)
        expected = model(test_inputs)
        output = runner_call(test_inputs)

        atol, rtol = 3e-4, 3e-4
        self.assertEqual(expected, output, atol=atol, rtol=rtol)

    @parametrize(
        "package_format",
        ["in_so", "binary_blob"],
    )
    def test_package_constants_serialization(self, package_format):
        # [FUNCTION DOC] Parametrized on package format (constants inside the library, or a binary
        # blob on disk). Compiles with constants kept, using a model with odd-sized weights,
        # bfloat16 and int8 buffers and an empty buffer so alignment padding and the empty-tensor
        # skip are exercised, then checks via check_model with tolerance 3e-4.
        # Exercises the packaged constant serialization path with a mix of
        # dtypes, odd shapes, and an empty buffer so alignment padding and
        # the numel==0 branch are actually stressed. Covers both packaging
        # formats since they share the same serialization code.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with three linear layers of odd sizes, a bfloat16 scale
            # buffer, an int8 offset buffer and an empty buffer.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Creates the layers and registers the three buffers on
                # the device.
                super().__init__()
                # Odd dims produce fp32 sizes that aren't multiples of
                # ALIGN_BYTES=64 (e.g. 7*13*4=364 bytes).
                self.fc1 = torch.nn.Linear(7, 13, device=device)
                self.fc2 = torch.nn.Linear(13, 17, device=device)
                self.fc3 = torch.nn.Linear(17, 23, device=device)
                # bf16 scale: 23 * 2 = 46 bytes.
                self.register_buffer(
                    "bf16_scale",
                    torch.randn(23, dtype=torch.bfloat16, device=device),
                )
                # int8 offset: 23 bytes.
                self.register_buffer(
                    "i8_offset",
                    torch.arange(-11, 12, dtype=torch.int8, device=device),
                )
                # numel==0 buffer exercises the size-zero skip path.
                self.register_buffer(
                    "empty", torch.empty(0, dtype=torch.float32, device=device)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the three linear layers, scales by the bfloat16
                # buffer, adds the int8 buffer, adds the sum of the empty buffer, and returns
                # the result.
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                x = x * self.bf16_scale.to(x.dtype)
                x = x + self.i8_offset.to(x.dtype)
                return x + self.empty.sum()

        example_inputs = (torch.randn(4, 7, device=self.device),)
        patches = {"always_keep_tensor_constants": True}
        if package_format == "in_so":
            patches["aot_inductor.package_constants_in_so"] = True
        else:
            patches["aot_inductor.package_constants_on_disk_format"] = "binary_blob"
        with config.patch(patches):
            self.check_model(Model(self.device), example_inputs, atol=3e-4, rtol=3e-4)

    def test_weight_on_disk_legacy(self):
        # [FUNCTION DOC] Compiles with constants stored on disk in the pickle weights format with
        # the package option on, packages the files into a pt2 archive, loads it with weights
        # loaded from disk, and compares the model output with eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 128, 2048, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)

        with (
            torch.no_grad(),
            config.patch(
                {
                    "always_keep_tensor_constants": True,
                    "aot_inductor.package_constants_in_so": False,
                    "aot_inductor.package_constants_on_disk_format": "pickle_weights",
                    "aot_inductor.package": True,
                }
            ),
        ):
            aoti_files = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        with WritableTempFile(suffix=".pt2") as f:
            package_path = package_aoti(
                f.name,
                {"model": aoti_files},
            )
            pt2_contents = load_pt2(package_path, load_weights_from_disk=True)
            loaded1 = pt2_contents.aoti_runners["model"]

        atol, rtol = 3e-4, 3e-4
        self.assertEqual(loaded1(a), model(a), atol=atol, rtol=rtol)

    def test_extract_constants_map(self):
        # [FUNCTION DOC] Compiles a linear model with constants kept, loads the runner and
        # exercises extraction of the constants map: the active map equals the original weights;
        # after updating the inactive buffer the inactive map equals the new weights; after
        # swapping the roles reverse.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        original_weights = {
            "L__self___weight": model.weight,
            "L__self___bias": model.bias,
        }
        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }

        # Extract weights with use_inactive = False, this should be the current weight.
        extracted_original_weights = runner.extract_constants_map(False)
        self.assertEqual(original_weights, extracted_original_weights)

        # update the inactive weights with new_weights, extract inactive weights.
        runner.update_constant_buffer(new_weights, True, False)
        extracted_new_weights = runner.extract_constants_map(True)
        self.assertEqual(new_weights, extracted_new_weights)

        # Swap constant buffer, this should give us the opposite weights.
        runner.swap_constant_buffer()

        extracted_inactive_weights = runner.extract_constants_map(True)
        extracted_active_weights = runner.extract_constants_map(False)
        self.assertEqual(original_weights, extracted_inactive_weights)
        self.assertEqual(new_weights, extracted_active_weights)

    def test_update_constant_buffer(self):
        # [FUNCTION DOC] Compiles a linear model with constants kept using the legacy API, checks
        # the constant name to original name mapping, then updates the active constant buffer with
        # new weights and checks the runner output matches the linear with the new weights.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        # Attribute naming has changed in the new export API, so still use the legacy API here.
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
            "L__self___bias": "L__self___bias",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        runner.update_constant_buffer(new_weights, False, False)
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output)

    def test_update_constant_buffer_simple(self):
        # [FUNCTION DOC] Compiles an add-with-constant model, checks the constant name mapping,
        # updates the active buffer with a new weight with user-managed buffer disabled, and
        # compares runner output against the eager model with the new weight.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a constant 3 by 3 weight to its input.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores a random constant weight on the device.
                super().__init__()
                self.weight = torch.randn((3, 3), device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns a plus the weight.
                return a + self.weight

        model = Model(self.device)
        a = torch.randn((3, 3), device=self.device)
        example_inputs = (a,)

        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        # Let's check whether the model has correct constant name mapping.
        expected_original_fqns = {
            "L__self___weight": "L__self___weight",
        }
        self.assertEqual(
            expected_original_fqns, runner.get_constant_names_to_original_fqns()
        )

        test_inputs = torch.randn((3, 3), device=self.device)
        new_weight = torch.randn((3, 3), device=self.device)
        model.weight = new_weight
        attach_weights = {"L__self___weight": new_weight}
        runner.update_constant_buffer(attach_weights, False, False, False)
        expected = model(test_inputs)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner: flattens args and kwargs with the
            # generic tree flatten, keeps tensor leaves, runs the runner and unflattens outputs
            # with the output spec from the call spec.
            call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = pytree.tree_flatten((args, kwargs))[0]
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
            return pytree.tree_unflatten(flat_outputs, out_spec)

        output = runner_call(test_inputs)
        self.assertEqual(expected, output)

    def test_update_inactive_constant_buffer(self):
        # [FUNCTION DOC] Compiles a linear model, updates the inactive constant buffer with new
        # weights and checks the output before the swap still uses the old weights and after the
        # swap uses the new ones.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )

        runner.update_constant_buffer(new_weights, True, False)
        output_before_swap = runner_call(test_inputs)
        runner.swap_constant_buffer()
        output_after_swap = runner_call(test_inputs)

        self.assertEqual(expected, output_before_swap)
        self.assertEqual(new_expected, output_after_swap)

    def test_free_inactive_buffer(self):
        # [FUNCTION DOC] Updates the inactive buffer, swaps, frees the inactive (old) buffer,
        # refills it with another set of weights and checks outputs. GPU only.
        # KNOWN ISSUE: the expected value for the refilled weights is computed from the earlier
        # weights and the refill is not swapped in before the second run, so the final comparison
        # checks that the active weights are unchanged by the refill rather than the new weights.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        # Check the outputs, make sure the model is correct here.
        self.assertEqual(expected, output)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        runner.update_constant_buffer(new_weights, True, False)

        # Make sure we have swapped buffer
        runner.swap_constant_buffer()
        output_after_swap = runner_call(test_inputs)
        self.assertEqual(new_expected, output_after_swap)

        # Free the secondary buffer
        runner.free_inactive_constant_buffer()

        # Create a new set of weights to refill into the already freed buffer.
        new_weights_1 = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        new_expected_1 = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        runner.update_constant_buffer(new_weights_1, True, False)

        output_after_swap_1 = runner_call(test_inputs)
        self.assertEqual(new_expected_1, output_after_swap_1)

        runner.free_inactive_constant_buffer()

    def test_create_with_external_constants(self):
        # [FUNCTION DOC] Creates a runner from caller-supplied external weights (CPU, XPU or CUDA
        # runner; skipped on MPS) and asserts that the constant loading hook is called for a normal
        # runner but not for the external one, and that the external runner output uses the
        # supplied weights rather than those baked into the library.
        # AOTInductorModelContainerCreateWithExternalConstants: a runner
        # built from caller-supplied weights must use those weights at
        # inference time, not the constants baked into the .so.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        with torch.no_grad(), config.patch({"always_keep_tensor_constants": True}):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        # Caller-allocated weights distinct from the .so-baked ones.
        external_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }

        if self.device == "cpu":
            normal_runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
            self.assertTrue(normal_runner.did_call_load_constants())
            del normal_runner
            runner = torch._C._aoti.AOTIModelContainerRunnerCpu(
                so_path, 1, external_weights
            )
        elif self.device == "xpu":
            normal_runner = torch._C._aoti.AOTIModelContainerRunnerXpu(
                so_path, 1, self.device
            )
            self.assertTrue(normal_runner.did_call_load_constants())
            del normal_runner
            runner = torch._C._aoti.AOTIModelContainerRunnerXpu(
                so_path, 1, self.device, "", external_weights
            )
        elif self.device == "mps":
            raise unittest.SkipTest("external-constants ctor not bound on MPS runner")
        else:
            normal_runner = torch._C._aoti.AOTIModelContainerRunnerCuda(
                so_path, 1, self.device
            )
            self.assertTrue(normal_runner.did_call_load_constants())
            del normal_runner
            runner = torch._C._aoti.AOTIModelContainerRunnerCuda(
                so_path, 1, self.device, "", external_weights
            )
        self.assertFalse(runner.did_call_load_constants())

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        # If the runner ignored external_weights and used .so-baked weights,
        # output would equal model(test_inputs) with the original (different)
        # weights, and this assertion would fail. The external constructor also
        # calls the release_* path after disabling blob release, so this checks
        # that no-op release did not drop caller-owned constants.
        expected = torch.nn.functional.linear(
            test_inputs,
            external_weights["L__self___weight"],
            external_weights["L__self___bias"],
        )
        output = runner_call(test_inputs)
        self.assertEqual(expected, output)
        self.assertFalse(runner.did_call_load_constants())

    def test_update_user_managed_buffer(self):
        # [FUNCTION DOC] Tests update_constant_buffer with and without the user-managed flag on
        # CUDA/XPU (CUDA also compiles weights with the caching allocator). Without the flag,
        # updating the inactive buffer allocates device memory and later in-place edits of the
        # source tensors do not change the runner output. With the flag, no memory is allocated,
        # the runner aliases the caller's tensors so in-place edits and a loaded state dict show up
        # in the output, which then matches the model but differs from the earlier expected value.
        # Compares with eager at tolerance 1e-3.
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias applying a linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Stores random constant weight and bias on the device.
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a.
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M, N, K = 1024, 4096, 4096
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)
        # Attribute naming has changed in the new export API, so still use the legacy API here.
        compile_config = {"always_keep_tensor_constants": True}
        if self.device == "cuda":
            compile_config["aot_inductor.weight_use_caching_allocator"] = True

        with torch.no_grad(), config.patch(compile_config):
            so_path = AOTIRunnerUtil.legacy_compile(
                model=model,
                example_inputs=example_inputs,
            )

        runner = AOTIRunnerUtil.legacy_load_runner(self.device, so_path)

        def constant_buffer_memory_used():
            # [FUNCTION DOC] HELPER: Returns the memory currently used by device allocations:
            # allocated bytes on CUDA, otherwise the negated free memory reported by the device.
            if self.device == "cuda":
                return torch.cuda.memory_allocated(self.device)
            free_memory, _ = getattr(torch, GPU_TYPE).mem_get_info(self.device)
            return -free_memory

        def runner_call(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the loaded runner like a function: flattens args and
            # kwargs with the stored input tree spec, keeps only tensor leaves, runs the runner and
            # unflattens the outputs with the output tree spec.
            import torch.fx._pytree as fx_pytree

            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        test_inputs = torch.randn(M, K, device=self.device)
        expected = model(test_inputs)
        output = runner_call(test_inputs)
        self.assertEqual(expected, output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        mem_before = constant_buffer_memory_used()
        # Do not use user managed_buffer, should allocate an owned constant buffer.
        runner.update_constant_buffer(new_weights, True, False, False)
        mem_after = constant_buffer_memory_used()
        self.assertGreater(mem_after, mem_before)

        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        # Inplace substitube tensor, without user managed buffer, result should be different.
        new_weights["L__self___weight"].add_(1)
        new_weights["L__self___bias"].add_(1)

        new_output = runner_call(test_inputs)
        # Same as the previous result
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        # Differ from latest result
        self.assertNotEqual(new_expected, new_output)

        # Clear out all buffers
        runner.free_inactive_constant_buffer()
        runner.swap_constant_buffer()
        runner.free_inactive_constant_buffer()

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }
        mem_before = constant_buffer_memory_used()
        # Try user managed_buffer, should not allocate an owned constant buffer.
        runner.update_constant_buffer(new_weights, True, False, True)
        mem_after = constant_buffer_memory_used()
        self.assertEqual(mem_before, mem_after, atol=1e-3, rtol=1e-3)

        runner.swap_constant_buffer()
        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        # Inplace substitube tensor, with user managed buffer, result should be the same.
        new_weights["L__self___weight"].add_(1)
        new_weights["L__self___bias"].add_(1)

        new_output = runner_call(test_inputs)
        new_expected = torch.nn.functional.linear(
            test_inputs, new_weights["L__self___weight"], new_weights["L__self___bias"]
        )
        self.assertEqual(new_expected, new_output, atol=1e-3, rtol=1e-3)

        new_weights = {
            "L__self___weight": torch.randn(N, K, device=self.device),
            "L__self___bias": torch.randn(N, device=self.device),
        }

        runner.update_constant_buffer(new_weights, True, False, True)
        runner.swap_constant_buffer()

        model.weight = torch.nn.Parameter(new_weights["L__self___weight"])
        model.bias = torch.nn.Parameter(new_weights["L__self___bias"])

        updated_state_dict = {
            "weight": torch.ones_like(model.weight),
            "bias": torch.zeros_like(model.bias),
        }

        model.load_state_dict(updated_state_dict)

        new_output = runner_call(test_inputs)
        expected_output = model(test_inputs)
        torch.testing.assert_close(new_output, expected_output, atol=1e-3, rtol=1e-3)

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(new_expected, new_output, atol=1e-3, rtol=1e-3)

    def test_load_constants_allow_h2d_copy(self):
        # [FUNCTION DOC] End-to-end test of loading constants from a CPU state dict into a GPU
        # model with the allow host-to-device copy option: it works and the result matches; without
        # the option the same call raises RuntimeError; combining it with user-managed buffers also
        # raises RuntimeError. GPU only, not MPS.
        # End-to-end check that AOTICompiledModel.load_constants(allow_h2d_copy=True)
        # accepts a CPU state_dict for a non-CPU model, the way a typical user
        # would after torch.load(..., map_location="cpu").
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single parameter weight applying a
            # bias-free linear.
            def __init__(self, n, k, device):
                # [FUNCTION DOC] HELPER: Creates a random weight parameter of shape n by k on
                # the device.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k, device=device))

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the linear of a with the weight.
                return torch.nn.functional.linear(a, self.weight)

        M, N, K = 8, 6, 16
        model = Model(N, K, self.device)
        a = torch.randn(M, K, device=self.device)
        example_inputs = (a,)

        package_path = AOTIRunnerUtil.compile(model, example_inputs)
        compiled = torch._inductor.aoti_load_package(package_path)

        (weight_fqn,) = compiled.get_constant_fqns()
        cpu_state_dict = {weight_fqn: torch.randn(N, K, device="cpu")}
        compiled.load_constants(
            cpu_state_dict, check_full_update=True, allow_h2d_copy=True
        )

        test_inputs = torch.randn(M, K, device=self.device)
        expected = torch.nn.functional.linear(
            test_inputs, cpu_state_dict[weight_fqn].to(self.device)
        )
        self.assertEqual(expected, compiled(test_inputs))

        # Without allow_h2d_copy, the same call must error out.
        with self.assertRaises(RuntimeError):
            compiled.load_constants(cpu_state_dict, check_full_update=True)

        # allow_h2d_copy + user_managed must be rejected up-front.
        with self.assertRaises(RuntimeError):
            compiled.load_constants(
                cpu_state_dict,
                check_full_update=True,
                user_managed=True,
                allow_h2d_copy=True,
            )

    def test_cond_share_predicate(self):
        # [FUNCTION DOC] Checks a model that uses the same predicate for two
        # successive conditionals.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying two conditionals sharing a predicate, each adding
            # one or two.
            def forward(self, predicate, x):
                # [FUNCTION DOC] HELPER: Applies the first conditional adding one or two to x, the
                # second adding one or two to that, and returns the result in a one-element tuple.
                y = torch.cond(
                    predicate,
                    lambda: x + 1,
                    lambda: x + 2,
                )

                z = torch.cond(
                    predicate,
                    lambda: y + 1,
                    lambda: y + 2,
                )
                return (z,)

        example_inputs = (
            torch.tensor([True]).to(self.device),
            torch.tensor([1, 2, 3]).to(self.device),
        )
        self.check_model(Model(), example_inputs)

    def test_cond_predicate_on_cpu(self):
        # [FUNCTION DOC] Compiles a model whose conditional predicate is a CPU buffer that gets set
        # to true after the first run, and checks the first call takes the false branch (result
        # 1.0) and the second takes the true branch (result 2.0).
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with a non-persistent CPU boolean buffer that is used as a
            # conditional predicate and set to true after use.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the false boolean buffer on CPU.
                super().__init__()
                self.register_buffer(
                    "is_cache_initialized",
                    torch.tensor([False], dtype=torch.bool, device="cpu"),
                    persistent=False,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Defines the two branches, applies the conditional on the
                # buffer, then fills the buffer with true in place and returns the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: adds one to x.
                    return x + 1.0

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: adds zero to x.
                    return x + 0.0

                out = torch.cond(
                    self.is_cache_initialized, true_fn, false_fn, operands=(x,)
                )
                self.is_cache_initialized.fill_(True)
                return out

        model = Model()
        example_inputs = (torch.tensor([1.0], device=self.device),)
        package_path = AOTIRunnerUtil.compile(model, example_inputs)
        optimized = torch._inductor.aoti_load_package(package_path)

        # First call: predicate is False
        result1 = optimized(*example_inputs)
        self.assertEqual(result1.item(), 1.0)

        # Second call: predicate is now True
        result2 = optimized(*example_inputs)
        self.assertEqual(result2.item(), 2.0)

    @unittest.skipIf(
        IS_FBCODE,
        "To enable after the C shim FC window ends",
    )
    def test_misaligned_input_1(self):
        # [FUNCTION DOC] Compiles with an aligned input and asserts the generated code has one
        # clone that preserves strides (a runtime alignment fix up), then calls the model with a
        # misaligned view and compares with eager. CUDA/XPU only; skipped in fbcode.
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("CUDA/XPU test only")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sin plus cos of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sin of x plus cos of x.
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        example_inputs = (arg,)
        model = Model()
        expected = model(*example_inputs)
        package_path = AOTIRunnerUtil.compile(model, example_inputs)
        optimized = torch._inductor.aoti_load_package(package_path)
        # If the model is compiled with aligned inputs, the generated
        # code will check inputs alignment at runtime
        self.code_check_count(
            model, example_inputs, "aoti_torch_clone_preserve_strides", 1
        )

        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        actual = optimized(misaligned_arg)
        torch.testing.assert_close(actual, expected)

    def test_misaligned_input_2(self):
        # [FUNCTION DOC] Compiles with an input that is already misaligned and asserts it matches
        # eager and the generated code has no alignment fix up clone. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("GPU test only")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sin plus cos of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sin of x plus cos of x.
                return x.sin() + x.cos()

        N = 64 * 64 * 64 + 64
        arg = torch.randn(N, device=self.device)
        misaligned_arg = torch.zeros(N + 1, device=self.device)
        misaligned_arg = misaligned_arg[1:]
        misaligned_arg.copy_(arg)
        example_inputs = (misaligned_arg,)

        model = Model()
        self.check_model(model, example_inputs)
        # If the model is already compiled with a misaligned input, the
        # generated code should NOT contain an alignment check for that input.
        self.code_check_count(
            model, example_inputs, "aoti_torch_clone_preserve_strides", 0
        )

    def test_autotuning_args_reuse(self):
        # [FUNCTION DOC] Checks that autotuning arguments are reused correctly when different
        # autotuned kernels (add and subtract) run over strided empty outputs with dynamic first
        # dims, using max-autotune. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running autotuned add kernels on two inputs and a subtract
            # kernel on one of the outputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Allocates strided empty outputs for x and y, runs the
                # add kernel on each and a subtract kernel on the first output, and returns
                # both outputs.
                x_out = torch.empty_strided(
                    (x.size()[0], x.size()[1]), (x.size()[1], 1), device=GPU_TYPE
                )
                x_out = torch.permute(x_out, [0, 1])
                add_kernel_autotuned[(4,)](x, x, x_out, 16)

                y_out = torch.empty_strided(
                    (y.size()[0], y.size()[1]), (y.size()[1], 1), device=GPU_TYPE
                )
                y_out = torch.permute(y_out, [0, 1])
                add_kernel_autotuned[(64,)](y, y, y_out, 64)

                sub_kernel_autotuned[(4,)](x, x, x_out, 16)

                return x_out, y_out

        example_inputs = (
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(8, 8, device=GPU_TYPE),
        )
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dim0_y = Dim("dim0_y", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}
        self.check_model(
            Model(),
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={"max_autotune": True},
        )

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    @unittest.skipIf(
        TEST_MPS and MACOS_VERSION < 14.0,
        "FFT operations are only supported on MacOS 14+",
    )
    def test_stft(self):
        # [FUNCTION DOC] Checks a short-time Fourier transform of a one-dimensional signal with a
        # Hann window and returns the squared magnitudes without the last frame. Skipped in fbcode
        # and on MPS before macOS 14.
        N_FFT = 400
        HOP_LENGTH = 160

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing squared STFT magnitudes with window size 400 and
            # hop 160.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates the Hann window, computes the complex STFT, drops
                # the last frame and returns the squared absolute value.
                window = torch.hann_window(N_FFT, device=x.device)
                stft = torch.stft(
                    x, N_FFT, HOP_LENGTH, window=window, return_complex=True
                )
                magnitudes = stft[..., :-1].abs() ** 2
                return magnitudes

        model = Model()
        example_inputs = (torch.randn(500, device=self.device),)
        self.check_model(model, example_inputs)

    def test_conv3d(self):
        # [FUNCTION DOC] Checks a convolution on a very large input (5160 by 5160 with 64 channels)
        # with dynamic spatial dims under max-autotune with the Triton conv backend and loose
        # tolerance. Skips without a big GPU or without enough device memory (32 GiB).
        if self.device != GPU_TYPE or not is_big_gpu():
            raise unittest.SkipTest("requires modern GPU to run max-autotune")

        if not _has_sufficient_memory(self.device, 2**35):
            raise unittest.SkipTest("insufficient memory")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the generic convolution op with stride, padding
            # and dilation all one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                convert_element_type_1271,
                convert_element_type_1272,
                convert_element_type_1273,
            ):
                # [FUNCTION DOC] HELPER: Returns the convolution of the input with the weight
                # and bias.
                return torch.ops.aten.convolution.default(
                    convert_element_type_1271,
                    convert_element_type_1272,
                    convert_element_type_1273,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                )

        example_inputs = (
            torch.randn(1, 64, 5160, 5160, device=self.device),
            torch.randn(3, 64, 3, 3, device=self.device),
            torch.randn(3, device=self.device),
        )
        dynamic_shapes = {
            "convert_element_type_1271": {
                3: torch.export.Dim.DYNAMIC,
                4: torch.export.Dim.DYNAMIC,
            },
            "convert_element_type_1272": None,
            "convert_element_type_1273": None,
        }
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_conv_backends": "TRITON",
            }
        ):
            self.check_model(
                Model(),
                example_inputs,
                atol=0.1,
                rtol=1e-3,
                dynamic_shapes=dynamic_shapes,
            )

    def test__int_mm(self):
        # [FUNCTION DOC] Checks the integer matrix multiply op on int8 inputs.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the integer matrix multiply.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the int8 matrix product of x and y.
                return torch._int_mm(x, y)

        example_inputs = (
            torch.randint(-10, 10, (64, 32), device=self.device, dtype=torch.int8),
            torch.randint(-10, 10, (32, 64), device=self.device, dtype=torch.int8),
        )
        self.check_model(Model(), example_inputs)

    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    def test_mm_out_dtype(self, input_dtype):
        # [FUNCTION DOC] Parametrized on half and bfloat16 inputs. Checks a matrix multiply with a
        # float32 output dtype. GPU only, not MPS.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doing a matrix multiply with output dtype float32.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and y in float32.
                return torch.mm(x, y, out_dtype=torch.float32)

        example_inputs = (
            torch.randn(64, 32, device=self.device, dtype=input_dtype),
            torch.randn(32, 64, device=self.device, dtype=input_dtype),
        )
        self.check_model(Model(), example_inputs)

    @parametrize("m", [32])
    @parametrize("n", [64])
    @parametrize("q_group", [32, 64])
    @parametrize("num_groups", [1, 2])
    def test__weight_int4pack_mm(self, m, n, q_group, num_groups):
        # [FUNCTION DOC] Parametrized on group size and group count. Checks the int4 packed weight
        # matmul with a bfloat16 activation and weights quantized in groups. GPU only, not MPS;
        # skipped on ROCm for unsupported architectures.
        if self.device != GPU_TYPE or self.device == "mps":
            raise unittest.SkipTest("requires GPU")

        if TEST_WITH_ROCM:
            if CDNA5OrLater():
                self.skipTest("int4 mm not yet implemented for gfx1250 (needs WMMA)")
            if not CDNA2OrLater():
                self.skipTest("_int4_mm is supported only for CDNA2 or later")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a packed int4 weight and scales and zeros,
            # applying the packed matmul.
            def __init__(self, weight, scale_and_zeros) -> None:
                # [FUNCTION DOC] HELPER: Stores the weight and the combined scales and
                # zeros tensor.
                super().__init__()
                self.weight = weight
                self.scale_and_zeros = scale_and_zeros

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the int4 packed matmul of a with the stored
                # weight, group size from the enclosing test and scales and zeros.
                return torch._weight_int4pack_mm(
                    a, self.weight, q_group, self.scale_and_zeros
                )

        def convert_weight_to_int4pack(b):
            # [FUNCTION DOC] HELPER: Quantizes a weight to four bits in groups, converts it to the
            # int4 packed layout and returns the packed weight and the scales and zeros.
            b_int32, b_scales_and_zeros = _group_quantize_tensor(
                b, n_bit=4, q_group_size=q_group
            )
            b_int4pack = torch._convert_weight_to_int4pack(b_int32, innerKTiles=2)
            return b_int4pack, b_scales_and_zeros

        k = q_group * num_groups
        a = torch.rand((m, k), device=self.device, dtype=torch.bfloat16)
        b = torch.rand((k, n), device=self.device, dtype=torch.bfloat16)
        b_int4pack, b_scales_and_zeros_f32 = convert_weight_to_int4pack(b)
        model = Model(b_int4pack, b_scales_and_zeros_f32)
        self.check_model(model, (a,))

    @parametrize("m", [32])
    @parametrize("n", [64])
    @parametrize("q_group", [32, 64])
    @parametrize("num_groups", [1, 2])
    def test__weight_int4pack_mm_with_scales_and_zeros(self, m, n, q_group, num_groups):
        # [FUNCTION DOC] Parametrized on group size and group count. Checks the int4 packed matmul
        # variant with separate scales and zeros tensors, with a custom group quantizer. Intel GPU
        # only; skipped on ROCm for unsupported architectures.
        if "xpu" not in self.device:
            raise unittest.SkipTest("requires Intel GPU")

        if TEST_WITH_ROCM:
            if CDNA5OrLater():
                self.skipTest("int4 mm not yet implemented for gfx1250 (needs WMMA)")
            if not CDNA2OrLater():
                self.skipTest("_int4_mm is supported only for CDNA2 or later")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a packed int4 weight, scales and zeros, applying
            # the packed matmul with separate scales and zeros.
            def __init__(self, weight, scale, zeros) -> None:
                # [FUNCTION DOC] HELPER: Stores the weight, scales and zeros.
                super().__init__()
                self.weight = weight
                self.scale = scale
                self.zeros = zeros

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the packed matmul of a with the stored weight,
                # group size from the enclosing test, scales and zeros.
                return torch._weight_int4pack_mm_with_scales_and_zeros(
                    a, self.weight, q_group, self.scale, self.zeros
                )

        def _group_quantize_tensor_xpu(w, n_bit=4, q_group_size=16):
            # [FUNCTION DOC] HELPER: Group-quantizes a 2D weight to the given bit width: transposes
            # it, validates group size and shape and absence of NaN (raising AssertionError
            # otherwise), computes per-group scales and integer zero points, quantizes, and (for
            # non-CPU) packs two values per byte. Returns quantized values, scales and zeros with
            # scales and zeros transposed contiguous.
            # w [k, n] = [32, 48]
            if w.dim() != 2:
                raise AssertionError(f"Expected 2D tensor, got {w.dim()}D")
            # w [n, k] = [48, 32]
            w = w.transpose(0, 1).contiguous()
            if q_group_size <= 1:
                raise AssertionError(f"Expected q_group_size > 1, got {q_group_size}")
            if w.shape[-1] % q_group_size != 0:
                raise AssertionError(
                    f"w.shape[-1] ({w.shape[-1]}) must be divisible by q_group_size ({q_group_size})"
                )

            # to_quant: [n * k / group_size, group_size]
            to_quant = w.reshape(-1, q_group_size)
            if torch.isnan(to_quant).sum() != 0:
                raise AssertionError("to_quant contains NaN values")

            max_val = to_quant.amax(dim=1, keepdim=True)
            min_val = to_quant.amin(dim=1, keepdim=True)
            max_int = 2**n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-6) / max_int
            if torch.isnan(scales).sum() != 0:
                raise AssertionError("scales contains NaN values")

            zeros = min_int - min_val.div(scales).round()
            zeros = torch.clamp(zeros, min_int, max_int)
            zeros = zeros.to(torch.int8)
            if torch.isnan(zeros).sum() != 0:
                raise AssertionError("zeros contains NaN values")

            out = to_quant.div(scales).add(zeros).round().clamp_(min_int, max_int)
            if torch.isnan(out).sum() != 0:
                raise AssertionError("out contains NaN values")

            # [n, k]
            out = out.to(dtype=torch.int32).reshape(w.shape)
            if out.device != torch.device("cpu"):
                out = (out[::, 1::2] << 4 | out[::, 0::2]).to(torch.uint8)

            # Scales and zeros for the same q-group should be contiguous, so we can
            # load as a 32-bit word
            scales = scales.view(w.shape[0], -1).transpose(0, 1).contiguous()
            zeros = zeros.view(w.shape[0], -1).transpose(0, 1).contiguous()

            return out, scales, zeros

        def convert_weight_to_int4pack(b):
            # [FUNCTION DOC] HELPER: Quantizes b with the XPU group quantizer, converts the result
            # to the int4 packed layout and returns the packed weight, scales and zeros.
            # b_uint8 [n, k //2]
            b_uint8, scales, zeros = _group_quantize_tensor_xpu(
                b, n_bit=4, q_group_size=q_group
            )
            # b_int4pack [k//8, n]
            b_int4pack = torch._convert_weight_to_int4pack(b_uint8, innerKTiles=2)

            return b_int4pack, scales, zeros

        k = q_group * num_groups
        a = torch.rand((m, k), device=self.device, dtype=torch.bfloat16)
        b = torch.rand((k, n), device=self.device, dtype=torch.bfloat16)
        b_int4pack, b_scales, zeros_int8 = convert_weight_to_int4pack(b)
        model = Model(b_int4pack, b_scales, zeros_int8)
        self.check_model(model, (a,))

    def test_assert_tensor_meta(self):
        # [FUNCTION DOC] Checks a model that asserts tensor metadata (int32 dtype) before adding
        # one, with implicit fallbacks disabled.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module asserting the dtype of its input then adding one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Asserts x is int32 via the metadata assert op and returns
                # x plus one in a one-element tuple.
                torch.ops.aten._assert_tensor_metadata.default(
                    x,
                    dtype=torch.int32,
                )
                return (x + 1,)

        example_inputs = (torch.tensor(1, dtype=torch.int32),)
        with config.patch(
            {
                "implicit_fallbacks": False,
            }
        ):
            self.check_model(
                Module(),
                example_inputs,
                atol=0.1,
                rtol=1e-3,
            )

    @runOnRocm
    @requires_autotune_at_compile_time
    def test_rocm_triton_autotuning(self):
        # [FUNCTION DOC] On ROCm only, compiles with the export aot_compile a model launching a
        # matmul kernel with an unusual autotune config, with autotuning on sample inputs and the
        # test class stack allocation and minimal arrayref settings. Compile-only. Needs autotune
        # at compile time.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the oddly configured matmul kernel with an
            # absolute-valued scalar argument.
            def forward(self, x, y, m):
                # [FUNCTION DOC] HELPER: Allocates a float32 output, defines a grid from the block
                # sizes, and launches the matmul kernel with x, y, the output, absolute m and the
                # dims; returns the output.
                _M, K = x.shape
                K, N = y.shape
                M = torch.abs(m)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4096], dtype=torch.int32, device=self.device)

        with (
            torch.no_grad(),
            config.patch(
                {
                    "triton.autotune_with_sample_inputs": True,
                    "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                    "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
                }
            ),
        ):
            torch._export.aot_compile(Model(), (x, y, m))

    @requires_autotune_at_compile_time
    def test_triton_autotuning(self):
        # [FUNCTION DOC] Compiles the oddly configured matmul kernel model with autotuning on
        # sample inputs, and asserts the first grid size found in the generated code is one of the
        # expected grids from the kernel's configs. GPU only; needs autotune at compile time.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the oddly configured matmul kernel with an
            # absolute-valued scalar argument.
            def forward(self, x, y, m):
                # [FUNCTION DOC] HELPER: Allocates a float32 output, defines a grid from the block
                # sizes, and launches the matmul kernel with x, y, the output, absolute m and the
                # dims; returns the output.
                _M, K = x.shape
                K, N = y.shape
                M = torch.abs(m)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4096], dtype=torch.int32, device=self.device)

        with config.patch("triton.autotune_with_sample_inputs", True):
            # Use dynamic grid checking (portable across different configs
            # and Triton versions)
            _, src_code = run_and_get_cpp_code(
                torch._export.aot_compile, Model(), (x, y, m)
            )
            actual_grid, expected_grids = get_triton_grid_info(
                strange_config_matmul_kernel, 4096 * 2046, src_code
            )
            self.assertTrue(
                actual_grid is not None, "Could not find grid_0 in generated code"
            )
            self.assertIn(
                actual_grid,
                expected_grids,
                lambda msg: f"{msg}\ngrid_0={actual_grid} not in expected {expected_grids} from kernel configs",
            )

    @requires_autotune_at_compile_time
    def test_triton_mutated_autotuning(self):
        # [FUNCTION DOC] Like the autotuning grid test, but the scalar argument is produced by
        # another triton kernel that mutates a buffer, to test autotuning with mutated arguments.
        # Asserts the found grid is among the expected grids. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Triton kernel: for each program, loads one element within bounds
        # from X, adds one and stores it to Y.
        @triton.jit
        def add_one_kernel(X, Y, N):
            pid = tl.program_id(axis=0)
            block_start = pid
            offsets = block_start + tl.arange(0, 1)

            x = tl.load(X + offsets, mask=offsets < N)
            y = x + 1
            tl.store(Y + offsets, y, mask=offsets < N)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a scalar with the add-one kernel and passing it
            # to the oddly configured matmul kernel.
            def forward(self, x, y, m):
                # [FUNCTION DOC] HELPER: Runs the add-one kernel on m into a one-element
                # buffer, then launches the matmul kernel with that buffer as its scalar, and
                # returns the output.
                _M, K = x.shape
                K, N = y.shape
                M = torch.empty((1), device=x.device, dtype=torch.int32)
                add_one_kernel[(1,)](m, M, 1)
                out = torch.empty((_M, N), device=x.device, dtype=torch.float32)
                grid = lambda META: (  # noqa: E731
                    triton.cdiv(
                        4096 * 2046, META["BLOCK_SIZE_M"] * META["BLOCK_SIZE_N"]
                    ),
                )
                strange_config_matmul_kernel[grid](
                    x,
                    y,
                    out,
                    M,
                    N,
                    K,
                )
                return out

        x = torch.randn(4096, 1024, device=self.device)
        y = torch.randn(1024, 2048, device=self.device)
        m = torch.tensor([4095], dtype=torch.int32, device=self.device)

        with config.patch("triton.autotune_with_sample_inputs", True):
            # Use dynamic grid checking (portable across different configs
            # and Triton versions)
            _, src_code = run_and_get_cpp_code(
                torch._export.aot_compile, Model(), (x, y, m)
            )
            actual_grid, expected_grids = get_triton_grid_info(
                strange_config_matmul_kernel, 4096 * 2046, src_code
            )
            self.assertTrue(
                actual_grid is not None, "Could not find grid_0 in generated code"
            )
            self.assertIn(
                actual_grid,
                expected_grids,
                lambda msg: f"{msg}\ngrid_0={actual_grid} not in expected {expected_grids} from kernel configs",
            )

    @patch.dict(os.environ, {"TRITON_DEBUG": "1"})
    @requires_autotune_at_compile_time
    def test_triton_dynamic_launcher_grid(self):
        # [FUNCTION DOC] With triton debug set, checks an autotuned kernel with device assertion
        # whose launch grid depends on an unbacked scalar, with sample input autotuning and dynamic
        # first dim. GPU only; needs autotune at compile time.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Autotuned triton kernel (two block-size configs keyed on the
        # element count): asserts the block start is below the element count, then loads, adds one
        # and stores.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["numel"],
        )
        @triton.jit
        def add_one_kernel(X, Y, numel, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            tl.device_assert(block_start < numel)
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            x = tl.load(X + offsets)
            y = x + 1
            tl.store(Y + offsets, y)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module launching the add-one kernel with a grid from a scalar
            # taken from a tensor.
            def forward(self, x, value):
                # [FUNCTION DOC] HELPER: Reads the scalar from value, allocates a zeros output,
                # launches the kernel with a grid based on the scalar and returns the output.
                numel = value.item()
                out = torch.zeros_like(x, dtype=torch.float16)

                grid = lambda META: (  # noqa: E731
                    triton.cdiv(numel, META["BLOCK_SIZE"]),
                )
                add_one_kernel[grid](x, out, numel)

                return out

        example_inputs = (
            torch.randn(1024, device=self.device),
            torch.tensor([1024], dtype=torch.int32, device=self.device),
        )

        with config.patch("triton.autotune_with_sample_inputs", True):
            dim0_x = Dim("dim0_x", min=2, max=8192)
            dynamic_shapes = {"x": {0: dim0_x}, "value": {0: Dim.AUTO}}
            self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    @patch.dict(os.environ, {"TRITON_DEBUG": "1"})
    @requires_autotune_at_compile_time
    def test_triton_dynamic_launcher_grid_infer_from_tensor(self):
        # [FUNCTION DOC] With triton debug set, like the previous test but the element count is a
        # product of a tensor dim and an unbacked scalar, so the launcher grid must be inferred
        # from tensor shapes. GPU only; needs autotune at compile time.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        # [FUNCTION DOC] HELPER: Autotuned triton kernel (two block-size configs keyed on the
        # element count): asserts the block start is below the element count, then loads, adds one
        # and stores.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["numel"],
        )
        @triton.jit
        def add_one_kernel(X, Y, numel, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            tl.device_assert(block_start < numel)

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + offsets)
            y = x + 1
            tl.store(Y + offsets, y)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating x by a scalar count and launching the add-one
            # kernel over the result.
            def forward(self, x, dim_D):
                # [FUNCTION DOC] HELPER: Computes the element count from the second dim and the
                # scalar, repeats x rows by the scalar, allocates a zeros output, launches the
                # kernel and returns the output.
                numel = x.shape[1] * dim_D.item()
                x = x.repeat(dim_D, 1)
                out = torch.zeros_like(x, dtype=torch.float16)

                grid = lambda META: (  # noqa: E731
                    triton.cdiv(numel, META["BLOCK_SIZE"]),
                )
                add_one_kernel[grid](x, out, numel)

                return out

        example_inputs = (
            torch.randn(1, 1024, device=self.device),
            torch.tensor([2], dtype=torch.int32, device=self.device),
        )

        with config.patch("triton.autotune_with_sample_inputs", True):
            dim1_x = Dim("dim1_x", min=2, max=8192)
            dynamic_shapes = {"x": {0: Dim.AUTO, 1: dim1_x}, "dim_D": {0: Dim.AUTO}}
            self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_composed_dynamic_size(self):
        # [FUNCTION DOC] Checks a model with a dynamic dimension constrained to be even (twice a
        # base dimension).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        example_inputs = (torch.randn(10, device=self.device),)
        dim = torch.export.Dim("dim_0")
        dim_even = 2 * dim
        dynamic_shapes = {
            "x": {0: dim_even},
        }
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_composed_dynamic_size_integer_symbol_recovery(self):
        # [FUNCTION DOC] Checks a dynamic dim that is a multiple of 1496 and an unflatten with an
        # inferred leading dim: asserts the generated code uses integer floor division and no
        # floating reciprocal of 1496, and compares outputs with eager for several batch sizes.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unflattening dim 0 into inferred by 1496 and summing over
            # dim 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unflattens and sums over the new dim.
                return x.unflatten(0, (-1, 1496)).sum(dim=1)

        model = Model().to(self.device)
        dynamic_shapes = {
            "x": {0: 1496 * Dim("batch", min=1, max=8)},
        }
        compile_inputs = (torch.ones(4 * 1496, 2, device=self.device),)
        package_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile,
            model,
            compile_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        self.assertIn("c10::div_floor_integer", code)
        self.assertNotIn("(1.0/1496.0)", code)

        optimized = torch._inductor.aoti_load_package(package_path)
        for batch in (1, 2, 3, 4, 8):
            inputs = (torch.ones(batch * 1496, 2, device=self.device),)
            self.assertEqual(optimized(*inputs), model(*inputs))

    def test_boolean_indexing(self):
        # [FUNCTION DOC] Checks boolean mask indexing on two tensors with the same mask followed by
        # concatenation, with several dynamic dims allowing zero sizes.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module masking two inputs with the same mask and concatenating
            # each with another tensor.
            def forward(self, x, y, z, x1, z1):
                # [FUNCTION DOC] HELPER: Selects rows of x and x1 by y, concatenates them with z
                # and z1 along dim 1 and returns both results.
                a = x[y]
                a1 = x1[y]
                b = torch.cat([a, z], dim=1)
                b1 = torch.cat([a1, z1], dim=1)
                return b, b1

        x = torch.randn(3, 5, device=self.device)
        y = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        z = torch.randn(2, 4, device=self.device)
        x1 = torch.randn(3, 5, device=self.device)
        z1 = torch.randn(2, 4, device=self.device)

        example_inputs = (x, y, z, x1, z1)
        s0 = Dim("s0", min=0, max=10240)
        s1 = Dim("s1", min=0, max=10240)
        s2 = Dim("s2", min=0, max=10240)
        s3 = Dim("s3", min=0, max=10240)
        dynamic_shapes = {
            "x": {0: s0, 1: s1},
            "y": {0: s0},
            "z": {0: s2, 1: s3},
            "x1": {0: s0, 1: s1},
            "z1": {0: s2, 1: s3},
        }
        self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)

    def test_sym_expr_indexing(self):
        # [FUNCTION DOC] Checks an embedding lookup whose index comes from stacking tensors with a
        # range and selecting the first, exercising symbolic expression indexing. CUDA/XPU only.
        if self.device not in ["cuda", "xpu"]:
            raise unittest.SkipTest("requires CUDA/XPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking six one-element index tensors (one built from a
            # range), selecting the first and looking it up in an embedding weight.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(
                self,
                episode_builder_position_encoding_observations_weight,
                add_15,
                add_16,
                add_17,
                add_18,
                add_13,
            ):
                # [FUNCTION DOC] HELPER: Builds the range plus 198, stacks it with the given index
                # tensors, selects the first entry (add_13) and returns the embedding lookup in a
                # one-element tuple.
                device = add_13.device
                arange_1 = torch.ops.aten.arange.start(
                    180,
                    181,
                    device=device,
                    pin_memory=False,
                )
                add_14 = torch.ops.aten.add.Tensor(arange_1, 198)
                arange_1 = None
                stack_1 = torch.ops.aten.stack.default(
                    [add_13, add_14, add_15, add_16, add_17, add_18]
                )
                add_13 = add_14 = add_15 = add_16 = add_17 = add_18 = None
                select_13 = torch.ops.aten.select.int(stack_1, 0, 0)
                stack_1 = None
                embedding_11 = torch.ops.aten.embedding.default(
                    episode_builder_position_encoding_observations_weight, select_13
                )
                episode_builder_position_encoding_observations_weight = select_13 = None
                return (embedding_11,)

        # Embedding weight: vocab_size x emb_dim
        episode_builder_position_encoding_observations_weight = torch.randn(
            100, 16, device=self.device
        )

        # These six must all be 1-D (shape [1]) and same dtype; use Long for embedding indices
        add_13 = torch.tensor(
            [7], dtype=torch.long, device=self.device
        )  # this one is used as the index
        add_15 = torch.tensor([5], dtype=torch.long, device=self.device)
        add_16 = torch.tensor([6], dtype=torch.long, device=self.device)
        add_17 = torch.tensor([7], dtype=torch.long, device=self.device)
        add_18 = torch.tensor([8], dtype=torch.long, device=self.device)

        # Instantiate and run
        m = Repro().to(self.device)

        example_inputs = (
            episode_builder_position_encoding_observations_weight,
            add_15,
            add_16,
            add_17,
            add_18,
            add_13,
        )
        self.check_model(m, example_inputs)

    def test_with_cudagraphs(self):
        # [FUNCTION DOC] Exports and AOTI-compiles a linear model with max-autotune, loads it in
        # single threaded mode and wraps it in a CUDA graph helper (keyed by input shapes), warms
        # up and checks the CUDA graph replay output equals eager. CUDA only.
        if self.device != "cuda":
            raise unittest.SkipTest("requires CUDA")

        # define CUDAGraph handling wrapper (only works with kwargs for simplicity)
        def cudagraph(f):
            # [FUNCTION DOC] HELPER: Decorator factory: returns a keyword-only wrapper that caches
            # a captured CUDA graph per tuple of tensor input shapes.
            _graphs = {}

            def f_(**kwargs):
                # [FUNCTION DOC] HELPER: Wrapper: computes a key from sorted tensor kwarg shapes;
                # for a known key it replays the cached graph. Otherwise it clones the inputs,
                # warms up the function, captures a graph, caches the wrapper, and runs it.
                key = hash(
                    tuple(
                        tuple(kwargs[a].shape)
                        for a in sorted(kwargs.keys())
                        if isinstance(kwargs[a], torch.Tensor)
                    )
                )
                if key in _graphs:
                    wrapped, *_ = _graphs[key]
                    return wrapped(**kwargs)
                g = torch.cuda.CUDAGraph()
                in_tensors = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                f(**in_tensors)  # stream warmup
                with torch.cuda.graph(g):
                    out_tensors = f(**in_tensors)

                def wrapped(**kwargs):
                    # [FUNCTION DOC] HELPER: Copies new inputs into the captured input buffers,
                    # replays the graph and returns clones of the output tensors. Raises ValueError
                    # if the output is not a tensor, list or tuple.
                    for key in kwargs:
                        in_tensors[key].copy_(kwargs[key])
                    g.replay()
                    if isinstance(out_tensors, torch.Tensor):
                        return out_tensors.clone()
                    elif isinstance(out_tensors, (list, tuple)):
                        return type(out_tensors)(o.clone() for o in out_tensors)
                    raise ValueError("unsupported output type encountered")

                _graphs[key] = (wrapped, g, in_tensors, out_tensors)
                return wrapped(**kwargs)

            return f_

        # define a simple model
        model = torch.nn.Linear(10, 20).to(device=self.device)

        # export + AOTI
        model_kwargs = {
            "input": torch.randn(3, 10, device=self.device),
        }
        ep = torch.export.export(model, args=(), kwargs=model_kwargs, strict=True)

        optimized = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                ep,
                inductor_configs={"max_autotune": True},
            ),
            # NB: this flag avoids a CUDAGraph + AOTI runtime multi-threading conflict
            # "Error: operation not permitted when stream is capturing"
            run_single_threaded=True,
        )

        # enable CUDAGraphs
        optimized = cudagraph(optimized)

        # warmup -> run with CUDAGraphs
        for _ in range(3):
            optimized(**model_kwargs)

        # compare against eager
        self.assertEqual(optimized(**model_kwargs), model(**model_kwargs))

    def test_custom_op_in_subgraph(self):
        # [FUNCTION DOC] Defines two custom ops (add one and add two) in a scoped library and
        # checks a model that selects one with a conditional on the input size, over two input
        # sizes with a dynamic dim.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_add1",
                "(Tensor a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_add1", "CompositeExplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo_add1", lib=lib)
            def foo_add1_impl(a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Implementation and fake implementation of the add-one op:
                # returns a plus one.
                return a + 1

            torch.library.define(
                "mylib::foo_add2",
                "(Tensor a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_add2", "CompositeExplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo_add2", lib=lib)
            def foo_add2_impl(a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Implementation and fake implementation of the add-two op:
                # returns a plus two.
                return a + 2

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a conditional choosing between the two custom ops
                # by the size of x.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the first op if the first dim is below five,
                    # else the second.
                    return torch.cond(
                        x.shape[0] < 5,
                        torch.ops.mylib.foo_add1,
                        torch.ops.mylib.foo_add2,
                        (x,),
                    )

            list_example_inputs = [
                (torch.ones(6, device=self.device),),
                (torch.ones(3, device=self.device),),
            ]
            self.check_model_with_multiple_inputs(
                M(), list_example_inputs, dynamic_shapes=({0: Dim.DYNAMIC},)
            )

    def test_clamp_decomposition(self):
        # [FUNCTION DOC] Checks clamp with a float minimum on an integer input (result must be
        # float32) and with an integer minimum (result stays integer), using two models.
        class Model1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping with a float minimum of 1.5.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x clamped from below at 1.5.
                return x.clamp(min=1.5)

        class Model2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping with an integer minimum of 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x clamped from below at 2.
                return x.clamp(min=2)

        x = torch.randint(4, (4,))

        # the output should have float32 type, not int
        self.check_model(Model1(), (x,))
        # the output should have int type
        self.check_model(Model2(), (x,))

    @unittest.skipIf(not IS_BIG_GPU, "Test requires large GPU memory")
    def test_upper_bound_i64(self):
        # [FUNCTION DOC] Exports an int8 add with a dynamic first dim up to 2 to the 33rd, compiles
        # and loads it, checks a small input against eager, then runs with 3 times 2 to the 30th
        # elements (beyond 32-bit index range) without checking values, to ensure no illegal memory
        # access. Skipped unless on a big GPU.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inp = (
            torch.randint(0, 100, (2**18,), device=self.device, dtype=torch.int8),
            torch.tensor([4], device=self.device, dtype=torch.int8),
        )
        ep = torch.export.export(
            Model(),
            inp,
            dynamic_shapes=({0: Dim("d", min=0, max=2**33)}, {0: Dim.STATIC}),
        )
        so_path = torch._inductor.aot_compile(ep.module(), inp)
        m = torch._export.aot_load(so_path, self.device)

        self.assertEqual(Model()(*inp), m(*inp))
        del inp

        inp = (
            torch.randint(0, 100, (3 * 2**30,), device=self.device, dtype=torch.int8),
            torch.tensor([4], device=self.device, dtype=torch.int8),
        )
        # don't check the accuracy of the result to reduce memory usage
        # this test is mostly checking to ensure there's no IMA.
        m(*inp)

    def test_using_model_name_for_files(self):
        # [FUNCTION DOC] Compiles a package with a custom model name for generated files and
        # asserts the zip archive contains the wrapper source, kernel source and wrapper library
        # under that name, then loads and compares with eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a 10x10 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model().to(self.device)
        with torch.no_grad():
            package_path: str = AOTIRunnerUtil.compile(
                model,
                example_inputs,
                inductor_configs={
                    "aot_inductor.model_name_for_generated_files": "test_model"
                },
            )

        with zipfile.ZipFile(package_path, "r") as zip_ref:
            all_files = zip_ref.namelist()
            base_dir = "test_model.wrapper/data/aotinductor/model/test_model"
            ext_type = get_module_ext_type()
            self.assertTrue(f"{base_dir}.wrapper.cpp" in all_files)
            self.assertTrue(f"{base_dir}.kernel.cpp" in all_files)
            self.assertTrue(f"{base_dir}.wrapper.{ext_type}" in all_files)

        aot_inductor_module = torch._inductor.aoti_load_package(package_path)
        self.assertEqual(aot_inductor_module(*example_inputs), model(*example_inputs))

    @skipIfRocmArch(NAVI_ARCH)
    def test_copy_non_blocking_is_pinned(self):
        # [FUNCTION DOC] Checks that non-blocking device-to-CPU copies in compiled code use pinned
        # memory: the generated code mentions pinned, and outputs match eager. Skipped on CPU, MPS
        # and Navi ROCm.
        if self.device == "cpu" or self.device == "mps":
            raise unittest.SkipTest("only matters for device-to-cpu copy")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module copying two inputs to CPU non-blocking, synchronizing on
            # an event and concatenating.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Copies a and b to CPU without blocking, records and waits
                # on an event, and returns the concatenation.
                a_cpu = a.to(device="cpu", non_blocking=True)
                b_cpu = b.to(device="cpu", non_blocking=True)
                a_to_cpu_event = torch.Event()
                a_to_cpu_event.record()
                a_to_cpu_event.synchronize()
                return torch.cat([a_cpu, b_cpu])

        model = Model()
        a = torch.randn(2, 2, device=self.device)
        b = torch.randn(2, 2, device=self.device)
        example_inputs = (a, b)
        outputs = model(*example_inputs)
        package_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, model, example_inputs
        )
        FileCheck().check("pinned").run(code)
        model_aoti = torch._inductor.aoti_load_package(package_path)
        outputs_aoti = model_aoti(*example_inputs)

        self.assertEqual(outputs, outputs_aoti)

    def test_pad_non_zero_memory_leak(self):
        # [FUNCTION DOC] Compiles a model that pads with a nonzero constant before a self matmul,
        # checks outputs with loose tolerance, and checks the generated code re-strides the buffer
        # and wraps it with an owning handle (to avoid a memory leak). GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("test is only for GPU_TYPE")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding one, padding the last dim by one with a large
            # constant and multiplying the result with itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one, pads, and returns the matrix product of the
                # result with itself.
                x = x + 1
                x = torch.ops.aten.constant_pad_nd(x, (0, 1, 0, 0), 12345.0)

                return x @ x

        model = Model()
        example_inputs = (torch.randn(2048, 2047, device=self.device),)
        package_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.compile, model, example_inputs
        )
        outputs = model(*example_inputs)
        model_aoti = torch._inductor.aoti_load_package(package_path)
        outputs_aoti = model_aoti(*example_inputs)

        self.assertEqual(outputs, outputs_aoti, atol=1e-2, rtol=1e-2)

        FileCheck().check_regex(
            r"aoti_torch_as_strided\(buf0_handle, .*, &buf0_handle_restrided\)"
        ).check("wrap_with_raii_handle_if_needed(buf0_handle);").check(
            "RAIIAtenTensorHandle buf0(buf0_handle_restrided);"
        ).run(code)

    @unittest.skipIf(
        IS_FBCODE,
        "different behavior in fbcode",
    )
    def test_codegen_int_array_var_fix_memory_leak(self):
        # [FUNCTION DOC] Regression test for issue 167630: compiles and packages an MLP three times
        # and asserts accelerator memory allocated after the second and third rounds is equal (no
        # leak from integer array variable codegen). CUDA/XPU only; skipped in fbcode.
        """
        Fix https://github.com/pytorch/pytorch/issues/167630
        """
        if self.device not in ("cuda", "xpu"):
            raise unittest.SkipTest("test is only for cuda or xpu")

        def make_mlp(in_dim=128, hidden=256, out_dim=64, depth=3):
            # [FUNCTION DOC] HELPER: Builds a sequential MLP with the given depth of linear plus
            # ReLU hidden layers and a final linear output layer.
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            layers += [nn.Linear(d, out_dim)]
            return nn.Sequential(*layers)

        batch = 32
        in_dim = 2048
        hidden = 512
        out_dim = 10
        depth = 6

        import gc

        allocated_memory = []
        for _ in range(3):
            torch.accelerator.reset_peak_memory_stats()

            model = make_mlp(in_dim, hidden, out_dim, depth).to(self.device)
            example_inputs = (torch.randn(batch, in_dim, device=self.device),)
            ep = torch.export.export(
                model,
                example_inputs,
            )
            torch._inductor.aoti_compile_and_package(ep)

            del model, example_inputs, ep
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            gc.collect()
            allocated_memory.append(torch.accelerator.memory_allocated())

        self.assertTrue(allocated_memory[1] == allocated_memory[2])

    @unittest.skipIf(IS_MACOS, "might have no readelf on Mac")
    def test_libtorch_free_so(self):
        # [FUNCTION DOC] Compiles a package with libtorch linking disabled, extracts it, and for
        # every shared library inspects its dynamic section with readelf, asserting none of the
        # needed libraries is a torch or c10 library. Skipped on macOS.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a 10x10 linear layer output to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the linear of y.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )

        model = Model().to(self.device)
        ep = torch.export.export(model, example_inputs)

        package_path = torch._inductor.aoti_compile_and_package(
            ep,
            inductor_configs={
                "aot_inductor.link_libtorch": False,
            },
        )

        torch_libs = {
            "libtorch.so",
            "libc10.so",
            "libtorch_cuda.so",
            "libc10_cuda.so",
            "libtorch_cpu.so",
            "libtorch_xpu.so",
            "libc10_xpu.so",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Unpack
            with zipfile.ZipFile(package_path, "r") as zf:
                zf.extractall(tmpdir)

            so_files = list(pathlib.Path(tmpdir).rglob("*.so"))
            self.assertTrue(len(so_files) > 0)

            for so_file in so_files:
                so_copy = pathlib.Path(tmpdir) / f"{so_file.name}.checkcopy"
                so_copy.write_bytes(so_file.read_bytes())

                result = subprocess.run(
                    ["readelf", "-d", str(so_copy)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.splitlines():
                    if "NEEDED" in line:
                        for lib in torch_libs:
                            self.assertTrue(lib not in line)

    def test_unbounded_expr_substitutions(self):
        # [FUNCTION DOC] Checks a model where two unbacked/backed scalars are tied by a runtime
        # equality check involving a max expression, used as an expand size; the legacy compiled
        # library output must match eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding x to a size given by the larger of a scalar and
            # 15, with a check tying it to another scalar, then matmul with y.
            def forward(self, x, y, a, b):
                # [FUNCTION DOC] HELPER: Reads two scalars from a and b, computes the max with 15,
                # adds the equality check, expands and clones x and returns the matmul with y.
                u0, s0 = a.item(), b.item()
                u_max = max(u0, 15)
                # construct the equality rule Max(15, u0) == s0 * Max(15, u0)
                torch._check(u_max == s0 * u_max)
                # size x - [Max(u0, 15), 64]
                x = x.expand(u_max, *x.shape).clone()
                return x @ y

        model = Model()

        example_inputs = (
            torch.randn((64,), dtype=torch.bfloat16, device=self.device),
            torch.randn((64, 16), dtype=torch.bfloat16, device=self.device),
            torch.tensor(19, device=self.device),
            torch.tensor(1, device=self.device),
        )
        torch._dynamo.mark_dynamic(example_inputs[-1], 0)

        so_path, code = run_and_get_cpp_code(
            AOTIRunnerUtil.legacy_compile, model, example_inputs
        )

        compiled = AOTIRunnerUtil.legacy_load(self.device, so_path)
        compiled_outputs = compiled(*example_inputs)

        eager_outputs = model(*example_inputs)
        torch.testing.assert_close(eager_outputs, compiled_outputs)

    @requires_gpu
    def test_mixed_device_1(self):
        # [FUNCTION DOC] Checks a model with CPU buffers and GPU inputs: scatter-add on CPU, move
        # to GPU, add to the vector, then matmul. The model is not moved to the device. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Mixed-device test requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CPU index and source buffers that builds a vector on
            # CPU and combines it with GPU inputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers two CPU int64 buffers.
                super().__init__()
                # Buffers are on CPU
                self.register_buffer(
                    "index", torch.tensor([1, 4, 1, 7], device="cpu", dtype=torch.int64)
                )
                self.register_buffer(
                    "src", torch.ones(4, device="cpu", dtype=torch.int64)
                )

            def forward(self, matrix, vector):
                # [FUNCTION DOC] HELPER: Scatter-adds the source into zeros on CPU, moves the
                # result to the GPU as the vector dtype, adds it to vector and returns the matrix
                # times that.
                # Inputs are on CUDA
                # 1. Operation on CPU tensors
                z = torch.zeros((vector.shape[0],), device="cpu", dtype=torch.int64)
                scatter_result = z.scatter_add(0, self.index, self.src)

                # 2. Move result to CUDA and continue on CUDA
                v = vector + scatter_result.to(vector.dtype).to(GPU_TYPE)
                return torch.matmul(matrix, v)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, device=self.device),
        )
        self.check_model(Model(), example_inputs, move_model_to_device=False)

    @requires_gpu
    def test_mixed_device_zero_size_constant(self):
        # [FUNCTION DOC] Checks a model with a zero-size GPU buffer plus CPU and GPU buffers,
        # verifying offset tracking of constants is correct. The model is not moved to the device.
        # GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("Mixed-device test requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty GPU buffer, CPU indices, GPU weights and
            # GPU bias.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the four buffers.
                super().__init__()
                # Zero-size CUDA buffer
                self.register_buffer(
                    "empty_buffer",
                    torch.empty(0, device=GPU_TYPE, dtype=torch.float32),
                )
                self.register_buffer(
                    "cpu_indices",
                    torch.tensor([0, 1, 2, 3], device="cpu", dtype=torch.int64),
                )
                self.register_buffer(
                    "cuda_weights",
                    torch.tensor(
                        [1.0, 2.0, 3.0, 4.0], device=GPU_TYPE, dtype=torch.float32
                    ),
                )
                # Another CUDA buffer to verify offset tracking is correct
                self.register_buffer(
                    "cuda_bias",
                    torch.tensor(
                        [0.5, 0.5, 0.5, 0.5], device=GPU_TYPE, dtype=torch.float32
                    ),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Moves the indices to the GPU, gathers the weights and
                # returns x times them plus the bias.
                idx_cuda = self.cpu_indices.to(GPU_TYPE)
                weights = self.cuda_weights[idx_cuda]
                return x * weights + self.cuda_bias

        example_inputs = (torch.randn(4, device=self.device),)
        self.check_model(Model(), example_inputs, move_model_to_device=False)

    @requires_gpu
    @requires_autotune_at_compile_time
    def test_constant_int_kernel_input(self):
        # [FUNCTION DOC] Runs a model creating random tensors sized by an unbacked scalar, using
        # sample input autotuning, via the runner utility. No numerics comparison. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module generating random tensors of an unbacked length and
            # combining them with a conditional select.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the length from x, builds a zeros tensor, two
                # random perturbations and a random mask, and returns the select between their
                # sum and difference.
                u0 = x.item()
                device = x.device

                inp = torch.zeros(u0, device=device)
                inp_rand0 = inp + torch.rand(u0, device=device)
                inp_rand1 = inp_rand0 - torch.rand(u0, device=device)
                return torch.where(
                    torch.rand(u0, device=device) > 0.5,
                    inp_rand0 + inp_rand1,
                    inp_rand0 - inp_rand1,
                )

        example_inputs = (torch.tensor(6, dtype=torch.int64, device=self.device),)

        with config.patch("triton.autotune_with_sample_inputs", True):
            AOTIRunnerUtil.run(Model(), example_inputs)

    @unittest.skipIf(
        IS_FBCODE, "Subprocess spawning doesn't work in fbcode Buck environment"
    )
    def test_aoti_load_package_in_fresh_subprocess(self):
        # [FUNCTION DOC] Compiles and packages a simple model, then loads it in a fresh Python
        # subprocess and asserts that the process exits with status zero. Skipped in fbcode.
        """
        Test that loading an AOTI package in a fresh subprocess works correctly.
        This catches initialization bugs that may not appear when loading in the
        same process where the package was compiled.
        """

        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        model = SimpleModel().to(self.device)
        example_input = (torch.randn(2, 10, device=self.device),)
        exported = torch.export.export(model, example_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported, package_path=str(model_path)
            )

            # Load in a fresh subprocess to reproduce potential bugs
            script = f"""
import torch

torch._inductor.aoti_load_package("{model_path}")
"""
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True
            )
            self.assertEqual(
                result.returncode,
                0,
                lambda msg: f"{msg}\nFailed to load package in subprocess: {result.stdout + result.stderr}",
            )

    @unittest.skipIf(
        IS_FBCODE, "Subprocess spawning doesn't work in fbcode Buck environment"
    )
    def test_python_custom_op_load_package_error_in_fresh_subprocess(self):
        # [FUNCTION DOC] Packages a model using a Python-defined custom op, then loads the package
        # in a fresh subprocess and asserts failure with an error message saying the schema could
        # not be found and that Python custom ops cannot run in AOTI without Python. CPU only;
        # skipped in fbcode.
        if self.device != "cpu":
            raise unittest.SkipTest("Only needs one no-Python custom op coverage")

        @torch.library.custom_op("aoti_python_custom_ops::custom_add", mutates_args=())
        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body: returns x plus y.
            return x + y

        @custom_add.register_fake
        def _(x, y):
            # [FUNCTION DOC] HELPER: Fake implementation of the custom op: returns x plus y.
            return x + y

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom op.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the custom op applied to x and y.
                return torch.ops.aoti_python_custom_ops.custom_add(x, y)

        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        exported = torch.export.export(Model(), example_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported, package_path=str(model_path)
            )

            script = f"""
import torch

torch._inductor.aoti_load_package("{model_path}")
"""
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Could not find schema for aoti_python_custom_ops::custom_add. "
                "Custom operators created via torch.library.custom_op cannot be run "
                "via AOTI without Python environment. Consider using C++ "
                "TORCH_LIBRARY APIs. See the Custom Operators Landing Page "
                "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html "
                "for more details.",
                result.stdout + result.stderr,
            )

    @unittest.skip(
        "Skip this test, only for local test. SIGFPE is produced when viewing "
        "empty tensor with dim=0. See D86125095 for model-side workaround."
    )
    def test_view_zero_dim_empty_tensor(self):
        # [FUNCTION DOC] Regression test (always skipped, since it would crash) for viewing an
        # empty tensor whose view size involves a division by a backed size that can be zero.
        # Compiles with unbacked and zero-allowed backed dims, checks the wrapper source has
        # division-by-zero guards, and then runs with empty inputs expecting a RuntimeError and the
        # guard message in the captured standard error stream (redirected through file descriptors
        # and restored afterwards).
        """
        Regression test for viewing an empty tensor with dim=0 in AOTInductor.

        When calling .view(s55, -1, 52) where:
        - The tensor being viewed has size u1 (unbacked symint from .item())
        - s55 is a backed symint from a DIFFERENT tensor's .size(0)
        - s55 can be 0 at runtime

        Previously, the AOTInductor C++ runtime crashed with SIGFPE because:
        - The -1 computation is: (u1*520) / (s55*52)
        - When s55=0, this is division by zero

        Now, we expect a proper RuntimeError to be raised with a meaningful
        error message instead of crashing the process.

        Key pattern from exported_model_graph_aoti.txt line 1182:
        view_90: "bf16[s55, ((10*u1)//s55), 52]" = _broadcast_impl.view(getitem_1, -1, 52)
        - _broadcast_impl has shape [u1, 520] (u1 is unbacked from .item())
        - getitem_1 is s55 (backed from embeddings_1.size(0))
        """

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating rows by batch sizes and viewing the result with
            # a leading size taken from another tensor.
            def forward(self, batch_sizes, embeddings):
                # [FUNCTION DOC] HELPER: Reads the leading size from embeddings and an unbacked
                # total from batch_sizes, repeats a zeros matrix by batch sizes and returns a view
                # with the backed size, inferred middle dim and last dim of 52.
                # batch_sizes: i64[s8] - batch size counts per item
                # embeddings: bf16[s55, s12] - embeddings tensor (s55 can be 0)
                # s8 and s55 are INDEPENDENT backed symints

                # Get s55 from embeddings.size(0) - backed symint
                s55 = embeddings.size(0)

                # Get u1 via sum().item() - unbacked symint
                # This mimics: item_1: "Sym(u1)" = sum_2.item()
                u1 = batch_sizes.sum().item()
                torch._check(u1 >= 0)

                # Create values tensor with shape [s8, 520]
                s8 = batch_sizes.size(0)
                values = embeddings.new_zeros(s8, 520)

                # Create data via repeat_interleave - result has shape [u1, 520]
                # This mimics:
                #   idx64: "i64[u1]" = torch.repeat_interleave(arange, batch_sizes, ...)
                #   _broadcast_impl: "bf16[u1, 520]" = cat_2[idx64]
                data = torch.repeat_interleave(values, batch_sizes, dim=0)

                # The problematic view: uses s55 (backed from embeddings, can be 0)
                # but the tensor data has size u1 (unbacked) - DECOUPLED from s55
                # When s55=0: -1 = (u1*520) / (s55*52) = 0/0
                # This mimics line 1182:
                #   view_90 = _broadcast_impl.view(getitem_1, -1, 52)
                result = data.view(s55, -1, 52)

                return result

        # Compile with non-empty tensors
        # At compile time: s8=3, s55=5, u1=5 (sum of batch_sizes = 2+2+1 = 5)
        compile_batch_sizes = torch.tensor(
            [2, 2, 1], dtype=torch.int64, device=self.device
        )
        compile_embeddings = torch.randn(5, 104, device=self.device)
        compile_inputs = (compile_batch_sizes, compile_embeddings)

        # Define dynamic shapes - s55 can be 0 at runtime!
        s8 = Dim("s8", min=1, max=1024)
        s55 = Dim("s55", min=0, max=1024)  # Can be 0!
        s12 = Dim("s12", min=1, max=1024)
        dynamic_shapes = {
            "batch_sizes": {0: s8},
            "embeddings": {0: s55, 1: s12},
        }

        # Export and compile
        model = Model().to(self.device)
        ep = torch.export.export(
            model, compile_inputs, dynamic_shapes=dynamic_shapes, strict=False
        )

        package_path = torch._inductor.aoti_compile_and_package(ep)

        with zipfile.ZipFile(package_path, "r") as z:
            for name in z.namelist():
                if name.endswith("wrapper.cpp"):
                    content = z.read(name).decode("utf-8", errors="replace")
                    # Verify that division and modulo operations are guarded
                    # The new implementation emits AOTI_TORCH_CHECK calls before
                    # computing size expressions that contain division/modulo
                    FileCheck().check("AOTI_TORCH_CHECK(").check("by zero").run(content)

        optimized = torch._inductor.aoti_load_package(package_path)

        # Run with s55=0 (embeddings is empty) and u1=0 (batch_sizes sum to 0)
        # data has shape [0, 520] (u1=0)
        # view(s55=0, -1, 52): -1 = 0*520 / (0*52) = 0/0 = division by zero
        # sum=0, u1=0
        run_batch_sizes = torch.tensor([0], dtype=torch.int64, device=self.device)
        run_embeddings = torch.randn(0, 104, device=self.device)  # s55=0
        run_inputs = (run_batch_sizes, run_embeddings)

        # Should raise RuntimeError instead of crashing with SIGFPE.
        # The generated AOTI_TORCH_CHECK logs the error (including "by zero") to stderr
        # before returning an error code, which is then converted to a RuntimeError.
        # We capture stderr at the file descriptor level to verify the message is logged.
        import tempfile

        # Save original stderr fd and redirect to temp file to capture C++ LOG(ERROR) output
        original_stderr_fd = os.dup(2)
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_stderr:
            temp_stderr_path = temp_stderr.name

        try:
            with open(temp_stderr_path, "w") as new_stderr:
                os.dup2(new_stderr.fileno(), 2)
                try:
                    with self.assertRaises(RuntimeError):
                        optimized(*run_inputs)
                finally:
                    # Flush and restore original stderr
                    new_stderr.flush()
                    os.dup2(original_stderr_fd, 2)
                    os.close(original_stderr_fd)

            # Read captured stderr and verify the "by zero" message was logged
            with open(temp_stderr_path) as f:
                stderr_content = f.read()
            self.assertIn("by zero", stderr_content)
        finally:
            # Clean up temp file
            if os.path.exists(temp_stderr_path):
                os.unlink(temp_stderr_path)

    def test_combo_kernel_grid_mixed_types(self):
        # [FUNCTION DOC] Checks combo kernels with transposed inputs where one pair has a dynamic
        # dim and the other is static, producing grid expressions mixing strings and integers,
        # which previously generated a max call with mismatched types in C++. GPU only.
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("combo kernels require GPU")

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two independent additions.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Returns a plus b and c plus d.
                return a + b, c + d

        # Non-contiguous (transposed) tensors force 2D Triton iteration,
        # creating both x and y range trees in the combo kernel.
        # Dynamic shapes on (a, b) make ynumel_0 dynamic (string in codegen),
        # while static shapes on (c, d) keep ynumel_1 as int.
        # This triggers GridExpr.maximum() with mixed [str, int] in C++ mode,
        # which previously generated std::max(long, int) causing a template
        # deduction error in the AOTInductor C++ wrapper.
        example_inputs = (
            torch.randn(30, 20, device=self.device),
            torch.randn(20, 30, device=self.device).t(),
            torch.randn(40, 30, device=self.device),
            torch.randn(30, 40, device=self.device).t(),
        )
        dim0 = Dim("dim0", min=1, max=100)
        dynamic_shapes = {
            "a": {0: dim0, 1: None},
            "b": {0: dim0, 1: None},
            "c": None,
            "d": None,
        }
        with config.patch({"combo_kernels": True}):
            self.check_model(Model(), example_inputs, dynamic_shapes=dynamic_shapes)


class AOTInductorLoggingTest(LoggingTestCase):
    # [CLASS DOC] Logging-based tests for AOTInductor compilation (shape environment
    # creation logs).
    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse(self, records):
        # [FUNCTION DOC] Checks, using captured dynamic-shapes debug logs, that exactly one shape
        # environment is created during an AOTI compile of an exported module with automatic
        # dynamic dims.
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus two.
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with torch.no_grad():
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)

    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse_zero_consts_use_consts_asm_false(self, records):
        # [FUNCTION DOC] Same as the previous test but with the consts assembly build disabled;
        # asserts exactly one shape environment creation log record.
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus two.
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with (
            torch.no_grad(),
            config.patch({"aot_inductor.use_consts_asm_build": False}),
        ):
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)


class TestAOTInductorConfig(TestCase):
    # [CLASS DOC] Tests for the helper that expands the standalone compile mode config into the
    # AOTI config options it implies.
    def test_no_compile_standalone(self):
        # [FUNCTION DOC] With standalone mode patched off, the helper returns an empty mapping for
        # an empty input.
        with config.patch({"aot_inductor_mode.compile_standalone": False}):
            result = maybe_aoti_standalone_config({})
            self.assertEqual(result, {})

    def test_compile_standalone_sets_package_cpp(self):
        # [FUNCTION DOC] With standalone mode enabled, asserts the helper result enables cpp-only
        # packaging, embedded kernel binaries, multi-arch kernels (except on ROCm), a default model
        # name and static linkage.
        result = maybe_aoti_standalone_config(
            {"aot_inductor_mode.compile_standalone": True}
        )
        self.assertEqual(result["aot_inductor.package_cpp_only"], True)
        self.assertEqual(result["aot_inductor_mode.compile_standalone"], True)
        self.assertEqual(result["aot_inductor.embed_kernel_binary"], True)
        self.assertEqual(
            result["aot_inductor.emit_multi_arch_kernel"], not torch.version.hip
        )
        self.assertEqual(
            result["aot_inductor.model_name_for_generated_files"], "aoti_model"
        )
        self.assertEqual(result["aot_inductor.dynamic_linkage"], False)

    def test_compile_standalone_explicit_set(self):
        # [FUNCTION DOC] When all standalone-related options are already explicitly set, asserts
        # the helper returns the same mapping unchanged.
        patches = {
            "aot_inductor_mode.compile_standalone": True,
            "aot_inductor.package_cpp_only": True,
            "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.dynamic_linkage": False,
            "aot_inductor.link_libtorch": False,
            "aot_inductor.emit_multi_arch_kernel": not torch.version.hip,
            "aot_inductor.model_name_for_generated_files": "aoti_model",
        }
        result = maybe_aoti_standalone_config(patches)
        self.assertEqual(result, patches)

    def test_compile_standalone_package_cpp_false_raises(self):
        # [FUNCTION DOC] Asserts RuntimeError when standalone mode is on but cpp-only packaging is
        # explicitly false, either in the passed options or in the global config.
        patches = {
            "aot_inductor_mode.compile_standalone": True,
            "aot_inductor.package_cpp_only": False,
        }
        with self.assertRaises(RuntimeError):
            maybe_aoti_standalone_config(patches)

        with config.patch({"aot_inductor.package_cpp_only": False}):
            patches = {
                "aot_inductor_mode.compile_standalone": True,
            }
            with self.assertRaises(RuntimeError):
                maybe_aoti_standalone_config(patches)

    def test_compile_standalone_cross_compile_windows_package_format(self):
        # [FUNCTION DOC] Asserts RuntimeError for a Windows cross-compile target combined with
        # constants packaged in the library.
        patches = {
            "aot_inductor.cross_target_platform": "windows",
            "aot_inductor.package_constants_in_so": True,
        }
        with self.assertRaises(RuntimeError):
            maybe_aoti_standalone_config(patches)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


def fail_cpu(is_skip=False):
    # [FUNCTION DOC] HELPER: Builds an expected-failure descriptor for the CPU suffix, which is
    # skipped instead of expected to fail when is_skip is true.
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_mps(is_skip=False):
    # [FUNCTION DOC] HELPER: Builds an expected-failure descriptor for the MPS suffix, which is
    # skipped instead of expected to fail when is_skip is true.
    return TestFailure(
        ("mps",),
        is_skip=is_skip,
    )


def fail_gpu(suffixes: tuple[str, ...], is_skip=False):
    # [FUNCTION DOC] HELPER: Builds an expected-failure descriptor for the given device suffixes,
    # skipped instead of expected to fail when is_skip is true.
    return TestFailure(
        suffixes,
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
GPU_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_gpu(("cuda", "xpu")),
    "test_quanatized_int8_linear": fail_gpu(("cuda", "xpu")),
    "test_quantized_linear_bias_none": fail_gpu(("cuda", "xpu")),
    # This test forces lazy dual-wrapper mode; torch.cond support for that
    # mode is covered by AOTInductorTestDualWrapper skips below.
    "test_cond_symint_input_disable_one_pass": fail_gpu(("cuda", "xpu"), is_skip=True),
}

MPS_TEST_FAILURES = {
    # aten::_scaled_dot_product_efficient_attention is not currently implemented for the MPS device.
    "test_scaled_dot_product_efficient_attention": fail_mps(),
    # aten::_int_mm is not implemented for MPS backend
    "test__int_mm": fail_mps(),
    # MPS doesn't support float64
    "test_while_loop_with_conv_dynamic_True": fail_mps(),
    "test_while_loop_with_conv_dynamic_False": fail_mps(),
    # MPS doesn't support float8
    "test_fp8": fail_mps(),
    "test_fp8_view_of_param": fail_mps(),
    # cannot initialize a parameter of type 'double' with an rvalue of type 'std::nullptr_t'
    "test_fallback_kernel_with_symexpr_output": fail_mps(),
    # correctness issue
    "test_index_put_with_none_index": fail_mps(),
    # Error device may not be nil
    "test_zero_size_weight": fail_mps(is_skip=True),
    # MPSGraph does not support tensor dims > INT_MAX
    "test_upper_bound_i64": fail_mps(is_skip=True),
    # MPS doesn't support triton
    "test_autotuning_args_reuse": fail_mps(),
    "test_triton_autotuning": fail_mps(),
    "test_triton_dynamic_launcher_grid": fail_mps(),
    "test_triton_dynamic_launcher_grid_infer_from_tensor": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_False_tma_version_new": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_False_tma_version_old": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_True_tma_version_new": fail_mps(),
    "test_triton_kernel_on_device_tma_dynamic_True_tma_version_old": fail_mps(),
    "test_size_with_unbacked_add_expr_transitive": fail_mps(),
    "test_size_with_unbacked_add_and_mul_expr": fail_mps(),
    "test_triton_next_power_of_2": fail_mps(),
    "test_sympy_cpp_printer_min_max_minmax0": fail_mps(),
    "test_sympy_cpp_printer_min_max_minmax1": fail_mps(),
    "test_triton_kernel_dynamic_shape_with_div": fail_mps(),
    "test_triton_kernel_reinterpret_view": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_False_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_False_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_True_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_1d_dynamic_True_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_False_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_False_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_True_tma_version_new_mps": fail_mps(),
    "test_triton_kernel_tma_descriptor_2d_dynamic_True_tma_version_old_mps": fail_mps(),
    "test_triton_kernel_sympy_expr_arg": fail_mps(),
    "test_triton_kernel_sympy_fn_like_arg": fail_mps(),
    "test_triton_kernel_with_none_input": fail_mps(),
    "test_triton_kernel_equal_to_1_arg": fail_mps(),
    "test_triton_kernel_with_none_inputs_and_equal_to_1_arg": fail_mps(),
    "test_triton_kernel_equal_to_1_float_arg_dynamic_True": fail_mps(),
    "test_triton_kernel_equal_to_1_float_arg_dynamic_False": fail_mps(),
    "test_triton_kernel_weird_param_order": fail_mps(),
    "test_triton_kernel_dynamic_grid": fail_mps(),
    "test_repeated_user_defined_triton_kernel_embed_kernel_binary_False": fail_mps(),
    "test_repeated_user_defined_triton_kernel_embed_kernel_binary_True": fail_mps(),
    "test_triton_kernel_extern_kernel_arg": fail_mps(),
    "test_triton_kernel_multi_output_arg": fail_mps(),
    "test_triton_kernel_reinterpret_view_mem_leak": fail_mps(),
    "test_triton_mutated_autotuning": fail_mps(),
    "test_sym_i64_input_codegen": fail_mps(),
    "test_none_args_aot_codegen": fail_mps(),
    "test_aoti_debug_printer_sym_inputs": fail_mps(),
    "test_aoti_debug_printer_user_defined_triton_kernel": fail_mps(),
    "test_autotune_int64_user_defined_triton_kernel": fail_mps(),
}


class AOTInductorTestABICompatibleCpu(TestCase):
    # [CLASS DOC] Concrete CPU test class: sets the device to CPU, binds the shared check helpers,
    # and disables stack allocation and the minimal arrayref interface. Test methods are copied in
    # from the template with CPU skip/expected-failure markings.
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "cpu",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleGpu(TestCase):
    # [CLASS DOC] Concrete GPU test class (device is the GPU type): binds the shared check helpers
    # and disables stack allocation and the minimal arrayref interface. Test methods are copied
    # from the template. Skipped on macOS.
    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)


# Lazy-autotune-mode-specific failures go here. Inherits regular GPU failures.
GPU_LAZY_AUTOTUNE_TEST_FAILURES = {
    **GPU_TEST_FAILURES,
    # This regression intentionally creates an incorrect fake dtype and
    # expects AOTI runtime validation to catch it.  Lazy autotune dual-wrapper
    # mode runs the generated JIT wrapper during compile, so it fails before
    # the AOTI package can be loaded.
    "test_aoti_custom_op_bad_fake_dtype_fails_fast": fail_gpu(
        ("cuda", "xpu"), is_skip=True
    ),
}


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestDualWrapper(TestCase):
    # [CLASS DOC] Concrete GPU test class that runs the template tests with compile-time triton
    # autotuning disabled (lazy compile, dual wrapper codegen). Uses the lazy-autotune failure
    # table. Skipped on macOS.
    """Run AOTInductor tests with autotune_at_compile_time=False, exercising
    the lazy Triton compile + dual-wrapper-mode codegen path."""

    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False

    def setUp(self):
        # [FUNCTION DOC] Enters the config patch disabling compile-time triton autotuning and
        # registers a cleanup to exit it after each test.
        super().setUp()
        ctx = torch._inductor.config.patch("triton.autotune_at_compile_time", False)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestDualWrapper,
    GPU_TYPE,
    GPU_LAZY_AUTOTUNE_TEST_FAILURES,
)


@unittest.skipIf(not torch.backends.mps.is_available(), "No MPS backend available")
class AOTInductorTestABICompatibleMps(TestCase):
    # [CLASS DOC] Concrete MPS test class: device is MPS, binds the shared check helpers
    # and disables stack allocation and the minimal arrayref interface. Skipped when MPS
    # is unavailable.
    device = "mps"
    device_type = "mps"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleMps,
    "mps",
    MPS_TEST_FAILURES,
)


class TestCheckLowerboundConfig(TestCase):
    # [CLASS DOC] Tests for the configuration option controlling lower-bound checks in
    # generated code.
    def test_aoti_check_lowerbound_codegen(self):
        # [FUNCTION DOC] Compiles a model with a dynamic dim bounded between 2 and 10 with the
        # lower-bound check option on, asserting exactly one dim too small message in the generated
        # code, and off, asserting none.
        """
        Test that check_lowerbound config controls lowerbound check codegen.
        When check_lowerbound=False, no lowerbound checks should be generated.
        """

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        model = Model()
        batch = Dim("batch", min=2, max=10)
        example_inputs = (torch.randn(4, 3),)

        # Test with check_lowerbound=True (default)
        with config.patch({"aot_inductor.check_lowerbound": True}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile,
                model,
                example_inputs,
                dynamic_shapes={"x": {0: batch}},
            )
            # Should have lowerbound checks
            FileCheck().check_count(
                "dim value is too small",
                1,
                exactly=True,
            ).run(code)

        # Test with check_lowerbound=False
        with config.patch({"aot_inductor.check_lowerbound": False}):
            result, code = run_and_get_cpp_code(
                AOTIRunnerUtil.legacy_compile,
                model,
                example_inputs,
                dynamic_shapes={"x": {0: batch}},
            )
            # Should NOT have lowerbound checks
            FileCheck().check_count(
                "dim value is too small",
                0,
                exactly=True,
            ).run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_GPU or sys.platform == "darwin":
        run_tests(needs="filelock")
