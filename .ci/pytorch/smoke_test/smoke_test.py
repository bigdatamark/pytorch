# [MODULE DOC] Smoke tests run against an installed PyTorch binary wheel in CI. Checks the
# torch (and optionally torchvision/torchaudio) versions against the release matrix or the
# nightly date, CUDA/cuDNN/NCCL versions and GPU init, conv2d, linalg, SDPA, torch.compile,
# NVSHMEM, and runs the companion libraries' own smoke tests. Import-time side effects: reads
# MATRIX_* / GPU_ARCH_VERSION / TARGET_OS environment variables into module globals, defines
# the MODULES table and imports torch. The main block runs everything; failures raise errors.
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from check_wheel_tags import check_mac_wheel_minos, check_wheel_platform_tag

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


if "MATRIX_GPU_ARCH_VERSION" in os.environ:
    gpu_arch_ver = os.getenv("MATRIX_GPU_ARCH_VERSION")
else:
    gpu_arch_ver = os.getenv("GPU_ARCH_VERSION")  # Use fallback if available
gpu_arch_type = os.getenv("MATRIX_GPU_ARCH_TYPE")
channel = os.getenv("MATRIX_CHANNEL")
package_type = os.getenv("MATRIX_PACKAGE_TYPE")
target_os = os.getenv("TARGET_OS", sys.platform)
BASE_DIR = Path(__file__).parent.parent.parent
PYTORCH_ROOT = BASE_DIR.parent

is_cuda_system = gpu_arch_type == "cuda"
NIGHTLY_ALLOWED_DELTA = 3

MODULES = [
    {
        "name": "torchvision",
        "repo": "https://github.com/pytorch/vision.git",
        "smoke_test": "./vision/test/smoke_test.py",
        "extension": "extension",
        "repo_name": "vision",
    },
    {
        "name": "torchaudio",
        "repo": "https://github.com/pytorch/audio.git",
        "smoke_test": "./audio/test/smoke_test/smoke_test.py --no-ffmpeg",
        "extension": "_extension",
        "repo_name": "audio",
    },
]


def get_modules_for_package(package: str) -> list:
    # [FUNCTION DOC] HELPER: Returns the companion module descriptors to test for a package
    # selector: all of them for all, only torchvision for torch_torchvision, else an empty list.
    if package == "all":
        return MODULES
    elif package == "torch_torchvision":
        return [m for m in MODULES if m["name"] == "torchvision"]
    else:
        return []


class Net(nn.Module):
    # [CLASS DOC] HELPER: Small CNN used as the model for the max-autotune compile smoke test.
    # State: two convolutions and one linear layer mapping 28x28 single-channel input to one
    # value per sample.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the two convolution layers and the linear layer.
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both convolutions, max pooling, flatten and the
        # linear layer; returns a tensor with one column. Expects a batch of 1x28x28 inputs.
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output


def load_json_from_basedir(filename: str):
    # [FUNCTION DOC] HELPER: Reads and parses a JSON file located under the base directory
    # (the .ci directory's parent). Raises ImportError if the file is missing or invalid.
    try:
        with open(BASE_DIR / filename) as fptr:
            return json.load(fptr)
    except FileNotFoundError as exc:
        raise ImportError(f"File {filename} not found error: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ImportError(f"Invalid JSON {filename}") from exc


def read_release_matrix():
    # [FUNCTION DOC] HELPER: Returns the parsed release matrix JSON (expected versions by
    # package name).
    return load_json_from_basedir("release_matrix.json")


def test_numpy():
    # [FUNCTION DOC] Checks that a numpy array converts to a torch tensor. If numpy is not
    # installed it prints a skip message instead of failing.
    try:
        import numpy as np

        x = np.arange(5)
        torch.tensor(x)
    except ImportError:
        print("Numpy check skipped. Numpy is not installed.")


def check_version(package: str) -> None:
    # [FUNCTION DOC] Validates installed versions. The expected stable version comes from the
    # release matrix when RELEASE_VERSION is set, else from MATRIX_STABLE_VERSION. For the
    # nightly channel it delegates to the date check. Otherwise requires torch's version to
    # start with the expected one and, with a release version, the same for torchvision and
    # torchaudio as selected by package. Raises RuntimeError on mismatch; prints a skip
    # message if no expected version is known.
    release_version = os.getenv("RELEASE_VERSION")
    # if release_version is specified, use it to validate the packages
    if release_version:
        release_matrix = read_release_matrix()
        stable_version = release_matrix["torch"]
    else:
        stable_version = os.getenv("MATRIX_STABLE_VERSION")

    # only makes sense to check nightly package where dates are known
    if channel == "nightly":
        check_nightly_binaries_date(package)
    elif stable_version is not None:
        if not torch.__version__.startswith(stable_version):
            raise RuntimeError(
                f"Torch version mismatch, expected {stable_version} for channel {channel}. But its {torch.__version__}"
            )

        if release_version and package in ["all", "torch_torchvision"]:
            for module in get_modules_for_package(package):
                imported_module = importlib.import_module(module["name"])
                module_version = imported_module.__version__
                if not module_version.startswith(release_matrix[module["name"]]):
                    raise RuntimeError(
                        f"{module['name']} version mismatch, expected: \
                            {release_matrix[module['name']]} for channel {channel}. But its {module_version}"
                    )
                else:
                    print(
                        f"{module['name']} version actual: {module_version} expected: \
                        {release_matrix[module['name']]} for channel {channel}."
                    )

    else:
        print(f"Skip version check for channel {channel} as stable version is None")


def check_nightly_binaries_date(package: str) -> None:
    # [FUNCTION DOC] Parses the dev date stamp from the torch version (and from companion
    # module versions when selected) and raises RuntimeError if a binary is too old.
    # KNOWN ISSUE: the age limit is applied inconsistently: torch fails at three or more days
    # old, while the companion modules fail only above three days.
    from datetime import datetime

    format_dt = "%Y%m%d"

    date_t_str = re.findall("dev\\d+", torch.__version__)
    date_t_delta = datetime.now() - datetime.strptime(date_t_str[0][3:], format_dt)
    if date_t_delta.days >= NIGHTLY_ALLOWED_DELTA:
        raise RuntimeError(
            f"the binaries are from {date_t_str} and are more than {NIGHTLY_ALLOWED_DELTA} days old!"
        )

    if package in ["all", "torch_torchvision"]:
        for module in get_modules_for_package(package):
            imported_module = importlib.import_module(module["name"])
            module_version = imported_module.__version__
            date_m_str = re.findall("dev\\d+", module_version)
            date_m_delta = datetime.now() - datetime.strptime(
                date_m_str[0][3:], format_dt
            )
            print(f"Nightly date check for {module['name']} version {module_version}")
            if date_m_delta.days > NIGHTLY_ALLOWED_DELTA:
                raise RuntimeError(
                    f"Expected {module['name']} to be less then {NIGHTLY_ALLOWED_DELTA} days. But its {date_m_delta}"
                )


def test_cuda_runtime_errors_captured() -> None:
    # [FUNCTION DOC] Triggers a device-side assertion with a zero tensor on the GPU and
    # requires that a RuntimeError mentioning CUDA is raised. Re-raises other RuntimeErrors;
    # raises RuntimeError if no error occurs. Leaves the CUDA context in an error state.
    cuda_exception_missed = True
    try:
        print("Testing test_cuda_runtime_errors_captured")
        torch._assert_async(torch.tensor(0, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0j, device="cuda"))
    except RuntimeError as e:
        if re.search("CUDA", f"{e}"):
            print(f"Caught CUDA exception with success: {e}")
            cuda_exception_missed = False
        else:
            raise e
    if cuda_exception_missed:
        raise RuntimeError("Expected CUDA RuntimeError but have not received!")


def test_cuda_gds_errors_captured() -> None:
    # [FUNCTION DOC] Skips on Windows and CUDA older than 12.6. Otherwise tries to open a
    # temporary file through the GPUDirect Storage wrapper and expects a RuntimeError saying
    # the file handle registration failed; on CUDA 13.2 and newer, successful registration
    # (compatibility mode) is the expected outcome. Raises RuntimeError on unexpected results.
    major_version = int(torch.version.cuda.split(".")[0])
    minor_version = int(torch.version.cuda.split(".")[1])

    if target_os == "windows":
        print(f"{target_os} is not supported for GDS smoke test")
        return

    if major_version < 12 or (major_version == 12 and minor_version < 6):
        print("CUDA version is not supported for GDS smoke test")
        return

    cuda_exception_missed = True
    try:
        print("Testing test_cuda_gds_errors_captured")
        with NamedTemporaryFile() as f:
            torch.cuda.gds.GdsFile(f.name, os.O_CREAT | os.O_RDWR)
        # cuFile >= 1.17 (CUDA 13.2+) compat mode: registration succeeds
        # without nvidia-fs driver, falling back to POSIX I/O
        if major_version > 13 or (major_version == 13 and minor_version >= 2):
            print("GDS handle registered successfully via compatibility mode")
            cuda_exception_missed = False
    except RuntimeError as e:
        expected_error = "cuFileHandleRegister failed"
        if re.search(expected_error, f"{e}"):
            print(f"Caught expected CUDA exception: {e}")
            cuda_exception_missed = False
        else:
            raise e
    if cuda_exception_missed:
        raise RuntimeError(
            "Expected cuFileHandleRegister failed RuntimeError but have not received!"
        )


def find_pypi_package_version(package: str) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the version of the first installed distribution whose
    # name starts with the given prefix, or None if there is none.
    from importlib import metadata

    dists = metadata.distributions()
    for dist in dists:
        if dist.metadata["Name"].startswith(package):
            return dist.version
    return None


def get_expected_cudnn_version_linux(cuda_version: str) -> str | None:
    # [FUNCTION DOC] HELPER: Reads the binary build matrix script in the repo and returns the
    # major.minor.patch cuDNN version pinned for the given CUDA version, using a regular
    # expression. Returns None (with a warning) if the file or entry is missing.
    """Parse expected cuDNN version from generate_binary_build_matrix.py for Linux.

    Reads PYTORCH_EXTRA_INSTALL_REQUIREMENTS and extracts the cudnn version
    for the given CUDA version (e.g. "12.6").
    """
    matrix_script = (
        PYTORCH_ROOT / ".github" / "scripts" / "generate_binary_build_matrix.py"
    )
    if not matrix_script.exists():
        print(f"Warning: {matrix_script} not found, skipping cuDNN version check")
        return None

    content = matrix_script.read_text()
    # Match the full cudnn package version like nvidia-cudnn-cu12==9.10.2.21
    # and extract major.minor.patch (dropping the build number)
    pattern = (
        rf'"{re.escape(cuda_version)}":\s*\(\s*'
        r"[\s\S]*?nvidia-cudnn-cu\d+==(\d+\.\d+\.\d+)\.\d+"
    )
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    return None


def get_expected_cudnn_version_windows(cuda_version: str) -> str | None:
    # [FUNCTION DOC] HELPER: Reads the Windows CUDA install batch file and returns the
    # expected cuDNN version in the block for the given CUDA version (dots removed), via a
    # regular expression. Returns None (with a warning) if the file or entry is missing.
    """Parse expected cuDNN version from cuda_install.bat for Windows.

    Reads the batch file and extracts EXPECTED_CUDNN_VERSION for the given
    CUDA version (e.g. "12.6" maps to CUDA_VER 126).
    """
    bat_file = (
        PYTORCH_ROOT / ".ci" / "pytorch" / "windows" / "internal" / "cuda_install.bat"
    )
    if not bat_file.exists():
        print(f"Warning: {bat_file} not found, skipping cuDNN version check")
        return None

    content = bat_file.read_text()
    # Convert "12.6" to "126" to match batch file's CUDA_VER format
    cuda_ver_nodot = cuda_version.replace(".", "")
    # Match: if %CUDA_VER% EQU 126 ( ... set EXPECTED_CUDNN_VERSION=9.10.2 )
    pattern = (
        rf"if %CUDA_VER% EQU {re.escape(cuda_ver_nodot)}\s*\("
        r"[\s\S]*?set EXPECTED_CUDNN_VERSION=(\d+\.\d+\.\d+)"
    )
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    return None


def check_cudnn_version(cuda_version: str, actual_cudnn_version: str) -> None:
    # [FUNCTION DOC] Looks up the expected cuDNN version for the platform (Linux or Windows)
    # and requires the loaded version string to start with it. Raises RuntimeError on
    # mismatch; prints and returns on unsupported platforms or when no expectation is found.
    """Validate cuDNN version matches expected version from build config files."""
    if sys.platform in ["linux", "linux2"]:
        expected = get_expected_cudnn_version_linux(cuda_version)
        source = "generate_binary_build_matrix.py"
    elif sys.platform == "win32":
        expected = get_expected_cudnn_version_windows(cuda_version)
        source = "cuda_install.bat"
    else:
        print(f"cuDNN version check not supported on platform {sys.platform}")
        return

    if expected is None:
        print(
            f"Warning: Could not determine expected cuDNN version for CUDA {cuda_version} "
            f"from {source}, skipping validation"
        )
        return

    if not actual_cudnn_version.startswith(expected):
        raise RuntimeError(
            f"cuDNN version mismatch for CUDA {cuda_version}. "
            f"Loaded: {actual_cudnn_version} Expected: {expected} (from {source})"
        )
    print(
        f"cuDNN version check passed: {actual_cudnn_version} matches "
        f"expected {expected} from {source}"
    )


def cudnn_to_version_str(cudnn_version: int) -> str:
    # [FUNCTION DOC] HELPER: Converts the integer cuDNN version code into a dotted
    # major.minor.patch string by digit grouping. Pure function.
    patch = int(cudnn_version % 10)
    minor = int((cudnn_version / 100) % 100)
    major = int((cudnn_version / 10000) % 10000)
    return f"{major}.{minor}.{patch}"


def compare_pypi_to_torch_versions(
    package: str, pypi_version: str, torch_version: str
) -> None:
    # [FUNCTION DOC] Requires the package version installed from PyPI to start with the
    # version torch reports for the same library. Raises RuntimeError if the PyPI version is
    # missing (None) or does not match; otherwise prints a confirmation.
    if pypi_version is None:
        raise RuntimeError(f"Can't find {package} in PyPI for Torch: {torch_version}")
    if pypi_version.startswith(torch_version):
        print(f"Found matching {package}. Torch: {torch_version} PyPI {pypi_version}")
    else:
        raise RuntimeError(
            f"Wrong {package} version. Torch: {torch_version} PyPI: {pypi_version}"
        )


def smoke_test_cuda(
    package: str,
    runtime_error_check: str,
    torch_compile_check: str,
    pypi_pkg_check: str,
) -> None:
    # [FUNCTION DOC] Main CUDA and compile checks. Raises if a CUDA build is expected but CUDA
    # is not available; prints companion libraries' CUDA versions; runs the compile smoke test
    # when enabled on supported OS and Python versions; and when CUDA is available checks the
    # CUDA version against the expected one, initializes CUDA, lists devices, compares cuDNN
    # runtime and compile versions and the build-config version, prints the NCCL version on
    # Linux, compares PyPI cuDNN/NCCL versions if enabled, and optionally tests that CUDA
    # runtime errors are captured. Raises RuntimeError on any mismatch.
    if not torch.cuda.is_available() and is_cuda_system:
        raise RuntimeError(f"Expected CUDA {gpu_arch_ver}. However CUDA is not loaded.")

    if package in ["all", "torch_torchvision"] and is_cuda_system:
        for module in get_modules_for_package(package):
            imported_module = importlib.import_module(module["name"])
            # TBD for vision move extension module to private so it will
            # be _extention.
            version = "N/A"
            if module["extension"] == "extension":
                version = imported_module.extension._check_cuda_version()
            else:
                version = imported_module._extension._check_cuda_version()
            print(f"{module['name']} CUDA: {version}")

    # torch.compile is not supported on Python 3.15+ yet (it raises at runtime),
    # so skip the compile smoke test there instead of failing the wheel test.
    if (
        torch_compile_check == "enabled"
        and sys.version_info < (3, 15)
        and target_os
        in [
            "linux",
            "linux-aarch64",
            "macos-arm64",
            "darwin",
        ]
    ):
        smoke_test_compile("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        if torch.version.cuda != gpu_arch_ver:
            raise RuntimeError(
                f"Wrong CUDA version. Loaded: {torch.version.cuda} Expected: {gpu_arch_ver}"
            )

        print(f"torch cuda: {torch.version.cuda}")
        torch.cuda.init()
        print("CUDA initialized successfully")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        print(f"cuDNN enabled? {torch.backends.cudnn.enabled}")
        torch_cudnn_version = cudnn_to_version_str(torch.backends.cudnn.version())
        print(f"Torch cuDNN version: {torch_cudnn_version}")

        torch_cudnn_compile_version = torch._C._cudnn.getCompileVersion()
        print(f"Torch cuDNN compile-time version: {torch_cudnn_compile_version}")
        torch_cudnn_runtime_version = tuple(
            [int(x) for x in torch_cudnn_version.split(".")]
        )
        if torch_cudnn_runtime_version != torch_cudnn_compile_version:
            raise RuntimeError(
                "cuDNN runtime version doesn't match comple version. "
                f"Loaded: {torch_cudnn_runtime_version} "
                f"Expected: {torch_cudnn_compile_version}"
            )

        check_cudnn_version(gpu_arch_ver, torch_cudnn_version)

        if sys.platform in ["linux", "linux2"]:
            torch_nccl_version = ".".join(str(v) for v in torch.cuda.nccl.version())
            print(f"Torch nccl; version: {torch_nccl_version}")

        # Pypi dependencies are installed on linux only and nccl is available only on Linux.
        if pypi_pkg_check == "enabled" and sys.platform in ["linux", "linux2"]:
            compare_pypi_to_torch_versions(
                "cudnn", find_pypi_package_version("nvidia-cudnn"), torch_cudnn_version
            )
            compare_pypi_to_torch_versions(
                "nccl", find_pypi_package_version("nvidia-nccl"), torch_nccl_version
            )

        if runtime_error_check == "enabled":
            test_cuda_runtime_errors_captured()


def smoke_test_conv2d() -> None:
    # [FUNCTION DOC] Builds several Conv2d layers and runs one on random CPU input; on CUDA
    # systems also runs a layer under autocast and repeats the convolution for half, single
    # and double precision. Raises AssertionError if an output is None.
    # KNOWN ISSUE: the first Conv2d is immediately overwritten and never used.
    import torch.nn as nn

    print("Testing smoke_test_conv2d")
    # With square kernels and equal stride
    m = nn.Conv2d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    if m is None:
        raise AssertionError("Conv2d with non-square kernels returned None")
    # non-square kernels and unequal stride and with padding and dilation
    basic_conv = nn.Conv2d(
        16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
    )
    input = torch.randn(20, 16, 50, 100)
    output = basic_conv(input)

    if is_cuda_system:
        print("Testing smoke_test_conv2d with cuda")
        conv = nn.Conv2d(3, 3, 3).cuda()
        x = torch.randn(1, 3, 24, 24, device="cuda")
        with torch.cuda.amp.autocast():
            out = conv(x)
        if out is None:
            raise AssertionError("Conv2d with cuda autocast returned None")

        supported_dtypes = [torch.float16, torch.float32, torch.float64]
        for dtype in supported_dtypes:
            print(f"Testing smoke_test_conv2d with cuda for {dtype}")
            conv = basic_conv.to(dtype).cuda()
            input = torch.randn(20, 16, 50, 100, device="cuda").type(dtype)
            output = conv(input)
            if output is None:
                raise AssertionError(f"Conv2d with cuda for {dtype} returned None")


def test_linalg(device="cpu") -> None:
    # [FUNCTION DOC] Runs singular value decompositions (reduced, full and batched) on random
    # matrices on the given device and checks the output shapes, raising AssertionError on a
    # mismatch; on cuda also runs a batched decomposition for single and double precision.
    print(f"Testing smoke_test_linalg on {device}")
    A = torch.randn(5, 3, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if not (
        U.shape == A.shape
        and S.shape == torch.Size([3])
        and Vh.shape == torch.Size([3, 3])
    ):
        raise AssertionError(
            f"SVD shapes mismatch: U.shape={U.shape}, S.shape={S.shape}, Vh.shape={Vh.shape}"
        )
    torch.dist(A, U @ torch.diag(S) @ Vh)

    U, S, Vh = torch.linalg.svd(A)
    if not (
        U.shape == torch.Size([5, 5])
        and S.shape == torch.Size([3])
        and Vh.shape == torch.Size([3, 3])
    ):
        raise AssertionError(
            f"SVD full_matrices shapes mismatch: U.shape={U.shape}, S.shape={S.shape}, Vh.shape={Vh.shape}"
        )
    torch.dist(A, U[:, :3] @ torch.diag(S) @ Vh)

    A = torch.randn(7, 5, 3, device=device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    torch.dist(A, U @ torch.diag_embed(S) @ Vh)

    if device == "cuda":
        supported_dtypes = [torch.float32, torch.float64]
        for dtype in supported_dtypes:
            print(f"Testing smoke_test_linalg with cuda for {dtype}")
            A = torch.randn(20, 16, 50, 100, device=device, dtype=dtype)
            torch.linalg.svd(A)


def test_sdpa(device="cpu", dtype=torch.float16) -> None:
    # [FUNCTION DOC] Runs scaled dot product attention with random inputs and a random
    # additive mask on the device and dtype, and raises AssertionError if the output has NaN.
    """Regression test for https://github.com/pytorch/pytorch/issues/167602
    Without nvrtc_builtins on CuDNN-9.13 on CUDA-13 fails with ` No valid execution plans built.`
    """
    print(f"Testing SDPA on {device} using type {dtype}")
    k, q, v = torch.rand(3, 1, 16, 77, 64, dtype=dtype, device=device).unbind(0)
    attn = torch.rand(1, 1, 77, 77, dtype=dtype, device=device)
    rc = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn)
    if rc.isnan().any().item() is not False:
        raise AssertionError("SDPA output contains NaN values")


def smoke_test_compile(device: str = "cpu") -> None:
    # [FUNCTION DOC] Compiles a small sin-plus-cos function for three float dtypes and checks
    # it matches eager results; on cpu requires a vectorized instruction set to be detected
    # (RuntimeError otherwise). Then resets the compiler state, sets high matmul precision
    # (global side effect) and compiles the Net model with max-autotune on random input.
    supported_dtypes = [torch.float16, torch.float32, torch.float64]

    def foo(x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the elementwise sine plus cosine of x.
        return torch.sin(x) + torch.cos(x)

    for dtype in supported_dtypes:
        print(f"Testing smoke_test_compile for {device} and {dtype}")
        x = torch.rand(3, 3, device=device).type(dtype)
        x_eager = foo(x)
        x_pt2 = torch.compile(foo)(x)
        torch.testing.assert_close(x_eager, x_pt2)

    # Check that SIMD were detected for the architecture
    if device == "cpu":
        from torch._inductor.codecache import pick_vec_isa

        isa = pick_vec_isa()
        if not isa:
            raise RuntimeError("Can't detect vectorized ISA for CPU")
        print(f"Picked CPU ISA {type(isa).__name__} bit width {isa.bit_width()}")

    # Reset torch dynamo since we are changing mode
    torch._dynamo.reset()
    dtype = torch.float32
    torch.set_float32_matmul_precision("high")
    print(f"Testing smoke_test_compile with mode 'max-autotune' for {dtype}")
    x = torch.rand(64, 1, 28, 28, device=device).type(torch.float32)
    model = Net().to(device=device)
    x_pt2 = torch.compile(model, mode="max-autotune")(x)


def smoke_test_nvshmem() -> None:
    # [FUNCTION DOC] Skips without CUDA or on Windows. Otherwise checks that the build exposes
    # NVSHMEM support: absence is tolerated for torch older than 2.9 and raises RuntimeError
    # for newer versions. Prints whether NVSHMEM is available at run time.
    if not torch.cuda.is_available() or target_os == "windows":
        print("Windows platform or CUDA is not available, skipping NVSHMEM test")
        return

    # Check if NVSHMEM is compiled in current build
    try:
        from torch._C._distributed_c10d import _is_nvshmem_available
    except ImportError:
        # Not built with NVSHMEM support.
        # torch is not compiled with NVSHMEM prior to 2.9
        from torch.torch_version import TorchVersion

        if TorchVersion(torch.__version__) < (2, 9):
            return
        else:
            # After 2.9: NVSHMEM is expected to be compiled in current build
            raise RuntimeError("torch not compiled with NVSHMEM") from None

    print("torch compiled with NVSHMEM")

    # Check if NVSHMEM is available on current system.
    print(f"NVSHMEM available at run time: {_is_nvshmem_available()}")


def smoke_test_modules(package: str):
    # [FUNCTION DOC] For each companion module of the package selector, clones its repository
    # (shallow) into the current directory if absent, then runs its smoke test script in a
    # shell subprocess with the platform's python command and prints the output. Raises
    # RuntimeError, including captured output, if cloning or the test fails.
    cwd = os.getcwd()
    for module in get_modules_for_package(package):
        if module["repo"]:
            if not os.path.exists(f"{cwd}/{module['repo_name']}"):
                print(f"Path does not exist: {cwd}/{module['repo_name']}")
                try:
                    subprocess.check_output(
                        f"git clone --depth 1 {module['repo']}",
                        stderr=subprocess.STDOUT,
                        shell=True,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Cloning {module['repo']} FAIL: {exc.returncode} Output: {exc.output}"
                    ) from exc
            try:
                smoke_test_command = f"python3 {module['smoke_test']}"
                if target_os == "windows":
                    smoke_test_command = f"python {module['smoke_test']}"
                output = subprocess.check_output(
                    smoke_test_command,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    universal_newlines=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Module {module['name']} FAIL: {exc.returncode} Output: {exc.output}"
                ) from exc
            else:
                print(f"Output: \n{output}\n")


def parse_args():
    # [FUNCTION DOC] HELPER: Parses options: package selector, and enabled/disabled switches
    # for runtime error check, torch compile check and PyPI package check (all default to
    # enabled; package defaults to all). Returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        help="Package to include in smoke testing",
        type=str,
        choices=["all", "torch_torchvision", "torchonly"],
        default="all",
    )
    parser.add_argument(
        "--runtime-error-check",
        help="No Runtime Error check",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--torch-compile-check",
        help="Check torch compile",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--pypi-pkg-check",
        help="Check pypi package versions cudnn and nccl",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
    )
    return parser.parse_args()


def main() -> None:
    # [FUNCTION DOC] Entry point. Prints torch version and parallel info, requires OpenMP
    # support (RuntimeError otherwise), then runs version, conv2d, linalg, numpy and SDPA
    # checks (plus GPU variants on CUDA systems), companion module tests, CUDA/compile checks,
    # the NVSHMEM check, and wheel platform tag and macOS minimum OS checks.
    options = parse_args()
    print(f"torch: {torch.__version__}")
    print(torch.__config__.parallel_info())
    # All PyTorch binary builds should be built with OpenMP
    if not torch.backends.openmp.is_available():
        raise RuntimeError("PyTorch must be built with OpenMP support")

    check_version(options.package)
    smoke_test_conv2d()
    test_linalg()
    test_numpy()
    test_sdpa()

    if is_cuda_system:
        test_linalg("cuda")
        test_cuda_gds_errors_captured()
        test_sdpa("cuda")

    if options.package in ["all", "torch_torchvision"]:
        smoke_test_modules(options.package)

    smoke_test_cuda(
        options.package,
        options.runtime_error_check,
        options.torch_compile_check,
        options.pypi_pkg_check,
    )

    smoke_test_nvshmem()

    check_wheel_platform_tag()
    check_mac_wheel_minos()


if __name__ == "__main__":
    main()
