# [MODULE DOC] Script that sanity-checks a Dynamo installation: verifies the Python, torch, CUDA
# and ROCm versions (mostly as warnings) and runs small compile and backward checks for the
# eager, aot_eager and inductor backends on cpu and cuda, exiting with status 1 on failure.
# Run as a script it calls main. Defines minimum version constants. No import-time side effects.
import os
import re
import subprocess
import sys
import traceback
import warnings


MIN_CUDA_VERSION = "12.1"
MIN_ROCM_VERSION = "5.4"
MIN_PYTHON_VERSION = (3, 10)


class VerifyDynamoError(BaseException):
    # [CLASS DOC] Exception raised when an environment check fails (missing CUDA or ROCm,
    # unparseable tool output, unsupported Python). Derives from BaseException, so a generic
    # exception handler does not catch it.
    pass


def check_python():
    # [FUNCTION DOC] Returns the interpreter version info; raises VerifyDynamoError if it is
    # older than the minimum supported Python version.
    if sys.version_info < MIN_PYTHON_VERSION:
        raise VerifyDynamoError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )
    return sys.version_info


def check_torch():
    # [FUNCTION DOC] Imports torch (ImportError if unavailable) and returns its version string.
    import torch

    return torch.__version__


# based on torch/utils/cpp_extension.py
def get_cuda_version():
    # [FUNCTION DOC] Finds the system CUDA install, runs its nvcc compiler with the version
    # flag and returns the parsed major.minor release as a TorchVersion. Raises
    # VerifyDynamoError if CUDA is not found or the version is not in the output; running nvcc
    # may raise subprocess errors.
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    CUDA_HOME = cpp_extension._find_cuda_home()
    if not CUDA_HOME:
        raise VerifyDynamoError(cpp_extension.CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    cuda_version_str = (
        subprocess.check_output([nvcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)
    if cuda_version is None:
        raise VerifyDynamoError("CUDA version not found in `nvcc --version` output")

    cuda_str_version = cuda_version.group(1)
    return TorchVersion(cuda_str_version)


def get_rocm_version():
    # [FUNCTION DOC] Finds the system ROCm install, runs hipcc with the version flag and
    # returns the parsed major.minor HIP version as a TorchVersion. Raises VerifyDynamoError if
    # ROCm is not found or the version is not in the output.
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    ROCM_HOME = cpp_extension._find_rocm_home()
    if not ROCM_HOME:
        raise VerifyDynamoError(
            "ROCM was not found on the system, please set ROCM_HOME environment variable"
        )

    hipcc = os.path.join(ROCM_HOME, "bin", "hipcc")
    hip_version_str = (
        subprocess.check_output([hipcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    hip_version = re.search(r"HIP version: (\d+[.]\d+)", hip_version_str)

    if hip_version is None:
        raise VerifyDynamoError("HIP version not found in `hipcc --version` output")

    hip_str_version = hip_version.group(1)

    return TorchVersion(hip_str_version)


def check_cuda():
    # [FUNCTION DOC] Returns None when CUDA is unavailable or torch is a ROCm build. Otherwise
    # compares the CUDA version torch was built with to the system one and to the minimum,
    # emitting warnings (not errors) on mismatch or too old, and returns the system version.
    # May raise VerifyDynamoError via get_cuda_version.
    # KNOWN ISSUE: compares TorchVersion objects to a plain version string for the minimum.
    import torch
    from torch.torch_version import TorchVersion

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None

    torch_cuda_ver = TorchVersion(torch.version.cuda)

    # check if torch cuda version matches system cuda version
    cuda_ver = get_cuda_version()
    if cuda_ver != torch_cuda_ver:
        # raise VerifyDynamoError(
        warnings.warn(
            f"CUDA version mismatch, `torch` version: {torch_cuda_ver}, env version: {cuda_ver}"
        )

    if torch_cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(`torch`) CUDA version not supported: {torch_cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )
    if cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(env) CUDA version not supported: {cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )

    return cuda_ver if torch.version.hip is None else "None"


def check_rocm():
    # [FUNCTION DOC] Returns None when no GPU is available or torch is not a ROCm build.
    # Otherwise compares torch's ROCm major.minor to the system version and to the minimum,
    # warning on mismatch or too old, and returns the system version. May raise
    # VerifyDynamoError via get_rocm_version.
    import torch
    from torch.torch_version import TorchVersion

    if not torch.cuda.is_available() or torch.version.hip is None:
        return None

    # Extracts main ROCm version from full string
    torch_rocm_ver = TorchVersion(".".join(list(torch.version.hip.split(".")[0:2])))

    # check if torch rocm version matches system rocm version
    rocm_ver = get_rocm_version()
    if rocm_ver != torch_rocm_ver:
        warnings.warn(
            f"ROCm version mismatch, `torch` version: {torch_rocm_ver}, env version: {rocm_ver}"
        )
    if torch_rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(`torch`) ROCm version not supported: {torch_rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )
    if rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(env) ROCm version not supported: {rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )

    return rocm_ver if torch.version.hip else "None"


def check_dynamo(backend, device, err_msg) -> None:
    # [FUNCTION DOC] Runs a tiny forward and backward check with the given dynamo backend on
    # the given device. Skips (printing a message) for cuda when CUDA or triton is unusable.
    # Resets dynamo, compiles a function and a module computing x plus x with full-graph mode,
    # runs both on random input, and compares outputs and gradients to expected values. On any
    # exception writes the traceback and err_msg to stderr and exits the process with status 1.
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torch._dynamo as dynamo

        if device == "cuda":
            from torch.utils._triton import has_triton

            if not has_triton():
                print(
                    f"WARNING: CUDA available but triton cannot be used. "
                    f"Your GPU may not be supported. "
                    f"Skipping CUDA check on {backend} backend\n"
                )
                return

        dynamo.reset()

        @dynamo.optimize(backend, nopython=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Test function returning the input added to itself.
            return x + x

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Parameterless test module whose forward doubles its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input added to itself.
                return x + x

        mod = Module()
        opt_mod = dynamo.optimize(backend, nopython=True)(mod)

        for f in (fn, opt_mod):
            x = torch.randn(10, 10).to(device)
            x.requires_grad = True
            y = f(x)
            torch.testing.assert_close(y, x + x)
            z = y.sum()
            z.backward()
            torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))
    except Exception:
        sys.stderr.write(traceback.format_exc() + "\n" + err_msg + "\n\n")
        sys.exit(1)


_SANITY_CHECK_ARGS = (
    ("eager", "cpu", "CPU eager sanity check failed"),
    ("eager", "cuda", "CUDA eager sanity check failed"),
    ("aot_eager", "cpu", "CPU aot_eager sanity check failed"),
    ("aot_eager", "cuda", "CUDA aot_eager sanity check failed"),
    ("inductor", "cpu", "CPU inductor sanity check failed"),
    (
        "inductor",
        "cuda",
        "CUDA inductor sanity check failed\n"
        + "NOTE: Please check that you installed the correct hash/version of `triton`",
    ),
)


def main() -> None:
    # [FUNCTION DOC] Entry point: checks and prints the Python, torch, CUDA and ROCm versions,
    # then runs check_dynamo for every backend/device pair in the sanity check table (warning
    # when Python is 3.15 or newer), and prints a success line. A failing check exits the
    # process.
    python_ver = check_python()
    torch_ver = check_torch()
    cuda_ver = check_cuda()
    rocm_ver = check_rocm()
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"`torch` version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}\n"
        f"ROCM version: {rocm_ver}\n"
    )
    for args in _SANITY_CHECK_ARGS:
        if sys.version_info >= (3, 15):
            warnings.warn("Dynamo not yet supported in Python 3.15.")
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()
