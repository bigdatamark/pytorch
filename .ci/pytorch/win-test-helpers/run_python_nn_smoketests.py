#!/usr/bin/env python3
# [MODULE DOC] Windows CI helper that runs small "python -c" checks against the installed
# torch (import, MKL, and with USE_CUDA=1 also CUDA tensors, magma, cuDNN). Defines the
# COMMON_TESTS and GPU_TESTS lists of (description, code string) pairs; the main block runs
# them in order and exits with the failing child's return code. No import-time side effects.

import os
import subprocess
import sys


COMMON_TESTS = [
    (
        "Checking that torch is available",
        "import torch",
    ),
    (
        "Checking that MKL is available",
        "import torch; exit(0 if torch.backends.mkl.is_available() else 1)",
    ),
]

GPU_TESTS = [
    (
        "Checking that CUDA archs are setup correctly",
        "import torch; torch.randn([3,5]).cuda()",
    ),
    (
        "Checking that magma is available",
        "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)",
    ),
    (
        "Checking that CuDNN is available",
        "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)",
    ),
]


if __name__ == "__main__":
    # Script entry: picks the test list from the USE_CUDA environment variable (GPU tests
    # only when it equals 1), runs each snippet in a child python process and prints its
    # description and command. On failure, if the Windows cdb debugger exists it reruns the
    # command under it to get a traceback, then exits with the original return code.
    if "USE_CUDA" in os.environ and os.environ["USE_CUDA"] == "1":
        TESTS = COMMON_TESTS + GPU_TESTS
    else:
        TESTS = COMMON_TESTS
    for description, python_commands in TESTS:
        print(description)
        command_args = ["python", "-c", python_commands]
        command_string = " ".join(command_args)
        print("Command:", command_string)
        try:
            subprocess.check_call(command_args)
        except subprocess.CalledProcessError as e:
            sdk_root = os.environ.get(
                "WindowsSdkDir", "C:\\Program Files (x86)\\Windows Kits\\10"
            )
            debugger = os.path.join(sdk_root, "Debuggers", "x64", "cdb.exe")
            if os.path.exists(debugger):
                command_args = [debugger, "-o", "-c", "~*g; q"] + command_args
                command_string = " ".join(command_args)
                print("Rerunning with traceback enabled")
                print("Command:", command_string)
                subprocess.run(command_args, check=False)
            sys.exit(e.returncode)
