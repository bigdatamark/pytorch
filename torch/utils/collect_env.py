# mypy: allow-untyped-defs

# [MODULE DOC] Environment report tool used when filing bug reports. Runs shell commands and
# queries torch to collect OS, compiler, CUDA/ROCm/XPU, driver, cuDNN, pip/conda and CPU info,
# and prints a formatted report. Defines the SystemEnv namedtuple, package-name pattern lists,
# many get_* collectors, get_env_info, pretty_str, get_pretty_env_info and main.
# Import-time side effect: tries to import torch and sets TORCH_AVAILABLE accordingly.

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
import datetime
import json
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple
from typing import cast as _cast, Dict as _Dict


try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "is_xpu_available",
        "pip_version",  # 'pip' or 'pip3'
        "pip_packages",
        "conda_packages",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
    ],
)

COMMON_PATTERNS = [
    "torch",
    "numpy",
    "triton",
    "optree",
]

NVIDIA_PATTERNS = [
    "cuda-cudart",
    "cuda-cupti",
    "cuda-libraries",
    "cuda-opencl",
    "cuda-nvrtc",
    "cuda-runtime",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvjitlink",
    "nvtx",
]

ONEAPI_PATTERNS = [
    "dpcpp-cpp-rt",
    "intel-cmplr-lib-rt",
    "intel-cmplr-lib-ur",
    "intel-cmplr-lic-rt",
    "intel-opencl-rt",
    "intel-sycl-rt",
    "mkl",
    "onemkl-sycl-blas",
    "onemkl-sycl-dft",
    "onemkl-sycl-lapack",
    "onemkl-sycl-rng",
    "onemkl-sycl-sparse",
    "intel-openmp",
    "tbb",
    "impi-rt",
    "impi-devel",
    "oneccl",
    "oneccl-devel",
    "intel-pti",
    "umf",
    "tcmlib",
]

CONDA_PATTERNS = [
    "cudatoolkit",
    "soumith",
    "mkl",
    "magma",
]

PIP_PATTERNS = [
    "mypy",
    "flake8",
    "onnx",
]


def run(command):
    # [FUNCTION DOC] Runs a command (string via shell, otherwise an argument list) and waits.
    # Returns (return code, stdout, stderr) decoded with the OEM code page on Windows or the
    # preferred locale encoding elsewhere, both stripped. Raises OSError if the process
    # cannot start.
    """Return (return-code, stdout, stderr)."""
    shell = type(command) is str
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == "win32":
        enc = "oem"
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    # [FUNCTION DOC] Runs command through run_lambda; returns the whole output when the return
    # code is zero, otherwise None.
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    # [FUNCTION DOC] Runs command via run_lambda and returns the first capture group of the first
    # regex match in the output. Returns None on nonzero return code or when nothing matches.
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda, command):
    # [FUNCTION DOC] Runs command via run_lambda and returns the first output line, or None if the
    # return code is nonzero.
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]


def get_conda_packages(run_lambda, patterns=None):
    # [FUNCTION DOC] Lists conda packages via the conda executable (CONDA_EXE or "conda") and
    # keeps non-comment lines containing any of the patterns (default: all pattern lists).
    # Returns the filtered text, or None when the conda command fails.
    if patterns is None:
        patterns = CONDA_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS + ONEAPI_PATTERNS
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, "{} list".format(conda))
    if out is None:
        return out

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#") and any(name in line for name in patterns)
    )


def get_gcc_version(run_lambda):
    # [FUNCTION DOC] Returns the gcc version text parsed from the gcc version output, or None.
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    # [FUNCTION DOC] Returns the clang version text parsed from the clang version output, or None.
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda):
    # [FUNCTION DOC] Returns the cmake version text parsed from the cmake version output, or None.
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_driver_version(run_lambda):
    # [FUNCTION DOC] Returns the NVIDIA driver version string, or None if unavailable. On macOS it
    # parses the loaded CUDA kernel extension; elsewhere it parses the nvidia-smi output.
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda):
    # [FUNCTION DOC] Returns a description of the GPUs. On macOS or ROCm builds it uses torch
    # (device name, plus the gcn architecture name on ROCm) when a device is available, else None.
    # Otherwise it lists GPUs with nvidia-smi and strips the UUIDs; None if the command fails.
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = " ({})".format(prop.gcnArchName)
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda):
    # [FUNCTION DOC] Returns the CUDA toolkit version parsed from the nvcc version output, or None.
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    # [FUNCTION DOC] Locates cuDNN libraries using a platform-specific search command (where on
    # Windows, ls on macOS, ldconfig on Linux). Resolves symlinks, keeps existing files and
    # sorts them.
    # Returns the single path, or a "Probably one of the following" list for several. Falls back to
    # the CUDNN_LIBRARY environment variable if the search finds nothing; otherwise returns None.
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, "System32", "where")
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-mac-os-x/index.html#installation
        # https://docs.nvidia.com/deeplearning/cudnn/installation/latest/
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get("CUDNN_LIBRARY")
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files_set = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files_set.add(fn)
    if not files_set:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_nvidia_smi():
    # [FUNCTION DOC] Returns the nvidia-smi command string. On Windows it prefers an existing
    # absolute path (System32, then the legacy NVSMI folder) wrapped in quotes; else the bare name.
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = os.path.join(
            program_files_root, "NVIDIA Corporation", "NVSMI", smi
        )
        new_path = os.path.join(system_root, "System32", smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = '"{}"'.format(candidate_smi)
                break
    return smi


def _detect_linux_pkg_manager():
    # [FUNCTION DOC] HELPER: Returns the first available Linux package manager among dpkg, dnf,
    # yum and zypper (checked with which), or "N/A" when not on Linux or none is found.
    if get_platform() != "linux":
        return "N/A"
    for mgr_name in ["dpkg", "dnf", "yum", "zypper"]:
        rc, _, _ = run(f"which {mgr_name}")
        if rc == 0:
            return mgr_name
    return "N/A"


def get_linux_pkg_version(run_lambda, pkg_name):
    # [FUNCTION DOC] Looks up the installed version of a package using the detected Linux package
    # manager command and a per-manager column index. Returns the version string, or "N/A" if no
    # manager is found, the command fails or prints nothing, or the output has too few columns.
    pkg_mgr = _detect_linux_pkg_manager()
    if pkg_mgr == "N/A":
        return "N/A"

    grep_version = {
        "dpkg": {
            "field_index": 2,
            "command": "dpkg -l | grep {}",
        },
        "dnf": {
            "field_index": 1,
            "command": "dnf list | grep {}",
        },
        "yum": {
            "field_index": 1,
            "command": "yum list | grep {}",
        },
        "zypper": {
            "field_index": 2,
            "command": "zypper info {} | grep Version",
        },
    }

    field_index: int = int(_cast(int, grep_version[pkg_mgr]["field_index"]))
    cmd: str = str(grep_version[pkg_mgr]["command"])
    cmd = cmd.format(pkg_name)
    ret = run_and_read_all(run_lambda, cmd)
    if ret is None or ret == "":
        return "N/A"
    lst = re.sub(" +", " ", ret).split(" ")
    if len(lst) <= field_index:
        return "N/A"
    return lst[field_index]


def get_intel_gpu_driver_version(run_lambda):
    # [FUNCTION DOC] Returns text listing Intel GPU driver versions, one bullet per entry. On Linux
    # it reports versions of known driver packages that are installed; on Windows/Cygwin it queries
    # PowerShell and parses the JSON. A JSON parse failure appends the raw text and error message.
    # KNOWN ISSUE: if the PowerShell command fails, the text is None and json.loads raises
    # TypeError, which is not caught (only ValueError is).
    lst = []
    platform = get_platform()
    if platform == "linux":
        pkgs = {  # type: ignore[var-annotated]
            "dpkg": {
                "intel-opencl-icd",
                "libze1",
                "level-zero",
            },
            "dnf": {
                "intel-opencl",
                "level-zero",
            },
            "yum": {
                "intel-opencl",
                "level-zero",
            },
            "zypper": {
                "intel-opencl",
                "level-zero",
            },
        }.get(_detect_linux_pkg_manager(), {})
        for pkg in pkgs:
            ver = get_linux_pkg_version(run_lambda, pkg)
            if ver != "N/A":
                lst.append(f"* {pkg}:\t{ver}")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName,DriverVersion,DriverDate\
            | ConvertTo-Json"',
        )
        try:
            obj = json.loads(txt)
            if type(obj) is list:
                for o in obj:
                    lst.append(
                        f'* {o["DeviceName"]}: {o["DriverVersion"]} ({o["DriverDate"]})'
                    )
            else:
                lst.append(f'* {obj["DriverVersion"]} ({obj["DriverDate"]})')
        except ValueError as e:
            lst.append(txt)
            lst.append(str(e))
    return "\n".join(lst)


def get_intel_gpu_onboard(run_lambda):
    # [FUNCTION DOC] Returns text listing Intel GPUs present, one bullet per device. On Linux it
    # parses xpu-smi discovery JSON; on Windows/Cygwin it queries PowerShell. Reports "N/A" when
    # the command output is empty or no devices are listed, and appends raw text plus the error
    # message when parsing fails.
    lst: list[str] = []
    platform = get_platform()
    if platform == "linux":
        txt = run_and_read_all(run_lambda, "xpu-smi discovery -j")
        if txt:
            try:
                obj = json.loads(txt)
                device_list = obj.get("device_list", [])
                if isinstance(device_list, list) and device_list:
                    lst.extend(f'* {device["device_name"]}' for device in device_list)
                else:
                    lst.append("N/A")
            except (ValueError, TypeError) as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName | ConvertTo-Json"',
        )
        if txt:
            try:
                obj = json.loads(txt)
                if isinstance(obj, list) and obj:
                    lst.extend(f'* {device["DeviceName"]}' for device in obj)
                else:
                    lst.append(f'* {obj.get("DeviceName", "N/A")}')
            except ValueError as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    return "\n".join(lst)


def get_intel_gpu_detected(run_lambda):
    # [FUNCTION DOC] Returns a bullet list of XPU devices seen by torch with their properties, or
    # "N/A" if torch or its xpu module is missing or no device exists. The argument is unused.
    if not TORCH_AVAILABLE or not hasattr(torch, "xpu"):
        return "N/A"

    device_count = torch.xpu.device_count()
    if device_count == 0:
        return "N/A"

    devices = [
        f"* [{i}] {torch.xpu.get_device_properties(i)}" for i in range(device_count)
    ]
    return "\n".join(devices)


# example outputs of CPU infos
#  * linux
#    Architecture:            x86_64
#      CPU op-mode(s):        32-bit, 64-bit
#      Address sizes:         46 bits physical, 48 bits virtual
#      Byte Order:            Little Endian
#    CPU(s):                  128
#      On-line CPU(s) list:   0-127
#    Vendor ID:               GenuineIntel
#      Model name:            Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#        CPU family:          6
#        Model:               106
#        Thread(s) per core:  2
#        Core(s) per socket:  32
#        Socket(s):           2
#        Stepping:            6
#        BogoMIPS:            5799.78
#        Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr
#                             sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl
#                             xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq monitor ssse3 fma cx16
#                             pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand
#                             hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced
#                             fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap
#                             avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1
#                             xsaves wbnoinvd ida arat avx512vbmi pku ospke avx512_vbmi2 gfni vaes vpclmulqdq
#                             avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid md_clear flush_l1d arch_capabilities
#    Virtualization features:
#      Hypervisor vendor:     KVM
#      Virtualization type:   full
#    Caches (sum of all):
#      L1d:                   3 MiB (64 instances)
#      L1i:                   2 MiB (64 instances)
#      L2:                    80 MiB (64 instances)
#      L3:                    108 MiB (2 instances)
#    NUMA:
#      NUMA node(s):          2
#      NUMA node0 CPU(s):     0-31,64-95
#      NUMA node1 CPU(s):     32-63,96-127
#    Vulnerabilities:
#      Itlb multihit:         Not affected
#      L1tf:                  Not affected
#      Mds:                   Not affected
#      Meltdown:              Not affected
#      Mmio stale data:       Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
#      Retbleed:              Not affected
#      Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
#      Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
#      Spectre v2:            Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
#      Srbds:                 Not affected
#      Tsx async abort:       Not affected
#  * win32
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU0
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142
#
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU1
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142


def get_cpu_info(run_lambda):
    # [FUNCTION DOC] Returns CPU information text: lscpu output on Linux, a PowerShell processor
    # query formatted as key-value lines (JSON parsed) on Windows, and the sysctl brand string on
    # macOS. Returns the command output on success, otherwise the error text. On other platforms
    # the return code stays zero and the result is empty.
    rc, out, err = 0, "", ""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            'powershell.exe "gwmi -Class Win32_Processor | Select-Object -Property Name,Manufacturer,Family,\
            Architecture,ProcessorType,DeviceID,CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision\
            | ConvertTo-Json"'
        )
        if rc == 0:
            lst = []
            try:
                obj = json.loads(out)
                if type(obj) is list:
                    for o in obj:
                        lst.append("----------------------")
                        lst.extend([f"{k}: {v}" for (k, v) in o.items()])
                else:
                    lst.extend([f"{k}: {v}" for (k, v) in obj.items()])
            except ValueError as e:
                lst.append(out)
                lst.append(str(e))
            out = "\n".join(lst)
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = "None"
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info


def get_platform():
    # [FUNCTION DOC] Normalizes sys.platform to "linux", "win32", "cygwin" or "darwin"; any other
    # value is returned unchanged.
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version(run_lambda):
    # [FUNCTION DOC] Returns the macOS product version string from sw_vers, or None.
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    # [FUNCTION DOC] Returns a Windows description "Caption (Version Architecture)" from a
    # PowerShell query. On JSON parse failure the raw text gets the error message appended.
    # KNOWN ISSUE: if the query fails the text is None, so json.loads raises an uncaught TypeError.
    ret = run_and_read_all(
        run_lambda,
        'powershell.exe "gwmi -Class Win32_OperatingSystem | Select-Object -Property Caption,\
        OSArchitecture,Version | ConvertTo-Json"',
    )
    try:
        obj = json.loads(ret)
        ret = f'{obj["Caption"]} ({obj["Version"]} {obj["OSArchitecture"]})'
    except ValueError as e:
        ret += f"\n{str(e)}"
    return ret


def get_lsb_version(run_lambda):
    # [FUNCTION DOC] Returns the distribution description from lsb_release, or None.
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )


def check_release_file(run_lambda):
    # [FUNCTION DOC] Returns the PRETTY_NAME value from the /etc release files, or None.
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_os(run_lambda):
    # [FUNCTION DOC] Returns a human-readable OS description. Windows uses PowerShell, macOS uses
    # sw_vers (None if unavailable), and Linux tries lsb_release, then the /etc release files, then
    # the bare platform name, appending the machine architecture. Other platforms return
    # their name.
    from platform import machine

    platform = get_platform()

    if platform in ["win32", "cygwin"]:
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "macOS {} ({})".format(version, machine())

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        return "{} ({})".format(platform, machine())

    # Unknown platform
    return platform


def get_python_platform():
    # [FUNCTION DOC] Returns the platform description string from the platform module.
    import platform

    return platform.platform()


def get_libc_version():
    # [FUNCTION DOC] Returns the libc name and version joined by a dash on Linux, otherwise "N/A".
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_pip_packages(run_lambda, patterns=None):
    # [FUNCTION DOC] Runs the pip list command through the current interpreter and keeps lines that
    # contain any pattern (default: all pattern lists). Returns (pip name, filtered text),
    # where the name is "pip3" on Python 3; the text is None if pip fails. Side effect:
    # sets the PIP_DISABLE_PIP_VERSION_CHECK environment variable.
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = PIP_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS + ONEAPI_PATTERNS

    pip_version = "pip3" if sys.version_info.major == 3 else "pip"

    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # People generally have pip as `pip` or `pip3`
    # But here it is invoked as `python -mpip`
    out = run_and_read_all(
        run_lambda, [sys.executable, "-mpip", "list", "--format=freeze"]
    )
    if out is None:
        return pip_version, out

    filtered_out = "\n".join(
        line for line in out.splitlines() if any(name in line for name in patterns)
    )

    return pip_version, filtered_out


def get_cachingallocator_config() -> _Dict[str, str]:
    # [FUNCTION DOC] Returns a mapping of the CUDA, HIP and generic allocator configuration
    # environment variables that are set and non-empty to their values.
    """Return the caching allocator configuration from environment variables.
    """
    # pyrefly: ignore [bad-return]
    return {
        var: os.environ.get(var)
        for var in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        )
        if os.environ.get(var)
    }


def get_cuda_module_loading_config():
    # [FUNCTION DOC] Returns the CUDA_MODULE_LOADING setting (empty string if unset) when torch is
    # present and CUDA is available, initializing CUDA as a side effect; otherwise "N/A".
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        config = os.environ.get("CUDA_MODULE_LOADING", "")
        return config
    else:
        return "N/A"


def is_xnnpack_available():
    # [FUNCTION DOC] Returns the string form of whether XNNPACK is enabled in torch, or "N/A" when
    # torch is not importable.
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack

        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]
    else:
        return "N/A"


def get_env_info():
    # [FUNCTION DOC] Collects all environment details into a SystemEnv tuple by running the
    # collectors above. Without torch, torch-derived fields are "N/A". Extends the XPU entry with
    # driver, onboard and detected device info when XPU is available, and fills ROCm versions from
    # the torch config when built with HIP. May initialize CUDA and set an environment variable.
    """
    Collects environment information to aid in debugging.

    The returned environment information contains details on torch version, is debug build
    or not, cuda compiled version, gcc version, clang version, cmake version, operating
    system, libc version, python version, python platform, CUDA availability, CUDA
    runtime version, CUDA module loading config, GPU model and configuration, Nvidia
    driver version, cuDNN version, pip version and versions of relevant pip and
    conda packages, HIP runtime version, MIOpen runtime version,
    Caching allocator config, XNNPACK availability and CPU information.

    Returns:
        SystemEnv (namedtuple): A tuple containing various environment details
            and system information.
    """
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        xpu_available_str = str(torch.xpu.is_available())
        if torch.xpu.is_available():
            xpu_available_str = (
                f"{xpu_available_str}\n"
                + f"XPU used to build PyTorch: {torch.version.xpu}\n"
                + f"Intel GPU driver version:\n{get_intel_gpu_driver_version(run_lambda)}\n"
                + f"Intel GPU models onboard:\n{get_intel_gpu_onboard(run_lambda)}\n"
                + f"Intel GPU models detected:\n{get_intel_gpu_detected(run_lambda)}"
            )
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version

            def get_version_or_na(cfg, prefix):
                # [FUNCTION DOC] HELPER: Returns the last whitespace-separated token of the first
                # config line containing prefix, or "N/A" if no line matches.
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else "N/A"

            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = xpu_available_str = "N/A"  # type: ignore[assignment]
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    conda_packages = get_conda_packages(run_lambda)

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        is_xpu_available=xpu_available_str,
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=conda_packages,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
Is XPU available: {is_xpu_available}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}
Caching allocator config: {caching_allocator_config}

CPU:
{cpu_info}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo):
    # [FUNCTION DOC] Formats a SystemEnv into the report text using env_info_fmt. Applies the
    # cleanup steps: "No CUDA" markers when torch has no CUDA and the dynamic CUDA fields are
    # missing, booleans to Yes/No, None to "Could not collect", empty package lists to a notice,
    # and package lines prefixed with the pip or conda tag. Empty allocator config becomes "N/A".
    def replace_nones(dct, replacement="Could not collect"):
        # [FUNCTION DOC] HELPER: Replaces None values in the mapping, in place, with the
        # replacement text.
        for key in dct:
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        # [FUNCTION DOC] HELPER: Replaces True and False values in the mapping, in place,
        # with the given strings.
        for key in dct:
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        # [FUNCTION DOC] HELPER: Prefixes every line of the text with the tag and returns
        # the joined result.
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text, replacement="No relevant packages"):
        # [FUNCTION DOC] HELPER: Returns the replacement for an empty string; other values
        # are returned unchanged.
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # [FUNCTION DOC] HELPER: Returns the string wrapped in newlines when it has multiple lines,
        # so it starts on its own line; otherwise returns it unchanged.
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(
        envinfo.nvidia_gpu_models
    )

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = dynamic_cuda_fields + ["cudnn_version"]
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields
    )
    if (
        TORCH_AVAILABLE
        and not torch.cuda.is_available()
        and all_dynamic_cuda_fields_missing
    ):
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"
        if envinfo.cuda_compiled_version is None:
            mutable_dict["cuda_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            mutable_dict["conda_packages"], "[conda] "
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info
    mutable_dict["caching_allocator_config"] = envinfo.caching_allocator_config
    if not envinfo.caching_allocator_config:
        mutable_dict["caching_allocator_config"] = "N/A"
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    # [FUNCTION DOC] Returns the formatted environment report string for get_env_info.
    """
    Returns a pretty string of environment information.

    This function retrieves environment information by calling the `get_env_info` function
    and then formats the information into a human-readable string. The retrieved environment
    information is listed in the document of `get_env_info`.
    This function is used in `python collect_env.py` that should be executed when reporting a bug.

    Returns:
        str: A pretty string of the environment information.
    """
    return pretty_str(get_env_info())


def main() -> None:
    # [FUNCTION DOC] Command-line entry: prints a progress line and the report to stdout.
    # On Linux, if a crash-handler minidump directory exists, prints a notice with the newest
    # dump path and creation time to stderr.
    # KNOWN ISSUE: an existing but empty minidump directory makes max raise ValueError on an
    # empty list.
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = (
                "\n*** Detected a minidump at {} created on {}, ".format(
                    latest, creation_time
                )
                + "if this is related to your bug please include it when you file a report ***"
            )
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
