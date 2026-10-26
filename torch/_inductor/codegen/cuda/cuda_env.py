# [MODULE DOC] Helpers that query the CUDA environment for Inductor's CUDA codegen:
# compute-capability string, CUDA toolkit version, Blackwell detection and nvcc
# availability. Results are memoized and the arch/version caches are cleared on
# fresh inductor cache. Config overrides come from config.cuda. No import-time
# side effects beyond creating a logger.
import functools
import logging
import shutil

import torch
from torch._inductor.utils import clear_on_fresh_cache

from ... import config


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None:
    # [FUNCTION DOC] Returns the CUDA compute capability as a string such as "90".
    # Uses config.cuda.arch if set, else the capability of device 0 (major*10+minor).
    # Returns None (and logs the exception) on any failure. Cached (size 1) and
    # cleared on fresh inductor cache.
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception:
        log.exception("Error getting cuda arch")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def is_datacenter_blackwell_arch() -> bool:
    # [FUNCTION DOC] True when the detected arch is in [100, 110), i.e. datacenter
    # Blackwell (sm100 family). False if the arch is unknown. Cached (size 1).
    arch = get_cuda_arch()
    if arch is None:
        return False
    arch_number = int(arch)
    return arch_number >= 100 and arch_number < 110


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None:
    # [FUNCTION DOC] Returns the CUDA version string: config.cuda.version if set,
    # else torch.version.cuda (may be None for non-CUDA builds). Logs and returns
    # None on exception. Cached (size 1).
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception:
        log.exception("Error getting cuda version")
        return None


@functools.cache
def nvcc_exist(nvcc_path: str | None = "nvcc") -> bool:
    # [FUNCTION DOC] True if nvcc_path is not None and resolves to an executable on
    # PATH (via shutil.which). Cached per argument.
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
