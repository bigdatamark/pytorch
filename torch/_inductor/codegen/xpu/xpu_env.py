# [MODULE DOC] Queries about the XPU hardware/software environment used by inductor XPU codegen:
# architecture name and runtime version. Results are cached and cleared on a fresh inductor cache.
import functools
import logging

import torch
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_arch() -> str | None:
    # [FUNCTION DOC] Return the Intel GPU architecture name (Xe12 for PVC, Xe20 for BMG) from the detected XPU codename.
    # Returns None and logs a warning if the codename is missing or unknown. Cached (single entry) and cleared on fresh cache.
    from torch.testing._internal.common_xpu import get_xpu_codename, XPUCodename

    name2arch = {
        XPUCodename.PVC: "Xe12",
        XPUCodename.BMG: "Xe20",
    }

    codename = get_xpu_codename()
    if not codename or codename not in name2arch:
        log.warning("Unknown XPU codename, cannot determine architecture")
        return None

    return name2arch[codename]


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_version() -> str | None:
    # [FUNCTION DOC] Return the XPU runtime version string from the torch version info (empty string if none).
    # Returns None and logs the exception if reading fails. Cached (single entry) and cleared on fresh cache.
    # string of version, like 20250101
    try:
        xpu_version = torch.version.xpu or ""
        return xpu_version
    except Exception:
        log.exception("Error getting xpu version")
        return None
