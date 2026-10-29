# [MODULE DOC] Test helpers for Intel XPU devices: enums for GPU codename (PVC, BMG) and
# architecture (Xe, Xe2), a table mapping PCI device ids to codenames and codenames to
# architectures, cached lookups for the current device, and lazy platform-support flags such as
# PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU. Import-time side effects: records whether XPU was already
# initialized and raises an error if importing this module itself initialized XPU.

import enum
import functools

import torch
import torch.xpu
from torch.testing._internal.common_utils import IS_WINDOWS, LazyVal, TEST_XPU


XPU_ALREADY_INITIALIZED_ON_IMPORT = torch.xpu.is_initialized()


class XPUCodename(enum.Enum):
    # [CLASS DOC] HELPER: Enumeration of known Intel GPU product codenames (PVC = Data Center GPU
    # Max, BMG = Arc Pro Battlemage); values are the codename strings.
    PVC = "PVC"  # Intel® Data Center GPU Max Series
    BMG = "BMG"  # Intel® Arc™ Pro Battlemage Graphics


class XPUArch(enum.IntEnum):
    # [CLASS DOC] HELPER: Ordered integer enumeration of Intel GPU architecture generations
    # (Unknown < Xe < Xe2), so callers can compare with greater-or-equal for "this or later".
    Unknown = 0
    Xe = 1  # Xe HPC
    Xe2 = 2


# device_id -> GPU codename
# From https://github.com/intel/intel-graphics-compiler/blob/master/inc/common/igfxfmid.h
_DEVICE_ID_TO_CODENAME = {
    0x0BD0: XPUCodename.PVC,
    0x0BD4: XPUCodename.PVC,
    0x0BD5: XPUCodename.PVC,
    0x0BD6: XPUCodename.PVC,
    0x0BD7: XPUCodename.PVC,
    0x0BD8: XPUCodename.PVC,
    0x0BD9: XPUCodename.PVC,
    0x0BDA: XPUCodename.PVC,
    0x0BDB: XPUCodename.PVC,
    0x0B69: XPUCodename.PVC,
    0x0B6E: XPUCodename.PVC,
    0xE202: XPUCodename.BMG,
    0xE20B: XPUCodename.BMG,
    0xE20C: XPUCodename.BMG,
    0xE20D: XPUCodename.BMG,
    0xE210: XPUCodename.BMG,
    0xE211: XPUCodename.BMG,
    0xE212: XPUCodename.BMG,
    0xE215: XPUCodename.BMG,
    0xE216: XPUCodename.BMG,
    0xE220: XPUCodename.BMG,
    0xE221: XPUCodename.BMG,
    0xE222: XPUCodename.BMG,
    0xE223: XPUCodename.BMG,
}

# GPU codename -> architecture
_CODENAME_TO_ARCH = {
    XPUCodename.PVC: XPUArch.Xe,
    XPUCodename.BMG: XPUArch.Xe2,
}


@functools.lru_cache(1)
def get_xpu_codename() -> XPUCodename | None:
    # [FUNCTION DOC] Returns the codename of the current XPU device by looking up its device id
    # (from the device capability info) in the id-to-codename table; None if the id is unknown.
    # Result is cached for the process, and querying the device initializes XPU.
    device_id = torch.xpu.get_device_capability()["device_id"]
    return _DEVICE_ID_TO_CODENAME.get(device_id)


@functools.lru_cache(1)
def get_xpu_arch() -> XPUArch | None:
    # [FUNCTION DOC] Returns the architecture of the current XPU device from its codename, or
    # XPUArch.Unknown when the codename is unknown. Cached for the process.
    # KNOWN ISSUE: the return annotation allows None, but the lookup always has a default, so
    # None is never returned.
    codename = get_xpu_codename()
    return _CODENAME_TO_ARCH.get(codename, XPUArch.Unknown)


Xe2_Or_Later = LazyVal(
    lambda: torch.xpu.is_available() and get_xpu_arch() >= XPUArch.Xe2
)


def evaluate_platform_supports_flash_attention():
    # [FUNCTION DOC] Decides whether flash attention is supported on XPU: true only when XPU
    # tests are enabled, the OS is not Windows, and the device is Xe2 or later; false otherwise.
    # KNOWN ISSUE: Xe2_Or_Later is a lazy value object, not a plain bool; it is returned
    # unevaluated and relies on the lazy object's truthiness at the point of use.
    if TEST_XPU:
        return not IS_WINDOWS and Xe2_Or_Later
    return False


PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU: bool = LazyVal(
    lambda: evaluate_platform_supports_flash_attention()
)

# Importing this module should NOT eagerly initialize XPU
if not XPU_ALREADY_INITIALIZED_ON_IMPORT:
    if torch.xpu.is_initialized():
        raise AssertionError("XPU should not be initialized on import")
