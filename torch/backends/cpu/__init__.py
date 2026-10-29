# [PACKAGE DOC] CPU backend package; exports get_cpu_capability, which reports the CPU vector instruction capability.
import torch


__all__ = [
    "get_cpu_capability",
]


def get_cpu_capability() -> str:
    # [FUNCTION DOC] Return the detected CPU capability as a string (e.g. DEFAULT, VSX, Z VECTOR, NO AVX, AVX2, AVX512, SVE256) by querying the C++ backend.
    r"""Return cpu capability as a string value.

    Possible values:
    - "DEFAULT"
    - "VSX"
    - "Z VECTOR"
    - "NO AVX"
    - "AVX2"
    - "AVX512"
    - "SVE256"
    """
    return torch._C._get_cpu_capability()
