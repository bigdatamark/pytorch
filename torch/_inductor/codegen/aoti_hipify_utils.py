# [MODULE DOC] Utility for converting generated CUDA source text to HIP (ROCm) text
# for AOT Inductor codegen. Exposes one function that applies the hipify_torch
# token mapping to a string of generated code. No import-time side effects.
import re

import torch


# It is not a good idea to directly apply hipify_torch to codegen, which will be vulnerable to cases like:
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# In such cases, we do not need to hipify_torch the original class/file name in codegen/codecache


def maybe_hipify_code_wrapper(source_codes: str, force_hipify: bool = False) -> str:
    # [FUNCTION DOC] Returns source_codes with CUDA identifiers replaced by their HIP
    # equivalents using the hipify PYTORCH_MAP/PYTORCH_TRIE tables. Returns the input
    # unchanged when not on a HIP build (unless force_hipify is true) or when the hipify
    # module cannot be imported. Uses a custom regex that has no leading word-boundary
    # requirement (so tokens at line start match) but keeps a trailing boundary (or end
    # of string) so repeated application is idempotent. Pure; no side effects.
    if torch.version.hip is None and not force_hipify:
        return source_codes

    try:
        from torch.utils.hipify.hipify_python import PYTORCH_MAP, PYTORCH_TRIE
    except ImportError:
        # hipify not available for non-AMD builds
        return source_codes

    def c2_repl(m: re.Match[str]) -> object:
        # [FUNCTION DOC] HELPER: regex substitution callback; looks up the matched CUDA
        # token in PYTORCH_MAP and returns its HIP replacement. Raises KeyError if absent.
        return PYTORCH_MAP[m.group(0)]

    # We need to redefine RE_PYTORCH_PREPROCESSOR here since in hipify_torch,
    # it will apply positive lookbehind (?<=\W) to the pattern to avoid matching
    # keyword at the beginning of code line. However, this can happen in codegen,
    # which will cause the pattern to not match.

    # Note that lookahead (?=\W|$) is still needed to keep hipification idempotent, for example
    # we need to skip replacing "getStreamFromExternal" in "getStreamFromExternalMasqueradingAsCUDA".
    # End-of-string must also count as a boundary so bare-token inputs like "CUfunction"
    # (returned by device_codegen.cpp_kernel_type()) get hipified too.
    RE_PYTORCH_PREPROCESSOR = re.compile(rf"({PYTORCH_TRIE.export_to_regex()})(?=\W|$)")

    source_codes = RE_PYTORCH_PREPROCESSOR.sub(c2_repl, source_codes)  # type: ignore[arg-type]
    return source_codes
