# [PACKAGE DOC] torch_openreg.openreg.amp: automatic mixed precision support for openreg.
# Exports get_amp_supported_dtype, which lists the low-precision dtypes usable for autocast.
import torch


# LITERALINCLUDE START: AMP GET_SUPPORTED_DTYPE
def get_amp_supported_dtype():
    # [FUNCTION DOC] Returns a list of the dtypes openreg supports under autocast: float16
    # and bfloat16.
    return [torch.float16, torch.bfloat16]


# LITERALINCLUDE END: AMP GET_SUPPORTED_DTYPE
