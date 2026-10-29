# [MODULE DOC] Meta-device (shape inference) kernels for the openreg custom operators. At
# import time creates a Library handle on the "openreg" namespace for the Meta key and
# registers the custom_abs meta implementation.
import torch


# LITERALINCLUDE START: CUSTOM OPERATOR META
lib = torch.library.Library("openreg", "IMPL", "Meta")  # noqa: SCOPED_LIBRARY


@torch.library.impl(lib, "custom_abs")
def custom_abs(self):
    # [FUNCTION DOC] Meta kernel for the custom_abs operator: returns an uninitialized tensor
    # with the same shape, dtype and device properties as the input (no data computed).
    return torch.empty_like(self)


# LITERALINCLUDE END: CUSTOM OPERATOR META
