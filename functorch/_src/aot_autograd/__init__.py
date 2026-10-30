# [PACKAGE DOC] Compatibility shim that re-exports a few names
# (aot_autograd_decompositions, KNOWN_TYPES, PytreeThunk) from
# torch._functorch.aot_autograd. Not public API.
# This file has moved to under torch/_functorch. It is not public API.
# If you are not a PyTorch developer and you are relying on the following
# imports, please file an issue.
from torch._functorch.aot_autograd import (
    aot_autograd_decompositions,
    KNOWN_TYPES,
    PytreeThunk,
)
