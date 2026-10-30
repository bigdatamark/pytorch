# [PACKAGE DOC] Compatibility shim that re-exports the private _swap_state helper from
# torch._functorch.make_functional. Not public API.
# This file has moved to under torch/_functorch. It is not public API.
# If you are not a PyTorch developer and you are relying on the following
# imports, please file an issue.
from torch._functorch.make_functional import _swap_state
