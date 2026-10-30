# [PACKAGE DOC] Compatibility shim that re-exports two private helpers (assert-wrapped
# and unwrap functional tensor) from torch._functorch.eager_transforms. Not public
# API.
# This file has moved to under torch/_functorch. It is not public API.
# If you are not a PyTorch developer and you are relying on the following
# imports, please file an issue.
from torch._functorch.eager_transforms import (
    _assert_wrapped_functional,
    _unwrap_functional_tensor,
)
