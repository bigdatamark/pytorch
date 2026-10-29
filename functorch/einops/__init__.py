# [PACKAGE DOC]
# functorch.einops package: re-exports the einops-style rearrange operation, which is
# implemented in torch._functorch.einops. Exports: rearrange.
# Import from the canonical location in torch.func
from torch._functorch.einops import rearrange


__all__ = ["rearrange"]
