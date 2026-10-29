# [PACKAGE DOC] Deprecated alias package exporting the dynamic QAT Linear (see __all__).
# The class is re-exported from torch.nn.qat.dynamic.modules.linear, itself a re-export of the
# implementation in torch.ao.nn.qat.dynamic.modules.linear.
from torch.nn.qat.dynamic.modules.linear import Linear


__all__ = ["Linear"]
