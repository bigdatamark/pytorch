# [MODULE DOC] Compatibility shim re-exporting the HigherOrderOperator base class from
# torch._ops. No logic and no import-time side effects beyond the import.
from torch._ops import HigherOrderOperator  # noqa: F401
