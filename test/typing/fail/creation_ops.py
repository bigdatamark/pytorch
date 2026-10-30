# flake8: noqa
# [MODULE DOC] Negative typing test. Type-checker input expecting errors when a string is
# passed as the dtype to the tensor, ones and zeros creation functions instead of a dtype
# object. Expected messages are given in the trailing error markers on each flagged line.
import torch


torch.tensor(
    [3],
    dtype="int32",  # E: Argument "dtype" to "tensor" has incompatible type "str"; expected "dtype | None"  [arg-type]
)
torch.ones(  # E: No overload variant of "ones" matches argument types "int", "str"
    3, dtype="int32"
)
torch.zeros(  # E: No overload variant of "zeros" matches argument types "int", "str"
    3, dtype="int32"
)
