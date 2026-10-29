# mypy: ignore-errors
# [MODULE DOC] Test fixture counterpart of future_div that imports torch and defines two
# constant-returning division functions, for TorchScript division semantics tests.

import torch  # noqa: F401


def div_int_nofuture():
    # [FUNCTION DOC] HELPER: Returns the result of dividing integer 1 by integer 2.
    return 1 / 2


def div_float_nofuture():
    # [FUNCTION DOC] HELPER: Returns the float result of dividing 3.14 by 0.125.
    return 3.14 / 0.125
