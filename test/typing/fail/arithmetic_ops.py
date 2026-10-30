# flake8: noqa
# [MODULE DOC] Negative typing test. Type-checker input that must report errors: it applies
# the bitwise operators and and or and xor with a Python float on the left and a Tensor on
# the right, each tagged with the expected unsupported-operand message.
# Shift operators and the Tensor-on-left forms are commented out because the checker
# currently misses them (false negatives, see the linked issue). Only runs under the type
# checker; at import time it would just build a random tensor.
from typing import Any
from typing_extensions import assert_type

from torch import randn, Tensor


# See ../pass/arithmetic_ops.py for more information

TENSOR, FLOAT = randn(3), 1.5

FLOAT & TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
FLOAT | TENSOR  # E: Unsupported operand types for | ("float" and "Tensor")
FLOAT ^ TENSOR  # E: Unsupported operand types for ^ ("float" and "Tensor")
# FIXME: false negatives (https://github.com/pytorch/pytorch/issues/155701)
#
# FLOAT << TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
# FLOAT >> TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
#
# TENSOR & FLOAT  # E: Unsupported operand types for & ("Tensor" and "float" )
# TENSOR | FLOAT  # E: Unsupported operand types for | ("Tensor" and "float" )
# TENSOR ^ FLOAT  # E: Unsupported operand types for ^ ("Tensor" and "float" )
# TENSOR << FLOAT  # E: Unsupported operand types for & ("Tensor" and "float")
# TENSOR >> FLOAT  # E: Unsupported operand types for & ("Tensor" and "float")
