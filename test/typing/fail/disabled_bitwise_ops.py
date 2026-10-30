# flake8: noqa
# [MODULE DOC] Negative typing test. Type-checker input expecting an unsupported-operand
# error when a plain int is the left operand of the bitwise or operator and the right operand
# is a float64 tensor. The comment lists the operator family the typing stubs restrict.
import torch


# binary ops: <<, >>, |, &, ~, ^

a = torch.ones(3, dtype=torch.float64)
i = int()

i | a  # E: Unsupported operand types
