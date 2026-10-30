# [MODULE DOC] Tiny module and function used by the framework overhead benchmark: a looped tensor
# addition function and an nn.Module that applies a supplied addition callable. Imports the loop
# count from the sibling utils module (script-style import, so the benchmark directory must be on
# the import path).
from utils import NUM_LOOP_ITERS

import torch


def add_tensors_loop(x, y):
    # [FUNCTION DOC] Add y to x, then add x repeatedly NUM_LOOP_ITERS more times, so a thousand
    # and one tensor additions happen per call. Returns the final tensor.
    z = torch.add(x, y)
    for i in range(NUM_LOOP_ITERS):
        z = torch.add(z, x)
    return z


class SimpleAddModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping a callable of two tensors (stored as add_op); forward just
    # applies it. Meant to be traced or run eagerly by the benchmark wrapper.
    def __init__(self, add_op):
        # [FUNCTION DOC] Initialize the base module and store the callable as add_op.
        super().__init__()
        self.add_op = add_op

    def forward(self, x, y):
        # [FUNCTION DOC] Return the callable applied to the two inputs.
        return self.add_op(x, y)
