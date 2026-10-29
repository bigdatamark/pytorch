# mypy: ignore-errors
# [MODULE DOC] HELPER: Test fixture defining a minimal module class Net with one linear layer
# (10 in, 20 out). Used by tests that save, load or import model definitions by module path.
# No import-time side effects beyond importing torch.nn.

import torch.nn as nn


class Net(nn.Module):
    # [CLASS DOC] HELPER: Tiny module with a single attribute, linear, a 10-to-20 linear layer.
    # It defines no forward method, so it is only meant for state and structure tests.

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Constructs the module and creates the linear layer attribute.
        super().__init__()
        self.linear = nn.Linear(10, 20)
