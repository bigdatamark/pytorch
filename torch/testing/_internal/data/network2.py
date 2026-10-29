# mypy: ignore-errors
# [MODULE DOC] HELPER: Test fixture defining a minimal module class Net with a linear layer
# (10 in, 20 out) and a ReLU. Differs from network1 only by the extra relu attribute, so tests can
# tell the two definitions apart. No import-time side effects beyond importing torch.nn.

import torch.nn as nn


class Net(nn.Module):
    # [CLASS DOC] HELPER: Tiny module with attributes linear (10-to-20 linear layer) and relu.
    # It defines no forward method, so it is only meant for state and structure tests.

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Constructs the module and creates the linear and relu attributes.
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.relu = nn.ReLU()
