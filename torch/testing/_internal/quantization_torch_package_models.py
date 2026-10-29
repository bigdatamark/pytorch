# mypy: ignore-errors
# [MODULE DOC] Small nn.Module fixtures (linear + relu using functional ops) for testing
# quantization with torch.package. Defines LinearReluFunctionalChild and
# LinearReluFunctional. No import-time side effects.

import math

import torch
import torch.nn as nn


class LinearReluFunctionalChild(nn.Module):
    # [CLASS DOC] HELPER: Module holding an N by N weight and an N bias as parameters; its
    # forward applies functional linear followed by functional relu.
    def __init__(self, N):
        # [FUNCTION DOC] HELPER: Creates parameter w1 (N by N, uninitialized then
        # kaiming-uniform initialized with a=sqrt(5), consuming global RNG) and b1 (N zeros).
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(N, N))
        self.b1 = nn.Parameter(torch.zeros(N))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns relu(linear(x, w1, b1)) using functional ops.
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        x = torch.nn.functional.relu(x)
        return x

class LinearReluFunctional(nn.Module):
    # [CLASS DOC] HELPER: Module with a LinearReluFunctionalChild submodule plus its own
    # N by N weight w1 and bias b1; forward runs the child, then a linear and a relu.
    def __init__(self, N):
        # [FUNCTION DOC] HELPER: Creates the child module and parameters w1 (kaiming-uniform
        # initialized, consuming global RNG) and b1 (zeros), all of size N.
        super().__init__()
        self.child = LinearReluFunctionalChild(N)
        self.w1 = nn.Parameter(torch.empty(N, N))
        self.b1 = nn.Parameter(torch.zeros(N))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the child, then functional linear with w1 and b1,
        # then relu; returns the result.
        x = self.child(x)
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        x = torch.nn.functional.relu(x)
        return x
