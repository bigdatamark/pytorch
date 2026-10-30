# [MODULE DOC] Script that scripts a trivial model (adds ten to the input) with TorchScript
# and saves it as aot_test_model.pt in the current directory. Used as input for the mobile NNC
# ahead-of-time compilation test. Does its work at import time.
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    # [CLASS DOC] Minimal nn.Module with no parameters whose forward adds a constant to input.
    def forward(self, x):
        # [FUNCTION DOC] Returns the input tensor plus the scalar ten via torch.add.
        return torch.add(x, 10)


model = NeuralNetwork()
script = torch.jit.script(model)
torch.jit.save(script, "aot_test_model.pt")
