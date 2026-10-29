# [MODULE DOC] CI smoke script for a tiny CNN on the first CUDA device (cuda:0).
# Runs at import time (no main guard): builds a one-layer conv/pool network, runs one forward
# pass, then one loss/backward/SGD step on random data and prints the output. Any missing
# component or absent GPU surfaces as an uncaught exception (nonzero exit). Exits 0 on success.
# KNOWN ISSUE: the module docstring says "rnn" but this script exercises a CNN.
r"""
It's used to check basic rnn features with cuda.
For example, it would throw exception if some components are missing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleCNN(nn.Module):
    # [CLASS DOC] HELPER: Minimal network for the smoke test. State: a 1-to-1 channel 3x3
    # convolution and a 2x2 max pool. Maps a single-channel 5x5 image to one scalar.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Builds the convolution and pooling layers (no arguments).
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Applies convolution, relu and pooling, then flattens the
        # result to a one-element tensor. Input must be shaped (1, 1, 5, 5) or the
        # reshape raises an error.
        output = self.pool(F.relu(self.conv(inputs)))
        output = output.view(1)
        return output


# Mock one infer
device = torch.device("cuda:0")
net = SimpleCNN().to(device)
net_inputs = torch.rand((1, 1, 5, 5), device=device)
outputs = net(net_inputs)
print(outputs)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

# Mock one step training
label = torch.full((1,), 1.0, dtype=torch.float, device=device)
loss = criterion(outputs, label)
loss.backward()
optimizer.step()
