# [MODULE DOC] CI smoke script for a two-layer RNN on the first CUDA device (cuda:0).
# Runs at import time (no main guard): builds the RNN and random input and initial hidden
# state, then runs one forward pass. Any missing component or absent GPU raises an uncaught
# exception (nonzero exit). Prints nothing.
r"""
It's used to check basic rnn features with cuda.
For example, it would throw exception if missing some components are missing
"""

import torch
import torch.nn as nn


device = torch.device("cuda:0")
rnn = nn.RNN(10, 20, 2).to(device)
inputs = torch.randn(5, 3, 10).to(device)
h0 = torch.randn(2, 3, 20).to(device)
output, hn = rnn(inputs, h0)
