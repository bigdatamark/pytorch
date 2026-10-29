# [MODULE DOC] HELPER: script that writes two dummy model files for loader tests. Run as a program
# with one argument, the output path: saves a TorchScript-compiled MLP there and a plain eager
# Sequential of the same layers to the same path plus the suffix .orig. Defines NeuralNetwork.
# Importing it has no side effects.
# Usage: python create_dummy_model.py <name_of_the_file>
import sys

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    # [CLASS DOC] HELPER: MNIST-style MLP: flatten, then linear 784 to 512, ReLU, linear 512 to
    # 512, ReLU, linear 512 to 10. Holds the submodules flatten and linear_relu_stack.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: builds the flatten module and the five-layer stack (three linear
        # layers separated by two ReLUs); no arguments.
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: flattens the input batch, runs the stack and returns the logits
        # (10 per sample).
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    jit_module = torch.jit.script(NeuralNetwork())
    torch.jit.save(jit_module, sys.argv[1])
    orig_module = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    torch.save(orig_module, sys.argv[1] + ".orig")
