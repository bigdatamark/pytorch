# [MODULE DOC] Helper script that runs a small convolution model under the MKLDNN verbose context
# manager so that oneDNN prints its verbose logging at the requested level. When run as a program
# it accepts an integer verbose level option (default 0), and prints any exception instead of
# raising it.
import argparse

import torch


class Module(torch.nn.Module):
    # [CLASS DOC] HELPER: Model wrapping one 2D convolution with 1 input channel, 10 output
    # channels, kernel 5 and stride 1.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the convolution layer.
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the convolution to the input and returns the result.
        y = self.conv(x)
        return y


def run_model(level):
    # [FUNCTION DOC] Puts the model in eval mode and runs it on a random 1 by 1 by 112 by 112 input
    # inside the MKLDNN verbose context for the given level. Returns nothing.
    m = Module().eval()
    d = torch.rand(1, 1, 112, 112)
    with torch.backends.mkldnn.verbose(level):
        m(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    try:
        run_model(args.verbose_level)
    except Exception as e:
        print(e)
