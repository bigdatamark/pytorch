# [MODULE DOC] Helper script that runs a small linear layer under the MKL verbose context manager
# so that MKL prints its verbose logging at the requested level. When run as a program it accepts
# an integer verbose level option (default 0), and prints any exception instead of raising it.
import argparse

import torch


def run_model(level):
    # [FUNCTION DOC] Runs a 20 to 30 linear layer on a random 128 by 20 input inside the
    # MKL verbose context for the given level. Returns nothing; output is the library's
    # verbose logging.
    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    with torch.backends.mkl.verbose(level):
        m(input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    try:
        run_model(args.verbose_level)
    except Exception as e:
        print(e)
