# [MODULE DOC] Command-line helper that checks a serialized TorchScript model. Run as a program
# with the model path as the first argument. It loads the scripted model from that path, loads a
# pickled original module from the same path with an extra orig suffix, prints the scripted module,
# runs the original on a random 2 by 784 input, and exits with status 0. No functions are defined.
import sys

import torch


if __name__ == "__main__":
    script_mod = torch.jit.load(sys.argv[1])
    # weights_only=False as this is loading a sharded model
    mod = torch.load(sys.argv[1] + ".orig", weights_only=False)
    print(script_mod)
    inp = torch.rand(2, 28 * 28)
    _ = mod(inp)
    sys.exit(0)
