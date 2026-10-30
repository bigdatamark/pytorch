# [PACKAGE DOC] Package for the fastrnns benchmarks. Re-exports everything from the cells and
# factory submodules and defines default sizes: sequence length 100, 2 layers, input and hidden
# size 512, mini-batch 64.
from .cells import *  # noqa: F403
from .factory import *  # noqa: F403


# (output, next_state) = cell(input, state)
seqLength = 100
numLayers = 2
inputSize = 512
hiddenSize = 512
miniBatch = 64
