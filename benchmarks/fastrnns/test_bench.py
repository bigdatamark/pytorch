# [MODULE DOC] Pytest benchmark tests for the fastrnns models using the pytest-benchmark plugin.
# A class-scoped fixture builds a model definition for each parametrized network name (supplied
# by the conftest), and TestBenchNetwork times its forward and backward passes on cuda.
import pytest

import torch

from .fuser import set_fuser
from .runner import get_nn_runners


@pytest.fixture(scope="class")
def modeldef(request, net_name, executor, fuser):
    # [FUNCTION DOC] Class-scoped pytest fixture: set the fuser and executor from the
    # parametrized values, look up the runner for net_name and build its model definition with
    # default sizes on cuda (seqLength 100, one layer, 512 sizes, batch 64, no seed). Returns the
    # ModelDef.
    set_fuser(fuser, executor)

    # Given a 'net_name' provided by generate_tests, build the thing
    name, rnn_creator, context = get_nn_runners(net_name)[0]
    creator_args = {
        "seqLength": 100,
        "numLayers": 1,
        "inputSize": 512,
        "hiddenSize": 512,
        "miniBatch": 64,
        "device": "cuda",
        "seed": None,
    }
    return rnn_creator(**creator_args)


def cuda_sync(func, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Call func with the arguments, synchronize cuda so asynchronous work
    # is included in the timing, and return the result.
    out = func(*args, **kwargs)
    torch.cuda.synchronize()
    return out


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    max_time=0.1,
    group="fastrnns",
)
class TestBenchNetwork:
    # [CLASS DOC] Pytest benchmark class (group fastrnns, warmup enabled, three warmup
    # iterations, garbage collection disabled, maximum time 0.1 s) whose tests use the modeldef
    # fixture.
    # See 'modeldef' fixture, which provides the things to benchmark
    def test_forward(self, modeldef, benchmark):
        # [FUNCTION DOC] Benchmark the forward callable of the model on its inputs with cuda
        # sync.
        benchmark(cuda_sync, modeldef.forward, *modeldef.inputs)

    def test_backward(self, modeldef, benchmark):
        # [FUNCTION DOC] Run one forward pass, apply the optional backward setup, and (if the
        # model has a backward) benchmark it with retained graph, then zero all parameter
        # gradients, raising AssertionError if any gradient is None. Does nothing in the
        # benchmark when backward is None.
        backward_input = modeldef.forward(*modeldef.inputs)
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(backward_input)

        if modeldef.backward is not None:
            benchmark(cuda_sync, modeldef.backward, *backward_input, retain_graph=True)

            with torch.no_grad():
                for param in modeldef.params:
                    if param.grad is None:
                        raise AssertionError("Parameter gradient must not be None")
                    param.grad.zero_()
