# [MODULE DOC] Registry of benchmark runners for fastrnns. Defines context managers (disable
# cuDNN, no-op, assert JIT disabled), the RNNRunner record (name, creator, context) and the
# nn_runners mapping from short names (cudnn, jit, resnet18, ...) to runners, with get_nn_runners
# to look them up. Import-time side effects: imports torchvision and builds the CNN creators.
from collections import namedtuple
from functools import partial

import torchvision.models as cnn

import torch

from .factory import (
    dropoutlstm_creator,
    imagenet_cnn_creator,
    layernorm_pytorch_lstm_creator,
    lnlstm_creator,
    lstm_creator,
    lstm_multilayer_creator,
    lstm_premul_bias_creator,
    lstm_premul_creator,
    lstm_simple_creator,
    pytorch_lstm_creator,
    varlen_lstm_creator,
    varlen_pytorch_lstm_creator,
)


class DisableCuDNN:
    # [CLASS DOC] Context manager that turns the cuDNN backend off while active and restores the
    # prior setting on exit (saved on the instance in an attribute named saved).
    def __enter__(self):
        # [FUNCTION DOC] Save the current cuDNN enabled flag and set it to false.
        self.saved = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] Restore the saved cuDNN enabled flag; exceptions are not suppressed.
        torch.backends.cudnn.enabled = self.saved


class DummyContext:
    # [CLASS DOC] Do nothing context manager used for runners that need no special environment.
    def __enter__(self):
        # [FUNCTION DOC] No-op.
        pass

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] No-op; exceptions propagate.
        pass


class AssertNoJIT:
    # [CLASS DOC] Context manager asserting that the JIT is disabled through the environment
    # variable.
    def __enter__(self):
        # [FUNCTION DOC] Read the environment variable controlling the JIT (default 1) and raise
        # AssertionError if the value is truthy.
        # KNOWN ISSUE: the value is a string, so even a value of 0 is truthy and raises; only an
        # unset check with an empty string passes.
        import os

        enabled = os.environ.get("PYTORCH_JIT", 1)
        if enabled:
            raise AssertionError("PYTORCH_JIT must be disabled")

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] No-op; exceptions propagate.
        pass


RNNRunner = namedtuple(
    "RNNRunner",
    [
        "name",
        "creator",
        "context",
    ],
)


def get_nn_runners(*names):
    # [FUNCTION DOC] Look up each name in the nn_runners mapping and return the list of runners
    # in the same order. An unknown name raises KeyError.
    return [nn_runners[name] for name in names]


nn_runners = {
    "cudnn": RNNRunner("cudnn", pytorch_lstm_creator, DummyContext),
    "cudnn_dropout": RNNRunner(
        "cudnn_dropout", partial(pytorch_lstm_creator, dropout=0.4), DummyContext
    ),
    "cudnn_layernorm": RNNRunner(
        "cudnn_layernorm", layernorm_pytorch_lstm_creator, DummyContext
    ),
    "vl_cudnn": RNNRunner("vl_cudnn", varlen_pytorch_lstm_creator, DummyContext),
    "vl_jit": RNNRunner(
        "vl_jit", partial(varlen_lstm_creator, script=True), DummyContext
    ),
    "vl_py": RNNRunner("vl_py", varlen_lstm_creator, DummyContext),
    "aten": RNNRunner("aten", pytorch_lstm_creator, DisableCuDNN),
    "jit": RNNRunner("jit", lstm_creator, DummyContext),
    "jit_premul": RNNRunner("jit_premul", lstm_premul_creator, DummyContext),
    "jit_premul_bias": RNNRunner(
        "jit_premul_bias", lstm_premul_bias_creator, DummyContext
    ),
    "jit_simple": RNNRunner("jit_simple", lstm_simple_creator, DummyContext),
    "jit_multilayer": RNNRunner(
        "jit_multilayer", lstm_multilayer_creator, DummyContext
    ),
    "jit_layernorm": RNNRunner("jit_layernorm", lnlstm_creator, DummyContext),
    "jit_layernorm_decom": RNNRunner(
        "jit_layernorm_decom",
        partial(lnlstm_creator, decompose_layernorm=True),
        DummyContext,
    ),
    "jit_dropout": RNNRunner("jit_dropout", dropoutlstm_creator, DummyContext),
    "py": RNNRunner("py", partial(lstm_creator, script=False), DummyContext),
    "resnet18": RNNRunner(
        "resnet18", imagenet_cnn_creator(cnn.resnet18, jit=False), DummyContext
    ),
    "resnet18_jit": RNNRunner(
        "resnet18_jit", imagenet_cnn_creator(cnn.resnet18), DummyContext
    ),
    "resnet50": RNNRunner(
        "resnet50", imagenet_cnn_creator(cnn.resnet50, jit=False), DummyContext
    ),
    "resnet50_jit": RNNRunner(
        "resnet50_jit", imagenet_cnn_creator(cnn.resnet50), DummyContext
    ),
}
