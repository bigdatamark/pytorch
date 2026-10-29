# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Support code for cuDNN-backed RNN modules: mode-name conversion, dropout state initialization, and a module wrapper exposing the rnn fp32 precision setting.
# Import-time side effects: tries to import the compiled cudnn extension (None if absent) and replaces this module in the module registry with CudnnRNNModule.
import sys

import torch._C
import torch.cuda
from torch.backends import (
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    PropModule,
)


try:
    from torch._C import _cudnn
except ImportError:
    # Uses of all the functions below should be guarded by torch.backends.cudnn.is_available(),
    # so it's safe to not emit any checks here.
    _cudnn = None  # type: ignore[assignment]


def get_cudnn_mode(mode):
    # [FUNCTION DOC] Maps an RNN mode name (RNN_RELU, RNN_TANH, LSTM, GRU) to the integer value of the cuDNN enum; raises ValueError for any other name.
    if mode == "RNN_RELU":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.rnn_relu)
    elif mode == "RNN_TANH":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.rnn_tanh)
    elif mode == "LSTM":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.lstm)
    elif mode == "GRU":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.gru)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# NB: We don't actually need this class anymore (in fact, we could serialize the
# dropout state for even better reproducibility), but it is kept for backwards
# compatibility for old models.
class Unserializable:
    # [CLASS DOC] HELPER: Wrapper holding an arbitrary object (dropout state) that deliberately refuses to be serialized; pickling stores a placeholder string and unpickling yields an empty wrapper.
    # Kept only for backward compatibility with old saved models.
    def __init__(self, inner):
        # [FUNCTION DOC] HELPER: Stores the wrapped object in attribute inner.
        self.inner = inner

    def get(self):
        # [FUNCTION DOC] HELPER: Returns the wrapped object.
        return self.inner

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Returns a placeholder string so the real state is never serialized.
        return "<unserializable>"

    def __setstate__(self, state):
        # [FUNCTION DOC] HELPER: Restores by setting the wrapped object to None, ignoring the saved state.
        self.inner = None


# we would like to use ContextProp from backends here but the
# frozen flags appears to be overzealous
class ContextProp:
    # [CLASS DOC] HELPER: Minimal descriptor that forwards attribute reads to a getter function and writes to a setter function; local replacement for the shared one in torch.backends.
    def __init__(self, getter, setter):
        # [FUNCTION DOC] HELPER: Stores the getter and setter callables.
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        # [FUNCTION DOC] HELPER: Returns the result of calling the getter.
        return self.getter()

    def __set__(self, obj, val):
        # [FUNCTION DOC] HELPER: Calls the setter with the assigned value.
        self.setter(val)


def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    # [FUNCTION DOC] Ensures the per-device dropout state mapping has a state entry for the current CUDA device and returns the dropout state tensor (or None).
    # Args: dropout probability, train flag, seed, and the mutable mapping of states. If missing or empty, creates an entry: a None wrapper when effective dropout is zero (not training or p is zero), else a cuDNN-initialized uint8 CUDA tensor wrapper.
    # Side effect: mutates the mapping. Duplicated by the static method of the same name in the module class.
    dropout_desc_name = "desc_" + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (
        dropout_state[dropout_desc_name].get() is None
    ):
        if dropout_p == 0:
            dropout_state[dropout_desc_name] = Unserializable(None)
        else:
            dropout_state[dropout_desc_name] = Unserializable(
                torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                    dropout_p,
                    train,
                    dropout_seed,
                    # pyrefly: ignore [unexpected-keyword]
                    self_ty=torch.uint8,
                    device=torch.device("cuda"),
                )
            )
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts


class CudnnRNNModule(PropModule):
    # [CLASS DOC] Module wrapper installed in place of this module; on construction attaches Unserializable, get_cudnn_mode and init_dropout_state to the original module, and exposes an rnn fp32_precision property for the cuda backend.
    def __init__(self, m, name):
        # [FUNCTION DOC] Initializes the wrapper around the original module and publishes the helper class and functions on it as attributes.
        super().__init__(m, name)
        self.m.Unserializable = Unserializable
        self.m.get_cudnn_mode = get_cudnn_mode
        self.m.init_dropout_state = init_dropout_state

    @staticmethod
    def init_dropout_state(dropout, train, dropout_seed, dropout_state):
        # [FUNCTION DOC] Same behavior as the module-level init_dropout_state: returns the per-device cached dropout state tensor, creating it when missing; mutates the given mapping.
        dropout_desc_name = "desc_" + str(torch.cuda.current_device())
        dropout_p = dropout if train else 0
        if (dropout_desc_name not in dropout_state) or (
            dropout_state[dropout_desc_name].get() is None
        ):
            if dropout_p == 0:
                dropout_state[dropout_desc_name] = Unserializable(None)
            else:
                dropout_state[dropout_desc_name] = Unserializable(
                    torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                        dropout_p,
                        train,
                        dropout_seed,
                        # pyrefly: ignore [unexpected-keyword]
                        self_ty=torch.uint8,
                        device=torch.device("cuda"),
                    )
                )
        dropout_ts = dropout_state[dropout_desc_name].get()
        return dropout_ts

    fp32_precision = ContextProp(
        _get_fp32_precision_getter("cuda", "rnn"),
        _set_fp32_precision_setter("cuda", "rnn"),
    )


sys.modules[__name__] = CudnnRNNModule(sys.modules[__name__], __name__)
