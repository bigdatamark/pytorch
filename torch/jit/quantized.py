# mypy: allow-untyped-defs
# [MODULE DOC] Tombstones for the removed TorchScript quantized module classes and helper
# functions. Every class constructor and function here only raises RuntimeError pointing to
# the torch.ao.nn.quantized.dynamic or torch.ao.quantization.quantize_dynamic replacements.
import torch


class QuantizedLinear(torch.jit.ScriptModule):
    # [CLASS DOC] Removed placeholder for the quantized linear module; cannot be instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Always raises RuntimeError saying the class is unsupported and to use
        # torch.ao.nn.quantized.dynamic.Linear.
        raise RuntimeError(
            "torch.jit.QuantizedLinear is no longer supported. Please use "
            "torch.ao.nn.quantized.dynamic.Linear instead."
        )


# FP16 weights
class QuantizedLinearFP16(torch.jit.ScriptModule):
    # [CLASS DOC] Removed placeholder for the FP16-weight quantized linear module; cannot be
    # instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Calls the base initializer, then always raises RuntimeError pointing to
        # torch.ao.nn.quantized.dynamic.Linear.
        super().__init__()
        raise RuntimeError(
            "torch.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.Linear instead."
        )


# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    # [CLASS DOC] Removed placeholder base class for quantized RNN cells; cannot be instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Always raises RuntimeError pointing to torch.ao.nn.quantized.dynamic.RNNCell.
        raise RuntimeError(
            "torch.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedRNNCell(QuantizedRNNCellBase):
    # [CLASS DOC] Removed placeholder for the quantized RNN cell; cannot be instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Always raises RuntimeError pointing to torch.ao.nn.quantized.dynamic.RNNCell.
        raise RuntimeError(
            "torch.jit.QuantizedRNNCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedLSTMCell(QuantizedRNNCellBase):
    # [CLASS DOC] Removed placeholder for the quantized LSTM cell; cannot be instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Calls the base initializer (which itself raises RuntimeError). The
        # following raise pointing to torch.ao.nn.quantized.dynamic.LSTMCell is unreachable.
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead."
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    # [CLASS DOC] Removed placeholder for the quantized GRU cell; cannot be instantiated.
    def __init__(self, other):
        # [FUNCTION DOC] Calls the base initializer (which itself raises RuntimeError). The
        # following raise pointing to torch.ao.nn.quantized.dynamic.GRUCell is unreachable.
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedGRUCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRUCell instead."
        )


class QuantizedRNNBase(torch.jit.ScriptModule):
    # [CLASS DOC] Removed placeholder base class for quantized RNN modules; cannot be instantiated.
    def __init__(self, other, dtype=torch.int8):
        # [FUNCTION DOC] Always raises RuntimeError pointing to torch.ao.nn.quantized.dynamic.
        raise RuntimeError(
            "torch.jit.QuantizedRNNBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic instead."
        )


class QuantizedLSTM(QuantizedRNNBase):
    # [CLASS DOC] Removed placeholder for the quantized LSTM module; cannot be instantiated.
    def __init__(self, other, dtype):
        # [FUNCTION DOC] Always raises RuntimeError pointing to torch.ao.nn.quantized.dynamic.LSTM.
        raise RuntimeError(
            "torch.jit.QuantizedLSTM is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTM instead."
        )


class QuantizedGRU(QuantizedRNNBase):
    # [CLASS DOC] Removed placeholder for the quantized GRU module; cannot be instantiated.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Accepts any arguments and always raises RuntimeError pointing to
        # torch.ao.nn.quantized.dynamic.GRU.
        raise RuntimeError(
            "torch.jit.QuantizedGRU is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRU instead."
        )


def quantize_rnn_cell_modules(module):
    # [FUNCTION DOC] Removed API; always raises RuntimeError pointing to quantize_dynamic.
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_linear_modules(module, dtype=torch.int8):
    # [FUNCTION DOC] Removed API; always raises RuntimeError pointing to quantize_dynamic.
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_rnn_modules(module, dtype=torch.int8):
    # [FUNCTION DOC] Removed API; always raises RuntimeError pointing to quantize_dynamic.
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )
