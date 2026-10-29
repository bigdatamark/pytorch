# mypy: allow-untyped-defs
# [MODULE DOC] Reference quantized recurrent modules: RNNCell, LSTMCell, GRUCell
# (single step) and LSTM, GRU (sequence). They keep float weights and per-weight
# quantization parameters, and at run time use quantize-dequantized weights with
# the float recurrent kernels so backends can lower them. Also exports
# get_quantized_weight. Contains helpers to collect a weight's quantization
# parameters. No import-time side effects.
from typing import Any

import torch
import torch.nn as nn
from torch import _VF, Tensor
from torch.nn.utils.rnn import PackedSequence

from .utils import _quantize_and_dequantize_weight, _quantize_weight


__all__ = [
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNNBase",
    "LSTM",
    "GRU",
    "get_quantized_weight",
]


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    # [FUNCTION DOC] HELPER: Reorders tensor along dimension dim (default 1) by
    # selecting the indices in permutation; returns the new tensor.
    return tensor.index_select(dim, permutation)


def _get_weight_and_quantization_params(module, wn):
    # [FUNCTION DOC] HELPER: Returns a list: the weight attribute named wn of the
    # module, followed by its qscheme, dtype, scale, zero_point and integer axis
    # attributes (named wn plus a suffix), each None if the module lacks it.
    weight = getattr(module, wn)
    params = [weight]
    for param_name in [
        wn + n for n in ["_qscheme", "_dtype", "_scale", "_zero_point", "_axis_int"]
    ]:
        if hasattr(module, param_name):
            param = getattr(module, param_name)
        else:
            param = None
        params.append(param)
    return params


def get_quantized_weight(module, wn):
    # [FUNCTION DOC] Returns the truly quantized version of the module's weight
    # named wn using its stored quantization parameters, or None if the module has
    # no such attribute.
    if not hasattr(module, wn):
        return None
    params = _get_weight_and_quantization_params(module, wn)
    weight = _quantize_weight(*params)
    return weight


def _get_quantize_and_dequantized_weight(module, wn):
    # [FUNCTION DOC] HELPER: Returns the quantize-then-dequantized (float) version
    # of the module's weight named wn, or None if the module has no such attribute.
    if not hasattr(module, wn):
        return None
    params = _get_weight_and_quantization_params(module, wn)
    weight = _quantize_and_dequantize_weight(*params)
    return weight


class RNNCellBase(nn.RNNCellBase):
    # [CLASS DOC] Reference base for the quantized RNN cells (derives from the float
    # cell base). State: per-weight quantization attributes (qscheme, dtype, scale,
    # zero_point, axis buffers and integer axis) for weight_ih and weight_hh, plus
    # the is_decomposed flag. Provides accessors for the quantized and fake-
    # quantized weights.
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
        weight_qparams_dict=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float cell base, then builds the weight
        # quantization state. If weight_qparams_dict is None it defaults to per-
        # tensor affine quint8 (scale 1, zero point 0) for weight_ih and
        # weight_hh with is_decomposed false. Raises AssertionError if the
        # mapping does not have exactly 3 entries.
        super().__init__(
            input_size, hidden_size, bias, num_chunks, device=device, dtype=dtype
        )
        # TODO(jerryzh168): maybe make this arg a required arg
        if weight_qparams_dict is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }
            weight_qparams_dict = {
                "weight_ih": weight_qparams,
                "weight_hh": weight_qparams,
                "is_decomposed": False,
            }
        if len(weight_qparams_dict) != 3:
            raise AssertionError(
                f"Expected length for weight_qparams_dict to be 3 for QuantizedRNNCellBase(Reference), "
                f"got {len(weight_qparams_dict)}"
            )
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        # [FUNCTION DOC] HELPER: For each weight key in the mapping (skipping
        # the is_decomposed entry, which is stored on the module), records
        # qscheme and dtype attributes (AssertionError for schemes other than
        # none, per-tensor affine, per-channel affine). When a scheme is set,
        # registers scale and zero point buffers (tensors cloned or created on
        # device) and an axis buffer (given axis for per-channel, else 0) plus
        # the integer axis attribute. Raises AssertionError if the mapping is
        # None.
        if weight_qparams_dict is None:
            raise AssertionError("weight_qparams_dict must not be None")
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        for key, weight_qparams in weight_qparams_dict.items():
            if key == "is_decomposed":
                continue
            # TODO: refactor the duplicated code to utils.py
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            if weight_qscheme not in [
                None,
                torch.per_tensor_affine,
                torch.per_channel_affine,
            ]:
                raise AssertionError(
                    f"qscheme: {weight_qscheme} is not supported in {self._get_name()}"
                )
            if weight_qscheme is not None:
                scale = weight_qparams["scale"]
                scale_tensor = (
                    scale.detach().clone()
                    if isinstance(scale, torch.Tensor)
                    else torch.tensor(scale, dtype=torch.float, device=device)
                )
                self.register_buffer(key + "_scale", scale_tensor)
                zp = weight_qparams["zero_point"]
                zp_tensor = (
                    zp.detach().clone()
                    if isinstance(zp, torch.Tensor)
                    else torch.tensor(zp, dtype=torch.int, device=device)
                )
                self.register_buffer(key + "_zero_point", zp_tensor)
                if weight_qscheme == torch.per_channel_affine:
                    axis = weight_qparams["axis"]
                    axis_tensor = (
                        axis.detach().clone()
                        if isinstance(axis, torch.Tensor)
                        else torch.tensor(axis, dtype=torch.int, device=device)
                    )
                    self.register_buffer(key + "_axis", axis_tensor)
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device)
                    )
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedRNNCellBase(Reference).
        return "QuantizedRNNCellBase(Reference)"

    def get_quantized_weight_ih(self):
        # [FUNCTION DOC] Returns the quantized input-to-hidden weight.
        return get_quantized_weight(self, "weight_ih")

    def get_quantized_weight_hh(self):
        # [FUNCTION DOC] Returns the quantized hidden-to-hidden weight.
        return get_quantized_weight(self, "weight_hh")

    def get_weight_ih(self):
        # [FUNCTION DOC] Returns the input-to-hidden weight after quantize-
        # dequantize.
        return _get_quantize_and_dequantized_weight(self, "weight_ih")

    def get_weight_hh(self):
        # [FUNCTION DOC] Returns the hidden-to-hidden weight after quantize-
        # dequantize.
        return _get_quantize_and_dequantized_weight(self, "weight_hh")


class RNNCell(RNNCellBase):
    # [CLASS DOC] Reference quantized RNN cell with tanh or relu nonlinearity
    # (stored as an attribute). Takes a weight_qparams_dict mapping weight names to
    # their quantization parameters.
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base cell with one gate chunk, passing
        # device, dtype and the quantization parameter mapping, and stores the
        # nonlinearity name.
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedRNNCell(Reference).
        return "QuantizedRNNCell(Reference)"

    # TODO: refactor nn.RNNCell to have a _forward that takes weight_ih and weight_hh as input
    # and remove duplicated code, same for the other two Cell modules
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        # [FUNCTION DOC] Runs one RNN step on a 1-D (unbatched) or 2-D (batched)
        # input with optional hidden state hx (zeros when None) using the
        # quantize-dequantized weights and the tanh or relu kernel. Returns the
        # new hidden state (batch dimension removed for unbatched input). Raises
        # AssertionError for input dimension not 1 or 2 and RuntimeError for an
        # unknown nonlinearity.
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input,
                hx,
                self.get_weight_ih(),
                self.get_weight_hh(),
                self.bias_ih,
                self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input,
                hx,
                self.get_weight_ih(),
                self.get_weight_hh(),
                self.bias_ih,
                self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # [FUNCTION DOC] Builds a reference RNNCell from a float RNNCell and a
        # quantization parameter mapping, then assigns the float module's
        # weights and biases (shared, not copied) to it. Returns the new module.
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.nonlinearity,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class LSTMCell(RNNCellBase):
    # [CLASS DOC] Reference quantized LSTM cell (four gate chunks). Takes a
    # weight_qparams_dict mapping weight names to quantization parameters.
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base cell with four gate chunks,
        # passing device, dtype and the quantization parameter mapping.
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedLSTMCell(Reference).
        return "QuantizedLSTMCell(Reference)"

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] Runs one LSTM step on a 1-D or 2-D input with optional
        # (hidden, cell) state (zeros when None) using the quantize-dequantized
        # weights. Returns the new (hidden, cell) pair, with the batch dimension
        # removed for unbatched input. Raises AssertionError for input dimension
        # not 1 or 2.
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
            input,
            hx,
            self.get_weight_ih(),
            self.get_weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a reference LSTMCell from a float LSTMCell and a
        # quantization parameter mapping, then assigns the float module's
        # weights and biases (shared, not copied). use_precomputed_fake_quant is
        # unused. Returns the new module.
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class GRUCell(RNNCellBase):
    # [CLASS DOC] Reference quantized GRU cell (three gate chunks). Takes a
    # weight_qparams_dict mapping weight names to quantization parameters.
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base cell with three gate chunks,
        # passing device, dtype and the quantization parameter mapping.
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "weight_qparams_dict": weight_qparams_dict,
        }
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedGRUCell(Reference).
        return "QuantizedGRUCell(Reference)"

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        # [FUNCTION DOC] Runs one GRU step on a 1-D or 2-D input with optional
        # hidden state (zeros when None) using the quantize-dequantized weights.
        # Returns the new hidden state (batch dimension removed for unbatched
        # input). Raises AssertionError for input dimension not 1 or 2.
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = _VF.gru_cell(
            input,
            hx,
            self.get_weight_ih(),
            self.get_weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # [FUNCTION DOC] Builds a reference GRUCell from a float GRUCell and a
        # quantization parameter mapping, then assigns the float module's
        # weights and biases (shared, not copied). Returns the new module.
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict,
        )
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod


class RNNBase(nn.RNNBase):
    # [CLASS DOC] Reference base for quantized sequence RNN modules (derives from
    # the float RNN base). State: per-flat-weight quantization attributes (qscheme,
    # dtype, scale and zero point buffers, axis buffer and integer axis) and the
    # is_decomposed flag.
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
        weight_qparams_dict: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float RNN base with the given mode and
        # sizes, then builds weight quantization state. If weight_qparams_dict
        # is None it defaults to per-tensor affine quint8 (scale 1, zero point
        # 0) for every flat weight whose name starts with weight, with
        # is_decomposed false.
        super().__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype,
        )
        # TODO(jerryzh168): maybe make this arg a required arg
        if weight_qparams_dict is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }
            weight_qparams_dict = {"is_decomposed": False}  # type: ignore[dict-item]
            for wn in self._flat_weights_names:
                if wn.startswith("weight"):
                    weight_qparams_dict[wn] = weight_qparams
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        # [FUNCTION DOC] HELPER: For each weight key in the mapping (skipping
        # is_decomposed, which is stored on the module), records qscheme and
        # dtype attributes (AssertionError for unsupported schemes). When a
        # scheme is set, registers scale and zero point buffers created from the
        # given numbers (float and int), and an axis buffer (given axis for per-
        # channel affine, else 0) plus the integer axis attribute.
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        for key, weight_qparams in weight_qparams_dict.items():
            if key == "is_decomposed":
                continue
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            if weight_qscheme not in [
                None,
                torch.per_tensor_affine,
                torch.per_channel_affine,
            ]:
                raise AssertionError(
                    f"qscheme: {weight_qscheme} is not supported in {self._get_name()}"
                )
            if weight_qscheme is not None:
                self.register_buffer(
                    key + "_scale",
                    torch.tensor(
                        weight_qparams["scale"], dtype=torch.float, device=device
                    ),
                )
                self.register_buffer(
                    key + "_zero_point",
                    torch.tensor(
                        weight_qparams["zero_point"], dtype=torch.int, device=device
                    ),
                )
                if weight_qscheme == torch.per_channel_affine:
                    self.register_buffer(
                        key + "_axis",
                        torch.tensor(
                            weight_qparams["axis"], dtype=torch.int, device=device
                        ),
                    )
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device)
                    )
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())


class LSTM(RNNBase):
    # [CLASS DOC] Reference quantized LSTM layer stack, using the float LSTM kernel
    # with quantize-dequantized flat weights. Takes a weight_qparams_dict mapping
    # flat weight names (for example weight_ih_l0) to quantization parameters.
    """Reference Quantized LSTM Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base RNN class in LSTM mode, forwarding
        # all arguments.
        super().__init__("LSTM", *args, **kwargs)

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(  # type: ignore[override]
        self,
        hx: tuple[Tensor, Tensor],
        permutation: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] Reorders the hidden and cell states of hx along the
        # batch dimension by permutation; returns hx unchanged when permutation
        # is None.
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]:
        # [FUNCTION DOC] Returns the expected cell state shape (layers times
        # directions, mini batch, hidden size) for the input, taking the mini
        # batch from batch_sizes when packed, otherwise from the batch dimension
        # per batch_first.
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(  # type: ignore[override]
        self,
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        batch_sizes: Tensor | None,
    ):
        # [FUNCTION DOC] Validates the input and checks that hidden[0] and
        # hidden[1] sizes match the expected hidden and cell sizes, via the base
        # class checks (which raise on mismatch).
        self.check_input(input, batch_sizes)
        self.check_hidden_size(
            hidden[0],
            self.get_expected_hidden_size(input, batch_sizes),
            "Expected hidden[0] size {}, got {}",
        )
        self.check_hidden_size(
            hidden[1],
            self.get_expected_cell_size(input, batch_sizes),
            "Expected hidden[1] size {}, got {}",
        )

    def get_quantized_weight_bias_dict(self):
        # [FUNCTION DOC] Returns a mapping from each flat weight name to its
        # truly quantized weight (for names starting with weight) or the
        # unquantized bias; None for attributes the module lacks.
        """dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
        quantized_weight_bias_dict = {}
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    weight_or_bias = getattr(self, wn)
            else:
                weight_or_bias = None
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict

    def get_flat_weights(self):
        # [FUNCTION DOC] Returns the list of flat weights in order, where each
        # weight whose name starts with weight is replaced by its quantize-
        # dequantized version; biases are returned as is and missing attributes
        # as None.
        flat_weights = []
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                weight = getattr(self, wn)
                if wn.startswith("weight"):
                    params = _get_weight_and_quantization_params(self, wn)
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                weight = None
            flat_weights.append(weight)
        return flat_weights

    def forward(self, input, hx=None):
        # [FUNCTION DOC] Runs the LSTM over a tensor or PackedSequence input
        # with optional (h0, c0), using the float LSTM kernel with the quantize-
        # dequantized flat weights. Handles unbatched 2-D input by adding then
        # removing a batch dimension, creates zero states when hx is None
        # (projection size honored for hidden), validates hx dimensions
        # (RuntimeError on mismatch) and permutes them by sorted indices.
        # Returns (output, (h_n, c_n)); output is a PackedSequence for packed
        # input.
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = (
                self.proj_size if self.proj_size > 0 else self.hidden_size
            )
            h_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        else:
            if batch_sizes is None:  # If not PackedSequence input.
                if is_batched:  # type: ignore[possibly-undefined]
                    if hx[0].dim() != 3 or hx[1].dim() != 3:
                        msg = (
                            "For batched 3-D input, hx and cx should "
                            f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = (
                            "For unbatched 2-D input, hx and cx should "
                            f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(
                input,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _VF.lstm(
                input,
                batch_sizes,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedLSTM(Reference).
        return "QuantizedLSTM(Reference)"

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # [FUNCTION DOC] Builds a reference LSTM from a float LSTM and a
        # quantization parameter mapping, then assigns each flat weight
        # attribute from the float module (shared). Returns the new module.
        # KNOWN ISSUE: does not forward proj_size to the constructor, so
        # projected LSTMs are not reproduced faithfully.
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict,
        )
        for wn in mod._flat_weights_names:
            setattr(ref_mod, wn, getattr(mod, wn))
        return ref_mod


class GRU(RNNBase):
    # [CLASS DOC] Reference quantized GRU layer stack, using the float GRU kernel
    # with quantize-dequantized flat weights. Takes a weight_qparams_dict mapping
    # flat weight names to quantization parameters.
    """Reference Quantized GRU Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base RNN class in GRU mode. Raises
        # ValueError if proj_size is passed as a keyword argument.
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        super().__init__("GRU", *args, **kwargs)

    def get_quantized_weight_bias_dict(self):
        # [FUNCTION DOC] Returns a mapping from each flat weight name to its
        # truly quantized weight (for names starting with weight) or the
        # unquantized bias; None for attributes the module lacks.
        """dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
        quantized_weight_bias_dict = {}
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    weight_or_bias = getattr(self, wn)
            else:
                weight_or_bias = None
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict

    def get_flat_weights(self):
        # [FUNCTION DOC] Returns the list of flat weights in order, where each
        # weight whose name starts with weight is replaced by its quantize-
        # dequantized version; biases are returned as is and missing attributes
        # as None.
        flat_weights = []
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                weight = getattr(self, wn)
                if wn.startswith("weight"):
                    params = _get_weight_and_quantization_params(self, wn)
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                weight = None
            flat_weights.append(weight)
        return flat_weights

    def forward(self, input, hx=None):
        # [FUNCTION DOC] Runs the GRU over a tensor or PackedSequence input with
        # optional initial hidden state, using the float GRU kernel with the
        # quantize-dequantized flat weights. Validates input dimension
        # (AssertionError) and hx dimension (RuntimeError), handles unbatched
        # input by adding then removing a batch dimension, creates zeros when hx
        # is None and permutes hx by sorted indices. Returns (output, h_n);
        # output is a PackedSequence for packed input.
        # Note: this is copied from the forward of GRU in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        # only changed self._flat_weights to self.get_flat_weights()
        # TODO: maybe we can try inheriting from that class and define get_flat_weights
        # as a @property? this might interfere with TorchScript, if we remove that
        # requirement in the future we should be able to do this
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise AssertionError(
                    f"GRU: Expected input to be 2-D or 3-D but received {input.dim()}-D tensor"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.gru(
                input,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _VF.gru(
                input,
                batch_sizes,
                hx,
                self.get_flat_weights(),
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = hidden.squeeze(1)

            return output, self.permute_hidden(hidden, unsorted_indices)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedGRU(Reference).
        return "QuantizedGRU(Reference)"

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # [FUNCTION DOC] Builds a reference GRU from a float GRU and a
        # quantization parameter mapping, then assigns each flat weight
        # attribute from the float module (shared). Returns the new module.
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict,
        )
        for wn in mod._flat_weights_names:
            setattr(ref_mod, wn, getattr(mod, wn))
        return ref_mod
