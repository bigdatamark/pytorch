# [MODULE DOC] Custom TorchScript LSTM building blocks for the fastrnns benchmarks: LSTM cell and
# layer-normalized cell, forward/reverse/bidirectional layers, stacked LSTMs (plain, nested
# direction states, with dropout), factories script_lstm and script_lnlstm that mimic nn.LSTM,
# state flattening helpers and numerical checks against the native nn.LSTM. Import-time side
# effect: the five test functions at the bottom are called with fixed small sizes when the module
# is imported, so importing it runs the comparisons (random inputs, AssertionError on mismatch).
import numbers
import warnings
from collections import namedtuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


"""
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
"""


def script_lstm(
    input_size,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
):
    # [FUNCTION DOC] Build a scripted LSTM stack that mimics nn.LSTM. bias must be true and
    # batch_first false, otherwise AssertionError is raised. Bidirectional selects StackedLSTM2
    # with BidirLSTMLayer (layer input width then doubles); otherwise dropout selects
    # StackedLSTMWithDropout; else a plain StackedLSTM. Bidirectional takes precedence over
    # dropout (dropout is ignored then). All use LSTMCell. Returns the module with num_layers
    # layers; weights are random.
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    if not bias:
        raise AssertionError("bias=False is not implemented")
    if batch_first:
        raise AssertionError("batch_first=True is not implemented")

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[LSTMCell, input_size, hidden_size],
        other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
    )


def script_lnlstm(
    input_size,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
    decompose_layernorm=False,
):
    # [FUNCTION DOC] Same as script_lstm but uses LayerNormLSTMCell. Raises AssertionError unless
    # bias is true, batch_first false and dropout false. Bidirectional selects StackedLSTM2 with
    # BidirLSTMLayer, otherwise StackedLSTM with LSTMLayer. decompose_layernorm selects the
    # scripted LayerNorm class over nn.LayerNorm inside the cells. Returns the stacked module.
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    if not bias:
        raise AssertionError("bias=False is not implemented")
    if batch_first:
        raise AssertionError("batch_first=True is not implemented")
    if dropout:
        raise AssertionError("dropout=True is not implemented")

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[
            LayerNormLSTMCell,
            input_size,
            hidden_size,
            decompose_layernorm,
        ],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size * dirs,
            hidden_size,
            decompose_layernorm,
        ],
    )


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


def reverse(lst: list[Tensor]) -> list[Tensor]:
    # [FUNCTION DOC] Return a reversed copy of a list of tensors (slice with negative step).
    return lst[::-1]


class LSTMCell(jit.ScriptModule):
    # [CLASS DOC] Scripted LSTM cell holding parameters weight_ih, weight_hh, bias_ih and bias_hh
    # (random normal init, first dimension four times hidden_size). forward consumes one time
    # step input and a (hx, cx) state and returns the output plus the new state.
    def __init__(self, input_size, hidden_size):
        # [FUNCTION DOC] Store sizes and create the four randomly initialised parameters.
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] One LSTM step: gates are the input and hidden products plus both biases,
        # split in four chunks (input, forget, cell, output). Returns (hy, (hy, cy)) where hy is
        # also the output.
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNorm(jit.ScriptModule):
    # [CLASS DOC] Scripted layer normalization over the last dimension, written out explicitly
    # (mean, biased std) so it can be decomposed and fused. Holds learnable weight (ones) and
    # bias (zeros).
    def __init__(self, normalized_shape):
        # [FUNCTION DOC] Accept an int or shape; convert it to a torch Size. Raises
        # AssertionError unless it has exactly one dimension. Creates weight and bias parameters
        # and stores the shape.
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        if len(normalized_shape) != 1:
            raise AssertionError(
                f"Expected normalized_shape to have length 1, but got {len(normalized_shape)}"
            )

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        # [FUNCTION DOC] Return the mean and the biased standard deviation over the last
        # dimension, both keeping the reduced dimension.
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        # [FUNCTION DOC] Normalize with the statistics, then scale by weight and shift by bias.
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias


class LayerNormLSTMCell(jit.ScriptModule):
    # [CLASS DOC] Scripted LSTM cell with layer normalization on the input gates, hidden gates
    # and the new cell state; there are no explicit biases (the norms supply them). The
    # decompose_layernorm flag selects the explicit LayerNorm class instead of nn.LayerNorm.
    def __init__(self, input_size, hidden_size, decompose_layernorm=False):
        # [FUNCTION DOC] Create input and hidden weights and three layernorm modules (sizes four
        # times hidden for gates, hidden for the cell state).
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] One step: normalized input and hidden products are summed and chunked
        # into four gates; the new cell state is layer normalized before the tanh. Returns (hy,
        # (hy, cy)).
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    # [CLASS DOC] Scripted layer that runs a cell over a sequence in time order. Holds one cell
    # built from the given cell class and arguments.
    def __init__(self, cell, *cell_args):
        # [FUNCTION DOC] Instantiate the cell class with the extra arguments and keep it as the
        # cell.
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] Unbind input along dim 0 and step the cell through every time step,
        # threading the state. Returns the stacked outputs and the final state.
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(list[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(jit.ScriptModule):
    # [CLASS DOC] Scripted layer that runs a cell over a sequence in reverse time order; outputs
    # are returned in original time order.
    def __init__(self, cell, *cell_args):
        # [FUNCTION DOC] Instantiate the cell class with the extra arguments and keep it as the
        # cell.
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] Reverse the unbound time steps, run the cell through them and reverse
        # the outputs back. Returns the stacked outputs and the final state.
        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(list[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(jit.ScriptModule):
    # [CLASS DOC] Scripted bidirectional layer holding a forward LSTMLayer and a ReverseLSTMLayer
    # in a module list; outputs of both directions are concatenated on the last dimension.
    __constants__ = ["directions"]

    def __init__(self, cell, *cell_args):
        # [FUNCTION DOC] Build the two directional layers from the same cell class and arguments.
        super().__init__()
        self.directions = nn.ModuleList(
            [
                LSTMLayer(cell, *cell_args),
                ReverseLSTMLayer(cell, *cell_args),
            ]
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        # [FUNCTION DOC] Run each direction with its own state (a list indexed by direction),
        # collect outputs and states. Returns the concatenation of outputs on the last dim and
        # the list of states.
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(list[Tensor], [])
        output_states = jit.annotate(list[tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1  # noqa: SIM113
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    # [FUNCTION DOC] HELPER: Build a module list of num_layers layers: the first constructed with
    # first_layer_args and the rest with other_layer_args (each call unpacks the argument list).
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
    # [CLASS DOC] Scripted multilayer LSTM. Holds a module list of layers; forward takes the
    # input and one state per layer and returns the last layer's output plus the list of output
    # states.
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        # [FUNCTION DOC] Create the layers through init_stacked_lstm.
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        # [FUNCTION DOC] Pass the output of each layer to the next, using the matching entry of
        # states for each layer. Returns the final output and the list of per-layer states.
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(list[tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1  # noqa: SIM113
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(jit.ScriptModule):
    # [CLASS DOC] Like StackedLSTM but each per-layer state is itself a list of states (one per
    # direction), used with bidirectional layers. Duplicated since script methods cannot be
    # overridden.
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        # [FUNCTION DOC] Create the layers through init_stacked_lstm.
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: list[list[tuple[Tensor, Tensor]]]
    ) -> tuple[Tensor, list[list[tuple[Tensor, Tensor]]]]:
        # [FUNCTION DOC] Run layers in sequence with the matching list-of-direction states.
        # Returns the final output and the list of per-layer lists of output states.
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(list[list[tuple[Tensor, Tensor]]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1  # noqa: SIM113
        return output, output_states


class StackedLSTMWithDropout(jit.ScriptModule):
    # [CLASS DOC] Scripted multilayer LSTM that applies dropout (probability 0.4) to the output
    # of every layer except the last one.
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        # [FUNCTION DOC] Create the layers, remember num_layers and create the dropout module.
        # Emits a warning if num_layers is 1, since dropout is then never applied.
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if num_layers == 1:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )

        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(
        self, input: Tensor, states: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        # [FUNCTION DOC] Same as StackedLSTM forward but applies dropout after each layer other
        # than the last. Returns the final output and per-layer states.
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(list[tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1  # noqa: SIM113
        return output, output_states


def flatten_states(states):
    # [FUNCTION DOC] Convert a list of per-layer (hx, cx) pairs into a list of two stacked
    # tensors (all hx stacked, all cx stacked). Raises AssertionError if the pairs do not have
    # exactly two parts.
    states = list(zip(*states))
    if len(states) != 2:
        raise AssertionError(f"Expected states to have length 2, but got {len(states)}")
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    # [FUNCTION DOC] HELPER: Flatten the nested per-layer, per-direction list of (hx, cx) states
    # into two tensors whose first two dimensions are merged (layers times directions). Returns a
    # list.
    # XXX: Can probably write this in a nicer way
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    # [FUNCTION DOC] Check LSTMLayer with LSTMCell against nn.LSTM with one layer. Copies the
    # custom parameters into the native LSTM, runs both on random input and compares output,
    # hidden and cell states with absolute tolerance 1e-5. Raises AssertionError on a shape or
    # value mismatch.
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        if lstm_param.shape != custom_param.shape:
            raise AssertionError(
                f"Shape mismatch: lstm_param.shape={lstm_param.shape}, custom_param.shape={custom_param.shape}"
            )
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    if (out - lstm_out).abs().max() >= 1e-5:
        raise AssertionError(
            f"Output mismatch: max diff={(out - lstm_out).abs().max()}"
        )
    if (out_state[0] - lstm_out_state[0]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Hidden state mismatch: max diff={(out_state[0] - lstm_out_state[0]).abs().max()}"
        )
    if (out_state[1] - lstm_out_state[1]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Cell state mismatch: max diff={(out_state[1] - lstm_out_state[1]).abs().max()}"
        )


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    # [FUNCTION DOC] Check a stacked script_lstm against a multilayer nn.LSTM: copy four
    # parameters per layer, run both on random input and states, compare output and final states
    # within 1e-5. Raises AssertionError on mismatch.
    inp = torch.randn(seq_len, batch, input_size)
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lstm(input_size, hidden_size, num_layers)
    out, out_state = rnn(inp, states)
    custom_state = flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    lstm_state = flatten_states(states)
    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer : 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer], custom_params):
            if lstm_param.shape != custom_param.shape:
                raise AssertionError(
                    f"Shape mismatch at layer {layer}: lstm_param.shape={lstm_param.shape}, custom_param.shape={custom_param.shape}"
                )
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    if (out - lstm_out).abs().max() >= 1e-5:
        raise AssertionError(
            f"Output mismatch: max diff={(out - lstm_out).abs().max()}"
        )
    if (custom_state[0] - lstm_out_state[0]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Hidden state mismatch: max diff={(custom_state[0] - lstm_out_state[0]).abs().max()}"
        )
    if (custom_state[1] - lstm_out_state[1]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Cell state mismatch: max diff={(custom_state[1] - lstm_out_state[1]).abs().max()}"
        )


def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    # [FUNCTION DOC] Same comparison for a bidirectional stacked script_lstm versus a
    # bidirectional nn.LSTM (four parameters per layer and direction, states doubly flattened).
    # Raises AssertionError on shape or value mismatch beyond 1e-5.
    inp = torch.randn(seq_len, batch, input_size)
    states = [
        [
            LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
            for _ in range(2)
        ]
        for _ in range(num_layers)
    ]
    rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
    out, out_state = rnn(inp, states)
    custom_state = double_flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index : 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index], custom_params):
                if lstm_param.shape != custom_param.shape:
                    raise AssertionError(
                        f"Shape mismatch at layer {layer}, direction {direct}: "
                        f"lstm_param.shape={lstm_param.shape}, custom_param.shape={custom_param.shape}"
                    )
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    if (out - lstm_out).abs().max() >= 1e-5:
        raise AssertionError(
            f"Output mismatch: max diff={(out - lstm_out).abs().max()}"
        )
    if (custom_state[0] - lstm_out_state[0]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Hidden state mismatch: max diff={(custom_state[0] - lstm_out_state[0]).abs().max()}"
        )
    if (custom_state[1] - lstm_out_state[1]).abs().max() >= 1e-5:
        raise AssertionError(
            f"Cell state mismatch: max diff={(custom_state[1] - lstm_out_state[1]).abs().max()}"
        )


def test_script_stacked_lstm_dropout(
    seq_len, batch, input_size, hidden_size, num_layers
):
    # [FUNCTION DOC] Smoke test: build a dropout script_lstm and run it once on random input and
    # states; no numerical check, only checks it runs.
    inp = torch.randn(seq_len, batch, input_size)
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)

    # just a smoke test
    out, out_state = rnn(inp, states)


def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size, num_layers):
    # [FUNCTION DOC] Smoke test: build a script_lnlstm and run it once on random input and
    # states; no numerical checks.
    inp = torch.randn(seq_len, batch, input_size)
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lnlstm(input_size, hidden_size, num_layers)

    # just a smoke test
    out, out_state = rnn(inp, states)


test_script_rnn_layer(5, 2, 3, 7)
test_script_stacked_rnn(5, 2, 3, 7, 4)
test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
test_script_stacked_lnlstm(5, 2, 3, 7, 4)
