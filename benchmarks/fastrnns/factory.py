# [MODULE DOC] Model factories for the fastrnns benchmarks. Each creator takes benchmark options
# and returns a ModelDef (inputs, params, forward, backward_setup, backward). Contains creators
# for native nn.LSTM, hand written LSTM loops (plain, scripted, premultiplied, simple,
# multilayer, variable length, layer norm and dropout variants) and ImageNet CNNs, plus input
# builders and backward helpers. No import-time side effects beyond imports.
from collections import namedtuple

import torch
from torch import Tensor

from .cells import flat_lstm_cell, lstm_cell, premul_lstm_cell, premul_lstm_cell_no_bias


# list[list[T]] -> list[T]
def flatten_list(lst):
    # [FUNCTION DOC] HELPER: Concatenate a list of lists into a single new list, preserving
    # order.
    result = []
    for inner in lst:
        result.extend(inner)
    return result


"""
Define a creator as a function:
(options) -> (inputs, params, forward, backward_setup, backward)
inputs: the inputs to the returned 'forward'. One can call
    forward(*inputs) directly.
params: List[Tensor] all requires_grad=True parameters.
forward: function / graph executor / module
    One can call rnn(rnn_inputs) using the outputs of the creator.
backward_setup: backward_inputs = backward_setup(*outputs)
    Then, we pass backward_inputs to backward. If None, then it is assumed to
    be the identity function.
backward: Given `output = backward_setup(*forward(*inputs))`, performs
    backpropagation. If None, then nothing happens.

fastrnns.bench times the forward and backward invocations.
"""


ModelDef = namedtuple(
    "ModelDef", ["inputs", "params", "forward", "backward_setup", "backward"]
)


def lstm_backward_setup(lstm_outputs, seed=None):
    # [FUNCTION DOC] HELPER: Backward setup for LSTM outputs: take the first element (the output
    # sequence) of the (output, state) pair and delegate to simple_backward_setup with the seed.
    hx, _ = lstm_outputs
    return simple_backward_setup(hx, seed)


def simple_backward_setup(output, seed=None):
    # [FUNCTION DOC] HELPER: Prepare backward inputs: raises AssertionError if output is not a
    # Tensor, seeds the global random generator when seed is truthy (so seed zero does not seed),
    # and returns the pair (output, random gradient with the same shape as output).
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"Expected output to be a Tensor, but got {type(output)}")
    if seed:
        torch.manual_seed(seed)
    grad_output = torch.randn_like(output)
    return output, grad_output


def simple_backward(output, grad_output, **kwargs):
    # [FUNCTION DOC] HELPER: Run backward on output with grad_output; extra keyword args are
    # passed to the autograd backward call. Returns its result (None).
    return output.backward(grad_output, **kwargs)


def pytorch_lstm_creator(**kwargs):
    # [FUNCTION DOC] Creator for the cuDNN based nn.LSTM. Builds the module through lstm_inputs
    # and returns a ModelDef whose inputs are the input and (hx, cx), params are all flattened
    # module weights, forward is the module, with LSTM backward setup and simple backward.
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=module,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_creator(script=True, **kwargs):
    # [FUNCTION DOC] Creator for the hand written LSTM loop built from lstm_cell (scripted when
    # script is true). Inputs are the input, hidden state and layer-0 weights; params are all
    # layer weights.
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lnlstm_creator(script=True, decompose_layernorm=False, **kwargs):
    # [FUNCTION DOC] Creator for the scripted layer normalized LSTM (one layer, always on cuda).
    # Raises AssertionError unless script is true. Reads inputSize, hiddenSize, seqLength and
    # miniBatch from kwargs. The inputs are an input tensor and a one element state list. Note
    # params is a one-shot generator of module parameters, so it is exhausted after the first
    # pass.
    if script is not True:
        raise AssertionError("lnlstm_creator requires script=True")
    from .custom_lstms import script_lnlstm

    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    ge = script_lnlstm(
        input_size, hidden_size, 1, decompose_layernorm=decompose_layernorm
    ).cuda()

    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    states = [
        (
            torch.randn(batch_size, hidden_size, device="cuda"),
            torch.randn(batch_size, hidden_size, device="cuda"),
        )
    ]

    return ModelDef(
        inputs=[input, states],
        params=ge.parameters(),
        forward=ge,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def dropoutlstm_creator(script=True, **kwargs):
    # [FUNCTION DOC] Creator for the scripted LSTM with dropout (cuda only). Raises
    # AssertionError unless script is true. Reads inputSize, hiddenSize, seqLength, miniBatch and
    # numLayers from kwargs; builds numLayers random LSTMState entries. params is a one-shot
    # generator.
    if script is not True:
        raise AssertionError("dropoutlstm_creator requires script=True")
    from .custom_lstms import LSTMState, script_lstm

    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    num_layers = kwargs["numLayers"]
    ge = script_lstm(input_size, hidden_size, num_layers, dropout=True).cuda()

    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    states = [
        LSTMState(
            torch.randn(batch_size, hidden_size, device="cuda"),
            torch.randn(batch_size, hidden_size, device="cuda"),
        )
        for _ in range(num_layers)
    ]
    return ModelDef(
        inputs=[input, states],
        params=ge.parameters(),
        forward=ge,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_premul_creator(script=True, **kwargs):
    # [FUNCTION DOC] Creator for the LSTM that premultiplies the whole input sequence by w_ih
    # once (premul_lstm_cell, scripted if script). Inputs are input, hidden and layer-0 weights.
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul(premul_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_premul_bias_creator(script=True, **kwargs):
    # [FUNCTION DOC] Like lstm_premul_creator but also folds the input bias into the
    # premultiplication step (premul_lstm_cell_no_bias is the cell used), with the same inputs
    # and params layout.
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul_bias(premul_lstm_cell_no_bias, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_simple_creator(script=True, **kwargs):
    # [FUNCTION DOC] Creator for the flat-state LSTM (flat_lstm_cell): the inputs are the input,
    # the first row of each hidden tensor entry (hx then cx) and the layer-0 weights.
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input] + [h[0] for h in hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_simple(flat_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_multilayer_creator(script=True, **kwargs):
    # [FUNCTION DOC] Creator for the multilayer hand written LSTM: inputs are input, hidden and
    # the flat list of all layer parameters; forward iterates the layers inside one function.
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden, flatten_list(params)]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_multilayer(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def imagenet_cnn_creator(arch, jit=True):
    # [FUNCTION DOC] Return a creator function for an ImageNet CNN constructor arch. When jit is
    # true the model is traced with a random 32x3x224x224 batch.
    def creator(device="cuda", **kwargs):
        # [FUNCTION DOC] HELPER: Build the model on the device, create a random input batch,
        # optionally trace it, and return a ModelDef with params from the model and the simple
        # backward helpers. The extra keyword arguments are ignored.
        model = arch().to(device)
        x = torch.randn(32, 3, 224, 224, device=device)
        if jit:
            model = torch.jit.trace(model, x)
        return ModelDef(
            inputs=(x,),
            params=list(model.parameters()),
            forward=model,
            backward_setup=simple_backward_setup,
            backward=simple_backward,
        )

    return creator


def varlen_lstm_inputs(
    minlen=30,
    maxlen=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    return_module=False,
    device="cuda",
    seed=None,
    **kwargs,
):
    # [FUNCTION DOC] Create inputs for the variable length LSTM benchmark. Optionally seeds the
    # RNG, draws miniBatch random lengths in the range minlen to maxlen, a list of random
    # sequences, random hx and cx, and an nn.LSTM on the device. Returns (sequences, lengths,
    # (hx, cx), all_weights, module); module is None unless return_module is true. Extra keyword
    # arguments are ignored.
    if seed is not None:
        torch.manual_seed(seed)
    lengths = torch.randint(
        low=minlen, high=maxlen, size=[miniBatch], dtype=torch.long, device=device
    )
    x = [torch.randn(length, inputSize, device=device) for length in lengths]
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers).to(device)

    if return_module:
        return x, lengths, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer]
        return x, lengths, (hx, cx), lstm.all_weights, None


def varlen_lstm_backward_setup(forward_output, seed=None):
    # [FUNCTION DOC] Backward setup for variable length outputs: optionally seed the RNG when
    # seed is truthy, pad the list of output sequences from the first element of forward_output,
    # and return (padded tensor, random gradient of the same shape).
    if seed:
        torch.manual_seed(seed)
    rnn_utils = torch.nn.utils.rnn
    sequences = forward_output[0]
    padded = rnn_utils.pad_sequence(sequences)
    grad = torch.randn_like(padded)
    return padded, grad


def varlen_pytorch_lstm_creator(**kwargs):
    # [FUNCTION DOC] Creator for variable length nn.LSTM: forward packs unsorted sequences, runs
    # the module and pads the packed output.
    rnn_utils = torch.nn.utils.rnn
    sequences, _, hidden, _, module = varlen_lstm_inputs(return_module=True, **kwargs)

    def forward(sequences, hidden):
        # [FUNCTION DOC] HELPER: Pack the list of sequences (unsorted allowed), run the LSTM
        # module and unpack. Returns (padded output, new hidden state).
        packed = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
        out, new_hidden = module(packed, hidden)
        padded, lengths = rnn_utils.pad_packed_sequence(out)
        # XXX: It's more efficient to store the output in its padded form,
        # but that might not be conducive to loss computation.
        # Un-padding the output also makes the backward pass 2x slower...
        # return [padded[:lengths[i], i, :] for i in range(lengths.size(0))]
        return padded, new_hidden

    return ModelDef(
        inputs=[sequences, hidden],
        params=flatten_list(module.all_weights),
        forward=forward,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def varlen_lstm_factory(cell, script):
    # [FUNCTION DOC] Return a variable length dynamic_rnn function around the given cell; when
    # script is true the cell and the function are compiled with TorchScript.
    def dynamic_rnn(
        sequences: list[Tensor],
        hiddens: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[list[Tensor], tuple[list[Tensor], list[Tensor]]]:
        # [FUNCTION DOC] HELPER: For each sequence in the batch, step the cell through its time
        # steps starting from that batch element's hidden state. Returns the list of stacked
        # outputs and the pair of lists of final hx and cx.
        hx, cx = hiddens
        hxs = hx.unbind(1)
        cxs = cx.unbind(1)
        # List of: (output, hx, cx)
        outputs = []
        hx_outs = []
        cx_outs = []

        for batch in range(len(sequences)):
            output = []
            hy, cy = hxs[batch], cxs[batch]
            inputs = sequences[batch].unbind(0)

            for seq_idx in range(len(inputs)):
                hy, cy = cell(
                    inputs[seq_idx].unsqueeze(0), (hy, cy), w_ih, w_hh, b_ih, b_hh
                )
                output += [hy]
            outputs += [torch.stack(output)]
            hx_outs += [hy.unsqueeze(0)]
            cx_outs += [cy.unsqueeze(0)]

        return outputs, (hx_outs, cx_outs)

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def varlen_lstm_creator(script=False, **kwargs):
    # [FUNCTION DOC] Creator for the pure-python or scripted variable length LSTM, with the
    # padded backward setup. Inputs are sequences, hidden and layer-0 weights.
    sequences, _, hidden, params, _ = varlen_lstm_inputs(return_module=False, **kwargs)
    inputs = [sequences, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=varlen_lstm_factory(lstm_cell, script),
        backward_setup=varlen_lstm_backward_setup,
        backward=simple_backward,
    )


# cudnn_layernorm_lstm: since cudnn does not have Layernorm LSTM, we cannot benchmark
# the lowerbound directly. Instead, we only benchmark the forward pass by mimicking the
# computation of a cudnn lstm + seq_len * 3 layernorm computation. This should serve
# as a perf lowerbound for the Layernorm LSTM forward pass(given that Layernorm itself
# is invariant), the lowerbound of backward pass is hard to get since we lose the
# intermediate results, we can still optimize the layernorm implementation to make
# a faster forward lowerbound though.
def layernorm_pytorch_lstm_creator(**kwargs):
    # [FUNCTION DOC] Creator for a lower bound of the layer norm LSTM: an nn.LSTM plus extra
    # layer norm calls per time step (two on a fixed input, one on the cell state). Forward only
    # (backward is None). Requires cuda and reads miniBatch and hiddenSize from kwargs.
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    batch_size = kwargs["miniBatch"]
    hidden_size = kwargs["hiddenSize"]
    ln_i = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_h = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_c = torch.nn.LayerNorm(hidden_size).cuda()
    ln_input1 = torch.randn(batch_size, 4 * hidden_size, device="cuda")

    def forward(input, hidden):
        # [FUNCTION DOC] HELPER: Run the native LSTM, then loop once per time step applying the
        # layer norms to mimic their cost. Returns (output, (hy, normalized cy)).
        out, new_hidden = module(input, hidden)
        # plus (seq_len * three laynorm cell computation) to mimic the lower bound of
        # Layernorm cudnn LSTM in the forward pass
        seq_len = len(input.unbind(0))
        hy, cy = new_hidden
        for i in range(seq_len):
            ln_i(ln_input1)
            ln_h(ln_input1)
            cy = ln_c(cy)

        return out, (hy, cy)

    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=forward,
        backward_setup=lstm_backward_setup,
        backward=None,
    )


# input: lstm.all_weights format (w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is w_ih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is w_hh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is b_ih with size (layer, 4*hiddenSize)
# packed_weights[3] is b_hh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    # [FUNCTION DOC] Convert the per layer weight lists into four stacked tensors (input weights,
    # hidden weights, input biases, hidden biases), each stacked across layers.
    def unzip_columns(mat):
        # [FUNCTION DOC] HELPER: Transpose a list of lists (layer-major to column-major). Raises
        # AssertionError if mat or its first element is not a list.
        if not isinstance(mat, list):
            raise AssertionError(f"Expected mat to be a list, but got {type(mat)}")
        if not isinstance(mat[0], list):
            raise AssertionError(
                f"Expected mat[0] to be a list, but got {type(mat[0])}"
            )
        layers = len(mat)
        columns = len(mat[0])
        return [[mat[layer][col] for layer in range(layers)] for col in range(columns)]

    # XXX: script fns have problems indexing multidim lists, so we try to
    # avoid them by stacking tensors
    all_weights = weights
    packed_weights = [torch.stack(param) for param in unzip_columns(all_weights)]
    return packed_weights


# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    dropout=0.0,
    return_module=False,
    device="cuda",
    seed=None,
):
    # [FUNCTION DOC] Create random inputs for the LSTM benchmarks. Optionally seeds the RNG;
    # builds input of shape seq length by batch by input size, hx and cx, and an nn.LSTM (moved
    # to cuda if the device string contains cuda). Returns (input, (hx, cx), all_weights,
    # module); module is None unless return_module is true.
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, dropout=dropout)
    if "cuda" in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell, script):
    # [FUNCTION DOC] Return a dynamic_rnn function that runs the given cell over all time steps
    # of the input; compiled with TorchScript (cell and function) when script is true.
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] HELPER: Step the cell over the unbound time steps from the first layer's
        # hidden state. Returns stacked outputs and the final (hy, cy) with a leading layer
        # dimension added.
        hx, cx = hidden
        outputs = []
        inputs = input.unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), w_ih, w_hh, b_ih, b_hh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights
def lstm_factory_premul(premul_cell, script):
    # [FUNCTION DOC] Like lstm_factory but the input projection for all time steps is computed
    # upfront with a matmul (without bias) and the premul cell consumes it with w_hh and both
    # biases.
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] HELPER: Premultiply the input, then loop over time steps with the premul
        # cell. Returns stacked outputs and the final state with a leading layer dimension.
        hx, cx = hidden
        outputs = []
        inputs = torch.matmul(input, w_ih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), w_hh, b_ih, b_hh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights, and add bias
def lstm_factory_premul_bias(premul_cell, script):
    # [FUNCTION DOC] Like lstm_factory_premul but the input bias is added during the upfront
    # projection, done with a flattened matrix product and views; the cell is given only w_hh and
    # b_hh.
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] HELPER: Flatten input to 2D, project with w_ih and add b_ih, reshape
        # back and loop over time steps with the cell. Returns stacked outputs and final state.
        hx, cx = hidden
        outputs = []
        inpSize = input.size()
        # add bias for all timesteps instead of going step-by-step, results in a single reduction kernel in the backward
        # FIXME matmul(x,y) + bias currently goes through jit AD, and backward formula in AD is not optimized for this
        # case. Workaround with mm and views.
        inpSize = input.size()
        inputs = torch.mm(input.view(-1, inpSize[2]), w_ih.t()) + b_ih
        inputs = inputs.view(inpSize[0], inpSize[1], -1).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), w_hh, b_hh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# simple: flat inputs (no tuples), no list to accumulate outputs
#         useful mostly for benchmarking older JIT versions
def lstm_factory_simple(cell, script):
    # [FUNCTION DOC] Return a dynamic_rnn function using flat state (separate hx and cx tensors,
    # no tuples or output list); compiled with TorchScript when script is true.
    def dynamic_rnn(input, hx, cx, w_ih, w_hh, b_ih, b_hh):
        # [FUNCTION DOC] HELPER: Loop over time steps calling the cell with flat states. Returns
        # the final (hy, cy); it does not return the per step outputs.
        hy = hx  # for scoping
        cy = cx  # for scoping
        inputs = input.unbind(0)
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], hy, cy, w_ih, w_hh, b_ih, b_hh)
        return hy, cy

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def lstm_factory_multilayer(cell, script):
    # [FUNCTION DOC] Return a dynamic_rnn function that handles multiple layers; compiled with
    # TorchScript when script is true.
    def dynamic_rnn(
        input: Tensor, hidden: tuple[Tensor, Tensor], params: list[Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # [FUNCTION DOC] HELPER: For each layer take four parameters (w_ih, w_hh, b_ih, b_hh,
        # stride four, assuming biases exist), run the cell over time steps and feed the outputs
        # to the next layer. Returns the last layer's stacked outputs and its final state with a
        # leading layer dimension.
        params_stride = 4  # NB: this assumes that biases are there
        hx, cx = hidden
        hy, cy = hidden  # for scoping...
        inputs, outputs = input.unbind(0), []
        for layer in range(hx.size(0)):
            hy = hx[layer]
            cy = cx[layer]
            base_idx = layer * params_stride
            w_ih = params[base_idx]
            w_hh = params[base_idx + 1]
            b_ih = params[base_idx + 2]
            b_hh = params[base_idx + 3]
            for seq_idx in range(len(inputs)):
                hy, cy = cell(inputs[seq_idx], (hy, cy), w_ih, w_hh, b_ih, b_hh)
                outputs += [hy]
            inputs, outputs = outputs, []
        return torch.stack(inputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn
