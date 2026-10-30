# [MODULE DOC] Pure tensor RNN cell functions used by the fastrnns benchmarks and as TorchScript
# inputs: a multiplicative-integration LSTM cell, standard LSTM cell variants (tuple state, flat
# state, with pre-multiplied input gates with and without bias), and GRU/RNN relu/RNN tanh cells.
# No import-time side effects.
import torch
from torch import Tensor


def milstm_cell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    # [FUNCTION DOC] Multiplicative-integration LSTM cell (section 2.1 of the cited paper).
    # Computes input and hidden projections, combines them as alpha*Wx*Uz + beta_i*Wx + beta_h*Uz
    # + bias, splits the result in four gate chunks along dim 1 and applies the usual LSTM
    # update. Returns (hy, cy).
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def lstm_cell(
    input: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Standard LSTM cell taking hidden as a (hx, cx) pair. Gates are the sum of
    # the input and hidden matrix products (with transposed weights) and both biases, split into
    # four chunks (input, forget, cell, output) with sigmoid/tanh activations. Returns the new
    # (hy, cy).
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def flat_lstm_cell(
    input: Tensor,
    hx: Tensor,
    cx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Same math as lstm_cell but the hidden and cell states are passed as two
    # separate tensors (hx, cx) instead of a tuple. Returns the new (hy, cy).
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell(
    igates: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] LSTM cell whose input projection was computed ahead of time: igates replaces
    # the input matrix product. Gates are igates plus hidden product plus both biases. Returns
    # (hy, cy).
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell_no_bias(
    igates: Tensor, hidden: tuple[Tensor, Tensor], w_hh: Tensor, b_hh: Tensor
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Pre-multiplied input LSTM cell with a single bias b_hh (no b_ih term);
    # otherwise identical to premul_lstm_cell. Returns the new (hy, cy).
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # [FUNCTION DOC] GRU cell. Computes input and hidden projections with biases, splits each
    # into three chunks, builds reset and input gates and the candidate new gate, and returns the
    # new hidden state (a single tensor) as newgate plus inputgate times (hidden minus newgate).
    gi = torch.mm(input, w_ih.t()) + b_ih
    gh = torch.mm(hidden, w_hh.t()) + b_hh
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # [FUNCTION DOC] Vanilla RNN cell with relu nonlinearity: relu of the input and hidden
    # projections (each with its bias). Returns the new hidden tensor.
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.relu(igates + hgates)


def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # [FUNCTION DOC] Vanilla RNN cell with tanh nonlinearity: tanh of the input and hidden
    # projections (each with its bias). Returns the new hidden tensor.
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.tanh(igates + hgates)
