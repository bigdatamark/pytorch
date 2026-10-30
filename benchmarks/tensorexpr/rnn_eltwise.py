# [MODULE DOC] Tensor-expression benchmark of the pointwise part of an LSTM cell (gate sums,
# activations and state update), with a static and a dynamic shape variant. Both classes are
# registered with the benchmark registry at import time.
import torch

from . import benchmark


class RNNEltwise(benchmark.Benchmark):
    # [CLASS DOC] Benchmark of the elementwise portion of an LSTM cell for batch b and hidden
    # size hs. Holds five random inputs: input and hidden projections, previous cell state,
    # and two bias matrices, all collected in the inputs list.
    def __init__(self, mode, device, dtype, b, hs):
        # [FUNCTION DOC] Stores batch and hidden sizes and creates the random tensors (four of
        # shape batch by four times hidden, one cell state of batch by hidden), tracking
        # gradients in the both mode. Sets the inputs list in the order input, hx, cx, b_ih,
        # b_hh.
        super().__init__(mode, device, dtype)
        self.b = b
        self.hs = hs
        self.input = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.hx = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.cx = self.rand(
            [b, hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_ih = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_hh = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    def forward(self, input, hx, cx, b_ih, b_hh):
        # [FUNCTION DOC] Sums the four gate inputs, splits the result into four chunks along
        # the feature dimension, applies sigmoid to the input, forget and output gates and
        # tanh to the cell gate, then computes the new cell state and the new hidden state.
        # Returns the pair of new hidden and cell states.
        gates = input + hx + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: batch size and hidden size.
        return [self.b, self.hs]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for this benchmark.
        return "rnn_eltwise"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping where both sol and algorithmic traffic equal the
        # total bytes of all inputs plus two cell-state sized outputs (new hidden and cell
        # states).
        def memsize(t):
            # [FUNCTION DOC] HELPER: Returns the size in bytes of a tensor from its element
            # count and element size.
            return t.numel() * t.element_size()

        input_size = sum(memsize(t) for t in self.inputs)
        output_size = 2 * memsize(self.cx)
        io_size = input_size + output_size
        return {"sol": io_size, "algorithmic": io_size}

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry of batch 64 and hidden
        # size 512.
        return [[64, 512]]


benchmark.register_benchmark_class(RNNEltwise)


class DynamicLSTM(benchmark.DynamicShape, RNNEltwise):
    # [CLASS DOC] Dynamic shape version of the LSTM elementwise benchmark; inputs of
    # randomized shapes are precomputed and cycled by the mixin.
    def __init__(self, mode, device, dtype, b, hs):
        # [FUNCTION DOC] Initializes the dynamic shape state, then the LSTM base with the
        # given mode, device, dtype, batch and hidden size.
        benchmark.DynamicShape.__init__(self)
        RNNEltwise.__init__(self, mode, device, dtype, b, hs)

    def instantiate_input(self):
        # [FUNCTION DOC] Draws randomized batch and hidden sizes near the configured ones,
        # creates a new set of five random tensors with consistent shapes, stores them as
        # attributes and assigns the inputs list, which the mixin records as a new sample.
        b, hs = self.rand_shape([self.b, self.hs])

        self.input = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.hx = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.cx = self.rand(
            [b, hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.b_ih = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.b_hh = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    @staticmethod
        # [FUNCTION DOC] Returns the module name string for the dynamic LSTM benchmark.
    def module():
        return "dynamic_lstm"


benchmark.register_benchmark_class(DynamicLSTM)
