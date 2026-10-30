# This is a copy of rnn_attention from MLPerf, with some common sizes hardcoded
# for benchmarking and some control flow stripped out.
# https://github.com/mlcommons/training/blob/master/retired_benchmarks/gnmt/pytorch/seq2seq/models/attention.py
# [MODULE DOC] Benchmark of Bahdanau (additive) attention scoring, copied from an MLPerf
# translation model with fixed sizes. Defines one benchmark class on top of the tensor
# expression benchmark base and registers it with the harness at import time.

import torch

from . import benchmark


class BahdanauAttention(benchmark.Benchmark):
    # [CLASS DOC] Benchmark computing additive attention scores: query and key tensors are
    # broadcast against each other, added with a bias, passed through tanh and reduced with a
    # vector by matmul, giving a score tensor of batch by query length by key length. Holds the
    # sizes and the four input tensors (query, keys, bias, projection vector), which require
    # gradients when the mode is both.
    def __init__(self, mode, device, dtype, b, t_q, t_k, n):
        # [FUNCTION DOC] Stores the batch, query length, key length and feature sizes and
        # creates the four random inputs through the base class: query (batch, query length,
        # features), keys (batch, key length, features), a bias vector and a projection vector
        # of the feature size. Collects them in the inputs list.
        super().__init__(mode, device, dtype)
        self.b = b
        self.t_q = t_q
        self.t_k = t_k
        self.n = n
        self.att_query = self.rand(
            [b, t_q, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.att_keys = self.rand(
            [b, t_k, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.normalize_bias = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.linear_att = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.att_query,
            self.att_keys,
            self.normalize_bias,
            self.linear_att,
        ]

    def forward(self, att_query, att_keys, normalize_bias, linear_att):
        # [FUNCTION DOC] Computes the attention scores: expands the query and keys to four
        # dimensions so every query position pairs with every key position, adds them and the
        # bias, applies tanh, and multiplies by the projection vector. Returns a tensor of batch
        # by query length by key length.
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        return b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys + normalize_bias
        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def reference(self):
        # [FUNCTION DOC] Returns the forward result on the stored inputs converted to a numpy
        # array, used as the reference for checking.
        return self.numpy(self.forward(*self.inputs))

    def config(self):
        # [FUNCTION DOC] Returns the list of sizes: batch, query length, key length, features.
        # Used in the benchmark description.
        return [self.b, self.t_q, self.t_k, self.n]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the benchmark module name used for selection and descriptions.
        return "attention"

    def memory_workload(self):
        # [FUNCTION DOC] Estimates data movement: the byte size of all four inputs plus the byte
        # size of the output (assuming four bytes per element) gives the minimum traffic; the
        # algorithmic figure adds the cost of writing and reading the intermediate sum when the
        # matmul is not fused (two times four bytes per element of the four-dimensional
        # intermediate). Returns a mapping with the entries sol and algorithmic.
        # KNOWN ISSUE: these figures are already in bytes, but the harness multiplies the
        #     workload by the dtype size again when computing bandwidth, so the reported GB/s
        #     figures are inflated.
        def memsize(t):
            # [FUNCTION DOC] HELPER: Returns the size in bytes of a tensor (element count times
            # element size).
            return t.numel() * t.element_size()

        input_size = (
            memsize(self.att_query)
            + memsize(self.att_keys)
            + memsize(self.normalize_bias)
            + memsize(self.linear_att)
        )
        output_size = 4 * torch.Size([self.b, self.t_q, self.t_k]).numel()
        io_size = input_size + output_size

        # If matmul is not fused, must write and then read `sum_qk`.
        intermediate_size = (
            2 * 4 * torch.Size([self.b, self.t_q, self.t_k, self.n]).numel()
        )
        return {"sol": io_size, "algorithmic": io_size + intermediate_size}

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns two default configs as lists of batch, query length, key length
        # and features: one from the MLPerf inference setup and one from an NVIDIA setup.
        mlperf_inference = [1280, 1, 66, 1024]
        nvidia = [128, 10, 128, 1024]
        return [mlperf_inference, nvidia]


benchmark.register_benchmark_class(BahdanauAttention)
