# [MODULE DOC] Helpers for the sparse benchmarks: a cpu timing shim imitating the cuda event
# interface, and generators of random sparse matrices in coo and csr layouts. Random draws use
# numpy and python random, so results depend on their global seeds. No import-time side effects.
import functools
import operator
import random
import time

import numpy as np

import torch


# shim for torch.cuda.Event when running on cpu
class Event:
    # [CLASS DOC] HELPER: Cpu stand-in for the cuda event type. It records a wall-clock
    # timestamp from the performance counter and reports differences between two events, so
    # benchmark code can use one interface for both devices. The timing flag passed at
    # construction is ignored.
    def __init__(self, enable_timing):
        # [FUNCTION DOC] HELPER: Accepts and ignores the timing-enabled flag; stores no state.
        pass

    def record(self):
        # [FUNCTION DOC] HELPER: Stores the current performance-counter time (seconds) as the
        # event time.
        self.time = time.perf_counter()

    def elapsed_time(self, end_event):
        # [FUNCTION DOC] HELPER: Returns the time difference end minus start between the
        # given end event and this event. The unit is seconds, unlike cuda events, which
        # report milliseconds. Raises an assertion error if the argument is not an event of
        # this shim class.
        if not isinstance(end_event, Event):
            raise AssertionError(f"Expected Event, but got {type(end_event)}")
        return end_event.time - self.time


def gen_sparse_csr(shape, nnz):
    # [FUNCTION DOC] Builds a random float64 sparse csr matrix of the given shape with exactly
    # the requested number of stored positions: generates normally distributed dense values for
    # all elements, zeroes a random sample of the other positions, and converts the dense tensor
    # to csr. Stored count can be lower in the unlikely case a normal sample is exactly zero.
    fill_value = 0
    total_values = functools.reduce(operator.mul, shape, 1)
    dense = np.random.randn(total_values)
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = fill_value
    dense = torch.from_numpy(dense.reshape(shape))

    return dense.to_sparse_csr()


def gen_sparse_coo(shape, nnz):
    # [FUNCTION DOC] Builds a random sparse coo tensor for a two-dimensional shape. Draws the
    # requested number of random row and column positions, taking the values from a dense normal
    # sample at those positions. The same position can be drawn more than once, so the result
    # may contain duplicate (uncoalesced) entries and fewer distinct entries than requested.
    dense = np.random.randn(*shape)
    values = []
    indices = [[], []]
    for n in range(nnz):
        row = random.randint(0, shape[0] - 1)
        col = random.randint(0, shape[1] - 1)
        indices[0].append(row)
        indices[1].append(col)
        values.append(dense[row, col])

    return torch.sparse_coo_tensor(indices, values, size=shape)


def gen_sparse_coo_and_csr(shape, nnz):
    # [FUNCTION DOC] Builds one random dense float64 tensor with exactly the requested number of
    # nonzero positions (the others zeroed by random sampling) and returns a pair: its coo
    # conversion and its csr conversion, which therefore hold identical data.
    total_values = functools.reduce(operator.mul, shape, 1)
    dense = np.random.randn(total_values)
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = 0

    dense = torch.from_numpy(dense.reshape(shape))
    return dense.to_sparse(), dense.to_sparse_csr()
