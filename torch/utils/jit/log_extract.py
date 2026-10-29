# mypy: allow-untyped-defs
# [MODULE DOC] Tools for extracting TorchScript graph IR from logs and benchmarking it under
# different fusers (no fusion, NNC, nvfuser). Functions parse IR text, rebuild a callable with
# random inputs, and time it on CPU or GPU. Imports the benchmark Timer; no other side effects.
from contextlib import contextmanager
from typing import Any, cast
import random
import torch
import time
from torch.utils.benchmark import Timer

def extract_ir(filename: str) -> list[str]:
    # [FUNCTION DOC] Read a log file and return the graph texts found between the export
    # begin and end markers. The text preceding each begin marker on its line is treated as a
    # log prefix and stripped from every graph line. Segments without an end marker are skipped.
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    graphs = []
    with open(filename) as f:
        split_strs = f.read().split(BEGIN)
        for i, split_str in enumerate(split_strs):
            if i == 0:
                continue
            end_loc = split_str.find(END)
            if end_loc == -1:
                continue
            s = split_str[:end_loc]
            pfx = split_strs[i - 1].splitlines()[-1]
            lines = [x[len(pfx):] for x in s.splitlines(keepends=True)]
            graphs.append(''.join(lines))

    return graphs


def make_tensor_from_type(inp_type: torch._C.TensorType):
    # [FUNCTION DOC] Create an uninitialized tensor matching the sizes, strides, device and
    # dtype recorded in a TensorType. Raises AssertionError if any of them is unknown (None).
    size = inp_type.sizes()
    stride = inp_type.strides()
    device = inp_type.device()
    dtype = inp_type.dtype()
    if size is None:
        raise AssertionError("make_tensor_from_type: 'size' is None (inp_type.sizes() returned None)")
    if stride is None:
        raise AssertionError("make_tensor_from_type: 'stride' is None (inp_type.strides() returned None)")
    if device is None:
        raise AssertionError("make_tensor_from_type: 'device' is None (inp_type.device() returned None)")
    if dtype is None:
        raise AssertionError("make_tensor_from_type: 'dtype' is None (inp_type.dtype() returned None)")
    return torch.empty_strided(size=size, stride=stride, device=device, dtype=dtype)

def load_graph_and_inputs(ir: str) -> tuple[Any, list[Any]]:
    # [FUNCTION DOC] Parse IR text into a graph (multiple outputs folded into a tuple), build
    # a callable function from it with shape information erased, and generate random inputs
    # per graph input type: float, int, bool or tensor. Returns (function, inputs list).
    # Raises NotImplementedError for any other input type. Uses the global random generator.
    graph = torch._C.parse_ir(ir, parse_tensor_constants=True)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            tensorType = cast(torch._C.TensorType, inp.type())
            inputs.append(make_tensor_from_type(tensorType))
        elif isinstance(inp.type(), torch._C.BoolType):
            inputs.append(random.randint(0, 1) == 1)
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)

def time_cuda(fn, inputs, test_runs):
    # [FUNCTION DOC] Return the median runtime in milliseconds of calling fn on inputs, via the
    # benchmark Timer's blocked autorange (which synchronizes the device).
    # KNOWN ISSUE: test_runs is accepted but ignored.
    t = Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs" : inputs})
    times = t.blocked_autorange()
    return times.median * 1000  # time in ms

def time_cpu(fn, inputs, test_runs):
    # [FUNCTION DOC] Return the mean wall-clock time in milliseconds per call of fn on inputs,
    # measured over test_runs consecutive calls.
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs * 1000  # time in ms

def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    # [FUNCTION DOC] Build a callable from the IR text, run it warmup_runs times, then time it
    # in ms with the CPU timer if the first tensor input is on CPU, else the CUDA timer.
    # Raises AssertionError if no tensor is among inputs.
    # KNOWN ISSUE: the inputs from load_graph_and_inputs are discarded; the given inputs are used.
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    if is_cpu is None:
        raise AssertionError("No tensor found in inputs")

    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    return out

@contextmanager
def no_fuser(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Context manager that disables graph executor optimization inside
    # the block and restores the previous setting on exit. Arguments are ignored.
    old_optimize = torch._C._get_graph_executor_optimize(False)
    try:
        yield
    finally:
        torch._C._get_graph_executor_optimize(old_optimize)

def run_baseline_no_fusion(ir, inputs) -> float:
    # [FUNCTION DOC] Time the IR with graph optimization disabled; returns milliseconds.
    with no_fuser():
        return run_test(ir, inputs)


def run_nnc(ir, inputs, dynamic) -> float:
    # [FUNCTION DOC] Time the IR with the NNC fuser (fuser1) and a fusion strategy of dynamic
    # or static shapes with depth 10; restores the previous strategy afterwards. Returns ms.
    # KNOWN ISSUE: if setting the strategy fails, the finally block uses an unbound old_strat.
    try:
        strat = [("DYNAMIC", 10)] if dynamic else [("STATIC", 10)]
        old_strat = torch.jit.set_fusion_strategy(strat)
        with torch.jit.fuser("fuser1"):
            return run_test(ir, inputs)
    finally:
        torch.jit.set_fusion_strategy(old_strat)

def run_nvfuser(ir, inputs) -> float:
    # [FUNCTION DOC] Time the IR with the nvfuser fuser (fuser2) enabled; returns milliseconds.
    with torch.jit.fuser("fuser2"):
        return run_test(ir, inputs)
