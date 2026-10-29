# [MODULE DOC] Benchmark script producing a CSV scorecard (fuser,device,operator,shape,time in
# microseconds) for pointwise operators over many tensor shapes/layouts. For each pair it
# measures eager, pointwise_operator-compiled ("pointwise", NaN if it fails) and TorchScript
# ("fuser") versions. Import-time side effects: sets one thread, disables fusion group
# inlining, validates each operator against both compiled versions (printing failures), then
# runs and prints all measurements to stdout. The output feeds process_scorecard.py.
# KNOWN ISSUE: pointwise_operator is imported from the functorch package, which may not export
# it, so import can fail.
import inspect
import itertools
import sys
import time

import torch
from functorch import pointwise_operator


torch.set_num_threads(1)
torch._C._debug_set_fusion_group_inlining(False)


def rand(*shape):
    # [FUNCTION DOC] HELPER: Random float tensor of the given shape with values in [1, 17).
    return torch.rand(*shape).mul(16).add(1)


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------
def scalar():
    # [FUNCTION DOC] HELPER: Shape case: two one-element tensors.
    return (rand(1), rand(1))


def small():
    # [FUNCTION DOC] HELPER: Shape case: two 1-D tensors of 32 elements.
    return (rand(32), rand(32))


def small_2d():
    # [FUNCTION DOC] HELPER: Shape case: two tensors of shape 1x32.
    return (rand(1, 32), rand(1, 32))


def small_broadcast():
    # [FUNCTION DOC] HELPER: Shape case: a 4x32 tensor and a 32-element tensor (broadcast).
    return (rand(4, 32), rand(32))


def medium():
    # [FUNCTION DOC] HELPER: Shape case: two contiguous tensors of shape 32x12x64x64.
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))


def medium_sliced():
    # [FUNCTION DOC] HELPER: Shape case: two 32x12x64x64 tensors sliced to every second element
    # of the last dimension (non-contiguous).
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])


def medium_transpose():
    # [FUNCTION DOC] HELPER: Shape case: two 32x12x64x64 tensors with the last two dimensions
    # transposed.
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )


def medium2():
    # [FUNCTION DOC] HELPER: Shape case: two contiguous tensors of shape 32x3x224x224.
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))


def medium3d():
    # [FUNCTION DOC] HELPER: Shape case: two tensors of shape 16x32x64.
    return (rand(16, 32, 64), rand(16, 32, 64))


def medium_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: two 32x3x224x224 tensors in channels-last layout.
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
    )


def medium_broadcast():
    # [FUNCTION DOC] HELPER: Shape case: a 32x12x64x64 tensor and a 64-element tensor.
    return (rand(32, 12, 64, 64), rand(64))


def medium_broadcast_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: a 32x3x223x223 channels-last tensor and a 3x1x1 tensor.
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))


def large():
    # [FUNCTION DOC] HELPER: Shape case: two contiguous 8192x8192 tensors.
    return (rand(8192, 8192), rand(8192, 8192))


def large_transpose():
    # [FUNCTION DOC] HELPER: Shape case: two transposed 8192x8192 tensors.
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))


def large_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: two 32x32x256x256 tensors in channels-last layout.
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
    )


def pathological_broadcast():
    # [FUNCTION DOC] HELPER: Shape case: tensors of shape 1x32x32x2 and 1024x1x1x2 that
    # broadcast against each other in both directions.
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------
def add(a, b):
    # [FUNCTION DOC] HELPER: Operator case: a plus b.
    return a + b


def sub(a, b):
    # [FUNCTION DOC] HELPER: Operator case: a minus b.
    return a - b


def mul(a, b):
    # [FUNCTION DOC] HELPER: Operator case: a times b.
    return a * b


def div(a, b):
    # [FUNCTION DOC] HELPER: Operator case: a divided by b.
    return a / b


def relu(a):
    # [FUNCTION DOC] HELPER: Operator case: rectified linear unit of a.
    return a.relu()


def sigmoid(a):
    # [FUNCTION DOC] HELPER: Operator case: logistic sigmoid of a.
    return a.sigmoid()


def tanh(a):
    # [FUNCTION DOC] HELPER: Operator case: hyperbolic tangent of a.
    return a.tanh()


def log(a):
    # [FUNCTION DOC] HELPER: Operator case: natural logarithm of a.
    return a.log()


def exp(a):
    # [FUNCTION DOC] HELPER: Operator case: exponential of a.
    return a.exp()


def square(a):
    # [FUNCTION DOC] HELPER: Operator case: a raised to the power two.
    return a**2


def fma(a, b):
    # [FUNCTION DOC] HELPER: Operator case: a times b plus b (the addend is b, not a third arg).
    return a * b + b


def hardswish(a):
    # [FUNCTION DOC] HELPER: Operator case: hardswish written out as a times (a plus 3 clamped
    # to the range 0 to 6) divided by 6.
    return a * (a + 3.0).clamp(0.0, 6.0) / 6.0


def native_hardswish(a):
    # [FUNCTION DOC] HELPER: Operator case: the built-in native hardswish kernel applied to a.
    return torch._C._nn.hardswish(a)


def softplus(a):
    # [FUNCTION DOC] HELPER: Operator case: softplus written as log1p of exp of a (beta 1).
    # Defined but not listed in the operators list, so it is never benchmarked.
    return (a * 1.0).exp().log1p() / 1.0


def mish(a):
    # [FUNCTION DOC] HELPER: Operator case: mish, a times tanh of softplus of a. Defined but not
    # listed in the operators list, so it is never benchmarked.
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def time_cpu(fn, args, iters):
    # [FUNCTION DOC] HELPER: Call fn with args iters times and return total wall-clock seconds.
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s


def time_cuda(fn, args, iters):
    # [FUNCTION DOC] HELPER: Call fn with args iters times between two CUDA events, synchronize,
    # and return the total elapsed time in seconds.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    # [FUNCTION DOC] HELPER: Warm up with 3 calls, calibrate with 1 timed call, choose an
    # iteration count targeting about one second, and return the mean seconds per call.
    # KNOWN ISSUE: a calibration time of zero would raise a division error.
    timer(fn, args, 3)
    calibration = timer(fn, args, 1)
    iters = int(1.0 / calibration)
    return timer(fn, args, iters) / iters


def benchmark(fn, args):
    # [FUNCTION DOC] HELPER: Benchmark fn on args using the CPU timer if the first argument is
    # on CPU, otherwise the CUDA timer. Returns mean seconds per call.
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    # [FUNCTION DOC] HELPER: Format seconds s as microseconds with one decimal place (string).
    return f"{s * 1e6:.1f}"


shapes = [
    scalar,
    small,
    small_2d,
    small_broadcast,
    medium,
    medium2,
    medium3d,
    medium_sliced,
    medium_transpose,
    medium_channels_last,
    medium_broadcast,
    medium_broadcast_channels_last,
    large,
    large_transpose,
    large_channels_last,
    pathological_broadcast,
]

operators = [
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    tanh,
    log,
    exp,
    square,
    fma,
    hardswish,
    native_hardswish,
]

nope = set()
for shape, operator in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]

    try:
        if shape is medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        pw_op = pointwise_operator(operator)
        torch.testing.assert_close(operator(*args), pw_op(*args))
    except Exception:
        print(f"pointwise_operator failed on {operator.__name__}, {shape.__name__}")
        nope.add((operator, shape))

    ts_op = torch.jit.script(operator)
    torch.testing.assert_close(operator(*args), ts_op(*args))


print("fuser,device,operator,shape,time")
results = []
for shape, operator in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]

    result = benchmark(operator, args)
    print(
        ",".join(
            [
                "eager",
                args[0].device.type,
                operator.__name__,
                shape.__name__,
                micros(result),
            ]
        )
    )
    try:
        if shape is medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        if (operator, shape) in nope:
            raise RuntimeError("pointwise_operator fails on medium_transpose")
        pw_op = pointwise_operator(operator)
        result = benchmark(pw_op, args)
        print(
            ",".join(
                [
                    "pointwise",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(result),
                ]
            )
        )
    except Exception:
        print(
            ",".join(
                [
                    "pointwise",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(float("nan")),
                ]
            )
        )

    ts_op = torch.jit.script(operator)
    result = benchmark(ts_op, args)
    print(
        ",".join(
            [
                "fuser",
                args[0].device.type,
                operator.__name__,
                shape.__name__,
                micros(result),
            ]
        )
    )
    sys.stdout.flush()
