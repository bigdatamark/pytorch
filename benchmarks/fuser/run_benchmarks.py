# [MODULE DOC] Command line benchmark comparing eager execution with three TorchScript fusers
# (nnc, nvfuser, legacy) on pointwise operators over many input shapes. Defines shape generator
# functions, operator functions, the timing helpers, functions that toggle the JIT fuser flags,
# and a click command that prints one CSV row per measurement (columns fuser, device, operator,
# shape, time in microseconds) to standard output. Import-time side effects: sets the number of
# CPU threads to one and disables fusion group inlining in the JIT. The output is consumed by
# plot_speedups.
import inspect
import itertools
import sys
import time

import click

import torch


torch.set_num_threads(1)
torch._C._debug_set_fusion_group_inlining(False)


def rand(*shape):
    # [FUNCTION DOC] HELPER: Create a random float tensor of the given shape with values uniform
    # in the range one to seventeen (uniform sample scaled by sixteen plus one), keeping inputs
    # away from zero and negatives so log and division are safe.
    return torch.rand(*shape).mul(16).add(1)


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------
def scalar():
    # [FUNCTION DOC] HELPER: Shape case: two tensors of one element each. Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(1), rand(1))


def small():
    # [FUNCTION DOC] HELPER: Shape case: two 1-D tensors of 32 elements. Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(32), rand(32))


def small_2d():
    # [FUNCTION DOC] HELPER: Shape case: two tensors of shape 1 by 32. Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(1, 32), rand(1, 32))


def small_broadcast():
    # [FUNCTION DOC] HELPER: Shape case: a 4 by 32 tensor and a 1-D tensor of 32 elements
    # (broadcast along the first dimension). Returns a tuple of two freshly created random
    # tensors, evaluated on every call.
    return (rand(4, 32), rand(32))


def medium():
    # [FUNCTION DOC] HELPER: Shape case: two contiguous tensors of shape 32 x 12 x 64 x 64.
    # Returns a tuple of two freshly created random tensors, evaluated on every call.
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))


def medium_sliced():
    # [FUNCTION DOC] HELPER: Shape case: two tensors from 32 x 12 x 64 x 64 inputs, each sliced
    # with stride two on the last dimension (non-contiguous views). Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])


def medium_transpose():
    # [FUNCTION DOC] HELPER: Shape case: two 32 x 12 x 64 x 64 tensors with the last two
    # dimensions transposed (non-contiguous views). Returns a tuple of two freshly created
    # random tensors, evaluated on every call.
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )


def medium2():
    # [FUNCTION DOC] HELPER: Shape case: two contiguous image-like tensors of shape 32 x 3 x 224
    # x 224. Returns a tuple of two freshly created random tensors, evaluated on every call.
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))


def medium3d():
    # [FUNCTION DOC] HELPER: Shape case: two 3-D tensors of shape 16 x 32 x 64. Returns a tuple
    # of two freshly created random tensors, evaluated on every call.
    return (rand(16, 32, 64), rand(16, 32, 64))


def medium_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: two 32 x 3 x 224 x 224 tensors converted to
    # channels-last memory format. Returns a tuple of two freshly created random tensors,
    # evaluated on every call.
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
    )


def medium_broadcast():
    # [FUNCTION DOC] HELPER: Shape case: a 32 x 12 x 64 x 64 tensor and a 1-D tensor of 64
    # elements (broadcast on the last dimension). Returns a tuple of two freshly created random
    # tensors, evaluated on every call.
    return (rand(32, 12, 64, 64), rand(64))


def medium_broadcast_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: a channels-last 32 x 3 x 223 x 223 tensor and a 3 x 1 x
    # 1 tensor (per-channel broadcast, odd spatial size). Returns a tuple of two freshly created
    # random tensors, evaluated on every call.
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))


def large():
    # [FUNCTION DOC] HELPER: Shape case: two large 8192 x 8192 tensors. Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(8192, 8192), rand(8192, 8192))


def large_transpose():
    # [FUNCTION DOC] HELPER: Shape case: two 8192 x 8192 tensors, each a transposed
    # (non-contiguous) view. Returns a tuple of two freshly created random tensors, evaluated on
    # every call.
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))


def large_channels_last():
    # [FUNCTION DOC] HELPER: Shape case: two large channels-last tensors of shape 32 x 32 x 256
    # x 256. Returns a tuple of two freshly created random tensors, evaluated on every call.
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
    )


def broadcast_narrow_57611():
    # [FUNCTION DOC] HELPER: Shape case regression input for issue 57611: a 1 x 32 x 32 x 2
    # tensor and a 1024 x 1 x 1 x 2 tensor, which broadcast against each other in both operands.
    # Returns a tuple of two freshly created random tensors, evaluated on every call.
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))


def large_broadcast_66816():
    # [FUNCTION DOC] HELPER: Shape case regression input for issue 66816: a 64 x 8 x 256 x 162
    # tensor and a 256 x 162 tensor (broadcast over leading dimensions). Returns a tuple of two
    # freshly created random tensors, evaluated on every call.
    return (rand(64, 8, 256, 162), rand(256, 162))


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------
def add(a, b):
    # [FUNCTION DOC] HELPER: Benchmark operator. Binary op: returns three times the first input
    # plus the second. Returns a tensor; no side effects.
    return 3 * a + b


def sub(a, b):
    # [FUNCTION DOC] HELPER: Benchmark operator. Binary op: returns three times the first input
    # minus the second. Returns a tensor; no side effects.
    return 3 * a - b


def mul(a, b):
    # [FUNCTION DOC] HELPER: Benchmark operator. Binary op: returns three times the first input
    # times the second. Returns a tensor; no side effects.
    return 3 * a * b


def div(a, b):
    # [FUNCTION DOC] HELPER: Benchmark operator. Binary op: returns three times the first input
    # divided by the second. Returns a tensor; no side effects.
    return 3 * a / b


def relu(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and applies
    # relu. Returns a tensor; no side effects.
    return (3 * a).relu()


def sigmoid(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and applies
    # sigmoid. Returns a tensor; no side effects.
    return (3 * a).sigmoid()


def tanh(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and applies
    # tanh. Returns a tensor; no side effects.
    return (3 * a).tanh()


def log(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and applies
    # the natural log. Returns a tensor; no side effects.
    return (3 * a).log()


def exp(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and applies
    # exp. Returns a tensor; no side effects.
    return (3 * a).exp()


def square(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: scales the input by three and squares
    # the result with the power operator. Returns a tensor; no side effects.
    return (3 * a) ** 2


def fma(a, b):
    # [FUNCTION DOC] HELPER: Benchmark operator. Binary op: multiply-add, the product of both
    # inputs plus the second input. Returns a tensor; no side effects.
    return a * b + b


def mul_mul_add_66816(a, b, c):
    # [FUNCTION DOC] HELPER: Benchmark operator. Ternary op for issue 66816: the product of the
    # first two inputs plus the product of the first and third inputs. Returns a tensor; no
    # side effects.
    return (a * b) + (a * c)


def hardswish_int(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: hardswish written with clamp bounds
    # as integers (zero and six); computes the input times the clamped input plus three, divided
    # by six. Returns a tensor; no side effects.
    return a * (a + 3).clamp(0, 6) / 6


def hardswish(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: hardswish written with float clamp
    # bounds; same formula as the integer-bound variant. Returns a tensor; no side effects.
    return a * (a + 3).clamp(0.0, 6.0) / 6


def native_hardswish(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: calls the native hardswish kernel on
    # the input scaled by three. Returns a tensor; no side effects.
    return torch._C._nn.hardswish(a * 3)


def softplus(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: softplus composed from exp, log1p and
    # divisions by one, with no overflow protection. Returns a tensor; no side effects.
    return (a * 1.0).exp().log1p() / 1.0


def mish(a):
    # [FUNCTION DOC] HELPER: Benchmark operator. Unary op: mish composed as the input times tanh
    # of the composed softplus value. Returns a tensor; no side effects.
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()


SHAPES = [
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
    broadcast_narrow_57611,
    large_broadcast_66816,
]

OPERATORS = [
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
    mul_mul_add_66816,
    hardswish_int,
    hardswish,
    native_hardswish,
    softplus,
    mish,
]


def time_cpu(fn, args, iters):
    # [FUNCTION DOC] HELPER: Time a CPU call loop. Calls fn with the unpacked args the given
    # number of times and returns the total wall clock seconds measured with a
    # performance counter.
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s


def time_cuda(fn, args, iters):
    # [FUNCTION DOC] HELPER: Time a CUDA call loop. Records CUDA events around the given number
    # of calls of fn on the unpacked args, synchronizes the device, and returns the total
    # elapsed time in seconds (the event milliseconds divided by one thousand). Requires CUDA.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    # [FUNCTION DOC] HELPER: Estimate per-call latency in seconds with an adaptive iteration
    # count. Runs three warm-up iterations, times one iteration to calibrate, then chooses a
    # count of iterations that should take about one second and returns the total time for that
    # many iterations divided by the count. Raises ZeroDivisionError if the calibration time is
    # so large that the count is zero (more than one second per call).
    timer(fn, args, 3)
    calibration = timer(fn, args, 1)
    iters = int(1.0 / calibration)
    return timer(fn, args, iters) / iters


def benchmark(fn, args):
    # [FUNCTION DOC] HELPER: Benchmark fn on args using the CPU timer if the first argument
    # lives on the CPU, otherwise the CUDA timer. Returns seconds per call.
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    # [FUNCTION DOC] HELPER: Format a time in seconds as microseconds with one decimal digit,
    # returned as a string.
    return f"{s * 1e6:.1f}"


def with_nvfuser():
    # [FUNCTION DOC] HELPER: Configure TorchScript globals so that the nvfuser fuser is used:
    # CPU and GPU fusion overrides and the texpr (nnc) fuser are turned off, nvfuser is on, and
    # the profiling executor and mode are on. Process-wide side effects.
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_nnc():
    # [FUNCTION DOC] HELPER: Configure TorchScript globals so that the nnc (texpr) fuser is
    # used: CPU and GPU fusion overrides and the texpr fuser are on, nvfuser is off, and the
    # profiling executor and mode are on. Process-wide side effects.
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_legacy():
    # [FUNCTION DOC] HELPER: Configure TorchScript globals for the legacy fuser: CPU and GPU
    # fusion overrides are on, the texpr fuser and nvfuser are off, and the profiling executor
    # and mode are off. Process-wide side effects.
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)


@click.command()
@click.option("--operators", default=None)
@click.option("--shapes", default=None)
def run_benchmarks(operators, shapes):
    # [FUNCTION DOC] Click command entry point. Options select comma separated operator names
    # and shape names (looked up by name in this module's global namespace; unknown names raise
    # KeyError); each defaults to the full list. For every pair of shape and operator it builds
    # the arguments (trimmed to the operator's parameter count, or padded by repeating the last
    # tensor), moves them all to CUDA, and prints a CSV header and then one row for eager mode
    # and one each for traced nnc, nvfuser and legacy runs, flushing output after each fuser
    # row. Requires a CUDA device. Changes global fuser flags.
    # KNOWN ISSUE: the device column comes from the CUDA tensors, so the CPU timer branch is
    # unreachable because the arguments are always moved to CUDA.
    if operators is None:
        operators = OPERATORS
    else:
        operators = [globals()[k] for k in operators.split(",")]
    if shapes is None:
        shapes = SHAPES
    else:
        shapes = [globals()[k] for k in shapes.split(",")]

    print("fuser,device,operator,shape,time")
    for shape, operator in itertools.product(shapes, operators):
        nargs = len(inspect.signature(operator).parameters)
        args = shape()
        if nargs > len(args):
            args = list(args)
            args += [args[-1]] * (nargs - len(args))
        args = args[:nargs]
        args = [arg.to("cuda") for arg in args]

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

        def bench(name):
            # [FUNCTION DOC] HELPER: Trace the current operator with the current arguments,
            # benchmark the traced function, and print a CSV row labelled with the given fuser
            # name, the device type, operator name, shape name and microsecond time. Flushes
            # standard output. Uses variables of the enclosing loop iteration.
            nnc_op = torch.jit.trace(operator, args)
            result = benchmark(nnc_op, args)
            print(
                ",".join(
                    [
                        name,
                        args[0].device.type,
                        operator.__name__,
                        shape.__name__,
                        micros(result),
                    ]
                )
            )
            sys.stdout.flush()

        with_nnc()
        bench("nnc")
        with_nvfuser()
        bench("nvfuser")
        with_legacy()
        bench("legacy")


if __name__ == "__main__":
    run_benchmarks()
