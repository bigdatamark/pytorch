# [MODULE DOC] Script comparing a CUDA convolution on a default-layout input against the same
# convolution on a channels-last copy, with profiler trace export and a numeric check. Importing
# has no side effects.
import torch
from torch._inductor import ir
from torch._inductor.runtime.benchmarking import benchmarker


def to_channels_last(x):
    # [FUNCTION DOC] Return a copy of a four dimensional tensor laid out in channels-last memory
    # order, built by cloning, viewing with strides derived from Inductor's flexible layout
    # stride ordering and copying the values in. Raises an AssertionError if the input is not 4D
    # or the copy differs from the input.
    if x.dim() != 4:
        raise AssertionError(f"Expected 4D tensor, but got {x.dim()}D")

    # NCHW -> NHWC
    stride_order = [3, 0, 2, 1]
    y = x.clone().as_strided(
        x.shape,
        ir.FlexibleLayout.stride_ordered(x.shape, stride_order),
    )
    y.copy_(x)
    if not torch.allclose(x, y):
        raise AssertionError("Tensor copy failed: x and y are not close")
    return y


def bench_conv(with_stack=True):
    # [FUNCTION DOC] Benchmark convolution of a random 256x3x224x224 input with 64 7x7 kernels
    # (stride 2, padding 3), both in default and channels-last layout. Warms up, runs both under
    # the profiler (with Python stacks if requested) and exports a Chrome trace to a fixed
    # temporary path as a side effect. Raises an AssertionError if outputs differ beyond
    # tolerance, then times each with the GPU benchmarker and prints times and speedup.
    x = torch.rand(256, 3, 224, 224).cuda()
    weight = torch.rand(64, 3, 7, 7).cuda()

    x_chan = to_channels_last(x)
    weight_chan = to_channels_last(weight)
    kwargs = {
        "stride": [2, 2],
        "padding": [3, 3],
        "dilation": [1, 1],
        "transposed": False,
        "output_padding": [0, 0],
        "groups": 1,
    }

    def baseline_fn():
        # [FUNCTION DOC] HELPER: Convolution of the default layout input and weight with the
        # shared options.
        return torch.convolution(x, weight, bias=None, **kwargs)

    def test_fn():
        # [FUNCTION DOC] HELPER: Convolution of the channels-last input and weight with the
        # shared options.
        return torch.convolution(x_chan, weight_chan, bias=None, **kwargs)

    # warmup
    baseline_fn()
    test_fn()

    torch.cuda.synchronize()
    with torch.profiler.profile(with_stack=with_stack) as p:
        baseline_out = baseline_fn()
        test_out = test_fn()
        torch.cuda.synchronize()

    p.export_chrome_trace("/tmp/chrome.json")
    if not torch.allclose(baseline_out, test_out, atol=1e-3, rtol=1e-3):
        raise AssertionError(
            f"baseline_out and test_out are not close: "
            f"baseline={baseline_out[0][0][0][:32]}, test={test_out[0][0][0][:32]}"
        )

    baseline_ms = benchmarker.benchmark_gpu(baseline_fn, rep=40)
    test_ms = benchmarker.benchmark_gpu(test_fn, rep=40)
    print(f"baseline {baseline_ms} test {test_ms} speedup {baseline_ms / test_ms:.3f}x")


def main():
    # [FUNCTION DOC] Run the convolution layout benchmark with its defaults.
    bench_conv()


if __name__ == "__main__":
    main()
