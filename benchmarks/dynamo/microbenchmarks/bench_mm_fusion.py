# [MODULE DOC] Script benchmarking Inductor-generated Triton matmul against the ATen matmul,
# with and without fused bias add and relu epilogues, on layer shapes from alexnet, BERT and
# hf_GPT2. Requires CUDA. Import-time side effects: enables dense indexing in the Inductor
# triton config, seeds the RNG, allows TF32 matmul, and runs the whole benchmark, printing a
# PrettyTable of TFLOPS per shape; there is no main guard.
from prettytable import PrettyTable

import torch
import torch._dynamo
import torch._inductor.config
from torch._inductor.runtime.benchmarking import benchmarker


# torch._inductor.config.debug = True
torch._inductor.config.triton.dense_indexing = True
torch.manual_seed(0)


# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True


class Func:
    # [CLASS DOC] Namespace of four Dynamo-optimized (Inductor backend) matmul variants, each
    # decorated at definition time: plain matmul, matmul plus bias, matmul then relu, and matmul
    # plus bias then relu. They take no self argument and are accessed on the class itself,
    # never on instances.
    # mm
    @torch._dynamo.optimize("inductor")
    def mm(a, b, bias):
        # [FUNCTION DOC] Compute the matrix product of a and b. The bias argument is accepted
        # for a uniform signature and ignored. Returns the product.
        y = torch.mm(a, b)
        return y

    # mm+bias
    @torch._dynamo.optimize("inductor")
    def mm_add(a, b, bias):
        # [FUNCTION DOC] Compute the matrix product of a and b and add bias to the result.
        # Returns the sum.
        y = torch.mm(a, b)
        return y + bias

    # relu(mm)
    @torch._dynamo.optimize("inductor")
    def mm_relu(a, b, bias):
        # [FUNCTION DOC] Compute the matrix product of a and b, then apply relu. The bias
        # argument is unused. Returns the activated result.
        y = torch.mm(a, b)
        return torch.relu(y)

    # relu(mm+bias)
    @torch._dynamo.optimize("inductor")
    def mm_add_relu(a, b, bias):
        # [FUNCTION DOC] Compute the matrix product of a and b, add bias in place to the
        # product, then apply relu. Returns the activated result.
        y = torch.mm(a, b)
        y += bias
        return torch.relu(y)


def bench(shape, layer_id, p, fusion_types=None):
    # [FUNCTION DOC] Benchmark one matmul shape and append one row to the table. The shape is a
    # pair of operand shapes; layer_id labels the row; p is the PrettyTable; fusion_types lists
    # variants (empty string for plain, otherwise a suffix selecting a Func method; default is
    # plain only). Enables inductor metrics logging and restores default logging at the end. For
    # each variant it times the ATen configuration, then resets Dynamo and metrics, times the
    # Triton configuration, and appends both TFLOPS values. Raises an error if the Triton run
    # did not generate exactly one kernel. Uses float16 CUDA tensors.
    # KNOWN ISSUE: the benchmarker GPU method returns a single float, but the code unpacks three
    #     values from it, which fails at run time; the Inductor triton config also has no option
    #     named mm, so the aten versus triton toggle likely has no effect.
    torch._logging.set_logs(inductor_metrics=True)
    if fusion_types is None:
        fusion_types = [""]
    dtype = torch.float16
    M, K = shape[0]
    _, N = shape[1]
    torch.manual_seed(0)
    # allocate inputs
    a = torch.randn(shape[0], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    def tflops(ms):
        # [FUNCTION DOC] HELPER: Convert a time in milliseconds to throughput using the
        # enclosing matrix sizes (M times K times N divided by the time, scaled by one
        # billionth). Note this counts multiply-accumulates, not two operations per element.
        return M * K * N / ms * 1e-9

    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == "":
            fn_mm = Func.mm
        else:
            fn_mm = getattr(Func, f"mm_{fusion_type}")

        if "add" in fusion_type:
            bias = torch.randn((M, N), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (a, b, bias)

        def fn():
            # [FUNCTION DOC] HELPER: Call the currently selected Dynamo-optimized variant on the
            # captured argument tuple and return its result; used as the zero-argument callable
            # for the GPU benchmarker.
            return fn_mm(*args)

        torch._inductor.config.triton.mm = "aten"
        torch_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        torch._inductor.config.triton.mm = "triton"
        # reset to force code gen new python code
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        triton_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        if torch._inductor.metrics.generated_kernel_count != 1:
            raise AssertionError(
                f"Expected 1 generated kernel, but got {torch._inductor.metrics.generated_kernel_count}"
            )
        row.extend([tflops(torch_mm_ms), tflops(triton_mm_ms)])

    p.add_row(row)
    torch._logging.set_logs()


fusion_types = ["", "add", "relu", "add_relu"]
shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
    ([128, 4096], [4096, 4096]),
    ([128, 4096], [4096, 1000]),
    # BERT
    ([2048, 768], [768, 768]),
    ([2048, 768], [768, 3072]),
    ([2048, 3072], [3072, 768]),
    # hf_GPT2
    ([1024, 768], [768, 768]),
    ([1024, 768], [768, 3072]),
    ([1024, 3072], [3072, 768]),
    ([1024, 768], [768, 2304]),
]
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("torch mm")
        field_names.append("triton mm")
    else:
        field_names.append(f"torch mm+{fusion_type}")
        field_names.append(f"triton mm+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
for id, shape in enumerate(shapes):
    bench(shape, id, p, fusion_types)

print(p)
