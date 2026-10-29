# [MODULE DOC] Benchmark script for the FX common-subexpression-elimination pass
# (fx_graph_cse). Requires a CUDA device. At import time it creates a seeded random CUDA input
# of 2**20 elements and runs profile_function on several small functions with duplicated
# subexpressions, printing one CSV-like line per function: name, average CUDA time before CSE,
# average CUDA time after CSE, number of nodes removed, original node count.
import torch
import torch.fx as fx
from functorch import make_fx
from torch._functorch.compile_utils import fx_graph_cse
from torch.profiler import profile, ProfilerActivity


def profile_it(f, inp):
    # [FUNCTION DOC] HELPER: Time callable f on inp: five warmup calls, then five calls under
    # the torch profiler (CUDA activity only). Returns the summed CUDA time over all profiled
    # events divided by the iteration count (5). Side effects: runs f on the GPU repeatedly.
    for _ in range(5):
        f(inp)

    itr = 5
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    timing = prof.key_averages()
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total / itr


def profile_function(name, f, inp):
    # [FUNCTION DOC] HELPER: Trace f with make_fx on inp, apply fx_graph_cse to the traced graph
    # and rebuild a GraphModule, then profile both versions with profile_it. Prints one line:
    # name, avg CUDA time original, avg CUDA time after CSE, node-count decrease, original
    # node count. Returns nothing.
    fx_g = make_fx(f)(inp)

    new_g = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g)
    # do not benchmark against the scripted version because script already does some CSE
    # script_f = torch.jit.script(fx_g)
    # script_g = torch.jit.script(new_g)
    # avg_cuda_time_f = profile_it(script_f, inp)
    # avg_cuda_time_g = profile_it(script_g, inp)
    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    print(
        f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {num_node_decrease}, {len(fx_g.graph.nodes)}"
    )


g_gpu = torch.Generator(device="cuda")
g_gpu.manual_seed(2147483647)
inp = torch.randn(2**20, device="cuda", generator=g_gpu)


def f1(x):
    # [FUNCTION DOC] HELPER: Benchmark case with no redundancy: cosine applied twice to x.
    return x.cos().cos()


profile_function("f1", f1, inp)


def fsum(x):
    # [FUNCTION DOC] HELPER: Benchmark case computing the sum of x four times and adding them;
    # CSE should merge the four identical reductions.
    a = x.sum()
    b = x.sum()
    c = x.sum()
    d = x.sum()
    return a + b + c + d


profile_function("fsum", fsum, inp)


def fconcat(x):
    # [FUNCTION DOC] HELPER: Benchmark case concatenating x with itself twice and adding the
    # results; CSE should merge the duplicate concatenations.
    a = torch.cat((x, x))
    b = torch.cat((x, x))
    return a + b


profile_function("fconcat", fconcat, inp)


def fsum2(x):
    # [FUNCTION DOC] HELPER: Benchmark case adding the sum of x to an accumulator 31 times in
    # total (one initial plus 30 loop iterations), all redundant reductions.
    a = x.sum()
    for _ in range(30):
        a = a + x.sum()
    return a


profile_function("fsum2", fsum2, inp)


def fsummulti(x):
    # [FUNCTION DOC] HELPER: Benchmark case alternating add and multiply with the sum of x for
    # 3 iterations, starting from the number zero; redundant reductions.
    a = 0
    for _ in range(3):
        a = a + x.sum()
        a = a * x.sum()
    return a


profile_function("fsummulti", fsummulti, inp)


def fsummulti2(x):
    # [FUNCTION DOC] HELPER: Same as fsummulti but with 30 iterations.
    a = 0
    for _ in range(30):
        a = a + x.sum()
        a = a * x.sum()
    return a


profile_function("fsummulti2", fsummulti2, inp)


def fcos(x):
    # [FUNCTION DOC] HELPER: Benchmark case adding the cosine of x to an accumulator 3 times;
    # CSE should merge the repeated cosine calls.
    a = 0
    for _ in range(3):
        a = a + x.cos()
    return a


profile_function("fcos", fcos, inp)


def fcos2(x):
    # [FUNCTION DOC] HELPER: Same as fcos but with 30 iterations.
    a = 0
    for _ in range(30):
        a = a + x.cos()
    return a


profile_function("fcos2", fcos2, inp)
