# [MODULE DOC] Standalone script comparing NNC (tensor expression, LLVM codegen) kernels
# against eager PyTorch for many unary, binary and custom elementwise ops over square sizes.
# Module-level code builds the op tables and the combined benchmark list at import time
# (importing matplotlib, pandas, seaborn and the tensor expression bindings). Run as a script,
# it benchmarks every op, prints timings and saves a heatmap image of the PyTorch to NNC time
# ratio in the current directory.
import argparse
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch._C._te as te


class kernel_arena_scope:
    # [CLASS DOC] Context manager that creates a tensor expression kernel scope on entry and
    # drops the reference on exit. Keeps the scope in an attribute; exceptions are not
    # suppressed.
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Creates a tensor expression kernel scope and stores it on the
        # instance. Returns nothing.
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        # [FUNCTION DOC] HELPER: Releases the stored kernel scope by clearing the attribute.
        # Ignores the exception arguments and returns nothing, so exceptions propagate.
        self.scope = None


unary_ops = [
    ("sin", torch.sin),
    ("cos", torch.cos),
    ("tan", torch.tan),
    ("asin", torch.asin),
    ("acos", torch.acos),
    ("atan", torch.atan),
    ("sinh", torch.sinh),
    ("cosh", torch.cosh),
    ("tanh", torch.tanh),
    ("sigmoid", torch.sigmoid),
    ("exp", torch.exp),
    ("expm1", torch.expm1),
    ("expm1", torch.expm1),
    ("abs", torch.abs),
    ("log", torch.log),
    ("fast_log", torch.log),
    ("log2", torch.log2),
    ("log10", torch.log10),
    ("log1p", torch.log1p),
    ("erf", torch.erf),
    ("erfc", torch.erfc),
    ("sqrt", torch.sqrt),
    ("rsqrt", torch.rsqrt),
    ("ceil", torch.ceil),
    ("floor", torch.floor),
    ("round", torch.round),
    ("trunc", torch.trunc),
    ("lgamma", torch.lgamma),
    # ("frac", torch.frac), # seems unimplemented
    # ("isnan", torch.isnan), # no out variant
]


def gen_unary_nnc_fun(nnc_name):
    # [FUNCTION DOC] HELPER: Builds the NNC compute-function factory for a unary op: the
    # returned function takes two placeholders and returns a per-element function that loads
    # the first placeholder at the index pair and calls the method with the given name on the
    # loaded expression. The second placeholder is ignored.
    def nnc_fun(A, B):
        # [FUNCTION DOC] HELPER: Receives the input placeholders and returns the per-element
        # compute function for the captured op name.
        def compute(i, j):
            # [FUNCTION DOC] HELPER: Per-element body: loads the first placeholder at the row
            # and column index and applies the named expression method to it.
            return getattr(A.load([i, j]), nnc_name)()

        return compute

    return nnc_fun


def gen_unary_torch_fun(torch_op):
    # [FUNCTION DOC] HELPER: Builds the eager-side factory for a unary torch op: the returned
    # function binds the input and the preallocated output tensor and yields a callable that
    # runs the op writing into the output.
    def torch_fun(a, b, out):
        # [FUNCTION DOC] HELPER: Binds the first tensor and the output tensor and returns a
        # zero-argument callable. The second tensor is ignored.
        def fun():
            # [FUNCTION DOC] HELPER: Runs the captured torch op on the bound input, writing
            # into the bound output tensor, and returns the result.
            return torch_op(a, out=out)

        return fun

    return torch_fun


def gen_binary_nnc_fun(fn):
    # [FUNCTION DOC] HELPER: Builds the NNC compute-function factory for a binary op given as
    # a callable on expressions: the returned function takes two placeholders and yields a
    # per-element function applying the callable to the loads of both at the index pair.
    def nnc_fun(A, B):
        # [FUNCTION DOC] HELPER: Receives the two input placeholders and returns the per-
        # element compute function for the captured binary callable.
        def compute(i, j):
            # [FUNCTION DOC] HELPER: Per-element body: loads both placeholders at the row and
            # column index and returns the captured callable applied to the two values.
            return fn(A.load([i, j]), B.load([i, j]))

        return compute

    return nnc_fun


def gen_binary_torch_fun(fn):
    # [FUNCTION DOC] HELPER: Builds the eager-side factory for a binary torch op: the returned
    # function binds both inputs and the output tensor and yields a callable that runs the op
    # with the output argument set.
    def pt_fun(a, b, out):
        # [FUNCTION DOC] HELPER: Binds the two input tensors and the output tensor and returns
        # a zero-argument callable.
        def fun():
            # [FUNCTION DOC] HELPER: Runs the captured torch op on the two bound inputs,
            # writing into the bound output tensor, and returns the result.
            return fn(a, b, out=out)

        return fun

    return pt_fun


def gen_int_comparison_tensors(N, M):
    # [FUNCTION DOC] HELPER: Creates the tensors for integer comparison ops: two random
    # integer matrices with values in zero to two of shape N by M, and an uninitialized
    # boolean matrix for the result. Returns them as a tuple.
    return (
        torch.randint(0, 3, (N, M)),
        torch.randint(0, 3, (N, M)),
        torch.empty((N, M), dtype=torch.bool),
    )


def gen_float_comparison_tensors(N, M):
    # [FUNCTION DOC] HELPER: Creates the tensors for float comparison ops: two uniform random
    # float matrices of shape N by M and an uninitialized boolean result matrix. Returns them
    # as a tuple.
    return (torch.rand(N, M), torch.rand(N, M), torch.empty((N, M), dtype=torch.bool))


te_bool = te.Dtype.Bool
binary_ops = [
    ("add", operator.add, torch.add),
    ("mul", operator.mul, torch.mul),
    ("sub", operator.sub, torch.sub),
    ("div", operator.truediv, torch.div),
    (
        "eq",
        (lambda a, b: te.Cast.make(te_bool, a == b)),
        torch.eq,
        gen_int_comparison_tensors,
    ),
    (
        "gt",
        (lambda a, b: te.Cast.make(te_bool, a > b)),
        torch.gt,
        gen_float_comparison_tensors,
    ),
    (
        "lt",
        (lambda a, b: te.Cast.make(te_bool, a < b)),
        torch.lt,
        gen_float_comparison_tensors,
    ),
    (
        "gte",
        (lambda a, b: te.Cast.make(te_bool, a >= b)),
        torch.greater_equal,
        gen_float_comparison_tensors,
    ),
    (
        "lte",
        (lambda a, b: te.Cast.make(te_bool, a <= b)),
        torch.less_equal,
        gen_float_comparison_tensors,
    ),
    # ('neq', (lambda a, b: a != b), None)), # no one-op equivalent
    # ('&', (lambda a, b: a & b), torch.bitwise_and), # requires more work to test
]


def nnc_relu(A, B):
    # [FUNCTION DOC] HELPER: NNC compute-function factory for relu: returns a per-element
    # function that selects zero where the loaded value is below zero and the value otherwise,
    # using the if-then-else expression. The second placeholder is ignored.
    def f(i, j):
        # [FUNCTION DOC] HELPER: Per-element body: loads the first placeholder at the row and
        # column index and returns the conditional relu expression for it.
        return torch._C._te.ifThenElse(
            A.load([i, j]) < torch._C._te.ExprHandle.float(0),
            torch._C._te.ExprHandle.float(0),
            A.load([i, j]),
        )

    return f


def pt_relu(a, b, c):
    # [FUNCTION DOC] HELPER: Eager-side relu used for the custom op table: applies relu to the
    # first tensor and returns it; the other two arguments are unused.
    return torch.relu(a)


custom_ops = [
    ("relu", nnc_relu, pt_relu),
    # ('nnc_mul_relu', nnc_mul_relu, pt_mul_relu)
    # ('manual_sigmoid', nnc_manual_sigmoid, lambda a, b, c: torch.sigmoid(a, out=c))
]


def gen_custom_torch_fun(fn):
    # [FUNCTION DOC] HELPER: Wraps a custom torch function that takes three tensors into the
    # same factory shape as the other generators: the returned function binds the tensors and
    # yields a zero-argument callable.
    def pt_fun(a, b, out):
        # [FUNCTION DOC] HELPER: Binds the two inputs and the output tensor and returns a
        # zero-argument callable.
        def fun():
            # [FUNCTION DOC] HELPER: Calls the captured custom function with the three bound
            # tensors and returns its result.
            return fn(a, b, out)

        return fun

    return pt_fun


def normalize_benchmarks(ops):
    # [FUNCTION DOC] HELPER: Pads benchmark table entries to a uniform shape: any three-field
    # entry gets a trailing missing shape generator, while four-field entries are kept.
    # Returns a new list.
    return [i + (None,) if len(i) == 3 else i for i in ops]


names = []
nnc_fns = []
pt_fns = []
shape_fns = []

for nnc_name, pt_op in unary_ops:
    names.append(nnc_name)
    nnc_fns.append(gen_unary_nnc_fun(nnc_name))
    pt_fns.append(gen_unary_torch_fun(pt_op))
    shape_fns.append(None)

for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(binary_ops):
    names.append(name)
    nnc_fns.append(gen_binary_nnc_fun(lmbda))
    pt_fns.append(gen_binary_torch_fun(pt_fn))
    shape_fns.append(shape_fn)

for name, lmbda, pt_fn, shape_fn in normalize_benchmarks(custom_ops):
    names.append(name)
    nnc_fns.append(lmbda)
    pt_fns.append(gen_custom_torch_fun(pt_fn))
    shape_fns.append(shape_fn)

benchmarks = list(zip(names, nnc_fns, pt_fns, shape_fns))


def run_benchmarks(benchmarks, sizes):
    # [FUNCTION DOC] Runs every benchmark over every size pair with gradients disabled and
    # returns a pandas table of results. For each case it makes input tensors (random, clamped
    # away from the edges, or from the shape generator), defines the NNC kernel via a tensor
    # expression compute over the matrix, lowers it with the LLVM codegen, times a size-
    # dependent number of calls after a warmup of ten, times the eager function the same way,
    # and appends one row with both times and their ratio. The iteration count scales
    # inversely with size. Prints progress and verifies that the NNC and eager outputs are
    # close, raising an AssertionError otherwise.
    # KNOWN ISSUE: the results are accumulated with the frame append method, which was removed
    # in recent pandas versions, so this raises an error there.
    df = pd.DataFrame(columns=["name", "N", "M", "nnc_time", "torch_time", "ratio"])
    with torch.no_grad():
        for name, nnc_fun, torch_fun, shape_fn in benchmarks:
            for N, M in sizes:
                iters = int(1e6 / (N + M))
                with kernel_arena_scope():
                    if shape_fn is None:
                        tA = torch.rand(M, N).clamp(0.01, 0.99)
                        tB = torch.rand(M, N).clamp(0.01, 0.99)
                        tX = torch.empty(M, N)
                        tR = torch.empty(M, N)
                    else:
                        tA, tB, tX = shape_fn(M, N)
                        tR = tX.clone()

                    def get_nnc_type(dtype):
                        # [FUNCTION DOC] HELPER: Maps a torch dtype to the matching tensor
                        # expression dtype for float and long; returns nothing (None) for any
                        # other dtype.
                        if dtype == torch.float:
                            return torch._C._te.Dtype.Float
                        elif dtype == torch.long:
                            return torch._C._te.Dtype.Long

                    dtype = get_nnc_type(tA.dtype)

                    dM = torch._C._te.ExprHandle.int(M)
                    dN = torch._C._te.ExprHandle.int(N)

                    A = torch._C._te.Placeholder("A", dtype, [dM, dN])
                    B = torch._C._te.Placeholder("B", dtype, [dM, dN])

                    dim_args = [
                        torch._C._te.DimArg(*args) for args in [(dM, "m"), (dN, "n")]
                    ]

                    compute = nnc_fun(A, B)
                    X = torch._C._te.Compute("X", dim_args, compute)
                    loopnest = torch._C._te.LoopNest([X])
                    loopnest.prepare_for_codegen()
                    stmt = torch._C._te.simplify(loopnest.root_stmt())
                    cg = torch._C._te.construct_codegen(
                        "llvm", stmt, [torch._C._te.BufferArg(x) for x in [A, B, X]]
                    )

                    # warmup
                    for _ in range(10):
                        cg.call([tA, tB, tX])
                    start = time.time()
                    for it in range(iters):
                        cg.call([tA, tB, tX])
                    time1 = time.time() - start

                    fn = torch_fun(tA, tB, tR)
                    # warmup
                    for _ in range(10):
                        tR = fn()
                    start = time.time()
                    for it in range(iters):
                        tR = fn()
                    time2 = time.time() - start

                    df = df.append(
                        {
                            "name": name,
                            "N": N,
                            "M": M,
                            "nnc_time": time1,
                            "torch_time": time2,
                            "ratio": time2 / time1,
                        },
                        ignore_index=True,
                    )
                    print(name, N, M)

                    print(time2 / time1, time1, time2)
                    print()

                    def check_correctness(a, b):
                        # [FUNCTION DOC] HELPER: Compares the two arrays with a tolerance
                        # check; if they differ it prints the op name and raises an
                        # AssertionError naming it.
                        if not np.allclose(a, b):
                            print(name)
                            raise AssertionError(f"Arrays not close for {name}")

                    check_correctness(tX, tR)
    return df


def dump_plot(df, sizes):
    # [FUNCTION DOC] Builds and saves a heatmap of the PyTorch to NNC time ratios. It keeps
    # only rows where the two sizes are equal, collects op names and ratios, reshapes the
    # ratios to one row per op and one column per size, and writes the image file nnc.png in
    # the current directory. Takes the result table and the size list; returns nothing.
    keys = []
    vals = []
    indexed = df[df["N"] == df["M"]]
    for index, row in indexed.iterrows():
        keys.append(row["name"])
        vals.append(row["ratio"])

    keys = keys[:: len(sizes)]
    sns.set(rc={"figure.figsize": (5.0, len(keys) * 0.5)})

    cmap = sns.diverging_palette(10, 120, n=9, as_cmap=True)
    np_vals = np.array([vals]).reshape(-1, len(sizes))
    g = sns.heatmap(np_vals, annot=True, cmap=cmap, center=1.0, yticklabels=True)
    plt.yticks(rotation=0)
    plt.title("PyTorch performance divided by NNC performance (single core)")
    plt.xlabel("Size of NxN matrix")
    plt.ylabel("Operation")
    g.set_yticklabels(keys)
    g.set_xticklabels(sizes)

    plt.savefig("nnc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs NNC microbenchmarks")
    parser.add_argument(
        "--multi-threaded",
        "--multi_threaded",
        action="store_true",
        help="Run with more than one thread",
    )
    args = parser.parse_args()
    if not args.multi_threaded:
        torch.set_num_threads(1)

    sizes = [1, 4, 16, 64, 256, 1024]
    df = run_benchmarks(benchmarks, [(i, i) for i in sizes])
    dump_plot(df, sizes)
