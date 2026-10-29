# mypy: allow-untyped-defs
# [MODULE DOC]
# Runnable example of the Timer and Compare APIs for sparse matrix multiplication. Builds timers
# over fake branches (emulated with a wrapper adding artificial sleep), densities, sizes and
# thread counts, serializes the measurements, and prints raw and formatted comparison tables.
# No import-time side effects; running as a script calls main.
"""Example of Timer and Compare APIs:

$ python -m examples.sparse.compare
"""

import pickle
import sys
import time

import torch
import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    # [CLASS DOC] HELPER: Wraps the real torch module and exposes a sparse namespace whose
    # matrix multiply sleeps in proportion to the result size, faking a slower library version.
    # State: the wrapped module and the extra nanoseconds per element.
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element) -> None:
        # [FUNCTION DOC] HELPER: Stores the real torch module and the per-element delay.
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    @property
    def sparse(self):
        # [FUNCTION DOC] HELPER: Returns a new fake sparse namespace sharing this wrapper's
        # module and delay setting.
        return self.Sparse(self._real_torch, self._extra_ns_per_element)

    class Sparse:
        # [CLASS DOC] HELPER: Fake sparse namespace; mm calls the real sparse matrix multiply and
        # then sleeps based on the result size. State: wrapped module and per-element delay.
        def __init__(self, real_torch, extra_ns_per_element) -> None:
            # [FUNCTION DOC] HELPER: Stores the real torch module and the per-element delay.
            self._real_torch = real_torch
            self._extra_ns_per_element = extra_ns_per_element

        def extra_overhead(self, result):
            # [FUNCTION DOC] HELPER: Sleeps for a time proportional to the sum of the result's
            # dimension sizes when that sum exceeds 5000, then returns the result unchanged.
            # KNOWN ISSUE: the sum of dimension lengths is used as the element count, which is
            # not the number of elements (the comment calls it numel).
            # time.sleep has a ~65 us overhead, so only fake a
            # per-element overhead if numel is large enough.
            size = sum(result.size())
            if size > 5000:
                time.sleep(size * self._extra_ns_per_element * 1e-9)
            return result

        def mm(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Real sparse matrix multiply followed by the artificial
            # delay; returns the product.
            return self.extra_overhead(self._real_torch.sparse.mm(*args, **kwargs))

def generate_coo_data(size, sparse_dim, nnz, dtype, device):
    # [FUNCTION DOC] HELPER: Generates random coordinate-format sparse data: an integer index
    # matrix of shape sparse_dim by nnz with each row scaled to the matching size entry, and a
    # random value vector of length nnz. Returns the pair (indices, values).
    # KNOWN ISSUE: when dtype is none it is replaced by a string name, which is not a valid
    # dtype for the random value generator.
    """
    Parameters
    ----------
    size : tuple
    sparse_dim : int
    nnz : int
    dtype : torch.dtype
    device : str
    Returns
    -------
    indices : torch.tensor
    values : torch.tensor
    """
    if dtype is None:
        dtype = 'float32'

    indices = torch.rand(sparse_dim, nnz, device=device)
    indices.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(indices))
    indices = indices.to(torch.long)

    values = torch.rand([nnz, ], dtype=dtype, device=device)
    return indices, values

def gen_sparse(size, density, dtype, device='cpu'):
    # [FUNCTION DOC] HELPER: Builds a 2-D sparse coordinate tensor of the given size with about
    # the given fraction of nonzero entries (nnz is size rows times columns times density,
    # truncated). Duplicate coordinates are possible. Returns the tensor.
    sparse_dim = len(size)
    nnz = int(size[0] * size[1] * density)
    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype, device)
    return torch.sparse_coo_tensor(indices, values, size, dtype=dtype, device=device)

def main() -> None:
    # [FUNCTION DOC] Builds timers for sparse matrix multiplication across fake branches, tasks,
    # densities, sizes and thread counts, runs each twice, serializes and deserializes the
    # measurements, and prints the Compare table raw and formatted. Prints to standard output;
    # returns nothing.
    tasks = [
        ("matmul", "x @ y", "torch.sparse.mm(x, y)"),
        ("matmul", "x @ y + 0", "torch.sparse.mm(x, y) + zero"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": gen_sparse(size=size, density=density, dtype=torch.float32),
                "y": torch.rand(size, dtype=torch.float32),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 10)]
        for label, sub_label, stmt in tasks
        for density in [0.05, 0.1]
        for size in [(8, 8), (32, 32), (64, 64), (128, 128)]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()
