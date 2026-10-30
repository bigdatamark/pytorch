# [MODULE DOC] PR-time benchmark that measures torch.export instruction counts on a model built
# from repeated unbind, item extraction, split, flip and cat steps. Defines the model and a
# Benchmark class (derived from BenchmarkBase); main appends results to a file given on the
# command line. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.export
from torch.fx.experimental._config import AggressiveGuardFreeMode


class UnbindSplitFlipCatModel(torch.nn.Module):
    # [CLASS DOC] Module whose forward runs a fixed number of iterations over paired lists of
    # size tensors and data tensors. Stores num_iterations and batch_size; has no parameters.
    """Model that performs unbind, split_with_sizes, flip, and cat operations."""

    def __init__(self, num_iterations: int = 40, batch_size: int = 128):
        # [FUNCTION DOC] Stores the iteration count and per-iteration batch size on the module.
        super().__init__()
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def forward(self, sizes_tensors: list, data_tensors: list):
        # [FUNCTION DOC] For each iteration, converts that iteration's sizes tensor to a 64-bit
        # integer tensor, unbinds it, and calls item on each of batch_size entries to get data-
        # dependent integers, asserting each is non-negative with torch._check. The integers are
        # also summed into a total that is never used. The data tensor is split by those
        # integers, every piece is flipped along dimension zero, and the pieces are
        # concatenated. Returns the list of concatenated tensors, one per iteration.
        results = []

        for i in range(self.num_iterations):
            sizes_tensor = sizes_tensors[i]
            data_tensor = data_tensors[i]

            to_result = sizes_tensor.to(torch.int64)
            unbind_result = torch.unbind(to_result)

            items = []
            for j in range(self.batch_size):
                getitem = unbind_result[j]
                item = getitem.item()
                torch._check(item >= 0)
                items.append(item)

            total = items[0]
            for j in range(1, self.batch_size):
                total = total + items[j]

            split_result = torch.split(data_tensor, items)

            flipped = []
            for j in range(self.batch_size):
                getitem = split_result[j]
                flip_result = torch.flip(getitem, [0])
                flipped.append(flip_result)

            cat_result = torch.cat(flipped)
            results.append(cat_result)

        return results


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark that exports the unbind/split/flip/cat model with the export backend
    # on cpu, in the category unbind_split_flip_cat. Holds num_iterations and batch_size, and
    # after _prepare_once also the model and the sizes_list and data_list inputs.
    """Benchmark for unbind/split/flip/cat pattern with torch.export."""

    def __init__(self, num_iterations: int = 40, batch_size: int = 128):
        # [FUNCTION DOC] Records num_iterations and batch_size, then initializes the base class
        # with the category, the export backend and the cpu device.
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        super().__init__(
            category="unbind_split_flip_cat",
            backend="export",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the benchmark name built from the category plus the iteration
        # and batch-size settings.
        return f"{self.category()}_iter{self.num_iterations}_batch{self.batch_size}"

    def description(self):
        # [FUNCTION DOC] Returns a fixed one-line description of the benchmark.
        return "Benchmark unbind/split/flip/cat pattern with torch.export"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup. Enables scalar output capture in dynamo, selects the
        # guard-free mode that skips range analysis, disables stack trace emission in fx (all
        # global config mutations), and seeds the random generator. Builds the model, then
        # builds num_iterations sizes tensors of random integers from 1 to 9 of length
        # batch_size, each paired with an integer arange data tensor whose length equals the sum
        # of that sizes tensor.
        torch._dynamo.config.capture_scalar_outputs = True
        torch.fx.experimental._config.aggressive_guard_free_semantics = (
            AggressiveGuardFreeMode.SKIP_RANGE_ANALYSIS
        )
        torch.fx.config.do_not_emit_stack_traces = True
        torch.manual_seed(0)

        self.model = UnbindSplitFlipCatModel(
            num_iterations=self.num_iterations,
            batch_size=self.batch_size,
        )

        self.sizes_list = []
        self.data_list = []

        for _ in range(self.num_iterations):
            sizes = torch.randint(1, 10, (self.batch_size,), dtype=torch.int64)
            total_size = int(sizes.sum().item())
            data = torch.arange(total_size, dtype=torch.int64)
            self.sizes_list.append(sizes)
            self.data_list.append(data)

    def _prepare(self):
        # [FUNCTION DOC] Per-measurement setup: resets dynamo state so each run starts from
        # scratch.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] The measured work: exports the model with the prepared sizes and data
        # lists as the single example input. The result is discarded.
        torch.export.export(
            self.model,
            (self.sizes_list, self.data_list),
        )


def main():
    # [FUNCTION DOC] Entry point. Takes the output path from the first command line argument,
    # runs the benchmark with instruction counting enabled across all collection steps, and
    # appends the results to that path.
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
