#!/usr/bin/env python3
# [MODULE DOC] Benchmark comparing the standard batch sampler against an alternative
# implementation over a sequential sampler of a fixed size, for several batch sizes with and
# without dropping the last partial batch. Prints the average time of each, the relative
# speedup, and a final grid table. Needs numpy and tabulate. No import-time side effects.

import time
from collections.abc import Iterable, Iterator

import numpy as np
from tabulate import tabulate

from torch.utils.data import BatchSampler, Sampler, SequentialSampler


class NewBatchSampler(Sampler[list[int]]):
    # [CLASS DOC] Alternative batch sampler used as the candidate in the benchmark. Stores
    # the underlying sampler, a positive integer batch size and a boolean drop-last flag
    # (validated at construction). Iteration yields lists of indices of the batch size;
    # length is the number of batches, defined only if the wrapped sampler has a length.
    """Alternative implementation of BatchSampler for benchmarking purposes."""

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # [FUNCTION DOC] Validate and store the wrapped sampler, batch size and drop-last flag.
        # Raises a value error if the batch size is not a positive integer (booleans are
        # rejected) or the drop-last flag is not a boolean.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        # [FUNCTION DOC] Generator over batches of indices. With drop-last it pulls exactly the
        # batch size items at a time and stops at the first incomplete batch, discarding it.
        # Otherwise it fills a preallocated list and yields it when full, allocating a new
        # one afterwards, and finally yields the leftover partial batch as a shorter list.
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of batches: the floor division of the sampler length
        # by the batch size when dropping the last batch, else the ceiling division. Fails if
        # the wrapped sampler has no length.
        # Can only be called if self.sampler has __len__ implemented
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def main():
    # [FUNCTION DOC] Run the benchmark. For each batch size and drop-last option, times full
    # iteration over the baseline sampler and the new sampler ten times each (sleeping briefly
    # between runs), averages the times, computes the percentage speedup of the new one
    # (not available if an average is zero), prints it, and finally prints all results as a
    # grid table. Prints output only; returns None.
    """Run benchmark with specified parameters."""
    DATA_SIZE = 99999
    AVG_TIMES = 10
    BATCH_SIZES = [4, 8, 64, 640, 6400, 64000]
    DROP_LAST_OPTIONS = [True, False]

    results = []

    # Set up samplers here, ensure right args are passed in
    baselineSampler = BatchSampler
    testSampler = NewBatchSampler

    for batch_size in BATCH_SIZES:
        for drop_last in DROP_LAST_OPTIONS:
            print(f"Benchmarking with batch_size={batch_size}, drop_last={drop_last}")

            # Benchmark baselineSampler
            original_times = []
            for _ in range(AVG_TIMES):
                start = time.perf_counter()
                for _ in baselineSampler(
                    sampler=SequentialSampler(range(DATA_SIZE)),
                    batch_size=batch_size,
                    drop_last=drop_last,
                ):
                    pass
                end = time.perf_counter()
                original_times.append(end - start)
                time.sleep(0.1)

            original_avg = float(np.mean(original_times))

            # Benchmark testSampler
            new_times = []
            for _ in range(AVG_TIMES):
                start = time.perf_counter()
                for _ in testSampler(
                    sampler=SequentialSampler(range(DATA_SIZE)),
                    batch_size=batch_size,
                    drop_last=drop_last,
                ):
                    pass
                end = time.perf_counter()
                new_times.append(end - start)
                time.sleep(0.1)  # Small delay to reduce system load

            new_avg = float(np.mean(new_times))

            # Calculate speedup
            if original_avg > 0 and new_avg > 0:
                speedup = (original_avg - new_avg) / original_avg * 100
                speedup_str = f"{speedup:.2f}%"
            else:
                speedup_str = "N/A"

            print(f"Speedup: {speedup_str}\n")

            results.append(
                [
                    batch_size,
                    drop_last,
                    f"{original_avg:.4f}",
                    f"{new_avg:.4f}",
                    speedup_str,
                ]
            )

    # Print results in a table
    headers = ["Batch Size", "Drop Last", "Original (s)", "New (s)", "Speedup"]
    print("\nBenchmark Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
