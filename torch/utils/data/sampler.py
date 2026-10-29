# mypy: allow-untyped-defs
# [MODULE DOC]
# Samplers that decide the order of dataset indices for a data loader: the Sampler base
# class, SequentialSampler, RandomSampler, SubsetRandomSampler, WeightedRandomSampler and
# BatchSampler (groups indices into lists). No import-time side effects.
import itertools
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Generic, TypeVar

import torch


# Note: For benchmarking changes to samplers, see:
# /benchmarks/data/samplers_bench.py
# This benchmark compares the performance of different sampler implementations
# and can be used to evaluate the impact of optimizations.


__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]


_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    # [CLASS DOC]
    # Abstract base for samplers: subclasses implement __iter__ to yield dataset indices (or
    # lists of indices) and may implement __len__. Deliberately has no default __len__ (see
    # the note below). Holds no state.
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC] Abstract: always raises NotImplementedError; subclasses return an
        # iterator over indices.
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising a `NotImplementedError` will propagate and make the call fail
    #     where it could have used `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler[int]):
    # [CLASS DOC]
    # Sampler yielding 0 up to len(data_source)-1 in order. State: the sized data source;
    # length is evaluated on use, so it follows later changes of the source.
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        # [FUNCTION DOC] Stores the data source.
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # [FUNCTION DOC] Returns an iterator over the integers from zero to the source length
        # minus one.
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        # [FUNCTION DOC] Length of the data source.
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    # [CLASS DOC]
    # Sampler yielding random indices in the range of the data source, either shuffled passes
    # without replacement or independent draws with replacement. State: data source,
    # replacement flag, optional requested sample count (default is the source length,
    # re-read each time) and optional generator. Each iteration draws fresh randomness.
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        generator=None,
    ) -> None:
        # [FUNCTION DOC] Stores the options. Raises TypeError if replacement is not a bool and
        # ValueError if the sample count is not a positive integer (this reads the length of
        # the source when no count was given).
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # [FUNCTION DOC] Property: the requested sample count, or the current source length
        # if none was requested.
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # [FUNCTION DOC] Generator of num_samples indices. Without a generator, seeds a new one
        # from the global torch RNG (consuming it). With replacement, draws random integers in
        # blocks of 32 plus a remainder block. Without replacement, yields whole random
        # permutations, then a prefix of one more for the remainder.
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    def __len__(self) -> int:
        # [FUNCTION DOC] The number of samples that iteration yields.
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    # [CLASS DOC]
    # Sampler yielding the entries of a given index sequence in random order, each once per
    # pass. State: the index sequence and an optional generator.
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        # [FUNCTION DOC] Stores the index sequence and the optional generator.
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # [FUNCTION DOC] Draws a random permutation of positions (using the generator if set,
        # else the global RNG) and yields the index sequence entries in that order.
        for i in torch.randperm(len(self.indices), generator=self.generator).tolist():
            yield self.indices[i]

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of indices in the sequence.
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    # [CLASS DOC]
    # Sampler drawing indices into a weight list with probability proportional to the weights
    # (which need not sum to one), using multinomial sampling. State: weights as a 1-D double
    # tensor, number of samples, replacement flag and optional generator.
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(
        ...     WeightedRandomSampler(
        ...         [0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True
        ...     )
        ... )
        [4, 4, 1, 4, 5]
        >>> list(
        ...     WeightedRandomSampler(
        ...         [0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False
        ...     )
        ... )
        [0, 1, 4, 3, 2]
    """

    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        # [FUNCTION DOC] Validates and stores the options; converts the weights to a double
        # tensor. Raises ValueError if num_samples is not a positive non-bool integer, if
        # replacement is not a bool, or if the weights are not 1-D.
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # [FUNCTION DOC] Draws num_samples indices in one multinomial call (with or without
        # replacement, using the generator if set) and yields them as integers. Errors from
        # the multinomial call (for example too many samples without replacement) propagate.
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        # [FUNCTION DOC] The configured number of samples.
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    # [CLASS DOC]
    # Sampler that groups the indices of another sampler or iterable into lists of batch_size.
    # State: the wrapped sampler, batch size and drop_last. A shorter final batch is yielded
    # unless drop_last is set. Length needs the wrapped sampler to have one.
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(
        ...     BatchSampler(
        ...         SequentialSampler(range(10)), batch_size=3, drop_last=False
        ...     )
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(
        ...     BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the sampler, batch size and drop_last flag. Raises ValueError if
        # batch_size is not a positive non-bool integer or drop_last is not a bool.
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
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
        # [FUNCTION DOC] Pulls indices from one iterator over the wrapped sampler and yields
        # lists of batch_size. With drop_last, stops at the first incomplete batch (using
        # grouped pulls from the same iterator); otherwise yields the smaller last batch.
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args, strict=False):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of batches: floor of wrapped length over batch_size with
        # drop_last, otherwise the ceiling. Raises TypeError if the wrapped sampler has no
        # length.
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
