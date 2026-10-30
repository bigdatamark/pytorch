# Owner(s): ["module: dataloader"]
# ruff: noqa: F841
# [MODULE DOC] Tests for torch.utils.data: dataset wrappers (random_split, Subset, TensorDataset,
# StackDataset, ConcatDataset), samplers, DataLoader single and multi process behavior, worker
# error handling, pin_memory, seeding and datapipe integration. Import time side effects: forces
# the spawn multiprocessing context for module level mp, enables faulthandler (with a SIGUSR1
# trace handler off Windows), raises if psutil is missing on CI, and computes feature flags such
# as HAS_NUMPY, TEST_CUDA_IPC and TEST_PIN_MEMORY.

import ctypes
import errno
import faulthandler
import functools
import gc
import itertools
import math
import operator
import os
import signal
import sys
import tempfile
import time
import unittest
import warnings
from unittest import mock

import torch
import torch.utils.data.datapipes as dp
from torch import multiprocessing as mp
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_JETSON,
    IS_LINUX,
    IS_MACOS,
    IS_S390X,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    parametrize,
    run_tests,
    skipIfNoDill,
    skipIfRocm,
    skipIfXpu,
    slowTest,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)
from torch.utils.data import (
    _utils,
    ChainDataset,
    ConcatDataset,
    DataLoader,
    dataloader,
    Dataset,
    IterableDataset,
    IterDataPipe,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.dataset import random_split


try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None
    err_msg = (
        "psutil not found. Some critical data loader tests relying on it "
        "(e.g., TestDataLoader.test_proper_exit) will not run."
    )
    if IS_CI:
        raise ModuleNotFoundError(err_msg) from None
    else:
        warnings.warn(err_msg)


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

TEST_CUDA_IPC = (
    torch.cuda.is_available()
    and sys.platform != "darwin"
    and sys.platform != "win32"
    and not IS_JETSON
    #    and not TEST_WITH_ROCM
)  # https://github.com/pytorch/pytorch/issues/90940

# pin_memory requires an accelerator with a pinned memory allocator.
# MPS reports as available but does not support pin_memory (see
# https://github.com/pytorch/pytorch/issues/86060), so exclude it.
TEST_PIN_MEMORY = torch.accelerator.is_available() and not (
    (acc := torch.accelerator.current_accelerator()) is not None and acc.type == "mps"
)

# We want to use `spawn` if able because some of our tests check that the
# data loader terminates gracefully. To prevent hanging in the testing
# process, such data loaders are run in a separate subprocess.
#
# We also want to test the `pin_memory=True` configuration, thus `spawn` is
# required to launch such processes and they initialize the accelerator context.
#
# Mixing different start method is a recipe for disaster (e.g., using a fork
# `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
# to avoid bugs.
#
# Get a multiprocessing context because some test / third party library will
# set start_method when imported, and setting again triggers `RuntimeError`.
mp = mp.get_context(method="spawn")


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


supported_multiprocessing_contexts = [None] + list(
    torch.multiprocessing.get_all_start_methods()
)


# The following collate functions are defined globally here for pickle purposes.


# collate_fn that returns the batch cloned
def _clone_collate(b):
    # [FUNCTION DOC] HELPER: collate_fn used as a picklable global; returns a list holding a clone
    # of each element of the batch, so results do not share storage with the dataset.
    return [x.clone() for x in b]


# collate_fn that returns the batch of sparse coo tensors cloned
def _sparse_coo_collate(b):
    # [FUNCTION DOC] HELPER: collate_fn for sparse COO tensors; clones each element, validates its
    # sparse tensor invariants (indices, values, size, coalesced flag, pinning check disabled) and
    # returns the list of clones. Raises if a tensor violates the invariants.
    lst = []
    for x in b:
        t = x.clone()
        lst.append(t)
        # Force sparse tensor invariants checks. check_pinning=True
        # reproduces gh-153143.
        torch._validate_sparse_coo_tensor_args(
            t._indices(), t._values(), t.size(), t.is_coalesced(), check_pinning=False
        )
    return lst


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDatasetRandomSplit(TestCase):
    # [CLASS DOC] Tests for random_split: length validation, sizes, disjointness, integer indices,
    # reproducibility with a seeded generator, generator side effects and slicing of subsets.
    def test_lengths_must_equal_dataset_size(self):
        # [FUNCTION DOC] Checks random_split raises ValueError when lengths do not sum to dataset
        # size.
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        # [FUNCTION DOC] Checks split sizes for integer lengths, fractional lengths, odd sizes and
        # the round-robin distribution of remainders for four fractions over 106 items.
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

        splits = random_split([1, 2, 3, 4, 5, 6], [0.5, 0.5])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 3)
        self.assertEqual(len(splits[1]), 3)

        # Odd size splits
        self.assertEqual(
            len(
                random_split(
                    range(3), [0.5, 0.5], generator=torch.Generator().manual_seed(1)
                )
            ),
            2,
        )

        # Odd sized round-robin splits
        splits = random_split(
            range(106), [0.1, 0.2, 0.3, 0.4], generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(len(splits[0]), 11)
        self.assertEqual(len(splits[1]), 22)
        self.assertEqual(len(splits[2]), 31)
        self.assertEqual(len(splits[3]), 42)

    def test_splits_are_mutually_exclusive(self):
        # [FUNCTION DOC] Checks splits are disjoint and together cover the data, for integer
        # lengths and for fractions, by sorting the concatenated split values and comparing to the
        # sorted data.
        data = [5, 2, 3, 4, 1, 6]
        splits = random_split(data, [2, 4])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        splits = random_split(data, [0.33, 0.67])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        data = [1, 2, 3, 4]
        splits = random_split(data, [0.25, 0.75])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

    def test_splits_indexing_type(self):
        # [FUNCTION DOC] Checks indices random_split passes to the underlying dataset are plain
        # int, for integer and fractional lengths, by iterating a DataLoader over the split.
        r"""Indices generated by random_split
        should be of integer type
        """

        class CustomDataset:
        # [CLASS DOC] HELPER: map style dataset over a list that asserts via the owning test case
        # that every key it receives has type int.
            def __init__(self, test_object, custom_list):
            # [FUNCTION DOC] HELPER: stores the backing list and the test case used for
            # assertions.
                self.data = custom_list
                self.test_object = test_object

            def __getitem__(self, key):
            # [FUNCTION DOC] HELPER: asserts the key type is int, then returns the list element at
            # the key.
                self.test_object.assertEqual(type(key), int)
                return self.data[key]

            def __len__(self):
            # [FUNCTION DOC] HELPER: returns the length of the backing list.
                return len(self.data)

        x = [1, 2, 3, 4, 5]
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [5])[0]
        data_loader = DataLoader(dataset)
        for _batch in data_loader:
            pass

        # fractional splitting
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [1.0])[0]
        data_loader = DataLoader(dataset)
        for _batch in data_loader:
            pass

    def test_splits_reproducibility(self):
        # [FUNCTION DOC] Checks that the same seed gives identical splits (explicit expected lists
        # for one case, equality of two calls for integer, fraction and three way fraction cases).
        self.assertEqual(
            [
                list(x)
                for x in random_split(
                    range(10), [3, 7], generator=torch.Generator().manual_seed(1)
                )
            ],
            [[5, 6, 1], [2, 0, 8, 9, 3, 7, 4]],
        )
        self.assertEqual(
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
        )
        self.assertEqual(
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
        )
        self.assertEqual(
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
        )

    def test_incomplete_fractional_splits(self):
        # [FUNCTION DOC] Checks ValueError is raised when fractions do not sum to one or one
        # exceeds one.
        with self.assertRaises(ValueError):
            # should raise since the sum of fractions is not 1
            random_split([1, 2, 3, 4], [0.1])

        with self.assertRaises(ValueError):
            # should raise since fraction > 1
            random_split([1, 2, 3, 4], [1.1])

    def test_splits_generator(self):
        # [FUNCTION DOC] Checks random_split without a generator consumes the global RNG while a
        # call with an explicit generator leaves the global RNG state untouched.
        # A random_split without a specific generator should affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5])
        b = torch.rand(10)
        self.assertNotEqual(a, b)

        # A random_split with a specific generator should not affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5], generator=torch.Generator().manual_seed(42))
        b = torch.rand(10)
        self.assertEqual(a, b)

    def test_slicing_of_subset_of_dataset(self):
        # [FUNCTION DOC] Checks slicing a Subset (and Subset from random_split) of a dataset
        # matches indexing the dataset with the corresponding index slice.
        # Testing slicing a subset initialized with a dataset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_dataset[:], dataset[:])
        self.assertEqual(subset_of_dataset[1:2], dataset[1:2])
        self.assertEqual(subset_of_dataset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset from random split
        subset1, subset2 = random_split(dataset, [3, 2])
        self.assertEqual(subset1[:], dataset[subset1.indices[:]])
        self.assertEqual(subset1[0:2], dataset[subset1.indices[0:2]])
        self.assertEqual(subset1[0:-1:2], dataset[subset1.indices[0:-1:2]])

    def test_slicing_of_subset_of_subset(self):
        # [FUNCTION DOC] Checks slicing a Subset of a Subset (also from nested random_split)
        # resolves through both index lists and matches indexing the base dataset.
        # Testing slicing a subset initialized with a subset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        subset_of_subset = Subset(subset_of_dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_subset[:], dataset[:])
        self.assertEqual(subset_of_subset[0:2], dataset[0:2])
        self.assertEqual(subset_of_subset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset of subset from random split
        subset1, subset2 = random_split(dataset, [4, 1])
        subset_of_subset1, subset_of_subset2 = random_split(subset1, [3, 1])
        idx = [subset1.indices[i] for i in subset_of_subset1.indices]
        self.assertEqual(subset_of_subset1[:], dataset[idx.copy()])
        self.assertEqual(subset_of_subset1[0:2], dataset[idx[0:2]])
        self.assertEqual(subset_of_subset1[0:-1:2], dataset[idx[0:-1:2]])


class CUDACountingDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset of a given length whose items are CUDA scalar tensors
# holding the index.
    def __init__(self, n):
    # [FUNCTION DOC] HELPER: stores the dataset length.
        super().__init__()
        self.n = n

    def __getitem__(self, i):
    # [FUNCTION DOC] HELPER: returns the index as a tensor created on the cuda device.
        return torch.as_tensor(i, device="cuda")

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored length.
        return self.n


class CountingDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset of a given length whose item at an index is the index.
    def __init__(self, n):
    # [FUNCTION DOC] HELPER: stores the dataset length.
        super().__init__()
        self.n = n

    def __getitem__(self, i):
    # [FUNCTION DOC] HELPER: returns the index unchanged.
        return i

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored length.
        return self.n


class CountingIterableDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset that yields the integers from zero up to a given count and
# reports that count as its length.
    def __init__(self, n):
    # [FUNCTION DOC] HELPER: stores the count.
        super().__init__()
        self.n = n

    def __iter__(self):
    # [FUNCTION DOC] HELPER: returns a fresh iterator over the integer range.
        return iter(range(self.n))

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored count.
        return self.n


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestTensorDataset(TestCase):
    # [CLASS DOC] Tests for TensorDataset length, indexing and the size mismatch check.
    def test_len(self):
        # [FUNCTION DOC] Checks the length equals the first dimension size shared by the tensors.
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        # [FUNCTION DOC] Checks item i is a tuple holding row i of each tensor (multi dimensional
        # data).
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        # [FUNCTION DOC] Checks the same row-wise indexing with one dimensional tensors.
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_single_tensor(self):
        # [FUNCTION DOC] Checks a dataset built from a single tensor yields one element tuples.
        t = torch.randn(5, 10)
        source = TensorDataset(t)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t[i], source[i][0])

    def test_many_tensors(self):
        # [FUNCTION DOC] Checks a dataset built from four tensors of different shapes indexes all
        # of them.
        t0 = torch.randn(5, 10, 2, 3, 4, 5)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10, 2, 5)
        t3 = torch.randn(5, 10, 3, 7)
        source = TensorDataset(t0, t1, t2, t3)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t0[i], source[i][0])
            self.assertEqual(t1[i], source[i][1])
            self.assertEqual(t2[i], source[i][2])
            self.assertEqual(t3[i], source[i][3])

    def test_size_mismatch(self):
        # [FUNCTION DOC] Checks construction raises AssertionError mentioning a size mismatch when
        # the tensors have different first dimension sizes.
        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(10, 3)
        tensor3 = torch.randn(20, 7)
        with self.assertRaisesRegex(AssertionError, "Size mismatch between tensors"):
            TensorDataset(tensor1, tensor2, tensor3)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestStackDataset(TestCase):
    # [CLASS DOC] Tests for StackDataset with positional (tuple output) and keyword (dict output)
    # datasets, including batched getitems behavior and its error cases.
    def test_empty(self):
        # [FUNCTION DOC] Checks constructing with no datasets raises ValueError.
        with self.assertRaisesRegex(
            ValueError, "At least one dataset should be passed"
        ):
            StackDataset()

    def test_mixed(self):
        # [FUNCTION DOC] Checks mixing positional and keyword datasets raises ValueError.
        with self.assertRaisesRegex(ValueError, "Supported either"):
            StackDataset(
                TensorDataset(torch.randn(15, 10)), a=TensorDataset(torch.randn(10, 15))
            )

    def test_size_mismatch(self):
        # [FUNCTION DOC] Checks datasets of different lengths raise ValueError, for positional and
        # keyword forms.
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(
                TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(10, 15))
            )
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(
                a=TensorDataset(torch.randn(15, 10)),
                b=TensorDataset(torch.randn(10, 15)),
            )

    def test_len(self):
        # [FUNCTION DOC] Checks the length equals the common dataset length for positional and
        # keyword construction with one or two datasets.
        source = StackDataset(
            TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(15))
        )
        self.assertEqual(len(source), 15)
        source = StackDataset(TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)
        source = StackDataset(
            a=TensorDataset(torch.randn(15, 10)), b=TensorDataset(torch.randn(15))
        )
        self.assertEqual(len(source), 15)
        source = StackDataset(a=TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)

    def test_single(self):
        # [FUNCTION DOC] Checks a single dataset gives tuple items (positional) or dict items
        # (keyword).
        t = TensorDataset(torch.randn(15, 10))
        source = StackDataset(t)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
        source = StackDataset(a=t)
        for i in range(15):
            self.assertEqual(t[i], source[i]["a"])

    def test_getitem(self):
        # [FUNCTION DOC] Checks item i combines item i of every dataset, as tuple entries or dict
        # values.
        t = TensorDataset(torch.randn(15, 10))
        l = TensorDataset(torch.randn(15, 5, 4))
        source = StackDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])
        source = StackDataset(a=t, b=l)
        for i in range(15):
            self.assertEqual(t[i], source[i]["a"])
            self.assertEqual(l[i], source[i]["b"])

    def test_getitems(self):
        # [FUNCTION DOC] Checks the batched getitems method, mixing a dataset implementing it with
        # a plain list, returns per index tuples (positional) or dicts (keyword).
        class GetItemsDataset(Dataset):
            # [CLASS DOC] HELPER: dataset of four random values supporting both single and batched
            # access.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates four random values.
                self.data = torch.randn(4)

            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: returns the value at the index.
                return self.data[item]

            def __getitems__(self, items):
                # [FUNCTION DOC] HELPER: returns the values at the given indices in one indexing
                # operation.
                return self.data[items]

            def __len__(self):
                # [FUNCTION DOC] HELPER: always reports length four.
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i][0])
            self.assertEqual(l[i], batch[i][1])

        source = StackDataset(t=t, l=l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i]["t"])
            self.assertEqual(l[i], batch[i]["l"])

    def test_getitems_raises_index_error(self):
        # [FUNCTION DOC] Checks the batched getitems method raises IndexError when an index is out
        # of range (index four on a dataset of length four).
        class GetItemsDataset(Dataset):
            # [CLASS DOC] HELPER: dataset of four random values supporting both single and batched
            # access.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates four random values.
                self.data = torch.randn(4)

            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: returns the value at the index.
                return self.data[item]

            def __getitems__(self, items):
                # [FUNCTION DOC] HELPER: returns the values at the given indices in one indexing
                # operation.
                return self.data[items]

            def __len__(self):
                # [FUNCTION DOC] HELPER: always reports length four.
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaises(IndexError):
            source.__getitems__([0, 4])

    def test_getitems_value_error(self):
        # [FUNCTION DOC] Checks batched getitems raises ValueError stating expected 4 got 3 when a
        # nested dataset returns fewer items than requested.
        class GetItemsDataset(Dataset):
            # [CLASS DOC] HELPER: dataset whose batched access deliberately drops the last result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates four random values.
                self.data = torch.randn(4)

            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: returns the value at the index.
                return self.data[item]

            def __getitems__(self, items):
                # [FUNCTION DOC] HELPER: returns the selected values minus the last one, to
                # trigger a size error.
                return self.data[items][:-1]  # return less

            def __len__(self):
                # [FUNCTION DOC] HELPER: always reports length four.
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaisesRegex(
            ValueError, "Nested dataset's output size mismatch. Expected 4, got 3"
        ):
            source.__getitems__([0, 1, 2, 3])


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestConcatDataset(TestCase):
    # [CLASS DOC] Tests for ConcatDataset indexing, the plus operator and rejection of iterables.
    def test_concat_two_singletons(self):
        # [FUNCTION DOC] Checks concatenating two one element lists gives length two with ordered
        # items.
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        # [FUNCTION DOC] Checks concatenating two five element lists gives length ten and cross
        # list indexing.
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # [FUNCTION DOC] Checks an empty dataset between two non empty ones does not affect
        # indexing.
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4], [], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        # [FUNCTION DOC] Checks an out of range index raises IndexError.
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        # [FUNCTION DOC] Checks adding three TensorDatasets with the plus operator gives a
        # concatenation of length 21 whose items come from the right component dataset.
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())

    def test_iterable_dataset_err(self):
        # [FUNCTION DOC] Checks ConcatDataset raises AssertionError (does not support
        # IterableDataset) when any component is an iterable dataset, in first, middle and last
        # positions.
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        it1 = CountingIterableDataset(5)
        it2 = CountingIterableDataset(10)

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([d1, it2, it1])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it2])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it1, d1])


# takes in dummy var so this can also be used as a `worker_init_fn`
def set_faulthander_if_available(_=None):
    # [FUNCTION DOC] HELPER: enables faulthandler on the original stderr and, off Windows,
    # registers SIGUSR1 to dump all thread tracebacks without chaining to the default handler. The
    # ignored argument lets this double as a worker_init_fn.
    faulthandler.enable(sys.__stderr__)
    if not IS_WINDOWS:
        # windows does not have faulthandler.register
        # chain=False prevents the default behavior of killing the process
        faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, chain=False)


set_faulthander_if_available()


# Process `pid` must have called `set_faulthander_if_available`
def print_traces_of_all_threads(pid):
    # [FUNCTION DOC] HELPER: asks the process with the given pid (which must have called the
    # faulthandler setup) to dump its thread traces: SIGUSR1 off Windows, SIGSEGV (which kills it)
    # on Windows. Then sleeps five seconds in the caller to let the output flush.
    if not IS_WINDOWS:
        # use the custom signal if available
        os.kill(pid, signal.SIGUSR1)
    else:
        # otherwise we can still use the handler given by faulthandler.enable()
        # at the cost of killing the process.
        os.kill(pid, signal.SIGSEGV)

    # wait in parent process to give subprocess some time to print
    time.sleep(5)


# The following `ErrorTrackingProcess` stores the first encountered exception in
# its `.exception` attribute.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):
    # [CLASS DOC] HELPER: multiprocessing Process (spawn context) that sends its first exception,
    # or None on success, through a pipe so the parent can retrieve it. Attributes: the pipe ends,
    # the stored exception and disable_stderr, which redirects the child stderr to the null
    # device.
    # Why no *args?
    #   py2 doesn't support def fn(x, *args, key=val, **kwargs)
    # Setting disable_stderr=True may generate a lot of unrelated error outputs
    # but could be helpful for debugging.
    def __init__(self, disable_stderr=True, **kwargs):
        # [FUNCTION DOC] HELPER: forwards keyword arguments to Process, creates the pipe, clears
        # the stored exception and records the disable_stderr flag.
        super().__init__(**kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.disable_stderr = disable_stderr

    def run(self):
        # [FUNCTION DOC] HELPER: child entry point; sets up faulthandler, optionally redirects
        # stderr to the null device, runs the target and sends None, or sends an ExceptionWrapper
        # of the raised exception through the pipe and re-raises it.
        set_faulthander_if_available()
        if self.disable_stderr:
            # Disable polluting stderr with errors that are supposed to happen.
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            super().run()
            self._cconn.send(None)
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    def print_traces_of_all_threads(self):
        # [FUNCTION DOC] HELPER: dumps the child's thread tracebacks. Raises AssertionError if the
        # process is not alive or if stderr was disabled. First reads the exception property (to
        # poll the pipe, since on platforms lacking SIGUSR1 the dump kills the process), then
        # calls the module level print_traces_of_all_threads with the child pid.
        if not self.is_alive():
            raise AssertionError(
                "can only use print_traces_of_all_threads if the process is alive"
            )
        if self.disable_stderr:
            raise AssertionError(
                "do not disable stderr if you use print_traces_of_all_threads"
            )
        # On platforms without `SIGUSR1`, `set_faulthander_if_available` sets
        # `faulthandler.enable()`, and `print_traces_of_all_threads` may kill
        # the process. So let's poll the exception first
        _ = self.exception
        print_traces_of_all_threads(self.pid)

    @property
    def exception(self):
        # [FUNCTION DOC] HELPER: property returning the exception raised in the child, or None.
        # Polls the pipe and caches any received ExceptionWrapper, then rebuilds a new exception
        # instance of the recorded type from the recorded message text (the original traceback is
        # not preserved).
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)

    # ESRCH means that os.kill can't finds alive proc
    def send_signal(self, signum, ignore_ESRCH=False):
        # [FUNCTION DOC] HELPER: sends the given signal number to the child pid. An OSError is
        # re-raised unless ignore_ESRCH is true and the error is "no such process" (the child
        # already exited).
        try:
            os.kill(self.pid, signum)
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


class ErrorDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset that reports a length but has no item access method, so
# fetching any item from a loader over it fails in the base class.
    def __init__(self, size):
    # [FUNCTION DOC] HELPER: stores the dataset size.
        self.size = size

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size


class SegfaultDataset(Dataset):
# [CLASS DOC] HELPER: dataset whose every item access reads memory at address zero through ctypes,
# crashing the (worker) process with a segmentation fault.
    def __init__(self, size):
    # [FUNCTION DOC] HELPER: stores the dataset size.
        self.size = size

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: reads from a null address via ctypes, which segfaults the process.
        return ctypes.string_at(0)

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size


class SleepDataset(Dataset):
# [CLASS DOC] HELPER: dataset that sleeps for a configured number of seconds on the first item
# access in each process (a flag marks that it already slept) and returns the index. Used to
# trigger DataLoader timeouts.
    def __init__(self, size, sleep_sec):
    # [FUNCTION DOC] HELPER: stores size and sleep seconds and sets the already-slept flag to
    # false.
        self.size = size
        self.sleep_sec = sleep_sec
        self.slept = False

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: on the first call in this process sleeps and sets the flag; always
    # returns the index.
        if not self.slept:
            time.sleep(self.sleep_sec)
            self.slept = True
        return idx

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size


class SeedDataset(Dataset):
# [CLASS DOC] HELPER: dataset whose items are the initial torch seed of the process computing
# them, used to inspect per worker seeding.
    def __init__(self, size):
    # [FUNCTION DOC] HELPER: stores the dataset size.
        self.size = size

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: returns the current process initial seed, ignoring the index.
        return torch.initial_seed()

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size


class WorkerSpecificIterableDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset where each DataLoader worker yields a range whose length is
# looked up by worker id in a list of per worker sizes. Only usable inside workers.
    def __init__(self, sizes_for_all_workers):
    # [FUNCTION DOC] HELPER: stores the list of per worker sizes.
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
    # [FUNCTION DOC] HELPER: reads the worker info (raises AssertionError if called outside a
    # worker) and returns an iterator over a range sized for this worker id.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            raise AssertionError("Expected worker_info to be available")
        return iter(range(self.sizes_for_all_workers[worker_info.id]))

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the sum of all per worker sizes.
        return sum(self.sizes_for_all_workers)


# Inspired by https://stackoverflow.com/a/26703365
# If all workers will call `sync_once`, they will be blocked until all workers
# reach the call (i.e., acting like a barrier).
# This can be used to ensure that each worker at least processes one data.
class SynchronizedDataset(Dataset):
# [CLASS DOC] HELPER: base dataset providing a cross process barrier so each worker processes at
# least one item. State: a shared locked counter, a semaphore used as the release gate, the worker
# count and the size. Subclasses implement item access and call sync_once.
    def __init__(self, size, batch_size, num_workers):
    # [FUNCTION DOC] HELPER: raises AssertionError if the size is below workers times batch size;
    # otherwise creates the shared counter, the semaphore and stores worker count and size.
        if size < num_workers * batch_size:
            raise AssertionError(
                f"Expected size >= num_workers * batch_size, got size={size}, "
                f"num_workers={num_workers}, batch_size={batch_size}"
            )
        self.count = mp.Value("i", 0, lock=True)
        self.barrier = mp.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def sync_once(self):
    # [FUNCTION DOC] HELPER: barrier. Atomically increments the shared counter and, when the last
    # worker arrives, releases the semaphore; then every caller acquires and immediately releases
    # it so all waiters pass once all workers have arrived.
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: not implemented in the base class; always raises NotImplementedError.
        raise NotImplementedError

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size


class EmptyTensorDataset(torch.utils.data.Dataset):
# [CLASS DOC] HELPER: dataset of a given length whose every item is an empty float tensor.
    def __init__(self, len):
    # [FUNCTION DOC] HELPER: stores the length.
        self.len = len

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored length.
        return self.len

    def __getitem__(self, any):
    # [FUNCTION DOC] HELPER: ignores the key and returns a new empty tensor.
        return torch.empty(0)


class SynchronizedSeedDataset(SynchronizedDataset):
# [CLASS DOC] HELPER: synchronized dataset whose items are the process initial seed, obtained
# after the barrier so all workers have started.
    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: waits at the barrier then returns the initial seed of this process.
        self.sync_once()
        return torch.initial_seed()


def _test_timeout(persistent_workers):
    # [FUNCTION DOC] HELPER: subprocess target. Builds a loader over the sleeping dataset with two
    # workers, batch size two and a one second timeout, and fetches one batch; the dataset sleeps
    # three seconds so a timeout error is expected to propagate. persistent_workers is passed on.
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))


def _test_timeout_pin_memory(persistent_workers):
    # [FUNCTION DOC] HELPER: same as the plain timeout target but with pin_memory enabled, so a
    # pin memory thread is also started; fetches one batch and lets the timeout error propagate.
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))


def _test_large_sampler_indices(persistent_workers):
    # [FUNCTION DOC] HELPER: subprocess target for large sampler index lists. Builds a one worker
    # loader over ten million empty items in batches of 40960 and on the first batch checks it is
    # empty (AssertionError otherwise) and then raises RuntimeError with a fixed message, to check
    # shutdown with large index queues.
    # See
    #   test_large_sampler_indices
    #   https://github.com/pytorch/pytorch/issues/48666

    dataloader = torch.utils.data.DataLoader(
        EmptyTensorDataset(10000000),
        batch_size=40960,
        persistent_workers=persistent_workers,
        num_workers=1,
    )

    it = iter(dataloader)

    for x in it:
        if x.numel() != 0:
            raise AssertionError(f"Expected empty tensor, got numel={x.numel()}")
        raise RuntimeError("My Error")


def disable_stderr(worker_id):
    # [FUNCTION DOC] HELPER: worker_init_fn that flushes stderr and redirects the stderr file
    # descriptor to the null device at OS level, hiding segfault messages printed by the worker
    # signal handler. The worker id argument is unused.
    r"""
    Avoids printing "ERROR: Unexpected segmentation fault encountered in worker."
    from workers. Since worker signal handler prints with low-level write(),
    this has to be done on OS level via dup.

    This is used as worker_init_fn for test_segfault.
    """
    sys.stderr.flush()  # flush library buffers that dup2 knows nothing about
    # Can't use a with-block because otherwise the fd will be closed when this
    # function ends.
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())


def _test_segfault():
    # [FUNCTION DOC] HELPER: subprocess target. Loads one batch from a loader with two workers
    # over the segfaulting dataset, with stderr silenced in workers; the parent is expected to
    # raise.
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=2, worker_init_fn=disable_stderr
    )
    _ = next(iter(dataloader))


def _test_no_segfault():
    # [FUNCTION DOC] HELPER: subprocess target for the fork plus many threads scenario. Ensures
    # the torch thread count is at least four (raising it if lower), then loads one batch from a
    # one worker loader using a fork multiprocessing context. Passes if no crash occurs.
    dataset = [1, 2, 3]
    num_threads = torch.get_num_threads()
    if num_threads < 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(num_threads)
    mp_ctx = torch.multiprocessing.get_context(method="fork")
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=disable_stderr,
        multiprocessing_context=mp_ctx,
    )
    _ = next(iter(dataloader))


class TestProperExitDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset for proper exit tests. Items are one element tensors of
# the index, but if the shared error event is set the last worker raises RuntimeError.
    def __init__(self, size, error_event):
    # [FUNCTION DOC] HELPER: stores the size and the (possibly None) multiprocessing error event.
        self.size = size
        self.error_event = error_event

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: raises RuntimeError if an event was given, is set, and this worker is
    # the last one; otherwise returns a one element tensor holding the index.
        worker_info = torch.utils.data.get_worker_info()
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            # only error in the last worker
            raise RuntimeError("Worker error")
        return torch.tensor([idx])


class TestProperExitIterableDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable counterpart used by proper exit tests; acts as its own iterator,
# yields a tensor of minus 1000 per item until the remaining count runs out, and the last worker
# raises RuntimeError once the error event is set.
    def __init__(self, size, error_event):
    # [FUNCTION DOC] HELPER: stores the error event, the size and a remaining counter equal to
    # size.
        self.error_event = error_event
        self.size = size
        self.remaining = size

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored size.
        return self.size

    def __iter__(self):
    # [FUNCTION DOC] HELPER: returns the dataset itself as iterator.
        return self

    def __next__(self):
    # [FUNCTION DOC] HELPER: raises RuntimeError if the error event is set and this worker is the
    # last one; otherwise decrements the remaining counter, raises StopIteration when it goes
    # negative, else returns a scalar tensor of minus 1000. Requires worker info (must run inside
    # a worker).
        worker_info = torch.utils.data.get_worker_info()
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            # only error in the last worker
            raise RuntimeError("Worker error")
        self.remaining -= 1
        if self.remaining < 0:
            raise StopIteration
        return torch.tensor(-1000)


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(
    is_iterable_dataset,
    use_workers,
    pin_memory,
    exit_method,
    hold_iter_reference,
    loader_setup_event,
    tester_setup_event,
    persistent_workers,
):
    # [FUNCTION DOC] HELPER: subprocess body for the proper exit test. Builds a map or iterable
    # dataset, a DataLoader (batch size one, no shuffle, two workers or none, optional pin_memory
    # and persistent workers), validates the dataset is long enough to outlast prefetching
    # (AssertionError otherwise), then iterates. At the first batch it optionally drops the
    # iterator and loader, signals loader_setup_event, waits for tester_setup_event, checks all
    # workers alive and sets the worker error event if used. At the third batch it triggers the
    # chosen failure: the loader raises, the loader process is killed, or the last worker is
    # killed. If the references were dropped it runs garbage collection at the end. exit_method
    # worker_error and worker_kill require workers.
    num_workers = 2 if use_workers else 0

    if exit_method == "worker_error" or exit_method == "worker_kill":
        if use_workers is not True:
            raise AssertionError("Expected use_workers=True for worker exit methods")

    if exit_method == "worker_error":
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    if is_iterable_dataset:
        ds = TestProperExitIterableDataset(7, worker_error_event)
    else:
        ds = TestProperExitDataset(12, worker_error_event)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=set_faulthander_if_available,
        persistent_workers=persistent_workers,
    )

    error_it = 2

    if use_workers:
        # 2 is the magical per-worker prefetch number...
        # FIXME: change this after the number becomes configurable.
        if is_iterable_dataset:
            if len(ds) * num_workers <= (error_it + 2 + 1):
                raise AssertionError(
                    "Expected iterable dataset size to exceed error threshold"
                )
        else:
            if len(loader) <= (error_it + 2 + 1) * num_workers:
                raise AssertionError("Expected loader length to exceed error threshold")
    else:
        if is_iterable_dataset:
            if len(ds) <= error_it + 1:
                raise AssertionError(
                    "Expected iterable dataset length to exceed error threshold"
                )
        else:
            if len(loader) <= error_it + 1:
                raise AssertionError("Expected loader length to exceed error threshold")

    it = iter(loader)
    if use_workers:
        workers = it._workers

    def kill_pid(pid):
        # [FUNCTION DOC] HELPER: kills the process with the given pid using psutil, waits up to
        # the join timeout and raises AssertionError if it is still running.
        psutil_p = psutil.Process(pid)
        psutil_p.kill()
        psutil_p.wait(JOIN_TIMEOUT)
        if psutil_p.is_running():
            raise AssertionError("Expected process to be terminated")

    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                del it
                del loader
            loader_setup_event.set()
            tester_setup_event.wait()
            # ensure that the workers are still alive
            if use_workers:
                for w in workers:
                    if not w.is_alive():
                        raise AssertionError("Expected worker process to be alive")
            if worker_error_event is not None:
                worker_error_event.set()

        if i == error_it:
            if exit_method == "loader_error":
                raise RuntimeError("Loader error")
            elif exit_method == "loader_kill":
                kill_pid(os.getpid())
            elif exit_method == "worker_kill":
                kill_pid(workers[-1].pid)  # kill last worker

    if not hold_iter_reference:
        # Tries to trigger the __del__ clean-up rather than the automatic
        # exiting of daemonic children. Technically it should be automatically
        # triggered, but I don't want to rely on the implementation detail of
        # Python gc.
        gc.collect()


class TestWorkerInfoDataset(SynchronizedDataset):
# [CLASS DOC] HELPER: synchronized dataset used with the worker info init function. Items are a
# tensor of the per worker value attribute (a worker id and pid pair set in the init function),
# returned after the barrier.
    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: waits at the barrier then returns a tensor built from the value
    # attribute; fails with AttributeError in a process where that attribute was never set (such
    # as the parent).
        self.sync_once()
        return torch.tensor(self.value)


# Should be used as worker_init_fn with TestWorkerInfoDataset.
# See _test_get_worker_info below for usage.
def _test_worker_info_init_fn(worker_id):
    # [FUNCTION DOC] HELPER: worker_init_fn verifying the worker info: the id matches the argument
    # and is in range, the seed matches the process initial seed, the dataset is a worker's own
    # copy of the right class without the value attribute, and the repr mentions id, num_workers,
    # seed, dataset and rng. Raises AssertionError on any failure. Finally stores the worker id
    # and pid pair on the worker's dataset copy.
    worker_info = torch.utils.data.get_worker_info()
    if worker_id != worker_info.id:
        raise AssertionError("worker_init_fn and worker_info should have consistent id")
    if worker_id >= worker_info.num_workers:
        raise AssertionError("worker_init_fn and worker_info should have valid id")
    if worker_info.seed != torch.initial_seed():
        raise AssertionError(
            "worker_init_fn and worker_info should have consistent seed"
        )
    dataset = worker_info.dataset
    if not isinstance(dataset, TestWorkerInfoDataset):
        raise AssertionError("worker_info should have correct dataset copy")
    if hasattr(dataset, "value"):
        raise AssertionError("worker_info should have correct dataset copy")
    for k in ["id", "num_workers", "seed", "dataset", "rng"]:
        if f"{k}=" not in repr(worker_info):
            raise AssertionError(f"Expected {k} in worker_info repr")
    dataset.value = [worker_id, os.getpid()]


def _test_get_worker_info():
    # [FUNCTION DOC] HELPER: subprocess target. Checks get_worker_info is None in the main
    # process, runs a loader with two workers and the init function above, collects all batches,
    # checks each returned pair's pid equals the pid of the worker with that id, checks worker
    # info is None again afterwards and that the main process dataset lacks the value attribute.
    # Raises AssertionError or RuntimeError on mismatch.
    # get_worker_info returns None in main proc
    if torch.utils.data.get_worker_info() is not None:
        raise AssertionError("Expected get_worker_info() to return None in main proc")
    num_workers = 2
    batch_size = 2
    dataset = TestWorkerInfoDataset(6, batch_size, num_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_test_worker_info_init_fn,
    )
    it = iter(dataloader)
    worker_pids = [w.pid for w in it._workers]
    data = []
    for d in it:
        data.append(d)  # noqa: PERF402
    data = torch.cat(data, 0)
    for d in data:
        # each `d` is a [worker_id, worker_pid] pair, which is set in
        # _test_worker_info_init_fn
        if d[1] != worker_pids[d[0]]:
            raise AssertionError(f"Expected worker pid {worker_pids[d[0]]}, got {d[1]}")
    # get_worker_info returns None in main proc after data loading
    if torch.utils.data.get_worker_info() is not None:
        raise AssertionError(
            "Expected get_worker_info() to return None after data loading"
        )
    # main proc dataset was never assigned this attribute
    if hasattr(dataset, "value"):
        raise AssertionError("Expected main dataset to not have 'value' attribute")
    try:
        _ = dataset[0]
    except AttributeError:
        return
    raise RuntimeError("Expected AttributeError")


# test custom init function
def init_fn(worker_id):
    # [FUNCTION DOC] HELPER: worker_init_fn that seeds torch with the constant 12345; the worker
    # id is ignored.
    torch.manual_seed(12345)


# used with test_error_in_init
class ErrorIterableDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset whose iteration always raises RuntimeError, to test errors
# at iterator creation.
    def __iter__(self):
    # [FUNCTION DOC] HELPER: always raises RuntimeError with a fixed message.
        raise RuntimeError("Error in __iter__")


# used with test_error_in_init
def error_worker_init_fn(_):
    # [FUNCTION DOC] HELPER: worker_init_fn that always raises RuntimeError with a fixed message.
    raise RuntimeError("Error in worker_init_fn")


class BulkLoadingDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset taking a whole list or tuple of indices per access and
# returning them as a tensor; used with a sampler yielding index lists and no automatic batching.
    def __init__(self, length):
    # [FUNCTION DOC] HELPER: stores the length.
        self.length = length

    def __getitem__(self, indices):
    # [FUNCTION DOC] HELPER: raises AssertionError unless the argument is a list or tuple; returns
    # it converted to a tensor.
        if not isinstance(indices, (list, tuple)):
            raise AssertionError(
                f"Expected indices to be list or tuple, got {type(indices)}"
            )
        return torch.as_tensor(indices)

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored length.
        return self.length


class BulkLoadingSampler(torch.utils.data.Sampler):
# [CLASS DOC] HELPER: sampler yielding lists of indices, one list per batch, from a random
# permutation split into chunks of the batch size.
    def __init__(self, dataset, batch_size):
    # [FUNCTION DOC] HELPER: stores the dataset and the batch size.
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
    # [FUNCTION DOC] HELPER: generator; permutes dataset indices randomly and yields each chunk as
    # a plain list.
        for x in torch.randperm(len(self.dataset)).split(self.batch_size):
            yield x.tolist()

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the number of batches, the dataset length divided by batch
    # size rounded up.
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))


class TestMultiEpochDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset where each worker yields its own worker id repeatedly,
# length divided by number of workers times. Requires being run in a worker.
    def __init__(self, length):
    # [FUNCTION DOC] HELPER: stores the total length.
        self.length = length

    def __iter__(self):
    # [FUNCTION DOC] HELPER: generator; raises AssertionError outside a worker, otherwise yields
    # the worker id, integer-division of length by worker count times.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            raise AssertionError("Expected worker_info to be available")
        worker_id = worker_info.id
        for _ in range(self.length // worker_info.num_workers):
            yield worker_id

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the stored length.
        return self.length


class CustomList(list):
# [CLASS DOC] HELPER: trivial list subclass used to check collation preserves custom list types.
    pass


class CustomDict(dict):
# [CLASS DOC] HELPER: trivial dict subclass used to check collation preserves custom mapping
# types.
    pass


def row_processor(row):
    # [FUNCTION DOC] HELPER: datapipe map function; adds one to each element of the row using
    # numpy (so numpy must be installed) and returns the result.
    return np.add(row, 1)


def filter_len(row):
    # [FUNCTION DOC] HELPER: datapipe filter predicate; true when the row has exactly four
    # elements.
    return len(row) == 4


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223",
)
class TestDataLoader(TestCase):
    # [CLASS DOC] Main DataLoader test case. State from setUp: random data tensor, label tensor, a
    # TensorDataset of both and a persistent_workers flag (false here; subclasses change it).
    # Loaders are built through the helper that drops persistent workers when no workers are used.
    def setUp(self):
        # [FUNCTION DOC] Creates a 100 sample random data tensor, labels that are a permutation of
        # fifty values repeated twice, the dataset over both, and sets the persistent workers flag
        # false.
        super().setUp()
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)
        self.persistent_workers = False

    def _get_data_loader(self, dataset, **kwargs):
        # [FUNCTION DOC] HELPER: builds a DataLoader forwarding keyword arguments. The
        # persistent_workers setting defaults to the test case attribute but is forced to false
        # when num_workers is zero (or absent), since persistent workers need worker processes.
        persistent_workers = kwargs.get("persistent_workers", self.persistent_workers)
        if persistent_workers and kwargs.get("num_workers", 0) == 0:
            persistent_workers = False
        kwargs["persistent_workers"] = persistent_workers
        return DataLoader(dataset, **kwargs)

    def _test_sequential(self, loader):
        # [FUNCTION DOC] HELPER: asserts a loader yields the dataset samples and labels in order,
        # for the unbatched case (sample by sample, last index is dataset length minus one) and
        # batched case (slices of batch size, last batch index matches floor division).
        batch_size = loader.batch_size
        if batch_size is None:
            for idx, (sample, target) in enumerate(loader):
                self.assertEqual(sample, self.data[idx])
                self.assertEqual(target, self.labels[idx])
            self.assertEqual(idx, len(self.dataset) - 1)
        else:
            for i, (sample, target) in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(sample, self.data[idx : idx + batch_size])
                self.assertEqual(target, self.labels[idx : idx + batch_size])
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        # [FUNCTION DOC] HELPER: asserts a shuffled loader yields every sample at most once and
        # each with its matching label, tracking found counts per dataset index; for unbatched and
        # batched loaders, and that the final batch count equals the expected number.
        found_data = dict.fromkeys(range(self.data.size(0)), 0)
        found_labels = dict.fromkeys(range(self.labels.size(0)), 0)
        batch_size = loader.batch_size
        if batch_size is None:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                sample, target = (batch_samples, batch_targets)
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
                self.assertEqual(sum(found_data.values()), (i + 1))
                self.assertEqual(sum(found_labels.values()), (i + 1))
            self.assertEqual(i, (len(self.dataset) - 1))
        else:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                for sample, target in zip(batch_samples, batch_targets):
                    for data_point_idx, data_point in enumerate(self.data):
                        if data_point.eq(sample).all():
                            self.assertFalse(found_data[data_point_idx])
                            found_data[data_point_idx] += 1
                            break
                    self.assertEqual(target, self.labels[data_point_idx])
                    found_labels[data_point_idx] += 1
                self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
                self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_error(self, loader):
        # [FUNCTION DOC] HELPER: iterates the loader, counting NotImplementedError exceptions from
        # next calls; on StopIteration asserts the error count equals the number of batches and
        # returns.
        it = iter(loader)
        errors = 0
        while True:
            try:
                next(it)
            except NotImplementedError:
                errors += 1
            except StopIteration:
                self.assertEqual(
                    errors, math.ceil(float(len(loader.dataset)) / loader.batch_size)
                )
                return

    def test_error_in_init(self):
        # [FUNCTION DOC] Checks that an error in the iterable dataset iteration method surfaces as
        # RuntimeError for zero and two workers, and an error in the worker init function surfaces
        # as RuntimeError.
        for num_workers in [0, 2]:
            loader = self._get_data_loader(
                ErrorIterableDataset(), num_workers=num_workers
            )
            with self.assertRaisesRegex(RuntimeError, "Error in __iter__"):
                list(iter(loader))

        loader = self._get_data_loader(
            self.dataset, num_workers=2, worker_init_fn=error_worker_init_fn
        )
        with self.assertRaisesRegex(RuntimeError, "Error in worker_init_fn"):
            list(iter(loader))

    def test_typing(self):
        # [FUNCTION DOC] Smoke test that subclassing the generic Dataset and annotating DataLoader
        # generics causes no TypeError at definition time. Defines but never calls anything.
        # Make sure there is no TypeError

        class SomeDatasetClass(Dataset[list[torch.Tensor]]):
            # [CLASS DOC] HELPER: empty subclass of a parametrized generic Dataset, only used to
            # test typing.
            pass

        def _create_dataloader(is_train: bool) -> DataLoader[list[torch.Tensor]]:
            # [FUNCTION DOC] HELPER: unused stub with a generic return annotation; does nothing.
            pass

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # [FUNCTION DOC] Runs a subprocess (skipped on Sandcastle and Windows) that sets a very
        # low open file limit and iterates a forked one worker loader over an iterable dataset
        # keeping tensors alive; expects the RuntimeError message to mention raising the ulimit
        # and changing the sharing strategy. The check_output call fails if the subprocess exits
        # non zero.
        # See NOTE [ DataLoader on Linux and open files limit ]
        import subprocess

        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
""",
            ]
        )

    def test_invalid_assign_after_init(self):
        # [FUNCTION DOC] Checks assigning to batch_size, sampler, batch_sampler, drop_last or
        # dataset on a constructed loader raises ValueError.
        dl = self._get_data_loader(self.dataset)
        for attr in ("batch_size", "sampler", "batch_sampler", "drop_last", "dataset"):

            def fn():
            # [FUNCTION DOC] HELPER: tries to assign an empty mapping to the current loop
            # attribute of the loader (closure over the loop variable).
                setattr(dl, attr, {})

            self.assertRaises(ValueError, fn)

    def test_sequential_nonbatch(self):
        # [FUNCTION DOC] Runs the sequential check for a loader with batch_size None.
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=None))

    def test_sequential_batch(self):
        # [FUNCTION DOC] Runs the sequential check for the default batch size and for batch size
        # two.
        self._test_sequential(self._get_data_loader(self.dataset))
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=2))

    def test_bulk_loading_nobatch(self):
        # [FUNCTION DOC] Checks loading with batch_size None and a sampler that yields lists of
        # indices, with zero and four workers and pin_memory per platform support: automatic
        # collation is off, sample pinning matches the flag, and the union of indices equals the
        # full range.
        n = 35
        bs = 4
        ds = BulkLoadingDataset(n)
        sampler = BulkLoadingSampler(ds, batch_size=4)

        for num_workers in [0, 4]:
            dl = self._get_data_loader(
                ds,
                num_workers=num_workers,
                batch_size=None,
                sampler=sampler,
                pin_memory=TEST_PIN_MEMORY,
            )
            self.assertFalse(dl._auto_collation)
            samples = list(dl)
            self.assertEqual(samples[0].is_pinned(), TEST_PIN_MEMORY)
            self.assertEqual(set(torch.cat(samples, 0).tolist()), set(range(n)))

    def test_growing_dataset(self):
        # [FUNCTION DOC] Checks loader length follows the growth of a list dataset appended after
        # the loaders were created, for sequential and shuffled loaders.
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = self._get_data_loader(dataset, shuffle=False)
        dataloader_shuffle = self._get_data_loader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_sequential_pin_memory(self):
        # [FUNCTION DOC] Checks batches from a pin_memory loader have pinned input and target
        # tensors. Skipped when no accelerator with pinned memory is available.
        loader = self._get_data_loader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_multiple_dataloaders(self):
        # [FUNCTION DOC] Creates and advances two iterators of loaders with one and two workers
        # (for each supported multiprocessing context), interleaving next calls then deleting
        # them, to check multiple live loaders coexist. Skipped without CUDA IPC.
        for multiprocessing_context in supported_multiprocessing_contexts:
            loader1_it = iter(self._get_data_loader(self.dataset, num_workers=1))
            loader2_it = iter(
                self._get_data_loader(
                    self.dataset,
                    num_workers=2,
                    multiprocessing_context=multiprocessing_context,
                )
            )
            next(loader1_it)
            next(loader1_it)
            next(loader2_it)
            next(loader2_it)
            next(loader1_it)
            next(loader2_it)
            del loader1_it
            del loader2_it

    @skipIfXpu
    @unittest.skipIf(IS_S390X, "Unexpectedly succeeds on s390x")
    # Test that DataLoader properly handles worker segfaults
    # Note: This test has inconsistent behavior across Linux distributions:
    # - Passes on RHEL 9.6 (segfault triggers correctly)
    # - Fails on Ubuntu (process may not terminate as expected)
    # Skipping on Linux due to kernel/distribution-dependent segfault behavior.
    @unittest.skipIf(
        IS_LINUX, "Segfault behavior is inconsistent across Linux distributions"
    )
    def test_segfault(self):
        # [FUNCTION DOC] Runs the segfault target in an error tracking process and checks it
        # exited with non zero code and the recorded exception is a RuntimeError reporting the
        # worker killed by a signal (OSError access violation on Windows). Terminates the process
        # at the end. Currently skipped on Linux for platform inconsistency, and on XPU and s390x.
        p = ErrorTrackingProcess(target=_test_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            if IS_WINDOWS:
                self.assertIsInstance(p.exception, OSError)
                self.assertRegex(str(p.exception), r"access violation reading ")
            else:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
        finally:
            p.terminate()

    # Tests if the child process forked by the DataLoader segfaults due to having more than 3 threads
    # in the parent process after at least one set_num_threads invocation in the parent process.
    # After forking, set_num_threads(1) in the child process entails handling some inherited data-structures
    # of the Caffe2 thread-pool of the parent process, culminating in a segfault.
    # Reference: https://github.com/pytorch/pytorch/issues/54752
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_no_segfault(self):
        # [FUNCTION DOC] Regression test for a crash after fork with more than three threads: runs
        # the no-segfault target in a child process; fails if the child recorded a worker killed
        # by signal error. Skipped on Windows.
        p = ErrorTrackingProcess(target=_test_no_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            if p.exception:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
                self.fail("Segfault occurred in worker process after fork")
        finally:
            p.terminate()

    def test_timeout(self):
        # [FUNCTION DOC] For each timeout target (plain, plus pin memory variant when an
        # accelerator pin allocator is available) runs it in an error tracking process and checks
        # the process exited non zero with a RuntimeError saying the DataLoader timed out after
        # some seconds. Always terminates the process.
        if TEST_PIN_MEMORY:
            # This test runs in a subprocess, which can only initialize the
            # accelerator with spawn. _test_timeout_pin_memory with pin_memory=True
            # initializes the accelerator when the iterator is constructed.
            targets = (_test_timeout, _test_timeout_pin_memory)
        else:
            targets = (_test_timeout,)
        for target in targets:
            p = ErrorTrackingProcess(target=target, args=(self.persistent_workers,))
            p.start()
            p.join(JOIN_TIMEOUT)
            try:
                self.assertFalse(p.is_alive())
                self.assertNotEqual(p.exitcode, 0)
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception), r"DataLoader timed out after \d+ seconds"
                )
            finally:
                p.terminate()

    def test_large_sampler_indices(self):
        # [FUNCTION DOC] Regression test for issue 48666: runs the large sampler target in a child
        # process and checks it terminates (does not hang) with non zero exit and a RuntimeError
        # whose message is the one raised by the target, i.e. cleanup works while index queue puts
        # block.
        # Test that the data loader cleanly exit when the process errors
        #   1. having an reference to the iterator
        #   2. using a sampler that yields big elements s.t. _index_queues putters block
        #
        # More context: https://github.com/pytorch/pytorch/issues/48666

        p = ErrorTrackingProcess(
            target=_test_large_sampler_indices, args=(self.persistent_workers,)
        )
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r"My Error")
        finally:
            p.terminate()

    def test_invalid_ctor_args_combinations(self):
        # [FUNCTION DOC] Checks the DataLoader constructor raises ValueError or TypeError with
        # specific messages for invalid combinations: negative workers or timeout, batch_size None
        # with drop_last, multiprocessing context without workers or with a bad name or wrong
        # type, sampler with shuffle, batch_sampler with batch size, shuffle, drop_last or
        # sampler, and for iterable datasets any shuffle, sampler or batch_sampler setting.
        # general
        with self.assertRaisesRegex(
            ValueError, "num_workers option should be non-negative"
        ):
            self._get_data_loader(self.dataset, num_workers=-1)
        with self.assertRaisesRegex(
            ValueError, "timeout option should be non-negative"
        ):
            self._get_data_loader(self.dataset, timeout=-1)

        # disable auto-batching
        with self.assertRaisesRegex(
            ValueError,
            "batch_size=None option disables auto-batching and is mutually exclusive",
        ):
            self._get_data_loader(self.dataset, batch_size=None, drop_last=True)

        valid_ctx = list(torch.multiprocessing.get_all_start_methods())[-1]
        with self.assertRaisesRegex(
            ValueError, r"multi-process loading \(num_workers > 0\), but got"
        ):
            self._get_data_loader(
                self.dataset, num_workers=0, multiprocessing_context=valid_ctx
            )
        with self.assertRaisesRegex(
            ValueError, "should specify a valid start method in"
        ):
            self._get_data_loader(
                self.dataset, num_workers=1, multiprocessing_context="bad"
            )
        with self.assertRaisesRegex(
            TypeError, "multiprocessing_context option should be a valid context "
        ):
            self._get_data_loader(
                self.dataset, num_workers=1, multiprocessing_context=object()
            )

        # map-style
        sampler = torch.utils.data.SequentialSampler(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 3, False)
        with self.assertRaisesRegex(
            ValueError, "sampler option is mutually exclusive with shuffle"
        ):
            self._get_data_loader(
                self.dataset, batch_size=11, sampler=sampler, shuffle=True
            )
        with self.assertRaisesRegex(
            ValueError, "sampler option is mutually exclusive with shuffle"
        ):
            self._get_data_loader(
                self.dataset, batch_sampler=batch_sampler, sampler=sampler, shuffle=True
            )
        with self.assertRaisesRegex(
            ValueError, "sampler option is mutually exclusive with shuffle"
        ):
            self._get_data_loader(
                self.dataset, batch_sampler=batch_sampler, sampler=sampler, shuffle=3
            )
        with self.assertRaisesRegex(
            ValueError, "batch_sampler option is mutually exclusive with"
        ):
            self._get_data_loader(
                self.dataset, batch_size=11, batch_sampler=batch_sampler
            )
        with self.assertRaisesRegex(
            ValueError, "batch_sampler option is mutually exclusive with"
        ):
            self._get_data_loader(
                self.dataset, shuffle=True, batch_sampler=batch_sampler
            )
        with self.assertRaisesRegex(
            ValueError, "batch_sampler option is mutually exclusive with"
        ):
            self._get_data_loader(
                self.dataset, drop_last=True, batch_sampler=batch_sampler
            )
        with self.assertRaisesRegex(
            ValueError, "batch_sampler option is mutually exclusive with"
        ):
            self._get_data_loader(
                self.dataset, drop_last=3, batch_sampler=batch_sampler
            )

        # iterable-style
        dataset = CountingIterableDataset(20)
        with self.assertRaisesRegex(
            ValueError, "DataLoader with IterableDataset: expected unspecified shuffle"
        ):
            self._get_data_loader(dataset, shuffle=True)
        with self.assertRaisesRegex(
            ValueError, "DataLoader with IterableDataset: expected unspecified shuffle"
        ):
            self._get_data_loader(dataset, shuffle=3)
        with self.assertRaisesRegex(
            ValueError, "DataLoader with IterableDataset: expected unspecified sampler"
        ):
            self._get_data_loader(
                dataset, sampler=torch.utils.data.SequentialSampler(dataset)
            )
        with self.assertRaisesRegex(
            ValueError, "DataLoader with IterableDataset: expected unspecified sampler"
        ):
            self._get_data_loader(dataset, sampler=3)
        with self.assertRaisesRegex(
            ValueError,
            "DataLoader with IterableDataset: expected unspecified batch_sampler",
        ):
            self._get_data_loader(
                dataset,
                batch_sampler=torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(dataset), 3, False
                ),
            )
        with self.assertRaisesRegex(
            ValueError,
            "DataLoader with IterableDataset: expected unspecified batch_sampler",
        ):
            self._get_data_loader(dataset, batch_sampler=3)

    def test_builtin_collection_conversion(self):
        # [FUNCTION DOC] Checks converting a loader to a list or tuple gives the expected items
        # for map style and iterable datasets, with and without auto batching, for zero and one
        # workers.
        for coll_ty in (list, tuple):
            for num_workers in (0, 1):
                # map-style dataset
                dataset = CountingDataset(20)
                # no auto-batching
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=None, num_workers=num_workers
                    )
                )
                self.assertEqual(fetched, coll_ty(range(20)))
                # auto-batching
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=2, num_workers=num_workers
                    )
                )
                self.assertEqual(
                    fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2))
                )

                # iterable-style dataset
                dataset = CountingIterableDataset(20)
                # no auto-batching
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=None, num_workers=num_workers
                    )
                )
                self.assertEqual(fetched, coll_ty(range(20)))
                # auto-batching
                # this IterableDataset isn't configured for each worker, so for
                # the equality test below to be valid, we cannot have more than 1 workers.
                if num_workers not in [0, 1]:
                    raise AssertionError("invalid test")
                fetched = coll_ty(
                    self._get_data_loader(
                        dataset, batch_size=2, num_workers=num_workers
                    )
                )
                self.assertEqual(
                    fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2))
                )

    def test_iterable_style_dataset(self):
        # [FUNCTION DOC] Checks loading an iterable style dataset: single process and multi
        # process (three workers with different per worker sizes, various prefetch factors),
        # unbatched, batched and batched with drop_last. Verifies item types, loader length
        # matching the dataset length, the warning emitted once more samples than the reported
        # length were fetched, batch contents, and that workers exit cleanly with code zero after
        # the iterator and loader are deleted.
        # [no auto-batching] single process loading
        dataset = CountingIterableDataset(20)
        dataloader = self._get_data_loader(dataset, batch_size=None)
        fetched = list(dataloader)
        self.assertEqual(len(fetched), 20)
        for i, d in enumerate(fetched):
            # non-batched should not convert ints into tensors
            self.assertIsInstance(d, int)
            self.assertEqual(d, i)
        # DataLoader should match len of the iterable-style dataset (if implemented)
        self.assertEqual(len(dataloader), len(dataset))

        # [no auto-batching] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(
            functools.reduce(
                operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []
            )
        )
        if len(sizes_for_all_workers) != num_workers:
            raise AssertionError("invalid test case")
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            dataloader = self._get_data_loader(
                dataset,
                num_workers=num_workers,
                batch_size=None,
                worker_init_fn=set_faulthander_if_available,
                prefetch_factor=prefetch_factor,
            )
            dataloader_iter = iter(dataloader)
            fetched = sorted(dataloader_iter)
            for a, b in zip(fetched, expected):
                # non-batched should not convert ints into tensors
                self.assertIsInstance(a, int)
                self.assertEqual(a, b)
            # DataLoader should match len of the iterable-style dataset (if implemented)
            self.assertEqual(len(dataloader), len(dataset))
            # When loading more than len(dataset) data, after accessing len(dataloader),
            # we should get a warning. See NOTE [ IterableDataset and __len__ ].
            dataset = CountingIterableDataset(20)
            dataloader = self._get_data_loader(
                dataset,
                num_workers=num_workers,
                worker_init_fn=set_faulthander_if_available,
                prefetch_factor=prefetch_factor,
            )
            it = iter(dataloader)
            for _ in range(40):
                self.assertNotWarn(
                    lambda: next(it), "Should not warn before accessing len(dataloader)"
                )
            self.assertEqual(len(dataloader), len(dataset))
            self.assertEqual(len(dataloader), 20)
            it = iter(dataloader)
            for _ in range(20):
                self.assertNotWarn(
                    lambda: next(it), "Should not warn before exceeding length"
                )
            for _ in range(3):
                with self.assertWarnsRegex(
                    UserWarning,
                    r"but [0-9]+ samples have been fetched\. For multiprocessing data-loading, this",
                    msg="Should always warn after exceeding length",
                ):
                    next(it)
        # [no auto-batching] test that workers exit gracefully
        workers = dataloader_iter._workers
        del dataloader_iter
        del dataloader
        try:
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive())
                self.assertEqual(w.exitcode, 0)
        finally:
            for w in workers:
                w.terminate()

        # [auto-batching] single process loading
        dataset = CountingIterableDataset(20)
        fetched = list(self._get_data_loader(dataset, batch_size=7))
        self.assertEqual(len(fetched), 3)
        self.assertEqual(fetched[0].tolist(), list(range(7)))
        self.assertEqual(fetched[1].tolist(), list(range(7, 14)))
        self.assertEqual(fetched[2].tolist(), list(range(14, 20)))

        # [auto-batching] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(
            functools.reduce(
                operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []
            )
        )
        if len(sizes_for_all_workers) != num_workers:
            raise AssertionError("invalid test case")
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            # worker 0 should return 0 batches
            # worker 1 should return 1 batches
            # worker 2 should return 3 batches
            dataloader = self._get_data_loader(
                dataset,
                num_workers=num_workers,
                batch_size=7,
                prefetch_factor=prefetch_factor,
            )
            dataloader_iter = iter(dataloader)
            fetched = list(dataloader_iter)
            self.assertEqual(len(fetched), 4)
            fetched = {tuple(t.tolist()) for t in fetched}
            self.assertEqual(
                fetched,
                {
                    tuple(range(4)),
                    tuple(range(7)),
                    tuple(range(7, 14)),
                    tuple(range(14, 20)),
                },
            )

            # [auto-batching] test that workers exit gracefully
            workers = dataloader_iter._workers
            del dataloader_iter
            del dataloader
            try:
                for w in workers:
                    w.join(JOIN_TIMEOUT)
                    self.assertFalse(w.is_alive())
                    self.assertEqual(w.exitcode, 0)
            finally:
                for w in workers:
                    w.terminate()
        # [auto-batching & drop_last] single process loading
        dataset = CountingIterableDataset(20)
        fetched = list(self._get_data_loader(dataset, batch_size=7, drop_last=True))
        self.assertEqual(len(fetched), 2)
        self.assertEqual(fetched[0].tolist(), list(range(7)))
        self.assertEqual(fetched[1].tolist(), list(range(7, 14)))

        # [auto-batching & drop_last] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(
            functools.reduce(
                operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []
            )
        )
        if len(sizes_for_all_workers) != num_workers:
            raise AssertionError("invalid test case")
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            # worker 0 should return 0 batches
            # worker 1 should return 1 batches
            # worker 2 should return 3 batches
            dataloader = self._get_data_loader(
                dataset,
                num_workers=num_workers,
                batch_size=7,
                drop_last=True,
                worker_init_fn=set_faulthander_if_available,
                prefetch_factor=prefetch_factor,
            )
            dataloader_iter = iter(dataloader)
            fetched = list(dataloader_iter)
            self.assertEqual(len(fetched), 2)
            fetched = {tuple(t.tolist()) for t in fetched}
            self.assertEqual(fetched, {tuple(range(7)), tuple(range(7, 14))})

            # [auto-batching & drop_last] test that workers exit gracefully
            workers = dataloader_iter._workers
            del dataloader_iter
            del dataloader
            try:
                for w in workers:
                    w.join(JOIN_TIMEOUT)
                    self.assertFalse(w.is_alive())
                    self.assertEqual(w.exitcode, 0)
            finally:
                for w in workers:
                    w.terminate()

    def test_chain_iterable_style_dataset(self):
        # [FUNCTION DOC] Checks chaining iterable datasets by the plus operator and ChainDataset
        # gives the concatenated sequence for zero and one workers, and that chaining a map style
        # dataset raises AssertionError when iterated.
        # chaining (concatenation)
        dataset1 = CountingIterableDataset(20)
        dataset2 = CountingIterableDataset(15)
        expected = list(range(20)) + list(range(15))
        for num_workers in [0, 1]:
            for chained_dataset in [
                dataset1 + dataset2,
                ChainDataset([dataset1, dataset2]),
            ]:
                fetched = list(
                    self._get_data_loader(chained_dataset, num_workers=num_workers)
                )
                self.assertEqual(len(fetched), len(expected))
                for e, d in zip(expected, fetched):
                    self.assertIsInstance(d, torch.Tensor)
                    self.assertEqual(e, d)

        with self.assertRaisesRegex(
            AssertionError, "ChainDataset only supports IterableDataset"
        ):
            list(iter(dataset1 + self.dataset))

        with self.assertRaisesRegex(
            AssertionError, "ChainDataset only supports IterableDataset"
        ):
            list(iter(ChainDataset([dataset1, self.dataset])))

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_multiprocessing_contexts(self):
        # [FUNCTION DOC] Checks loaders over a counting dataset give identical batches for every
        # supported multiprocessing context, passed as name or as context object; uses a CUDA
        # dataset for spawn and forkserver when CUDA is available on supported platforms. Skipped
        # without CUDA IPC.
        reference = [
            torch.arange(3),
            torch.arange(3, 6),
            torch.arange(6, 9),
            torch.arange(9, 11),
        ]
        counting_ds_n = 11
        dl_common_args = dict(num_workers=3, batch_size=3, pin_memory=False)
        for ctx in supported_multiprocessing_contexts:
            # windows and jetson devices don't support sharing cuda tensor; ROCm does not yet fully support IPC
            if (
                ctx in ["spawn", "forkserver"]
                and TEST_CUDA
                and not IS_WINDOWS
                and not IS_JETSON
            ):
                ds_cls = CUDACountingDataset
            else:
                ds_cls = CountingDataset
            self.assertEqual(
                reference,
                list(
                    self._get_data_loader(
                        ds_cls(counting_ds_n),
                        multiprocessing_context=ctx,
                        **dl_common_args,
                    )
                ),
            )
            if ctx is not None:
                # test ctx object
                ctx = mp.get_context(ctx)
                self.assertEqual(
                    reference,
                    list(
                        self._get_data_loader(
                            ds_cls(counting_ds_n),
                            multiprocessing_context=ctx,
                            **dl_common_args,
                        )
                    ),
                )

    def _test_multiprocessing_iterdatapipe(self, with_dill):
        # [FUNCTION DOC] HELPER: builds a datapipe over two lists, maps it with a module level
        # function and filters rows by a length of four, using a lambda when dill is allowed or a
        # global function otherwise. For each multiprocessing context (name and object) loads with
        # two workers and compares to the expected tensors (converted to int64).
        # Testing to make sure that function from global scope (e.g. imported from library) can be serialized
        # and used with multiprocess DataLoader

        reference = [
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),
        ]
        datapipe: IterDataPipe = IterableWrapper([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
        datapipe = datapipe.map(row_processor)
        datapipe = (
            datapipe.filter(lambda row: len(row) == 4)
            if with_dill
            else datapipe.filter(filter_len)
        )

        dl_common_args = dict(
            num_workers=2, batch_size=2, shuffle=True, pin_memory=False
        )
        for ctx in supported_multiprocessing_contexts:
            self.assertEqual(
                reference,
                [
                    t.type(torch.int64)
                    for t in self._get_data_loader(
                        datapipe, multiprocessing_context=ctx, **dl_common_args
                    )
                ],
            )
            if ctx is not None:
                # test ctx object
                ctx = mp.get_context(ctx)
                self.assertEqual(
                    reference,
                    [
                        t.type(torch.int64)
                        for t in self._get_data_loader(
                            datapipe, multiprocessing_context=ctx, **dl_common_args
                        )
                    ],
                )

    @skipIfNoNumpy
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_multiprocessing_iterdatapipe(self):
        # [FUNCTION DOC] Runs the datapipe multiprocessing check without dill. Needs NumPy and
        # CUDA IPC.
        self._test_multiprocessing_iterdatapipe(with_dill=False)

    @unittest.expectedFailure
    @skipIfNoNumpy
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @skipIfNoDill
    def test_multiprocessing_iterdatapipe_with_dill(self):
        # [FUNCTION DOC] Same check using a lambda (needs dill); marked expected failure. Needs
        # NumPy, CUDA IPC and dill.
        self._test_multiprocessing_iterdatapipe(with_dill=True)

    def test_worker_seed(self):
        # [FUNCTION DOC] Checks each of six workers gets a distinct initial seed, using a dataset
        # with a barrier so every worker handles an item.
        num_workers = 6
        batch_size = 1
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)
        dataloader = self._get_data_loader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )
        seeds = set()
        seeds.update(batch[0] for batch in dataloader)
        self.assertEqual(len(seeds), num_workers)

    def test_worker_seed_reproducibility(self):
        # [FUNCTION DOC] Checks that two loaders created with the same seeded generator give the
        # same set of worker seeds.
        def get_dataloader():
            # [FUNCTION DOC] HELPER: builds a loader over the enclosing dataset using the
            # enclosing batch size and worker count and a generator seeded with 42.
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(42),
            )

        num_workers = 6
        batch_size = 1
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)
        self.assertEqual(
            {int(batch) for batch in get_dataloader()},
            {int(batch) for batch in get_dataloader()},
        )

    def test_multi_epochs_reproducibility(self):
        # [FUNCTION DOC] Checks over three epochs that each batch of the multi epoch iterable
        # dataset contains the id of the expected worker repeated batch size times.
        num_workers = 2
        batch_size = 10
        num_epochs = 3

        dataset = TestMultiEpochDataset(batch_size * num_workers)
        dataloader = self._get_data_loader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        for _ in range(num_epochs):
            for batch_idx, sample in enumerate(dataloader):
                self.assertEqual(
                    sample.tolist(), [batch_idx % num_workers] * batch_size
                )

    def test_worker_init_fn(self):
        # [FUNCTION DOC] Checks the worker_init_fn that seeds with a constant makes every item of
        # a seed dataset equal that constant.
        dataset = SeedDataset(4)
        dataloader = self._get_data_loader(
            dataset, batch_size=2, num_workers=2, worker_init_fn=init_fn
        )
        for batch in dataloader:
            self.assertEqual(12345, batch[0])
            self.assertEqual(12345, batch[1])

    @unittest.skipIf(
        IS_WINDOWS or IS_MACOS,
        "`ValueError: cannot find context for 'forkserver'` in Windows",
    )
    def test_worker_init_fn_forkserver(self):
        # [FUNCTION DOC] Sets the global start method to forkserver (force) and uses a locally
        # defined init function, which cannot be pickled; expects a UserWarning about a pickle
        # error and an exception on the first batch. Side effect: changes the global
        # multiprocessing start method. Skipped on Windows and macOS.
        def local_init_fn(worker_id):
            # [FUNCTION DOC] HELPER: worker init function that seeds with a constant; the id is
            # unused.
            torch.manual_seed(12345)

        import multiprocessing as py_mp

        py_mp.set_start_method("forkserver", force=True)

        dataset = SeedDataset(4)
        dataloader = self._get_data_loader(
            dataset, batch_size=2, num_workers=2, worker_init_fn=local_init_fn
        )
        with self.assertWarnsRegex(UserWarning, "Got pickle error when"):
            with self.assertRaises(Exception):
                next(iter(dataloader))

    def test_get_worker_info(self):
        # [FUNCTION DOC] Runs the worker info target in a child process and checks it exits with
        # code zero.
        p = ErrorTrackingProcess(target=_test_get_worker_info)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertEqual(p.exitcode, 0)
        finally:
            p.terminate()

    def test_shuffle(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled loader.
        self._test_shuffle(self._get_data_loader(self.dataset, shuffle=True))

    def test_shuffle_batch_none(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled loader with batch_size None.
        self._test_shuffle(DataLoader(self.dataset, batch_size=None, shuffle=True))

    def test_shuffle_batch(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled loader with batch size two.
        self._test_shuffle(
            self._get_data_loader(self.dataset, batch_size=2, shuffle=True)
        )

    def test_shuffle_reproducibility(self):
        # [FUNCTION DOC] Checks that two loaders with the same seeded generator give identical
        # shuffled results, with zero and two workers.
        for fn in (
            lambda: DataLoader(
                self.dataset,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: DataLoader(
                self.dataset,
                shuffle=True,
                num_workers=2,
                generator=torch.Generator().manual_seed(42),
            ),
        ):
            self.assertEqual(list(fn()), list(fn()))

    def test_sequential_workers(self):
        # [FUNCTION DOC] Runs the sequential check on a four worker loader.
        self._test_sequential(self._get_data_loader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        # [FUNCTION DOC] Runs the sequential check on a four worker loader with batch size two.
        self._test_sequential(
            self._get_data_loader(self.dataset, batch_size=2, num_workers=4)
        )

    def test_seqential_batch_workers_prefetch(self):
        # [FUNCTION DOC] Runs the sequential check on a four worker batched loader with prefetch
        # factor three.
        self._test_sequential(
            DataLoader(self.dataset, batch_size=2, num_workers=4, prefetch_factor=3)
        )

    def test_shuffle_workers(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled four worker loader.
        self._test_shuffle(
            self._get_data_loader(self.dataset, shuffle=True, num_workers=4)
        )

    def test_shuffle_batch_workers(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled four worker loader with batch size
        # two.
        self._test_shuffle(
            self._get_data_loader(
                self.dataset, batch_size=2, shuffle=True, num_workers=4
            )
        )

    def test_shuffle_batch_workers_prefetch(self):
        # [FUNCTION DOC] Runs the shuffle check on a shuffled batched four worker loader with
        # prefetch factor three.
        self._test_shuffle(
            DataLoader(
                self.dataset,
                batch_size=2,
                shuffle=True,
                num_workers=4,
                prefetch_factor=3,
            )
        )

    def test_random_sampler(self):
        # [FUNCTION DOC] Checks RandomSampler statistics (repeated count, min, max, total) for
        # sampling with replacement, without replacement with default or larger or smaller sample
        # counts, and that a non boolean replacement argument raises TypeError.
        from collections import Counter

        from torch.utils.data import RandomSampler

        def sample_stat(sampler, num_samples):
            # [FUNCTION DOC] HELPER: counts how often each sampled index occurs and returns the
            # number of indices drawn more than once, smallest and largest index and the total
            # drawn. The sample count argument is unused.
            counts = Counter(sampler)
            count_repeated = sum(val > 1 for val in counts.values())
            return (
                count_repeated,
                min(counts.keys()),
                max(counts.keys()),
                sum(counts.values()),
            )

        # test sample with replacement
        n = len(self.dataset) + 1  # ensure at least one sample is drawn more than once
        sampler_with_replacement = RandomSampler(
            self.dataset, replacement=True, num_samples=n
        )
        count_repeated, minval, maxval, count_total = sample_stat(
            sampler_with_replacement, n
        )
        self.assertTrue(count_repeated > 0)
        self.assertTrue(minval >= 0)
        self.assertTrue(maxval < len(self.dataset))
        self.assertTrue(count_total == n)

        # test sample without replacement and without specified num_samples
        sampler_without_replacement = RandomSampler(self.dataset)
        count_repeated, minval, maxval, count_total = sample_stat(
            sampler_without_replacement, len(self.dataset)
        )
        self.assertTrue(count_repeated == 0)
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == len(self.dataset))

        # test sample without replacement and with specified num_samples
        n = len(self.dataset) * 2
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(
            sampler_without_replacement, len(self.dataset)
        )
        self.assertTrue(count_repeated == len(self.dataset))
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == n)

        n = len(self.dataset) - 1
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(
            sampler_without_replacement, len(self.dataset)
        )
        self.assertTrue(count_repeated == 0)
        self.assertTrue(minval >= 0)
        self.assertTrue(maxval < len(self.dataset))
        self.assertTrue(count_total == n)

        n = len(self.dataset) + 1
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(
            sampler_without_replacement, len(self.dataset)
        )
        self.assertTrue(count_repeated == 1)
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == n)

        # raise error when replacement is non-boolean
        with self.assertRaisesRegex(
            TypeError, "replacement should be a boolean value, but got replacement=0"
        ):
            RandomSampler(self.dataset, replacement=0)

    def test_random_sampler_len_with_replacement(self):
        # [FUNCTION DOC] Checks the length of a RandomSampler with replacement and extra samples
        # equals the sample count, matches the number of iterated items, and the loader length for
        # batch sizes one and six.
        from torch.utils.data import RandomSampler

        # add 5 extra samples
        num_samples = len(self.dataset) + 5
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=num_samples)
        # test len method
        self.assertEqual(num_samples, len(sampler))

        # test with iteration
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # test with dataloader, batch_size = 1
        batch_size = 1
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # test with dataloader, batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(
            int(math.ceil(float(num_samples) / batch_size)),
            count_num_samples_in_data_loader,
        )

    def test_random_sampler_len_without_replacement(self):
        # [FUNCTION DOC] Same as the with replacement variant but sampling without replacement
        # with more samples than the dataset has.
        from torch.utils.data import RandomSampler

        # add 5 extra samples
        num_samples = len(self.dataset) + 5
        sampler = RandomSampler(
            self.dataset, replacement=False, num_samples=num_samples
        )
        # test len method
        self.assertEqual(num_samples, len(sampler))

        # test with iteration
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # test with dataloader, batch_size = 1
        batch_size = 1
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # test with dataloader, batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(
            self._get_data_loader(self.dataset, batch_size=batch_size, sampler=sampler)
        )
        self.assertEqual(
            num_samples // batch_size + (num_samples % batch_size > 0),
            count_num_samples_in_data_loader,
        )

    def test_distributed_sampler_invalid_rank(self):
        # [FUNCTION DOC] Checks DistributedSampler raises ValueError mentioning an invalid rank
        # when the rank equals the replica count or is negative.
        from torch.utils.data.distributed import DistributedSampler

        dataset = torch.IntTensor(range(10))
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, 3)

        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, -1)

    def test_duplicating_data_with_drop_last(self):
        # [FUNCTION DOC] Checks that with drop_last and four simulated processes using
        # DistributedSampler over nine items, the data seen across all ranks has no duplicates.
        from torch.utils.data.distributed import DistributedSampler

        num_processes = 4
        num_batches = 9
        data_set = torch.IntTensor(range(num_batches))
        scanned_data = torch.IntTensor([])
        for i in range(num_processes):
            s = DistributedSampler(data_set, num_processes, i)
            d_loader = self._get_data_loader(
                data_set,
                batch_size=int(num_batches / num_processes),
                drop_last=True,
                sampler=s,
            )
            for data in d_loader:
                scanned_data = torch.cat((scanned_data, data), 0)

        self.assertEqual(scanned_data.size(), scanned_data.unique().size())

    def test_sampler_reproducibility(self):
        # [FUNCTION DOC] Checks sampler reproducibility: two samplers built with the same seeded
        # generator (random with and without replacement, weighted with and without replacement,
        # subset random) give equal sequences; and for samplers without generator, reseeding the
        # global RNG before iterating gives equal sequences both for consecutive full passes and
        # for two interleaved iterators.
        from torch.utils.data import (
            RandomSampler,
            SubsetRandomSampler,
            WeightedRandomSampler,
        )

        weights = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
        for fn in (
            lambda: RandomSampler(
                self.dataset,
                num_samples=5,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: RandomSampler(
                self.dataset,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: WeightedRandomSampler(
                weights,
                num_samples=5,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: WeightedRandomSampler(
                weights,
                num_samples=5,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            ),
            lambda: SubsetRandomSampler(
                range(10), generator=torch.Generator().manual_seed(42)
            ),
        ):
            self.assertEqual(list(fn()), list(fn()))

        for sampler in (
            RandomSampler(self.dataset, num_samples=5, replacement=True),
            RandomSampler(self.dataset, replacement=False),
            WeightedRandomSampler(weights, num_samples=5, replacement=True),
            WeightedRandomSampler(weights, num_samples=5, replacement=False),
            SubsetRandomSampler(range(10)),
        ):
            torch.manual_seed(0)
            l1 = list(sampler) + list(sampler)

            torch.manual_seed(0)
            l2 = list(sampler) + list(sampler)
            self.assertEqual(l1, l2)

            its = (iter(sampler), iter(sampler))
            ls = ([], [])
            for idx in range(len(sampler)):
                for i in range(2):
                    if idx == 0:
                        torch.manual_seed(0)
                    ls[i].append(next(its[i]))
            self.assertEqual(ls[0], ls[1])

    def _test_sampler(self, **kwargs):
        # [FUNCTION DOC] HELPER: builds a loader (batch size two, extra keyword arguments
        # forwarded) with a plain range from two to twelve as sampler; asserts length five and
        # each batch equals the expected slice of the data.
        indices = range(2, 12)  # using a regular iterable
        dl = self._get_data_loader(
            self.dataset, sampler=indices, batch_size=2, **kwargs
        )
        self.assertEqual(len(dl), 5)
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 2)
            self.assertEqual(input, self.data[i * 2 + 2 : i * 2 + 4])

    def test_sampler(self):
        # [FUNCTION DOC] Runs the sampler check in-process and with four workers, then a third
        # call.
        # KNOWN ISSUE: the third call uses the batch sampler helper (with the spawn context)
        # instead of the sampler helper, so the spawn variant of the plain sampler path is never
        # checked.
        self._test_sampler()
        self._test_sampler(num_workers=4)
        self._test_batch_sampler(num_workers=4, multiprocessing_context="spawn")

    def _test_batch_sampler(self, **kwargs):
        # [FUNCTION DOC] HELPER: builds a list of tuple batches alternating sizes two and three
        # over the first twenty indices and uses it as batch_sampler; asserts loader length eight
        # and that each batch has the expected size and equals the matching slice of the data.
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []  # using a regular iterable
        for i in range(0, 20, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = self._get_data_loader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 8)
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                self.assertEqual(len(input), 2)
                self.assertEqual(input, self.data[offset : offset + 2])
            else:
                offset = i * 5 // 2
                self.assertEqual(len(input), 3)
                self.assertEqual(input, self.data[offset : offset + 3])

    def test_batch_sampler(self):
        # [FUNCTION DOC] Runs the batch sampler check in-process, with four workers, and with four
        # workers under the spawn context.
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)
        self._test_batch_sampler(num_workers=4, multiprocessing_context="spawn")

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_shuffle_pin_memory(self):
        # [FUNCTION DOC] Checks a shuffled four worker loader with pin_memory yields pinned inputs
        # and targets. Skipped without accelerator pinned memory support.
        loader = self._get_data_loader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy(self):
        # [FUNCTION DOC] Checks a dataset returning NumPy arrays is collated into a double tensor
        # of shape twelve by two by three by four. Skipped without NumPy.
        import numpy as np

        class TestDataset(torch.utils.data.Dataset):
            # [CLASS DOC] HELPER: dataset of 1000 items, each a NumPy array of ones times the
            # index.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: returns a NumPy array of ones with shape two by three by
                # four scaled by i.
                return np.ones((2, 3, 4)) * i

            def __len__(self):
                # [FUNCTION DOC] HELPER: always returns 1000.
                return 1000

        loader = self._get_data_loader(TestDataset(), batch_size=12)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.DoubleTensor)
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_gen_state(self):
        # [FUNCTION DOC] Checks the worker state generation helper reproduces reference four word
        # states produced by NumPy for a table of base seed and worker id pairs. Skipped without
        # NumPy.
        from torch.utils.data._utils.worker import _generate_state

        # Using NumPy generated states as the reference to test `_generate_state`
        # having the same result.
        # Test case: ((worker_id, base_seed), expected_state)
        test_cases = [
            (
                (4, 13434589827475259383),
                (2884386318, 1088094898, 3523808998, 3860348662),
            ),
            ((1, 15014285634777110771), (1934848465, 763213760, 2959016433, 179751970)),
            (
                (10, 978296274032934101),
                (1759791917, 3550927336, 1225977135, 1036538043),
            ),
            (
                (12, 11868770762134256968),
                (3974661794, 3331131333, 3630387033, 2885815368),
            ),
            (
                (9, 15378787925219019706),
                (3815056996, 3162224466, 2735102421, 3190253477),
            ),
            ((5, 9055612723125076328), (3522565701, 3368424109, 959377806, 621878693)),
            (
                (15, 14617792358407278405),
                (3402479508, 1588702753, 1169536393, 3675067356),
            ),
            (
                (9, 17363320784006640087),
                (957989458, 2518334477, 1421725660, 3086155459),
            ),
            (
                (12, 480002904169484764),
                (2732851467, 1762620729, 4055801988, 1277640511),
            ),
            (
                (15, 16803975943592702950),
                (3479415043, 4022359553, 295994005, 3358606349),
            ),
            (
                (9, 11704776406047813044),
                (1968928009, 710113752, 2442656196, 1587420279),
            ),
            (
                (10, 16357891985431864516),
                (1271733898, 4197047399, 3727213786, 2338547348),
            ),
            (
                (2, 17423369006318065007),
                (544294336, 1911284083, 3299147734, 3231058347),
            ),
            ((2, 2889492011444113593), (3721591783, 2595811276, 2212881745, 977682627)),
            ((0, 8979703111668486195), (4276723937, 2556068849, 2962827292, 233130238)),
            (
                (6, 6269787272229682235),
                (2548857855, 1216457374, 1012973562, 2999759647),
            ),
        ]

        for (worker_id, base_seed), exp in test_cases:
            self.assertEqual(exp, _generate_state(base_seed, worker_id))

    def test_error(self):
        # [FUNCTION DOC] Runs the error check on a shuffled batched loader over the error dataset.
        self._test_error(
            self._get_data_loader(ErrorDataset(100), batch_size=2, shuffle=True)
        )

    def test_error_workers(self):
        # [FUNCTION DOC] Runs the error check on the same kind of loader with four workers.
        self._test_error(
            self._get_data_loader(
                ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4
            )
        )

    def test_subset_custom_getitem(self):
        # [FUNCTION DOC] Regression test for issue 163184: a Subset subclass overriding item
        # access (and batched access) with a doubling transform must have that transform applied
        # when loaded by a DataLoader; checks direct indexing and the batch values.
        """
        Regression test for issue #163184 where DataLoader ignores custom
        transformations in Subset subclasses that override __getitem__.
        """

        class SimpleDataset(Dataset):
            # [CLASS DOC] HELPER: dataset of the integers zero to nineteen.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the data tensor.
                self.data = torch.arange(20)

            def __len__(self):
                # [FUNCTION DOC] HELPER: returns 20.
                return 20

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: returns the element at the index.
                return self.data[idx]

        class TransformSubset(Subset):
            # [CLASS DOC] HELPER: Subset subclass whose items are doubled values of the underlying
            # dataset item at the mapped index; the batched method applies the same per-item
            # function.
            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: maps the index through the stored indices, reads the base
                # item and returns it times two.
                # transform: multiply by 2
                original_idx = self.indices[idx]
                value = self.dataset[original_idx]
                return value * 2

            def __getitems__(self, indices):
                # [FUNCTION DOC] HELPER: returns the transformed item for each requested index.
                return [self.__getitem__(idx) for idx in indices]

        dataset = SimpleDataset()
        subset = TransformSubset(dataset, [0, 1, 2, 3])

        self.assertEqual(subset[0].item(), 0)
        self.assertEqual(subset[1].item(), 2)
        self.assertEqual(subset[2].item(), 4)

        loader = self._get_data_loader(subset, batch_size=2, shuffle=False)
        batches = list(loader)

        self.assertTrue(torch.equal(batches[0], torch.tensor([0, 2])))
        self.assertTrue(torch.equal(batches[1], torch.tensor([4, 6])))

    def test_subset_custom_getitem_with_tuple_data(self):
        # [FUNCTION DOC] Same regression as above for a dataset of tuples: a Subset subclass
        # returning the sum of the tuple parts must be honored by the loader.
        """Test Subset custom __getitem__ with tuple data (like the original issue)."""

        class TupleDataset(Dataset):
            # [CLASS DOC] HELPER: dataset of ten pairs of values from two aranges (zero to nine
            # and 100 to 109).
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the two tensors.
                self.a = torch.arange(10)
                self.b = torch.arange(100, 110)

            def __len__(self):
                # [FUNCTION DOC] HELPER: returns 10.
                return 10

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: returns the pair of values at the index.
                return self.a[idx], self.b[idx]

        class SumSubset(Subset):
            # [CLASS DOC] Subset subclass whose items are the sum of the pair from the underlying
            # dataset.
            """Subset that returns sum instead of tuple"""

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: maps the index, reads the pair and returns the sum of its
                # two parts.
                original_idx = self.indices[idx]
                a, b = self.dataset[original_idx]
                return a + b

            def __getitems__(self, indices):
                # [FUNCTION DOC] HELPER: returns the summed item for each requested index.
                return [self.__getitem__(idx) for idx in indices]

        dataset = TupleDataset()
        subset = SumSubset(dataset, [0, 1, 2, 3])

        self.assertEqual(subset[0].item(), 100)
        self.assertEqual(subset[1].item(), 102)

        loader = self._get_data_loader(subset, batch_size=2, shuffle=False)
        batches = list(loader)

        self.assertTrue(torch.equal(batches[0], torch.tensor([100, 102])))
        self.assertTrue(torch.equal(batches[1], torch.tensor([104, 106])))

    def test_subset_override_getitem_requires_getitems(self):
        # [FUNCTION DOC] Checks that subclassing Subset and overriding only item access, without
        # batched access, raises NotImplementedError at construction whose message names the class
        # and both methods.
        """
        Test that subclassing Subset and overriding only __getitem__ without
        __getitems__ raises a NotImplementedError at instantiation time.
        """

        class SimpleDataset(Dataset):
            # [CLASS DOC] HELPER: dataset of the integers zero to nine.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the data tensor.
                self.data = torch.arange(10)

            def __len__(self):
                # [FUNCTION DOC] HELPER: returns 10.
                return 10

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: returns the element at the index.
                return self.data[idx]

        class IncompleteSubset(Subset):
            # [CLASS DOC] HELPER: Subset subclass that overrides only item access (doubling),
            # deliberately missing the batched method, so construction is expected to fail.
            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: maps the index, reads the base item and returns it times
                # two.
                original_idx = self.indices[idx]
                return self.dataset[original_idx] * 2

        dataset = SimpleDataset()

        with self.assertRaises(NotImplementedError) as cm:
            subset = IncompleteSubset(dataset, [0, 1, 2, 3])

        error_message = str(cm.exception)
        self.assertIn("IncompleteSubset", error_message)
        self.assertIn("__getitem__", error_message)
        self.assertIn("__getitems__", error_message)

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    def test_partial_workers(self):
        # [FUNCTION DOC] Checks workers (and the pin memory thread when used) terminate after the
        # iterator is deleted before exhaustion: reads eleven batches from a four worker loader,
        # then deletes it and joins all workers, asserting none is alive. Skipped on Windows.
        r"""Check that workers exit even if the iterator is not exhausted."""
        if TEST_PIN_MEMORY:
            pin_memory_configs = (True, False)
        else:
            pin_memory_configs = (False,)

        for pin_memory in pin_memory_configs:
            loader = iter(
                self._get_data_loader(
                    self.dataset, batch_size=2, num_workers=4, pin_memory=pin_memory
                )
            )
            workers = loader._workers
            if pin_memory:
                pin_memory_thread = loader._pin_memory_thread
            for i, _ in enumerate(loader):
                if i == 10:
                    break
            if i != 10:
                raise AssertionError(f"Expected to stop at i=10, got i={i}")
            del loader
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive(), "subprocess not terminated")
            if pin_memory:
                pin_memory_thread.join(JOIN_TIMEOUT)
                self.assertFalse(pin_memory_thread.is_alive())

    # Takes 2.5min to finish, see https://github.com/pytorch/pytorch/issues/46065
    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    @slowTest
    def test_proper_exit(self):
        # [FUNCTION DOC] Slow test (about 2.5 minutes, needs psutil) of orderly shutdown. Iterates
        # over combinations of iterable or map dataset, workers or not, pin_memory and holding an
        # iterator reference, and over exit methods (clean, loader error, worker error, loader
        # kill without workers; worker kill is skipped as flaky). Runs the proper-exit target in
        # an error tracking process, waits for its setup event, signals the tester event, then
        # asserts that the loader and all its worker children terminate within timeouts and that
        # the exit code and exception match the exit method, reporting process diagnostics on
        # failure. Always terminates the loader process.
        """There might be ConnectionResetError or leaked semaphore warning
        (due to dirty process exit), but they are all safe to ignore"""

        # TODO: test the case where the pin_memory_thread triggers an
        #       error/fatal signal. I haven't found out how to properly do that.

        for (
            is_iterable_dataset,
            use_workers,
            pin_memory,
            hold_iter_reference,
        ) in itertools.product([True, False], repeat=4):
            # `hold_iter_reference` specifies whether we hold a reference to the
            # iterator. This is interesting because Python3 error traces holds a
            # reference to the frames, which hold references to all the local
            # variables including the iterator, and then the iterator dtor may
            # not be called before process end. It is important to see that the
            # processes still exit in both cases.

            if pin_memory and (not TEST_PIN_MEMORY or IS_WINDOWS):
                # This test runs in a subprocess, which can only initialize the
                # accelerator with spawn. DataLoader with pin_memory=True initializes
                # the accelerator when its iterator is constructed.
                # For windows, pin_memory sometimes causes OOM.
                continue

            # `exit_method` controls the way the loader process ends.
            #   - `*_kill` means that `*` is killed by OS.
            #   - `*_error` means that `*` raises an error.
            #   - `None` means that no error happens.
            # In all cases, all processes should end properly.
            if use_workers:
                # TODO: Fix test for 'loader_kill' that would cause running out of shared memory.
                # Killing loader process would prevent DataLoader iterator clean up all queues
                # and worker processes
                exit_methods = [None, "loader_error", "worker_error", "worker_kill"]
                persistent_workers = self.persistent_workers
            else:
                exit_methods = [None, "loader_error", "loader_kill"]
                persistent_workers = False

            for exit_method in exit_methods:
                if exit_method == "worker_kill":
                    # FIXME: This sometimes hangs. See #16608.
                    continue

                desc = []
                desc.append(f"is_iterable_dataset={is_iterable_dataset}")
                desc.append(f"use_workers={use_workers}")
                desc.append(f"pin_memory={pin_memory}")
                desc.append(f"hold_iter_reference={hold_iter_reference}")
                desc.append(f"exit_method={exit_method}")
                desc = "test_proper_exit with " + ", ".join(desc)

                # Event that the loader process uses to signal testing process
                # that various things are setup, including that the worker pids
                # are specified in `worker_pids` array.
                loader_setup_event = mp.Event()

                # Event that this process has finished setting up, and the
                # loader process can now proceed to trigger error events or
                # finish normally.
                tester_setup_event = mp.Event()

                loader_p = ErrorTrackingProcess(
                    target=_test_proper_exit,
                    args=(
                        is_iterable_dataset,
                        use_workers,
                        pin_memory,
                        exit_method,
                        hold_iter_reference,
                        loader_setup_event,
                        tester_setup_event,
                        persistent_workers,
                    ),
                    disable_stderr=False,
                )
                loader_p.start()
                loader_psutil_p = psutil.Process(loader_p.pid)

                # Wait for loader process to set everything up, e.g., starting
                # workers.
                loader_setup_event.wait(timeout=JOIN_TIMEOUT)
                if not loader_setup_event.is_set():
                    fail_msg = (
                        desc + ": loader process failed to setup within given time"
                    )
                    if loader_p.exception is not None:
                        fail_msg += f", and had exception {loader_p.exception}"
                    elif not loader_p.is_alive():
                        fail_msg += f", and exited with code {loader_p.exitcode} but had no exception"
                    else:
                        fail_msg += ", and is still alive."
                    if loader_p.is_alive():
                        # this may kill the process, needs to run after the above lines
                        loader_p.print_traces_of_all_threads()
                    self.fail(fail_msg)

                # We are certain that the workers have started now.
                worker_psutil_ps = loader_psutil_p.children()

                def fail(reason):
                    # [FUNCTION DOC] HELPER: builds a failure message from the combination
                    # description, the reason and psutil diagnostics of the loader and worker
                    # processes (dumping thread traces of those still running), then fails the
                    # test with it.
                    report_psutil_attrs = [
                        "pid",
                        "name",
                        "cpu_times",
                        "io_counters",
                        "memory_full_info",
                        "num_ctx_switches",
                        "open_files",
                        "threads",
                        "status",
                        "nice",
                        "ionice",
                    ]
                    if reason is None:
                        err_msg = desc
                    else:
                        err_msg = f"{desc}: {reason}"
                    err_msg += "\nLoader info:\n\t"
                    if loader_psutil_p.is_running():
                        err_msg += str(
                            loader_psutil_p.as_dict(attrs=report_psutil_attrs)
                        )
                        # this may kill the process, needs to run after the above line
                        loader_p.print_traces_of_all_threads()
                    else:
                        err_msg += f"exited with code {loader_p.exitcode}"
                    if use_workers:
                        err_msg += "\nWorker(s) info:"
                        for idx, worker_psutil_p in enumerate(worker_psutil_ps):
                            err_msg += f"\n\tWorker {idx}:\n\t\t"
                            if worker_psutil_p.is_running():
                                err_msg += str(
                                    worker_psutil_p.as_dict(attrs=report_psutil_attrs)
                                )
                                # this may kill the process, needs to run after the above line
                                print_traces_of_all_threads(worker_psutil_p.pid)
                            else:
                                err_msg += "exited with unknown code"
                    self.fail(err_msg)

                tester_setup_event.set()

                try:
                    loader_p.join(JOIN_TIMEOUT + MP_STATUS_CHECK_INTERVAL)
                    if loader_p.is_alive():
                        fail_reason = "loader process did not terminate"
                        if loader_p.exception is not None:
                            fail(
                                fail_reason
                                + f", and had exception {loader_p.exception}"
                            )
                        else:
                            fail(fail_reason + ", and had no exception")
                    _, alive = psutil.wait_procs(
                        worker_psutil_ps,
                        timeout=(MP_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT),
                    )
                    if len(alive) > 0:
                        fail(
                            "worker process (pid(s) {}) did not terminate".format(
                                ", ".join(str(p.pid) for p in alive)
                            )
                        )
                    if exit_method is None:
                        if loader_p.exitcode != 0:
                            fail(
                                f"loader process had nonzero exitcode {loader_p.exitcode}"
                            )
                    else:
                        if loader_p.exitcode == 0:
                            fail("loader process had zero exitcode")
                        if exit_method == "loader_error":
                            if not isinstance(
                                loader_p.exception, RuntimeError
                            ) or "Loader error" not in str(loader_p.exception):
                                fail(
                                    f"loader process did not raise expected exception, but had {loader_p.exception}"
                                )
                        elif exit_method == "worker_kill":
                            if isinstance(loader_p.exception, RuntimeError):
                                if "DataLoader worker (pid" not in str(
                                    loader_p.exception
                                ):
                                    fail(
                                        f"loader process did not raise expected exception, but had {loader_p.exception}"
                                    )
                            elif isinstance(loader_p.exception, ConnectionRefusedError):
                                # Sometimes, when the worker is being killed and is freeing its
                                # resources, the unpickling in loader process will be met an
                                # a `ConnectionRefusedError` as it can not open a socket to receive
                                # resource. In such cases, the worker may not have fully exited,
                                # and the loader can't know this via `is_alive` check or `SIGCHLD`
                                # handler. So we permit this as an allowed error as well.
                                # After all, we are happy as long as it terminates.
                                pass
                            else:
                                fail(
                                    f"loader process did not raise expected exception, but had {loader_p.exception}"
                                )
                        elif exit_method == "worker_error":
                            if not isinstance(
                                loader_p.exception, RuntimeError
                            ) or "Worker error" not in str(loader_p.exception):
                                fail(
                                    f"loader process did not raise expected exception, but had {loader_p.exception}"
                                )
                finally:
                    loader_p.terminate()

    def test_len(self):
        # [FUNCTION DOC] Checks len of a dataset and of loaders with batch sizes two and three (50
        # and 34 batches), and that iteration count equals len.
        def check_len(dl, expected):
            # [FUNCTION DOC] HELPER: asserts the length of the object equals the expected value
            # and that iterating it yields that many items.
            self.assertEqual(len(dl), expected)
            n = 0
            for _ in dl:
                n += 1
            self.assertEqual(n, expected)

        check_len(self.dataset, 100)
        check_len(self._get_data_loader(self.dataset, batch_size=2), 50)
        check_len(self._get_data_loader(self.dataset, batch_size=3), 34)

    def test_iterabledataset_len(self):
        # [FUNCTION DOC] Checks loader length for an iterable dataset defining a length of ten,
        # for batch sizes one, two and three with and without drop_last (rounded up unless
        # drop_last).
        class IterableDataset(torch.utils.data.IterableDataset):
            # [CLASS DOC] HELPER: iterable dataset of the integers zero to nine that reports
            # length ten.
            def __len__(self):
                # [FUNCTION DOC] HELPER: returns 10.
                return 10

            def __iter__(self):
                # [FUNCTION DOC] HELPER: returns an iterator over a range of ten.
                return iter(range(10))

        iterable_loader = DataLoader(IterableDataset(), batch_size=1)
        self.assertEqual(len(iterable_loader), 10)
        iterable_loader = DataLoader(IterableDataset(), batch_size=1, drop_last=True)
        self.assertEqual(len(iterable_loader), 10)

        iterable_loader = DataLoader(IterableDataset(), batch_size=2)
        self.assertEqual(len(iterable_loader), 5)
        iterable_loader = DataLoader(IterableDataset(), batch_size=2, drop_last=True)
        self.assertEqual(len(iterable_loader), 5)

        iterable_loader = DataLoader(IterableDataset(), batch_size=3)
        self.assertEqual(len(iterable_loader), 4)
        iterable_loader = DataLoader(IterableDataset(), batch_size=3, drop_last=True)
        self.assertEqual(len(iterable_loader), 3)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_scalars(self):
        # [FUNCTION DOC] Checks NumPy scalar types (eight float and integer types) returned by a
        # dataset are collated into the matching torch tensor types. Skipped without NumPy.
        import numpy as np

        class ScalarDataset(torch.utils.data.Dataset):
            # [CLASS DOC] HELPER: dataset of four items, each a fresh zero scalar of the given
            # NumPy type.
            def __init__(self, dtype):
                # [FUNCTION DOC] HELPER: stores the scalar type.
                self.dtype = dtype

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: returns a newly constructed scalar of the stored type.
                return self.dtype()

            def __len__(self):
                # [FUNCTION DOC] HELPER: returns 4.
                return 4

        dtypes = {
            np.float64: torch.DoubleTensor,
            np.float32: torch.FloatTensor,
            np.float16: torch.HalfTensor,
            np.int64: torch.LongTensor,
            np.int32: torch.IntTensor,
            np.int16: torch.ShortTensor,
            np.int8: torch.CharTensor,
            np.uint8: torch.ByteTensor,
        }
        for dt, tt in dtypes.items():
            dset = ScalarDataset(dt)
            loader = self._get_data_loader(dset, batch_size=2)
            batch = next(iter(loader))
            self.assertIsInstance(batch, tt)

    def test_default_convert_mapping_keep_type(self):
        # [FUNCTION DOC] Checks default_convert on a dict subclass returns an equal mapping.
        data = CustomDict({"a": 1, "b": 2})
        converted = dataloader.default_convert(data)

        self.assertEqual(converted, data)

    def test_default_convert_sequence_keep_type(self):
        # [FUNCTION DOC] Checks default_convert on a list subclass returns an equal sequence.
        data = CustomList([1, 2, 3])
        converted = dataloader.default_convert(data)

        self.assertEqual(converted, data)

    def test_default_convert_sequence_dont_keep_type(self):
        # [FUNCTION DOC] Checks default_convert on a range yields a plain list of its values.
        data = range(2)
        converted = dataloader.default_convert(data)

        self.assertEqual(converted, [0, 1])

    def test_default_collate_dtype(self):
        # [FUNCTION DOC] Checks default_collate dtype results: ints become int64, floats become
        # float64, bools become bool, and a list of strings is returned unchanged.
        arr = [1, 2, -1]
        collated = dataloader.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.int64)

        arr = [1.1, 2.3, -0.9]
        collated = dataloader.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr, dtype=torch.float64))

        arr = [True, False]
        collated = dataloader.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.bool)

        # Should be a no-op
        arr = ["a", "b", "c"]
        self.assertEqual(arr, dataloader.default_collate(arr))

    def test_default_collate_mapping_keep_type(self):
        # [FUNCTION DOC] Checks default_collate of dict subclass samples returns an equal dict
        # subclass with stacked tensors per key.
        batch = [CustomDict({"a": 1, "b": 2}), CustomDict({"a": 3, "b": 4})]
        collated = dataloader.default_collate(batch)

        expected = CustomDict({"a": torch.tensor([1, 3]), "b": torch.tensor([2, 4])})
        self.assertEqual(collated, expected)

    def test_default_collate_sequence_keep_type(self):
        # [FUNCTION DOC] Checks default_collate of list subclass samples returns an equal list
        # subclass with a stacked tensor per position.
        batch = [CustomList([1, 2, 3]), CustomList([4, 5, 6])]
        collated = dataloader.default_collate(batch)

        expected = CustomList(
            [
                torch.tensor([1, 4]),
                torch.tensor([2, 5]),
                torch.tensor([3, 6]),
            ]
        )
        self.assertEqual(collated, expected)

    def test_default_collate_sequence_dont_keep_type(self):
        # [FUNCTION DOC] Checks default_collate of ranges returns a plain list of stacked tensors.
        batch = [range(2), range(2)]
        collated = dataloader.default_collate(batch)

        self.assertEqual(collated, [torch.tensor([0, 0]), torch.tensor([1, 1])])

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_bad_numpy_types(self):
        # [FUNCTION DOC] Checks default_collate on NumPy string arrays passes the array through,
        # and raises TypeError for nested string arrays and for object arrays. Skipped without
        # NumPy.
        import numpy as np

        # Should be a no-op
        arr = np.array(["a", "b", "c"])
        self.assertEqual(arr, dataloader.default_collate(arr))

        arr = np.array([[["a", "b", "c"]]])
        self.assertRaises(TypeError, lambda: dataloader.default_collate(arr))

        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: dataloader.default_collate(arr))

        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: dataloader.default_collate(arr))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_numpy_memmap(self):
        # [FUNCTION DOC] Checks default_collate converts a list of rows of a memory mapped NumPy
        # file (temporary file) into the expected tensor. Skipped without NumPy.
        import numpy as np

        with tempfile.TemporaryFile() as f:
            arr = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
            arr_memmap = np.memmap(f, dtype=arr.dtype, mode="w+", shape=arr.shape)
            arr_memmap[:] = arr[:]
            arr_new = np.memmap(f, dtype=arr.dtype, mode="r", shape=arr.shape)
            tensor = dataloader.default_collate(list(arr_new))

        self.assertTrue(
            (tensor == tensor.new_tensor([[0, 1], [2, 3], [4, 5], [6, 7]])).all().item()
        )

    def test_default_collate_bad_sequence_type(self):
        # [FUNCTION DOC] Checks default_collate raises RuntimeError when sequences in a batch have
        # different lengths, in either order.
        batch = [["X"], ["X", "X"]]
        self.assertRaises(RuntimeError, lambda: dataloader.default_collate(batch))
        self.assertRaises(RuntimeError, lambda: dataloader.default_collate(batch[::-1]))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_shared_tensor(self):
        # [FUNCTION DOC] Checks default_collate output is not in shared memory in the main
        # process, and is shared when the module worker info marker is faked non-null (to emulate
        # a worker), for both tensor and NumPy inputs; the marker is restored afterwards. Skipped
        # without NumPy.
        import numpy as np

        t_in = torch.zeros(1)
        n_in = np.zeros(1)

        self.assertEqual(t_in.is_shared(), False)

        self.assertEqual(dataloader.default_collate([t_in]).is_shared(), False)
        self.assertEqual(dataloader.default_collate([n_in]).is_shared(), False)

        # FIXME: fix the following hack that makes `default_collate` believe
        #        that it is in a worker process (since it tests
        #        `get_worker_info() != None`), even though it is not.
        old = _utils.worker._worker_info
        try:
            _utils.worker._worker_info = "x"
            self.assertEqual(dataloader.default_collate([t_in]).is_shared(), True)
            self.assertEqual(dataloader.default_collate([n_in]).is_shared(), True)
        finally:
            _utils.worker._worker_info = old

    def test_excessive_thread_creation_warning(self):
        # [FUNCTION DOC] Checks constructing a loader with 1000 workers emits a UserWarning about
        # excessive worker creation.
        with self.assertWarnsRegex(
            UserWarning,
            r"excessive worker creation might get DataLoader running slow or even freeze",
        ):
            dataloader = DataLoader(self.dataset, batch_size=2, num_workers=1000)


class TestDataLoaderDeviceType(TestCase):
# [CLASS DOC] Device generic tests (instantiated per device type) for loading nested and sparse
# tensors through multiprocess loaders.
    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_nested_tensor_multiprocessing(self, device, context):
        # [FUNCTION DOC] For each non-default multiprocessing context (skipping fork on CUDA),
        # loads a list of ten nested tensors on the device with four workers and a clone collate
        # function, with and without pin_memory (CPU only), comparing each batch to the dataset
        # item. Then checks the default collate function raises RuntimeError saying nested tensors
        # are unsupported. Skipped without CUDA IPC.
        # The 'fork' multiprocessing context doesn't work for CUDA so skip it
        if "cuda" in device and context == "fork":
            self.skipTest(
                f"{context} multiprocessing context not supported for {device}"
            )

        dataset = [
            torch.nested.nested_tensor([torch.randn(5)], device=device)
            for _ in range(10)
        ]

        pin_memory_settings = [False]
        if device == "cpu" and TEST_PIN_MEMORY:
            pin_memory_settings.append(True)

        for pin_memory in pin_memory_settings:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                collate_fn=_clone_collate,
                pin_memory=pin_memory,
                multiprocessing_context=context,
            )

            for i, batch in enumerate(loader):
                self.assertEqual(batch[0], dataset[i])

        # Error case: default collate_fn doesn't currently support batches of nested tensors.
        # Following the current semantics, we'd need to stack them, which isn't possible atm.
        with self.assertRaisesRegex(
            RuntimeError, "not currently supported by the default collate_fn"
        ):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                multiprocessing_context=context,
            )

            next(iter(loader))

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/179979")
    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_sparse_tensor_multiprocessing(self, device, context):
        # [FUNCTION DOC] Same shape of test for a list of ten sparse COO tensors on the device:
        # for each non-default multiprocessing context (fork skipped on CUDA) loads with four
        # workers using the sparse clone collate function, with and without pin_memory (CPU only),
        # and compares each batch to the dataset item. Skipped without CUDA IPC and on ROCm.
        # The 'fork' multiprocessing context doesn't work for CUDA so skip it
        if "cuda" in device and context == "fork":
            self.skipTest(
                f"{context} multiprocessing context not supported for {device}"
            )

        dataset = [torch.randn(5, 5).to_sparse().to(device) for _ in range(10)]

        pin_memory_settings = [False]
        if device == "cpu" and TEST_PIN_MEMORY:
            pin_memory_settings.append(True)

        for pin_memory in pin_memory_settings:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                collate_fn=_sparse_coo_collate,
                pin_memory=pin_memory,
                multiprocessing_context=context,
            )

            for i, batch in enumerate(loader):
                self.assertEqual(batch[0], dataset[i])

    @onlyAccelerator
    def test_pin_memory_for_device(self, device):
        # [FUNCTION DOC] Checks a default loader with pin_memory enabled yields pinned tensors on
        # the current accelerator. Only instantiated for accelerator device types.
        # pin_memory pins to the current accelerator automatically;
        # @onlyAccelerator ensures this runs only when one is present.
        dataset = [torch.randn(3, 5) for _ in range(10)]
        loader = DataLoader(dataset, batch_size=2, pin_memory=True)
        for batch in loader:
            self.assertTrue(batch[0].is_pinned())


class IntegrationTestDataLoaderDataPipe(TestCase):
    # [CLASS DOC] Integration tests of datapipes used with the DataLoader.
    r"""
    Verify the behavior of a certain ``DataPipes`` with ``DataLoader``
    """

    def test_shuffler_iterdatapipe(self):
        # [FUNCTION DOC] Checks the shuffle datapipe is controlled by the loader: for several
        # buffer sizes, worker counts and persistent worker settings, results contain all items;
        # the same global seed before each epoch gives identical order; a different seed changes
        # the order but not the contents. Shuts down persistent workers of the iterator at the end
        # of each configuration.
        r"""
        Verify ``IterDataPipe.shuffle`` is controlled by ``DataLoader``
        to generate different seeds deterministically per epoch.
        """
        exp = list(range(100))

        def _create_dp(buffer_size):
            # [FUNCTION DOC] HELPER: wraps the enclosing list as an iterable datapipe, shuffles it
            # with the given buffer size and applies the sharding filter; returns the pipe.
            input_ds = dp.iter.IterableWrapper(exp)
            return input_ds.shuffle(buffer_size=buffer_size).sharding_filter()

        for bs in (5, 20, 33):
            # Test Deterministic
            for num_workers, pw in itertools.product((0, 1, 2), (True, False)):
                if num_workers == 0 and pw:
                    continue

                shuffle_dp = _create_dp(bs)

                mp_ctx = "spawn" if num_workers > 0 else None
                dl = DataLoader(
                    shuffle_dp,
                    num_workers=num_workers,
                    shuffle=True,
                    multiprocessing_context=mp_ctx,
                    persistent_workers=pw,
                )

                # No seed
                dl_res_ns = list(dl)
                self.assertEqual(sorted(dl_res_ns), exp)

                # Same seeds
                dl_res = []
                for _epoch in range(2):
                    torch.manual_seed(123)
                    dl_res.append(list(dl))
                self.assertEqual(dl_res[0], dl_res[1])
                self.assertEqual(sorted(dl_res[0]), exp)

                # Different seeds
                torch.manual_seed(321)
                dl_res.append(list(dl))

                self.assertEqual(len(dl_res[0]), len(dl_res[2]))
                self.assertNotEqual(dl_res[0], dl_res[2])
                self.assertEqual(sorted(dl_res[0]), sorted(dl_res[2]))

                if dl._iterator is not None:
                    dl._iterator._shutdown_workers()
                    dl._iterator = None
                del dl


class StringDataset(Dataset):
# [CLASS DOC] HELPER: dataset over the string of digits one to five; an item is a pair of the
# character and its index.
    def __init__(self) -> None:
    # [FUNCTION DOC] HELPER: stores the string.
        self.s = "12345"

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the string length.
        return len(self.s)

    def __getitem__(self, ndx):
    # [FUNCTION DOC] HELPER: returns a pair of the character and the index.
        return (self.s[ndx], ndx)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestStringDataLoader(TestCase):
    # [CLASS DOC] Tests loading datasets with string items.
    def setUp(self):
        # [FUNCTION DOC] Creates the string dataset.
        super().setUp()
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_shuffle_pin_memory(self):
        # [FUNCTION DOC] Checks a shuffled four worker loader with pin_memory returns strings
        # unchanged and pinned index tensors. Skipped without pinned memory support.
        loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        for s, n in loader:
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
# [CLASS DOC] HELPER: dataset of four items, each a mapping holding a 4 by 2 tensor filled with
# the index and a nested mapping with the integer index.
    def __len__(self):
    # [FUNCTION DOC] HELPER: returns 4.
        return 4

    def __getitem__(self, ndx):
    # [FUNCTION DOC] HELPER: builds the nested mapping for the index.
        return {
            "a_tensor": torch.empty(4, 2).fill_(ndx),
            "another_dict": {"a_number": ndx},
        }


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDictDataLoader(TestCase):
    # [CLASS DOC] Tests collation and pinning of mapping valued samples.
    def setUp(self):
        # [FUNCTION DOC] Creates the dict dataset.
        super().setUp()
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        # [FUNCTION DOC] Checks batches keep the mapping structure with stacked tensors of the
        # right size and values, once without workers and once with a single persistent worker.
        for persistent_workers in (False, True):
            if persistent_workers:
                loader = DataLoader(
                    self.dataset,
                    batch_size=2,
                    shuffle=False,
                    persistent_workers=persistent_workers,
                    num_workers=1,
                )
            else:
                loader = DataLoader(
                    self.dataset,
                    batch_size=2,
                    shuffle=False,
                    persistent_workers=persistent_workers,
                )
            batch_size = loader.batch_size
            for i, sample in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(set(sample.keys()), {"a_tensor", "another_dict"})
                self.assertEqual(set(sample["another_dict"].keys()), {"a_number"})

                t = sample["a_tensor"]
                self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
                self.assertTrue((t[0] == idx).all())
                self.assertTrue((t[1] == idx + 1).all())

                n = sample["another_dict"]["a_number"]
                self.assertEqual(n.size(), torch.Size([batch_size]))
                self.assertEqual(n[0], idx)
                self.assertEqual(n[1], idx + 1)

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_pin_memory(self):
        # [FUNCTION DOC] Checks pin_memory pins tensors nested at every level of the mapping.
        # Skipped without pinned memory support.
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for sample in loader:
            self.assertTrue(sample["a_tensor"].is_pinned())
            self.assertTrue(sample["another_dict"]["a_number"].is_pinned())

    @unittest.skipIf(TEST_PIN_MEMORY, "test requires no accelerator")
    def test_pin_memory_no_accelerator(self):
        # [FUNCTION DOC] Checks that when no accelerator is available the pin_memory flag leaves
        # tensors unpinned. Skipped when pinned memory is supported.
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for sample in loader:
            self.assertFalse(sample["a_tensor"].is_pinned())
            self.assertFalse(sample["another_dict"]["a_number"].is_pinned())

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_pin_memory_device(self):
        # [FUNCTION DOC] Checks the deprecated pin_memory_device argument emits a UserWarning but
        # pinning still works on the current accelerator. Skipped without pinned memory support.
        # pin_memory_device is deprecated; verify the deprecation warning fires
        # and that pin_memory still works via the current accelerator.
        acc_type = torch.accelerator.current_accelerator().type
        with self.assertWarnsRegex(UserWarning, "pin_memory_device is deprecated"):
            loader = DataLoader(
                self.dataset,
                batch_size=2,
                pin_memory=True,
                pin_memory_device=acc_type,
            )
            for sample in loader:
                self.assertTrue(sample["a_tensor"].is_pinned())
                self.assertTrue(sample["another_dict"]["a_number"].is_pinned())

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_pin_memory_with_only_device(self):
        # [FUNCTION DOC] Checks that giving only the pin memory device (without enabling
        # pin_memory) does not pin tensors. Skipped without pinned memory support.
        # pin_memory_device alone (without pin_memory=True) should not pin tensors.
        acc_type = torch.accelerator.current_accelerator().type
        loader = DataLoader(self.dataset, batch_size=2, pin_memory_device=acc_type)
        for sample in loader:
            self.assertFalse(sample["a_tensor"].is_pinned())
            self.assertFalse(sample["another_dict"]["a_number"].is_pinned())


class DummyDataset(torch.utils.data.Dataset):
# [CLASS DOC] HELPER: dataset over a list of integers zero to nine. Its item access expects an
# externally assigned start attribute equal to zero in the process that serves it, which is how
# the persistent worker test detects a dataset being copied again.
    def __init__(self) -> None:
    # [FUNCTION DOC] HELPER: creates the list of ten integers.
        self.data = list(range(10))

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the list length.
        return len(self.data)

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: converts a tensor index to a list, asserts the start attribute is
    # zero (AssertionError otherwise; AttributeError if never assigned) and returns the element.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # The persistent workers always maintain the original
        # dataset through the dataloader lifetime
        # so the attributes will remain the same as the
        # first time the workers where spawned (dataloader iteration)
        if self.start != 0:
            raise AssertionError(f"Expected start=0, got {self.start}")
        return self.data[idx]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDataLoaderPersistentWorkers(TestDataLoader):
    # [CLASS DOC] Reruns all tests of the main DataLoader test case with persistent workers
    # enabled, plus tests specific to persistent workers (cleanup at exit, file descriptor growth,
    # dataset caching).
    def setUp(self):
        # [FUNCTION DOC] Runs the parent setup and then turns the persistent workers flag on.
        super().setUp()
        self.persistent_workers = True

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # [FUNCTION DOC] Same subprocess test as the parent class but the child loader also uses
        # persistent workers; expects the file descriptor limit RuntimeError message. Skipped on
        # Sandcastle and Windows.
        # See NOTE [ DataLoader on Linux and open files limit ]
        import subprocess

        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1, persistent_workers=True):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
""",
            ]
        )

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_persistent_workers_pin_memory_atexit_registered_once(self):
        # [FUNCTION DOC] Checks the atexit cleanup hook for persistent workers with pin memory is
        # registered only once across eight loader lifetimes: patches the registration function
        # with a spy and the module level registration flag and weak set, creates and deletes
        # loaders, checks both objects are garbage collected, the weak set is empty and the hook
        # was registered exactly once.
        import atexit
        import weakref

        cleanup_callback = dataloader._MultiProcessingDataLoaderIter._clean_up_persistent_workers_atexit
        register_spy = mock.Mock(wraps=atexit.register)
        persistent_workers_atexit = weakref.WeakSet()

        with (
            mock.patch("atexit.register", register_spy),
            mock.patch.object(
                dataloader, "_persistent_workers_atexit_registered", False
            ),
            mock.patch.object(
                dataloader, "_persistent_workers_atexit", persistent_workers_atexit
            ),
        ):
            for _ in range(8):
                loader = self._get_data_loader(
                    self.dataset, batch_size=2, num_workers=1, pin_memory=True
                )
                it = iter(loader)
                next(it)
                loader_ref = weakref.ref(loader)
                it_ref = weakref.ref(it)
                del it
                del loader
                gc.collect()
                self.assertIsNone(it_ref())
                self.assertIsNone(loader_ref())

        self.assertEqual(len(persistent_workers_atexit), 0)
        self.assertEqual(
            sum(
                1
                for call in register_spy.call_args_list
                if call.args and call.args[0] is cleanup_callback
            ),
            1,
        )

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_persistent_workers_pin_memory_atexit_uses_iterator_shutdown(self):
        # [FUNCTION DOC] Checks the atexit cleanup hook calls the iterator's own shutdown method
        # exactly once: with patched module state, creates a persistent pin memory iterator, wraps
        # its shutdown with a spy and invokes the class cleanup hook.
        import weakref

        with (
            mock.patch.object(
                dataloader, "_persistent_workers_atexit_registered", True
            ),
            mock.patch.object(
                dataloader, "_persistent_workers_atexit", weakref.WeakSet()
            ),
        ):
            loader = self._get_data_loader(
                self.dataset, batch_size=2, num_workers=1, pin_memory=True
            )
            it = iter(loader)
            next(it)

            shutdown_workers_spy = mock.Mock(wraps=it._shutdown_workers)

            with mock.patch.object(
                it,
                "_shutdown_workers",
                shutdown_workers_spy,
            ):
                dataloader._MultiProcessingDataLoaderIter._clean_up_persistent_workers_atexit()

            self.assertEqual(shutdown_workers_spy.call_count, 1)

    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    @unittest.skipIf(not IS_LINUX, "fd counting is only reliable on Linux")
    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_persistent_workers_pin_memory_fd_does_not_grow_linearly(self):
        # [FUNCTION DOC] Checks the process open file descriptor count does not grow linearly when
        # repeatedly creating and discarding persistent pin memory loaders: after one warm up
        # round and fifteen more, it must be at most eight above the baseline. Linux, psutil and
        # pinned memory needed.
        proc = psutil.Process()

        def run_loader_once() -> None:
            # [FUNCTION DOC] HELPER: creates a persistent single worker pin memory loader, fetches
            # a batch, deletes the loader and iterator, runs garbage collection, asserts both are
            # freed and sleeps briefly so asynchronous cleanup can finish.
            import weakref

            loader = self._get_data_loader(
                self.dataset,
                batch_size=2,
                num_workers=1,
                pin_memory=True,
            )
            it = iter(loader)
            next(it)
            loader_ref = weakref.ref(loader)
            it_ref = weakref.ref(it)
            del it
            del loader
            gc.collect()
            self.assertIsNone(it_ref())
            self.assertIsNone(loader_ref())
            time.sleep(0.05)

        run_loader_once()
        baseline_fds = proc.num_fds()
        for _ in range(15):
            run_loader_once()
        after_fds = proc.num_fds()
        self.assertLessEqual(after_fds, baseline_fds + 8)

    def test_dataset_not_reset(self):
        # [FUNCTION DOC] Checks persistent workers keep their original dataset copy across epochs:
        # over ten epochs the parent changes the start attribute after each, but the workers' copy
        # still has zero. Also with pin_memory when supported.
        dataset = DummyDataset()
        pin_memory_configs = [False]
        if TEST_PIN_MEMORY:
            pin_memory_configs.append(True)
        for pin_memory in pin_memory_configs:
            dataloader = self._get_data_loader(
                dataset, num_workers=2, pin_memory=pin_memory
            )
            dataset.start = 0
            for i in range(10):
                for _ in dataloader:
                    pass
                # Changing the start value here doesn't have any effect in the dataset
                # cached by the workers. since they are not recreated between epochs
                # and can cache values safely
                dataset.start = i

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_early_exit(self):
        # [FUNCTION DOC] Runs a subprocess script that creates a persistent worker, pin memory
        # loader with fork context, takes one batch and exits early; check_output fails if the
        # interpreter does not exit cleanly. Skipped on Sandcastle and Windows.
        import subprocess

        proc = subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import torch
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

if __name__ == '__main__':
    dl = DataLoader(
        RandomDataset(64, (28, 28)),
        batch_size=16,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    for _ in dl:
        break
""",
            ]
        )


class NamedTupleDataset(Dataset):
# [CLASS DOC] HELPER: dataset of four items, each a named tuple of a nested named tuple (positive
# and negative index), a string label and a random tensor. Defines the two named tuple types as
# class attributes.
    from collections import namedtuple

    Batch = namedtuple("Batch", ["data", "label", "random_tensor"])
    Data = namedtuple("Data", ["positive", "negative"])

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns 4.
        return 4

    def __getitem__(self, ndx):
    # [FUNCTION DOC] HELPER: builds the nested named tuple item for the index.
        return self.Batch(
            data=self.Data(positive=ndx, negative=-ndx),
            label=str(ndx),
            random_tensor=torch.randn(3),
        )


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestNamedTupleDataLoader(TestCase):
    # [CLASS DOC] Tests that named tuples survive collation and pinning.
    def setUp(self):
        # [FUNCTION DOC] Creates the named tuple dataset.
        super().setUp()
        self.dataset = NamedTupleDataset()

    def test_dataloader_with_namedtuple(self):
        # [FUNCTION DOC] Checks batches keep the named tuple types (outer and nested), with
        # tensors pinned depending on platform support, with auto collation (index becomes tensor)
        # and without (index stays a plain value).
        # auto-collation
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=TEST_PIN_MEMORY)
        for batch in loader:
            self.assertIsInstance(batch, NamedTupleDataset.Batch)
            self.assertEqual(batch.random_tensor.is_pinned(), TEST_PIN_MEMORY)
            self.assertIsInstance(batch.data, NamedTupleDataset.Data)
            self.assertIsInstance(batch.data.positive, torch.Tensor)
            self.assertEqual(batch.data.positive.is_pinned(), TEST_PIN_MEMORY)
        # no auto-collation
        loader = DataLoader(self.dataset, batch_size=None, pin_memory=TEST_PIN_MEMORY)
        for batch in loader:
            self.assertIsInstance(batch, NamedTupleDataset.Batch)
            self.assertEqual(batch.random_tensor.is_pinned(), TEST_PIN_MEMORY)
            self.assertIsInstance(batch.data, NamedTupleDataset.Data)
            self.assertNotIsInstance(batch.data.positive, torch.Tensor)


class SimpleCustomBatch:
# [CLASS DOC] HELPER: custom batch type holding stacked inputs and targets built from a list of
# input and target pairs; supports pinning itself.
    def __init__(self, data):
    # [FUNCTION DOC] HELPER: transposes the list of pairs and stacks inputs and targets into two
    # tensors.
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
    # [FUNCTION DOC] HELPER: pins both tensors in place and returns the batch itself.
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

    def is_pinned(self):
    # [FUNCTION DOC] HELPER: true when both tensors are pinned.
        return self.inp.is_pinned() and self.tgt.is_pinned()


# Workaround for https://github.com/pytorch/pytorch/issues/50661
# Classes from  `__main__` can not be correctly unpickled from spawned module
# See https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
self_module = __import__(os.path.splitext(os.path.basename(__file__))[0])


def collate_wrapper(batch):
    # [FUNCTION DOC] HELPER: collate function building the custom batch through the module
    # imported by name, so it can be unpickled in spawned workers (workaround for issue 50661).
    return self_module.SimpleCustomBatch(batch)


def collate_into_packed_sequence(batch):
    # [FUNCTION DOC] HELPER: collate function stacking the first part of each sample along a new
    # dimension one, giving time by batch data, and packing it into a PackedSequence with random
    # lengths between one and the time length (unsorted allowed).
    data = torch.stack([sample[0] for sample in batch], 1)
    t, b = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, enforce_sorted=False)


def collate_into_packed_sequence_batch_first(batch):
    # [FUNCTION DOC] HELPER: batch first variant of the packing collate function: stacks along
    # dimension zero and packs with the batch first option and random lengths.
    data = torch.stack([sample[0] for sample in batch], 0)
    b, t = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(
        data, lengths, batch_first=True, enforce_sorted=False
    )


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestCustomPinFn(TestCase):
    # [CLASS DOC] Tests custom batch types with their own pinning method.
    def setUp(self):
        # [FUNCTION DOC] Creates a ten sample tensor dataset with identical input and target
        # tensors.
        super().setUp()
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.dataset = TensorDataset(inps, tgts)

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_custom_batch_pin(self):
        # [FUNCTION DOC] Checks, with pin_memory in the main process, that the three custom
        # collate functions give instances of the expected class and that they are pinned. Skipped
        # without pinned memory support.
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(
                self.dataset, batch_size=2, collate_fn=collate_fn, pin_memory=True
            )
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())

    @unittest.skipIf(not TEST_PIN_MEMORY, "pin_memory requires accelerator")
    def test_custom_batch_pin_worker(self):
        # [FUNCTION DOC] Same as the previous test but with one worker process doing the loading.
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(
                self.dataset,
                batch_size=2,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=1,
            )
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())


class TestWorkerQueueDataset(Dataset):
# [CLASS DOC] HELPER: dataset over a list whose items are pairs of the worker id that served them
# and the data value; the worker id is recorded by its init function.
    def __init__(self, data):
    # [FUNCTION DOC] HELPER: stores the data and an unset worker id.
        self.data = data
        self.worker_id = None

    def worker_init_fn(self, worker_id):
    # [FUNCTION DOC] HELPER: records the worker id on the dataset copy in that worker.
        self.worker_id = worker_id

    def __getitem__(self, item):
    # [FUNCTION DOC] HELPER: returns the pair of recorded worker id and data value.
        return self.worker_id, self.data[item]

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the data length.
        return len(self.data)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestIndividualWorkerQueue(TestCase):
    # [CLASS DOC] Tests that batches are assigned to workers in round robin order.
    def setUp(self):
        # [FUNCTION DOC] Creates a dataset of 128 values.
        super().setUp()
        self.dataset = TestWorkerQueueDataset(list(range(128)))

    def _run_ind_worker_queue_test(self, batch_size, num_workers):
        # [FUNCTION DOC] HELPER: loads unshuffled with given batch size and worker count, with a
        # timeout and the dataset init function; asserts each batch comes from the next worker in
        # cyclic order and has the expected consecutive values.
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            timeout=JOIN_TIMEOUT,
            worker_init_fn=self.dataset.worker_init_fn,
        )
        current_worker_idx = 0
        for i, (worker_ids, sample) in enumerate(loader):
            self.assertEqual(worker_ids.tolist(), [current_worker_idx] * batch_size)
            self.assertEqual(
                sample.tolist(), list(range(i * batch_size, (i + 1) * batch_size))
            )
            current_worker_idx += 1
            if current_worker_idx == num_workers:
                current_worker_idx = 0

    @unittest.skipIf(
        IS_WINDOWS or IS_MACOS,
        "Flaky on Windows and MacOS https://github.com/pytorch/pytorch/issues/68643",
    )
    def test_ind_worker_queue(self):
        # [FUNCTION DOC] Runs the round robin check for batch sizes 8, 32, 64 and one to five
        # workers. Skipped on Windows and macOS (flaky).
        for batch_size in (8, 32, 64):
            for num_workers in range(1, 6):
                self._run_ind_worker_queue_test(
                    batch_size=batch_size, num_workers=num_workers
                )


class SetAffinityDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset yielding the CPU ids in the iterating process affinity
# mask; may carry a CPU id that worker init functions should pin themselves to.
    def __init__(self, expected_affinity=None):
    # [FUNCTION DOC] HELPER: stores the expected CPU id (or none).
        self.expected_affinity = expected_affinity

    def __iter__(self):
    # [FUNCTION DOC] HELPER: reads the process CPU affinity mask and returns an iterator over it.
        affinity_mask = os.sched_getaffinity(0)
        return iter(affinity_mask)


def _worker_set_affinity_init(worker_id):
    # [FUNCTION DOC] HELPER: worker_init_fn; inside a worker whose dataset carries an expected
    # CPU, sets the worker process affinity to just that CPU. Does nothing otherwise. The id
    # argument is unused.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if (
            isinstance(dataset, SetAffinityDataset)
            and dataset.expected_affinity is not None
        ):
            os.sched_setaffinity(0, [dataset.expected_affinity])


@unittest.skipIf(
    not hasattr(os, "sched_setaffinity"), "os.sched_setaffinity is not available"
)
class TestSetAffinity(TestCase):
    # [CLASS DOC] Tests CPU affinity changes made from worker init functions. Skipped when the
    # platform has no affinity setter.
    def test_set_affinity_in_worker_init(self):
        # [FUNCTION DOC] Picks the last CPU from the current affinity mask (skips the test when
        # the mask is empty), loads from a two worker loader whose init function pins workers to
        # that CPU, and checks every sample is a list holding only that CPU.
        # Query the current affinity mask to avoid setting a disallowed one
        old_affinity = os.sched_getaffinity(0)
        if not old_affinity:
            self.skipTest("No affinity information")
        # Choose any
        expected_affinity = list(old_affinity)[-1]

        # Pass expected affinity through the dataset
        dataset = SetAffinityDataset(expected_affinity=expected_affinity)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            worker_init_fn=_worker_set_affinity_init,
        )

        for sample in dataloader:
            self.assertEqual(sample, [expected_affinity])


class ConvDataset(Dataset):
# [CLASS DOC] HELPER: single item dataset whose item is a one dimensional convolution of a ones
# tensor with a small kernel; the convolution is also run once in the parent at construction.
    def __init__(self) -> None:
    # [FUNCTION DOC] HELPER: creates the input tensor and runs one item access in the parent
    # process (so convolution state is initialized before forking).
        self.x = torch.ones(1, 1, 24000)
        # Call convolution on parent process
        self[0]

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns 1.
        return 1

    def __getitem__(self, index):
    # [FUNCTION DOC] HELPER: returns the convolution of the input with a kernel of two ones.
        return torch.nn.functional.conv1d(self.x, torch.ones(1, 1, 2))


@unittest.skipIf(IS_WINDOWS, "Needs fork")
class TestConvAfterFork(TestCase):
# [CLASS DOC] Regression tests for convolution after fork. Skipped on Windows.
    # Tests crash reported in https://github.com/pytorch/pytorch/issues/53565
    def test_conv_after_fork(self):
        # [FUNCTION DOC] Regression test for issue 53565: a forked worker runs convolution after
        # the parent already did; the loaded result must have the expected output shape.
        loader = DataLoader(ConvDataset(), num_workers=1)
        for x in loader:
            self.assertEqual(x.shape, (1, 1, 1, 23999))


class TestSlowIndexDataset(Dataset):
# [CLASS DOC] HELPER: map style dataset recording the id of the worker serving it and sleeping one
# second when a given slow index is accessed; items are pairs of worker id and index.
    def __init__(self, end: int, slow_index: int):
    # [FUNCTION DOC] HELPER: stores the end, the slow index and an unset worker id.
        self.end = end
        self.slow_index = slow_index
        self._worker_id = None

    def __getitem__(self, idx):
    # [FUNCTION DOC] HELPER: lazily records the worker id from worker info, sleeps one second for
    # the slow index and returns a pair of worker id and index.
        if not self._worker_id:
            worker_info = torch.utils.data.get_worker_info()
            self._worker_id = worker_info.id
        if idx == self.slow_index:
            time.sleep(1.0)
        return (self._worker_id, idx)

    def __len__(self):
    # [FUNCTION DOC] HELPER: returns the end.
        return self.end


class TestSlowIterableDataset(IterableDataset):
# [CLASS DOC] HELPER: iterable dataset splitting a range evenly among workers where each worker
# yields pairs of worker id and value, with a one second sleep at the midpoint value. Requires
# worker context.
    def __init__(self, start: int, end: int):
    # [FUNCTION DOC] HELPER: stores the start and end and computes the midpoint of the range
    # length.
        self.start = start
        self.end = end
        self.mid = math.ceil((self.end - self.start) / 2)

    def give_data(self, worker_id, iter_start, iter_end):
    # [FUNCTION DOC] HELPER: generator yielding pairs of worker id and value over the given range,
    # sleeping one second when the value equals the midpoint.
        for i in range(iter_start, iter_end):
            if i == self.mid:
                time.sleep(1.0)
            yield (worker_id, i)

    def __iter__(self):
    # [FUNCTION DOC] HELPER: computes this worker's share of the range from worker info (rounded
    # up division) and returns the generator for it.
        worker_info = torch.utils.data.get_worker_info()
        per_worker = int(
            math.ceil((self.end - self.start) / float(worker_info.num_workers))
        )
        worker_id = worker_info.id
        iter_start = self.start + worker_id * per_worker
        iter_end = min(iter_start + per_worker, self.end)
        return self.give_data(worker_id, iter_start, iter_end)


class TestOutOfOrderDataLoader(TestCase):
    # [CLASS DOC] Tests the in_order option of the DataLoader, with slow items to force
    # reordering.
    def test_in_order_index_ds(self):
        # [FUNCTION DOC] Checks with in-order delivery and a slow first item that outputs
        # alternate between two workers in index order.
        dataset = TestSlowIndexDataset(end=10, slow_index=0)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            in_order=True,
        )

        expected_worker_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        expected_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        outputs = list(dataloader)
        worker_ids = [o[0] for o in outputs]
        data = [o[1] for o in outputs]
        self.assertEqual(expected_worker_ids, worker_ids)
        self.assertEqual(expected_data, data)

    def test_out_of_order_index_ds(self):
        # [FUNCTION DOC] Checks with out-of-order delivery, prefetch factor two and a stuck first
        # item, that worker zero's results come last and the order differs from index order,
        # matching the expected worker ids and data.
        dataset = TestSlowIndexDataset(end=10, slow_index=0)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            prefetch_factor=2,
            in_order=False,
        )

        # worker_id = 0 gets 'stuck' on 0 and also has 2 in it's queue
        # due to prefetch_factor being 2
        # this makes the test more deterministic as [0, 2] will be the last elements
        expected_worker_ids = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        expected_data = [1, 3, 4, 5, 6, 7, 8, 9, 0, 2]
        outputs = list(dataloader)
        worker_ids = [o[0].item() for o in outputs]
        data = [o[1].item() for o in outputs]
        self.assertEqual(expected_worker_ids, worker_ids)
        self.assertNotEqual(data, list(range(10)))
        self.assertEqual(expected_data, data)

    def test_in_order_iterable_ds(self):
        # [FUNCTION DOC] Checks in-order delivery for the slow iterable dataset interleaves the
        # two workers' outputs.
        dataset = TestSlowIterableDataset(start=0, end=10)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            in_order=True,
        )

        expected_worker_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        expected_data = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        outputs = list(dataloader)
        worker_ids = [o[0] for o in outputs]
        data = [o[1] for o in outputs]
        self.assertEqual(expected_worker_ids, worker_ids)
        self.assertEqual(expected_data, data)

    def test_out_of_order_iterable_ds(self):
        # [FUNCTION DOC] Checks out-of-order delivery for the slow iterable dataset: worker zero's
        # values all come first, and the order differs from the interleaved in-order order.
        dataset = TestSlowIterableDataset(start=0, end=10)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            in_order=False,
        )

        # worker 0 has [0, 1, 2, 3, 4], worker 1 has [5, 6, 7, 8, 9]
        # index 5 is slow, so expect all of worker 0 before worker 1
        expected_worker_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        expected_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        outputs = list(dataloader)
        worker_ids = [o[0] for o in outputs]
        data = [o[1] for o in outputs]
        self.assertEqual(expected_worker_ids, worker_ids)
        self.assertEqual(sum(worker_ids), 5)
        self.assertNotEqual(data, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
        self.assertEqual(expected_data, data)


instantiate_device_type_tests(TestDataLoaderDeviceType, globals())


if __name__ == "__main__":
    run_tests()
