#!/usr/bin/env python3
# [MODULE DOC] Unit tests for CyclingIterator from torch.distributed.elastic.utils.data, which
# chains per-epoch iterators produced by a generator function for a fixed number of epochs. Pure
# unittest; no import-time side effects.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from torch.distributed.elastic.utils.data import CyclingIterator


class CyclingIteratorTest(unittest.TestCase):
    # [CLASS DOC] Tests that CyclingIterator yields the concatenation of the per-epoch generators
    # from the start epoch through the last epoch and then raises StopIteration.
    def generator(self, epoch, stride, max_epochs):
        # [FUNCTION DOC] HELPER: Returns an iterator over a list of stride consecutive integers
        # beginning at stride times epoch, so successive epochs continue the count. The max_epochs
        # argument is unused.
        # generate an continuously incrementing list each epoch
        # e.g. [0,1,2] [3,4,5] [6,7,8] ...
        return iter([stride * epoch + i for i in range(stride)])

    def test_cycling_iterator(self):
        # [FUNCTION DOC] Builds a CyclingIterator over 90 epochs of stride 3 and checks it yields 0
        # through 269 in order, then raises StopIteration.
        stride = 3
        max_epochs = 90

        def generator_fn(epoch):
            # [FUNCTION DOC] HELPER: Closure adapting the test generator to the one-argument epoch
            # callable that CyclingIterator expects.
            return self.generator(epoch, stride, max_epochs)

        it = CyclingIterator(n=max_epochs, generator_fn=generator_fn)
        for i in range(stride * max_epochs):
            self.assertEqual(i, next(it))

        with self.assertRaises(StopIteration):
            next(it)

    def test_cycling_iterator_start_epoch(self):
        # [FUNCTION DOC] Builds a CyclingIterator with 2 epochs and a start epoch of 1 and checks
        # it yields only the second epoch values (3 to 5) before raising StopIteration.
        stride = 3
        max_epochs = 2
        start_epoch = 1

        def generator_fn(epoch):
            # [FUNCTION DOC] HELPER: Closure adapting the test generator to the one-argument epoch
            # callable that CyclingIterator expects.
            return self.generator(epoch, stride, max_epochs)

        it = CyclingIterator(max_epochs, generator_fn, start_epoch)
        for i in range(stride * start_epoch, stride * max_epochs):
            self.assertEqual(i, next(it))

        with self.assertRaises(StopIteration):
            next(it)
