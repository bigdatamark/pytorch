#!/usr/bin/env python3

# [PACKAGE DOC] Data loading utilities for elastic training: re-exports CyclingIterator (repeats an iterator over several epochs) and ElasticDistributedSampler (distributed sampler that can start from a given dataset index).
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .cycling_iterator import CyclingIterator
from .elastic_distributed_sampler import ElasticDistributedSampler
