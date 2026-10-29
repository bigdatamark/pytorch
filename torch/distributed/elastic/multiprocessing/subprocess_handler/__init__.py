#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [PACKAGE DOC] Subprocess launching helpers for elastic multiprocessing. Re-exports
# SubprocessHandler (a Popen wrapper) and get_subprocess_handler (its factory).
from torch.distributed.elastic.multiprocessing.subprocess_handler.handlers import (
    get_subprocess_handler,
)
from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import (
    SubprocessHandler,
)


__all__ = ["SubprocessHandler", "get_subprocess_handler"]
