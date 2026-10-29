#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Tiny logging helper that reports expired timers for a run. Creates a module logger via the elastic get_logger helper at import.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch.distributed.elastic.utils.logging import get_logger


logger = get_logger(__name__)

__all__ = ["log_debug_info_for_expired_timers"]


def log_debug_info_for_expired_timers(
    run_id: str,
    expired_timers: dict[int, list[str]],
):
    # [FUNCTION DOC] Log at info level the run id and the mapping of expired timers (keyed by worker/process id to list of scope strings). Does nothing if the mapping is empty. Returns None.
    if expired_timers:
        logger.info("Timers expired for run:[%s] [%s].", run_id, expired_timers)
