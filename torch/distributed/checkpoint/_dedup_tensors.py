# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC]
# Simple deduplication of replicated write items across save plans, keeping the first
# occurrence. Exports dedup_tensors. Import-time side effect: creates the module logger with a
# console handler at INFO level and no propagation.
import dataclasses
import logging
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_tensors"]


def init_logger() -> logging.Logger:
    # [FUNCTION DOC] HELPER: configures and returns this module's logger: INFO level, a console
    # stream handler with a detailed formatter, no propagation to parent loggers. Mutates logging state.
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = init_logger()


# TODO add docstring for dedup_tensors
def dedup_tensors(all_plans: list[SavePlan]) -> list[SavePlan]:
    # [FUNCTION DOC] Finds write items (by index) present in more than one plan and removes them
    # from every plan except the first (lowest position) that has them. Logs the removed keys at
    # info level. Returns a new list; modified plans are copies, originals are not mutated.
    all_plans = list(all_plans)
    key_to_plan: dict[MetadataIndex, list[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # Remove duplicates by always keeping the first entry.
    # Compute the per-rank remove set.
    plan_to_keys: dict[int, list[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)
    if len(plan_to_keys) > 0:
        logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in key_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
