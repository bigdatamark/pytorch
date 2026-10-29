# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the c10d exception logger decorator. At import it exits the process
# silently (status 0) if torch.distributed is unavailable or dev-asan is enabled, and computes
# WORLD_SIZE (2 to 4, from the accelerator device count). Defines a with_comms decorator and
# C10dErrorLoggerTest, which checks the logger's handler and the logged message content.

import json
import logging
import re
import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _c10d_logger, _exception_logger


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistributedTestBase, TEST_SKIPS
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = get_devtype().type

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

WORLD_SIZE = min(4, max(2, torch.get_device_module(device_type).device_count()))


def with_comms(func=None):
    # [FUNCTION DOC] HELPER: decorator for test methods. Usable with or without call parentheses.
    # The wrapper exits with the multi-gpu skip code if fewer devices than world_size exist,
    # otherwise creates the process group, runs the test, then destroys the comms.
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: skip-or-run wrapper; extra args are ignored, the test is
        # called with self only, and its return value is discarded.
        if torch.get_device_module(device_type).device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.create_pg(device_type)
        func(self)
        self.destroy_comms()

    return wrapper


class C10dErrorLoggerTest(DistributedTestBase):
    # [CLASS DOC] Distributed test case with WORLD_SIZE ranks verifying that _exception_logger
    # logs failure details to _c10d_logger at debug level and still re-raises.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: property; the module-level WORLD_SIZE.
        return WORLD_SIZE

    @property
    def process_group(self):
        # [FUNCTION DOC] HELPER: property; the default (WORLD) process group.
        return dist.group.WORLD

    def destroy_comms(self):
        # [FUNCTION DOC] HELPER: barriers across all ranks, then destroys the default group.
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def test_get_or_create_logger(self):
        # [FUNCTION DOC] Asserts the c10d logger exists and has exactly one handler, a
        # NullHandler.
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)

    @_exception_logger
    def _failed_broadcast_raise_exception(self):
        # [FUNCTION DOC] HELPER: broadcasts with an out-of-range source rank, so the error
        # propagates; the decorator logs it and re-raises.
        tensor = torch.arange(2, dtype=torch.int64)
        dist.broadcast(tensor, self.world_size + 1)

    @_exception_logger
    def _failed_broadcast_not_raise_exception(self):
        # [FUNCTION DOC] HELPER: same invalid broadcast, but the exception is caught and
        # swallowed inside, so nothing propagates out of this method; the decorator on it
        # itself therefore logs nothing.
        try:
            tensor = torch.arange(2, dtype=torch.int64)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception:
            pass

    @with_comms
    def test_exception_logger(self) -> None:
        # [FUNCTION DOC] Asserts the raising helper raises. Then captures debug logs while the
        # swallowing helper runs, parses the first captured message as a mapping, and checks
        # its size (9 keys on NCCL, else 8) and the values of pg_name, func_name, backend,
        # nccl_version (NCCL only), group_size, world_size, global_rank and local_rank.
        # The logged record comes from the logger decorator on the inner broadcast call, since
        # the helper swallows its own exception.
        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception()

        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._failed_broadcast_not_raise_exception()
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            # NCCL adds additional nccl_version data to the error_msg_dict
            if self.backend(device_type) == dist.Backend.NCCL:
                self.assertEqual(len(error_msg_dict), 9)
            else:
                self.assertEqual(len(error_msg_dict), 8)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual(self.backend(device_type), error_msg_dict["backend"])

            if self.backend(device_type) == dist.Backend.NCCL:
                self.assertIn("nccl_version", error_msg_dict.keys())
                nccl_ver = torch.cuda.nccl.version()
                self.assertEqual(
                    ".".join(str(v) for v in nccl_ver), error_msg_dict["nccl_version"]
                )

            # In this test case, group_size = world_size, since we don't have multiple processes on one node.
            self.assertIn("group_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["group_size"])

            self.assertIn("world_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["world_size"])

            self.assertIn("global_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["global_rank"])

            # In this test case, local_rank = global_rank, since we don't have multiple processes on one node.
            self.assertIn("local_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["local_rank"])


if __name__ == "__main__":
    run_tests()
