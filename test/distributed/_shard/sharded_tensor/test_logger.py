# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Single-process unit test for the sharded tensor logger helper, which must
# return a logger with exactly one NullHandler attached.

import logging

from torch.distributed._shard.sharded_tensor.logger import _get_or_create_logger
from torch.testing._internal.common_utils import run_tests, TestCase


class ShardingSpecLoggerTest(TestCase):
    # [CLASS DOC] Plain test case for the sharded tensor logger helper.
    def test_get_or_create_logger(self):
        # [FUNCTION DOC] Calls the logger helper; asserts it returns a non-None logger with
        # exactly one handler, which is a NullHandler.
        logger = _get_or_create_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)


if __name__ == "__main__":
    run_tests()
