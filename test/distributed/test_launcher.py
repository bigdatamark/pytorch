# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Test for torch.distributed.launch: launches a 4-process single-node job with
# the spawn start method on a free local port, running bin/test_script.py. Exits at import if
# distributed is unavailable or when running under dev-dbg ASAN.

import os
import sys
from contextlib import closing

import torch.distributed as dist
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


def path(script):
    # [FUNCTION DOC] HELPER: Returns the absolute path of a script name relative to this
    # test file's directory.
    return os.path.join(os.path.dirname(__file__), script)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


class TestDistributedLaunch(TestCase):
    def test_launch_user_script(self):
        # [FUNCTION DOC] Picks a free port by binding a socket and closing it, then calls
        # torch.distributed.launch main with arguments for 1 node, 4 processes per node, spawn
        # start, localhost master, node rank 0 and the use-env flag, running
        # bin/test_script.py. Passes if the launcher returns without raising.
        nnodes = 1
        nproc_per_node = 4
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            "--use-env",
            path("bin/test_script.py"),
        ]
        launch.main(args)


if __name__ == "__main__":
    run_tests()
