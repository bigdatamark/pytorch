#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC] Tests for the deprecated torch.distributed.launch module entry point (main).
# Launches a 4 process single node job with spawn start method, with and without --use-env,
# using worker scripts under bin/. The module raises RuntimeError when run directly because
# these tests are not currently enabled in discover_tests.py.

import os
import shutil
import tempfile
import unittest
from contextlib import closing

import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


def path(script):
    # [FUNCTION DOC] HELPER: returns the absolute path of a script located next to this file.
    return os.path.join(os.path.dirname(__file__), script)


class LaunchTest(unittest.TestCase):
    # [CLASS DOC] Tests for torch.distributed.launch.main. State: test_dir is a fresh temporary
    # directory per test, removed afterwards; a sentinel environment variable is exported to
    # the parent process so workers inherit it.
    def setUp(self):
        # [FUNCTION DOC] HELPER: creates the temporary directory test_dir and sets the process
        # environment variable TEST_SENTINEL_PARENT to FOOBAR (never removed afterwards, so it
        # leaks into later tests in the same process).
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        # [FUNCTION DOC] HELPER: recursively deletes test_dir.
        shutil.rmtree(self.test_dir)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_without_env(self):
        # [FUNCTION DOC] Reserves a free port, then launches 4 local workers (node rank 0,
        # localhost master) running bin/test_script_local_rank.py without --use-env, so the
        # launcher must pass --local-rank on each worker's command line. Passes if launch.main
        # returns without error.
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
            path("bin/test_script_local_rank.py"),
        ]
        launch.main(args)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_with_env(self):
        # [FUNCTION DOC] Launches 4 local workers with --use-env running bin/test_script.py,
        # which touches one file per global rank in test_dir. Asserts the set of file names
        # equals the ranks 0 to 3, confirming every worker ran.
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
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
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
