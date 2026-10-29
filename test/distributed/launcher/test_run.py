#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC] Tests for the torchrun entry point torch.distributed.run (imported as launch).
# Covers launching worker scripts from bin/ in python, bash (--no-python) and run-path modes,
# nproc-per-node parsing (number, auto, gpu, xpu, invalid), PET_ environment variable
# configuration, standalone and c10d elastic rendezvous, multiple agents, error propagation
# from workers and the agent, rendezvous shutdown, is_torchelastic_launched, logs specs,
# init_method handling, log capture with a line prefix, and the virtual-local-rank option.
# Helper functions here wrap launching and process inspection; no import-time side effects.

import io
import multiprocessing as mp
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import closing, redirect_stderr, redirect_stdout
from unittest import mock, skipIf
from unittest.mock import MagicMock, Mock, patch

import pytest

import torch
import torch.distributed.run as launch
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.utils import get_socket_with_port
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_distributed import (
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


def launch_in_proc(args):
    # [FUNCTION DOC] HELPER: runs torchrun's main in the current process with the argument list.
    launch.main(args)


def path(script):
    # [FUNCTION DOC] HELPER: returns the absolute path of a script located next to this file.
    return os.path.join(os.path.dirname(__file__), script)


def get_child_pids(pid):
    # [FUNCTION DOC] HELPER: lists the direct child process ids of a process by running pgrep
    # with the parent option through a shell, waiting for it, and parsing each non-empty output
    # line as an integer. Returns a list of ints (empty if none).
    # KNOWN ISSUE: waits for pgrep before reading its piped output, which could block on large
    #   output; the loop variable also shadows the pid parameter.
    pgrep = subprocess.Popen(args=f"pgrep -P {pid}", shell=True, stdout=subprocess.PIPE)
    pgrep.wait()
    out = pgrep.stdout.read().decode("utf-8").rstrip().split("\n")
    pids = []
    for pid in out:
        if pid:
            pids.append(int(pid))
    return pids


def pid_exists(pid):
    # [FUNCTION DOC] HELPER: probes a process with the null signal. Returns True if the signal
    # call succeeds and False on any OS error.
    # KNOWN ISSUE: a permission error (process exists but belongs to another user) is also an OS
    #   error and is reported as not existing.
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class MockException(Exception):
    # [CLASS DOC] HELPER: marker exception used as the side effect of a mocked agent run so tests
    # can check the launcher re-raises the agent's original exception.
    pass


class ElasticLaunchTest(TestCase):
    # [CLASS DOC] End to end tests of torchrun (torch.distributed.run.main) that spawn real
    # worker processes. State: test_dir is a per-test temporary directory into which the worker
    # scripts write marker files named by global rank. Most tests are skipped under dev/dbg ASAN
    # builds. Fixtures clear PET_ variables and set a sentinel environment variable.
    def setUp(self):
        # [FUNCTION DOC] HELPER: creates test_dir, deletes every environment variable whose name
        # starts with PET_ (torchrun's config prefix) and sets TEST_SENTINEL_PARENT to FOOBAR,
        # which worker scripts assert is inherited.
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

        # remove any lingering environment variables
        for env in os.environ.keys():  # noqa: SIM118
            if env.startswith("PET_"):
                del os.environ[env]

        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        # [FUNCTION DOC] HELPER: recursively removes test_dir.
        shutil.rmtree(self.test_dir)

    def test_launch_user_script_python(self):
        # [FUNCTION DOC] Runs the shared python script launch scenario (see the helper below).
        self._test_launch_user_script_python()

    def _test_launch_user_script_python(self):
        # [FUNCTION DOC] HELPER: launches 4 python workers on one node with a random rdzv id,
        # running bin/test_script.py, then asserts test_dir contains exactly one file per global
        # rank 0..3.
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_launch_user_script_python_caffe2_bc(self):
        # [FUNCTION DOC] Backward compatibility launch using the older style flags: a free master
        # port, --master-addr=localhost and --node-rank=0 instead of rendezvous options. Asserts
        # all 4 workers ran (one marker file per rank).
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
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_user_script_bash(self):
        # [FUNCTION DOC] Launches 4 workers of bin/test_script.sh with --no-python. First checks
        # that combining --no-python with --module raises ValueError, then launches for real and
        # asserts one marker file per rank exists.
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            launch.main(args + ["--module"] + script_args)

        launch.main(args + script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_user_script_default_nproc(self):
        # [FUNCTION DOC] Same as the bash launch test but omits nproc-per-node, so the default of
        # a single process is used; checks the --no-python plus --module ValueError, then asserts
        # only rank 0's marker file exists.
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        world_size = 1
        args = [
            f"--nnodes={nnodes}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            launch.main(args + ["--module"] + script_args)

        launch.main(args + script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_with_env_vars(self):
        # [FUNCTION DOC] Configures torchrun purely through PET_ environment variables (nnodes,
        # nproc per node, rdzv id, monitor interval, start method, no python), setting PET_MODULE
        # to 1 first and expecting ValueError, then to 0 and launching bin/test_script.sh.
        # Asserts one marker file per rank. Side effect: mutates os.environ (cleared by setUp
        # later).
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node

        os.environ["PET_NNODES"] = str(nnodes)
        os.environ["PET_NPROC_PER_NODE"] = str(nproc_per_node)
        os.environ["PET_RDZV_ID"] = run_id
        os.environ["PET_MONITOR_INTERVAL"] = "1"
        os.environ["PET_START_METHOD"] = "spawn"
        os.environ["PET_NO_PYTHON"] = "1"

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no-python cannot be used with --module
            os.environ["PET_MODULE"] = "1"
            launch.main(script_args)

        os.environ["PET_MODULE"] = "0"
        launch.main(script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def _test_nproc_launch_configuration(self, nproc_type, expected_number):
        # [FUNCTION DOC] HELPER: launches bin/test_script.sh on one node with nproc-per-node set
        # to the given string (for example auto, gpu, xpu or a number) and asserts marker files
        # exist for exactly expected_number ranks. Invalid values propagate the launcher's
        # ValueError.
        run_id = str(uuid.uuid4().int)
        nnodes = 1

        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_type}",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        launch.main(args + script_args)

        world_size = nnodes * expected_number
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=False)
    def test_nproc_launch_auto_configurations(self, _mock1):
        # [FUNCTION DOC] With CUDA reported unavailable, nproc-per-node=auto must use one worker
        # per CPU core (torch._utils.cpu_count).
        expected = torch._utils.cpu_count()
        self._test_nproc_launch_configuration("auto", expected)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_number_configurations(self):
        # [FUNCTION DOC] An explicit nproc-per-node of 4 launches exactly 4 workers.
        self._test_nproc_launch_configuration("4", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_nproc_launch_unknown_configurations(self):
        # [FUNCTION DOC] An unrecognized nproc-per-node value must make the launcher raise
        # ValueError.
        with self.assertRaises(ValueError):
            self._test_nproc_launch_configuration("unknown", 4)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=3)
    @patch("torch.accelerator.is_available", return_value=True)
    @patch("torch.accelerator.device_count", return_value=3)
    @patch("torch.accelerator.current_accelerator", return_value=MagicMock(type="gpu"))
    def test_nproc_gpu_launch_configurations(
        self, _mock1, _mock2, _mock3, _mock4, _mock5
    ):
        # [FUNCTION DOC] With CUDA and the accelerator API mocked to report 3 gpu devices, both
        # auto and gpu nproc-per-node settings must launch 3 workers. Mock arguments are unused.
        self._test_nproc_launch_configuration("auto", 3)
        self._test_nproc_launch_configuration("gpu", 3)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @patch("torch.xpu.is_available", return_value=True)
    @patch("torch.xpu.device_count", return_value=3)
    @patch("torch.accelerator.is_available", return_value=True)
    @patch("torch.accelerator.device_count", return_value=3)
    @patch("torch.accelerator.current_accelerator", return_value=MagicMock(type="xpu"))
    def test_nproc_xpu_launch_configurations(
        self, _mock1, _mock2, _mock3, _mock4, _mock5
    ):
        # [FUNCTION DOC] With the xpu and accelerator APIs mocked to report 3 devices, both auto
        # and xpu nproc-per-node settings must launch 3 workers. Mock arguments are unused.
        self._test_nproc_launch_configuration("auto", 3)
        self._test_nproc_launch_configuration("xpu", 3)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        # [FUNCTION DOC] Launches with an elastic range of 1 to 2 nodes over the c10d rendezvous
        # on a free local port with short timeouts, but only one node joins; 4 workers run and
        # the test asserts a marker file for each of ranks 0..3.
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # we are only launching 1 node (even though max = 2)
        world_size = nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @mock.patch("torch.distributed.elastic.events.record")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        # [FUNCTION DOC] Launches workers that fail (--fail) with zero restarts. Expects the
        # launcher to raise ChildFailedError and the events record function to be called exactly
        # once.
        """
        Asserts that when the worker program fails and lancher raieses exception
        to indicate that worker process failed

        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv-conf='join_timeout=5,last_call_timeout=1,timeout=5'",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            "--fail",
        ]
        with self.assertRaises(ChildFailedError):
            launch.main(args)

        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        # [FUNCTION DOC] Makes the local elastic agent's run raise MockException. Expects the
        # launcher to re-raise that same exception and to record exactly one event.
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception
        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv_conf=timeout=5",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--max-restarts=0",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]

        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            launch.main(args)
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_standalone(self):
        # [FUNCTION DOC] Launches 4 workers with --standalone (single node rendezvous set up by
        # the launcher) and asserts one marker file per rank.
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--standalone",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_run_path(self):
        # [FUNCTION DOC] Launches 4 workers with --run-path, which executes the script in the
        # worker via runpy semantics, and asserts one marker file per rank.
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            "--run-path",
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        # [FUNCTION DOC] Simulates two nodes: one extra process (multiprocessing) runs
        # launch.main with the same c10d rendezvous arguments while the test process runs
        # another. The helper process must exit with code 0, and marker files must exist for all
        # 8 ranks.
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            "--rdzv_conf=timeout=5",
            f"--rdzv-id={run_id}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[args])
            procs.append(p)
            p.start()
        launch.main(args)
        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_min_max_nodes_parse(self):
        # [FUNCTION DOC] Checks parse_min_max_nnodes: a single number gives equal min and max, a
        # colon separated pair gives (min, max), and three parts raise RuntimeError.
        min_nodes, max_nodes = launch.parse_min_max_nnodes("1")
        self.assertEqual(min_nodes, max_nodes)
        self.assertEqual(1, min_nodes)
        min_nodes, max_nodes = launch.parse_min_max_nnodes("2:20")
        self.assertEqual(2, min_nodes)
        self.assertEqual(20, max_nodes)
        with self.assertRaises(RuntimeError):
            launch.parse_min_max_nnodes("2:20:30")

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    def test_launch_shutdown(self, agent_mock_cls):
        # [FUNCTION DOC] With the agent class mocked to return a succeeded result and the
        # rendezvous handler lookup mocked, asserts launch.main calls the handler's shutdown
        # exactly once.
        nnodes = 1
        nproc_per_node = 4
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            launch.main(args)
            rdzv_handler_mock.shutdown.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched(self):
        # [FUNCTION DOC] Launches bin/test_script_is_torchelastic_launched.py via torchrun with
        # one worker and asserts the file it writes contains the text True.
        # launch test script with torchelastic and validate that
        # torch.distributed.is_torchelastic_launched() returns True

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )

        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual("True", is_torchelastic_launched)

    @patch("torch.distributed.run.metadata")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_is_torchelastic_launched_with_logs_spec_defined(self, metadata_mock):
        # [FUNCTION DOC] Same check as above but with --logs_specs=default, where the importlib
        # metadata entry points lookup is mocked to return DefaultLogsSpecs. Removes any stale
        # out file first and asserts the written text is True.
        # mock the entrypoint API to avoid version issues.
        entrypoints = MagicMock()
        metadata_mock.entry_points.return_value = entrypoints

        group = MagicMock()
        entrypoints.select.return_value = group

        ep = MagicMock()
        ep.load.return_value = DefaultLogsSpecs

        group.select.return_value = ep
        group.__getitem__.return_value = ep

        out_file = f"{os.path.join(self.test_dir, 'out')}"
        if os.path.exists(out_file):
            os.remove(out_file)
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--monitor-interval=1",
                "--logs_specs=default",
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ]
        )

        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual("True", is_torchelastic_launched)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_logs_logs_spec_entrypoint_must_be_defined(self):
        # [FUNCTION DOC] Passing a logs_specs name with no registered entry point must raise
        # ValueError from launch.main.
        with self.assertRaises(ValueError):
            launch.main(
                [
                    "--run-path",
                    "--nnodes=1",
                    "--nproc-per-node=1",
                    "--monitor-interval=1",
                    "--logs_specs=DOESNOT_EXIST",
                    path("bin/test_script_is_torchelastic_launched.py"),
                ]
            )

    def test_is_not_torchelastic_launched(self):
        # [FUNCTION DOC] Runs the same script directly in this interpreter via runpy (with argv
        # patched) rather than torchrun, and asserts the file it writes contains False.
        # launch test script without torchelastic and validate that
        # torch.distributed.is_torchelastic_launched() returns False

        out_file = f"{os.path.join(self.test_dir, 'out')}"

        # need to run the script with runpy in the same interpreter
        # as the test because otherwise (depending on the environment)
        # it will not find torch as a dependency
        with patch.object(
            sys,
            "argv",
            [
                path("bin/test_script_is_torchelastic_launched.py"),
                f"--out-file={out_file}",
            ],
        ):
            runpy.run_path(sys.argv[0], run_name="__main__")
            with open(out_file) as fp:
                is_torchelastic_launched = fp.readline()
                self.assertEqual("False", is_torchelastic_launched)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_init_method_tcp_with_torchelastic(self):
        # [FUNCTION DOC] Launches 4 workers of bin/test_script_init_method.py with a tcp:// init
        # method pointing at the master address and a free port. No assertions beyond completion
        # without error.
        port = get_free_port()
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=localhost",
                f"--master-port={port}",
                "--monitor-interval=1",
                path("bin/test_script_init_method.py"),
                f"--init-method=tcp://localhost:{port}",
            ]
        )
        # nothing to validate, just make sure it runs

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_init_method_env_with_torchelastic(self):
        # [FUNCTION DOC] Same as the tcp init method test but with the env:// init method;
        # success means launch.main returns without error.
        port = get_free_port()
        launch.main(
            [
                "--run-path",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=localhost",
                f"--master-port={port}",
                "--monitor-interval=1",
                path("bin/test_script_init_method.py"),
                "--init-method=env://",
            ]
        )
        # nothing to validate, just make sure it runs

    def test_capture_logs_using_default_logs_specs(self):
        # [FUNCTION DOC] Launches 4 bash workers with redirect and tee both set to 3 while the
        # TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE is patched to a rank prefix, capturing stdout.
        # Asserts the output contains the creating message prefixed by each rank's tag.
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            f"--rdzv-id={run_id}",
            "--redirect=3",
            "--tee=3",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--no-python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            with patch.dict(
                os.environ, {"TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": "[rank${rank}]: "}
            ):
                launch.main(args + script_args)

        for i in range(nproc_per_node):
            self.assertTrue(f"[rank{i}]: creating " in captured_out.getvalue())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    @skipIf(not TEST_CUDA, "requires CUDA")
    # ElasticLaunchTest is a plain TestCase, but this test launches
    # `torchrun --nproc-per-node=2` which needs 2 GPUs. That process spawning
    # happens via torchrun, not MultiProcessTestCase, so the conftest heuristic
    # (see test/conftest.py) can't detect it; mark it multigpu explicitly.
    @pytest.mark.multigpu
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_virtual_local_rank(self):
        # [FUNCTION DOC] Requires CUDA and 2 GPUs. Launches script_deviceid.py on 2 workers with
        # and without --virtual-local-rank, capturing the generated inductor code per rank.
        # Without the flag the two ranks' code must differ (different device ids); with it the
        # code must be identical. Also asserts code was captured for both ranks in both runs.
        """
        Test that virtual-local-rank ensures consistent device IDs across ranks.
        Without it, ranks may compile to different devices, leading to different code.
        """
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 2

        # Helper function to run and capture output
        def run_test(use_virtual_local_rank):
            # [FUNCTION DOC] HELPER: launches the 2 process job (optionally adding
            # --virtual-local-rank), redirecting and teeing worker output, captures stdout and
            # returns it as a string.
            args = [
                f"--nnodes={nnodes}",
                f"--nproc-per-node={nproc_per_node}",
                f"--rdzv-id={run_id}",
                "--monitor-interval=1",
                "--start-method=spawn",
                "--redirect=3",
                "--tee=3",
            ]
            if use_virtual_local_rank:
                args.append("--virtual-local-rank")

            args.append(path("script_deviceid.py"))

            captured_out = io.StringIO()
            captured_err = io.StringIO()
            with redirect_stdout(captured_out), redirect_stderr(captured_err):
                launch.main(args)

            return captured_out.getvalue()

        def split_ranks(output):
            # [FUNCTION DOC] HELPER: keeps only output lines containing cuda: and splits them
            # into two lists by the default0 / default1 rank prefix, stripping the 11 character
            # prefix. Returns the pair (rank 0 lines, rank 1 lines).
            default0 = []
            default1 = []
            for line in output.splitlines():
                if "cuda:" not in line:
                    continue
                if line.startswith("[default0]:"):
                    default0.append(line[11:])
                elif line.startswith("[default1]:"):
                    default1.append(line[11:])
            return default0, default1

        # First, run WITHOUT virtual-local-rank - outputs should differ
        output = run_test(use_virtual_local_rank=False)
        rank0, rank1 = split_ranks(output)

        # Verify we actually captured compiled code from both ranks
        self.assertGreater(
            len(rank0), 0, "Expected to capture compiled code from rank 0"
        )
        self.assertGreater(
            len(rank1), 0, "Expected to capture compiled code from rank 1"
        )

        # Without virtual-local-rank, the ranks should have DIFFERENT compiled code
        # because they see different device IDs (cuda:0 vs cuda:1)
        self.assertNotEqual(
            rank0,
            rank1,
            "Expected different compiled code without --virtual-local-rank",
        )

        # Now run WITH virtual-local-rank - outputs should be identical
        output = run_test(use_virtual_local_rank=True)
        rank0, rank1 = split_ranks(output)

        # Verify we actually captured compiled code from both ranks
        self.assertGreater(
            len(rank0),
            0,
            "Expected to capture compiled code from rank 0 with --virtual-local-rank",
        )
        self.assertGreater(
            len(rank1),
            0,
            "Expected to capture compiled code from rank 1 with --virtual-local-rank",
        )

        # With virtual-local-rank, both ranks should have IDENTICAL compiled code
        # because they both see cuda:0 during compilation
        self.assertEqual(
            rank0, rank1, "Expected identical compiled code with --virtual-local-rank"
        )


if __name__ == "__main__":
    run_tests()
