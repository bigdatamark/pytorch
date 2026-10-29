#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for torch.distributed.launcher.api (elastic_launch, launch_agent,
# LaunchConfig, _get_entrypoint_name). Defines worker entrypoints and helpers (rank-scaling,
# failing and all-reduce functions, config builder, multi-process wrapper) and ElasticLaunchTest,
# which starts one shared etcd server and launches real worker processes with the spawn method.
# Several tests mock the agent, rendezvous handler and event recording. Importing has no side
# effects, but running the file directly raises an error: it is not run as a script and must be
# enabled via the test discovery list.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import closing
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.distributed as dist
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.elastic.multiprocessing.api import SignalException
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.utils import get_socket_with_port
from torch.distributed.launcher.api import (
    _get_entrypoint_name,
    elastic_launch,
    launch_agent,
    LaunchConfig,
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


def path(script):
    # [FUNCTION DOC] HELPER: Returns the absolute path of the named file next to this test module.
    return os.path.join(os.path.dirname(__file__), script)


def simple_rank_scale():
    # [FUNCTION DOC] HELPER: Worker entrypoint; reads the RANK environment variable and returns
    # ten plus that rank. Raises a lookup error if RANK is unset.
    rank = int(os.environ["RANK"])
    return 10 + rank


def function_with_bug():
    # [FUNCTION DOC] HELPER: Worker entrypoint that always raises a RuntimeError with message
    # "test error". Not used by any test in this file.
    raise RuntimeError("test error")


def get_test_launch_config(
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str = "",
    rdzv_backend: str = "etcd",
    config: dict[str, Any] | None = None,
) -> LaunchConfig:
    # [FUNCTION DOC] HELPER: Builds a LaunchConfig for tests with the given node bounds,
    # processes per node, run id, rendezvous endpoint and backend (default etcd), plus extra
    # rendezvous configs copied from config. Fixed settings: 0.1s monitor interval, spawn start
    # method and zero restarts.
    rdzv_configs = {}
    if config:
        rdzv_configs.update(config)
    return LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=run_id,
        rdzv_endpoint=rdzv_endpoint,
        monitor_interval=0.1,
        rdzv_backend=rdzv_backend,
        start_method="spawn",
        max_restarts=0,
        rdzv_configs=rdzv_configs,
    )


def elastic_launch_wrapper(
    test_dir: str,
    rdzv_endpoint: str,
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str,
):
    # [FUNCTION DOC] HELPER: Process target that launches this script's test_script.py through
    # elastic_launch with the sys.executable interpreter, touching rank files in test_dir. Used to
    # simulate extra nodes in separate processes; an exception in the agent ends the process with
    # a non-zero exit code. Returns nothing.
    """A wrapper function for class `elastic_launch.` in order to make multiprocess returns correct exit code."""
    elastic_launch(
        get_test_launch_config(
            rdzv_endpoint, min_nodes, max_nodes, nproc_per_node, run_id
        ),
        sys.executable,
    )("-u", path("bin/test_script.py"), f"--touch-file-dir={test_dir}")


def _dist_sum(wait=0):
    # [FUNCTION DOC] HELPER: Worker entrypoint; initializes a gloo process group from the launcher
    # environment, optionally sleeps wait seconds, all-reduces (sum) a tensor holding this rank
    # and returns the Python number. Side effect: leaves the default process group initialized.
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.reduce_op.SUM)
    return t.item()


ELASTIC_AGENT_RUN = "torch.distributed.launcher.api.LocalElasticAgent.run"
EVENTS_RECORD = "torch.distributed.launcher.api.events.record"
GET_RDZV_HANDLER = (
    "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
)


class MockException(Exception):
    # [CLASS DOC] HELPER: Marker exception raised by mocks to check the launcher re-raises the
    # original error type.
    pass


def short_hash():
    # [FUNCTION DOC] HELPER: Returns the first dash-separated segment of a random UUID string,
    # used as a fake run id.
    return str(uuid.uuid4()).split("-")[0]


class ElasticLaunchTest(unittest.TestCase):
    # [CLASS DOC] Test case for elastic_launch and launch_agent. Class state: one standalone etcd
    # server and its endpoint, shared by all tests. Per test: a temporary directory (test_dir)
    # where workers touch files named by their global rank, and a clean PET_ environment.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Starts a single-process etcd server and stores the server and its
        # endpoint on the class for every test to use.
        # start a standalone, single process etcd server to use for all tests.
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()
        cls._etcd_endpoint = cls._etcd_server.get_endpoint()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Stops the shared etcd server.
        # stop the standalone etcd server.
        cls._etcd_server.stop()

    def setUp(self):
        # [FUNCTION DOC] Creates a fresh temporary test_dir, deletes all environment variables
        # starting with PET_, and sets TEST_SENTINEL_PARENT and OMP_NUM_THREADS in the parent
        # environment so child workers can verify propagation.
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

        # remove any lingering environment variables.
        for env in os.environ.keys():  # noqa:SIM118
            if env.startswith("PET_"):
                del os.environ[env]

        # set a sentinel env var on the parent proc.
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``.
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"
        os.environ["OMP_NUM_THREADS"] = str(1)

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temporary test_dir.
        shutil.rmtree(self.test_dir)

    def check_works_ran(self, world_size: int):
        # [FUNCTION DOC] HELPER: Asserts the files in test_dir are named exactly the ranks 0 up to
        # world_size minus one, proving each worker ran once.
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python(self):
        # [FUNCTION DOC] Launches 4 workers on one node running test_script.py with the Python
        # interpreter via etcd rendezvous, then checks every rank touched its file.
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # make sure all the workers ran.
        # each worker touches a file with its global rank as the name.
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_python_local_rank_transfer(self):
        # [FUNCTION DOC] Same launch and check as test_launch_script_python (4 workers, one node).
        # KNOWN ISSUE: the body is identical to test_launch_script_python, so despite its name it
        # does not exercise any local-rank transfer.
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        # make sure all the workers ran.
        # each worker touches a file with its global rank as the name.
        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_script_bash(self):
        # [FUNCTION DOC] Launches 4 workers running the test_script.sh shell script, passing
        # test_dir as its argument, and checks every rank touched its file.
        nnodes = 1
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            path("bin/test_script.sh"),
        )(f"{self.test_dir}")

        world_size = nnodes * nproc_per_node
        self.check_works_ran(world_size)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_function(self):
        # [FUNCTION DOC] Launches 4 workers running a Python function and checks the returned
        # mapping of rank to result, sorted, equals 10 through 13.
        nnodes = 1
        nproc_per_node = 4

        res = elastic_launch(
            get_test_launch_config(self._etcd_endpoint, nnodes, nnodes, nproc_per_node),
            simple_rank_scale,
        )()

        expected_res = [10, 11, 12, 13]
        actual_res = sorted(value for value in res.values())
        self.assertEqual(expected_res, actual_res)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_dist_sum_with_static_rdzv(self):
        # [FUNCTION DOC] Launches 4 workers with the static rendezvous backend on a free local
        # port (node rank 0), each all-reducing its rank with gloo, and checks every worker
        # returned the sum of all ranks.
        nnodes = 1
        nproc_per_node = 4
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        rdzv_endpoint = f"127.0.0.1:{master_port}"
        rank = 0
        rdzv_config = {
            "rank": rank,
        }

        res = elastic_launch(
            get_test_launch_config(
                rdzv_endpoint,
                nnodes,
                nnodes,
                nproc_per_node,
                rdzv_backend="static",
                config=rdzv_config,
            ),
            _dist_sum,
        )()

        expected_res = [sum(range(nproc_per_node))] * nproc_per_node
        actual_res = sorted(value for value in res.values())
        self.assertEqual(expected_res, actual_res)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic(self):
        # [FUNCTION DOC] Launches 4 workers with an elastic node range of 1 to 2 nodes and checks
        # every rank touched its file.
        nproc_per_node = 4

        elastic_launch(
            get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
            sys.executable,
        )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

        world_size = nproc_per_node
        self.check_works_ran(world_size)

    @mock.patch("torch.distributed.elastic.events.record")
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        # [FUNCTION DOC] Launches workers whose script is told to fail and asserts the launcher
        # raises ChildFailedError and records exactly one event (the events record is mocked).
        """
        Asserts that when the worker program fails and lancher raieses exception
        to indicate that worker process failed.
        """
        nproc_per_node = 4

        with self.assertRaises(ChildFailedError):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, nproc_per_node),
                sys.executable,
            )("-u", path("bin/test_script.py"), "--fail")

        record_mock.assert_called_once()

    @mock.patch("torch.distributed.elastic.events.record")
    @mock.patch(
        "torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run"
    )
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        # [FUNCTION DOC] Sets a side effect of MockException and expects the launcher to raise it,
        # with exactly one recorded event.
        # KNOWN ISSUE: decorator order maps the first argument to the agent run patch and the
        # second to the events record patch, but the names are swapped; the side effect and the
        # single-call assertion therefore target the opposite mocks to what the names suggest.
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception.
        """
        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 2, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_elastic_multiple_agents(self):
        # [FUNCTION DOC] Simulates 2 nodes: spawns one extra process that runs the launch wrapper
        # with a shared run id, runs the wrapper for the other node in this process, joins the
        # child and asserts it exited with code 0, then checks all 8 global ranks touched files.
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        ctx = mp.get_context("spawn")
        for _ in range(nnodes - 1):
            p = ctx.Process(
                target=elastic_launch_wrapper,
                args=(
                    self.test_dir,
                    self._etcd_endpoint,
                    min_nodes,
                    max_nodes,
                    nproc_per_node,
                    run_id,
                ),
            )
            procs.append(p)
            p.start()

        elastic_launch_wrapper(
            self.test_dir,
            self._etcd_endpoint,
            min_nodes,
            max_nodes,
            nproc_per_node,
            run_id,
        )

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        world_size = nnodes * nproc_per_node
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    def test_launch_shutdown(self, agent_mock_cls):
        # [FUNCTION DOC] With a mocked agent returning success and a mocked rendezvous handler,
        # runs elastic_launch and asserts the handler's shutdown was called exactly once.
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        with patch(
            "torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            elastic_launch(
                get_test_launch_config(self._etcd_endpoint, 1, 1, 4),
                sys.executable,
            )("-u", path("bin/test_script.py"), f"--touch-file-dir={self.test_dir}")

            rdzv_handler_mock.shutdown.assert_called_once()

    def test_get_entrypoint_name(self):
        # [FUNCTION DOC] Checks _get_entrypoint_name: a function gives its own name; a binary
        # with no args or only flag args gives an empty string; a binary plus a script argument
        # gives the script name; a missing entrypoint gives an empty string.
        self.assertEqual(
            "simple_rank_scale", _get_entrypoint_name(simple_rank_scale, [])
        )
        self.assertEqual("", _get_entrypoint_name(sys.executable, []))
        self.assertEqual("", _get_entrypoint_name(sys.executable, ["-u"]))
        self.assertEqual(
            "test_script.py",
            _get_entrypoint_name(sys.executable, ["-u", "test_script.py"]),
        )
        self.assertEqual("", _get_entrypoint_name(None, []))

    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    def test_rdzv_handler_shutdown_on_agent_signal(self, mock_get_rdzv, mock_agent_run):
        # [FUNCTION DOC] For SIGTERM and SIGINT, makes the mocked agent raise SignalException and
        # asserts launch_agent re-raises it, does not shut down the rendezvous handler, and
        # records one event.
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        for sigval in [signal.SIGTERM, signal.SIGINT]:
            with patch(EVENTS_RECORD) as record_event_mock:
                rdzv_handler_mock = MagicMock()
                rdzv_handler_mock.get_run_id.return_value = short_hash()
                mock_get_rdzv.return_value = rdzv_handler_mock

                mock_agent_run.side_effect = SignalException("test", sigval)
                with self.assertRaises(SignalException):
                    launch_agent(config, simple_rank_scale, [])
                rdzv_handler_mock.shutdown.assert_not_called()
                record_event_mock.assert_called_once()

    @patch(ELASTIC_AGENT_RUN)
    @patch(GET_RDZV_HANDLER)
    def test_rdzv_handler_shutdown_on_agent_error(self, mock_get_rdzv, mock_agent_run):
        # [FUNCTION DOC] Makes the mocked agent raise a RuntimeError and asserts launch_agent
        # re-raises it, shuts down the rendezvous handler once, and records one event.
        config = get_test_launch_config(
            self._etcd_endpoint, min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        with patch(EVENTS_RECORD) as record_event_mock:
            rdzv_handler_mock = MagicMock()
            rdzv_handler_mock.get_run_id.return_value = short_hash()
            mock_get_rdzv.return_value = rdzv_handler_mock

            mock_agent_run.side_effect = RuntimeError("any other exception")
            with self.assertRaises(RuntimeError):
                launch_agent(config, simple_rank_scale, [])
            rdzv_handler_mock.shutdown.assert_called_once()
            record_event_mock.assert_called_once()


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
