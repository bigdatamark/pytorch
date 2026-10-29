#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# [MODULE DOC] Tests for LocalElasticAgent
# (torch.distributed.elastic.agent.server.local_elastic_agent) driven end to end: worker
# entrypoints defined at module level, a Conf dataclass describing one agent, and
# LocalElasticAgentTest which launches agents in subprocesses over c10d, etcd and etcd-v2
# rendezvous backends (starting a standalone etcd server per class). Also unit tests for log prefix
# macros and uninterruptible (D state) detection with mocked processes. No import-time side effects
# beyond imports; the __main__ guard raises an error because this file is not wired into
# discover_tests.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import signal
import socket
import tempfile
import time
import unittest
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
import torch.distributed.rpc as rpc
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
    TORCHELASTIC_ENABLE_FILE_TIMER,
    TORCHELASTIC_HEALTH_CHECK_PORT,
    TORCHELASTIC_TIMER_FILE,
    TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT,
)
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.utils.process_state import read_proc_state
from torch.distributed.rpc.backend_registry import BackendType
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def init_rpc(name, backend):
    # [FUNCTION DOC] HELPER: Worker-side helper that reads RANK and WORLD_SIZE from the environment
    # (KeyError if missing) and calls rpc.init_rpc with the given name and backend.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rpc.init_rpc(
        name=name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def rpc_master(msg):
    # [FUNCTION DOC] HELPER: Worker entrypoint for rank acting as master. Initializes RPC as
    # 'master' on the tensorpipe backend, makes a synchronous RPC to 'worker' calling _echo with
    # msg, shuts RPC down and returns the echoed text with ' from worker' appended.
    init_rpc("master", BackendType.TENSORPIPE)
    ret = rpc.rpc_sync(to="worker", func=_echo, args=(msg,))
    rpc.shutdown()
    return f"{ret} from worker"


def rpc_worker():
    # [FUNCTION DOC] HELPER: Worker entrypoint that initializes RPC as 'worker' on tensorpipe and
    # immediately shuts down (waiting for outstanding calls). Returns None.
    init_rpc("worker", BackendType.TENSORPIPE)
    rpc.shutdown()


def _happy_function():
    # [FUNCTION DOC] HELPER: Worker entrypoint that does nothing and returns None.
    return


def _sad_function():
    # [FUNCTION DOC] HELPER: Worker entrypoint that always raises RuntimeError.
    raise RuntimeError("sad because i throw")


def dummy_compute() -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Worker entrypoint returning a random 100x100 tensor.
    """
    returns a predefined size random Tensor
    """
    return torch.rand(100, 100)


def dummy_compute_simulate_rank_failure() -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Worker entrypoint that kills its own process with signal 9 when RANK
    # is 1 and TORCHELASTIC_RESTART_COUNT is 0 (first attempt only); otherwise returns a random
    # 100x100 tensor. Requires both env vars (KeyError otherwise).
    """
    fails rank 1 once
    in other cases, returns a predefined size random Tensor
    """
    if os.environ["RANK"] == "1" and os.environ["TORCHELASTIC_RESTART_COUNT"] == "0":
        os.kill(os.getpid(), 9)
    return torch.rand(100, 100)


def _fatal_signal_function(expected_error_index: int, sig: int):
    # [FUNCTION DOC] HELPER: Worker entrypoint that sends the given signal to its own process if
    # its RANK equals expected_error_index; other ranks return normally.
    rank = int(os.environ["RANK"])
    if rank == expected_error_index:
        os.kill(os.getpid(), sig)


def _check_master_port_addr_override(
    expected_master_addr: str, expected_master_port: int
):
    # [FUNCTION DOC] HELPER: Worker entrypoint comparing MASTER_ADDR and MASTER_PORT from the
    # environment with the expected values.
    # KNOWN ISSUE: the condition uses a logical and, so the error is raised only if both address
    # and port differ; a mismatch of just one is not detected.
    actual_master_addr = os.environ["MASTER_ADDR"]
    actual_master_port = int(os.environ["MASTER_PORT"])
    if (
        expected_master_addr != actual_master_addr
        and expected_master_port != actual_master_port
    ):
        raise RuntimeError(
            f"Expected addr: {expected_master_addr}:{expected_master_port}, got addr: {actual_master_addr}:{actual_master_port}"
        )


def _bipolar_function():
    # [FUNCTION DOC] HELPER: Worker entrypoint where even ranks succeed and odd ranks raise
    # RuntimeError.
    rank = int(os.environ["RANK"])
    if rank % 2 == 0:
        _happy_function()
    else:
        _sad_function()


def _bipolar_sleep_function(sleep_sec):
    # [FUNCTION DOC] HELPER: Worker entrypoint where even ranks sleep for sleep_sec seconds and odd
    # ranks fail immediately with RuntimeError.
    rank = int(os.environ["RANK"])
    if rank % 2 == 0:
        _sleep(sleep_sec)
    else:
        _sad_function()


def _dist_sum(wait=0):
    # [FUNCTION DOC] HELPER: Worker entrypoint that initializes a gloo process group, sleeps for
    # the given wait, all-reduces a tensor holding its rank with SUM and raises RuntimeError if the
    # result differs from the sum of ranks 0..WORLD_SIZE-1. Requires RANK and WORLD_SIZE in the
    # environment.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.reduce_op.SUM)

    expected_sum = sum(range(world_size))
    actual = t.item()
    if expected_sum != actual:
        raise RuntimeError(f"Expected rank sum {expected_sum}, got {actual}")


def _sleep(sleep_sec) -> int:
    # [FUNCTION DOC] HELPER: Worker entrypoint that sleeps for sleep_sec seconds and returns its
    # own RANK as an int.
    time.sleep(sleep_sec)
    return int(os.environ["RANK"])


@dataclass
class RankInfo:
    # [CLASS DOC] HELPER: Plain record of the rank-related environment values (rank, role_rank,
    # group_rank, role_world_size, world_size) a worker observed; returned by workers so the test
    # can check rank assignment.
    rank: int
    role_rank: int
    group_rank: int
    role_world_size: int
    world_size: int


def _get_role_info() -> RankInfo:
    # [FUNCTION DOC] HELPER: Worker entrypoint that reads RANK, ROLE_RANK, GROUP_RANK,
    # ROLE_WORLD_SIZE and WORLD_SIZE from the environment and returns them as a RankInfo.
    rank = int(os.environ["RANK"])
    role_rank = int(os.environ["ROLE_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])
    role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    return RankInfo(rank, role_rank, group_rank, role_world_size, world_size)


def _echo(msg):
    # [FUNCTION DOC] HELPER: Returns its argument unchanged; used as an entrypoint and as an RPC
    # target.
    return msg


def _check_env_function():
    # [FUNCTION DOC] HELPER: Worker entrypoint that reads each environment variable the elastic
    # agent must provide (ranks, sizes, master address/port, restart/run-id settings, agent store
    # flag and NCCL async error handling) and fails with KeyError if any is missing.
    # just check these env vars exist, os.environ[...] will naturally throw
    # if the variable does not exist
    env_vars = [
        "RANK",
        "LOCAL_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "GROUP_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    for var in env_vars:
        _ = os.environ[var]


def _check_env_value(key: str, expected: str):
    # [FUNCTION DOC] HELPER: Worker entrypoint that raises RuntimeError if environment variable key
    # is absent or its value differs from expected.
    # checks if the env var ``key`` matches ``value``
    # this function is intended to be used as the entrypoint to the elastic run
    if key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    else:
        actual = os.getenv(key)
        if expected != actual:
            raise RuntimeError(
                f"os.environ['{key}']={actual}"
                f" does not equal the expected value: {expected}"
            )


def _check_local_watchdog_setup(key: str, should_exist: bool):
    # [FUNCTION DOC] HELPER: Worker entrypoint that raises RuntimeError if the environment variable
    # key is missing when should_exist is true, or present when should_exist is false.
    if should_exist and key not in os.environ:
        raise RuntimeError(f"Environment variable {key} not found in os.environ")
    if not should_exist and key in os.environ:
        raise RuntimeError(f"Environment variable {key} found in os.environ")


def acquire_available_port():
    # [FUNCTION DOC] HELPER: Finds a free local TCP port by binding to port 0 on each address
    # returned for localhost, reading the assigned port and closing the socket. Prints failures per
    # address; raises RuntimeError if no socket could be created. The port can be taken by another
    # process after return.
    """
    Uses sockets to acquire an available port from the os for use.

    Note: To reduce the race condition where another process grabs the port
          after this function returns an available port, we should aim to use
          the port as quickly as possible.
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError as e:
            s.close()
            print(f"Socket creation attempt failed: {e}")

    raise RuntimeError("Failed to create a socket")


@dataclass
class Conf:
    # [CLASS DOC] HELPER: Dataclass describing the work of one simulated agent: entrypoint
    # callable, local_world_size, entrypoint args, role (default 'default') and the redirects and
    # tee settings for worker output (both default Std.NONE).
    """
    Holds arguments to launch an agent (e.g. simulates an agent run on a node).

    """

    entrypoint: Callable
    local_world_size: int
    args: tuple = ()
    role: str = "default"
    redirects: Std = Std.NONE
    tee: Std = Std.NONE


class LocalElasticAgentTest(unittest.TestCase):
    # [CLASS DOC] End-to-end tests of LocalElasticAgent. Class setup starts one standalone etcd
    # server shared by all tests; each test gets a temp directory and a unique run id. Tests set
    # the rendezvous backend and endpoint through run_test_with_backend, then run one or more
    # agents (often in subprocesses) and check the RunResults. Methods without a test prefix are
    # scenario bodies and test_* methods run them per backend.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Starts a standalone single-process etcd server shared by all tests in the
        # class.
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Stops the shared etcd server.
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def setUp(self):
        # [FUNCTION DOC] Creates a temporary test directory and a short random run id for the
        # rendezvous.
        super().setUp()
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self._run_id = str(uuid.uuid4()).split("-")[0]

    def tearDown(self):
        # [FUNCTION DOC] Deletes the temporary test directory tree.
        shutil.rmtree(self._test_dir)

    def log_dir(self) -> str:
        # [FUNCTION DOC] Creates and returns a new temporary log directory under the test
        # directory.
        return tempfile.mkdtemp(prefix="torchelastic_", dir=self._test_dir)

    def get_worker_spec(
        self,
        node_config: Conf,
        min_nodes=1,
        max_nodes=1,
        max_restarts=0,
        monitor_interval=0.01,
        master_addr_override: str | None = None,
        master_port_override: int | None = None,
        is_host=True,
    ):
        # [FUNCTION DOC] Builds a WorkerSpec for the node config using a rendezvous handler created
        # from the current backend and endpoint (set by run_test_with_backend), min/max nodes and
        # is_host flag. Takes max_restarts, monitor_interval and optional master address and port
        # overrides. Returns the spec.
        rdzv_params = RendezvousParameters(
            backend=self._backend,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            is_host=is_host,
        )
        rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_params)
        return WorkerSpec(
            role=node_config.role,
            local_world_size=node_config.local_world_size,
            entrypoint=node_config.entrypoint,
            args=node_config.args,
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
            master_addr=master_addr_override,
            master_port=master_port_override,
        )

    def get_agent(
        self,
        spec: WorkerSpec,
        node_config: Conf,
        start_method: str = "spawn",
        exit_barrier_timeout=5,
        log_line_prefix_template: str | None = None,
    ) -> LocalElasticAgent:
        # [FUNCTION DOC] Creates a LocalElasticAgent for the spec with the given start method, exit
        # barrier timeout and optional log line prefix template, using DefaultLogsSpecs with a
        # fresh log dir and the redirects and tee from node_config.
        return LocalElasticAgent(
            spec,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            logs_specs=DefaultLogsSpecs(
                log_dir=self.log_dir(),
                redirects=node_config.redirects,
                tee=node_config.tee,
            ),
            log_line_prefix_template=log_line_prefix_template,
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.multiprocessing.errors.record`.
    @record
    def run_agent(
        self,
        conf: Conf,
        agent_results: mp.Queue | None = None,  # (role, agent_result)
        min_nodes=1,
        max_nodes=1,
        start_method: str = "spawn",
        max_restarts: int = 0,
        exit_barrier_timeout=5,
        master_addr_override: str | None = None,
        master_port_override: int | None = None,
        is_host=True,
        monitor_interval=0.01,
        log_line_prefix_template: str | None = None,
    ) -> RunResult | None:
        # [FUNCTION DOC] Builds a spec and agent and runs it to completion, then shuts the
        # rendezvous handler down. If agent_results is given, puts (role, result) in the queue; if
        # the run failed raises ChildFailedError built from the failures; otherwise returns the
        # RunResult (only when no queue was given). Wrapped by record so errors are written to the
        # error file.
        """
        Runs a single agent. This method can be called either on a separate process
        or the main test process. When calling this method on a separate process make
        sure to pass the ``agent_results`` multiprocessing Queue so that the agent's
        run results can be returned. If ``agent_results`` is omitted, then the
        run result is returned from the method.
        """

        spec = self.get_worker_spec(
            node_config=conf,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_restarts=max_restarts,
            master_addr_override=master_addr_override,
            master_port_override=master_port_override,
            is_host=is_host,
            monitor_interval=monitor_interval,
        )
        agent = self.get_agent(
            spec=spec,
            node_config=conf,
            start_method=start_method,
            exit_barrier_timeout=exit_barrier_timeout,
            log_line_prefix_template=log_line_prefix_template,
        )

        result = agent.run()
        spec.rdzv_handler.shutdown()

        if agent_results:
            agent_results.put((conf.role, result))

        if result.is_failed():
            raise ChildFailedError(spec.get_entrypoint_name(), result.failures)
        else:
            if not agent_results:
                return result

    def run_job(
        self,
        node_configs: list[Conf],
        exit_barrier_timeout: int = 5,
        log_line_prefix_template: str | None = None,
    ) -> dict[str, list[RunResult]]:
        # [FUNCTION DOC] Simulates a job by launching one agent process per node config, passing
        # the queue for results; min and max nodes equal the node count and only node 0 is host.
        # Waits for all processes and returns a mapping from role to the list of RunResults
        # collected. Despite the docstring, no agent runs in the main process (all are
        # subprocesses).
        """
        Simulates running a distributed job by running multiple agents
        (one on each process). Agent 0 is run on the main process for
        test coverage and ease of debugging
        """

        nnodes = len(node_configs)

        # each element in this queue holds a tuple (role, RunResult) for each agent
        agent_results = mp.Queue()

        # run first agent of first config on main process for test coverage + ease of debugging
        # it is important we loop in reverse order b/c running fn on the main process blocks
        procs = []
        for node_idx in reversed(range(len(node_configs))):
            conf = node_configs[node_idx]
            run_agent_args = {
                "conf": conf,
                "agent_results": agent_results,
                "min_nodes": nnodes,
                "max_nodes": nnodes,
                "start_method": "spawn",
                "max_restarts": 0,
                "exit_barrier_timeout": exit_barrier_timeout,
                "is_host": node_idx == 0,
                "log_line_prefix_template": log_line_prefix_template,
            }
            p = mp.Process(target=self.run_agent, kwargs=run_agent_args)
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

        results: dict[str, list[RunResult]] = {}
        while not agent_results.empty():
            role, run_result = agent_results.get()
            results.setdefault(role, []).append(run_result)
        return results

    def run_test_with_backend(self, backend: str, test_to_run: Callable):
        # [FUNCTION DOC] Sets the rendezvous backend; for etcd backends the endpoint comes from the
        # shared etcd server, otherwise a localhost endpoint with a freshly acquired free port is
        # used. Then calls test_to_run with no arguments.
        """
        Sets the backend and determines the endpoint before running the
        given test.

        Note: This method must be invoked to run any test functions that spawn
              an agent. This is because this function sets the backend and
              endpoint parameters.
        """
        self._backend = backend

        if self._backend == "etcd-v2" or self._backend == "etcd":
            self._endpoint = self._etcd_server.get_endpoint()
        else:
            # the default is c10d backend
            self._endpoint = f"localhost:{acquire_available_port()}"

        test_to_run()

    def dummy_compute(self):
        # [FUNCTION DOC] Scenario: runs two workers computing random tensors and checks the run did
        # not fail and every return value is a 100x100 tensor.
        res = self.run_agent(Conf(entrypoint=dummy_compute, local_world_size=2))
        self.assertFalse(res.is_failed())
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            self.assertEqual((100, 100), return_value.shape)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_c10d(self):
        # [FUNCTION DOC] Runs the dummy compute scenario on the c10d backend.
        self.run_test_with_backend(backend="c10d", test_to_run=self.dummy_compute)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_etcd(self):
        # [FUNCTION DOC] Runs the dummy compute scenario on the etcd backend.
        self.run_test_with_backend(backend="etcd", test_to_run=self.dummy_compute)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_dummy_compute_etcd_v2(self):
        # [FUNCTION DOC] Runs the dummy compute scenario on the etcd-v2 backend.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.dummy_compute)

    def run_happy_function(self):
        # [FUNCTION DOC] Scenario: two workers run a no-op; the run must not fail and both return
        # values are None.
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])
        self.assertIsNone(res.return_values[1])

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_c10d(self):
        # [FUNCTION DOC] Runs the happy function scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_happy_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_etcd(self):
        # [FUNCTION DOC] Runs the happy function scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_happy_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_happy_function_etcd_v2(self):
        # [FUNCTION DOC] Runs the happy function scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_happy_function
        )

    def check_master_addr_port_override(self):
        # [FUNCTION DOC] Scenario: runs one worker with master address 'test_host' and port 42
        # overrides and checks the run succeeds with a None return value.
        master_addr = "test_host"
        master_port = 42
        res = self.run_agent(
            Conf(
                entrypoint=_check_master_port_addr_override,
                args=(master_addr, master_port),
                local_world_size=1,
            ),
            master_addr_override=master_addr,
            master_port_override=master_port,
        )
        self.assertFalse(res.is_failed())
        self.assertIsNone(res.return_values[0])

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_check_master_addr_port_override_etcd(self):
        # [FUNCTION DOC] Runs the master address/port override scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.check_master_addr_port_override
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_check_master_addr_port_override_etcd_v2(self):
        # [FUNCTION DOC] Runs the master address/port override scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.check_master_addr_port_override
        )

    def run_check_env_function(self):
        # [FUNCTION DOC] Scenario: runs one worker that verifies all required environment variables
        # are set, and checks the run did not fail.
        # just checks that all env vars that we need to set on the user script
        # is actually set
        res = self.run_agent(Conf(entrypoint=_check_env_function, local_world_size=1))
        self.assertFalse(res.is_failed())

    def run_check_nccl_async_error_handling_env(self):
        # [FUNCTION DOC] Scenario: with TORCH_NCCL_ASYNC_ERROR_HANDLING patched to 0 in the parent
        # environment, the worker must see value 0 (the agent honors an existing setting).
        # make sure TORCH_NCCL_ASYNC_ERROR_HANDLING set in os.environ is honored
        with patch.dict(os.environ, {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}):
            res = self.run_agent(
                Conf(
                    entrypoint=_check_env_value,
                    local_world_size=1,
                    args=("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0"),
                )
            )
            self.assertFalse(res.is_failed())

    def run_check_nccl_async_error_handling_env_default(self):
        # [FUNCTION DOC] Scenario: when TORCH_NCCL_ASYNC_ERROR_HANDLING is not preset, the worker
        # must see the default value 1. Does not clear the variable itself, so it relies on the
        # ambient environment.
        # if not present in env var it should default to 1
        res = self.run_agent(
            Conf(
                entrypoint=_check_env_value,
                local_world_size=1,
                args=("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
            )
        )
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_enabled(self):
        # [FUNCTION DOC] Scenario: sets the file timer env variable to a temp path and checks the
        # worker environment contains it. Side effect: the variable is left set in os.environ after
        # the test.
        # KNOWN ISSUE: the environment variable is never restored or removed.
        # Set the env for watchdog
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = "/tmp/watchdog_timer_" + str(uuid.uuid4())
        os.environ[watchdog_env_name] = watchdog_file_path
        # Run the agent
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_TIMER_FILE, True),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    def run_agent_local_watchdog_setup_disabled(self):
        # [FUNCTION DOC] Scenario: removes the file timer env variable if present and checks the
        # worker environment does not contain it.
        # Do not set the env for watchdog
        watchdog_env_name = TORCHELASTIC_TIMER_FILE
        if watchdog_env_name in os.environ:
            del os.environ[watchdog_env_name]
        # Run the agent
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_TIMER_FILE, False),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_enabled_etcd(self):
        # [FUNCTION DOC] Runs the watchdog-enabled scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_enabled_c10d(self):
        # [FUNCTION DOC] Runs the watchdog-enabled scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_disabled_etcd(self):
        # [FUNCTION DOC] Runs the watchdog-disabled scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_local_watchdog_setup_disabled_c10d(self):
        # [FUNCTION DOC] Runs the watchdog-disabled scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_local_watchdog_setup_disabled
        )

    def run_agent_healthcheck_setup_enabled(self):
        # [FUNCTION DOC] Scenario: sets the health check port env variable to 12345 and checks the
        # worker environment contains it.
        # KNOWN ISSUE: the environment variable is left set in os.environ after the test.
        # Set the env for healthcheck
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        os.environ[healthcheck_port_env_name] = "12345"
        # Run the agent
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_HEALTH_CHECK_PORT, True),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    def run_agent_healthcheck_setup_disabled(self):
        # [FUNCTION DOC] Scenario: removes the health check port env variable if present and checks
        # the worker environment lacks it.
        # Do not set the env for healthcheck
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        if healthcheck_port_env_name in os.environ:
            del os.environ[healthcheck_port_env_name]
        # Run the agent
        node_conf = Conf(
            entrypoint=_check_local_watchdog_setup,
            local_world_size=1,
            args=(TORCHELASTIC_HEALTH_CHECK_PORT, False),
        )
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        res = agent.run()
        self.assertFalse(res.is_failed())

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_enabled_etcd(self):
        # [FUNCTION DOC] Runs the health check enabled scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_healthcheck_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_enabled_c10d(self):
        # [FUNCTION DOC] Runs the health check enabled scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_healthcheck_setup_enabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_disabled_etcd(self):
        # [FUNCTION DOC] Runs the health check disabled scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_agent_healthcheck_setup_disabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_agent_healthcheck_setup_disabled_c10d(self):
        # [FUNCTION DOC] Runs the health check disabled scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_agent_healthcheck_setup_disabled
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_check_env_function_etcd(self):
        # [FUNCTION DOC] Runs the required-environment-variables scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_check_env_function
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_check_nccl_async_error_handling_env_c10d(self):
        # [FUNCTION DOC] Runs the preset NCCL async error handling scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_check_nccl_async_error_handling_env
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_check_nccl_async_error_handling_env_default_c10d(self):
        # [FUNCTION DOC] Runs the default NCCL async error handling scenario on c10d.
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=self.run_check_nccl_async_error_handling_env_default,
        )

    def run_function_with_return_value(self):
        # [FUNCTION DOC] Scenario: two workers return the string 'foo'; both entries of the
        # result's return values must equal it.
        res = self.run_agent(Conf(entrypoint=_echo, args=("foo",), local_world_size=2))
        self.assertFalse(res.is_failed())
        self.assertEqual("foo", res.return_values[0])
        self.assertEqual("foo", res.return_values[1])

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_function_with_return_value_c10d(self):
        # [FUNCTION DOC] Runs the return value scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_function_with_return_value
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_function_with_return_value_etcd(self):
        # [FUNCTION DOC] Runs the return value scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_function_with_return_value
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_function_with_return_value_etcd_v2(self):
        # [FUNCTION DOC] Runs the return value scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_function_with_return_value
        )

    def simple_dist_sum(self):
        # [FUNCTION DOC] Scenario: two workers all-reduce their ranks; the worker itself validates
        # the sum, so the test only checks the run did not fail.
        res = self.run_agent(Conf(entrypoint=_dist_sum, local_world_size=2))
        self.assertFalse(res.is_failed())
        # _dist_sum internally checks that the sum computed is valid

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_simple_dist_sum_c10d(self):
        # [FUNCTION DOC] Runs the simple distributed sum scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.simple_dist_sum)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_simple_dist_sum_etcd(self):
        # [FUNCTION DOC] Runs the simple distributed sum scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.simple_dist_sum)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_simple_dist_sum_etcd_v2(self):
        # [FUNCTION DOC] Runs the simple distributed sum scenario on etcd-v2.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.simple_dist_sum)

    def run_distributed_sum_homogeneous(
        self, log_line_prefix_template: str | None = None
    ):
        # [FUNCTION DOC] Scenario: two agents with four workers each (tee to all output) run the
        # distributed sum, optionally with a log line prefix template. Checks two results for role
        # 'sum', none failed, and the union of worker ranks is 0..7.
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4, tee=Std.ALL),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=4, tee=Std.ALL),
        ]
        # When the process method is spawn, the coverage collector hangs
        # due to getting stuck on the _dist_sum in waiting for TCPStore workers
        # to join the cluster
        # TODO(aivanou): t83447589 come up with the proper fix
        res = self.run_job(
            node_configs, log_line_prefix_template=log_line_prefix_template
        )
        self.assertEqual(2, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(4 + 4)), ranks)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_c10d(self):
        # [FUNCTION DOC] Runs the homogeneous distributed sum scenario on c10d (skipped under asan
        # and tsan).
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_homogeneous
        )

    def test_run_with_custom_log_lines(self):
        # [FUNCTION DOC] Runs the homogeneous distributed sum scenario on c10d with a log prefix
        # template using hostname, role name, local rank and rank macros.
        # KNOWN ISSUE: unlike its siblings it has no skip decorator for asan or tsan.
        log_line_prefix_template = "[${hostname}-${role_name}-${local_rank}:${rank}]:"
        self.run_test_with_backend(
            backend="c10d",
            test_to_run=lambda: self.run_distributed_sum_homogeneous(
                log_line_prefix_template
            ),
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd(self):
        # [FUNCTION DOC] Runs the homogeneous distributed sum scenario on etcd (skipped under asan
        # and tsan).
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_run_distributed_sum_homogeneous_etcd_v2(self):
        # [FUNCTION DOC] Runs the homogeneous distributed sum scenario on etcd-v2 (skipped under
        # asan and tsan).
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_homogeneous
        )

    def run_distributed_sum_heterogeneous(self):
        # [FUNCTION DOC] Scenario: three agents with 1, 2 and 3 workers run the distributed sum
        # (total of ranks 0..5 is 15). Checks three results, none failed, and ranks 0..5 all
        # present.
        # sums all ranks on 3 agents; each running 1, 2, 3 workers respectively
        # sum should be equal to 0 + (1 + 2) + (3 + 4 + 5) = 15
        # sum asserted inside _dist_sum()
        node_configs = [
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=1),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=2),
            Conf(role="sum", entrypoint=_dist_sum, local_world_size=3),
        ]
        # When the process method is spawn, the coverage collector hangs
        # due to getting stuck on the _dist_sum in waiting for TCPStore workers
        # to join the cluster
        # TODO(aivanou): t83447589 come up with the proper fix
        res = self.run_job(node_configs)
        self.assertEqual(3, len(res["sum"]))
        ranks = set()
        for run_results in res["sum"]:
            self.assertFalse(run_results.is_failed())
            ranks.update(run_results.return_values.keys())
        self.assertSetEqual(set(range(1 + 2 + 3)), ranks)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_c10d(self):
        # [FUNCTION DOC] Runs the heterogeneous distributed sum scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_etcd(self):
        # [FUNCTION DOC] Runs the heterogeneous distributed sum scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_distributed_sum_heterogeneous
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_distributed_sum_heterogeneous_etcd_v2(self):
        # [FUNCTION DOC] Runs the heterogeneous distributed sum scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_distributed_sum_heterogeneous
        )

    def run_sad_function(self):
        # [FUNCTION DOC] Scenario: with TORCHELASTIC_ERROR_FILE pointing into the test dir, two
        # failing workers must make run_agent raise ChildFailedError; the first failure's rank and
        # local rank are 0 or 1, exit code 1, and its error message and timestamp match what was
        # written to the error file.
        """
        checks error propagation logic
        """
        replyfile = os.path.join(self._test_dir, "error.json")
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": replyfile}):
            with self.assertRaises(ChildFailedError) as cm:
                self.run_agent(Conf(entrypoint=_sad_function, local_world_size=2))

            rank, failure = cm.exception.get_first_failure()
            failure_data = failure.error_file_data["message"]
            with open(replyfile) as fp:
                data = json.load(fp)["message"]

                # ran two; both failed; first failure is either rank 0 or 1
                self.assertTrue(rank in {0, 1})
                self.assertTrue(failure.local_rank in {0, 1})
                self.assertEqual(1, failure.exitcode)
                self.assertEqual(data["message"], failure_data["message"])
                self.assertEqual(int(data["extraInfo"]["timestamp"]), failure.timestamp)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_c10d(self):
        # [FUNCTION DOC] Runs the failing function scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.run_sad_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_etcd(self):
        # [FUNCTION DOC] Runs the failing function scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.run_sad_function)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_sad_function_etcd_v2(self):
        # [FUNCTION DOC] Runs the failing function scenario on etcd-v2.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.run_sad_function)

    def run_bipolar_function(self):
        # [FUNCTION DOC] Scenario: four workers (half fail) with max_restarts 2. After agent.run
        # the result is failed, all restarts are consumed, the worker group state is FAILED and
        # total execution time is positive.
        """
        checks agent failure handling logic
        """
        node_conf = Conf(entrypoint=_bipolar_function, local_world_size=4)
        spec = self.get_worker_spec(node_conf, max_restarts=2)
        agent = self.get_agent(spec, node_config=node_conf)
        run_result = agent.run()
        self.assertTrue(run_result.is_failed())
        self.assertEqual(0, agent._remaining_restarts)
        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        self.assertTrue(agent._total_execution_time > 0)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_c10d(self):
        # [FUNCTION DOC] Runs the bipolar scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.run_bipolar_function
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_etcd(self):
        # [FUNCTION DOC] Runs the bipolar scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.run_bipolar_function
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_run_bipolar_function_etcd_v2(self):
        # [FUNCTION DOC] Runs the bipolar scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.run_bipolar_function
        )

    def correct_rank_assignment_heterogeneous(self):
        # [FUNCTION DOC] Scenario: seven agents across roles master (8 workers), trainer (1,2,3,4)
        # and ps (5,2) report their ranks; checks result counts per role and, via
        # assert_rank_consistency, role world sizes 8, 10 and 7.
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=8),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=2),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=5),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=2),
        ]
        results = self.run_job(node_configs)
        print(f"heterogeneous job result: {results}")
        self.assertEqual(1, len(results["master"]))
        self.assertEqual(4, len(results["trainer"]))
        self.assertEqual(2, len(results["ps"]))
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={
                "master": 8,
                "trainer": 1 + 2 + 3 + 4,
                "ps": 5 + 2,
            },
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_heterogeneous_etcd(self):
        # [FUNCTION DOC] Runs the heterogeneous rank assignment scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_heterogeneous_etcd_v2(self):
        # [FUNCTION DOC] Runs the heterogeneous rank assignment scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_heterogeneous
        )

    def correct_rank_assignment_homogeneous(self):
        # [FUNCTION DOC] Scenario: seven agents with equal sizes within each role (master 1, four
        # trainers with 4, two ps with 3) report ranks; checks result counts and role world sizes
        # 1, 16 and 6.
        node_configs = [
            Conf(role="master", entrypoint=_get_role_info, local_world_size=1),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="trainer", entrypoint=_get_role_info, local_world_size=4),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
            Conf(role="ps", entrypoint=_get_role_info, local_world_size=3),
        ]
        results = self.run_job(node_configs)
        print(f"homogeneous job result: {results}")
        self.assertEqual(1, len(results["master"]))
        self.assertEqual(4, len(results["trainer"]))
        self.assertEqual(2, len(results["ps"]))
        self.assert_rank_consistency(
            results,
            expected_role_world_sizes={"master": 1, "trainer": 4 * 4, "ps": 3 * 2},
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_homogeneous_etcd(self):
        # [FUNCTION DOC] Runs the homogeneous rank assignment scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.correct_rank_assignment_homogeneous
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_correct_rank_assignment_homogeneous_etcd_v2(self):
        # [FUNCTION DOC] Runs the homogeneous rank assignment scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.correct_rank_assignment_homogeneous
        )

    def assert_rank_consistency(
        self,
        run_results: dict[str, list[RunResult]],
        expected_role_world_sizes: dict[str, int],
    ):
        # [FUNCTION DOC] Checks the RankInfo values returned by workers: world size equals the sum
        # of expected role world sizes, each role's world size matches, global ranks are exactly
        # 0..world_size-1, role ranks per role are exactly 0..role_world_size-1, and within each
        # agent (group rank) both global and role ranks are consecutive.
        """
        Asserts that ranks are consecutive w.r.t role_rank. If local world sizes are 4:
        role_rank_0 -> ranks: 0,1,2,3
        role_rank_1 -> ranks: 4,5,6,7
        ... etc ...
        """

        global_ranks: list[int] = []
        # role -> [role_rank,...]
        role_ranks: dict[str, list[int]] = {}
        # group rank -> [(rank, role_rank),...]
        grouped_ranks: dict[int, list[tuple[int, int]]] = {}

        # global world size == sum of all the role world sizes
        expected_world_size = sum(expected_role_world_sizes.values())
        for role, results in run_results.items():
            for result in results:
                res = result.return_values
                for role_info in res.values():
                    rank = role_info.rank
                    role_rank = role_info.role_rank
                    group_rank = role_info.group_rank
                    role_world_size = role_info.role_world_size
                    world_size = role_info.world_size

                    self.assertEqual(expected_world_size, world_size)
                    self.assertEqual(expected_role_world_sizes[role], role_world_size)
                    grouped_ranks.setdefault(group_rank, []).append((rank, role_rank))
                    role_ranks.setdefault(role, []).append(role_rank)
                    global_ranks.append(rank)

        global_ranks = sorted(global_ranks)
        self.assertEqual(list(range(expected_world_size)), global_ranks)
        for role, expected_role_world_size in expected_role_world_sizes.items():
            self.assertEqual(
                list(range(expected_role_world_size)), sorted(role_ranks[role])
            )
        # Make sure that each agent assigns consecutive ranks to workers
        # The first argument is the global_rank and the second argument
        # is role_rank
        for ranks_lst in grouped_ranks.values():
            self.assert_ranks_sequential(ranks_lst, 0)
            self.assert_ranks_sequential(ranks_lst, 1)

    def assert_ranks_sequential(self, ranks_pairs, rank_idx):
        # [FUNCTION DOC] Takes (global rank, role rank) pairs and a position index; sorts that
        # component and asserts it forms a consecutive integer range.
        ranks = sorted(rank_pair[rank_idx] for rank_pair in ranks_pairs)
        start_rank, end_rank = ranks[0], ranks[-1]
        self.assertEqual(list(range(start_rank, end_rank + 1)), ranks)

    def double_agent_fault_tolerance(self):
        # [FUNCTION DOC] Scenario: starts two agent processes (max_restarts 2) running the
        # distributed sum with a wait, kills the odd-indexed one and starts a replacement, then
        # requires every final process to exit with code 0.
        """
        start ``nnodes`` agents, kill and restart odd ones, validate fault-tolerance works
        """
        nnodes = 2
        wait = 2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        agent_results = mp.Queue()
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": nnodes,
            "max_nodes": nnodes,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(nnodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # restart odd agents
        for i in range(nnodes):
            if i % 2 != 0:
                procs[i].kill()
                p = mp.Process(
                    target=self.run_agent,
                    kwargs=agent_args,
                )
                procs[i] = p
                p.start()

        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd(self):
        # [FUNCTION DOC] Runs the agent fault tolerance scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_fault_tolerance
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_fault_tolerance_etcd_v2(self):
        # [FUNCTION DOC] Runs the agent fault tolerance scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_fault_tolerance
        )

    def no_exit_barrier_on_failure(self):
        # [FUNCTION DOC] Scenario: two agents whose workers fail on odd ranks, max_restarts 0, exit
        # barrier timeout 300 and monitor interval 0.5. Both agent processes must exit with nonzero
        # code and well before the exit barrier timeout.
        # KNOWN ISSUE: the docstring is copied from the fault tolerance test, and the elapsed-time
        # variable is updated by subtracting itself from the current time, so the final timing
        # assertion does not measure what it claims.
        """
        start ``nnodes`` agents, kill and restart odd ones, validate fault-tolerance works
        """
        nnodes = 2
        wait = 20
        node_conf = Conf(
            entrypoint=_bipolar_sleep_function, args=(wait,), local_world_size=2
        )
        agent_results = mp.Queue()
        monitor_interval_s = 0.5
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": nnodes,
            "max_nodes": nnodes,
            "max_restarts": 0,
            "exit_barrier_timeout": 300,
            "monitor_interval": monitor_interval_s,
        }

        procs = []
        for _ in range(nnodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # wait for all processes to finish should not take exit barrier time
        exit_interval_between_agents = 0
        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertNotEqual(0, p.exitcode)
            exit_interval_between_agents = (
                time.monotonic() - exit_interval_between_agents
            )

        # Validate that the processes finish close to each other.
        # Using a slightly higher timeout than 2 * monitor_interval (0.01) to make it less flaky
        self.assertGreater(
            2 * monitor_interval_s,
            exit_interval_between_agents,
            "Agents are not cleaned up until 2 * monitor_interval",
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_no_exit_barrier_on_failure(self):
        # [FUNCTION DOC] Runs the no-exit-barrier-on-failure scenario on c10d (skipped under asan
        # and tsan).
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.no_exit_barrier_on_failure
        )

    def double_agent_elastic(self):
        # [FUNCTION DOC] Scenario: starts two agents (min 1, max 2 nodes, max_restarts 2), kills
        # the odd one without restart; the surviving even agent must exit 0 (scale down works) and
        # the killed one with SIGKILL.
        """
        start ``nnodes`` agents, kill odd ones (do not restart), validate
        elasticity (scale-down) works. (scale-up covered in fault_tolerance test)
        """
        min_nodes = 1
        max_nodes = 2
        wait = 2
        node_conf = Conf(entrypoint=_dist_sum, args=(wait,), local_world_size=2)
        agent_results = mp.Queue()
        agent_args = {
            "conf": node_conf,
            "agent_results": agent_results,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "max_restarts": 2,
        }

        procs = []
        for _ in range(max_nodes):
            p = mp.Process(
                target=self.run_agent,
                kwargs=agent_args,
            )
            procs.append(p)
            p.start()

        # kill odd agents
        for i in range(max_nodes):
            if i % 2 != 0:
                procs[i].kill()

        for i in range(max_nodes):
            p = procs[i]
            p.join()
            if i % 2 == 0:
                self.assertEqual(0, p.exitcode)
            else:
                self.assertEqual(-signal.SIGKILL, p.exitcode)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_c10d(self):
        # [FUNCTION DOC] Runs the elastic scale-down scenario on c10d.
        self.run_test_with_backend(
            backend="c10d", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd(self):
        # [FUNCTION DOC] Runs the elastic scale-down scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.double_agent_elastic
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_double_agent_elastic_etcd_v2(self):
        # [FUNCTION DOC] Runs the elastic scale-down scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.double_agent_elastic
        )

    def torch_rpc(self):
        # [FUNCTION DOC] Scenario: two single-worker agents, a master and a worker role, use torch
        # RPC; checks the master's only return value is the echoed message plus ' from worker'.
        """
        Simple torch rpc example with torchelastic.
        Creates two agents (to simulate two node job),
        each agent runs a single worker. worker0 calls an rpc_sync on
        worker1.
        """
        msg = "hello world"
        node_configs = [
            Conf(
                role="master",
                entrypoint=rpc_master,
                args=(msg,),
                local_world_size=1,
                tee=Std.ALL,
            ),
            Conf(
                role="worker",
                entrypoint=rpc_worker,
                args=(),
                local_world_size=1,
                tee=Std.ALL,
            ),
        ]

        results = self.run_job(node_configs)
        master_retvals = results["master"][0].return_values
        # there is only one master but the global rank is not stable
        # so compare the master return value as a collection
        self.assertEqual([f"{msg} from worker"], list(master_retvals.values()))

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_c10d(self):
        # [FUNCTION DOC] Runs the torch RPC scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd(self):
        # [FUNCTION DOC] Runs the torch RPC scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.torch_rpc)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_torch_rpc_etcd_v2(self):
        # [FUNCTION DOC] Runs the torch RPC scenario on etcd-v2.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.torch_rpc)

    def workers_drift_success(self):
        # [FUNCTION DOC] Scenario: two single-worker agents sleep 0 and 2 seconds with an exit
        # barrier timeout of 4 seconds; neither run fails and every worker returns its own rank.
        """
        two agents (one worker each) finishes within ``sec`` seconds of each other,
        exit barrier timeout set to ``sec * 2 * 2``.
        """

        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(2 * sec,), local_world_size=1),
        ]
        results = self.run_job(node_configs, exit_barrier_timeout=2 * 2 * sec)
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd(self):
        # [FUNCTION DOC] Runs the worker drift success scenario on etcd.
        self.run_test_with_backend(
            backend="etcd", test_to_run=self.workers_drift_success
        )

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_success_etcd_v2(self):
        # [FUNCTION DOC] Runs the worker drift success scenario on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_success
        )

    def workers_drift_fail(self):
        # [FUNCTION DOC] Scenario: two single-worker agents sleep 0 and 4 seconds with exit barrier
        # timeout 0; the barrier timing out must not fail the job and every worker returns its
        # rank.
        """
        two agents (one worker each) finishes within ``4 x sec`` seconds of each other,
        exit barrier timeout set to 0. Exit barriers should NOT fail the job.
        """
        sec = 1
        node_configs = [
            Conf(role="zzz", entrypoint=_sleep, args=(0 * sec,), local_world_size=1),
            Conf(role="zzz", entrypoint=_sleep, args=(4 * sec,), local_world_size=1),
        ]
        results = self.run_job(node_configs, exit_barrier_timeout=0)
        for i in range(2):
            run_results = results["zzz"][i]
            self.assertFalse(run_results.is_failed())
            for rank, output in run_results.return_values.items():
                # _sleep() returns its own rank
                self.assertEqual(rank, output)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd(self):
        # [FUNCTION DOC] Runs the worker drift scenario with expired barrier on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.workers_drift_fail)

    @unittest.skipIf(
        TEST_WITH_DEV_DBG_ASAN or TEST_WITH_TSAN,
        "test incompatible with dev/dbg asan or tsan",
    )
    def test_workers_drift_fail_etcd_v2(self):
        # [FUNCTION DOC] Runs the worker drift scenario with expired barrier on etcd-v2.
        self.run_test_with_backend(
            backend="etcd-v2", test_to_run=self.workers_drift_fail
        )

    @patch("torch.distributed.elastic.utils.store.barrier")
    def barrier_failed(self, barrier_mock):
        # [FUNCTION DOC] Scenario: the store barrier is patched to raise RuntimeError; a single
        # happy worker must still succeed and the barrier is called exactly once.
        """
        Failure during the barrier should NOT fail the job.
        """
        barrier_mock.side_effect = RuntimeError("test error")
        res = self.run_agent(Conf(entrypoint=_happy_function, local_world_size=1))
        self.assertFalse(res.is_failed())
        barrier_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_c10d(self):
        # [FUNCTION DOC] Runs the failed barrier scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.barrier_failed)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_etcd(self):
        # [FUNCTION DOC] Runs the failed barrier scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.barrier_failed)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_barrier_failed_etcd_v2(self):
        # [FUNCTION DOC] Runs the failed barrier scenario on etcd-v2.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.barrier_failed)

    @patch("torch.distributed.elastic.agent.server.local_elastic_agent.start_processes")
    def shutdown_called(self, start_processes_mock):
        # [FUNCTION DOC] Scenario: start_processes is patched to return a mock process context,
        # monitoring returns SUCCEEDED, and agent.run is called with a role argument; checks the
        # context's close method is called once.
        pcontext_mock = Mock()
        pcontext_mock.pids.return_value = {0: 0}
        start_processes_mock.return_value = pcontext_mock
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)
        spec = self.get_worker_spec(node_conf, max_restarts=0)
        agent = self.get_agent(spec, node_config=node_conf)
        with patch.object(agent, "_monitor_workers") as monitor_mock:
            monitor_mock.return_value = RunResult(
                state=WorkerState.SUCCEEDED, return_values={0: 0}
            )
            agent.run("worker")
        pcontext_mock.close.assert_called_once()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_shutdown_called_c10d(self):
        # [FUNCTION DOC] Runs the shutdown scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.shutdown_called)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_shutdown_called_etcd(self):
        # [FUNCTION DOC] Runs the shutdown scenario on etcd.
        self.run_test_with_backend(backend="etcd", test_to_run=self.shutdown_called)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_shutdown_called_etcd_v2(self):
        # [FUNCTION DOC] Runs the shutdown scenario on etcd-v2.
        self.run_test_with_backend(backend="etcd-v2", test_to_run=self.shutdown_called)

    def fail_rank_one_once(self):
        # [FUNCTION DOC] Scenario: two workers where rank 1 kills itself on the first attempt,
        # max_restarts 3; the run finally succeeds and returns 100x100 tensors.
        res = self.run_agent(
            Conf(entrypoint=dummy_compute_simulate_rank_failure, local_world_size=2),
            max_restarts=3,
        )
        self.assertFalse(res.is_failed())
        for return_value in res.return_values.values():
            self.assertIsInstance(return_value, torch.Tensor)
            self.assertEqual((100, 100), return_value.shape)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_rank_restart_after_failure(self):
        # [FUNCTION DOC] Runs the rank-one-fails-once scenario on c10d.
        self.run_test_with_backend(backend="c10d", test_to_run=self.fail_rank_one_once)

    def test_get_alive_time_without_watchdog(self):
        # [FUNCTION DOC] With a mocked rendezvous handler and no worker watchdog set,
        # _get_alive_time returns the current integer time (between before and after readings).
        """Test _get_alive_time returns current time when watchdog is not set."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        # Use a mocked rendezvous handler to avoid blocking on real connections
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)

        # Watchdog is None by default at initialization
        self.assertIsNone(agent._worker_watchdog)

        # _get_alive_time should return current time when watchdog is None
        before = int(time.time())
        alive_time = agent._get_alive_time()
        after = int(time.time())

        self.assertGreaterEqual(alive_time, before)
        self.assertLessEqual(alive_time, after)

    def test_get_alive_time_with_watchdog(self):
        # [FUNCTION DOC] With a mock watchdog installed, _get_alive_time returns the watchdog's
        # last progress time and asks it exactly once.
        """Test _get_alive_time returns watchdog time when watchdog is active."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        # Use a mocked rendezvous handler to avoid blocking on real connections
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)

        # Mock the watchdog with a specific return value
        mock_watchdog = Mock()
        expected_time = 1234567890
        mock_watchdog.get_last_progress_time.return_value = expected_time
        agent._worker_watchdog = mock_watchdog

        # _get_alive_time should delegate to watchdog when available
        alive_time = agent._get_alive_time()

        self.assertEqual(alive_time, expected_time)
        mock_watchdog.get_last_progress_time.assert_called_once()

    def test_get_alive_time_during_exit_barrier(self):
        # [FUNCTION DOC] While the exit-barrier flag is set, _get_alive_time returns the current
        # time instead of the stale watchdog time; without the flag it returns the watchdog time.
        """When _in_exit_barrier is True, _get_alive_time returns current time
        even if the watchdog exists (since watchdog progress time is stale)."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)

        # Set up a mock watchdog with a stale time
        mock_watchdog = Mock()
        mock_watchdog.get_last_progress_time.return_value = 1000  # very old
        agent._worker_watchdog = mock_watchdog

        # Without exit barrier flag, should return watchdog time
        agent._in_exit_barrier = False
        self.assertEqual(agent._get_alive_time(), 1000)

        # With exit barrier flag, should return current time (not stale watchdog time)
        agent._in_exit_barrier = True
        alive_time = agent._get_alive_time()
        now = int(time.time())
        self.assertAlmostEqual(alive_time, now, delta=2)

    def test_healthcheck_reports_healthy_during_exit_barrier(self):
        # [FUNCTION DOC] Simulates a stale watchdog (10 minutes old): alive time is stale before
        # the exit barrier, current during it, and stale again after it.
        """Verify health check callback returns current time during exit barrier,
        preventing TW from killing the task while agents coordinate shutdown."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)

        # Simulate post-training state: watchdog exists but progress time is stale
        mock_watchdog = Mock()
        stale_time = int(time.time()) - 600  # 10 minutes ago
        mock_watchdog.get_last_progress_time.return_value = stale_time
        agent._worker_watchdog = mock_watchdog

        # Before exit barrier: alive_time is stale (would fail TW health check)
        alive_time = agent._get_alive_time()
        self.assertEqual(alive_time, stale_time)

        # During exit barrier: alive_time is current (TW health check passes)
        agent._in_exit_barrier = True
        alive_time = agent._get_alive_time()
        now = int(time.time())
        self.assertAlmostEqual(alive_time, now, delta=2)

        # After exit barrier: back to watchdog time
        agent._in_exit_barrier = False
        alive_time = agent._get_alive_time()
        self.assertEqual(alive_time, stale_time)

    def test_in_exit_barrier_initialized_false(self):
        # [FUNCTION DOC] A freshly built agent has the exit-barrier flag false.
        """Verify _in_exit_barrier is False on agent construction."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)
        self.assertFalse(agent._in_exit_barrier)

    def test_setup_healthcheck_idempotent(self):
        # [FUNCTION DOC] With a health check server already set and the health check port env
        # variable set, _setup_healthcheck (justknobs patched true) must neither replace the server
        # nor start it. Restores the env variable in a finally block.
        """Test _setup_healthcheck is idempotent and does not recreate server when JK is enabled."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)

        # Use a mocked rendezvous handler to avoid blocking on real connections
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role=node_conf.role,
            local_world_size=node_conf.local_world_size,
            entrypoint=node_conf.entrypoint,
            args=node_conf.args,
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        agent = self.get_agent(spec, node_config=node_conf)

        # Pre-set a mock health check server
        mock_server = Mock()
        agent._health_check_server = mock_server

        # Set the env for healthcheck port
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        original_value = os.environ.get(healthcheck_port_env_name)
        os.environ[healthcheck_port_env_name] = "12345"

        try:
            # Patch justknobs_check to return True for the healthcheck knob
            with patch(
                "torch.distributed.elastic.agent.server.local_elastic_agent.justknobs_check",
                return_value=True,
            ):
                # Call _setup_healthcheck - should be a no-op since server already exists
                agent._setup_healthcheck()

            # Verify the mock server was not replaced or modified
            self.assertIs(agent._health_check_server, mock_server)
            # Verify start was not called on the mock (it was already "running")
            mock_server.start.assert_not_called()
        finally:
            # Restore env
            if original_value is None:
                if healthcheck_port_env_name in os.environ:
                    del os.environ[healthcheck_port_env_name]
            else:
                os.environ[healthcheck_port_env_name] = original_value

    def test_setup_healthcheck_creates_server_when_none(self):
        # [FUNCTION DOC] With a real c10d spec, a free health check port in the environment and no
        # server, _setup_healthcheck creates a health check server. Stops it and restores the env
        # variable in a finally block.
        """Test _setup_healthcheck creates server when none exists."""
        node_conf = Conf(entrypoint=_happy_function, local_world_size=1)
        self._backend = "c10d"
        self._endpoint = f"localhost:{acquire_available_port()}"
        spec = self.get_worker_spec(node_conf)
        agent = self.get_agent(spec, node_config=node_conf)

        # Ensure no health check server exists
        self.assertIsNone(agent._health_check_server)

        # Set the env for healthcheck
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        original_value = os.environ.get(healthcheck_port_env_name)
        healthcheck_port = str(acquire_available_port())
        os.environ[healthcheck_port_env_name] = healthcheck_port

        try:
            # Call _setup_healthcheck
            agent._setup_healthcheck()

            # Verify a health check server was created
            self.assertIsNotNone(agent._health_check_server)
        finally:
            # Cleanup
            if agent._health_check_server is not None:
                agent._health_check_server.stop()
                agent._health_check_server = None
            if original_value is None:
                if healthcheck_port_env_name in os.environ:
                    del os.environ[healthcheck_port_env_name]
            else:
                os.environ[healthcheck_port_env_name] = original_value

    def test_healthcheck_with_watchdog_enabled(self):
        # [FUNCTION DOC] Sets the file timer enable flag and a free health check port in the
        # environment, then runs a c10d agent (justknobs patched true) whose worker verifies the
        # flag is visible; the run must not fail. Restores both environment variables afterwards.
        """Test healthcheck works with watchdog enabled during agent run."""
        # Set the env for watchdog and healthcheck
        watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT

        original_watchdog = os.environ.get(watchdog_env_name)
        original_healthcheck = os.environ.get(healthcheck_port_env_name)

        os.environ[watchdog_env_name] = "1"
        os.environ[healthcheck_port_env_name] = str(acquire_available_port())

        try:
            node_conf = Conf(
                entrypoint=_check_local_watchdog_setup,
                local_world_size=1,
                args=(TORCHELASTIC_ENABLE_FILE_TIMER, True),
            )

            def run_test():
                # [FUNCTION DOC] HELPER: Builds a spec and agent for the enclosing test's node
                # config, runs the agent with justknobs_check patched to true and asserts the
                # result did not fail.
                spec = self.get_worker_spec(node_conf, max_restarts=0)
                agent = self.get_agent(spec, node_config=node_conf)

                # Run the agent with justknobs_check returning True
                with patch(
                    "torch.distributed.elastic.agent.server.local_elastic_agent.justknobs_check",
                    return_value=True,
                ):
                    res = agent.run()
                self.assertFalse(res.is_failed())

            self.run_test_with_backend(backend="c10d", test_to_run=run_test)
        finally:
            # Restore env
            if original_watchdog is None:
                if watchdog_env_name in os.environ:
                    del os.environ[watchdog_env_name]
            else:
                os.environ[watchdog_env_name] = original_watchdog
            if original_healthcheck is None:
                if healthcheck_port_env_name in os.environ:
                    del os.environ[healthcheck_port_env_name]
            else:
                os.environ[healthcheck_port_env_name] = original_healthcheck


class LocalElasticAgentLogPrefixTest(unittest.TestCase):
    # [CLASS DOC] Unit tests for log_line_prefix_template macro expansion. They mock out
    # start_processes and inspect the per-rank prefixes passed to it; no etcd or real rendezvous is
    # needed.
    """Unit tests for ``log_line_prefix_template`` macro substitution.

    These tests do not require etcd or a real rendezvous; they mock out
    ``start_processes`` and inspect the prefixes the agent hands to it.
    """

    def _make_agent(self, log_line_prefix_template: str) -> LocalElasticAgent:
        # [FUNCTION DOC] HELPER: Builds a LocalElasticAgent with a mocked c10d rendezvous handler,
        # two local workers, a temp log dir (removed at cleanup) and the given log line prefix
        # template.
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role="default",
            local_world_size=2,
            entrypoint=_happy_function,
            args=(),
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        log_dir = tempfile.mkdtemp(prefix="LocalElasticAgentLogPrefixTest")
        self.addCleanup(shutil.rmtree, log_dir, ignore_errors=True)
        return LocalElasticAgent(
            spec,
            start_method="spawn",
            exit_barrier_timeout=5,
            logs_specs=DefaultLogsSpecs(log_dir=log_dir),
            log_line_prefix_template=log_line_prefix_template,
        )

    @staticmethod
    def _captured_prefixes(agent: LocalElasticAgent) -> dict[int, str]:
        # [FUNCTION DOC] HELPER: Builds a worker group with fake rendezvous results (two workers,
        # world size equal to local world size), calls the agent's _start_workers with
        # start_processes patched, and returns the log_line_prefixes mapping passed to it.
        """Run ``_start_workers`` with the spawn mocked out, return the prefixes."""
        spec = agent._worker_group.spec
        worker_group = WorkerGroup(spec)
        worker_group.store = Mock()
        worker_group.group_rank = 0
        worker_group.group_world_size = 1
        worker_group.master_addr = "localhost"
        worker_group.master_port = 0
        for global_rank, worker in enumerate(worker_group.workers):
            worker.global_rank = global_rank
            worker.role_rank = global_rank
            worker.world_size = spec.local_world_size
            worker.role_world_size = spec.local_world_size

        with patch(
            "torch.distributed.elastic.agent.server.local_elastic_agent.start_processes"
        ) as mock_start_processes:
            agent._start_workers(worker_group)
        return mock_start_processes.call_args.kwargs["log_line_prefixes"]

    def test_hostname_macro(self):
        # [FUNCTION DOC] The hostname macro expands to the machine hostname in each worker's
        # prefix, together with the rank macro.
        """``${hostname}`` expands to the name of the node the agent runs on."""
        agent = self._make_agent("${hostname}:${rank}: ")
        hostname = socket.gethostname()
        self.assertEqual(
            {0: f"{hostname}:0: ", 1: f"{hostname}:1: "},
            self._captured_prefixes(agent),
        )

    def test_preexisting_macros_unchanged(self):
        # [FUNCTION DOC] A template using role name and local rank macros still renders as before
        # (e.g. default0 and default1).
        """Adding ${hostname} must not alter how existing templates render."""
        agent = self._make_agent("[${role_name}${local_rank}]:")
        self.assertEqual(
            {0: "[default0]:", 1: "[default1]:"}, self._captured_prefixes(agent)
        )


class LocalElasticAgentUninterruptibleStateTest(unittest.TestCase):
    # [CLASS DOC] Unit tests for detection of workers stuck in uninterruptible (D) state:
    # read_proc_state parsing and the agent's timeout check and monitor integration. Uses mocked
    # process contexts and no real rendezvous.
    """Unit tests for uninterruptible (D-state) detection helpers in LocalElasticAgent.

    These tests do not require etcd or a real rendezvous and directly
    exercise the helpers with a mocked process context.
    """

    def _make_agent(
        self, uninterruptible_state_timeout: float | None = None
    ) -> LocalElasticAgent:
        # [FUNCTION DOC] HELPER: Builds a LocalElasticAgent with a mocked c10d rendezvous handler,
        # one local worker, a temp log dir (removed at cleanup) and the given uninterruptible state
        # timeout (None means use env variable).
        rdzv_handler = Mock()
        rdzv_handler.get_backend.return_value = "c10d"
        spec = WorkerSpec(
            role="default",
            local_world_size=1,
            entrypoint=_happy_function,
            args=(),
            rdzv_handler=rdzv_handler,
            max_restarts=0,
            monitor_interval=0.01,
        )
        log_dir = tempfile.mkdtemp(prefix="LocalElasticAgentUninterruptibleStateTest")
        self.addCleanup(shutil.rmtree, log_dir, ignore_errors=True)
        return LocalElasticAgent(
            spec,
            start_method="spawn",
            exit_barrier_timeout=5,
            logs_specs=DefaultLogsSpecs(log_dir=log_dir),
            uninterruptible_state_timeout=uninterruptible_state_timeout,
        )

    @staticmethod
    def _stat_data(state: str) -> str:
        # [FUNCTION DOC] HELPER: Returns a fake /proc stat line for pid 1234 with the given state
        # character.
        # /proc/<pid>/stat format: pid (comm) state ppid ...
        return f"1234 (python) {state} 1 1 0 0 -1 0 0 0\n"

    def test_read_proc_state_parses_state(self):
        # [FUNCTION DOC] read_proc_state returns the state character from mocked stat file content.
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=self._stat_data("D")),
        ):
            self.assertEqual(read_proc_state(1234), "D")

    def test_read_proc_state_handles_comm_with_spaces(self):
        # [FUNCTION DOC] read_proc_state parses correctly when the command name contains spaces and
        # parentheses (uses the last closing parenthesis).
        # comm field can contain spaces and parens; parser must use last ')'
        data = "1234 (py (worker)) R 1 1 0 0 -1 0 0 0\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(read_proc_state(1234), "R")

    def test_read_proc_state_returns_none_when_missing(self):
        # [FUNCTION DOC] read_proc_state returns None when the stat file cannot be opened.
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            self.assertIsNone(read_proc_state(1234))

    def test_check_uninterruptible_state_timeout_below_threshold(self):
        # [FUNCTION DOC] The first observation of a D-state pid records its first-seen time and
        # returns None.
        agent = self._make_agent()
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: 4321}
        with mock.patch(
            "torch.distributed.elastic.agent.server.local_elastic_agent.read_proc_state",
            return_value="D",
        ):
            # First observation: just records timestamp, returns None.
            self.assertIsNone(
                agent._check_uninterruptible_state_timeout(
                    agent._worker_group, timeout=300
                )
            )
            self.assertIn(4321, agent._uninterruptible_state_first_seen)

    def test_check_uninterruptible_state_timeout_fires(self):
        # [FUNCTION DOC] A pid in D state seen 600 s ago with a 60 s timeout yields an UNHEALTHY
        # result and sets remaining restarts to 0 so the agent will not relaunch on the same host.
        agent = self._make_agent()
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: 4321}
        # Pre-seed the first-seen timestamp far in the past.
        agent._uninterruptible_state_first_seen[4321] = time.monotonic() - 600
        agent._remaining_restarts = 3
        with mock.patch(
            "torch.distributed.elastic.agent.server.local_elastic_agent.read_proc_state",
            return_value="D",
        ):
            result = agent._check_uninterruptible_state_timeout(
                agent._worker_group, timeout=60
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.state, WorkerState.UNHEALTHY)
        # D-state workers wedge the host; restarting on the same host would
        # conflict. _check_uninterruptible_state_timeout should disable
        # restarts so the supervising loop exits via _stop_workers instead
        # of relaunching.
        self.assertEqual(agent._remaining_restarts, 0)

    def test_check_uninterruptible_state_timeout_clears_on_recovery(self):
        # [FUNCTION DOC] When the pid is no longer in D state its first-seen record is removed and
        # None is returned.
        agent = self._make_agent()
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: 4321}
        agent._uninterruptible_state_first_seen[4321] = time.monotonic() - 5
        with mock.patch(
            "torch.distributed.elastic.agent.server.local_elastic_agent.read_proc_state",
            return_value="S",
        ):
            result = agent._check_uninterruptible_state_timeout(
                agent._worker_group, timeout=1
            )
        self.assertIsNone(result)
        self.assertNotIn(4321, agent._uninterruptible_state_first_seen)

    def test_check_uninterruptible_state_timeout_drops_dead_pids(self):
        # [FUNCTION DOC] Bookkeeping for pids that are no longer live is removed and None is
        # returned.
        agent = self._make_agent()
        agent._pcontext = Mock()
        # No live pids; bookkeeping for stale pid should be removed.
        agent._pcontext.pids.return_value = {}
        agent._uninterruptible_state_first_seen[9999] = time.monotonic() - 10
        result = agent._check_uninterruptible_state_timeout(
            agent._worker_group, timeout=1
        )
        self.assertIsNone(result)
        self.assertNotIn(9999, agent._uninterruptible_state_first_seen)

    def test_monitor_workers_disabled_when_arg_unset(self):
        # [FUNCTION DOC] With neither constructor argument nor environment variable the timeout is
        # 0 and _monitor_workers does not call the D-state check, reporting HEALTHY.
        # Construct with no env var and no constructor arg: check is disabled.
        env = {
            k: v
            for k, v in os.environ.items()
            if k != TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT
        }
        with mock.patch.dict(os.environ, env, clear=True):
            agent = self._make_agent()
        self.assertEqual(agent._uninterruptible_state_timeout, 0.0)
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: 4321}
        agent._pcontext.wait.return_value = None
        agent._worker_group.workers = [
            type("W", (), {"id": 4321, "global_rank": 0, "local_rank": 0})()
        ]
        with mock.patch.object(
            agent, "_check_uninterruptible_state_timeout"
        ) as check_mock:
            result = agent._monitor_workers(agent._worker_group)
        check_mock.assert_not_called()
        self.assertEqual(result.state, WorkerState.HEALTHY)

    def test_monitor_workers_calls_check_when_arg_set(self):
        # [FUNCTION DOC] With a timeout argument of 5, _monitor_workers calls the D-state check
        # once and returns its UNHEALTHY result.
        agent = self._make_agent(uninterruptible_state_timeout=5.0)
        self.assertEqual(agent._uninterruptible_state_timeout, 5.0)
        agent._pcontext = Mock()
        agent._pcontext.pids.return_value = {0: 4321}
        agent._pcontext.wait.return_value = None
        agent._worker_group.workers = [
            type("W", (), {"id": 4321, "global_rank": 0, "local_rank": 0})()
        ]
        with mock.patch.object(
            agent,
            "_check_uninterruptible_state_timeout",
            return_value=RunResult(state=WorkerState.UNHEALTHY),
        ) as check_mock:
            result = agent._monitor_workers(agent._worker_group)
        check_mock.assert_called_once()
        self.assertEqual(result.state, WorkerState.UNHEALTHY)

    def test_env_var_used_when_arg_omitted(self):
        # [FUNCTION DOC] With no constructor argument the timeout is read from the environment once
        # at construction (7.5) and later environment changes do not affect it.
        # When the constructor arg is None the env var is read exactly once.
        with mock.patch.dict(
            os.environ, {TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT: "7.5"}
        ):
            agent = self._make_agent()
        self.assertEqual(agent._uninterruptible_state_timeout, 7.5)
        # Subsequent env mutation does NOT affect the resolved timeout.
        with mock.patch.dict(
            os.environ, {TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT: "99"}
        ):
            self.assertEqual(agent._uninterruptible_state_timeout, 7.5)

    def test_arg_overrides_env_var(self):
        # [FUNCTION DOC] A constructor argument (2.0) takes precedence over the environment
        # variable value.
        with mock.patch.dict(
            os.environ, {TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT: "7.5"}
        ):
            agent = self._make_agent(uninterruptible_state_timeout=2.0)
        self.assertEqual(agent._uninterruptible_state_timeout, 2.0)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
