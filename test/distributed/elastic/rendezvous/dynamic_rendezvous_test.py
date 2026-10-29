# [MODULE DOC] Unit and integration tests for the dynamic rendezvous implementation in
# torch.distributed.elastic.rendezvous.dynamic_rendezvous. Covers RendezvousTimeout, node
# descriptors, the pickled rendezvous state size, the backend-backed state holder (sync,
# caching, sanitizing), the distributed op executor, the join/exit/close/keep-alive operations
# (as pure functions from state to action), the DynamicRendezvousHandler public API,
# create_handler, and multi-threaded integration runs with an in-memory backend. Test doubles
# defined here: a fake backend, a fake state holder, a dummy store and a capturing thread. Many
# tests patch the datetime, time and _delay names used inside the module.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import pickle
import socket
import threading
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock, patch, PropertyMock

import torch.distributed as dist
from torch.distributed import HashStore, Store
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    _Action,
    _BackendRendezvousStateHolder,
    _DistributedRendezvousOpExecutor,
    _NodeDesc,
    _NodeDescGenerator,
    _RendezvousCloseOp,
    _RendezvousContext,
    _RendezvousExitOp,
    _RendezvousJoinOp,
    _RendezvousKeepAliveOp,
    _RendezvousState,
    _RendezvousStateHolder,
    create_handler,
    DynamicRendezvousHandler,
    RendezvousBackend,
    RendezvousSettings,
    RendezvousTimeout,
    Token,
)


TEST_PORT = 54321
TEST_ADDR = "host"


class CustomAssertMixin:
    # [CLASS DOC] HELPER: Mixin giving test classes helpers to compare _RendezvousState objects
    # by their attribute dictionaries. Requires the host class to provide assertDictEqual.
    assertDictEqual: Callable

    def assert_state_equal(
        self, actual: _RendezvousState, expected: _RendezvousState
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts two states are equal by comparing the dictionaries of
        # their attributes.
        self.assertDictEqual(vars(actual), vars(expected))

    def assert_state_empty(self, actual: _RendezvousState) -> None:
        # [FUNCTION DOC] HELPER: Asserts the state equals a freshly constructed (empty) state.
        self.assertDictEqual(vars(actual), vars(_RendezvousState()))


class RendezvousTimeoutTest(TestCase):
    # [CLASS DOC] Tests for RendezvousTimeout construction: explicit values, defaults
    # and validation.
    def test_init_initializes_timeout(self) -> None:
        # [FUNCTION DOC] Explicit join, last_call, close and heartbeat timeouts must be stored
        # as given.
        timeout = RendezvousTimeout(
            timedelta(seconds=50),
            timedelta(seconds=60),
            timedelta(seconds=70),
            timedelta(seconds=80),
        )

        self.assertEqual(timeout.join, timedelta(seconds=50))
        self.assertEqual(timeout.last_call, timedelta(seconds=60))
        self.assertEqual(timeout.close, timedelta(seconds=70))
        self.assertEqual(timeout.heartbeat, timedelta(seconds=80))

    def test_init_initializes_timeout_if_no_timeout_is_specified(self) -> None:
        # [FUNCTION DOC] With no arguments the defaults must be join 600 s, last_call 30 s,
        # close 30 s and heartbeat 5 s.
        timeout = RendezvousTimeout()

        self.assertEqual(timeout.join, timedelta(seconds=600))
        self.assertEqual(timeout.last_call, timedelta(seconds=30))
        self.assertEqual(timeout.close, timedelta(seconds=30))
        self.assertEqual(timeout.heartbeat, timedelta(seconds=5))

    def test_init_raises_error_if_timeout_is_not_positive(self) -> None:
        # [FUNCTION DOC] A zero or negative join timeout must raise ValueError saying the join
        # timeout must be positive.
        join_timeouts = [timedelta(seconds=0), timedelta(seconds=-1)]

        for join_timeout in join_timeouts:
            with self.subTest(join_timeout=join_timeout):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The join timeout \({join_timeout}\) must be positive.$",
                ):
                    RendezvousTimeout(join_timeout)


class NodeDescTest(TestCase):
    # [CLASS DOC] Tests for the _NodeDesc node descriptor (fully qualified name, pid, local id).
    def test_repr(self) -> None:
        # [FUNCTION DOC] The repr of a descriptor must join fqdn, pid and local id
        # with underscores.
        desc = _NodeDesc("dummy_fqdn", 3, 5)

        self.assertEqual(repr(desc), "dummy_fqdn_3_5")

    def test_hash(self) -> None:
        # [FUNCTION DOC] Two descriptors differing in pid and local id can both be put in a set
        # and are found in it.
        # KNOWN ISSUE: it only checks membership of the same objects, so it does not verify hash
        # equality of equal descriptors.
        desc1 = _NodeDesc("dummy_fqdn", 2, 4)
        desc2 = _NodeDesc("dummy_fqdn", 3, 5)

        descs = {desc1, desc2}

        self.assertIn(desc1, descs)
        self.assertIn(desc2, descs)


class NodeDescGeneratorTest(TestCase):
    # [CLASS DOC] Tests for _NodeDescGenerator, which produces descriptors for the current host.
    def test_generate(self) -> None:
        # [FUNCTION DOC] Four successive generate calls must give descriptors with the machine
        # fqdn, the current process id and local ids 0 to 3 in order (checked through repr).
        desc_generator = _NodeDescGenerator()

        fqdn = socket.getfqdn()

        pid = os.getpid()

        for local_id in range(4):
            with self.subTest(fqdn=fqdn, pid=pid, local_id=local_id):
                desc = desc_generator.generate()

                self.assertEqual(repr(desc), f"{fqdn}_{pid}_{local_id}")


class RendezvousStateTest(TestCase):
    # [CLASS DOC] Tests the serialized size of _RendezvousState.
    def test_encoded_size_is_within_expected_limit(self) -> None:
        # [FUNCTION DOC] Fills a state with n running participants and n waiting nodes (with
        # heartbeats) for 10, 100, 1000 and 10000 machines in total, pickles it, base64 encodes
        # it and asserts the size stays within 2 KB, 16 KB, 160 KB and 1.6 MB respectively,
        # because backends store the encoded state.
        state = _RendezvousState()
        state.round = 1
        state.complete = True
        state.deadline = datetime.now(timezone.utc)
        state.closed = True

        # fmt: off
        expected_max_sizes = (
            (   5,    2 * (2 ** 10),),  #    10 machines <=   2KB  # noqa: E201, E241, E262
            (  50,   16 * (2 ** 10),),  #   100 machines <=  16KB  # noqa: E201, E241, E262
            ( 500,  160 * (2 ** 10),),  #  1000 machines <= 160KB  # noqa: E201, E241, E262
            (5000, 1600 * (2 ** 10),),  # 10000 machines <= 1.6MB  # noqa: E201, E241, E262
        )
        # fmt: on

        for num_nodes, max_byte_size in expected_max_sizes:
            with self.subTest(num_nodes=num_nodes, max_byte_size=max_byte_size):
                for i in range(num_nodes):
                    node_running = _NodeDesc(
                        f"dummy{i}.dummy1-dummy1-dummy1-dummy1.com", 12345, i
                    )
                    node_waiting = _NodeDesc(
                        f"dummy{i}.dummy2-dummy2-dummy2-dummy2.com", 67890, i
                    )

                    state.participants[node_running] = i

                    state.wait_list.add(node_waiting)

                    state.last_heartbeats[node_running] = datetime.now(timezone.utc)
                    state.last_heartbeats[node_waiting] = datetime.now(timezone.utc)

                bits = pickle.dumps(state)

                base64_bits = b64encode(bits)

                self.assertLessEqual(len(base64_bits), max_byte_size)


class FakeRendezvousBackend(RendezvousBackend):
    # [CLASS DOC] HELPER: In-memory RendezvousBackend for tests. Holds the state bytes and an
    # integer token starting at 0 (meaning no state yet). set_state succeeds only when the
    # given token (None counted as 0) equals the current one, then increments the token. Extra
    # helpers read, write and corrupt the stored state as a pickled _RendezvousState without a
    # token check.
    _state: bytes | None
    _token: int

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no state and token 0.
        self._state = None
        self._token = 0

    @property
    def name(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the fixed backend name "fake_backend".
        return "fake_backend"

    def get_state(self) -> tuple[bytes, Token] | None:
        # [FUNCTION DOC] HELPER: Returns None while the token is 0, otherwise the state bytes
        # and the token.
        if self._token == 0:
            return None

        return self._state, self._token  # type: ignore[return-value]

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        # [FUNCTION DOC] HELPER: Compare-and-set. If the token (None treated as 0) matches the
        # current token, stores the state and increments the token and sets the written flag;
        # otherwise leaves things unchanged. Always returns the current state, current token and
        # the flag.
        if token is None:
            token = 0

        if token == self._token:
            self._state = state
            self._token += 1

            has_set = True
        else:
            has_set = False

        return self._state, self._token, has_set  # type: ignore[return-value]

    def get_state_internal(self) -> _RendezvousState:
        # [FUNCTION DOC] HELPER: Unpickles and returns the stored state; the state must exist.
        return pickle.loads(cast(bytes, self._state))

    def set_state_internal(self, state: _RendezvousState) -> None:
        # [FUNCTION DOC] HELPER: Pickles the given state into the backend and increments the
        # token, ignoring any token check.
        self._state = pickle.dumps(state)
        self._token += 1

    def corrupt_state(self) -> None:
        # [FUNCTION DOC] HELPER: Replaces the stored state with bytes that are not a valid
        # pickle and increments the token.
        self._state = b"corrupt_state"
        self._token += 1


class BackendRendezvousStateHolderTest(TestCase, CustomAssertMixin):
    # [CLASS DOC] Tests for _BackendRendezvousStateHolder using the fake backend whose get_state
    # and set_state are wrapped in mocks so calls can be counted. Fixtures: settings with a 30 s
    # keep-alive interval, a configurable cache duration (default 0) and a patched module
    # datetime whose now returns a fixed time. The class assumes heartbeats older than 90
    # seconds are considered dead during sync.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the fake backend and replaces its get_state/set_state
        # with call-recording mocks, builds the settings, fixes the current time at 2000-01-01
        # and starts a patch of the module datetime. Sets cache duration to 0.
        super().setUp()
        self._backend = FakeRendezvousBackend()

        mock_get_state = MagicMock(wraps=self._backend.get_state)
        mock_set_state = MagicMock(wraps=self._backend.set_state)

        self._mock_backend = Mock()
        self._mock_backend.get_state = mock_get_state
        self._mock_backend.set_state = mock_set_state

        setattr(self._backend, "get_state", mock_get_state)  # noqa: B010
        setattr(self._backend, "set_state", mock_set_state)  # noqa: B010

        self._settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            timeout=RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

        self._cache_duration = 0

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.now.return_value = self._now

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the datetime patch.
        self._datetime_patch.stop()

    def _create_state(self) -> _RendezvousState:
        # [FUNCTION DOC] HELPER: Builds a complete closed state at round 999 with three
        # participants, two waiting nodes and heartbeats aged 0, 15, 30, 60 and 90 seconds
        # relative to the fixed now.
        state = _RendezvousState()
        state.round = 999
        state.complete = True
        state.deadline = self._now
        state.closed = True
        state.participants = {
            _NodeDesc("dummy1", 1, 1): 0,
            _NodeDesc("dummy2", 1, 1): 1,
            _NodeDesc("dummy3", 1, 1): 2,
        }
        state.wait_list = {
            _NodeDesc("dummy4", 1, 1),
            _NodeDesc("dummy5", 1, 1),
        }
        state.last_heartbeats = {
            _NodeDesc("dummy1", 1, 1): self._now,
            _NodeDesc("dummy2", 1, 1): self._now - timedelta(seconds=15),
            _NodeDesc("dummy3", 1, 1): self._now - timedelta(seconds=30),
            _NodeDesc("dummy4", 1, 1): self._now - timedelta(seconds=60),
            _NodeDesc("dummy5", 1, 1): self._now - timedelta(seconds=90),
        }

        return state

    def _create_state_holder(self) -> _BackendRendezvousStateHolder:
        # [FUNCTION DOC] HELPER: Constructs the state holder under test with the fake backend,
        # the settings and the current cache duration.
        return _BackendRendezvousStateHolder(
            self._backend, self._settings, self._cache_duration
        )

    def test_init_initializes_state_holder(self) -> None:
        # [FUNCTION DOC] A new holder must have an empty state and must not have touched
        # the backend.
        state_holder = self._create_state_holder()

        self.assert_state_empty(state_holder.state)

        self._mock_backend.assert_not_called()

    def test_sync_gets_empty_state_if_backend_state_does_not_exist(self) -> None:
        # [FUNCTION DOC] Syncing when the backend has no state must return None, leave the local
        # state empty, call get_state once and never call set_state.
        state_holder = self._create_state_holder()

        has_set = state_holder.sync()

        self.assertIsNone(has_set)

        self.assert_state_empty(state_holder.state)

        self.assertEqual(self._mock_backend.get_state.call_count, 1)
        self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_gets_backend_state_if_local_state_is_clean(self) -> None:
        # [FUNCTION DOC] When the local state is clean, sync must fetch the backend state each
        # time (three rounds with a changed round number), return None, mirror the backend state
        # and make one get_state call and no set_state call.
        state_holder = self._create_state_holder()

        expected_state = self._create_state()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                expected_state.round = attempt

                self._backend.set_state_internal(expected_state)

                has_set = state_holder.sync()

                self.assertIsNone(has_set)

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 1)
                self.assertEqual(self._mock_backend.set_state.call_count, 0)

                self._mock_backend.reset_mock()

    def test_sync_gets_backend_state_if_local_state_is_old_and_dirty(self) -> None:
        # [FUNCTION DOC] When the local state is dirty but the backend token has moved on, sync
        # must try set_state once, report False (not written), and adopt the backend's current
        # state without calling get_state.
        state_holder = self._create_state_holder()

        expected_state = self._create_state()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                self._backend.set_state_internal(expected_state)  # Increment token.

                state_holder.state.round = attempt
                state_holder.mark_dirty()

                has_set = state_holder.sync()

                self.assertFalse(has_set)

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                self._mock_backend.reset_mock()

    def test_sync_sets_backend_state_if_local_state_is_new_and_dirty(self) -> None:
        # [FUNCTION DOC] When the local state is dirty and the backend token is current, sync
        # must write it with one set_state call, return True and keep the state equal to what
        # the backend stores, with no get_state call.
        state_holder = self._create_state_holder()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                state_holder.state.round = attempt
                state_holder.mark_dirty()

                has_set = state_holder.sync()

                self.assertTrue(has_set)

                expected_state = self._backend.get_state_internal()

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                self._mock_backend.reset_mock()

    def test_sync_uses_cached_state_if_cache_duration_is_specified(self) -> None:
        # [FUNCTION DOC] With cache durations of 1, 5 and 10 and a patched monotonic clock, two
        # syncs at the same time (and two at the time exactly one duration later) must hit the
        # backend only once in total, returning None and never writing.
        state = self._create_state()

        self._backend.set_state_internal(state)

        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
            for cache_duration in [1, 5, 10]:
                with self.subTest(cache_duration=cache_duration):
                    self._cache_duration = cache_duration

                    state_holder = self._create_state_holder()

                    mock_time.monotonic.return_value = 5

                    state_holder.sync()

                    has_set = state_holder.sync()

                    self.assertIsNone(has_set)

                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    self.assertEqual(self._mock_backend.set_state.call_count, 0)

                    mock_time.monotonic.return_value = 5 + self._cache_duration

                    state_holder.sync()

                    has_set = state_holder.sync()

                    self.assertIsNone(has_set)

                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    self.assertEqual(self._mock_backend.set_state.call_count, 0)

                    self._mock_backend.get_state.reset_mock()

    def test_sync_gets_backend_state_if_cached_state_has_expired(self) -> None:
        # [FUNCTION DOC] With cache duration 1, a second sync before the duration passes uses
        # the cache (one get_state call); after the clock moves past the duration plus 0.01 a
        # sync refreshes and get_state has been called twice in total.
        state = self._create_state()

        self._backend.set_state_internal(state)

        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
            self._cache_duration = 1

            state_holder = self._create_state_holder()

            mock_time.monotonic.return_value = 5

            state_holder.sync()

            has_set = state_holder.sync()

            self.assertIsNone(has_set)

            self.assertEqual(self._mock_backend.get_state.call_count, 1)
            self.assertEqual(self._mock_backend.set_state.call_count, 0)

            mock_time.monotonic.return_value = 5 + self._cache_duration + 0.01

            state_holder.sync()

            has_set = state_holder.sync()

            self.assertIsNone(has_set)

            self.assertEqual(self._mock_backend.get_state.call_count, 2)
            self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_sanitizes_state(self) -> None:
        # [FUNCTION DOC] Adds dead nodes (heartbeats older than 90 seconds) to participants and
        # the wait list in the backend state; after sync, the holder's state must equal the
        # original state without the dead nodes.
        state = self._create_state()

        expected_state = copy.deepcopy(state)

        dead_node1 = _NodeDesc("dead1", 1, 1)
        dead_node2 = _NodeDesc("dead2", 1, 1)
        dead_node3 = _NodeDesc("dead3", 1, 1)
        dead_node4 = _NodeDesc("dead4", 1, 1)
        dead_node5 = _NodeDesc("dead5", 1, 1)

        state.last_heartbeats[dead_node1] = self._now - timedelta(seconds=91)
        state.last_heartbeats[dead_node2] = self._now - timedelta(seconds=100)
        state.last_heartbeats[dead_node3] = self._now - timedelta(seconds=110)
        state.last_heartbeats[dead_node4] = self._now - timedelta(seconds=120)
        state.last_heartbeats[dead_node5] = self._now - timedelta(seconds=130)

        state.participants[dead_node1] = 0
        state.participants[dead_node2] = 0
        state.participants[dead_node3] = 0

        state.wait_list.add(dead_node4)
        state.wait_list.add(dead_node5)

        self._backend.set_state_internal(state)

        state_holder = self._create_state_holder()

        state_holder.sync()

        self.assert_state_equal(state_holder.state, expected_state)

    def test_sync_sanitizes_state_if_no_participants_is_left(self) -> None:
        # [FUNCTION DOC] If every heartbeat is stale, sync must remove all participants, wait
        # list and heartbeat entries, mark the rendezvous incomplete and bump the round from 999
        # to 1000.
        state = self._create_state()

        expected_state = copy.deepcopy(state)

        for node in state.last_heartbeats:
            state.last_heartbeats[node] = self._now - timedelta(seconds=100)

        expected_state.complete = False
        expected_state.round = 1000
        expected_state.participants = {}
        expected_state.wait_list = set()
        expected_state.last_heartbeats = {}

        self._backend.set_state_internal(state)

        state_holder = self._create_state_holder()

        state_holder.sync()

        self.assert_state_equal(state_holder.state, expected_state)

    def test_sync_raises_error_if_backend_state_is_corrupt(self) -> None:
        # [FUNCTION DOC] If the backend state cannot be unpickled, sync must raise
        # RendezvousStateError with the "state is corrupt" message.
        self._backend.corrupt_state()

        state_holder = self._create_state_holder()

        with self.assertRaisesRegex(
            RendezvousStateError,
            r"^The rendezvous state is corrupt. See inner exception for details.$",
        ):
            state_holder.sync()


class FakeRendezvousStateHolder(_RendezvousStateHolder):
    # [CLASS DOC] HELPER: Fake _RendezvousStateHolder keeping a plain state object and a dirty
    # flag. sync returns whether the state was dirty (True) or None, and resets the flag;
    # mark_dirty sets it. It never talks to a backend.
    _state: _RendezvousState
    _dirty: bool | None

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty state and no dirty flag.
        self._state = _RendezvousState()
        self._dirty = None

    @property
    def state(self) -> _RendezvousState:
        # [FUNCTION DOC] HELPER: Returns the held state.
        return self._state

    @state.setter
    def state(self, value) -> None:
        # [FUNCTION DOC] HELPER: Replaces the held state with the given value.
        self._state = value

    def sync(self) -> bool | None:
        # [FUNCTION DOC] HELPER: Returns True if the state was marked dirty since the last sync,
        # otherwise None, and clears the flag.
        self._dirty, dirty = None, self._dirty

        return dirty

    def mark_dirty(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the dirty flag.
        self._dirty = True


class DistributedRendezvousOpExecutorTest(TestCase, CustomAssertMixin):
    # [CLASS DOC] Tests for _DistributedRendezvousOpExecutor, which repeatedly asks an operation
    # for the next action, applies the action to the shared state, and syncs the state holder.
    # Each test feeds a single action followed by FINISH and compares the resulting state with
    # an expected one. Uses a fake state holder with mocked sync and mark_dirty and a patched
    # datetime that returns a fixed time.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Creates this node, a fake state holder whose sync and
        # mark_dirty are wrapped in mocks (the recording mock exposes mark_dirty under the name
        # mark), default 1-node settings, default timeouts, a fixed now and the datetime patch.
        super().setUp()
        self._node = _NodeDesc("this_node", 1, 1)

        self._state_holder = FakeRendezvousStateHolder()

        mock_sync = MagicMock(wraps=self._state_holder.sync)
        mock_mark = MagicMock(wraps=self._state_holder.mark_dirty)

        self._mock_state_holder = Mock()
        self._mock_state_holder.sync = mock_sync
        self._mock_state_holder.mark = mock_mark

        setattr(self._state_holder, "sync", mock_sync)  # noqa: B010
        setattr(self._state_holder, "mark_dirty", mock_mark)  # noqa: B010

        self._state = self._state_holder.state

        self._min_nodes = 1
        self._max_nodes = 1

        self._timeout = RendezvousTimeout()

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.now.return_value = self._now

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the datetime patch.
        self._datetime_patch.stop()

    def _create_settings(self) -> RendezvousSettings:
        # [FUNCTION DOC] HELPER: Builds RendezvousSettings from the current min/max nodes and
        # timeout, a 30 s keep-alive interval and 3 keep-alive attempts.
        return RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

    def _create_op_executor(
        self, settings: RendezvousSettings | None = None
    ) -> _DistributedRendezvousOpExecutor:
        # [FUNCTION DOC] HELPER: Puts the test's state into the state holder and returns a
        # _DistributedRendezvousOpExecutor for this node, using the given settings or
        # default ones.
        self._state_holder.state = self._state

        if settings is None:
            settings = self._create_settings()

        return _DistributedRendezvousOpExecutor(
            self._node, self._state_holder, settings
        )

    def _run_action(self, action: _Action) -> None:
        # [FUNCTION DOC] HELPER: Runs the executor with a mocked operation that returns the
        # given action first and FINISH second, with deadline 1.
        op_executor = self._create_op_executor()

        op = MagicMock(side_effect=[action, _Action.FINISH])

        op_executor.run(op, deadline=1)

    def _assert_action(self, action: _Action, expected_state: _RendezvousState) -> None:
        # [FUNCTION DOC] HELPER: Runs the action, asserts the resulting state equals the
        # expected state, and asserts the state holder saw exactly sync, mark (dirty) and sync.
        self._run_action(action)

        self.assert_state_equal(self._state, expected_state)

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.mark(), call.sync()]
        )

    def test_run_passes_expected_context_and_deadline_to_state_handler(self) -> None:
        # [FUNCTION DOC] The operation must be called with a context holding this node, the
        # shared state and the settings, together with the given deadline (3).
        settings = self._create_settings()

        op_executor = self._create_op_executor(settings)

        op = MagicMock(return_value=_Action.FINISH)

        op_executor.run(op, deadline=3)

        ctx, deadline = op.call_args[0]  # args

        self.assertIs(ctx.node, self._node)
        self.assertIs(ctx.state, self._state)
        self.assertIs(ctx.settings, settings)

        self.assertEqual(deadline, 3)

    def test_run_keeps_alive(self) -> None:
        # [FUNCTION DOC] The KEEP_ALIVE action must only set this node's heartbeat to now.
        expected_state = _RendezvousState()

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.KEEP_ALIVE, expected_state)

    def test_run_adds_to_participants(self) -> None:
        # [FUNCTION DOC] ADD_TO_PARTICIPANTS on an empty state with min=max=2 must add this node
        # as a participant with rank 0 and set its heartbeat, without completing the rendezvous.
        expected_state = _RendezvousState()

        expected_state.participants[self._node] = 0

        expected_state.last_heartbeats[self._node] = self._now

        self._min_nodes = 2
        self._max_nodes = 2

        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def test_run_adds_to_participants_if_node_was_in_waitlist(self) -> None:
        # [FUNCTION DOC] Same as above, but the node was in the wait list; it must be moved from
        # the wait list to participants.
        self._state.wait_list.add(self._node)

        expected_state = _RendezvousState()

        expected_state.participants[self._node] = 0

        expected_state.last_heartbeats[self._node] = self._now

        self._min_nodes = 2
        self._max_nodes = 2

        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def _add_participants(
        self, num_participants: int, state: _RendezvousState, ranked: bool = False
    ) -> None:
        # [FUNCTION DOC] HELPER: Adds the given number of participants named dummy0.. with
        # heartbeats at now to the state. If ranked, ranks equal the index; otherwise names are
        # created in reverse order and all ranks are 0.
        for i in range(num_participants):
            if ranked:
                node = _NodeDesc(f"dummy{i}", 1, 1)
                rank = i
            else:
                node = _NodeDesc(
                    f"dummy{num_participants - i - 1}", 1, 1
                )  # Add in reverse.
                rank = 0

            state.participants[node] = rank

            state.last_heartbeats[node] = self._now

    def test_run_adds_to_participants_and_starts_last_call_if_min_nodes_is_reached(
        self,
    ) -> None:
        # [FUNCTION DOC] When adding this node brings participants to the minimum (for 0 to 2
        # existing ones, min = existing + 1, max = existing + 2), the node is added with rank 0
        # and a last call deadline of now plus the last_call timeout is set; the rendezvous is
        # not complete.
        for num_participants in range(3):
            self._state = _RendezvousState()

            self._add_participants(num_participants, self._state)

            self._state.wait_list.add(self._node)

            expected_state = _RendezvousState()

            self._add_participants(num_participants, expected_state)

            expected_state.participants[self._node] = 0

            expected_state.last_heartbeats[self._node] = self._now

            expected_state.deadline = self._now + self._timeout.last_call

            with self.subTest(num_participants=num_participants):
                self._min_nodes = num_participants + 1
                self._max_nodes = num_participants + 2

                self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                self._mock_state_holder.reset_mock()

    def test_run_adds_to_participants_and_completes_rendezvous_if_max_nodes_is_reached(
        self,
    ) -> None:
        # [FUNCTION DOC] When adding this node reaches max nodes (also with min equal to max or
        # zero), the rendezvous becomes complete: participants are ranked in sorted node order,
        # this node gets the last rank, and the last call deadline is cleared.
        for min_max_nodes_equal in [False, True]:
            for num_participants in range(3):
                rank = num_participants

                self._state = _RendezvousState()

                self._add_participants(num_participants, self._state)

                self._state.wait_list.add(self._node)

                self._state.deadline = self._now + self._timeout.last_call

                expected_state = _RendezvousState()

                self._add_participants(num_participants, expected_state, ranked=True)

                expected_state.participants[self._node] = rank

                expected_state.last_heartbeats[self._node] = self._now

                expected_state.complete = True
                expected_state.deadline = None

                with self.subTest(num_participants=num_participants):
                    self._min_nodes = num_participants + 1 if min_max_nodes_equal else 0
                    self._max_nodes = num_participants + 1

                    self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                    self._mock_state_holder.reset_mock()

    def test_run_adds_to_waitlist(self) -> None:
        # [FUNCTION DOC] ADD_TO_WAIT_LIST must put this node in the wait list and set
        # its heartbeat.
        expected_state = _RendezvousState()

        expected_state.wait_list.add(self._node)

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.ADD_TO_WAIT_LIST, expected_state)

    def test_run_removes_from_participants(self) -> None:
        # [FUNCTION DOC] REMOVE_FROM_PARTICIPANTS with two other participants must remove this
        # node and its heartbeat while keeping the complete flag, last call deadline and round
        # as they were (checked for both an incomplete and a complete rendezvous).
        for complete, last_call_deadline in [(False, self._now), (True, None)]:
            self._state = _RendezvousState()

            self._add_participants(2, self._state)

            self._state.participants[self._node] = 0

            self._state.last_heartbeats[self._node] = self._now

            self._state.complete = complete
            self._state.deadline = last_call_deadline

            self._state.round = 1

            expected_state = _RendezvousState()

            self._add_participants(2, expected_state)

            expected_state.complete = complete
            expected_state.deadline = last_call_deadline

            expected_state.round = 1

            with self.subTest(complete=complete):
                self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

                self._mock_state_holder.reset_mock()

    def test_run_removes_from_participants_and_moves_to_next_round_if_node_is_last_participant(
        self,
    ) -> None:
        # [FUNCTION DOC] Removing the last participant of a complete rendezvous must clear the
        # complete flag and increase the round from 1 to 2, leaving an otherwise empty state.
        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.complete = True

        self._state.round = 1

        expected_state = _RendezvousState()

        expected_state.complete = False

        expected_state.round = 2

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_participants_and_clears_last_call_if_rendezvous_has_less_than_min_nodes(
        self,
    ) -> None:
        # [FUNCTION DOC] Removing a participant so that fewer than min nodes remain (min 3)
        # while a last call deadline is set must clear that deadline.
        self._add_participants(2, self._state)

        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.deadline = self._now

        expected_state = _RendezvousState()

        self._add_participants(2, expected_state)

        self._min_nodes = 3
        self._max_nodes = 4

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_waitlist(self) -> None:
        # [FUNCTION DOC] REMOVE_FROM_WAIT_LIST must remove this node from the wait list and drop
        # its heartbeat, giving an empty state.
        self._state.wait_list.add(self._node)

        self._state.last_heartbeats[self._node] = self._now

        expected_state = _RendezvousState()

        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST, expected_state)

    def test_run_marks_rendezvous_closed(self) -> None:
        # [FUNCTION DOC] MARK_RENDEZVOUS_CLOSED must set the closed flag.
        expected_state = _RendezvousState()

        expected_state.closed = True

        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED, expected_state)

    def test_run_raises_error_if_rendezvous_is_closed(self) -> None:
        # [FUNCTION DOC] ERROR_CLOSED must make run raise RendezvousClosedError after only one
        # sync (the state is not marked dirty).
        with self.assertRaises(RendezvousClosedError):
            self._run_action(_Action.ERROR_CLOSED)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_raises_error_if_operation_timed_out(self) -> None:
        # [FUNCTION DOC] ERROR_TIMEOUT must make run raise RendezvousTimeoutError after only
        # one sync.
        with self.assertRaises(RendezvousTimeoutError):
            self._run_action(_Action.ERROR_TIMEOUT)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_delays_execution_if_sync_requested(self) -> None:
        # [FUNCTION DOC] The SYNC action must call the module _delay helper once with one second
        # and sync the state holder twice in total, without marking the state dirty.
        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay"
        ) as mock_delay:
            self._run_action(_Action.SYNC)

            mock_delay.assert_called_once_with(seconds=1)

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.sync()]
        )


class AbstractTestRendezvousOp(ABC):
    # [CLASS DOC] HELPER: Base mixin for tests of individual rendezvous operations. It builds a
    # state with one other participant, a node, min/max nodes (1 and 2), a 30 s keep-alive
    # interval and a deadline of 10, and patches the module datetime (fixed now) and time
    # (monotonic clock returning 10). Subclasses supply the operation via _create_op and assert
    # which action it returns for the prepared state.
    assertEqual: Callable

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the fixture described on the class and starts the
        # datetime and time patches.
        self._node = _NodeDesc("this_node", 1, 1)

        self._min_nodes = 1
        self._max_nodes = 2

        self._keep_alive_interval = timedelta(seconds=30)

        self._state = _RendezvousState()
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 1

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._deadline = 10

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.now.return_value = self._now

        self._time_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        )

        mock_time = self._time_patch.start()
        mock_time.monotonic.return_value = self._deadline

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the time and datetime patches.
        self._time_patch.stop()
        self._datetime_patch.stop()

    def _get_next_action(self) -> _Action:
        # [FUNCTION DOC] HELPER: Builds the operation, wraps the current node, state and
        # settings in a _RendezvousContext and returns the action the operation yields for the
        # current deadline.
        op = self._create_op()

        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        ctx = _RendezvousContext(self._node, self._state, settings)

        return op(ctx, self._deadline)

    @abstractmethod
    def _create_op(self) -> Callable:
        # [FUNCTION DOC] HELPER: Abstract factory for the operation under test.
        pass

    def _assert_action(self, expected_action) -> None:
        # [FUNCTION DOC] HELPER: Asserts the operation returns the expected action.
        action = self._get_next_action()

        self.assertEqual(action, expected_action)


class TestRendezvousExitOp(AbstractTestRendezvousOp, TestCase):
    # [CLASS DOC] Tests for _RendezvousExitOp: actions chosen for a node leaving the rendezvous.
    def _create_op(self) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a new _RendezvousExitOp.
        return _RendezvousExitOp()

    def test_removes_from_participants_if_node_is_participant(self) -> None:
        # [FUNCTION DOC] A node that is a participant gets REMOVE_FROM_PARTICIPANTS.
        self._state.participants[self._node] = 1

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        # [FUNCTION DOC] With the deadline already passed and the node a participant, the action
        # is ERROR_TIMEOUT.
        self._deadline = 0

        self._state.participants[self._node] = 1

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_finishes_if_node_is_not_participant(self) -> None:
        # [FUNCTION DOC] A node that is not a participant gets FINISH.
        self._assert_action(_Action.FINISH)


class TestRendezvousJoinOp(AbstractTestRendezvousOp, TestCase):
    # [CLASS DOC] Tests for _RendezvousJoinOp: the action chosen for a node trying to join given
    # state flags (closed, complete, participant, wait list, redundancy list), the node limit,
    # last call deadline, heartbeat age and operation deadline.
    def _create_op(self) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a new _RendezvousJoinOp.
        return _RendezvousJoinOp()

    def test_raises_closed_if_rendezvous_is_closed(self) -> None:
        # [FUNCTION DOC] A closed rendezvous gives ERROR_CLOSED.
        self._state.closed = True

        self._assert_action(_Action.ERROR_CLOSED)

    def test_finishes_if_rendezvous_is_complete_and_node_is_participant(self) -> None:
        # [FUNCTION DOC] A complete rendezvous in which this node is a participant gives FINISH.
        self._state.participants[self._node] = 0

        self._state.complete = True

        self._assert_action(_Action.FINISH)

    def _assert_waits_rendezvous_completion(self) -> None:
        # [FUNCTION DOC] HELPER: With the heartbeat exactly one keep-alive interval old the
        # expected action is KEEP_ALIVE; one second fresher it is SYNC.
        keep_alive_time = self._now - self._keep_alive_interval

        for delta, expected_action in [
            (timedelta(seconds=0), _Action.KEEP_ALIVE),
            (timedelta(seconds=1), _Action.SYNC),
        ]:
            self._state.last_heartbeats[self._node] = keep_alive_time + delta

            self._assert_action(expected_action)

    def test_treat_as_redundancy_for_next_rendezvous_if_rendezvous_is_complete(
        self,
    ) -> None:
        # [FUNCTION DOC] If the rendezvous is complete and already has max nodes (max 1), a new
        # node is added to the redundancy list (ADD_TO_REDUNDANCY_LIST).
        self._max_nodes = 1

        self._state.complete = True

        self._assert_action(_Action.ADD_TO_REDUNDANCY_LIST)

    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_redundant(
        self,
    ) -> None:
        # [FUNCTION DOC] A node already in the redundancy list of a full, complete rendezvous
        # just waits for the next round (KEEP_ALIVE or SYNC depending on heartbeat age).
        self._state.redundancy_list.add(self._node)

        self._max_nodes = 1

        self._state.complete = True

        self._assert_waits_rendezvous_completion()

    def test_remove_from_rednundancy_list(self) -> None:
        # [FUNCTION DOC] A node in the redundancy list of a complete rendezvous that has room
        # again (max 2) gets REMOVE_FROM_REDUNDANCY_LIST. The name has a typo (rednundancy).
        self._state.redundancy_list.add(self._node)

        self._max_nodes = 2

        self._state.complete = True

        self._assert_action(_Action.REMOVE_FROM_REDUNDANCY_LIST)

    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_in_wait_list(
        self,
    ) -> None:
        # [FUNCTION DOC] A node in the wait list of a complete rendezvous waits for the next
        # round (KEEP_ALIVE or SYNC depending on heartbeat age).
        self._state.wait_list.add(self._node)

        self._state.complete = True

        self._assert_waits_rendezvous_completion()

    def test_adds_to_wait_list_if_rendezvous_is_complete_and_num_nodes_is_less_than_max_nodes(
        self,
    ) -> None:
        # [FUNCTION DOC] In a complete rendezvous with fewer than max nodes, a node outside any
        # list gets ADD_TO_WAIT_LIST.
        self._state.complete = True

        self._assert_action(_Action.ADD_TO_WAIT_LIST)

    def test_waits_rendezvous_to_complete_if_node_is_participant(self) -> None:
        # [FUNCTION DOC] An incomplete rendezvous (max 3) in which this node is a participant
        # waits for completion while the last call deadline has not passed (KEEP_ALIVE or SYNC).
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now

        self._assert_waits_rendezvous_completion()

    def test_marks_rendezvous_complete_if_node_is_participant_and_last_call_deadline_exceeded(
        self,
    ) -> None:
        # [FUNCTION DOC] A participant in an incomplete rendezvous whose last call deadline has
        # passed gets MARK_RENDEZVOUS_COMPLETE.
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now - timedelta(seconds=1)

        self._assert_action(_Action.MARK_RENDEZVOUS_COMPLETE)

    def test_adds_to_participants(self) -> None:
        # [FUNCTION DOC] A new node joining an open rendezvous gets ADD_TO_PARTICIPANTS.
        self._assert_action(_Action.ADD_TO_PARTICIPANTS)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        # [FUNCTION DOC] With the deadline already passed and the node not yet in any list, the
        # action is ERROR_TIMEOUT.
        self._deadline = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_participant(
        self,
    ) -> None:
        # [FUNCTION DOC] If the deadline is 0 (past the rollback deadline too) and the node is a
        # participant, the action is ERROR_TIMEOUT.
        self._deadline = 0

        self._state.participants[self._node] = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_in_wait_list(
        self,
    ) -> None:
        # [FUNCTION DOC] If the deadline is 0 (past the rollback deadline too) and the node is
        # in the wait list, the action is ERROR_TIMEOUT.
        self._deadline = 0

        self._state.wait_list.add(self._node)

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_removes_from_participants_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        # [FUNCTION DOC] If the deadline has passed (5) but the rollback deadline has not, a
        # participant node is rolled back with REMOVE_FROM_PARTICIPANTS.
        self._deadline = 5

        self._state.participants[self._node] = 0

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)

    def test_removes_from_wait_list_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        # [FUNCTION DOC] If the deadline has passed (5) but the rollback deadline has not, a
        # node in the wait list is rolled back with REMOVE_FROM_WAIT_LIST.
        self._deadline = 5

        self._state.wait_list.add(self._node)

        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST)

    def test_no_timeout_for_redundant_node(self) -> None:
        # [FUNCTION DOC] A node in the redundancy list never times out, even at deadline 0; the
        # action is SYNC.
        self._max_nodes = 1
        self._deadline = 0
        self._state.complete = True

        self._state.redundancy_list.add(self._node)

        self._assert_action(_Action.SYNC)

    def test_keep_alive_for_redundant_node(self) -> None:
        # [FUNCTION DOC] A redundant node with an old enough heartbeat gets KEEP_ALIVE even when
        # the deadline has passed.
        self._deadline = 0
        self._max_nodes = 1
        self._state.complete = True

        self._state.redundancy_list.add(self._node)

        keep_alive_time = self._now - self._keep_alive_interval
        self._state.last_heartbeats[self._node] = keep_alive_time
        self._assert_action(_Action.KEEP_ALIVE)


class TestRendezvousCloseOp(AbstractTestRendezvousOp, TestCase):
    # [CLASS DOC] Tests for _RendezvousCloseOp.
    def _create_op(self) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a new _RendezvousCloseOp.
        return _RendezvousCloseOp()

    def test_finishes_if_rendezvous_is_closed(self) -> None:
        # [FUNCTION DOC] An already closed rendezvous gives FINISH.
        self._state.closed = True

        self._assert_action(_Action.FINISH)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        # [FUNCTION DOC] With the deadline passed and the rendezvous open, the action
        # is ERROR_TIMEOUT.
        self._deadline = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_marks_rendezvous_closed(self) -> None:
        # [FUNCTION DOC] An open rendezvous within the deadline gives MARK_RENDEZVOUS_CLOSED.
        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED)


class TestRendezvousKeepAliveOp(AbstractTestRendezvousOp, TestCase):
    # [CLASS DOC] Tests for _RendezvousKeepAliveOp, which refreshes a node's heartbeat when it
    # is at least one keep-alive interval old.
    def _create_op(self) -> Callable:
        # [FUNCTION DOC] HELPER: Returns a new _RendezvousKeepAliveOp.
        return _RendezvousKeepAliveOp()

    def test_updates_keep_alive_if_needed(self) -> None:
        # [FUNCTION DOC] When the heartbeat is exactly one interval old or older, the action
        # is KEEP_ALIVE.
        keep_alive_time = self._now - self._keep_alive_interval

        for delta in [timedelta(seconds=0), timedelta(seconds=-1)]:
            with self.subTest(delta=delta):
                self._state.last_heartbeats[self._node] = keep_alive_time + delta

                self._assert_action(_Action.KEEP_ALIVE)

    def test_raises_timeout_if_deadlined_exceeded(self) -> None:
        # [FUNCTION DOC] An old heartbeat with an expired deadline gives ERROR_TIMEOUT.
        self._deadline = 0

        self._state.last_heartbeats[self._node] = self._now - self._keep_alive_interval

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_finishes_if_no_keep_alive_update_is_needed(self) -> None:
        # [FUNCTION DOC] A heartbeat one second newer than the interval needs no update; the
        # action is FINISH.
        delta = timedelta(seconds=1)

        self._state.last_heartbeats[self._node] = (
            self._now - self._keep_alive_interval + delta
        )

        self._assert_action(_Action.FINISH)


class DummyStore(Store):
    # [CLASS DOC] HELPER: Store subclass used by handler tests; it adds only a fixed port
    # property. Tests replace its get and set with mocks.
    @property
    def port(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the constant TEST_PORT.
        return TEST_PORT


class DynamicRendezvousHandlerTest(TestCase):
    # [CLASS DOC] Tests for DynamicRendezvousHandler using a fake state holder (with mocked
    # sync), a dummy store with mocked get and set, and DynamicRendezvousHandler's creation of a
    # TCP store server patched to return a dummy store. Attributes such as the join, close and
    # heartbeat timeouts and min/max nodes are changed per test before the handler is created.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the fixture (node, one-node limits, unset timeouts,
        # dummy store with mocked get returning b"123" and mocked set, fake state holder with
        # mocked sync, patch of _create_tcp_store_server that returns a dummy store); the
        # patcher is stopped via addCleanup.
        super().setUp()
        self._node = _NodeDesc("this_node", 1, 1)

        self._min_nodes = 1
        self._max_nodes = 1

        self._join_timeout: timedelta | None = None
        self._close_timeout: timedelta | None = None
        self._heartbeat_timeout: timedelta | None = None

        self._keep_alive_interval = timedelta(seconds=30)

        self._store = DummyStore()

        self._mock_store_get = MagicMock(return_value=b"123")
        self._mock_store_set = MagicMock()

        setattr(self._store, "get", self._mock_store_get)  # noqa: B010
        setattr(self._store, "set", self._mock_store_set)  # noqa: B010

        self._state_holder = FakeRendezvousStateHolder()

        self._mock_sync = MagicMock(wraps=self._state_holder.sync)

        setattr(self._state_holder, "sync", self._mock_sync)  # noqa: B010

        self._state = self._state_holder.state

        self._tcp_store_mock = DummyStore()

        patcher = patch.object(
            DynamicRendezvousHandler,
            "_create_tcp_store_server",
            return_value=self._tcp_store_mock,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _create_handler(self) -> DynamicRendezvousHandler:
        # [FUNCTION DOC] HELPER: Builds settings from the current fixture values (3 keep-alive
        # attempts), installs the test state into the state holder and returns a handler for the
        # node, backend name dummy_backend, store and state holder.
        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(
                join=self._join_timeout,
                close=self._close_timeout,
                heartbeat=self._heartbeat_timeout,
            ),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        self._state_holder.state = self._state

        return DynamicRendezvousHandler(
            self._node, settings, "dummy_backend", self._store, self._state_holder
        )

    def test_share_store_creates_tcp_store(self):
        # [FUNCTION DOC] With RendezvousStoreInfo.build patched, next_rendezvous must expose
        # the bootstrap address and port from it, and the handler must hold the mocked TCP
        # store server both after the first and second call (the store server is created and
        # then kept).
        handler = self._create_handler()

        shared_store_info = RendezvousStoreInfo(TEST_ADDR, TEST_PORT)
        with patch.object(RendezvousStoreInfo, "build", return_value=shared_store_info):
            rdzv_info = handler.next_rendezvous()
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, TEST_ADDR)
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, TEST_PORT)
        self.assertEqual(handler._shared_tcp_store_server, self._tcp_store_mock)

        rdzv_info = handler.next_rendezvous()
        self.assertEqual(handler._shared_tcp_store_server, self._tcp_store_mock)

    def test_share_store_when_tcp_store(self):
        # [FUNCTION DOC] With the handler's store replaced by a mock TCP store (port one less
        # than TEST_PORT) and PrefixStore replaced by a fake, next_rendezvous (called twice)
        # must report the address and port read through the prefix store, and the shared TCP
        # store server must differ from the handler's own store.
        handler = self._create_handler()

        class CustomPrefixStore(Mock):
            def get(self, key):
                # [FUNCTION DOC] HELPER: Fake store get; returns the host name bytes for the key
                # MASTER_ADDR and the port number as bytes for any other key.
                return (
                    TEST_ADDR.encode("utf-8")
                    if key == "MASTER_ADDR"
                    else bytes(str(TEST_PORT), "utf-8")
                )

            def set(self, key, value):
                # [FUNCTION DOC] HELPER: Fake store set; does nothing.
                pass

        with patch.object(dist, "PrefixStore", new=CustomPrefixStore):
            handler._store = Mock(spec=dist.TCPStore)
            type(handler._store).host = PropertyMock(return_value=TEST_ADDR)
            type(handler._store).port = PropertyMock(return_value=TEST_PORT - 1)
            rdzv_info = handler.next_rendezvous()
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, TEST_ADDR)
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, TEST_PORT)
            self.assertNotEqual(handler._shared_tcp_store_server, handler._store)

            rdzv_info = handler.next_rendezvous()
            self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, TEST_ADDR)
            self.assertEqual(rdzv_info.bootstrap_store_info.master_port, TEST_PORT)
            self.assertNotEqual(handler._shared_tcp_store_server, handler._store)

    def test_share_store_creates_tcp_store_when_rank_changes_to_zero(self):
        # [FUNCTION DOC] Starts with another participant sorted first, so this node gets rank 1
        # and no shared TCP store server is created. After that participant is removed and the
        # settings change to one node, the next rendezvous gives this node rank 0 and the
        # handler creates the shared TCP store server.
        self._max_nodes = 2
        other_node = _NodeDesc("aaa_node", 1, 1)
        self._state.participants[other_node] = 0
        self._state.last_heartbeats[other_node] = datetime.now(timezone.utc)

        handler = self._create_handler()

        shared_store_info = RendezvousStoreInfo(TEST_ADDR, TEST_PORT)
        with patch.object(RendezvousStoreInfo, "build", return_value=shared_store_info):
            rdzv_info = handler.next_rendezvous()

        self.assertEqual(rdzv_info.rank, 1)
        self.assertIsNone(handler._shared_tcp_store_server)

        del self._state.participants[other_node]
        del self._state.last_heartbeats[other_node]
        self._min_nodes = 1
        self._max_nodes = 1
        handler._settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            timeout=RendezvousTimeout(),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        rdzv_info = handler.next_rendezvous()

        self.assertEqual(rdzv_info.rank, 0)
        self.assertEqual(handler._shared_tcp_store_server, self._tcp_store_mock)

    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay")
    def test_next_rendezvous_skews_the_first_join_attempt(self, mock_delay) -> None:
        # [FUNCTION DOC] Patches the module _delay. next_rendezvous must call _delay once when
        # the round is 0 (to skew the first join attempt) and not at all when the round is 1.
        for round, expected_call_count in [(0, True), (1, False)]:
            with self.subTest(round=round):
                self._state.round = round

                handler = self._create_handler()

                handler.next_rendezvous()

                self.assertEqual(mock_delay.call_count, expected_call_count)

                mock_delay.reset_mock()

    def test_next_rendezvous_returns_expected_value(self) -> None:
        # [FUNCTION DOC] With two other participants and max 3, next_rendezvous must give rank 2
        # and world size 3, and the returned store must prefix keys with the run id and round
        # ("torch.rendezvous.dummy_run_id.0/") when forwarding to the underlying store.
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 0
        self._state.participants[_NodeDesc("dummy2", 1, 1)] = 0

        self._max_nodes = 3

        handler = self._create_handler()

        rdzv_info = handler.next_rendezvous()

        self.assertEqual(rdzv_info.rank, 2)
        self.assertEqual(rdzv_info.world_size, 3)

        _ = rdzv_info.store.get("dummy_key")

        self._mock_store_get.assert_called_with(
            "torch.rendezvous.dummy_run_id.0/dummy_key"
        )

    def test_next_rendezvous_respects_the_requested_timeout(self) -> None:
        # [FUNCTION DOC] When every state sync takes 0.3 s and the join timeout is 0.2 s,
        # next_rendezvous must raise RendezvousTimeoutError.
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._join_timeout = timedelta(seconds=0.2)

        handler = self._create_handler()

        with self.assertRaises(RendezvousTimeoutError):
            handler.next_rendezvous()

    def test_next_rendezvous_moves_to_next_round_if_called_repeatedly(self) -> None:
        # [FUNCTION DOC] Calling next_rendezvous four times on one handler must advance the
        # state round to 0, 1, 2 and 3 after each call.
        handler = self._create_handler()

        for i in range(4):
            handler.next_rendezvous()

            self.assertEqual(self._state.round, i)

    def test_is_closed_returns_expected_value(self) -> None:
        # [FUNCTION DOC] is_closed must return the state's closed flag (False and True) and sync
        # the state exactly once per call.
        for closed in [False, True]:
            with self.subTest(closed=closed):
                self._state.closed = closed

                handler = self._create_handler()

                self.assertEqual(handler.is_closed(), closed)

                self._mock_sync.assert_called_once()

                self._mock_sync.reset_mock()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_is_closed_records_and_raises_exceptions(self, record_mock) -> None:
        # [FUNCTION DOC] When sync raises RendezvousError, is_closed must propagate it.
        # KNOWN ISSUE: the record_mock assertion sits after the call inside the raises block, so
        # it is never executed; event recording is not actually verified.
        self._mock_sync.side_effect = RendezvousError("test error")
        handler = self._create_handler()
        with self.assertRaises(RendezvousError):
            handler.is_closed()
            record_mock.assert_called_once()

    def test_set_closed_closes_rendezvous(self) -> None:
        # [FUNCTION DOC] set_closed must mark the state closed.
        handler = self._create_handler()

        handler.set_closed()

        self.assertTrue(self._state.closed)

    def test_set_closed_respects_the_requested_timeout(self) -> None:
        # [FUNCTION DOC] With slow syncs (0.3 s) and a close timeout of 0.2 s, set_closed must
        # raise RendezvousTimeoutError.
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._close_timeout = timedelta(seconds=0.2)

        handler = self._create_handler()

        with self.assertRaises(RendezvousTimeoutError):
            handler.set_closed()

    def test_set_closed_can_be_called_multiple_times(self) -> None:
        # [FUNCTION DOC] Calling set_closed twice must not fail and the state stays closed.
        handler = self._create_handler()

        handler.set_closed()
        handler.set_closed()

        self.assertTrue(self._state.closed)

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_set_closed_records_and_raises_exceptions(self, record_mock) -> None:
        # [FUNCTION DOC] When the handler's _close raises RendezvousError, set_closed must
        # propagate it.
        # KNOWN ISSUE: the record_mock assertion is after the raising call inside the raises
        # block and never runs.
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            close_mock.side_effect = RendezvousError("test error")
            handler = self._create_handler()
            with self.assertRaises(RendezvousError):
                handler.set_closed()
                record_mock.assert_called_once()

    def test_num_nodes_waiting_returns_expected_value(self) -> None:
        # [FUNCTION DOC] num_nodes_waiting must return the wait list size (2) and sync once.
        self._state.wait_list.add(_NodeDesc("dummy1", 1, 1))
        self._state.wait_list.add(_NodeDesc("dummy2", 1, 1))

        handler = self._create_handler()

        self.assertEqual(handler.num_nodes_waiting(), 2)

        self._mock_sync.assert_called_once()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_num_nodes_waiting_records_and_raises_exceptions(self, record_mock) -> None:
        # [FUNCTION DOC] When sync raises RendezvousError, num_nodes_waiting must propagate it.
        # KNOWN ISSUE: the record_mock assertion is after the raising call inside the raises
        # block and never runs.
        self._mock_sync.side_effect = RendezvousError("test error")
        handler = self._create_handler()
        with self.assertRaises(RendezvousError):
            handler.num_nodes_waiting()
            record_mock.assert_called_once()

    def test_shutdown_closes_rendezvous_and_returns_true(self) -> None:
        # [FUNCTION DOC] shutdown must close the rendezvous and return True.
        handler = self._create_handler()

        result = handler.shutdown()

        self.assertTrue(result)

        self.assertTrue(self._state.closed)

    def test_shutdown_returns_false_if_rendezvous_cannot_be_closed(self) -> None:
        # [FUNCTION DOC] If closing fails with RendezvousError, shutdown must swallow it and
        # return False.
        self._mock_sync.side_effect = [RendezvousError]

        handler = self._create_handler()

        result = handler.shutdown()

        self.assertFalse(result)

    def test_shutdown_can_be_called_multiple_times(self) -> None:
        # [FUNCTION DOC] Calling shutdown twice must not fail and the state stays closed.
        handler = self._create_handler()

        handler.shutdown()
        handler.shutdown()

        self.assertTrue(self._state.closed)

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_shutdown_records_and_raises_exceptions(self, record_mock) -> None:
        # [FUNCTION DOC] When _close raises RuntimeError, shutdown must propagate it.
        # KNOWN ISSUE: the record_mock assertion is after the raising call inside the raises
        # block and never runs.
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            close_mock.side_effect = RuntimeError("test error")
            handler = self._create_handler()
            with self.assertRaises(RuntimeError):
                handler.shutdown()
                record_mock.assert_called_once()

    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime")
    def test_keep_alive_updates_last_heartbeat(self, mock_datetime) -> None:
        # [FUNCTION DOC] With the module datetime patched to a fixed time and this node's
        # heartbeat older than twice the keep-alive interval, calling _keep_alive must set the
        # heartbeat to the fixed now.
        now = datetime(2000, 1, 1, hour=0, minute=0)

        mock_datetime.now.return_value = now

        self._state.last_heartbeats[self._node] = now - (self._keep_alive_interval * 2)

        handler = self._create_handler()

        handler._keep_alive()

        self.assertEqual(self._state.last_heartbeats[self._node], now)

    def _assert_keep_alive_swallows_rendezvous_errors(self) -> None:
        # [FUNCTION DOC] HELPER: Sets an old heartbeat, creates a handler, calls _keep_alive
        # and asserts the heartbeat is unchanged, meaning errors or timeouts during keep-alive
        # were swallowed.
        last_heartbeat_time = datetime.now(timezone.utc) - (
            self._keep_alive_interval * 2
        )

        self._state.last_heartbeats[self._node] = last_heartbeat_time

        handler = self._create_handler()

        handler._keep_alive()

        self.assertEqual(self._state.last_heartbeats[self._node], last_heartbeat_time)

    def test_keep_alive_swallows_rendezvous_errors(self) -> None:
        # [FUNCTION DOC] If sync raises RendezvousError, _keep_alive must swallow it and leave
        # the heartbeat unchanged.
        self._mock_sync.side_effect = [RendezvousError]

        self._assert_keep_alive_swallows_rendezvous_errors()

    def test_keep_alive_respects_the_requested_timeout(self) -> None:
        # [FUNCTION DOC] With 0.3 s syncs and a heartbeat timeout of 0.2 s, _keep_alive must
        # swallow the timeout and leave the heartbeat unchanged.
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._heartbeat_timeout = timedelta(seconds=0.2)

        self._assert_keep_alive_swallows_rendezvous_errors()

    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_shutdown(
        self,
    ) -> None:
        # [FUNCTION DOC] For a node with local id 2, no thread named RendezvousKeepAliveTimer_2
        # exists before next_rendezvous, one exists after it, and none exists after shutdown.
        self._node = _NodeDesc("this_node", 1, 2)

        name = "RendezvousKeepAliveTimer_2"

        handler = self._create_handler()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

        handler.next_rendezvous()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        handler.shutdown()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_finalizer(
        self,
    ) -> None:
        # [FUNCTION DOC] For a node with local id 3, the keep-alive timer thread named
        # RendezvousKeepAliveTimer_3 starts with next_rendezvous and is gone after the handler
        # is deleted (finalizer stops it).
        self._node = _NodeDesc("this_node", 1, 3)

        name = "RendezvousKeepAliveTimer_3"

        handler = self._create_handler()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

        handler.next_rendezvous()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        del handler

        self.assertTrue(all(t.name != name for t in threading.enumerate()))


class DummyRendezvousBackend(RendezvousBackend):
    # [CLASS DOC] HELPER: Backend stub named "dummy_backend" with no state: get_state returns
    # None and set_state returns None.
    @property
    def name(self):
        # [FUNCTION DOC] HELPER: Returns the backend name dummy_backend.
        return "dummy_backend"

    def get_state(self):
        # [FUNCTION DOC] HELPER: Always reports that no state exists (returns None).
        return None

    def set_state(self, state, token):
        # [FUNCTION DOC] HELPER: Ignores the state and token and returns None.
        return None


class DynamicRendezvousHandlerFromBackendTest(TestCase):
    # [CLASS DOC] Tests for DynamicRendezvousHandler.from_backend: stored settings and
    # argument validation.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Sets run id, dummy store and backend, min 3, max 6 and a
        # default timeout object.
        super().setUp()
        self._run_id = "dummy_run_id"
        self._store = DummyStore()
        self._backend = DummyRendezvousBackend()
        self._min_nodes = 3
        self._max_nodes = 6
        self._timeout: RendezvousTimeout | None = RendezvousTimeout()

    def _create_handler(self) -> DynamicRendezvousHandler:
        # [FUNCTION DOC] HELPER: Calls from_backend with the current fixture values and returns
        # the handler.
        return DynamicRendezvousHandler.from_backend(
            run_id=self._run_id,
            store=self._store,
            backend=self._backend,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
        )

    def test_init_initializes_handler(self) -> None:
        # [FUNCTION DOC] The handler must report the backend name and run id, and its settings
        # must hold the run id, min and max nodes; the timeout must be the object passed in, or
        # any non-None default when no timeout was given.
        handler = self._create_handler()

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._run_id)
        self.assertEqual(handler.settings.run_id, self._run_id)
        self.assertEqual(handler.settings.min_nodes, self._min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._max_nodes)

        if self._timeout is None:
            self.assertIsNotNone(handler.settings.timeout)
        else:
            self.assertIs(handler.settings.timeout, self._timeout)

    def test_init_initializes_handler_if_timeout_is_not_specified(self) -> None:
        # [FUNCTION DOC] With no timeout argument, a default timeout is created (re-runs the
        # initialization checks).
        self._timeout = None

        self.test_init_initializes_handler()

    def test_init_initializes_handler_if_min_and_max_nodes_are_equal(self) -> None:
        # [FUNCTION DOC] Min and max nodes of 3 and 3 must be accepted (re-runs the checks).
        self._min_nodes = 3
        self._max_nodes = 3

        self.test_init_initializes_handler()

    def test_init_raises_error_if_min_nodes_is_not_positive(self) -> None:
        # [FUNCTION DOC] A min nodes value of 0 or -10 must raise ValueError saying it must be
        # greater than zero.
        for num in [0, -10]:
            with self.subTest(min_nodes=num):
                self._min_nodes = num

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of nodes \({num}\) must be greater than zero.$",
                ):
                    self._create_handler()

    def test_init_raises_error_if_max_nodes_is_less_than_min(self) -> None:
        # [FUNCTION DOC] Max nodes less than min (2 versus 3) must raise ValueError saying max
        # must be at least the minimum.
        self._min_nodes = 3
        self._max_nodes = 2

        with self.assertRaisesRegex(
            ValueError,
            rf"^The maximum number of nodes \({self._max_nodes}\) must be greater than or equal to "
            "the minimum number of nodes "
            rf"\({self._min_nodes}\).$",
        ):
            self._create_handler()


class CreateHandlerTest(TestCase):
    # [CLASS DOC] Tests for create_handler, which builds a DynamicRendezvousHandler from
    # RendezvousParameters (timeouts as strings of seconds, local address, node limits).
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a dummy store, backend and parameters (min 3, max 6,
        # join 50, last call 60, close 70) plus the expected RendezvousTimeout.
        super().setUp()
        self._store = DummyStore()

        self._backend = DummyRendezvousBackend()

        self._params = RendezvousParameters(
            backend=self._backend.name,
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=3,
            max_nodes=6,
            join_timeout="50",
            last_call_timeout="60",
            close_timeout="70",
        )

        self._expected_timeout = RendezvousTimeout(
            timedelta(seconds=50), timedelta(seconds=60), timedelta(seconds=70)
        )

    def test_create_handler_returns_handler(self) -> None:
        # [FUNCTION DOC] The handler must expose the backend name, run id, min and max nodes and
        # the join, last call and close timeouts from the parameters.
        handler = create_handler(self._store, self._backend, self._params)

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._params.run_id)
        self.assertEqual(handler.settings.min_nodes, self._params.min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._params.max_nodes)
        self.assertEqual(handler.settings.timeout.join, self._expected_timeout.join)
        self.assertEqual(
            handler.settings.timeout.last_call, self._expected_timeout.last_call
        )
        self.assertEqual(handler.settings.timeout.close, self._expected_timeout.close)

    def test_create_handler_returns_handler_if_timeout_is_not_specified(self) -> None:
        # [FUNCTION DOC] With the three timeout options removed, the handler must use the
        # default timeouts (re-runs the checks).
        del self._params.config["join_timeout"]
        del self._params.config["last_call_timeout"]
        del self._params.config["close_timeout"]

        self._expected_timeout = RendezvousTimeout()

        self.test_create_handler_returns_handler()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_create_handler_records_and_raises_exceptions(self, record_mock) -> None:
        # [FUNCTION DOC] When from_backend raises RendezvousError, create_handler must
        # propagate it.
        # KNOWN ISSUE: the record_mock assertion is after the raising call inside the raises
        # block and never runs.
        with patch.object(DynamicRendezvousHandler, "from_backend") as from_mock:
            from_mock.side_effect = RendezvousError("test error")
            with self.assertRaises(RendezvousError):
                create_handler(self._store, self._backend, self._params)
                record_mock.assert_called_once()

    def test_create_handler_rdzv_local_addr(self) -> None:
        # [FUNCTION DOC] With local_addr 127.0.0.2 in the parameters and a hash store,
        # next_rendezvous must report 127.0.0.2 as the bootstrap master address.
        params = RendezvousParameters(
            backend=self._backend.name,
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            join_timeout="50",
            last_call_timeout="60",
            close_timeout="70",
            local_addr="127.0.0.2",
        )
        store = HashStore()
        handler = create_handler(store, self._backend, params)
        rdzv_info = handler.next_rendezvous()
        self.assertEqual(rdzv_info.bootstrap_store_info.master_addr, "127.0.0.2")


def _ignore_exception(exception_type: Exception, fn: Callable):
    # [FUNCTION DOC] HELPER: Calls fn and swallows exceptions of the given type; other
    # exceptions propagate.
    # KNOWN ISSUE: the annotation says the parameter is an instance of Exception, but it is used
    # as an exception class.
    try:
        fn()
    except exception_type:
        pass


def _wait_for(condition, timeout=10, interval=1, name=None):
    # [FUNCTION DOC] HELPER: Polls the condition every interval seconds on a background thread
    # and waits up to timeout seconds for that thread to finish. Returns nothing and does not
    # raise if the condition never becomes true.
    # KNOWN ISSUE: the polling thread is not a daemon and is not stopped on timeout, so it can
    # keep polling forever after a timeout.
    def _wait_while():
        # [FUNCTION DOC] HELPER: Loop run by the waiter thread; returns as soon as the condition
        # is true, otherwise sleeps for the interval and retries.
        while True:
            if condition():
                break
            else:
                time.sleep(interval)

    wait_thread = threading.Thread(target=_wait_while, name=name)
    wait_thread.start()
    wait_thread.join(timeout=timeout)


class _CapturingThread(threading.Thread):
    # [CLASS DOC] HELPER: Thread that stores the return value of its target so join can return
    # it. Exceptions raised by the target are not captured and surface in the thread only.
    def __init__(self, target=None, name=None, args=None, kwargs=None):
        # [FUNCTION DOC] HELPER: Defaults args to an empty tuple and kwargs to an empty mapping,
        # calls the Thread constructor and initializes the result to None.
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        threading.Thread.__init__(
            self, target=target, args=args, kwargs=kwargs, name=name
        )
        self._result = None

    def run(self):
        # [FUNCTION DOC] HELPER: Runs the target, if any, and saves its return value.
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        # [FUNCTION DOC] HELPER: Joins the thread (forwarding the timeout argument) and returns
        # the saved result, which is None if the thread has not finished or failed.
        threading.Thread.join(self, *args)
        return self._result


class IntegrationTest(TestCase):
    # [CLASS DOC] Integration tests that run several DynamicRendezvousHandler instances in
    # threads against a shared hash store and an in-memory backend. Handlers get distinct local
    # addresses 127.0.0.<n> and are tracked so heartbeats can be stopped in tearDown.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the hash store, an empty handler list and the
        # in-memory backend.
        super().setUp()
        self._store = HashStore()
        self._handlers = []
        self._backend = _InMemoryRendezvousBackend()

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the keep-alive heartbeat threads of every
        # handler created.
        for handler in self._handlers:
            handler._stop_heartbeats()

    def _create_handler(self, **kwargs) -> DynamicRendezvousHandler:
        # [FUNCTION DOC] HELPER: Builds parameters with defaults (min and max nodes 2, join
        # timeout 5, local address by handler count) updated by the keyword overrides, creates
        # a handler with create_handler on the shared store and backend, records it and
        # returns it.
        params = {
            "backend": self._backend.name,
            "endpoint": "dummy_endpoint",
            "run_id": "dummy_run_id",
            "min_nodes": 2,
            "max_nodes": 2,
            "join_timeout": "5",
            "local_addr": f"127.0.0.{len(self._handlers)}",
        }
        params.update(**kwargs)

        rzdv_params = RendezvousParameters(**params)

        handler = create_handler(self._store, self._backend, rzdv_params)
        self._handlers.append(handler)
        return handler

    def test_all_nodes_join_rendezvous(self) -> None:
        # [FUNCTION DOC] Two handlers (min=max=2) run next_rendezvous in threads. Both results
        # must share the same underlying store, have different ranks and world size 2.
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        rdzv_info1: RendezvousInfo = handler1_thread.join()
        rdzv_info2: RendezvousInfo = handler2_thread.join()
        self.assertEqual(rdzv_info1.store.underlying_store, self._store)
        self.assertEqual(rdzv_info2.store.underlying_store, self._store)

        self.assertNotEqual(rdzv_info1.rank, rdzv_info2.rank)

        self.assertEqual(rdzv_info1.world_size, 2)
        self.assertEqual(rdzv_info2.world_size, 2)

    def test_redundancy_list(self) -> None:
        # [FUNCTION DOC] Two handlers complete a rendezvous with max 2; a third handler's
        # attempt then registers in the redundancy list (its timeout error is ignored). After
        # waiting for that, the redundancy list must hold only the address 127.0.0.2.
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)
        handler3 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)
        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        handler1_thread.start()
        handler2_thread.start()

        # establish successful rendezvous
        handler1_thread.join()
        handler2_thread.join()

        # expect to register in redundancy list
        handler3_thread.start()

        # wait until the handler3 is registered in the redundancy list
        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        state_and_token = self._backend.get_state()
        state = pickle.loads(state_and_token[0])
        addresses = [node.addr for node in state.redundancy_list]
        self.assertListEqual(addresses, ["127.0.0.2"])

    def test_redundancy_transition_to_wait_list_then_join_rendezvous(self) -> None:
        # [FUNCTION DOC] Three handlers (min 1, max 2, one with a 1 s keep-alive interval): two
        # rendezvous, the third becomes redundant. After stopping the second's heartbeats the
        # test waits until participants drop to 1 and the wait list holds 1, i.e. the redundant
        # node is promoted to the wait list.
        handler1 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )
        handler2 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
            keep_alive_interval=timedelta(seconds=1),
        )
        handler3 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        handler1_thread.start()
        handler2_thread.start()

        # establish successful rendezvous
        handler1_thread.join()
        handler2_thread.join()

        handler3_thread.start()

        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        handler2._stop_heartbeats()

        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).participants) == 1
        )
        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).wait_list) == 1
        )

    def test_use_agent_store_is_true_by_default(self):
        # [FUNCTION DOC] By default the handler must use the agent store (use_agent_store true).
        handler = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        self.assertTrue(handler.use_agent_store)

    @patch.dict(os.environ, {"TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"})
    def test_use_agent_store_is_disabled(self):
        # [FUNCTION DOC] With the environment variable TORCH_DISABLE_SHARE_RDZV_TCP_STORE set to
        # 1, use_agent_store must be false.
        handler = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        self.assertFalse(handler.use_agent_store)

    @patch.object(dist, "PrefixStore")
    def test_share_tcp_store_from_backend(self, prefix_store_class_mock):
        # [FUNCTION DOC] With PrefixStore mocked and the provided store a mock TCP store, two
        # handlers rendezvous; both must return the prefix store, which is created with prefix
        # "torch.rendezvous.dummy_run_id.0" over the TCP store, and share the same bootstrap
        # info with the address and port read from that prefix store.
        expected_addr = "expected_address"
        expected_port = 54231

        class CustomPrefixStore(Mock):
            def get(self, key):
                # [FUNCTION DOC] HELPER: Fake store get; returns the expected address bytes for
                # key MASTER_ADDR and the expected port as bytes for any other key.
                return (
                    expected_addr.encode("utf-8")
                    if key == "MASTER_ADDR"
                    else bytes(str(expected_port), "utf-8")
                )

            def set(self, key, value):
                # [FUNCTION DOC] HELPER: Fake store set; does nothing.
                pass

        prefix_store = CustomPrefixStore()
        prefix_store_class_mock.return_value = prefix_store
        tcp_store = Mock(spec=dist.TCPStore)
        original_addr = "original_addr"
        original_port = TEST_PORT
        type(tcp_store).host = PropertyMock(return_value=original_addr)
        type(tcp_store).port = PropertyMock(return_value=original_port)
        # this will be injected
        self._store = tcp_store

        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        rdzv_info1: RendezvousInfo = handler1_thread.join()
        rdzv_info2: RendezvousInfo = handler2_thread.join()

        self.assertEqual(rdzv_info1.store, prefix_store)
        self.assertEqual(rdzv_info2.store, prefix_store)
        prefix_store_class_mock.assert_called_with(
            "torch.rendezvous.dummy_run_id.0", tcp_store
        )

        self.assertEqual(
            rdzv_info1.bootstrap_store_info, rdzv_info2.bootstrap_store_info
        )

        self.assertEqual(rdzv_info1.bootstrap_store_info.master_addr, expected_addr)
        self.assertEqual(rdzv_info1.bootstrap_store_info.master_port, expected_port)

    @patch.dict(os.environ, {"TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"})
    @patch.object(dist, "PrefixStore")
    def test_share_tcp_store_is_disabled(self, prefix_store_class_mock):
        # [FUNCTION DOC] With sharing of the TCP store disabled by environment variable, two
        # handlers rendezvous using a mocked prefix store whose get returns b"123": both get the
        # prefix store (built over the injected store) and bootstrap master port 123.
        prefix_store = Mock()
        prefix_store_class_mock.return_value = prefix_store

        prefix_store.set.return_value = None
        prefix_store.get.return_value = b"123"
        tcp_store = Mock(spec=dist.TCPStore)
        # this will be injected
        self._store = tcp_store

        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        rdzv_info1: RendezvousInfo = handler1_thread.join()
        rdzv_info2: RendezvousInfo = handler2_thread.join()

        self.assertEqual(rdzv_info1.store, prefix_store)
        self.assertEqual(rdzv_info2.store, prefix_store)
        prefix_store_class_mock.assert_called_with(
            "torch.rendezvous.dummy_run_id.0", self._store
        )
        self.assertEqual(rdzv_info1.bootstrap_store_info.master_port, 123)
        self.assertEqual(rdzv_info2.bootstrap_store_info.master_port, 123)


class _InMemoryRendezvousBackend(RendezvousBackend):
    # [CLASS DOC] HELPER: Thread-safe in-memory RendezvousBackend. State and token start as
    # None; the token becomes 0 on the first write and is incremented on each later one. A write
    # is accepted only if the given token equals the current token.
    # KNOWN ISSUE: set_state returns None on success as well as on failure instead of the
    # expected state, token and written flag, so it never reports a successful write.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a lock and sets state and token to None.
        self._lock = threading.Lock()
        self._state = None
        self._token = None

    @property
    def name(self):
        # [FUNCTION DOC] HELPER: Returns the backend name _in_memory_backend.
        return "_in_memory_backend"

    def get_state(self):
        # [FUNCTION DOC] HELPER: Under the lock returns None if no state was stored, otherwise
        # the state bytes and token.
        # KNOWN ISSUE: the final return statement after the locked block is unreachable.
        with self._lock:
            if self._state is None:
                return None
            return (self._state, self._token)

        return self._state

    def set_state(self, state, token):
        # [FUNCTION DOC] HELPER: Raises ValueError if the state is None. Under the lock, returns
        # None if the token is None while state exists or the token differs from the current
        # one; otherwise stores the state and moves the token to 0 or increments it. Returns
        # None in every case.
        if state is None:
            raise ValueError("State cannot be None.")
        with self._lock:
            if token is None and self._token is not None:
                return None
            if self._token != token:
                return None

            self._state = state
            self._token = self._token + 1 if self._token is not None else 0
