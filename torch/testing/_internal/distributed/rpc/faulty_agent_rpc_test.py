# mypy: allow-untyped-defs

# [MODULE DOC]
# Tests RPC and RRef behavior under the faulty TensorPipe agent: which messages are failed or
# delayed, RRef creation errors when remote calls are dropped or time out, and RPC timeout
# semantics (per-call, default and zero timeouts). Defines small module-level helpers (a sleeping
# function, a scripted add, an RRef adder) used as RPC targets. The scripted helper is
# compiled at import time.

import time

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
    dist_init,
    wait_until_owners_and_forks_on_rank,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def my_sleep_func(seconds=1):
    # [FUNCTION DOC] HELPER: RPC target that sleeps for the given seconds, then returns a
    # scalar tensor equal to one.
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))


@torch.jit.script
def my_script_func(tensor):
    # [FUNCTION DOC] HELPER: Scripted RPC target returning the tensor added to itself.
    return torch.add(tensor, tensor)


def add_rref_to_value(rref, value):
    # [FUNCTION DOC] HELPER: RPC target that fetches the RRef value to the local worker and
    # returns it plus the given value.
    return rref.to_here() + value


class FaultyAgentRpcTest(RpcAgentTestFixture):
    # [CLASS DOC]
    # Test case run with the faulty agent fixture. The dist_init arguments faulty_messages and
    # messages_to_delay choose which message types are dropped or delayed for each test.
    # Stateless; most tests run only on rank 0 and target the next worker or itself.
    # no faulty_messages defined so this fails all retryable messages - see
    # faulty_rpc_agent_test_fixture.py for the list of retryable messages.
    @dist_init(messages_to_delay={})
    def test_check_failed_messages(self):
        # [FUNCTION DOC]
        # With default failing messages and no delays, rank 0 creates a remote tensor add on one
        # worker and passes the RRef to a remote call on another, then checks the fetched value
        # equals the sum. Finally deletes all user RRefs explicitly to ensure clean shutdown.
        if self.rank == 0:
            dst_worker_b = worker_name((self.rank + 1) % self.world_size)
            dst_worker_c = worker_name((self.rank + 2) % self.world_size)

            # Worker0 sends RPC to Worker1 and creates an RRef there
            rref = rpc.remote(
                dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2))
            )
            # Worker0 sends an RPC to Worker2 with the RRef as an arg
            rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
            # check if the output is as expected
            self.assertEqual(
                rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2))
            )
        # explicitly delete all User RRefs
        _delete_all_user_and_unforked_owner_rrefs()

    @dist_init
    def test_verify_backend_options(self):
        # [FUNCTION DOC]
        # Checks the fixture exposes the faulty backend with 8 threads, 3 failed sends, 4
        # failing message types, 2 delayed message types and the default RPC timeout.
        self.assertEqual(
            self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE
        )
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
        self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
        self.assertEqual(
            self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC
        )

    @dist_init(faulty_messages=["RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"])
    def test_custom_faulty_messages(self):
        # [FUNCTION DOC]
        # Checks that the failing messages passed to dist_init replace the defaults in the
        # backend options.
        self.assertEqual(
            {"RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"},
            set(self.rpc_backend_options.messages_to_fail),
        )

    @dist_init(faulty_messages=[])
    def test_no_faulty_messages(self):
        # [FUNCTION DOC]
        # Checks that an empty failing-message list yields no failing messages in the options.
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 0)

    @dist_init(messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_custom_messages_to_delay(self):
        # [FUNCTION DOC]
        # Checks that the delay mapping passed to dist_init replaces the default in the options.
        self.assertEqual(
            self.rpc_backend_options.messages_to_delay, {"SCRIPT_CALL": 1.5}
        )

    def _test_remote_message_dropped_pickle(self, dst=None):
        # [FUNCTION DOC] HELPER: Rank 0 only. Issues a remote call whose creation message is
        # dropped (destination defaults to the next rank), flushes pending futures, then checks
        # that serializing the RRef and sending it as an RPC argument both raise an error
        # mentioning RRef creation.
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Attempt to fork the RRef should raise an error indicating the rpc.remote timeout.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref._serialize()
        # Test that using RRef as arg over RPC (which forks) results in the same
        # error
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle(self):
        # [FUNCTION DOC] Runs the dropped-remote-call pickle check against another worker.
        self._test_remote_message_dropped_pickle()

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle_to_self(self):
        # [FUNCTION DOC] Runs the dropped-remote-call pickle check with the destination being
        # the calling worker itself.
        self._test_remote_message_dropped_pickle(self.rank)

    def _test_remote_message_dropped_timeout(self, func, args, dst=None):
        # [FUNCTION DOC] HELPER: Rank 0 only. Creates a remote call of func with args whose
        # creation message is dropped, flushes pending futures, and checks that fetching the
        # value raises an error mentioning RRef creation. Destination defaults to the next rank.
        if self.rank != 0:
            return

        # test the case where rpc.remote() message creation is completely dropped.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, func, args=args)
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()
        # Note: during shutdown, logs will indicate "Could not find OwnerRRef..."
        # on the owning nodes, this is expected because the OwnerRRef was never
        # successfully created. Therefore, delAllUsers will work as expected.

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout(self):
        # [FUNCTION DOC] Dropped-creation check for a builtin op (tensor add) on another worker.
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout_to_self(self):
        # [FUNCTION DOC] Dropped-creation check for a builtin op with the local worker as target.
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout(self):
        # [FUNCTION DOC] Dropped-creation check for a Python function on another worker.
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout_to_self(self):
        # [FUNCTION DOC] Dropped-creation check for a Python function targeting the local
        # worker.
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    def _test_remote_message_delay_timeout(self, func, args, dst=None):
        # [FUNCTION DOC] HELPER: Rank 0 only. Issues a remote call with a tiny timeout while the
        # creation message is delayed; checks that the creation future raises a timeout error and
        # that fetching the value then reports an RRef creation error. When the target is another
        # worker, also checks that a fetch with a tiny timeout on a slow remote call raises a
        # timeout, then waits until the owner has confirmed pending owners and forks so shutdown
        # does not race with user deletes.
        # KNOWN ISSUE: the comment says 10 ms timeout but the value used is 0.001 seconds (1 ms).
        if self.rank != 0:
            return
        # Test the case where remote message is eventually processed on the owner,
        # but the future on the creator times out before the response comes back.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 10 ms timeout
        rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
        # Future corresponding to the remote creation should time out.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()

        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # to_here() should now pick up that rpc.remote() creation has failed.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        # Test the case where rpc.remote() times out, but to_here() has already
        # started blocking before.
        # NOTE: we only test this when not sending to self, as to_here() calls
        # calls localValue(), which does not send an RPC and thus does not have
        # a timeout. This can be supported by allowing future.wait() to
        # take in an optional timeout (https://github.com/pytorch/pytorch/issues/39280)
        if dst_rank != self.rank:
            slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)

            with self.assertRaisesRegex(RuntimeError, expected_error):
                # to_here() should raise timeout error, since it does not know about the
                # status of rpc.remote().
                slow_rref.to_here(0.001)
        # Note: If we proceed with shutdown, UserRRef will send out a RRefUserDelete
        # but this can be a noop since it may not exist on the owner yet. Later,
        # the owner can process the RRef creation and wait for the delete message,
        # thus leading to a timeout.
        # Therefore, we wait until we get notification that pending owners have
        # been confirmed before sending out RRefUserDeletes.
        if dst_rank != self.rank:
            wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout(self):
        # [FUNCTION DOC] Delay-timeout check for a Python function on another worker.
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout_to_self(self):
        # [FUNCTION DOC] Delay-timeout check for a Python function with the local worker as target.
        func = my_sleep_func
        args = (1,)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout(self):
        # [FUNCTION DOC] Delay-timeout check for a builtin op on another worker.
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout_to_self(self):
        # [FUNCTION DOC] Delay-timeout check for a builtin op with the local worker as target.
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout(self):
        # [FUNCTION DOC] Delay-timeout check for a scripted function on another worker.
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout_to_self(self):
        # [FUNCTION DOC] Delay-timeout check for a scripted function targeting the local
        # worker.
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout(self):
        # [FUNCTION DOC]
        # Rank 0 only. With RRef fetch calls delayed, fetching with a 0.01s timeout must raise a
        # timeout error; a following fetch without timeout must succeed.
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref.to_here(0.01)

        rref.to_here()

    @dist_init(faulty_messages=[])
    def test_rpc_builtin_timeout(self):
        # [FUNCTION DOC]
        # With the default delays, checks builtin RPC timeouts on the next worker: a 1s explicit
        # timeout raises for sync and async calls, the default timeout lets delayed calls
        # finish, a new tiny default raises, and a zero timeout runs to completion. Resets the
        # global default RPC timeout at the end.
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        # PYTHON_CALL message types which correspond to Python UDF over RPC
        # by default get a delay (see faulty_rpc_agent_test_fixture)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(
                dst_worker,
                torch.add,
                args=(torch.tensor(1), torch.tensor(1)),
                timeout=1,
            )

        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_rpc_script_timeout(self):
        # [FUNCTION DOC]
        # Same timeout scenarios as the builtin test, but calling a scripted function with
        # script calls delayed 1.5s. Resets the global default RPC timeout at the end.
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)

        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)
