# mypy: allow-untyped-defs
# [MODULE DOC]
# Tests rpc_async and RRef to_here called from TorchScript under the faulty agent, to verify
# timeout handling and "RRef creation" errors. Defines scripted helpers that issue an async RPC
# to a scripted function with two args and two kwargs (returning the result or the future,
# with or without a timeout) and helpers that fetch an RRef value. Scripted functions are
# compiled at import time.


import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
    dist_init,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


@torch.jit.script
def two_args_two_kwargs(
    first_arg,
    second_arg,
    first_kwarg=torch.tensor([3, 3]),
    second_kwarg=torch.tensor([4, 4]),
):
    # [FUNCTION DOC] HELPER: Scripted RPC target returning the sum of its two positional and
    # two keyword tensors (kwargs default to 3s and 4s).
    return first_arg + second_arg + first_kwarg + second_kwarg


@torch.jit.script
def script_rpc_async_call(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    # [FUNCTION DOC] HELPER: Scripted async RPC of the two-args-two-kwargs function using the
    # default timeout; waits and returns the tensor (raises if the RPC times out).
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    ret = fut.wait()
    return ret


@torch.jit.script
def rpc_async_call_with_timeout(
    dst_worker_name: str,
    args: tuple[Tensor, Tensor],
    kwargs: dict[str, Tensor],
    timeout: float,
):
    # [FUNCTION DOC] HELPER: Like the default-timeout call but passes an explicit timeout in
    # seconds (zero means no timeout); waits inside script and returns the tensor.
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)
    ret = fut.wait()
    return ret


@torch.jit.script
def rpc_async_call_with_timeout_future_ret(
    dst_worker_name: str,
    args: tuple[Tensor, Tensor],
    kwargs: dict[str, Tensor],
    timeout: float,
):
    # [FUNCTION DOC] HELPER: Scripted async RPC with an explicit timeout that returns the
    # future without waiting, so the caller waits from Python.
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)
    return fut


@torch.jit.script
def rpc_async_call_future_ret(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    # [FUNCTION DOC] HELPER: Scripted async RPC with the default timeout that returns the
    # future without waiting.
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    return fut


@torch.jit.script
def rref_to_here(rref_var: RRef[Tensor]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function fetching the RRef value to the caller; raises
    # if the RRef creation failed.
    return rref_var.to_here()


@torch.jit.script
def rref_to_here_with_timeout(rref_var: RRef[Tensor], timeout: float) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function fetching the RRef value with a timeout in seconds.
    return rref_var.to_here(timeout)


@torch.jit.script
def rpc_async_with_rref_arg(dst_worker_name: str, args: tuple[RRef[Tensor]]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted async RPC that sends an RRef as an argument to the RRef
    # fetch function and returns the fetched tensor; the RRef is pickled in JIT on send.
    fut = rpc.rpc_async(dst_worker_name, rref_to_here, args)
    ret = fut.wait()
    return ret


class JitFaultyAgentRpcTest(RpcAgentTestFixture):
    # [CLASS DOC]
    # Test case for JIT rpc_async and RRef to_here timeouts under the faulty agent fixture.
    # Stateless; tests mostly run only on rank 0 and talk to the next worker.
    """
    Run tests for rpc_async in JIT under the faulty agent test fixture to test
    arbitrary timeouts.
    """

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_timeout_in_torchscript_function(self):
        # [FUNCTION DOC]
        # Rank 0 only. With script calls delayed 1.5s, checks that the scripted RPC raises a
        # timeout error for an explicit short timeout and for a tiny default timeout, and
        # completes with the expected sum when the timeout is zero. Restores the default RPC
        # timeout at the end (side effect on global RPC state).
        # Call rpc_async + fut.wait() in torchscript function and ensure that
        # timeout is raised.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()
        # Ensure that we get a timeout if we override the default timeout and
        # the RPC takes longer to execute.
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0.5)

        # Ensure that we timeout if we don't specify a timeout but the default
        # is less than the RPC takes to execute.
        rpc._set_rpc_timeout(0.001)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            script_rpc_async_call(dst_worker_name, args, kwargs)

        # Ensure that we run to completion if zero timeout is specified.
        ret = rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0)
        self.assertEqual(ret, torch.tensor([8, 8]))
        # reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_timeout_in_python(self):
        # [FUNCTION DOC]
        # Rank 0 only. Same scenarios as the in-script test, but the future is returned from
        # script and waited on in Python: explicit and default short timeouts raise, a zero
        # timeout completes. Resets the default RPC timeout at the end.
        # Ensures timeouts are raised if we call rpc_async from within a
        # torchscript function, but wait on the future in python.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()

        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0.5)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure timeout if we don't specify but the default is less than the
        # RPC takes to execute.
        rpc._set_rpc_timeout(0.001)
        fut = rpc_async_call_future_ret(dst_worker_name, args, kwargs)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if zero timeout is specified
        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0)
        result = fut.wait()
        self.assertEqual(result, torch.tensor([8, 8]))
        # reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_remote_timeout_to_here_in_jit(self):
        # [FUNCTION DOC]
        # Rank 0 only. Drops the remote creation message, flushes pending futures, then checks
        # that the scripted to_here raises an error mentioning RRef creation.
        # Test that calling to_here() in JIT will raise timeout error if
        # rpc.remote failed.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call to_here() within a ScriptFunction and ensure it raises
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref_to_here(rref)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout_in_jit(self):
        # [FUNCTION DOC]
        # Rank 0 only. With RRef fetch calls delayed 1s, a scripted to_here with a 0.01s timeout
        # must raise a timeout error, while a later call with a large timeout succeeds.
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref_to_here_with_timeout(rref, 0.01)

        rref_to_here_with_timeout(rref, 100)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_in_jit(self):
        # [FUNCTION DOC]
        # Rank 0 only. With the remote creation dropped, a scripted RPC that passes the failed
        # RRef as an argument must raise an error mentioning RRef creation during pickling.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call RPC with RRef arg in JIT, which will go through JIT pickling and
        # ensure error is raised.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc_async_with_rref_arg(dst_worker, (rref,))

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_script_func(self):
        # [FUNCTION DOC]
        # Rank 0 only. Like the JIT pickle test, but the RRef is sent via a Python rpc_sync to
        # a script function; expects an error mentioning RRef creation.
        # Similar to above test, but calls python rpc with script function.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call RPC with script function that takes RRef, ensure timeout during pickling
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_sync(dst_worker, rref_to_here, args=(rref,))
