# mypy: allow-untyped-defs

# [MODULE DOC]
# Shared test logic for distributed autograd over RPC. Provides module-level state used to
# synchronize workers (flags and context ids recording which peer finished an RPC, and the set of
# all context ids ever seen), helper functions executed on remote workers (tensor factories, add
# functions in Python and TorchScript, nested-call helpers, trainer routines), a custom autograd
# function that can simulate a backward error, an enum of execution modes, a common base test
# class with graph and backward verification helpers, and the concrete test class that exercises
# dist autograd contexts, graphs, backward, gradients, cleanup, and failure handling.
# Import-time side effects: creates global state, a tensor requiring grad, and compiles several
# TorchScript functions.

import random
import sys
import threading
import time
import unittest
from datetime import timedelta
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    IS_MACOS,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.dist_utils import (
    dist_init,
    initialize_pg,
    wait_until_node_failure,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


# Right now we test up to 3-layer nested rpc calls.
# rpc_done[1] and ctx_ids[1] represent rpc is done in prev rank, and context id
# sent from prev rank respectively.
# rpc_done[2] and ctx_ids[2] represents for prev of prev rank.
# rpc_done[3] and ctx_ids[3] represents for prev of prev of prev rank.
# rpc_done[0] and ctx_ids[0] represents for current rank, but mostly not used.
rpc_done = [False, False, False, False]
ctx_ids = [-1, -1, -1, -1]

known_context_ids = set()

requires_grad_tensor = torch.ones(3, 3, requires_grad=True)


# Send rpc done info and context_id to
# dst_rank = (self.rank + rank_distance) % self.world_size
# we don't need a lock here since the GIL is held while executing remote
# python UDFs, so access is serialized across several workers.
def _set_rpc_done(ctx_id, rank_distance):
    # [FUNCTION DOC] HELPER: Runs (via RPC) on the receiving worker. Marks the done flag at index
    # rank_distance as true, stores the given context id at that index, and adds it to the global
    # set of known context ids. Mutates module-level state without a lock (relies on the GIL).
    global rpc_done
    global ctx_ids
    global known_context_ids
    rpc_done[rank_distance] = True
    ctx_ids[rank_distance] = ctx_id
    known_context_ids.add(ctx_id)


def _check_rpc_done(rank_distance):
    # [FUNCTION DOC] HELPER: Blocks, polling every 0.1 seconds, until the done flag at index
    # rank_distance has been set by _set_rpc_done. Has no timeout.
    while not rpc_done[rank_distance]:
        time.sleep(0.1)


def _torch_ones(sizes, requires_grad=False):
    # [FUNCTION DOC] HELPER: Returns a tensor of ones with the given sizes and requires_grad flag.
    return torch.ones(sizes, requires_grad=requires_grad)


# This method must be called on the rref owner, and verifies that the grad of
# rref tensor equals to the given grad.
def _compare_owner_value(context_id, rref, grad):
    # [FUNCTION DOC] HELPER: Must run on the owner of the reference. Looks up the gradient that
    # the given dist autograd context accumulated for the referenced tensor and returns whether it
    # equals the given grad. Raises an assertion error if exactly one of the two is sparse; when
    # both are sparse they are densified before comparing.
    grads = dist_autograd.get_gradients(context_id)
    x = grads[rref.local_value()]
    if x.is_sparse:
        if not grad.is_sparse:
            raise AssertionError("Expected grad to be sparse")
        x = x.to_dense()
        grad = grad.to_dense()
    else:
        if grad.is_sparse:
            raise AssertionError("Expected grad to not be sparse")
    return torch.equal(x, grad)


def create_tensor():
    # [FUNCTION DOC] HELPER: Returns a 3x3 tensor of ones that requires grad.
    return torch.ones((3, 3), requires_grad=True)


def build_sparse_tensor(coalesce=False, requires_grad=True, dtype=torch.float32):
    # [FUNCTION DOC] HELPER: Builds a fixed 3x3 sparse COO tensor with three entries, using the
    # given dtype and requires_grad flag; coalesces it first when requested. Returns the tensor.
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3.2, 4.1, 5.3]
    tensor = torch.sparse_coo_tensor(
        i, v, (3, 3), requires_grad=requires_grad, dtype=dtype
    )
    if coalesce:
        tensor = tensor.coalesce()
    return tensor


@torch.jit.script
def create_torchscript_tensor() -> torch.Tensor:
    # [FUNCTION DOC] HELPER: TorchScript function returning a 3x3 tensor of ones that requires
    # grad.
    return torch.ones((3, 3)).requires_grad_()


def my_py_add(t1, t2):
    # [FUNCTION DOC] HELPER: Python function returning the elementwise sum of two tensors.
    return torch.add(t1, t2)


def my_scalar_add(a, b):
    # [FUNCTION DOC] HELPER: Returns the sum of the two arguments using plain addition.
    return a + b


def my_rref_add(rref_t1, t2):
    # [FUNCTION DOC] HELPER: Must run on the reference owner. Adds the locally held referenced
    # tensor and t2 and returns the sum.
    ret = torch.add(rref_t1.local_value(), t2)
    return ret


@torch.jit.script
def my_script_add(t1, t2):
    # [FUNCTION DOC] HELPER: TorchScript function returning the elementwise sum of two tensors.
    return torch.add(t1, t2)


@torch.jit.script
def my_script_ref_add(ref_t1: RRef[torch.Tensor], t2: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: TorchScript function that fetches the referenced tensor to the
    # current worker (a remote copy if needed) and returns its sum with t2.
    t1 = ref_t1.to_here()
    return torch.add(t1, t2)


def my_nested_rref_add(dst, rref_t1, t2):
    # [FUNCTION DOC] HELPER: Makes a synchronous RPC to worker dst that runs the reference add
    # helper on the reference and t2, and returns its result.
    return rpc.rpc_sync(dst, my_rref_add, args=(rref_t1, t2))


def ret_requires_grad():
    # [FUNCTION DOC] HELPER: Returns the module-level tensor that requires grad (same object).
    return requires_grad_tensor


def my_py_nested_call(t1, t2, dst, world_size, hops):
    # [FUNCTION DOC] HELPER: Forwards the call around the ring of workers. The next destination is
    # dst plus one modulo world_size. While hops is positive it makes a synchronous RPC to the next
    # worker running this same function with hops decreased by one; at zero hops it makes a
    # synchronous RPC to the next worker running the Python add helper on t1 and t2. Returns the
    # final sum.
    next_dst = (dst + 1) % world_size
    if hops > 0:
        return rpc.rpc_sync(
            worker_name(next_dst),
            my_py_nested_call,
            args=(t1, t2, next_dst, world_size, hops - 1),
        )
    else:
        return rpc.rpc_sync(worker_name(next_dst), my_py_add, args=(t1, t2))


# after dist autograd context is cleaned up, it should be cleaned up on other
# nodes. This helper allows timeout_seconds for those RPCs to be completed, and
# ensures that all the contexts have been cleaned up in that timeframe.any
def _all_contexts_cleaned_up(timeout_seconds=10):
    # [FUNCTION DOC] HELPER: Repeatedly tries to retrieve every known context id until each
    # retrieval raises a runtime error (meaning the context was cleaned up) or the timeout in
    # seconds elapses. Busy-loops without sleeping. Returns true if all known contexts are gone.
    global known_context_ids
    start = time.time()
    context_id_to_raised = set()
    while (
        time.time() - start < timeout_seconds
        and context_id_to_raised != known_context_ids
    ):
        for context_id in known_context_ids:
            try:
                dist_autograd._retrieve_context(context_id)
            except RuntimeError:
                context_id_to_raised.add(context_id)
    # all contexts have been cleaned up if trying to retrieve any context resulted in a RuntimeError.
    success = context_id_to_raised == known_context_ids
    return success


# This function creates a dist autograd context, run rpc_sync on the given ps,
# and then blocks until the ps has verified the grads are correctly accumulated.
def _run_trainer(rref_t1, t2, ps, rank_diff, sparse):
    # [FUNCTION DOC] HELPER: Trainer routine run on a worker. Within a new dist autograd context it
    # calls the parameter server worker ps to add the referenced tensor and t2, sums the result
    # (sparse sum when sparse is true), runs distributed backward, then tells ps that this trainer
    # (identified by rank_diff) is done together with the context id, and blocks until ps signals
    # that it has finished verification (done flag at index zero).
    with dist_autograd.context() as context_id:
        ret = rpc.rpc_sync(ps, my_rref_add, args=(rref_t1, t2))
        if sparse:
            loss = torch.sparse.sum(ret)
        else:
            loss = ret.sum()
        dist_autograd.backward(context_id, [loss])
        # prevent deleting dist autograd context
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))


# This function is the same as _run_trainer, except rpc calls torchscript
# function "my_script_ref_add" instead of python function "my_rref_add"
def _run_trainer_torchscript(rref_t1, t2, ps, rank_diff, sparse):
    # [FUNCTION DOC] HELPER: Same as the Python trainer routine but the remote call runs the
    # TorchScript reference-add function instead of the Python one.
    with dist_autograd.context() as context_id:
        ret = rpc.rpc_sync(ps, my_script_ref_add, args=(rref_t1, t2))
        if sparse:
            loss = torch.sparse.sum(ret)
        else:
            loss = ret.sum()
        dist_autograd.backward(context_id, [loss])
        # prevent deleting dist autograd context
        rpc.rpc_sync(ps, _set_rpc_done, args=(context_id, rank_diff))
        rpc.rpc_sync(ps, _check_rpc_done, args=(0,))


class SimulateBackwardError(Function):
    # [CLASS DOC] HELPER: Identity autograd function whose backward raises a generic exception
    # while the class-level flag _simulate_error is true (the default); when the flag is false
    # backward passes the gradient through unchanged. Used to test error propagation in
    # distributed backward.
    _simulate_error = True

    @staticmethod
    def forward(ctx, input):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return input

    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        # [FUNCTION DOC] HELPER: Raises an exception with message "Simulate error on backward pass"
        # when the class flag is true; otherwise returns the incoming gradient unchanged.
        if SimulateBackwardError._simulate_error:
            raise Exception("Simulate error on backward pass")  # noqa: TRY002
        else:
            return input


class ExecMode(Enum):
    # [CLASS DOC] HELPER: Enumeration of how an operation is executed in tests: locally, by
    # synchronous RPC, by remote call (with fetch of the result), or by asynchronous RPC.
    LOCAL = 1  # Run the operation locally.
    RPC_SYNC = 2  # Run the operation using rpc_sync
    REMOTE = 3  # Run the operation using remote.
    RPC_ASYNC = 4  # Run the operation using rpc_async


# Common utils for both CPU and CUDA test suites
class CommonDistAutogradTest(RpcAgentTestFixture):
    # [CLASS DOC] Base test class shared by the CPU and CUDA suites. Holds no state of its own
    # except a lazily created destination rank cursor (dst_rank) used to rotate through peers.
    # Provides helpers to run a function under a chosen ExecMode, to compute the next peer rank,
    # to verify dist autograd gradients against local ones, and the generic test bodies for graph
    # structure, nested calls, argument handling, context cleanup, and backward scenarios.
    def _exec_func_with_dst(self, dst, exec_mode, method, *args):
        # [FUNCTION DOC] HELPER: Runs method on the given args according to exec_mode. Local mode
        # calls it directly (a single list argument is unpacked into positional args). Sync RPC,
        # remote (result fetched) and async RPC (waited on) run it on worker dst and return the
        # result. Raises a value error for an unknown mode.
        if ExecMode.LOCAL == exec_mode:
            if len(args) == 1 and isinstance(args[0], list):
                return method(*args[0])
            return method(*args)
        elif ExecMode.RPC_SYNC == exec_mode:
            return rpc.rpc_sync(worker_name(dst), method, args=(args))
        elif ExecMode.REMOTE == exec_mode:
            return rpc.remote(worker_name(dst), method, args=(args)).to_here()
        elif ExecMode.RPC_ASYNC == exec_mode:
            fut = rpc.rpc_async(worker_name(dst), method, args=(args))
            return fut.wait()
        else:
            raise ValueError(f"Unrecognized ExecMode {exec_mode}")

    def _exec_func(self, exec_mode, method, *args):
        # [FUNCTION DOC] HELPER: Like the destination-explicit variant but picks the destination
        # as the next peer rank from the rotating cursor.
        return self._exec_func_with_dst(self._next_rank(), exec_mode, method, *args)

    def _next_rank(self):
        # [FUNCTION DOC] HELPER: Returns the next destination rank and stores it in the dst_rank
        # attribute. First call yields rank plus one modulo world size; later calls advance by one
        # modulo world size, skipping this worker's own rank.
        if hasattr(self, "dst_rank"):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                return self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank

    def _check_rpc_done(self, rank_distance):
        # [FUNCTION DOC] HELPER: Method wrapper (overridable) around the module-level blocking
        # wait for the done flag at the given index.
        _check_rpc_done(rank_distance)

    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        # [FUNCTION DOC] HELPER: In local mode runs ordinary backward on the given tensors and
        # returns the gradient of each arg (to be used as the reference). In other modes runs
        # distributed backward and verifies against local_grads, returning nothing.
        if exec_mode == ExecMode.LOCAL:
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]
        else:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)

    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        # [FUNCTION DOC] HELPER: Runs distributed backward in the context, fetches the context's
        # gradient mapping, and asserts that each arg with a non-missing local gradient appears
        # with an equal gradient, each arg with a missing local gradient is absent, and the mapping
        # has no extra entries.
        dist_autograd.backward(context_id, tensors)

        # Verify grads were accumulated appropriately.
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)

        self.assertEqual(ngrads, len(grads))

    def _test_graph(self, fn, exec_mode, sparse):
        # [FUNCTION DOC] HELPER: Initializes the process group, then in a dist autograd context
        # sends two tensors (dense or sparse) to the next rank via sync RPC or remote call running
        # fn, and notifies the peer. Verifies that the context holds exactly one send and one recv
        # function with the expected graph, waits for the previous rank, verifies the graph of that
        # rank's context, and after the context exits asserts it is cleaned up and no current
        # context exists. Raises a value error for modes other than sync RPC or remote.
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor()
                t2 = build_sparse_tensor()
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), fn, args=(t1, t2)).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))

            # Verify graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                next(iter(recv_functions.values())),
                t1,
                t2,
                ret,
            )

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # Verify graph for previous context id.
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

        # autograd context should be cleaned up by now.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # No autograd context available.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    # 3-layer nested calls
    def _test_graph_for_py_nested_call(self, exec_mode, sparse):
        # [FUNCTION DOC] HELPER: Sends tensors through a nested call chain of one extra hop and
        # uses barriers and done notifications to ensure all RPCs finished. Then verifies four
        # graphs on this rank: the context for its own first call, the contexts of the previous
        # ranks for the first and second nested calls, and the context of the rank at distance
        # three where the add executed.
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(t1, t2, dst_rank, self.world_size, 1),
                )
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(t1, t2, dst_rank, self.world_size, 1),
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            # Barrier to ensure all RPCs are done.
            dist.barrier()

            for rd in [1, 2, 3]:
                rpc.rpc_sync(
                    worker_name((self.rank + rd) % self.world_size),
                    _set_rpc_done,
                    args=(context_id, rd),
                )

            # Barrier to ensure all set_rpc_done have completed.
            dist.barrier()

            # For self.rank, it has 4 graphs to verify
            # One is for current context id when this rank send first rpc call.
            # Second one is for prev context id when this rank make 1st nested
            # call.
            # Third one is for prev prev context id when this rank make
            # 2nd nested call.
            # Last one is for prev prev prev context id when this rank
            # execute the torch.add() operator.

            # Verify first graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                next(iter(recv_functions.values())),
                t1,
                t2,
                ret,
            )

            # Verify second graph for 1st nested call.
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)

            # Verify third graph for 2nd nested call.
            ctx = dist_autograd._retrieve_context(ctx_ids[2])
            self._verify_graph_for_nested_rpc_call(ctx)

            # verify last graph for rpc call execution.
            ctx = dist_autograd._retrieve_context(ctx_ids[3])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    # Rank0->Rank1->Rank0
    def _test_graph_for_py_nested_call_itself(self, exec_mode, sparse):
        # [FUNCTION DOC] HELPER: Nested call that comes back to the originating rank (zero extra
        # hops, so the add executes on the originator). Verifies the current context has two send
        # and two recv functions with the expected graphs, then waits for the previous rank and
        # verifies its context for the nested call.
        dst_rank = (self.rank + 1) % self.world_size

        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(
                        t1,
                        t2,
                        (self.rank - 1 + self.world_size) % self.world_size,
                        self.world_size,
                        0,
                    ),
                )
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank),
                    my_py_nested_call,
                    args=(
                        t1,
                        t2,
                        (self.rank - 1 + self.world_size) % self.world_size,
                        self.world_size,
                        0,
                    ),
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(
                worker_name((self.rank + 1) % self.world_size),
                _set_rpc_done,
                args=(context_id, 1),
            )

            # For self.rank, it has 2 graphs to verify.
            # One is for current context id when this rank send first rpc
            # call and execute the torch.add() operator.
            # Another one is for prev context id when this rank make
            # nested call.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(2, len(recv_functions))
            self._verify_graph_for_first_rpc_call(
                next(iter(send_functions.values())),
                list(recv_functions.values())[1],
                t1,
                t2,
                ret,
            )
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[1])

            # Verify two pairs of send and recv functions for nested
            # call
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    def _test_no_graph_with_tensors_not_require_grad(self, exec_mode, sparse):
        # [FUNCTION DOC] HELPER: Sends tensors that do not require grad over RPC and asserts the
        # local context records no send or recv functions, while the peer's context for the
        # message still exists (the context id is always propagated).
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=False)
                t2 = build_sparse_tensor(requires_grad=False)
            else:
                t1 = torch.ones(3, 3, requires_grad=False)
                t2 = torch.zeros(3, 3, requires_grad=False)
            if ExecMode.RPC_SYNC == exec_mode:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                rpc.remote(worker_name(dst_rank), torch.add, args=(t1, t2)).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))

            ctx = dist_autograd._current_context()
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # NB: RRef.to_here() always passes the autograd context to
            # the callee, as the caller does not know whether the return
            # value would contain a requires_grad tensor or not.
            #
            # rpc/remote with udf (_set_rpc_done here) also always passes the
            # autograd context to the callee due to the same reason.
            self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
            dist.barrier()

    def _test_rpc_complex_args(self, exec_mode, sparse):
        # [FUNCTION DOC] HELPER: Sends a list of ten tensors (every other one requiring grad) to
        # the next peer to be stacked, asserts the result equals a local stack, that the send
        # function links to an accumulate-grad node for each tensor, and that the context records
        # exactly the destination as a known worker id.
        with dist_autograd.context():
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                if sparse:
                    tensor = build_sparse_tensor(requires_grad=(i % 2 == 0))
                else:
                    tensor = torch.ones(3, 3, requires_grad=(i % 2 == 0))
                tensors.append(tensor)
            dst_rank = self._next_rank()
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.stack, args=(tensors,))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(
                    worker_name(dst_rank), torch.stack, args=(tensors,)
                ).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            self.assertEqual(torch.stack(tensors), ret)

            # Verify appropriate tensors have been attached the autograd graph.
            next_funcs = next(
                iter(dist_autograd._current_context()._send_functions().values())
            ).next_functions
            for i in range(len(next_funcs)):
                self.assertEqual(
                    "torch::autograd::AccumulateGrad", next_funcs[i][0].name()
                )
                self.assertEqual(tensors[i], next_funcs[i][0].variable)

            # Verify that the worker id has been recorded in the context
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            self.assertEqual(worker_ids, {dst_rank})

    def context_cleanup_test_helper(self, rpc_args, func, nested=False):
        # [FUNCTION DOC] HELPER: Initializes the process group and, within a context, calls func
        # with rpc_args on every other worker (or only the next one, plus a nested done
        # notification for the one after, when nested is true). After the context exits asserts it
        # is gone locally, waits at a barrier, and asserts all contexts known to this worker are
        # eventually cleaned up.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # test that in dist autograd, in the case that tensors communicated over RPC do
        # NOT require grad, we still cleanup the dist autograd contexts created
        # on other nodes. This is because the autograd context is still
        # communicated over RPC even if tensor arguments do not require grad, as
        #  it is possible that the response could.
        if nested:
            dst_rank = (self.rank + 1) % self.world_size
            nested_dst_rank = (dst_rank + 1) % self.world_size
            dst_ranks = {dst_rank}
        else:
            dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}

        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
                if nested:
                    rpc.rpc_sync(
                        worker_name(nested_dst_rank),
                        _set_rpc_done,
                        args=(context_id, 2),
                    )
        # the thread's context id should be cleaned up
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        # Ensure all peers have finished mutating the
        # `known_context_ids` set.
        dist.barrier()
        # check that all contexts have been cleaned up.
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    def _backward_no_grad_on_tensor(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Adds t1 and t2 on the next peer, runs distributed backward and
        # asserts the tensors' own grad fields stay empty. Then populates them with a local
        # backward and asserts a further distributed backward does not modify them.
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            self.assertIsNone(t1.grad)
            self.assertIsNone(t2.grad)

            # Now populate .grad with local autograd engine and
            # verify dist autograd doesn't mess with it.
            loss_local = torch.add(t1, t2)
            if sparse:
                loss_local = torch.sparse.sum(loss_local)
            else:
                loss_local = loss_local.sum()
            loss_local.backward()
            self.assertIsNotNone(t1.grad)
            self.assertIsNotNone(t2.grad)

            t1_grad_before = t1.grad
            t2_grad_before = t2.grad
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(t1_grad_before, t1.grad)
            self.assertEqual(t2_grad_before, t2.grad)

    # The current rank first creates a tensor on the rref_owner, and then passes
    # the rref with another tensor to the callee to run either my_rref_add or
    # my_nested_rref_add, depending on whether the callee is the rref owner.
    # The grad of tensor lives on the current rank, and the grad of the rref
    # tensor lives on the rref owner.
    def _backward_rref(self, callee, rref_owner, t1, t2, local_grads, sparse):
        # [FUNCTION DOC] HELPER: Creates a tensor on rref_owner, passes its reference with t2 to
        # callee to add them (directly, or through a nested call when callee is not the owner),
        # runs distributed backward on the fetched result, and asserts the gradient of t2 is
        # recorded on the caller and the gradient of the referenced tensor on the owner equals
        # t1.grad. Note: local_grads is an unused argument; t1.grad is expected to have been
        # populated by the local backward run at the start.
        local_ret = torch.add(t1, t2)
        if sparse:
            local_ret = torch.sparse.sum(local_ret)
        else:
            local_ret = local_ret.sum()
        local_ret.backward()
        with dist_autograd.context() as context_id:
            if sparse:
                rref_t1 = rpc.remote(
                    rref_owner,
                    build_sparse_tensor,
                    args=(
                        False,
                        True,
                    ),
                )
            else:
                rref_t1 = rpc.remote(
                    rref_owner,
                    _torch_ones,
                    args=((3, 3),),
                    kwargs={"requires_grad": True},
                )
            if callee == rref_owner:
                rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
            else:
                rref = rpc.remote(
                    callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2)
                )
            ret = rref.to_here()
            if sparse:
                ret = torch.sparse.sum(ret)
            else:
                ret = ret.sum()
            dist_autograd.backward(context_id, [ret])

            # verify grads on caller
            grads = dist_autograd.get_gradients(context_id)
            self.assertIn(t2, grads)
            self.assertEqual(grads[t2], t2.grad)

            # verify grads on rref owner
            self.assertTrue(
                rpc.rpc_sync(
                    rref_owner,
                    _compare_owner_value,
                    args=(context_id, rref_t1, t1.grad),
                )
            )

    # In this test, every rank will serve as a parameter server (ps) and a
    # driver, and then kicks off trainers on the other three ranks. So, we have:
    # ps = rank0 with trainers = rank1/2/3
    # ps = rank2 with trainers = rank2/3/0
    # ps = rank3 with trainers = rank3/0/1
    # ps = rank4 with trainers = rank0/1/2
    #
    # These four test ps-trainer groups run on completely separate autograd
    # graphs, but they share the same set of underlying RpcAgents.
    def _test_trainer_ps(self, create_ref_fn, trainer_fn, sparse):
        # [FUNCTION DOC] HELPER: Acts as parameter server. Computes reference gradients locally,
        # creates a reference on itself using create_ref_fn, and starts trainer_fn asynchronously
        # on the three workers at rank distances one to three. Waits for each trainer's done
        # signal, checks the gradient stored in each trainer's context for the referenced tensor
        # equals the local gradient, then releases the trainers and waits for all futures.
        # KNOWN ISSUE: the comment above this method lists ps/trainer groupings that do not match
        # the code (trainers are always the three following ranks) and assumes four ranks.
        if sparse:
            t1 = build_sparse_tensor(requires_grad=True)
            t2 = build_sparse_tensor(requires_grad=True)
        else:
            t1 = torch.ones((3, 3), requires_grad=True)
            t2 = torch.zeros((3, 3), requires_grad=True)

        local_ret = torch.add(t1, t2)
        if sparse:
            torch.sparse.sum(local_ret).backward()
        else:
            local_ret.sum().backward()

        # create rref on self
        rref_t1 = rpc.remote(worker_name(self.rank), create_ref_fn, args=())

        # kick off forward and backward pass on three other workers (trainers)
        rank_diffs = [1, 2, 3]
        futures = [
            rpc.rpc_async(
                worker_name((self.rank + rank_diff) % self.world_size),
                trainer_fn,
                args=(rref_t1, t2, worker_name(self.rank), rank_diff, sparse),
            )
            for rank_diff in rank_diffs
        ]

        # check if the trainers have done with their backward pass
        for rank_diff in rank_diffs:
            self._check_rpc_done(rank_diff)

        # trainers are done and holding the context for verification
        for rank_diff in rank_diffs:
            # make sure grads are accumulated for the same tensors and values
            # are all correct
            ctx_id = ctx_ids[rank_diff]
            grads = dist_autograd.get_gradients(ctx_id)
            local_t1 = rref_t1.to_here()
            self.assertIn(local_t1, grads)
            self.assertEqual(grads[local_t1], t1.grad)

        # unblock trainers
        _set_rpc_done(None, 0)

        # wait until all trainers are done
        torch.futures.wait_all(futures)

    def _backward_multiple_round_trips(self, t1, t2, t3, t4, t5, local_grads, sparse):
        # [FUNCTION DOC] HELPER: For local, sync RPC and remote modes builds a computation with
        # several operations each executed on a rotating peer (add, multiply, stack, then batched
        # or elementwise products), reduces it to a loss and verifies gradients for the five
        # inputs against the local run, carrying the local gradients over to later modes.
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                # Multiple RPCs between different nodes.
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                if sparse:
                    val = self._exec_func(exec_mode, torch.mul, s1, s2)
                    val = self._exec_func(exec_mode, torch.mul, val, val)
                    loss = torch.sparse.sum(val)
                else:
                    val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                    val = self._exec_func(exec_mode, torch.matmul, val, val)
                    loss = val.sum()

                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5
                )
                local_grads = ret if ret else local_grads

    def _backward_different_dtypes(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Adds t1 and t2 (which may differ in dtype) locally and remotely,
        # reduces to a loss and checks the distributed gradients match the local ones.
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                loss = self._exec_func(exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(loss)
                else:
                    loss = loss.sum()
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple_python_udf(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Runs the Python add function locally and by remote call, reduces
        # to a loss and verifies the distributed gradients equal the local ones.
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, my_py_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple_script_call(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Runs the TorchScript add function in local, sync RPC, async RPC
        # and remote modes, reduces to a loss and verifies gradients against the local run.
        local_grads = None
        for exec_mode in [
            ExecMode.LOCAL,
            ExecMode.RPC_SYNC,
            ExecMode.RPC_ASYNC,
            ExecMode.REMOTE,
        ]:
            with dist_autograd.context() as context_id:
                forward_ret = self._exec_func(exec_mode, my_script_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(forward_ret)
                else:
                    loss = forward_ret.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    def _nested_backward_accumulate_grads(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Calls the nested accumulate-grads test function on a peer, reduces
        # the result to a loss and runs distributed backward twice (the first retaining the graph)
        # to exercise gradient accumulation across a nested call.
        with dist_autograd.context() as context_id:
            ret = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._test_nested_backward_accumulate_grads,
                args=(t1, t2, self._next_rank()),
            )
            if sparse:
                loss = torch.sparse.sum(ret)
            else:
                loss = ret.sum()
            # Run backward twice.
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            dist_autograd.backward(context_id, [loss])

    def _backwards_nested_python_udf(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Computes a composite loss locally and runs ordinary backward, then
        # computes the same loss through a nested Python function on a peer, runs distributed
        # backward and asserts the context gradients of t1 and t2 equal the local grads.
        t3 = t1 * t2
        t4 = t1 + t2
        res = t3 + t4
        loss = t1 * t2 * t3 * t4 * res
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        torch.autograd.backward([loss])

        # Now run distributed autograd.
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._nested_python_udf,
                args=(t1, t2, self._next_rank()),
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    def _mixed_requires_grad(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Runs the mixed requires-grad operation on a peer via sync RPC and
        # remote call with t1 requiring grad and t2 not. Asserts the product value, that only t1
        # receives a gradient in the context (equal to t2), and that requires_grad flags are
        # unchanged.
        for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(
                    exec_mode, DistAutogradTest._mixed_requires_grad_operaton, t1, t2
                )
                self.assertEqual(t1 * t2, ret)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                dist_autograd.backward(context_id, [loss])
                self.assertTrue(t1.requires_grad)
                self.assertFalse(t2.requires_grad)
                grads = dist_autograd.get_gradients(context_id)
                self.assertIn(t1, grads)
                self.assertNotIn(t2, grads)
                self.assertEqual(t2, grads[t1])

    def _multiple_backward(self, t1, t2, sparse):
        # [FUNCTION DOC] HELPER: Adds t1 and t2 on a peer and runs distributed backward 1000 times
        # on the same retained graph; passes if nothing raises.
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            )
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            # Run backward in a loop multiple times.
            for _ in range(1000):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    # For current context, this rank sends t1 and t2 tensors to dst_rank,
    # then get t3 = torch.add(t1, t2) result tensor.
    # For the current context in this rank, it expects graph like this:
    #  send function:
    #              rpcSendBackward
    #                  /          \
    #  t1.AccumulateGrad         t2.AccumulateGrad
    #
    #  recv function:
    #
    #            |
    #          t3.rpcRecvBackward
    #
    def _verify_graph_for_first_rpc_call(
        self, send_function, recv_function, t1, t2, ret
    ):
        # [FUNCTION DOC] HELPER: Asserts the send function has exactly two next functions, each an
        # accumulate-grad node for t1 and t2 respectively (output index zero), and that the
        # result's grad function is the given recv function.
        # Retrieve the next functions in the graph.
        next_funcs = send_function.next_functions
        self.assertEqual(2, len(next_funcs))

        # We should now hit t1 and t2 in the autograd graph.
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])

        # Test recv functions.
        self.assertEqual(ret.grad_fn, recv_function)

    # Run the same code locally and with dist autograd and verify gradients
    # are same.
    def _backward_simple(self, dst, t1, t2, local_grads, sparse):
        # [FUNCTION DOC] HELPER: Adds t1 and t2 on worker rank dst in local, sync RPC and remote
        # modes, reduces to a loss, and verifies distributed gradients against the local ones,
        # carrying local gradients forward from the local run.
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func_with_dst(dst, exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, executes torch.add(t1, t2) and sends
    # result tensor t3 back.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions:
    #       rpcSendBackward
    #           |
    #          t3.AddBackward0
    #          /             \
    # t1.recvRpcBackward    t2.recvRpcBackward
    def _verify_graph_for_rpc_call_exec(self, send_function):
        # [FUNCTION DOC] HELPER: Asserts the send function leads to a single add-backward node
        # whose two inputs are the same receive-backward node.
        # Verify next function is AddBackward0
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual("AddBackward0", add_backward_fn.name())

        # Verify the next two functions are the same recv backward function.
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, forwards t1 and t2 tensors using
    # nested rpc call to next dst. In return route, receive result tensor t3
    # from next dst and forwarding t3 back to previous calls.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions for receiving and forwarding t1 and t2:
    #       rpcSendBackward
    #          /          \
    # t1.recvRpcBackward    t2.recvRpcBackward
    #  send and recv functions for receiving and forwarding t3:
    #       rpcSendBackward
    #             |
    #           t3.recvRpcBackward
    def _verify_graph_for_nested_rpc_call(self, ctx):
        # [FUNCTION DOC] HELPER: Asserts the context has two send functions: the first has two
        # inputs that are the same receive-backward node (forwarded tensors), and the second has
        # one input that is a receive-backward node (the returned result).
        send_functions = ctx._send_functions()
        self.assertEqual(2, len(send_functions))

        # For send function when making nested rpc call,
        # next functions of the send function are two recv functions
        # for received two tensors from previous call
        next_funcs = next(iter(send_functions.values())).next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

        # For send function when returning response to previous call
        # next function of the send function is the recv function
        # for received tensor result returned from nested call
        next_funcs = list(send_functions.values())[1].next_functions
        self.assertEqual(1, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )


class TensorPipeAgentDistAutogradTest(CommonDistAutogradTest):
    # [CLASS DOC] Dist autograd tests using sparse tensors; sparse gradients are supported only by
    # the TensorPipe agent. Each test delegates to a shared helper with the sparse flag set.
    # Sparse tests only work with TensorPipeAgent.
    @dist_init
    def test_graph_for_builtin_call_sparse(self):
        # [FUNCTION DOC] Graph verification for a built-in add via sync RPC with sparse tensors.
        self._test_graph(torch.add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_python_call_sparse(self):
        # [FUNCTION DOC] Graph verification for a Python function via sync RPC, sparse tensors.
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_builtin_remote_call_sparse(self):
        # [FUNCTION DOC] Graph verification for a built-in add via remote call, sparse tensors.
        self._test_graph(torch.add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_python_remote_call_sparse(self):
        # [FUNCTION DOC] Graph verification for a Python function via remote call, sparse tensors.
        self._test_graph(my_py_add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_sparse(self):
        # [FUNCTION DOC] Nested-call graph verification via sync RPC with sparse tensors.
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_sparse(self):
        # [FUNCTION DOC] Nested-call graph verification via remote call with sparse tensors.
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_itself_sparse(self):
        # [FUNCTION DOC] Self-returning nested-call graph check via sync RPC, sparse tensors.
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_itself_sparse(self):
        # [FUNCTION DOC] Self-returning nested-call graph check via remote call, sparse tensors.
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_sparse(self):
        # [FUNCTION DOC] No-graph check for sparse tensors without grad, via sync RPC.
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote_sparse(self):
        # [FUNCTION DOC] No-graph check for sparse tensors without grad, via remote call.
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, True)

    @dist_init
    def test_rpc_complex_args_sparse(self):
        # [FUNCTION DOC] Complex-argument (list of tensors) check via sync RPC, sparse tensors.
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_remote_complex_args_sparse(self):
        # [FUNCTION DOC] Complex-argument (list of tensors) check via remote call, sparse tensors.
        self._test_rpc_complex_args(ExecMode.REMOTE, True)

    @dist_init
    def test_context_cleanup_tensor_with_grad_sparse(self):
        # [FUNCTION DOC] Context cleanup check when sparse tensors requiring grad are sent.
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_context_cleanup_tensor_no_grad_sparse(self):
        # [FUNCTION DOC] Context cleanup check when a sparse tensor not requiring grad is sent.
        t1 = build_sparse_tensor(requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    def test_context_cleanup_nested_rpc_sparse(self):
        # [FUNCTION DOC] Context cleanup check for a nested call chain with sparse tensors.
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(
            rpc_args=args, func=my_py_nested_call, nested=True
        )

    @dist_init
    def test_backward_no_grad_on_tensor_sparse(self):
        # [FUNCTION DOC] Checks distributed backward leaves tensor grad fields untouched, sparse.
        self._backward_no_grad_on_tensor(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backward_simple_sparse(self):
        # [FUNCTION DOC] Simple add backward against the next peer, with sparse tensors.
        self._backward_simple(
            self._next_rank(),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_simple_self_sparse(self):
        # [FUNCTION DOC] Simple add backward where the destination is this rank, sparse tensors.
        self._backward_simple(
            self.rank,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_rref_multi_sparse(self):
        # [FUNCTION DOC] All ranks except rank 0 use worker0 as both callee and reference owner
        # for the reference backward check with sparse tensors; rank 0 does nothing.
        if self.rank > 0:
            callee = "worker0"
            rref_owner = callee
            self._backward_rref(
                callee,
                rref_owner,
                build_sparse_tensor(requires_grad=True),
                build_sparse_tensor(requires_grad=True),
                None,
                True,
            )

    @dist_init
    def test_backward_rref_sparse(self):
        # [FUNCTION DOC] Reference backward check where callee and owner are the next peer.
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_rref_nested_sparse(self):
        # [FUNCTION DOC] Reference backward check with callee at rank plus one and owner at rank
        # plus two (nested call), sparse tensors.
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(
            callee,
            rref_owner,
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_trainer_ps_sparse(self):
        # [FUNCTION DOC] Parameter-server/trainer check with sparse tensors and Python trainers.
        self._test_trainer_ps(build_sparse_tensor, _run_trainer, True)

    @dist_init
    def test_backward_multiple_round_trips_sparse(self):
        # [FUNCTION DOC] Multi-round-trip backward check with five sparse tensors, alternating
        # requires-grad flags.
        self._backward_multiple_round_trips(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            build_sparse_tensor(requires_grad=True),
            None,
            True,
        )

    @dist_init
    def test_backward_different_dtypes_sparse(self):
        # [FUNCTION DOC] Backward check with sparse tensors of 32-bit and 64-bit float dtypes.
        self._backward_different_dtypes(
            build_sparse_tensor(requires_grad=True, dtype=torch.float32),
            build_sparse_tensor(requires_grad=True, dtype=torch.float64),
            True,
        )

    @dist_init
    def test_backward_simple_python_udf_sparse(self):
        # [FUNCTION DOC] Backward check through a Python function with sparse tensors.
        self._backward_simple_python_udf(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backward_simple_script_call_sparse(self):
        # [FUNCTION DOC] Backward check through a TorchScript function with sparse tensors.
        self._backward_simple_script_call(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_nested_backward_accumulate_grads_sparse(self):
        # [FUNCTION DOC] Nested-call gradient accumulation check with sparse tensors.
        self._nested_backward_accumulate_grads(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_backwards_nested_python_udf_sparse(self):
        # [FUNCTION DOC] Nested Python function backward check against local autograd, sparse.
        # Run equivalent of _nested_python_udf locally.
        self._backwards_nested_python_udf(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_mixed_requires_grad_sparse(self):
        # [FUNCTION DOC] Mixed requires-grad check with one sparse tensor needing grad, one not.
        self._mixed_requires_grad(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=False),
            True,
        )

    @dist_init
    def test_multiple_backward_sparse(self):
        # [FUNCTION DOC] Repeated backward (1000 times) check with sparse tensors.
        self._multiple_backward(
            build_sparse_tensor(requires_grad=True),
            build_sparse_tensor(requires_grad=True),
            True,
        )

    @dist_init
    def test_embedding_bag_with_no_grad_tensors(self):
        # [FUNCTION DOC] Creates a sparse-gradient embedding bag remotely and locally, runs local
        # backward twice for reference, then runs the remote embedding through an RPC and
        # distributed backward twice (accumulating sparse gradients), and asserts the remote
        # module's weight gradient equals the local one.
        dst = self._next_rank()
        remote_embedding = rpc.remote(
            worker_name(dst),
            torch.nn.EmbeddingBag,
            args=(16, 16),
            kwargs={"mode": "sum", "sparse": True},
        )
        local_embedding = torch.nn.EmbeddingBag(16, 16, mode="sum", sparse=True)

        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        # requires_grad = True to record send/recv functions
        per_sample_weights = torch.rand((8), requires_grad=True)
        offsets = torch.LongTensor([0, 4])

        local_res = local_embedding(input, offsets, per_sample_weights)

        # Run backward twice.
        torch.autograd.backward([local_res.sum()], retain_graph=True)
        torch.autograd.backward([local_res.sum()])
        local_grad = local_embedding.weight.grad

        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._call_remote_embedding,
                args=(remote_embedding, input, offsets, per_sample_weights),
            )

            # Run backward twice to test accumulation of sparse gradients.
            dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [res.sum()])

            remote_grad = rpc.rpc_sync(
                worker_name(dst),
                DistAutogradTest._get_grad,
                args=(remote_embedding, context_id),
            )

            self.assertEqual(local_grad, remote_grad)


class DistAutogradTest(CommonDistAutogradTest):
    # [CLASS DOC] Main multi-process test case for distributed autograd with dense tensors:
    # context management, graph structure, cleanup, backward correctness, error propagation,
    # failure handling, and related static helper functions executed on remote workers. Has a
    # class attribute flag _backward_done (initially false) used by the test with a dying node.
    @dist_init
    def test_autograd_context(self):
        # [FUNCTION DOC] Checks the maximum context id, then creates 200 contexts verifying each
        # id can be retrieved, carries the worker id in its upper 16 bits, and after exit each
        # context can no longer be found (error message names the id).
        # Verify max possible id.
        max_auto_increment = 281474976710655
        self.assertEqual(
            max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id()
        )

        context_ids = []
        for _ in range(200):
            with dist_autograd.context() as context_id:
                self.assertEqual(
                    context_id,
                    dist_autograd._retrieve_context(context_id)._context_id(),
                )
                # First 16 bits should be worker_id.
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)

        for context_id in context_ids:
            with self.assertRaisesRegex(
                RuntimeError,
                f"Could not find autograd context with id: {context_id}",
            ):
                dist_autograd._retrieve_context(context_id)

    @dist_init
    def test_nested_context(self):
        # [FUNCTION DOC] Asserts that entering a second context in the same thread raises a
        # runtime error saying an autograd context id already exists.
        with (
            dist_autograd.context(),
            self.assertRaisesRegex(
                RuntimeError, "Already have an autograd context id for this thread"
            ),
            dist_autograd.context(),
        ):
            pass

    @dist_init
    def test_graph_for_builtin_call(self):
        # [FUNCTION DOC] Graph verification for a built-in add via sync RPC, dense tensors.
        self._test_graph(torch.add, ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_python_call(self):
        # [FUNCTION DOC] Graph verification for a Python function via sync RPC, dense tensors.
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_builtin_remote_call(self):
        # [FUNCTION DOC] Graph verification for a built-in add via remote call, dense tensors.
        self._test_graph(torch.add, ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_python_remote_call(self):
        # [FUNCTION DOC] Graph verification for a Python function via remote call, dense tensors.
        self._test_graph(my_py_add, ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_py_nested_call(self):
        # [FUNCTION DOC] Nested-call graph verification via sync RPC, dense tensors.
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_py_nested_remote_call(self):
        # [FUNCTION DOC] Nested-call graph verification via remote call, dense tensors.
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_py_nested_call_itself(self):
        # [FUNCTION DOC] Self-returning nested-call graph check via sync RPC, dense tensors.
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_py_nested_remote_call_itself(self):
        # [FUNCTION DOC] Self-returning nested-call graph check via remote call, dense tensors.
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, False)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad(self):
        # [FUNCTION DOC] No-graph check for dense tensors without grad, via sync RPC.
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote(self):
        # [FUNCTION DOC] No-graph check for dense tensors without grad, via remote call.
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, False)

    def _test_grad_only_on_return_value(self, exec_mode):
        # [FUNCTION DOC] HELPER: Calls a function on the next rank that returns a tensor requiring
        # grad (arguments need none), runs distributed backward on the result, notifies the peer,
        # waits for the previous rank, and asserts that rank's context holds exactly one gradient,
        # for the returned tensor, equal to ones. Ends with a barrier so contexts outlive the
        # lookup. Raises a value error for modes other than sync RPC or remote.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), ret_requires_grad)
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), ret_requires_grad).to_here()
            else:
                raise ValueError(f"Unrecognized ExecMode {exec_mode}")

            dist_autograd.backward(context_id, [ret.sum()])

            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            grads = dist_autograd.get_gradients(ctx_ids[1])
            self.assertEqual(1, len(grads))
            self.assertIn(requires_grad_tensor, grads)
            self.assertEqual(torch.ones_like(ret), grads[requires_grad_tensor])
            # due to the above get_gradients call, ensure that dist autograd
            # contexts aren't cleaned up until all workers exit context managers
            dist.barrier()

    @dist_init
    def test_grad_only_on_return_value(self):
        # [FUNCTION DOC] Return-value-only gradient check via sync RPC.
        self._test_grad_only_on_return_value(ExecMode.RPC_SYNC)

    @dist_init
    def test_grad_only_on_return_value_remote(self):
        # [FUNCTION DOC] Return-value-only gradient check via remote call.
        self._test_grad_only_on_return_value(ExecMode.REMOTE)

    @dist_init
    def test_rpc_complex_args(self):
        # [FUNCTION DOC] Complex-argument (list of tensors) check via sync RPC, dense tensors.
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_remote_complex_args(self):
        # [FUNCTION DOC] Complex-argument (list of tensors) check via remote call, dense tensors.
        self._test_rpc_complex_args(ExecMode.REMOTE, False)

    @dist_init
    def test_context_cleanup_tensor_with_grad(self):
        # [FUNCTION DOC] Context cleanup check when dense tensors requiring grad are sent.
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_context_cleanup_tensor_no_grad(self):
        # [FUNCTION DOC] Context cleanup check when a dense tensor not requiring grad is sent.
        t1 = torch.ones(3, 3, requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    def test_context_cleanup_no_tensors(self):
        # [FUNCTION DOC] Context cleanup check when only plain numbers are sent.
        self.context_cleanup_test_helper(rpc_args=(1, 1), func=my_scalar_add)

    @dist_init
    def test_context_cleanup_nested_rpc(self):
        # [FUNCTION DOC] Context cleanup check for a nested call chain with dense tensors.
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(
            rpc_args=args, func=my_py_nested_call, nested=True
        )

    @dist_init
    def test_worker_ids_recorded(self):
        # [FUNCTION DOC] Sends adds to every other worker, first with tensors not requiring grad
        # and then requiring grad, and asserts after each phase that the context's known worker
        # ids equal the set of all other ranks.
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            # if no tensors require grad, we should still record worker_ids, as
            # the autograd context ID is still passed to other workers.
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            # all worker_ids in dst_ranks should be recorded.
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)

            # worker_ids should be recorded when tensors do require grad
            t1.requires_grad = True
            t2.requires_grad = True
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            # all worker_ids in dst_ranks should be recorded.
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)

    @dist_init
    def test_dist_autograd_profiling(self):
        # [FUNCTION DOC] Profiles a distributed backward and asserts one send-backward and one
        # recv-backward event were recorded, and that the total CPU time of the distributed
        # backward event exceeds each of them.
        # Defines a nested helper that finds the first profiler event whose name contains a key.
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.add, args=(t1, t2)
            ).sum()
            with torch.autograd.profiler.profile() as p:
                dist_autograd.backward(context_id, [loss])

        function_events = p.function_events

        def get_event(partial_key):
            # [FUNCTION DOC] HELPER: Returns the first profiler event whose name contains the key;
            # raises a stop-iteration error if none matches.
            return next(event for event in function_events if partial_key in event.name)

        send_event = get_event("SendRpcBackward")
        recv_event = get_event("RecvRpcBackward")
        backward_event = get_event("torch::distributed::autograd::backward")
        # There should be at least 1 send and recv_events each, corresponding to send/recv functions executed.
        self.assertEqual(send_event.count, 1)
        self.assertEqual(recv_event.count, 1)
        # The CPU total for backward event should be great than send and recv, since
        # applying those functions in the backwards pass is a subset of the entire backward pass.
        self.assertGreater(backward_event.cpu_time_total, send_event.cpu_time_total)
        self.assertGreater(backward_event.cpu_time_total, recv_event.cpu_time_total)

    @dist_init
    def test_error_in_context(self):
        # [FUNCTION DOC] Asserts an RPC matmul of mismatched shapes made inside a context raises a
        # runtime error.
        with dist_autograd.context():
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(6, 6, requires_grad=True)

            with self.assertRaises(RuntimeError):
                # This should throw an error since matrix sizes don't match.
                rpc.rpc_sync(
                    worker_name(self._next_rank()), torch.matmul, args=(t1, t2)
                )

    @dist_init
    def test_backward_no_grad_on_tensor(self):
        # [FUNCTION DOC] Checks distributed backward leaves tensor grad fields untouched, dense.
        self._backward_no_grad_on_tensor(
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3), requires_grad=True),
            False,
        )

    @dist_init
    def test_backward_simple(self):
        # [FUNCTION DOC] Simple add backward against the next peer, dense tensors.
        self._backward_simple(
            self._next_rank(),
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3), requires_grad=True),
            None,
            False,
        )

    @dist_init
    def test_backward_simple_self(self):
        # [FUNCTION DOC] Simple add backward where the destination is this rank, dense tensors.
        self._backward_simple(
            self.rank,
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3), requires_grad=True),
            None,
            False,
        )

    @dist_init
    def test_backward_rref(self):
        # [FUNCTION DOC] Reference backward check where callee and owner are the next peer.
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(
            callee,
            rref_owner,
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3), requires_grad=True),
            None,
            False,
        )

    @dist_init
    def test_backward_rref_multi(self):
        # [FUNCTION DOC] All ranks except rank 0 use worker0 as both callee and reference owner
        # for the reference backward check; rank 0 does nothing.
        if self.rank > 0:
            callee = "worker0"
            rref_owner = callee
            self._backward_rref(
                callee,
                rref_owner,
                torch.rand((3, 3), requires_grad=True),
                torch.rand((3, 3), requires_grad=True),
                None,
                False,
            )

    @dist_init
    def test_backward_rref_nested(self):
        # [FUNCTION DOC] Reference backward check with callee at rank plus one and owner at rank
        # plus two (nested call), dense tensors.
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(
            callee,
            rref_owner,
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3), requires_grad=True),
            None,
            False,
        )

    @dist_init
    def test_trainer_ps(self):
        # [FUNCTION DOC] Parameter-server/trainer check with dense tensors and Python trainers.
        self._test_trainer_ps(create_tensor, _run_trainer, False)

    @dist_init
    def test_trainer_ps_torchscript_functions(self):
        # [FUNCTION DOC] Same as the parameter-server check but with a TorchScript tensor factory
        # and TorchScript trainers. Sets the RPC API module's flag that ignores reference leaks at
        # shutdown (a process-wide side effect that is never reset).
        # TODO, need more investigation
        # there is rref leak when shutting down, suspect it is because
        # ref as arg is passed to pybind boundary, and the ref is not garbage
        # collected by python when calling shutdown()
        import torch.distributed.rpc.api as api

        api._ignore_rref_leak = True

        self._test_trainer_ps(
            create_torchscript_tensor, _run_trainer_torchscript, False
        )

    @dist_init
    def test_backward_multiple_round_trips(self):
        # [FUNCTION DOC] Multi-round-trip backward check with five dense tensors, alternating
        # requires-grad flags.
        self._backward_multiple_round_trips(
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3)),
            torch.rand((3, 3), requires_grad=True),
            torch.rand((3, 3)),
            torch.rand((3, 3), requires_grad=True),
            None,
            False,
        )

    @dist_init
    def test_backward_different_tensor_dims(self):
        # [FUNCTION DOC] Chains a matrix product and a multi-matrix product of tensors with
        # different shapes in local, sync RPC and remote modes, and verifies gradients against the
        # local run.
        # KNOWN ISSUE: the verification argument list passes the second tensor twice and omits
        # nothing else; it appears to be a typo but is harmless as the second tensor has no grad.
        local_grads = None
        t1 = torch.rand((4, 6), requires_grad=True)
        t2 = torch.rand((6, 5))
        t3 = torch.rand((5, 7), requires_grad=True)
        t4 = torch.rand((7, 9))

        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.matmul, t1, t2)
                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (val, t3, t4))
                loss = val.sum()

                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4
                )
                local_grads = ret if ret else local_grads

    @dist_init
    def test_backward_unused_tensors(self):
        # [FUNCTION DOC] Stacks three tensors and multiplies only the first and last slices, so
        # the middle tensor is unused; verifies gradients (including absence for unused) in local,
        # sync RPC and remote modes.
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                s = self._exec_func(exec_mode, torch.stack, (t1, t2, t3))
                val = self._exec_func(
                    exec_mode,
                    torch.matmul,
                    torch.narrow(s, 0, 0, 1),
                    torch.narrow(s, 0, 2, 1),
                )

                loss = val.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2, t3
                )
                local_grads = ret if ret else local_grads

    @dist_init
    def test_backward_multiple_output_tensors(self):
        # [FUNCTION DOC] Splits one tensor into chunks via the execution mode, combines three of
        # the chunks with a multi-matrix product, and verifies the gradient of the source tensor
        # in local, sync RPC and remote modes.
        local_grads = None
        t = torch.rand((10, 2), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                tensor_list = self._exec_func(exec_mode, torch.split, t, 2)
                t1 = tensor_list[0]
                t2 = tensor_list[2]
                t3 = tensor_list[4]

                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (t1, t2, t3))

                loss = val.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t
                )
                local_grads = ret if ret else local_grads

    def _run_test_backward_unused_send_function_in_thread(self):
        # [FUNCTION DOC] HELPER: Thread body. Makes an RPC whose result is unused, then runs
        # distributed backward on an unrelated local loss; the backward is expected to hang
        # forever because a send function is never reached.
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            # We don't use the result of an RPC function, as a result the
            # backward pass would hang in the "FAST" mode.
            rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))

            val = torch.mul(t1, t2)

            # Run backward, this would hang forever.
            dist_autograd.backward(context_id, [val.sum()])

    @dist_init
    def test_backward_unused_send_function(self):
        # [FUNCTION DOC] Runs the hanging-backward body in a daemon thread, waits 10 seconds and
        # asserts the thread is still alive. The daemon thread is left running.
        # Run the test in a thread which would never finish.
        t = threading.Thread(
            target=self._run_test_backward_unused_send_function_in_thread
        )
        t.daemon = True
        t.start()
        t.join(10)  # Wait for 10s.

        # Verify thread is still alive (indicating backward hasn't completed yet).
        self.assertTrue(t.is_alive())

    @dist_init
    def test_backward_autograd_engine_error(self):
        # [FUNCTION DOC] Inserts the error-simulating autograd function into a computation, passes
        # the result through four chained RPC operations on different peers, and asserts that
        # distributed backward raises a runtime error naming a node and the simulated message.
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            # Perform some ops before error simulation.
            tmp = (t1 + t2) * (t1 + t2)
            t3 = SimulateBackwardError.apply(tmp)

            # Run multiple round trips across different nodes and verify the
            # original node receives an error thrown on a node deep in the chain.
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t2, t3))
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.mul, args=(val, t2)
            )
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.matmul, args=(val, t2)
            )
            val = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.div, args=(val, t2)
            )

            with self.assertRaisesRegex(
                RuntimeError, "Error on Node [0-9]+: Simulate error on backward pass"
            ):
                # Run backwards, and validate we receive an error.
                dist_autograd.backward(context_id, [val.sum()])

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_backward_node_failure(self):
        # [FUNCTION DOC] Sets a 5-second RPC timeout (global side effect) and runs an RPC add.
        # After a barrier, odd ranks exit while even ranks wait for each odd rank to fail and
        # assert that distributed backward raises a runtime error matching the shutdown error
        # pattern.
        # Skipped (passes) in sandcastle on macOS because of flakiness.
        rpc._set_rpc_timeout(5)  # 5 seconds
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))

            # Wait for all RPCs to be done.
            dist.barrier()

            # Kill all odd rank nodes.
            if self.rank % 2 == 0:
                shutdown_error_regex = self.get_shutdown_error_regex()
                # Wait for all other nodes to die.
                for rank in range(self.world_size):
                    if rank % 2 != 0:
                        wait_until_node_failure(rank, shutdown_error_regex)

                # Shutdown sequence is not very well defined and as a result
                # we might see any error given by get_shutdown_error_regex()
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    # Run backwards, and validate we receive an error since all
                    # other nodes are dead.
                    dist_autograd.backward(context_id, [res.sum()])
            else:
                # Exit all other nodes.
                pass

    @dist_init
    def test_backward_without_context(self):
        # [FUNCTION DOC] Asserts distributed backward with a nonexistent context id (100) raises a
        # runtime error saying the context could not be found.
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        context_id = 100  # dummy context_id
        with self.assertRaisesRegex(
            RuntimeError,
            f"Could not find autograd context with id: {context_id}",
        ):
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])

    @dist_init
    def test_backward_without_rpc(self):
        # [FUNCTION DOC] Runs distributed backward on a purely local add and asserts the context
        # holds exactly two gradients, both ones.
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)

            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_backward_invalid_args(self):
        # [FUNCTION DOC] Asserts the expected errors for invalid backward inputs: wrong argument
        # types (type error), empty root list, root not requiring grad, non-scalar root, and a root
        # without a gradient function (runtime errors with matching messages).
        with dist_autograd.context() as context_id:
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                dist_autograd.backward(context_id, None)

            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                dist_autograd.backward(None, None)

            with self.assertRaisesRegex(
                RuntimeError, "No tensors provided for gradient computation"
            ):
                dist_autograd.backward(context_id, [])

            with self.assertRaisesRegex(RuntimeError, "requires_grad not set on"):
                t = torch.rand(3, 3)
                dist_autograd.backward(context_id, [t])

            with self.assertRaisesRegex(
                RuntimeError, "is not a scalar, all roots need to be scalar"
            ):
                t = torch.rand(3, 3, requires_grad=True)
                dist_autograd.backward(context_id, [t])

            with self.assertRaisesRegex(
                RuntimeError, "does not have a valid gradient function"
            ):
                t = torch.rand(1, requires_grad=True)
                dist_autograd.backward(context_id, [t])

    @dist_init
    def test_backward_multiple_roots(self):
        # [FUNCTION DOC] Computes four scalar roots (add, mul, cos, div reductions) in local and
        # sync RPC modes, runs backward on all roots at once and verifies the gradients.
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
            with dist_autograd.context() as context_id:
                r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
                r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
                r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()

                local_grads = self._verify_backwards(
                    exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2
                )

    @dist_init
    def test_backward_different_dtypes(self):
        # [FUNCTION DOC] Backward check with dense tensors of 32-bit and 64-bit float dtypes.
        self._backward_different_dtypes(
            torch.rand((3, 3), requires_grad=True, dtype=torch.float32),
            torch.rand((3, 3), requires_grad=True, dtype=torch.float64),
            False,
        )

    @dist_init
    def test_backward_simple_python_udf(self):
        # [FUNCTION DOC] Backward check through a Python function with dense tensors.
        self._backward_simple_python_udf(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False,
        )

    @dist_init
    def test_backward_simple_script_call(self):
        # [FUNCTION DOC] Backward check through a TorchScript function with dense tensors.
        self._backward_simple_script_call(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False,
        )

    @staticmethod
    def _complex_python_udf(t1, t2):
        # [FUNCTION DOC] HELPER: Static function chaining three linear operations on the inputs
        # and returning the multi-matrix product of all five tensors.
        t3 = torch.nn.functional.linear(t1, t2)
        t4 = torch.nn.functional.linear(t2, t3)
        t5 = torch.nn.functional.linear(t3, t4)
        return torch.linalg.multi_dot([t1, t2, t3, t4, t5])

    @dist_init
    def test_backward_complex_python_udf(self):
        # [FUNCTION DOC] Runs the complex Python function locally and by remote call and verifies
        # that the distributed gradients equal the local ones.
        # Run the same code locally and with dist autograd and verify gradients
        # are same.
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(
                    exec_mode, DistAutogradTest._complex_python_udf, t1, t2
                )
                loss = ret.sum()
                local_grads = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )

    @staticmethod
    def _python_udf_with_backward_error(t1, t2):
        # [FUNCTION DOC] HELPER: Static function adding the inputs, passing the sum through the
        # error-simulating autograd function, and returning a multi-matrix product.
        t3 = t1 + t2
        t4 = SimulateBackwardError.apply(t3)
        return torch.linalg.multi_dot([t1, t2, t3, t4])

    @staticmethod
    def _nested_rpc_call_backward_error(t1, t2, dst):
        # [FUNCTION DOC] HELPER: Static function that derives two tensors from the inputs, calls
        # the backward-error function on worker rank dst via sync RPC, and returns a multi-matrix
        # product of the tensors and the result.
        t1 = t1 * t2
        t2 = t1 + t2
        res = rpc.rpc_sync(
            worker_name(dst),
            DistAutogradTest._python_udf_with_backward_error,
            args=(t1, t2),
        )
        return torch.linalg.multi_dot([t1, t2, res])

    @dist_init
    def test_backward_python_udf_error(self):
        # [FUNCTION DOC] Asserts that an error raised in a backward pass deep in a nested call
        # chain propagates to the caller's distributed backward as a runtime error with the
        # simulated message.
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                worker_name(self._next_rank()),
                DistAutogradTest._nested_rpc_call_backward_error,
                args=(t1, t2, self._next_rank()),
            )
            with self.assertRaisesRegex(
                RuntimeError, "Simulate error on backward pass"
            ):
                dist_autograd.backward(context_id, [loss.sum()])

    _backward_done = False

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_backward_node_failure_python_udf(self):
        # [FUNCTION DOC] Sets a 5-second RPC timeout (global side effect) and makes a nested call
        # chain (one extra hop) from each rank. After a barrier rank 2, the last hop, returns early
        # and stops. Rank 0 waits for rank 2 to fail, asserts distributed backward raises a runtime
        # error matching the shutdown error pattern, then sets a done key in the shared store; the
        # remaining ranks wait up to 10 seconds for that key. Skipped (passes) in sandcastle on
        # macOS.
        # Set a short timeout to quickly time out failed RPCs.
        rpc._set_rpc_timeout(5)  # 5 seconds
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            dst = self._next_rank()
            res = rpc.rpc_sync(
                worker_name(dst),
                my_py_nested_call,
                args=(t1, t2, dst, self.world_size, 1),
            )

            dist.barrier()

            # Kill rank 2 (last hop of nested rpc) and verify rank 0 receives an error.
            if self.rank == 2:
                return

            store = dist.distributed_c10d._get_default_store()
            if self.rank == 0:
                # Wait for rank 2 to die.
                shutdown_error_regex = self.get_shutdown_error_regex()
                wait_until_node_failure(2, shutdown_error_regex)
                # Shutdown sequence is not very well defined and as a result
                # we might see any error given by get_shutdown_error_regex().
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    # Run backwards, and validate we receive an error since rank 2 is dead.
                    dist_autograd.backward(context_id, [res.sum()])

                # Mark rank 0 is done in the store, since the RPC framework on
                # some nodes might be broken at this point.
                store.set("test_backward_node_failure_python_udf_rank0_done", "True")
            else:
                # Wait for backward to finish on rank 0.
                store.wait(
                    ["test_backward_node_failure_python_udf_rank0_done"],
                    timedelta(seconds=10),
                )

    @staticmethod
    def _nested_python_udf(t1, t2, dst):
        # [FUNCTION DOC] HELPER: Static function that computes a product and a sum of the inputs,
        # adds them on worker rank dst via sync RPC, and returns the product of the inputs, the
        # product, the sum, and the remote result.
        t3 = t1 * t2
        t4 = t1 + t2
        res = rpc.rpc_sync(worker_name(dst), my_py_add, args=(t3, t4))
        return t1 * t2 * t3 * t4 * res

    @dist_init
    def test_backwards_nested_python_udf(self):
        # [FUNCTION DOC] Nested Python function backward check against local autograd, dense.
        # Run equivalent of _nested_python_udf locally.
        self._backwards_nested_python_udf(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False,
        )

    _test_clean_context_backward_context_id = None

    class MyBackwardFunc(Function):
        # [CLASS DOC] HELPER: Identity autograd function whose backward releases the shared test
        # context while the backward pass is running, to simulate an error that cleans up the
        # context early.
        @staticmethod
        def forward(ctx, input):
            # [FUNCTION DOC] HELPER: Returns the input unchanged.
            return input

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            # [FUNCTION DOC] HELPER: Requires the class-level test context id to be set. Waits at a
            # barrier so all nodes reach backward, releases that context, asserts all contexts are
            # cleaned up, and returns the gradient unchanged.
            if DistAutogradTest._test_clean_context_backward_context_id is None:
                raise AssertionError(
                    "Expected _test_clean_context_backward_context_id to not be None"
                )

            # Release the context to simulate error (use barrier before releasing
            # context to ensure all nodes execute the backward function).
            dist.barrier()
            dist_autograd._release_context(
                DistAutogradTest._test_clean_context_backward_context_id
            )

            # Verify all contexts are cleaned up.
            if not _all_contexts_cleaned_up():
                raise AssertionError("Expected all contexts to be cleaned up")

            return input

    @dist_init
    def test_clean_context_during_backward(self):
        # [FUNCTION DOC] Creates a context without the context manager, shares its id with all
        # other nodes, builds a chain of 100 remote adds ending in the context-releasing function,
        # and asserts that distributed backward with a bogus context id raises a runtime error
        # (the process must not crash). Finally kills the workers: barrier, non-graceful RPC
        # shutdown and process exit with status zero, because other nodes would otherwise hang in
        # the autograd engine.
        # KNOWN ISSUE: the backward is called with a dummy id (100), not the real context, so the
        # early-release path in the custom function appears never to be exercised by this call.
        """
        This test simulates the situation where the 'backward' call might throw
        an exception locally which would lead to the autograd context being
        cleaned up if we're using the context manager. As a result, the autograd
        context might be cleaned up while some threads are still using the
        autograd context.

        It is fine for the 'backward' call to throw an exception in this test,
        but the process should not crash.
        """
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        context = dist_autograd._new_context()
        context_id = context._context_id()
        DistAutogradTest._test_clean_context_backward_context_id = context_id

        # Send the context id to all nodes.
        for i in range(self.world_size):
            if i != self.rank:
                rank_distance = (i - self.rank + self.world_size) % self.world_size
                rpc.rpc_sync(
                    worker_name(i),
                    _set_rpc_done,
                    args=(context_id, rank_distance),
                )

        dist.barrier()

        # Verify all context ids have been received.
        self.assertEqual(self.world_size - 1, len(known_context_ids))

        t1 = torch.rand((3, 3), requires_grad=True)
        for _ in range(100):
            dst = self._next_rank()
            t1 = rpc.rpc_sync(worker_name(dst), torch.add, args=(t1, t1))

        # Call MyBackwardFunc as the first op of the backward pass to
        # ensure we release the context early in the backward pass.
        t1 = DistAutogradTest.MyBackwardFunc.apply(t1)
        self.assertEqual(100, len(context._send_functions()))

        context_id = 100  # dummy context_id
        with self.assertRaisesRegex(
            RuntimeError,
            f"Could not find autograd context with id: {context_id}",
        ):
            dist_autograd.backward(context_id, [t1.sum()])

        # HACK: Killing workers since otherwise the autograd engine gets stuck on
        # other nodes. The proper fix would be addressing:
        # https://github.com/pytorch/pytorch/issues/27643, which would inform
        # other nodes about the failure.
        # The autograd engine gets stuck on other nodes since they're waiting to
        # receive gradients from the node that received an error (and as a
        # result it didn't execute the rest of the graph).
        dist.barrier()
        rpc.shutdown(graceful=False)
        sys.exit(0)

    @classmethod
    def _call_remote_embedding(cls, embedding_rref, input, offsets, per_sample_weights):
        # [FUNCTION DOC] HELPER: Class method run on the reference owner: applies the locally held
        # embedding module to the input, offsets and per-sample weights and returns the output.
        embedding = embedding_rref.local_value()
        return embedding(input, offsets, per_sample_weights)

    @classmethod
    def _get_grad(cls, embedding_rref, context_id):
        # [FUNCTION DOC] HELPER: Class method run on the reference owner: returns the gradient the
        # given context accumulated for the locally held embedding module's weight.
        embedding = embedding_rref.local_value()
        grad_map = dist_autograd.get_gradients(context_id)
        return grad_map[embedding.weight]

    @classmethod
    def _mixed_requires_grad_operaton(cls, t1, t2):
        # [FUNCTION DOC] HELPER: Class method returning the difference of the inputs if t2 requires
        # grad, otherwise their product. (The name keeps a historical misspelling.)
        if t2.requires_grad:
            return t1 - t2
        else:
            return t1 * t2

    @dist_init
    def test_mixed_requires_grad(self):
        # [FUNCTION DOC] Mixed requires-grad check with dense tensors (first needs grad, second
        # does not).
        self._mixed_requires_grad(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=False),
            False,
        )

    class TestDebugInfoFunc(Function):
        # [CLASS DOC] HELPER: Identity autograd function whose backward checks the debug info of
        # the distributed autograd engine while a backward pass is in progress.
        @staticmethod
        def forward(ctx, input):
            # [FUNCTION DOC] HELPER: Returns the input unchanged.
            return input

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            # [FUNCTION DOC] HELPER: Reads the engine debug info and asserts the number of current
            # backward passes is between 1 and 4 (raises an assertion error otherwise); returns
            # the gradient unchanged. Does not block, to avoid deadlocking the single CPU thread.
            debug_info = dist_autograd._get_debug_info()
            if debug_info is None:
                raise AssertionError("Expected debug_info to not be None")
            backward_passes = int(debug_info["num_current_backward_passes"])

            # Hard to validate exact numbers because of the distributed nature.
            # We can't use a barrier() here since that would block the single
            # CPU thread available for autograd and can cause deadlocks.
            if not (1 <= backward_passes <= 4):
                raise AssertionError(
                    f"Expected 1 <= backward_passes <= 4, got {backward_passes}"
                )
            return input

    @dist_init
    def test_debug_info(self):
        # [FUNCTION DOC] Builds a chain of remote adds through all other ranks, a debug-checking
        # function in the middle, and another chain, then runs distributed backward and asserts
        # the debug info has between 1 and 4 contexts. After all ranks signal done and meet at a
        # barrier, asserts zero current backward passes, exactly two debug entries, and that
        # contexts are eventually cleaned up (zero contexts).
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            i = 0
            res = {}
            res[i] = t1
            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(
                        worker_name(rank), torch.add, args=(res[i], t2)
                    )
                    i += 1

            # Call custom function in middle of backward pass to ensure all
            # nodes are still waiting on a backward().
            res[i + 1] = DistAutogradTest.TestDebugInfoFunc.apply(res[i])
            i += 1

            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(
                        worker_name(rank), torch.add, args=(res[i], t2)
                    )
                    i += 1

            dist_autograd.backward(context_id, [res[i].sum()])

            debug_info = dist_autograd._get_debug_info()
            num_autograd_context = int(debug_info["num_autograd_contexts"])
            # Need at least one context and not more than 4.
            self.assertTrue(num_autograd_context >= 1 and num_autograd_context <= 4)

        for rd in range(self.world_size - 1):
            rpc.rpc_sync(
                worker_name((self.rank + rd + 1) % self.world_size),
                _set_rpc_done,
                args=(context_id, rd + 1),
            )

        dist.barrier()

        # Validate information
        debug_info = dist_autograd._get_debug_info()
        if debug_info is None:
            raise AssertionError("Expected debug_info to not be None")
        self.assertEqual(0, int(debug_info["num_current_backward_passes"]))
        # only have `num_current_backward_passes` and `num_autograd contexts`
        self.assertTrue(len(debug_info) == 2)

        self.assertTrue(_all_contexts_cleaned_up())

        # All contexts should be cleaned up.
        debug_info = dist_autograd._get_debug_info()
        self.assertEqual(0, int(debug_info["num_autograd_contexts"]))

    @staticmethod
    def _workload_thread():
        # [FUNCTION DOC] HELPER: Thread body. Inside a new context, chains four RPC operations
        # (add, multiply, matrix product, add) all executed on worker0 and runs distributed
        # backward on the sum of the final result.
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = rpc.rpc_sync("worker0", torch.add, args=(t1, t2))
            t4 = rpc.rpc_sync("worker0", torch.mul, args=(t2, t3))
            t5 = rpc.rpc_sync("worker0", torch.matmul, args=(t3, t4))
            t6 = rpc.rpc_sync("worker0", torch.add, args=(t4, t5))

            dist_autograd.backward(context_id, [t6.sum()])

    @dist_init
    def test_async_dist_autograd(self):
        # [FUNCTION DOC] Initializes the process group; every rank except rank 0 starts 20 threads
        # that each run the workload against worker0 and joins them, stressing asynchronous
        # processing of many concurrent backward calls on one node. All ranks end at a barrier.
        """
        This test ensures async processing for distributed autograd works
        appropriately. This is achieved by spawning multiple threads and
        hammering a single node with a lot of backward() calls.
        """

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank != 0:
            # All other ranks schedule work on rank 0.
            threads = []
            for _ in range(20):
                t = threading.Thread(target=DistAutogradTest._workload_thread)
                t.start()
                threads.append(t)

            for thread in threads:
                thread.join()

        dist.barrier()

    @dist_init
    def test_backward_accumulate_grads(self):
        # [FUNCTION DOC] Runs a local matrix product with ordinary backward twice, then the same
        # product on a peer with distributed backward twice (first retaining the graph), and
        # asserts the context holds two gradients equal to the tensors' accumulated local grads.
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = torch.matmul(t1, t2)
            # Run backward twice.
            torch.autograd.backward([t3.sum()], retain_graph=True)
            torch.autograd.backward([t3.sum()])

            t3 = rpc.rpc_sync(
                worker_name(self._next_rank()), torch.matmul, args=(t1, t2)
            )
            # Run backward twice.
            dist_autograd.backward(context_id, [t3.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [t3.sum()])

            # Verify the gradients are same for local and remote execution.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    @staticmethod
    def _test_nested_backward_accumulate_grads(t1, t2, dst_rank):
        # [FUNCTION DOC] HELPER: Static function that adds the tensors on worker rank dst_rank via
        # sync RPC and returns the result.
        return rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))

    @dist_init
    def test_nested_backward_accumulate_grads(self):
        # [FUNCTION DOC] Nested-call gradient accumulation check with dense tensors.
        self._nested_backward_accumulate_grads(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False,
        )

    @dist_init
    def test_multiple_backward(self):
        # [FUNCTION DOC] Repeated backward (1000 times) check with dense tensors.
        self._multiple_backward(
            torch.rand(3, 3, requires_grad=True),
            torch.rand(3, 3, requires_grad=True),
            False,
        )

    @dist_init(clean_shutdown=False)
    def test_multiple_backward_with_errors(self):
        # [FUNCTION DOC] Runs distributed backward 100 times on a retained graph whose remote part
        # contains the error-simulating function. The first 50 calls must raise the simulated
        # error; at the middle iteration all ranks synchronize and turn the class-level error flag
        # off; the remaining calls must succeed. A final barrier and reset of the flag to true
        # always run (global state restored).
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(
                f"worker{self._next_rank()}",
                DistAutogradTest._python_udf_with_backward_error,
                args=(t1, t2),
            ).sum()

            try:
                # Run backward in a loop multiple times.
                for i in range(100):
                    if i < 50:
                        with self.assertRaisesRegex(
                            RuntimeError, "Simulate error on backward pass"
                        ):
                            dist_autograd.backward(
                                context_id, [loss], retain_graph=True
                            )
                    elif i > 50:
                        # Recovered from error.
                        dist_autograd.backward(context_id, [loss], retain_graph=True)
                    else:
                        dist.barrier()
                        SimulateBackwardError._simulate_error = False
                        dist.barrier()
            finally:
                # Sync before resetting flag.
                dist.barrier()

                # Reset the flag.
                SimulateBackwardError._simulate_error = True

    @dist_init
    def test_backward_verify_hooks(self):
        # [FUNCTION DOC] Registers a hook doubling the gradient of one tensor, then verifies in
        # local, sync RPC and remote modes that distributed gradients match the local ones, so
        # hooks are honored by distributed backward.
        t1 = torch.ones((3, 3), requires_grad=True)
        # Double the gradient.
        t1.register_hook(lambda grad: grad * 2)
        t2 = torch.ones((3, 3), requires_grad=True)
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, torch.matmul, t1, t2)
                loss = ret.sum()
                ret = self._verify_backwards(
                    exec_mode, [loss], context_id, local_grads, t1, t2
                )
                local_grads = ret if ret else local_grads

    @dist_init
    def test_no_grad_copy(self):
        # [FUNCTION DOC] Uses custom autograd functions that record the memory address of the
        # gradient they receive. Verifies distributed backward copies a non-contiguous gradient,
        # does not copy (same address) a gradient that feeds a single input, and does copy
        # gradients shared between two inputs into distinct buffers.
        """
        Similar to test in test_autograd.py.
        """

        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Two-input add whose backward records the address of the incoming
            # gradient in a class attribute and returns the gradient for both inputs.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Records the gradient's data address and returns the same
                # gradient for both inputs.
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class MyFuncSingleGrad(Function):
            # [CLASS DOC] HELPER: Identity function whose backward records the address of the
            # incoming gradient in a class attribute and passes the gradient through.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inp

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Records the gradient's data address and returns it.
                MyFuncSingleGrad.static_grad_ptr = grad.data_ptr()
                return grad

        class NonContGradFunc(Function):
            # [CLASS DOC] HELPER: Reduces its input to a one-element tensor; its backward returns
            # a non-contiguous (expanded) gradient with the input's shape.
            @staticmethod
            def forward(ctx, inp1):
                # [FUNCTION DOC] HELPER: Remembers the input size and returns a constant
                # one-element tensor.
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Returns a single-element ones tensor expanded to the
                # input size (non-contiguous).
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        with dist_autograd.context() as context_id:
            dist_autograd.backward(
                context_id, [NonContGradFunc.apply(MyFunc.apply(a, b))]
            )
            grads = dist_autograd.get_gradients(context_id)
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)

        # test case that should trigger no copy for a
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFuncSingleGrad.apply(a)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFuncSingleGrad.static_grad_ptr
            p_a = grads[a].data_ptr()
            # Verify there was no clone.
            self.assertTrue(p_a == p_g)

        # Test case that should trigger copy for both of a,b. This is
        # different in the distributed autograd case since we hold
        # a reference to all grads in a vector until all accumulation is done.
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFunc.apply(a, b)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a].data_ptr()
            p_b = grads[b].data_ptr()
            # check a,b uses different grad buffer
            self.assertFalse(p_a == p_b)
            # both should be copied.
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)

    @dist_init
    def test_no_grad_copy_sparse(self):
        # [FUNCTION DOC] Sparse-gradient variant of the no-copy check using embedding bags.
        # Verifies a sparse gradient's values buffer is reused (same address) when it feeds a
        # single input, and is copied when the indices and values are non-contiguous. Also runs
        # backward ten more times in each case to check accumulation.
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Identity function whose backward records the address of the
            # sparse gradient's values buffer in a class attribute and passes the gradient on.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inp

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Records the values buffer address and returns the grad.
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad

        class NonContGradFunc(Function):
            # [CLASS DOC] HELPER: Two-input add whose backward ignores the incoming gradient and
            # returns a sparse gradient built from expanded (non-contiguous) indices and values.
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Builds a sparse gradient with non-contiguous indices and
                # values, records its values buffer address, and returns it for both inputs.
                # Create a sparse tensor with non-contiguous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for a.
        with dist_autograd.context() as context_id:
            emb_matrix = MyFunc.apply(a)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            # check a uses the same buffer
            self.assertTrue(p_a == p_g)

            # Run backwards multiple times.
            for _ in range(10):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        with dist_autograd.context() as context_id:
            emb_matrix = NonContGradFunc.apply(a, b)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = NonContGradFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            p_b = grads[b]._values().data_ptr()
            # check a,b uses different grad buffer
            self.assertFalse(p_a == p_b)
            # Verify we cloned both grads.
            self.assertFalse(p_a == p_g)
            self.assertFalse(p_b == p_g)

            # Run backwards multiple times to verify accumulation.
            for _ in range(10):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    @dist_init
    def test_grad_copy_sparse_indices_extra_ref(self):
        # [FUNCTION DOC] Checks that holding extra references to a sparse gradient's indices and
        # values (which are views and do not raise the buffer reference count) does not prevent
        # distributed backward from taking over the values buffer without copying.
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            # [CLASS DOC] HELPER: Identity function whose backward records the address of the
            # sparse gradient's values buffer and keeps extra references to its indices and values
            # views in class attributes.
            static_grad_ptr = None
            static_grad_indices_ref = None
            static_grad_values_ref = None

            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return inp

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Records the values buffer address, stores views of the
                # indices and values, and returns the gradient.
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                # indices() and values() return views, so holding onto
                # references of them would not increment refcount of indices
                # and values inside the sparse tensor.
                MyFunc.static_grad_indices_ref = grad._indices()
                MyFunc.static_grad_values_ref = grad._values()
                return grad

        a = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        with dist_autograd.context() as context_id:
            emb_matrix = MyFunc.apply(a)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            self.assertIsNotNone(MyFunc.static_grad_indices_ref)
            self.assertIsNotNone(MyFunc.static_grad_values_ref)
            # grad would be stolen, since static_grad_indices_ref and
            # static_grad_values_ref are holding onto views and don't bump the
            # refcount.
            self.assertTrue(p_g == p_a)

    @dist_init
    def test_post_hooks(self):
        # [FUNCTION DOC] Registers post hooks (adding one and two to a counter) on the two
        # accumulate-grad nodes of a sum of a tensor with itself: two hooks on the first node and
        # one on the second. Asserts the counter is 5 after distributed backward and the context
        # holds a single gradient for the tensor.
        self.hook_called_times = 0

        def post_hook_add_one(output_grads, input_grads):
            # [FUNCTION DOC] HELPER: Adds one to the test's call counter and returns the output
            # grads unchanged.
            self.hook_called_times += 1
            return output_grads

        def post_hook_add_two(output_grads, input_grads):
            # [FUNCTION DOC] HELPER: Adds two to the test's call counter and returns the output
            # grads unchanged.
            self.hook_called_times += 2
            return output_grads

        t = torch.rand(10, 10, requires_grad=True)
        a = t + t

        # Register post hooks
        accumulate_grad_0 = a.grad_fn.next_functions[0][0]
        accumulate_grad_0.register_hook(post_hook_add_one)
        accumulate_grad_0.register_hook(post_hook_add_two)

        accumulate_grad_1 = a.grad_fn.next_functions[1][0]
        accumulate_grad_1.register_hook(post_hook_add_two)

        with dist_autograd.context() as context_id:
            loss = a.sum()
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(5, self.hook_called_times)
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(1, len(grads))
            self.assertTrue(t in grads)

    @staticmethod
    def _slow_add(t1, t2):
        # [FUNCTION DOC] HELPER: Static function that sleeps one second, then returns the sum with
        # requires_grad set to true on the result.
        time.sleep(1)
        t3 = t1 + t2
        t3.requires_grad = True
        return t3

    @dist_init
    def test_thread_local_context_id(self):
        # [FUNCTION DOC] Creates a remote slow add on the next rank, then runs distributed
        # backward on its fetched result so the continuation executes on a different thread than
        # the one that created the reference. Asserts the owner's gradient equals the locally
        # computed gradient, showing the context id is propagated across threads.
        t1 = torch.rand((3, 3))
        t2 = torch.rand((3, 3))

        t3 = t1 + t2
        t3.requires_grad = True
        t3.sum().backward()

        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, DistAutogradTest._slow_add, args=(t1, t2))

        with dist_autograd.context() as context_id:
            loss = rref.to_here().sum()
            # due to slow add, the continuation of this backward pass will be
            # invoked by the previous rpc.remote thread which does not have a
            # valid context_id. So, this can test whether we propagate
            # thread_local states properly when jumping across threads on the
            # server side.
            dist_autograd.backward(context_id, [loss])
            self.assertTrue(
                rpc.rpc_sync(
                    dst, _compare_owner_value, args=(context_id, rref, t3.grad)
                )
            )


class CudaDistAutogradTest(CommonDistAutogradTest):
    # [CLASS DOC] Dist autograd tests that involve CUDA tensors; each test requires at least one
    # GPU.
    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/70753")
    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_simple(self):
        # [FUNCTION DOC] Adds two GPU tensors, runs ordinary backward as the reference, then
        # distributed backward in a context, and asserts two gradients equal the reference grads.
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        t2 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        (t1 + t2).sum().backward()
        with dist_autograd.context() as context_id:
            t3 = t1 + t2
            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_to_cpu_continuation(self):
        # [FUNCTION DOC] For three iterations, builds a computation that alternates between CPU
        # work (local or sync RPC) and GPU work, with a CPU root, and verifies distributed
        # gradients against the local run.
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        t2 = torch.rand(3, 3, requires_grad=True)
        # Run a few iterations.
        for _ in range(3):
            t1.grad = None
            t2.grad = None
            # Root is CPU
            local_grads = None
            for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
                with dist_autograd.context() as context_id:
                    t3 = self._exec_func(exec_mode, torch.add, t2, t2)
                    t4 = t3.cuda(0) + t1
                    t5 = self._exec_func(exec_mode, torch.add, t4.cpu(), t2)
                    t6 = t5.cuda(0) + t4
                    t7 = self._exec_func(exec_mode, torch.add, t6.cpu(), t5)
                    # Autograd graph consists of CPU -> GPU -> CPU execution.
                    ret = self._verify_backwards(
                        exec_mode, [t7.sum()], context_id, local_grads, t1, t2
                    )
                    local_grads = ret if ret else local_grads

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_gpu_to_cpu_continuation_gpu_root(self):
        # [FUNCTION DOC] Like the CPU-root variant but the computation ends on a GPU tensor.
        # KNOWN ISSUE: the inline comment says the root is CPU, but the root (last tensor) is on
        # the GPU in this test.
        t1 = torch.rand(3, 3, requires_grad=True, device="cuda:0")
        t2 = torch.rand(3, 3, requires_grad=True)
        # Run a few iterations.
        for _ in range(3):
            t1.grad = None
            t2.grad = None
            # Root is CPU
            local_grads = None
            for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
                with dist_autograd.context() as context_id:
                    t3 = self._exec_func(exec_mode, torch.add, t2, t2)
                    t4 = t3.cuda(0) + t1
                    t5 = self._exec_func(exec_mode, torch.add, t4.cpu(), t2)
                    t6 = t5.cuda(0) + t4
                    # Autograd graph consists of CPU -> GPU -> CPU execution.
                    ret = self._verify_backwards(
                        exec_mode, [t6.sum()], context_id, local_grads, t1, t2
                    )
                    local_grads = ret if ret else local_grads


class FaultyAgentDistAutogradTest(RpcAgentTestFixture):
    # [CLASS DOC] Dist autograd tests run with the faulty TensorPipe agent that deliberately fails
    # some sends, checking that contexts are still cleaned up and the backend options are as
    # configured.
    # Reusing a simplified helper function from DistAutogradTest to ensure
    # autograd context is successfully cleaned up even when RPCs are failing.
    def context_cleanup_test_helper(self, rpc_args, func):
        # [FUNCTION DOC] HELPER: Simplified copy of the common cleanup helper. Initializes the
        # process group, calls func with rpc_args and a done notification on every other worker
        # within a context, then asserts the local context is gone, waits at a barrier and asserts
        # all known contexts are eventually cleaned up.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # test that in dist autograd, in the case that tensors communicated over RPC do
        # NOT require grad, we still cleanup the dist autograd contexts created
        # on other nodes. This is because the autograd context is still
        # communicated over RPC even if tensor arguments do not require grad, as
        # it is possible that the response could.
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}

        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
        # the thread's context id should be cleaned up
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        # Ensure all peers have finished mutating the
        # `known_context_ids` set.
        dist.barrier()
        # check that all contexts have been cleaned up.
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    # no faulty_messages defined so this fails all retryable messages - see
    # faulty_rpc_agent_test_fixture.py for the list of retryable messages.
    @dist_init
    def test_context_cleanup_tensor_with_grad(self):
        # [FUNCTION DOC] Context cleanup check with dense tensors requiring grad while the faulty
        # agent fails sends of all retryable message types.
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_verify_backend_options(self):
        # [FUNCTION DOC] Asserts the backend is the faulty TensorPipe type with 8 worker threads, 3
        # failed sends, and 4 message types configured to fail.
        self.assertEqual(
            self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE
        )
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)


class WrapperModule(nn.Module):
    # [CLASS DOC] HELPER: Wraps a model moved to a given device so it can be hosted on a remote
    # worker; exposes the wrapped model's forward and a way to read its parameter gradients from
    # a dist autograd context.
    def __init__(self, model, device):
        # [FUNCTION DOC] HELPER: Moves the model to the device and stores it.
        super().__init__()
        self.model = model.to(device)

    def forward(self, *args):
        # [FUNCTION DOC] HELPER: Calls the wrapped model on the arguments and returns its output.
        return self.model(*args)

    def gradients(self, ctx_id):
        # [FUNCTION DOC] HELPER: Returns the list of gradients, in parameter order, that the given
        # dist autograd context accumulated for the wrapped model's parameters.
        grads = dist_autograd.get_gradients(ctx_id)
        return [grads[p] for p in self.model.parameters()]


class TensorPipeCudaDistAutogradTest(RpcAgentTestFixture):
    # [CLASS DOC] Dist autograd tests for the TensorPipe agent with CUDA device maps; each test
    # needs at least four GPUs and initializes and shuts down RPC itself.
    @skip_if_lt_x_gpu(4)
    def test_device_maps_backward_pass(self):
        # [FUNCTION DOC] Configures a device map from this rank's GPU to the next rank's GPU for
        # the next worker, initializes RPC, adds two GPU tensors remotely, and asserts distributed
        # backward returns ones gradients placed on the original devices (reverse map used).
        # Shuts RPC down at the end.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)

        # The reverse of this device mapping should be used for the backward pass.
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        t1 = torch.rand(10, device=self.rank, requires_grad=True)
        t2 = torch.rand(10, device=self.rank, requires_grad=True)
        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(dst, torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(torch.ones(10), grads[t1])
            self.assertEqual(torch.ones(10), grads[t2])
            self.assertEqual(t1.device, grads[t1].device)
            self.assertEqual(t2.device, grads[t2].device)

        rpc.shutdown()

    class MyRemoteCompute(torch.nn.Module):
        # [CLASS DOC] HELPER: Module hosted remotely that multiplies its input by two.
        def forward(self, input):
            # [FUNCTION DOC] HELPER: Returns the input multiplied by 2.0.
            input = input * 2.0
            return input

    class MyLocalCompute(torch.nn.Module):
        # [CLASS DOC] HELPER: Module that forwards its input to a remote stage (held by reference)
        # through a synchronous RPC and returns the result.
        def __init__(self, next_stage):
            # [FUNCTION DOC] HELPER: Stores the reference to the remote stage.
            super().__init__()
            self.next_stage = next_stage

        def forward(self, input):
            # [FUNCTION DOC] HELPER: Calls the remote stage's forward with the input via sync RPC.
            return self.next_stage.rpc_sync().forward(input)

    @skip_if_lt_x_gpu(4)
    def test_dist_autograd_sync_streams(self):
        # [FUNCTION DOC] With a device map to the next worker and RPC initialized, ten times runs a
        # local doubling with ordinary backward and the same through a remote doubling module with
        # distributed backward (random loss scaling) on a large GPU tensor, and asserts the
        # gradients match, verifying stream synchronization. Shuts RPC down at the end.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)

        # The reverse of this device mapping should be used for the backward pass.
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        remote_compute = rpc.remote(dst, TensorPipeCudaDistAutogradTest.MyRemoteCompute)
        local_compute = TensorPipeCudaDistAutogradTest.MyLocalCompute(remote_compute)
        for _ in range(10):
            input = torch.rand([1000, 10000], device=self.rank, requires_grad=True)
            # Run local autograd
            result = input * 2.0
            r = random.random()
            loss = result.sum() * r
            loss.backward()

            # Run distributed autograd
            with dist_autograd.context() as context_id:
                result = local_compute(input)
                loss = result.sum() * r
                dist_autograd.backward(context_id, [loss])

                # Compare grads.
                grads = dist_autograd.get_gradients(context_id)
                self.assertEqual(input.grad, grads[input])

        rpc.shutdown()

    @skip_if_lt_x_gpu(4)
    def test_gradients_synchronizations(self):
        # [FUNCTION DOC] Sets device maps from this rank's GPU to every peer and initializes RPC.
        # On rank 0, builds a stack of linear layers hosted on the other ranks (wrapped modules),
        # runs a local model for reference, then a remote forward chain and distributed backward,
        # fetches each remote layer's gradients asynchronously, and asserts they equal the local
        # gradients. Other ranks only serve. Shuts RPC down at the end.
        options = self.rpc_backend_options
        for peer_rank in range(self.world_size):
            options.set_device_map(worker_name(peer_rank), {self.rank: peer_rank})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 0:
            # this is master
            layers = [nn.Linear(2000, 2000) for _ in range(self.world_size - 1)]
            local_layers = [l.to(0) for l in layers]
            remote_layers = [
                rpc.remote(
                    worker_name(rank), WrapperModule, args=(layers[rank - 1], rank)
                )
                for rank in range(1, self.world_size)
            ]

            x = torch.randn(5000, 2000).to(0)
            # local iteration
            local_model = nn.Sequential(*local_layers)
            local_model(x).sum().backward()

            # remote iteration
            with dist_autograd.context() as context_id:
                for remote_layer in remote_layers:
                    x = remote_layer.rpc_sync().forward(x)

                dist_autograd.backward(context_id, [x.sum()])

                futs = []
                for remote_layer in remote_layers:
                    futs.append(remote_layer.rpc_async().gradients(context_id))

                for i in range(len(futs)):
                    local_gradients = [p.grad for p in local_layers[i].parameters()]
                    for g1, g2 in zip(futs[i].wait(), local_gradients, strict=True):
                        self.assertEqual(g1, g2)

        rpc.shutdown()
