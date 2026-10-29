# mypy: allow-untyped-defs
# [MODULE DOC]
# Tests distributed autograd APIs (get_gradients, backward) when called from TorchScript, and
# context propagation across JIT fork and RPC threads. Defines scripted helper functions that
# run a local add, an add through rpc_async, and an add through a JIT fork. The scripted
# functions are compiled at import time.


import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import rpc_async
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


@torch.jit.script
def local_add(t1, t2):
    # [FUNCTION DOC] HELPER: Scripted function returning the elementwise sum of two tensors.
    return torch.add(t1, t2)


@torch.jit.script
def remote_add(t1, t2, dst: str):
    # [FUNCTION DOC] HELPER: Scripted function that runs the local add on worker dst through
    # an async RPC and blocks for the resulting tensor.
    return rpc_async(dst, local_add, (t1, t2)).wait()


@torch.jit.script
def fork_add(t1, t2, dst: str):
    # [FUNCTION DOC] HELPER: Scripted function that forks the remote add into a JIT task and
    # waits for it, so the RPC is issued from a different thread than the caller.
    fut = torch.jit._fork(remote_add, t1, t2, dst)
    return torch.jit._wait(fut)


class JitDistAutogradTest(RpcAgentTestFixture):
    # [CLASS DOC]
    # Test case (run under a backend-specific fixture mixin) checking that distributed autograd
    # works from TorchScript. Stateless; every test is wrapped by dist_init so RPC is
    # initialized per test.
    @dist_init
    def test_get_gradients(self):
        # [FUNCTION DOC]
        # Scripts a function wrapping get_gradients, checks the graph contains that op, then
        # runs a local add under a dist autograd context, calls backward and verifies the
        # scripted call returns gradients for both inputs, each equal to ones.
        @torch.jit.script
        def dist_get_gradients(context_id: int) -> dict[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Scripted wrapper returning the gradient mapping of a context.
            return dist_autograd.get_gradients(context_id)

        FileCheck().check("get_gradients").run(str(dist_get_gradients.graph))
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)

            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_get_gradients(context_id)

            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_dist_backward(self):
        # [FUNCTION DOC]
        # Runs only on rank 0. Scripts a function that calls distributed backward, checks the
        # graph contains the op, then computes a loss from an RPC add on the next worker and
        # runs the scripted backward on it. Passes if no error is raised.
        if self.rank != 0:
            return

        @torch.jit.script
        def dist_backward_script(context_id: int, loss: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted wrapper running distributed backward on the loss.
            dist_autograd.backward(context_id, [loss])

        FileCheck().check("dist_backward").run(str(dist_backward_script.graph))
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            loss = rpc.rpc_sync(dst_worker_name, torch.add, args=(t1, t2)).sum()
            dist_backward_script(context_id, loss)

    @dist_init
    def test_jit_fork_within_context(self):
        # [FUNCTION DOC]
        # Within a dist autograd context, calls the forked remote add on the next worker, runs
        # backward on the summed result and checks that gradients exist for both inputs, i.e. the
        # context survives the JIT fork thread.
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            res = fork_add(t1, t2, dst_worker_name)
            loss = res.sum()
            dist_autograd.backward(context_id, [loss])

            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)

    @dist_init
    def test_restore_context_after_swtich_to_jit_thread(self):
        # [FUNCTION DOC]
        # Runs only on rank 0. Scripted code does two sequential async RPCs; after the first
        # wait the script continues on a new JIT thread, and the test checks the autograd context
        # is restored so backward yields equal gradients for both inputs.
        # KNOWN ISSUE: the method name misspells "switch"; the in-script comment says the context
        # is lost, while the test expects it to be restored.
        if self.rank != 0:
            return

        @torch.jit.script
        def forward_script(
            context_id: int, dst_worker_name: str, t1: Tensor, t2: Tensor
        ) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function issuing two sequential async RPC adds
            # (doubling each tensor) and returning the sum of each result as two losses.
            res1_fut = rpc.rpc_async(dst_worker_name, local_add, (t1, t1))
            res1 = res1_fut.wait()  # After this, the script runs in a new JIT thread.
            loss1 = res1.sum()

            # SendRpcBackward is not attached, since DistAutogradContext is lost here.
            res2_fut = rpc.rpc_async(dst_worker_name, local_add, (t2, t2))
            res2 = res2_fut.wait()
            loss2 = res2.sum()

            return loss1, loss2

        with dist_autograd.context() as context_id:
            t1 = torch.ones((2, 3), requires_grad=True)
            t2 = torch.ones((2, 3), requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            loss0, loss1 = forward_script(context_id, dst_worker_name, t1, t2)
            dist_autograd.backward(context_id, [loss0, loss1])
            grad0, grad1 = dist_autograd.get_gradients(context_id)
            self.assertEqual(grad0, grad1)
