# mypy: allow-untyped-defs

# [MODULE DOC]
# Multi-process RPC test of a batch-updating parameter server. Defines a server object that
# accumulates gradients from several trainers and releases an updated model through a future once
# all trainers of the batch have reported, a trainer that computes gradients and sends them, a
# driver that launches trainers, and the test case class. Module-level constants set the batch
# size, feature sizes and number of batches. No import-time side effects beyond imports.

# If you need to modify this file to make this test pass, please also apply same edits accordingly to
# https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py
# and https://pytorch.org/tutorials/intermediate/rpc_async_execution.html#batch-updating-parameter-server

import threading
from datetime import datetime
from time import perf_counter

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


batch_size = 20
in_features = 100
out_features = 30
num_batches = 4


def timed_log(text):
    # [FUNCTION DOC] HELPER: Prints the given text prefixed with the current wall-clock time
    # (hours:minutes:seconds) to standard output.
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer:
    # [CLASS DOC] Parameter server that holds a linear model and applies one optimizer step per
    # batch of trainer updates. State: the model, a lock, a future that resolves to the model when
    # the current batch completes, the number of updates required per batch, the count received so
    # far, and an SGD optimizer. Invariant: the count is reset to zero and a fresh future is
    # created each time a batch is applied. Key method: update_and_fetch_model.
    def __init__(self, batch_update_size):
        # [FUNCTION DOC] Creates the linear model, lock, pending future, SGD optimizer (learning
        # rate 0.001, momentum 0.9), and sets every parameter gradient to zeros.
        # Arg: batch_update_size, the number of trainer updates that make up one batch.
        self.model = nn.Linear(in_features, out_features)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        # [FUNCTION DOC] Returns the server's current model object (not a copy).
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        # [FUNCTION DOC] Async-execution static method run on the server. Takes a remote reference
        # to the server and a list of gradients (one per model parameter), adds them into the
        # parameter gradients, and counts the update under the lock. When the count reaches the
        # batch size it averages the gradients, steps and zeroes the optimizer, resolves the
        # pending future with the model and installs a new future. Returns the future that was
        # current when this call arrived, so every trainer in a batch receives the updated model.
        # Side effects: mutates model gradients and weights, logs progress.
        self = ps_rref.local_value()
        for p, g in zip(self.model.parameters(), grads, strict=True):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
        with self.lock:
            timed_log(
                f"PS got {self.curr_update_size}/{self.batch_update_size} updates"
            )
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer:
    # [CLASS DOC] Trainer that repeatedly computes gradients on random batches and sends them to
    # the parameter server. State: a remote reference to the server and an L1 loss function.
    def __init__(self, ps_rref):
        # [FUNCTION DOC] Stores the server reference and creates the L1 loss.
        self.ps_rref = ps_rref
        self.loss_fn = nn.L1Loss()

    def get_next_batch(self):
        # [FUNCTION DOC] Generator yielding num_batches pairs of random input features and
        # all-zero labels, sized by the module constants.
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, in_features)
            labels = torch.zeros(batch_size, out_features)
            yield inputs, labels

    def train(self):
        # [FUNCTION DOC] Fetches the model from the server, then for each batch runs the forward
        # and backward pass, sends the parameter gradients (moved to CPU) to the server through a
        # synchronous RPC and replaces the local model with the returned updated model.
        # Blocks until the server completes each batch. Logs progress.
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model()
        for inputs, labels in self.get_next_batch():
            timed_log(f"{name} processing one batch")
            self.loss_fn(m(inputs), labels).backward()
            timed_log(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            )
            timed_log(f"{name} got updated model")


def run_trainer(ps_rref):
    # [FUNCTION DOC] Entry point executed on a trainer worker: builds a Trainer for the given
    # server reference and runs its training loop to completion.
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    # [FUNCTION DOC] Runs on the server worker. Takes a list of trainer worker names, creates a
    # server (batch size equal to the number of trainers) wrapped in a remote reference, launches
    # the trainer entry point on every trainer asynchronously, waits for all to finish, and logs
    # the elapsed training time.
    timed_log("Start training")
    start = perf_counter()
    ps_rref = rpc.RRef(BatchUpdateParameterServer(len(trainers)))
    futs = [
        rpc.rpc_async(trainer, run_trainer, args=(ps_rref,)) for trainer in trainers
    ]

    torch.futures.wait_all(futs)
    stop = perf_counter()
    timed_log("Finish training")
    timed_log(f"Time spent training: {stop - start}s")


class ParameterServerTest(RpcAgentTestFixture):
    # [CLASS DOC] Multi-process test case for the batch-updating parameter server example.
    @dist_init(setup_rpc=False)
    def test_batch_updating_parameter_server(self):
        # [FUNCTION DOC] Every rank initializes RPC itself (the decorator is told not to). Rank 0
        # then acts as the server and runs training against all other ranks as trainers; all ranks
        # finally shut RPC down. Passes if no exception occurs. Both branches of the rank check
        # perform the same initialization.
        if self.rank != 0:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        else:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            run_ps([f"{worker_name(r)}" for r in range(1, self.world_size)])

        rpc.shutdown()
