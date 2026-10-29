# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Point-to-point tests (send/recv, isend/irecv, batch_isend_irecv, event
# ordering, dropped work lifetime) run once per backend in C10D_BACKENDS. Defines the
# abstract AbstractP2PTest mixin and instantiates concrete per-backend classes with
# instantiate_backend_tests. Import-time side effects: exits the process with status 0 if
# torch.distributed is unavailable; creates the generated test classes in module globals.
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests


COUNTS = (0, 4)


class AbstractP2PTest(C10dBackendTest):
    # [CLASS DOC] Backend-agnostic ring-style P2P tests. Each rank sends to the next rank
    # and receives from the previous one (modulo world size), with even ranks issuing the
    # send first and odd ranks the receive first to avoid deadlock. Relies on backend
    # attributes (dtypes, supports_dropped_p2p_work) injected by instantiate_backend_tests.
    def _peers(self):
        # [FUNCTION DOC] HELPER: Returns a pair (next_rank, previous_rank) for this rank in
        # a ring over the world size.
        next_rank = (self.rank + 1) % self.world_size
        previous_rank = (self.rank - 1) % self.world_size
        return next_rank, previous_rank

    def _tensor(self, count, dtype, value):
        # [FUNCTION DOC] HELPER: Builds a one-dimensional tensor of length count on the
        # test device, filled with value as float32 and then cast to dtype (so any dtype,
        # including bool, can be produced). Returns the tensor.
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _test_send_recv(self, count, dtype):
        # [FUNCTION DOC] HELPER: One blocking ring exchange. Sends a tensor filled with
        # this rank to the next rank and receives into an empty tensor from the previous
        # rank (even ranks send first, odd ranks receive first), then asserts the received
        # values equal the previous rank's id. Count may be zero (empty tensors).
        next_rank, previous_rank = self._peers()
        send = self._tensor(count, dtype, self.rank)
        recv = torch.empty_like(send)
        if self.rank % 2 == 0:
            dist.send(send, next_rank)
            dist.recv(recv, previous_rank)
        else:
            dist.recv(recv, previous_rank)
            dist.send(send, next_rank)
        self.assertEqual(recv, self._tensor(count, dtype, previous_rank))

    def test_send_recv(self):
        # [FUNCTION DOC] Initializes the process group, then runs the blocking ring
        # exchange for every count in COUNTS and every dtype of the backend, each as a
        # subtest.
        self._init_pg()
        for count in COUNTS:
            for dtype in self.dtypes:
                with self.subTest(count=count, dtype=dtype):
                    self._test_send_recv(count, dtype)

    def test_isend_irecv(self):
        # [FUNCTION DOC] Same ring exchange as the blocking test but using the
        # asynchronous isend and irecv calls (order depends on rank parity); waits on both
        # work handles and checks the received values, for each count and dtype subtest.
        self._init_pg()
        next_rank, previous_rank = self._peers()
        for count in COUNTS:
            for dtype in self.dtypes:
                with self.subTest(count=count, dtype=dtype):
                    send = self._tensor(count, dtype, self.rank)
                    recv = torch.empty_like(send)
                    if self.rank % 2 == 0:
                        works = (
                            dist.isend(send, next_rank),
                            dist.irecv(recv, previous_rank),
                        )
                    else:
                        works = (
                            dist.irecv(recv, previous_rank),
                            dist.isend(send, next_rank),
                        )
                    for work in works:
                        work.wait()
                    self.assertEqual(recv, self._tensor(count, dtype, previous_rank))

    def _test_batch_isend_irecv(self, dtype, recv_first, num_ops):
        # [FUNCTION DOC] HELPER: Builds num_ops send/recv P2POp pairs (send to next rank,
        # receive from previous rank), each send value offset by 100 times its index. If
        # recv_first is set each pair is reversed so the receive precedes the send. Issues
        # them with batch_isend_irecv, waits on all returned works, and asserts each
        # received value equals the previous rank plus the index offset.
        next_rank, previous_rank = self._peers()
        sends = [self._tensor(1, dtype, self.rank + i * 100) for i in range(num_ops)]
        recvs = [torch.empty_like(sends[0]) for _ in range(num_ops)]
        ops = []
        for send, recv in zip(sends, recvs):
            pair = [
                dist.P2POp(dist.isend, send, next_rank),
                dist.P2POp(dist.irecv, recv, previous_rank),
            ]
            ops.extend(reversed(pair) if recv_first else pair)
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        for i, recv in enumerate(recvs):
            self.assertEqual(recv, self._tensor(1, dtype, previous_rank + i * 100))

    def test_batch_isend_irecv(self):
        # [FUNCTION DOC] Initializes the process group and runs the batched P2P helper over
        # every dtype, both op orderings (recv_first false/true) and 1 or 2 ops per batch.
        self._init_pg()
        for dtype in self.dtypes:
            for recv_first in (False, True):
                for num_ops in (1, 2):
                    with self.subTest(
                        dtype=dtype,
                        recv_first=recv_first,
                        num_ops=num_ops,
                    ):
                        self._test_batch_isend_irecv(dtype, recv_first, num_ops)

    def test_send_recv_event_ordering(self):
        # [FUNCTION DOC] Stress check (see the existing comment below for its origin): for 50
        # iterations with distinct values, does an asynchronous ring exchange of one
        # million float32 elements, waits on both works, and immediately asserts the
        # received data matches the previous rank's value for that iteration, which would
        # fail if the work completed before the transfer.
        # Regression test for the single-op P2P group-wrap fix (TorchComms
        # D109625550): the nccl2 sendImpl/recvImpl now wrap ncclSend/ncclRecv in
        # ncclGroupStart/End so the kernel is enqueued before the end CUDA event
        # is recorded. Without it a non-blocking comm could complete the event
        # ahead of the transfer and wait() would return stale/partial data. Loop
        # with distinct per-iteration values and read back immediately after
        # wait(): an early completion surfaces as a value mismatch. Generic P2P
        # ordering check, so it runs across every backend.
        self._init_pg()
        next_rank, previous_rank = self._peers()
        numel = 1024 * 1024
        for i in range(50):
            send = self._tensor(numel, torch.float32, self.rank + i)
            recv = torch.empty_like(send)
            if self.rank % 2 == 0:
                send_work = dist.isend(send, next_rank)
                recv_work = dist.irecv(recv, previous_rank)
            else:
                recv_work = dist.irecv(recv, previous_rank)
                send_work = dist.isend(send, next_rank)
            send_work.wait()
            recv_work.wait()
            self.assertEqual(
                recv, self._tensor(numel, torch.float32, previous_rank + i)
            )

    def test_async_work_lifetime(self):
        # [FUNCTION DOC] Skipped unless the backend retains dropped P2P work. Issues an
        # asynchronous ring exchange of four floats, deletes the send work handle without
        # waiting, waits only on the receive work, and asserts the data arrived correctly.
        if not self.supports_dropped_p2p_work:
            self.skipTest(f"{self.backend_name} does not retain dropped P2P work")
        self._init_pg()
        next_rank, previous_rank = self._peers()
        send = self._tensor(4, torch.float32, self.rank)
        recv = torch.empty_like(send)
        if self.rank % 2 == 0:
            send_work = dist.isend(send, next_rank)
            recv_work = dist.irecv(recv, previous_rank)
        else:
            recv_work = dist.irecv(recv, previous_rank)
            send_work = dist.isend(send, next_rank)
        del send_work
        recv_work.wait()
        self.assertEqual(recv, self._tensor(4, torch.float32, previous_rank))


instantiate_backend_tests(globals(), "P2P", AbstractP2PTest, C10D_BACKENDS)


if __name__ == "__main__":
    run_tests()
