# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for pickle-based object collectives (all_gather_object, gather_object,
# send/recv_object_list, broadcast_object_list, scatter_object_list), with weights_only on
# and off, plus subgroup variants. Defines the with_comms decorator and the
# TestObjectCollectives class. Import-time side effects: exits with status 0 if
# torch.distributed is unavailable or under dev-asan; computes module globals device_type
# (current accelerator type or cpu) and device_count.
import sys
from functools import partial, wraps

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistributedTestBase, TEST_SKIPS
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    TEST_WITH_DEV_DBG_ASAN,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)
device_count = torch.accelerator.device_count()


def with_comms(func=None):
    # [FUNCTION DOC] HELPER: Decorator usable with or without parentheses. With no function
    # it returns a partial of itself; otherwise returns a wrapper that creates the process
    # group for the test and destroys the default group afterwards (even on failure).
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: On an accelerator with fewer devices than the world size,
        # exits the process with the matching multi-gpu skip exit code. Otherwise stores
        # a new process group from create_pg on the test (self.pg), runs the test body and
        # returns its result, and always destroys the process group.
        if device_type != "cpu" and device_count < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=device_type)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class TestObjectCollectives(DistributedTestBase):
    # [CLASS DOC] Multi-process tests of object collectives, each run under with_comms so a
    # process group exists; the rank is available as self.rank. Parametrized over
    # weights_only (restricted unpickling) via instantiate_parametrized_tests below.
    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_all_gather_object(self, weights_only):
        # [FUNCTION DOC] Each rank contributes its rank as the object; asserts the gathered
        # list on every rank holds ranks in order 0..world_size-1.
        output = [None] * dist.get_world_size()
        dist.all_gather_object(
            object_list=output, obj=self.rank, weights_only=weights_only
        )

        for i, v in enumerate(output):
            self.assertEqual(i, v, lambda msg: f"{msg}\nrank: {self.rank}")

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_gather_object(self, weights_only):
        # [FUNCTION DOC] Gathers each rank's id to rank 0 (only rank 0 supplies an output
        # list); rank 0 asserts the list holds ranks in order.
        output = [None] * dist.get_world_size() if self.rank == 0 else None
        dist.gather_object(
            obj=self.rank, object_gather_list=output, weights_only=weights_only
        )

        if self.rank == 0:
            for i, v in enumerate(output):
                self.assertEqual(i, v, lambda msg: f"{msg}\nrank: {self.rank}")

    @skipIfHpu
    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_send_recv_object_list(self, weights_only):
        # [FUNCTION DOC] Rank 0 sends a list filled with 99 to rank 1, which receives into
        # its own list. Ranks 0 and 1 assert the first element is 99; any other rank
        # asserts it remains None. Skipped on HPU.
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        if self.rank == 0:
            dist.send_object_list(object_list, 1, weights_only=weights_only)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, weights_only=weights_only)

        if self.rank < 2:
            self.assertEqual(99, object_list[0])
        else:
            self.assertEqual(None, object_list[0])

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_broadcast_object_list(self, weights_only):
        # [FUNCTION DOC] Rank 0 holds 99 in every list slot, others hold None; after
        # broadcast from the default source (rank 0) every rank asserts the first element
        # is 99. The device argument is not covered (see the TODO).
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        # TODO test with broadcast_object_list's device argument
        dist.broadcast_object_list(object_list=object_list, weights_only=weights_only)

        self.assertEqual(99, object_list[0])

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_scatter_object_list(self, weights_only):
        # [FUNCTION DOC] Rank 0 provides the list of ranks 0..world_size-1 (others provide
        # none); each rank receives one element in a one-slot output list and asserts it
        # equals its own rank.
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        output_list = [None]
        dist.scatter_object_list(
            scatter_object_output_list=output_list,
            scatter_object_input_list=input_list,
            weights_only=weights_only,
        )

        self.assertEqual(self.rank, output_list[0])

    @with_comms()
    def test_weights_only_rejects_unsafe_object(self):
        # [FUNCTION DOC] Gathers a function object (the with_comms decorator) with
        # weights_only enabled and asserts that some exception is raised, since arbitrary
        # callables are refused by restricted unpickling.
        # An arbitrary function is not deserializable under weights_only=True
        output = [None] * dist.get_world_size()
        with self.assertRaises(Exception):
            dist.all_gather_object(
                object_list=output, obj=with_comms, weights_only=True
            )

    # Test Object Collectives With Sub Pg

    def setup_sub_pg(self):
        # [FUNCTION DOC] HELPER: Pairs ranks (0,1), (2,3), ... by rounding this rank down to
        # an even base. Creates a group over the pair with local synchronization (only
        # members call). Returns the tuple (own rank, pair rank list, group).
        rank = dist.get_rank()
        base_rank = rank - (rank % 2)
        ranks = [base_rank, base_rank + 1]
        my_pg = dist.new_group(ranks, use_local_synchronization=True)
        return rank, ranks, my_pg

    @with_comms()
    def test_subpg_scatter_object(self):
        # [FUNCTION DOC] In the rank pair subgroup, the first rank scatters the pair's rank
        # list; each member asserts it received its own rank.
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        dist.scatter_object_list(out_list, ranks, src=ranks[0], group=my_pg)
        self.assertEqual(rank, out_list[0])

    @with_comms()
    def test_subpg_all_gather_object(self):
        # [FUNCTION DOC] All-gathers each member's rank within the pair subgroup and asserts
        # the result equals the pair's rank list.
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks)
        dist.all_gather_object(out_list, rank, group=my_pg)
        self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_gather_object(self):
        # [FUNCTION DOC] Gathers ranks to the pair's first rank (destination given as a
        # global rank); that rank asserts the result equals the pair's rank list.
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks) if rank == ranks[0] else None
        dist.gather_object(rank, out_list, dst=ranks[0], group=my_pg)
        if rank == ranks[0]:
            self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_broadcast_object(self):
        # [FUNCTION DOC] The pair's first rank puts its rank in a one-slot list and
        # broadcasts within the pair subgroup; both members assert the value equals that
        # first rank.
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        if rank == ranks[0]:
            out_list[0] = rank
        dist.broadcast_object_list(out_list, src=ranks[0], group=my_pg)
        self.assertEqual(ranks[0], out_list[0])


instantiate_parametrized_tests(TestObjectCollectives)


if __name__ == "__main__":
    run_tests()
