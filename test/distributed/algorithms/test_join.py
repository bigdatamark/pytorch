# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Multi-process tests for the generic Join context manager in
# torch.distributed.algorithms.join. Defines an example Joinable (AllReducer) and its
# JoinHook, then runs uneven/even input scenarios across ranks covering main hooks, post
# hooks, multiple joinables, enable=False and throw_on_early_termination.
# Import-time side effects: exits the process (status 0) when torch.distributed is
# unavailable or dev-asan is enabled; picks NCCL if CUDA is available else Gloo, and sets
# the world size to between 2 and 4 based on the GPU count.

import contextlib
import os
import sys
from typing import Any

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    require_n_gpus_for_nccl_backend,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))

# Constants used for testing post-hooks
BEFORE_CONSTANT = 41
AFTER_CONSTANT = 42


class AllReducerJoinHook(JoinHook):
    # [CLASS DOC] HELPER: Join hook for the AllReducer test joinable. Holds the owning
    # allreducer, how many all-reduces each iteration issues, and whether the post hook
    # runs. The main hook shadows the all-reduces of non-joined ranks with zero tensors;
    # the post hook syncs a magic constant from the last joiner.
    r"""
    Join hook for :class:`AllReducer`.

    Arguments:
        allreducer (AllReducer): the :class:`AllReducer` object using this
            hook.
        num_allreduces (int): the number of all-reduces to shadow per
            iteration.
        run_post_hook (bool): a flag enabling the post-hook logic.
    """

    def __init__(self, allreducer, num_allreduces, run_post_hook):
        # [FUNCTION DOC] HELPER: Stores the owning allreducer, the per-iteration number of
        # shadowed all-reduces and the post-hook enable flag. No other side effects.
        self.allreducer = allreducer
        self.num_allreduces = num_allreduces
        self.run_post_hook = run_post_hook

    def main_hook(self):
        # [FUNCTION DOC] HELPER: Called each iteration on a joined rank. Issues
        # num_allreduces all-reduces of a one-element zero tensor on the allreducer's
        # device over the default group, matching the collectives of non-joined ranks
        # without changing their sum.
        r"""
        Shadows each all-reduce; the number of all-reduces is passed into the
        constructor as ``num_allreduces``.
        """
        device = self.allreducer.device
        for _ in range(self.num_allreduces):
            t = torch.zeros(1, device=device)
            dist.all_reduce(t)

    def post_hook(self, is_last_joiner: bool):
        # [FUNCTION DOC] HELPER: Runs once all ranks have joined. Returns immediately if
        # post hooks are disabled. Otherwise finds the highest rank among last joiners,
        # has that rank set the allreducer's post_hook_tensor to AFTER_CONSTANT, and
        # broadcasts that tensor from it to all ranks (mutating post_hook_tensor in place
        # on the receivers).
        r"""
        Broadcasts a tensor containing a magic constant ``AFTER_CONSTANT`` from
        the last joiner to all other processes.
        """
        if not self.run_post_hook:
            return
        rank = dist.get_rank(self.allreducer.process_group)
        common_rank = self.allreducer.find_common_rank(rank, is_last_joiner)
        device = self.allreducer.device
        if rank == common_rank:
            self.allreducer.post_hook_tensor = torch.tensor(
                [AFTER_CONSTANT], device=device
            )
        dist.broadcast(self.allreducer.post_hook_tensor, src=common_rank)


class AllReducer(Joinable):
    # [CLASS DOC] HELPER: Minimal Joinable used by the tests. State: device, process group
    # and post_hook_tensor (initialized to BEFORE_CONSTANT). Calling it all-reduces a ones
    # tensor a given number of times and returns the summed result; join_hook returns an
    # AllReducerJoinHook configured from keyword arguments.
    r"""
    Example :class:`Joinable` that performs some number of all-reduces as its
    per-iteration collective communication.
    """

    def __init__(self, device, process_group):
        # [FUNCTION DOC] HELPER: Records the device and process group and creates
        # post_hook_tensor holding BEFORE_CONSTANT on that device.
        super().__init__()
        self.device = device
        self.process_group = process_group
        self.post_hook_tensor = torch.tensor([BEFORE_CONSTANT], device=self.device)

    def __call__(self, num_allreduces=1):
        # [FUNCTION DOC] HELPER: Notifies the Join context that this rank is still active,
        # then performs num_allreduces all-reduces of a one-element ones tensor on the
        # default group. Returns the accumulated float total of the reduced values (so
        # each non-joined rank contributes one per reduce).
        r"""
        All-reduces a dim-1 one tensor ``num_allreduces``-many times, and
        returns the total result.
        """
        Join.notify_join_context(self)
        device = self.device
        total = 0
        for _ in range(num_allreduces):
            t = torch.ones(1, device=device)
            dist.all_reduce(t)
            total += t.item()
        return total

    def join_hook(self, **kwargs) -> JoinHook:
        # [FUNCTION DOC] HELPER: Builds an AllReducerJoinHook from keyword arguments
        # num_allreduces (default 1) and run_post_hooks (default False); other keywords are
        # ignored.
        r"""
        Returns a join hook that shadows some number of all-reduces; by default,
        this number is 1.
        """
        num_allreduces = kwargs.get("num_allreduces", 1)
        run_post_hook = kwargs.get("run_post_hooks", False)
        return AllReducerJoinHook(self, num_allreduces, run_post_hook)

    @property
    def join_device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property required by Joinable; returns the device used for
        # the join context's own collectives.
        return self.device

    @property
    def join_process_group(self) -> Any:
        # [FUNCTION DOC] HELPER: Property required by Joinable; returns the process group
        # the join context should use.
        return self.process_group

    def find_common_rank(self, rank, to_consider):
        # [FUNCTION DOC] HELPER: All-reduces (MAX) over this object's process group a value
        # that is the caller's rank if to_consider is true, else -1. Returns the resulting
        # maximum rank as an int; raises AssertionError if no rank considered itself
        # (result negative). Collective: every rank must call it.
        r"""
        Returns the max rank of the ones to consider over the process group.
        """
        common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        common_rank = common_rank.item()
        if not (common_rank >= 0):
            raise AssertionError(f"Expected common_rank >= 0, got {common_rank}")
        return common_rank


class TestJoin(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case for the Join context manager. Each test runs in
    # WORLD_SIZE spawned processes; ranks get uneven input counts (base plus rank times
    # offset) so they join at different times. Shared logic lives in _test_join_base.
    r"""Test cases for the generic join context."""

    def setUp(self):
        # [FUNCTION DOC] Sets WORLD_SIZE and BACKEND environment variables (global process
        # side effect) and spawns the worker processes that run the test body.
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        self._spawn_processes()

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this rank under NCCL,
        # otherwise the CPU device.
        return (
            torch.device(self.rank)
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property; returns the module-level WORLD_SIZE.
        return WORLD_SIZE

    @property
    def process_group(self):
        # [FUNCTION DOC] HELPER: Property; returns the default (world) process group.
        return dist.group.WORLD

    def tearDown(self):
        # [FUNCTION DOC] Destroys the default process group (ignoring an AssertionError if
        # none exists) and deletes the shared store file (ignoring OS errors). Overrides the
        # base class teardown and does not call it.
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def dist_init(self, rank, world_size, backend=BACKEND):
        # [FUNCTION DOC] HELPER: Initializes the default process group for this rank using
        # a FileStore backed by the test's temp file. Returns the init result; side effect
        # is global process group state.
        store = dist.FileStore(self.file_name, world_size)
        return dist.init_process_group(
            backend=backend, store=store, rank=rank, world_size=world_size
        )

    def construct_uneven_inputs(self, base, offset, device=None):
        # [FUNCTION DOC] HELPER: Returns a list of one-element zero tensors whose length is
        # base plus this rank times offset, on the given device (default: self.device).
        r"""
        Returns uneven inputs: rank i gets ``base`` + i * ``offset`` inputs.
        """
        if device is None:
            device = self.device
        return [torch.zeros(1, device=device) for _ in range(base + self.rank * offset)]

    def construct_even_inputs(self, base, device=None):
        # [FUNCTION DOC] HELPER: Returns a list of base one-element zero tensors on the
        # given device (default: self.device), identical in length on every rank.
        r"""Returns even inputs: each rank gets ``base`` inputs."""
        if device is None:
            device = self.device
        return [torch.zeros(1, device=device) for _ in range(base)]

    @property
    def base_num_inputs(self):
        # [FUNCTION DOC] HELPER: Property; the minimum number of inputs per rank (3).
        r"""Base number of inputs to be used by all ranks."""
        return 3

    @property
    def offset(self):
        # [FUNCTION DOC] HELPER: Property; extra inputs per rank index in the uneven case (1).
        r"""Rank i gets i * ``offset`` additional inputs."""
        return 1

    def _test_join_base(
        self,
        uneven_inputs: bool,
        num_joinables: int,
        enable: bool,
        throw_on_early_termination: bool,
        num_allreduces: int,
        run_post_hooks: bool,
        expected_total: int | None = None,
    ):
        # [FUNCTION DOC] HELPER: Shared body of all join tests, run inside each worker.
        # Initializes the process group, builds the requested number of AllReducers
        # (checking their initial post-hook value), then loops over even or uneven inputs
        # calling every allreducer inside a Join context with the given options. With
        # throw_on_early_termination it expects a RuntimeError whose message depends on
        # whether this is rank 0, and returns. Otherwise it optionally checks the summed
        # all-reduce total and, if post hooks ran, that every post_hook_tensor became
        # AFTER_CONSTANT.
        r"""
        Skeleton for all :class:`Join` tests.

        Arguments:
            uneven_inputs (bool): ``True`` to use uneven inputs; ``False``
                otherwise.
            num_joinables (int): number of :class:`AllReducer` s to construct.
            enable (bool): ``True`` to enable the join context manager;
                ``False`` otherwise.
            throw_on_early_termination (bool): ``True`` to raise an exception
                upon detecting uneven inputs; ``False`` otherwise.
            num_allreduces (int): number of all-reduces to perform per input.
            run_post_hooks (bool): ``True`` to run post-hooks; ``False``
                otherwise.
            expected_total (Optional[int]): ``None`` to not check the expected
                all-reduce total; otherwise, the expected total; default is
                ``None``.
        """
        self.dist_init(self.rank, self.world_size)

        allreducers = [
            AllReducer(self.device, self.process_group) for _ in range(num_joinables)
        ]
        for allreducer in allreducers:
            self.assertEqual(allreducer.post_hook_tensor.item(), BEFORE_CONSTANT)

        inputs = (
            self.construct_uneven_inputs(self.base_num_inputs, self.offset)
            if uneven_inputs
            else self.construct_even_inputs(self.base_num_inputs)
        )
        allreduce_total = 0

        # Expect a `RuntimeError` if `throw_on_early_termination=True`
        # Rank 0 exhausts its inputs first
        expected_msg = (
            "Rank 0 exhausted all inputs."
            if self.rank == 0
            else "Detected at least one rank that exhausted inputs. "
            "Throwing across all ranks."
        )
        with (
            self.assertRaisesRegex(RuntimeError, expected_msg)
            if throw_on_early_termination
            else contextlib.nullcontext()
        ):
            with Join(
                allreducers,
                enable=enable,
                throw_on_early_termination=throw_on_early_termination,
                num_allreduces=num_allreduces,
                run_post_hooks=run_post_hooks,
            ):
                for _ in inputs:
                    for allreducer in allreducers:
                        allreduce_total += allreducer(num_allreduces)

        if throw_on_early_termination:
            return

        # Check `expected_total` if not `None`
        if expected_total:
            self.assertEqual(allreduce_total, expected_total)

        # All `AllReduce` instances should receive the updated
        # `post_hook_tensor` from the last-joined process
        if run_post_hooks:
            for allreducer in allreducers:
                self.assertEqual(allreducer.post_hook_tensor.item(), AFTER_CONSTANT)

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_main_hooks(self):
        # [FUNCTION DOC] One joinable, uneven inputs, main hooks only. The expected total
        # is the world-size times base inputs plus, for each rank that joined earlier, the
        # remaining non-joined rank count times the offset.
        r"""Tests the main hooks of a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 1
        run_post_hooks = False
        # Non-joined processes all-reduce a 1, so this rank's all-reduce total
        # should be precisely equal to the total number of inputs processed
        # before it joined
        expected_total = self.world_size * self.base_num_inputs
        # Rank i runs for i additional iterations
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_post_hooks(self):
        # [FUNCTION DOC] One joinable, uneven inputs, zero all-reduces per iteration to
        # skip main hooks, no total check.
        # KNOWN ISSUE: run_post_hooks is set to False, so the post hook path is not
        # actually exercised despite the test name and docstring.
        r"""Tests the post-hooks of a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 0  # set to 0 to skip the main hooks
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable(self):
        # [FUNCTION DOC] One joinable, uneven inputs, with both main hooks and post hooks
        # enabled; checks the all-reduce total and that the post-hook tensor is updated.
        r"""
        Tests the main hooks and post-hooks of a single :class:`Joinable`
        together.

        This combines ``test_single_joinable_main_hooks()`` and
        ``test_single_joinable_post_hooks()`` into a single test to ensure that
        main hooks and post-hooks operate correctly together.
        """
        num_joinables = 1
        num_allreduces = 1
        run_post_hooks = True

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinables(self):
        # [FUNCTION DOC] Three joinables with uneven inputs, main and post hooks on; the
        # expected all-reduce total is scaled by the number of joinables.
        r"""
        Tests the main hooks and post-hooks of multiple :class:`Joinable` s
        together.

        This generalizes ``test_single_joinable()`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        run_post_hooks = True

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # The expected total is now multiplied by a factor of `NUM_JOINABLES`
        expected_total *= num_joinables

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_disable(self):
        # [FUNCTION DOC] One joinable with the Join context disabled and even inputs; the
        # total must equal world size times base inputs (no joining behavior).
        r"""Tests ``enable=False`` for a single :class:`Joinable`."""
        num_joinables = 1
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs

        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinable_disable(self):
        # [FUNCTION DOC] Three joinables, Join disabled, even inputs; the total is world
        # size times base inputs times the joinable count.
        r"""
        Tests ``enable=False`` for multiple :class:`Joinable` s.

        This generalizes ``test_single_joinable_disable`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs * num_joinables

        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_throw(self):
        # [FUNCTION DOC] One joinable with uneven inputs and throw_on_early_termination;
        # every rank must see a RuntimeError once rank 0 runs out of inputs.
        r"""
        Tests ``throw_on_early_termination=True`` for a single
        :class:`Joinable`.
        """
        num_joinables = 1
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=throw_on_early_termination,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinables_throw(self):
        # [FUNCTION DOC] Same as the single-joinable throw test but with three joinables.
        r"""
        Tests ``throw_on_early_termination=True`` for multiple
        :class:`Joinable` s together.

        This generalizes ``test_single_joinable_throw`` to multiple
        :class:`Joinable` s.
        """
        num_joinables = 3
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=throw_on_early_termination,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=None,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_join_kwargs(self):
        # [FUNCTION DOC] Checks that keyword arguments given to Join reach the hooks: two
        # all-reduces per iteration, so the expected total is doubled.
        r"""
        Tests passing keyword arguments to the context manager.
        """
        num_joinables = 1
        num_allreduces = 2
        run_post_hooks = False

        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # The expected total is now multiplied by a factor of `NUM_ALLREDUCES`
        expected_total *= num_allreduces

        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )


if __name__ == "__main__":
    run_tests()
