# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests of FSDP collective communication behavior.
# Defines TestCommunication, which counts all-gather and reduce-scatter calls (via mocked
# torch.distributed single-tensor collectives) per iteration for FULL_SHARD / SHARD_GRAD_OP,
# with and without no_sync(), and TestExplicitUnshard, which checks the private
# _unshard(async_op=...) API overlaps with other async collectives and works after backward.
# Import-time side effects: exits the process (status 0) if torch.distributed is unavailable
# or under dev-asan; computes the device type; instantiates device-generic test classes for
# cuda/hpu/xpu at module level. Running the file calls run_tests().
import sys
from contextlib import nullcontext
from enum import auto, Enum
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch._utils import _get_device_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    get_devtype,
    MLP,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class PassType(Enum):
    # [CLASS DOC] HELPER: Enum naming the two phases of an iteration (FWD, BWD) whose
    # all-gather counts are computed separately; iteration order is forward then backward.
    __order__ = "FWD BWD"
    FWD = auto()
    BWD = auto()


class TestCommunication(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case that verifies the number of all-gathers and
    # reduce-scatters FSDP issues per iteration against analytically derived reference counts.
    # Stateless apart from the inherited process group; helper methods compute the references.
    """Tests ``FullyShardedDataParallel``'s collective communication usage."""

    def _init_model(
        self,
        device,
        nested_model: bool,
        sharding_strategy: ShardingStrategy,
    ):
        # [FUNCTION DOC] HELPER: Builds the FSDP model under test. With nested_model, wraps a
        # recursively wrapped NestedWrappedModule in an outer FSDP; otherwise builds a
        # recursively wrapped TransformerWithSharedParams. The device argument is unused (the
        # module-level device_type is used). Returns the FSDP model.
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "device_id": device_type.type,
        }
        if nested_model:
            model = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_AFTER,
                fsdp_kwargs,
            )
            fsdp_model: FSDP = FSDP(
                model,
                self.process_group,
                **fsdp_kwargs,
            )
        else:
            fsdp_model: FSDP = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                fsdp_kwargs,
            )
        return fsdp_model

    def _run_iter(self, fsdp_model, batch, use_no_sync: bool):
        # [FUNCTION DOC] HELPER: Runs one forward, loss computation and backward on the model,
        # inside no_sync() if use_no_sync else a null context. Accumulates gradients; returns
        # nothing.
        """Runs an iteration inside or outside the ``no_sync()`` context."""
        context = fsdp_model.no_sync() if use_no_sync else nullcontext()
        with context:
            output = fsdp_model(*batch)
            loss = fsdp_model.module.get_loss(batch, output)
            loss.backward()

    def _get_ref_num_reduce_scatters(
        self,
        num_fsdp: int,
        in_no_sync: bool,
    ) -> int:
        # [FUNCTION DOC] HELPER: Expected reduce-scatter count for one iteration: one per FSDP
        # instance that manages parameters, or zero when the iteration ran in no_sync().
        """Returns the reference number of reduce-scatters for an iteration
        in the ``no_sync()`` context."""
        return num_fsdp if not in_no_sync else 0

    def _get_ref_num_all_gathers(
        self,
        num_fsdp: int,
        sharding_strategy: ShardingStrategy | None,
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ) -> int:
        # [FUNCTION DOC] HELPER: Expected all-gather count for a whole iteration: the sum of
        # the per-pass reference counts over the forward and backward passes.
        """Returns the reference number of all-gathers in an iteration, summing
        over the forward and backward passes."""
        return sum(
            self._get_ref_num_all_gathers_in_pass(
                num_fsdp,
                sharding_strategy,
                pass_type,
                is_first_iter,
                is_last_iter_no_sync,
            )
            for pass_type in PassType
        )

    def _get_ref_num_all_gathers_in_pass(
        self,
        num_fsdp: int,
        sharding_strategy: ShardingStrategy | None,
        pass_type: PassType,
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ):
        # [FUNCTION DOC] HELPER: Expected all-gather count for a single pass. A None strategy
        # is treated as FULL_SHARD. Forward: zero for SHARD_GRAD_OP if the previous iteration
        # ran in no_sync() (parameters were never freed), else one per FSDP instance; on the
        # first iteration the forward count is tripled because execution-order validation adds
        # two extra all-gathers per real one. Backward: FULL_SHARD gives one per instance
        # except the root; SHARD_GRAD_OP gives zero. Raises AssertionError for any other
        # strategy/pass combination.
        """Returns the reference number of all-gathers for a given setting."""
        if sharding_strategy is None:
            sharding_strategy = ShardingStrategy.FULL_SHARD  # default
        # Forward pass:
        if (
            pass_type == PassType.FWD
            and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
            and is_last_iter_no_sync
        ):
            # Modules do not free the full parameters in the last
            # iteration's backward pass if it was in `no_sync()`
            num_all_gathers = 0
        elif pass_type == PassType.FWD:
            # Otherwise, all modules all-gather the full parameters in the
            # forward pass
            num_all_gathers = num_fsdp
        # Backward pass:
        elif (
            pass_type == PassType.BWD
            and sharding_strategy == ShardingStrategy.FULL_SHARD
        ):
            # Root does not free the full parameters at the end of the
            # forward pass
            num_all_gathers = num_fsdp - 1
        elif (
            pass_type == PassType.BWD
            and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        ):
            # Modules do not free the full parameters at the end of the
            # forward pass
            num_all_gathers = 0
        else:
            raise AssertionError(
                f"Unsupported: add a branch for pass_type={pass_type} "
                f"is_first_iter={is_first_iter} "
                f"is_last_iter_no_sync={is_last_iter_no_sync} "
                f"sharding_strategy={sharding_strategy}"
            )
        if is_first_iter and pass_type == PassType.FWD:
            # With execution order validation, on the first iteration, we have
            # an additional two all-gathers before every actual all-gather in
            # the forward pass
            num_all_gathers *= 3
        return num_all_gathers

    def _print_ref_num_all_gathers_in_pass(
        self,
        num_fsdp: int,
        sharding_strategy: ShardingStrategy,
        pass_type: PassType,
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ):
        # [FUNCTION DOC] HELPER: Debugging aid that prints, on rank 0 only, the pass type,
        # iteration flags, strategy and the reference all-gather count for that setting.
        # Not called by the tests; prints to stdout.
        """Helper method for printing the number of all-gathers for a specific
        setting. This may be helpful since the branching is complex."""
        if self.rank != 0:
            return  # only print on one rank
        num_all_gathers = self._get_ref_num_all_gathers_in_pass(
            num_fsdp,
            sharding_strategy,
            pass_type,
            is_first_iter,
            is_last_iter_no_sync,
        )
        print(
            f"Pass: {pass_type}\n"
            f"Is First Iteration: {is_first_iter}\n"
            f"Sharding Strategy: {sharding_strategy}\n"
            f"Last iteration in `no_sync()`: {is_last_iter_no_sync}\n"
            f"Number of all-gathers: {num_all_gathers}"
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("nested_model", [False, True])
    @parametrize("use_no_sync", [False, True])
    @parametrize("sharding_strategy", [ShardingStrategy.SHARD_GRAD_OP, None])
    def test_communication(
        self,
        device,
        nested_model: bool,
        use_no_sync: bool,
        sharding_strategy: ShardingStrategy | None,
    ):
        # [FUNCTION DOC] Parameterized over nested model, no_sync use and sharding strategy
        # (SHARD_GRAD_OP or default). Enables detailed distributed debug level, builds the
        # model, then with all-gather and reduce-scatter single-tensor collectives mocked,
        # runs three iterations inside no_sync() (only if requested) followed by three
        # iterations outside it, asserting after each that the mock call counts equal the
        # reference counts. Side effect: sets the global distributed debug level.
        """
        Tests FSDP's communication cost in terms of calls to collective
        communication primitives (i.e. all-gather and reduce-scatter).
        Arguments:
            nested_model (bool): If ``True``, uses ``NestedWrappedModule``,
                which has nested FSDP instances; if ``False``, uses the default
                model, which does not have nested FSDP instances.
            use_no_sync (bool): If ``True``, runs some iterations inside the
                ``no_sync()`` context manager to accumulate gradients, followed
                by some iterations outside the context manager; if ``False``,
                only runs some iterations outside the context manager.
            sharding_strategy (Optional[ShardingStrategy]): Configures the
                FSDP algorithm.
        """
        # Enable execution order checking
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        # Initialize the model and inputs
        fsdp_model = self._init_model(device_type, nested_model, sharding_strategy)
        batch = fsdp_model.module.get_input(device_type)
        # Count the number of FSDP instances that manage parameters since the
        # number of collectives are a function of this number
        num_fsdp = sum(
            (isinstance(m, FSDP) and len(m.params) > 0) for m in fsdp_model.modules()
        )
        # If `use_no_sync=True`, we run `num_iters` iterations inside
        # `no_sync()` followed by `num_iters` iterations outside `no_sync()`,
        # and if `use_no_sync=False`, we only run `num_iters` iterations
        # outside `no_sync()`
        num_iters = 3
        with (
            patch("torch.distributed.all_gather_single") as mock_all_gather,
            patch("torch.distributed.reduce_scatter_single") as mock_reduce_scatter,
        ):

            def reset_mocks():
                # [FUNCTION DOC] HELPER: Clears the recorded calls of both collective mocks
                # so the next iteration's counts start from zero.
                mock_all_gather.reset_mock()
                mock_reduce_scatter.reset_mock()

            # Check the communication cost when using `no_sync()`
            if use_no_sync:
                for i in range(num_iters):
                    reset_mocks()
                    self._run_iter(fsdp_model, batch, use_no_sync=True)
                    num_all_gathers = mock_all_gather.call_count
                    num_reduce_scatters = mock_reduce_scatter.call_count
                    ref_num_all_gathers = self._get_ref_num_all_gathers(
                        num_fsdp,
                        sharding_strategy,
                        is_first_iter=i == 0,
                        is_last_iter_no_sync=i > 0,
                    )
                    ref_num_reduce_scatters = self._get_ref_num_reduce_scatters(
                        num_fsdp,
                        in_no_sync=True,
                    )
                    self.assertEqual(num_all_gathers, ref_num_all_gathers)
                    self.assertEqual(num_reduce_scatters, ref_num_reduce_scatters)
            # Check the normal communication cost (when not using `no_sync()`)
            for i in range(num_iters):
                reset_mocks()
                self._run_iter(fsdp_model, batch, use_no_sync=False)
                num_all_gathers = mock_all_gather.call_count
                num_reduce_scatters = mock_reduce_scatter.call_count
                ref_num_all_gathers = self._get_ref_num_all_gathers(
                    num_fsdp,
                    sharding_strategy,
                    is_first_iter=not use_no_sync and i == 0,
                    is_last_iter_no_sync=use_no_sync and i == 0,
                )
                ref_num_reduce_scatters = self._get_ref_num_reduce_scatters(
                    num_fsdp,
                    in_no_sync=False,
                )
                self.assertEqual(num_all_gathers, ref_num_all_gathers)
                self.assertEqual(num_reduce_scatters, ref_num_reduce_scatters)


class TestExplicitUnshard(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case for FSDP's private explicit-unshard API; compares
    # against a DDP reference model and checks the handle returned by _unshard.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Number of processes to launch: the accelerator device count, capped
        # at 2.
        return min(_get_device_module(self.device_type).device_count(), 2)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    def test_unshard_async(self, device, use_orig_params: bool):
        # [FUNCTION DOC] Checks that asynchronously unsharding FSDP-wrapped MLPs while
        # non-differentiable async all-reduces are in flight yields the same losses as a plain
        # DDP reference. Builds the reference and a model whose inner MLPs are FSDP-wrapped
        # (SHARD_GRAD_OP, parameters ignored by the outer DDP), then runs ten training steps
        # on both with Adam, asserting equal losses and waiting on unshard streams each step.
        class ReduceModule(nn.Module):
            # [CLASS DOC] HELPER: Linear-plus-relu module that also launches an async
            # all-reduce on its output over the given group and returns both result and work.
            def __init__(self, dim: int, group: dist.ProcessGroup):
                # [FUNCTION DOC] HELPER: Stores the group and creates a random dim x dim
                # weight parameter.
                super().__init__()
                self.group = group
                self.weight = nn.Parameter(torch.randn(dim, dim))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Computes relu of x times the weight, starts an async
                # in-place all-reduce on it, and returns the tensor and the work handle (the
                # caller must wait on the work).
                y = F.relu(x @ self.weight)
                # NOTE: This all-reduce is not differentiable and is included
                # to exercise the overlap.
                work = dist.all_reduce(y, group=self.group, async_op=True)
                return y, work

        class MLPs(nn.Module):
            # [CLASS DOC] HELPER: Three independent MLPs applied to three inputs whose
            # all-reduces must complete first; the FSDP-wrapped submodule in the test.
            def __init__(self, dim: int):
                # [FUNCTION DOC] HELPER: Creates three MLP submodules of width dim.
                super().__init__()
                self.mlp1 = MLP(dim)
                self.mlp2 = MLP(dim)
                self.mlp3 = MLP(dim)

            def forward(self, ys: list[torch.Tensor], works: list[dist.Work]):
                # [FUNCTION DOC] HELPER: For each of the three (input, work) pairs in order,
                # waits on the pending all-reduce then applies the matching MLP; returns the
                # sum of the three outputs.
                (y1, y2, y3), (work1, work2, work3) = ys, works
                work1.wait()
                z1 = self.mlp1(y1)
                work2.wait()
                z2 = self.mlp2(y2)
                work3.wait()
                z3 = self.mlp3(y3)
                return z1 + z2 + z3

        class ReduceModel(nn.Module):
            # [CLASS DOC] HELPER: Model of three ReduceModules feeding one MLPs block; its
            # forward triggers explicit async unshard of each FSDP-wrapped MLP to overlap with
            # the later all-reduces.
            def __init__(self, dim: int, group: dist.ProcessGroup):
                # [FUNCTION DOC] HELPER: Creates three ReduceModules and the MLPs block.
                super().__init__()
                self.reduce_module1 = ReduceModule(dim, group)
                self.reduce_module2 = ReduceModule(dim, group)
                self.reduce_module3 = ReduceModule(dim, group)
                self.mlps = MLPs(dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Runs the three reduce modules on x; after each, if the
                # corresponding MLP is an FSDP instance, starts its asynchronous unshard so it
                # overlaps with the next all-reduce. Returns the MLPs block output.
                y1, work1 = self.reduce_module1(x)
                if isinstance(self.mlps.mlp1, FSDP):
                    self.mlps.mlp1._unshard(async_op=True)
                y2, work2 = self.reduce_module2(x)
                if isinstance(self.mlps.mlp2, FSDP):
                    self.mlps.mlp2._unshard(async_op=True)
                y3, work3 = self.reduce_module3(x)
                if isinstance(self.mlps.mlp3, FSDP):
                    self.mlps.mlp3._unshard(async_op=True)
                return self.mlps([y1, y2, y3], [work1, work2, work3])

        group = self.process_group
        batch_size, dim = 2, 8
        torch.manual_seed(42)
        ref_model = DDP(ReduceModel(dim, group).to(device_type), device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        torch.manual_seed(42)
        model = ReduceModel(dim, group)
        model.mlps = FSDP(
            model.mlps,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            auto_wrap_policy=ModuleWrapPolicy((MLP,)),
            device_id=device_type.type,
            use_orig_params=use_orig_params,
        )
        model.mlps.check_is_root()
        mlp_params = set(model.mlps.parameters())
        mlp_param_names = {n for n, p in model.named_parameters() if p in mlp_params}
        DDP._set_params_and_buffers_to_ignore_for_model(model, mlp_param_names)
        model = DDP(model.to(device_type), device_ids=[self.rank])
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((batch_size, dim), device=device_type)
        for _ in range(10):
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
                _optim.zero_grad()
            self.assertEqual(losses[0], losses[1])
            model.module.mlps._wait_unshard_streams_on_current_stream()

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    @parametrize("async_op", [False, True])
    def test_unshard_after_backward(
        self, device, use_orig_params: bool, async_op: bool
    ):
        # [FUNCTION DOC] Regression test: after a forward and backward through a model whose
        # first layer is FSDP-wrapped, calling _unshard must not raise. With async_op it must
        # return a handle that can be waited on; without, it must return None. Afterwards the
        # wrapped parameters must be non-empty.
        """
        Tests that _unshard() works correctly after backward().
        This is a regression test for issue #168947 where a Stream was
        incorrectly passed instead of an Event to UnshardHandle.
        """
        model = nn.Sequential(
            MLP(dim=8),
            nn.Linear(8, 4, device=device_type.type),
        )
        model[0] = FSDP(
            model[0],
            device_id=device_type.type,
            use_orig_params=use_orig_params,
        )

        inp = torch.randn(2, 8, device=device_type.type)
        output = model(inp)
        loss = output.sum()
        loss.backward()

        # This should not raise "AttributeError: 'Stream' object has no attribute 'wait'"
        handle = model[0]._unshard(async_op=async_op)

        if async_op:
            self.assertIsNotNone(handle)
            handle.wait()
        else:
            self.assertIsNone(handle)

        for param in model[0].parameters():
            self.assertIsNotNone(param)
            self.assertTrue(param.numel() > 0)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestCommunication, globals(), only_for=devices, allow_xpu=True
)
instantiate_device_type_tests(
    TestExplicitUnshard, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
