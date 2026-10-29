# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests of the composable replicate API combined with torch.compile and compiled
# autograd: result parity against eager DDP and replicate (with no_sync, activation
# checkpointing, bf16/fp16 comm hooks, backward-only compilation), DDP graph splitting and
# gradient bucketing with fused all-reduce passes checked via generated code, and a
# (disabled) DDP plus tensor-parallel case. Defines a 4-layer Net and helper test bases.
# Import-time: computes device_type and DIM (2000).

import contextlib
import functools
import unittest
from collections.abc import Callable
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import _inductor as inductor, nn
from torch._C import FileCheck
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import counters
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
    sm_is_or_higher_than,
)
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import IS_LINUX, run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils.checkpoint import checkpoint


device_type = get_devtype().type

DIM = 2000


class Net(nn.Module):
    # [CLASS DOC] HELPER: Four stacked DIM x DIM linear layers (fc1 to fc4) without
    # activations. The use_checkpoint flag makes forward run fc1 under non-reentrant
    # activation checkpointing.
    def __init__(self, checkpoint=False):
        # [FUNCTION DOC] HELPER: Creates the four linear layers and stores the checkpoint
        # flag as use_checkpoint.
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, DIM)
        self.use_checkpoint = checkpoint

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs fc1 (via non-reentrant checkpoint if enabled), then
        # fc2, fc3, fc4, and returns the output.
        if self.use_checkpoint:
            _fc1 = checkpoint(self.fc1, x, use_reentrant=False)
        else:
            _fc1 = self.fc1(x)
        return self.fc4(self.fc3(self.fc2(_fc1)))


def compiler_fn(no_inductor=False):
    # [FUNCTION DOC] HELPER: Returns a compiler callback for compiled autograd. The callback
    # takes a graph module and returns it wrapped with a full-graph torch.compile whose
    # backend returns the graph unchanged when no_inductor is true, else compiles it with
    # Inductor.
    def _compiler_fn(gm):
        # [FUNCTION DOC] HELPER: Wraps gm with full-graph torch.compile using the inner
        # backend and returns the compiled callable.
        def inner_compiler(gm_, example_inputs_):
            # [FUNCTION DOC] HELPER: Backend: returns gm_ itself if no_inductor was set,
            # otherwise the result of Inductor compiling gm_ with the example inputs.
            if no_inductor:
                return gm_
            else:
                return inductor.compile(gm_, example_inputs_)

        gm = torch.compile(gm, fullgraph=True, backend=inner_compiler)
        return gm

    return _compiler_fn


class MultiProcessInductorTestCase(DistributedTestBase, InductorTestCase):
    # [CLASS DOC] HELPER: Combines the distributed multi-process test base with the
    # Inductor test case so each process isolates its Inductor cache directory. No added
    # behavior.
    """
    A version of MultiProcessTestCase that derives from the Inductor TestCase
    to handle isolation of the inductor cache dir.
    """


class ReplicateTest(MultiProcessInductorTestCase):
    # [CLASS DOC] Multi-process test case (up to 2 ranks) checking replicate plus torch.compile
    # and compiled autograd against eager DDP and uncompiled replicate, and DDP bucketing.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of 2 and the number of devices of the
        # current device type.
        return min(2, torch.get_device_module(device_type).device_count())

    def _test_compile(
        self,
        *,
        no_sync: bool,
        setup_func: Callable | None = None,
        no_inductor: bool = False,
        no_compile_forward: bool = False,
        checkpoint: bool = False,
        device: str | torch.device,
    ):
        # [FUNCTION DOC] HELPER: Core parity test. Creates the process group, sets the
        # global dynamo optimize_ddp option to python_reducer (side effect), and builds
        # three copies of a Net: eager replicate, a compiled replicate (compiled forward
        # unless no_compile_forward) and compiled DDP, each with Adam. Optional setup_func
        # may register hooks on all three. For 2 iterations with rank-dependent inputs it
        # runs forward/backward (backward of the compiled ones under compiled autograd,
        # using no_inductor to pick the backend) toggling gradient sync when no_sync is set,
        # and on sync iterations asserts gradients equal across all three and steps and
        # zeroes optimizers. Finally asserts parameters match, and destroys the group.
        # KNOWN ISSUE: the sync context chosen in the loop is unconditionally overwritten by
        # a null context before use, so the DDP no_sync context is never actually entered.
        self.create_pg(device)
        torch._dynamo.config.optimize_ddp = "python_reducer"
        torch.manual_seed(123)
        if device_type == "xpu":
            torch.use_deterministic_algorithms(True, warn_only=True)
        model = Net(checkpoint=checkpoint).to(device)
        input = torch.randn([1, DIM], device=device)

        compiled_replicate_model = replicate(deepcopy(model))
        if not no_compile_forward:
            compiled_replicate_model = torch.compile(
                compiled_replicate_model, fullgraph=False
            )
        compiled_replicate_optim = torch.optim.Adam(
            compiled_replicate_model.parameters()
        )
        compiled_ddp_model = DDP(deepcopy(model))
        if not no_compile_forward:
            compiled_ddp_model = torch.compile(compiled_ddp_model, fullgraph=True)
        compiled_ddp_optim = torch.optim.Adam(compiled_ddp_model.parameters())
        model = replicate(model)
        optim = torch.optim.Adam(model.parameters())

        if setup_func:
            setup_func(model, compiled_replicate_model, compiled_ddp_model)

        models = [model, compiled_replicate_model, compiled_ddp_model]
        optims = [optim, compiled_replicate_optim, compiled_ddp_optim]
        sync_contexts = [
            contextlib.nullcontext(),
            contextlib.nullcontext(),
            compiled_ddp_model.no_sync(),
        ]

        # Run multiple iterations so that we could test no_sync
        for i in range(2):
            # Setting a different random seed so that if the allreduces are not
            # executed correctly, the gradients won't be correct compared to the
            # eager DDP.
            torch.manual_seed(123 + self.rank + i)
            input = torch.randn([1, DIM], device=device)

            for model_idx in range(3):
                if no_sync and i % 2 == 0:
                    context = sync_contexts[model_idx]
                    if model_idx <= 1:
                        models[model_idx].set_requires_gradient_sync(False)
                else:
                    context = contextlib.nullcontext()
                    if model_idx <= 1:
                        models[model_idx].set_requires_gradient_sync(True)
                context = contextlib.nullcontext()

                with context:
                    bwd_context = (
                        contextlib.nullcontext()
                        if model_idx == 0
                        else compiled_autograd._enable(compiler_fn(no_inductor))
                    )
                    with bwd_context:
                        loss = models[model_idx](input).sum()
                        loss.backward()

            if not no_sync or i % 2 == 1:
                for p1, p2, p3 in zip(
                    model.parameters(),
                    compiled_replicate_model.parameters(),
                    compiled_ddp_model.parameters(),
                ):
                    self.assertEqual(p1.grad, p2.grad)
                    self.assertEqual(p1.grad, p3.grad)
                for optim in optims:
                    optim.step()
                    optim.zero_grad()

        self.assertEqual(
            tuple(model.parameters()), tuple(compiled_replicate_model.parameters())
        )
        self.assertEqual(
            tuple(model.parameters()), tuple(compiled_ddp_model.parameters())
        )
        dist.destroy_process_group()

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/160597")
    def test_compile_cpu(self):
        # [FUNCTION DOC] Sets the global Inductor DDP fusion passes to the coalesced-op
        # fusion and comm-wait scheduling (side effect, not restored), then runs the parity
        # test on CPU with gradient sync. Skipped on Linux (the skip decorator is tied to a
        # tracked issue), so it effectively never runs there.
        # Test the coalesced_op with CPU.
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
        self._test_compile(no_sync=False, device="cpu")

    def test_compile_cpu_no_sync(self):
        # [FUNCTION DOC] Same as the CPU test (also sets the global fusion passes, not
        # restored) but alternates gradient sync off and on across iterations.
        # Test the coalesced_op with CPU.
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
        self._test_compile(no_sync=True, device="cpu")

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    @torch._inductor.config.patch(
        reorder_for_locality=False, reorder_for_peak_memory=False
    )
    def test_compile_gpu(self):
        # [FUNCTION DOC] Runs the parity test on the accelerator without checkpointing, with
        # Inductor locality and peak-memory reordering disabled by the decorator.
        self._test_compile(no_sync=False, checkpoint=False, device=device_type)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @torch._inductor.config.patch(
        reorder_for_locality=False, reorder_for_peak_memory=False
    )
    def test_compile_gpu_ac(self):
        # [FUNCTION DOC] Runs the parity test on the accelerator with activation
        # checkpointing enabled in the model, reorderings disabled by the decorator.
        self._test_compile(no_sync=False, checkpoint=True, device=device_type)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compile_bf16(self):
        # [FUNCTION DOC] Skips unless the device has compute capability 8.0 or higher (or is
        # HIP). Runs the parity test with a setup that registers the bf16 compression
        # communication hook on the eager model, the original module under the compiled
        # replicate model, and the compiled DDP model.
        # Check device capability wrt bf16
        if (
            not sm_is_or_higher_than(torch.device(device_type), 8, 0)
            and torch.version.hip is None
        ):
            self.skipTest("bf16 requires sm >= 8.0")

        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            # [FUNCTION DOC] HELPER: Registers the bf16 compress hook (no state) on each of
            # the three models; for the compiled replicate model it uses the wrapped
            # original module.
            model.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )

        self._test_compile(no_sync=False, setup_func=setup, device=device_type)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_compile_fp16(self):
        # [FUNCTION DOC] Runs the parity test with the fp16 compression hook registered on
        # all three models, with Inductor disabled for compiled autograd (a TODO notes the
        # cause is unknown).
        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            # [FUNCTION DOC] HELPER: Registers the fp16 compress hook (no state) on each of
            # the three models; for the compiled replicate model it uses the wrapped
            # original module.
            model.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )

        # TODO: figure out why we need to disable Inductor to avoid test errors.
        self._test_compile(
            no_sync=False, setup_func=setup, no_inductor=True, device=device_type
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compile_backward_only(self):
        # [FUNCTION DOC] Runs the parity test without compiling the forward pass, so only
        # backward goes through compiled autograd.
        self._test_compile(no_sync=False, no_compile_forward=True, device=device_type)

    def test_ddp_optimizer_splits_graph(self):
        # [FUNCTION DOC] Skips if world size is below 2. Initializes a gloo group from a
        # file store, sets optimize_ddp to python_reducer, compiles a replicated Net, runs
        # forward and a compiled-autograd backward, and asserts the Inductor ddp_buckets
        # counter is positive. Destroys the group.
        if self.world_size < 2:
            self.skipTest("DDP bucketing requires world_size >= 2")
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )
        torch._dynamo.config.optimize_ddp = "python_reducer"

        model = Net()
        compiled_model = torch.compile(replicate(model), fullgraph=False)

        input = torch.randn([1, DIM])
        loss = compiled_model(input).sum()

        with compiled_autograd._enable(compiler_fn()):
            loss.backward()

        self.assertGreater(counters["inductor"]["ddp_buckets"], 0)

        dist.destroy_process_group()

    def _test_bucketing(self, init_process_group=True, loop=1):
        # [FUNCTION DOC] HELPER: Optionally initializes a gloo group, compiles a replicated
        # Net, and runs forward loop times, running backward under compiled autograd for all
        # but the last iteration. The last backward is run through the helper that captures
        # the generated Triton/C++ code. Asserts the ddp_buckets counter equals 3 and
        # returns the captured code string. The group is not destroyed here.
        if init_process_group:
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
                store=dist.FileStore(self.file_name, self.world_size),
            )
        model = Net()
        input = torch.randn([1, DIM])
        torch._dynamo.config.optimize_ddp = "python_reducer"
        compiled_replicate_model = torch.compile(
            replicate(deepcopy(model)), fullgraph=False
        )

        def bwd(loss):
            # [FUNCTION DOC] HELPER: Runs backward on loss under compiled autograd.
            with compiled_autograd._enable(compiler_fn()):
                loss.backward()

        for i in range(loop):
            loss = compiled_replicate_model(input).sum()
            if i != loop - 1:
                # Leave the last bwd for the run_and_get_triton_code.
                bwd(loss)

        code = run_and_get_triton_code(functools.partial(bwd, loss=loss))

        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        return code

    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
    )
    # todo: This pass mucks things up since Inductor thinks its inference
    # and can apply this. Should turn off these passes in compiled autograd
    @torch._inductor.config.patch(
        reorder_for_locality=False,
        reorder_for_peak_memory=False,
        # The correctness of this test relies on the pointless permute ops
        # in the joint graph does not get eliminated..
        pattern_matcher=False,
    )
    def test_bucketing_coalesced_op(self):
        # [FUNCTION DOC] With the coalesced-op fusion pass, checks generated code twice
        # (fresh run, then two iterations): three fused CPU kernels each followed by a
        # coalesced all-reduce, then three wait ops. Skips below world size 2.
        # The second block's comment says gradient is None though it uses a second iteration.
        if self.world_size < 2:
            self.skipTest("DDP bucketing requires world_size >= 2")
        # Gradient is None
        code = self._test_bucketing()
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for _ in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        for _ in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        fc.run(code)

        # Gradient is None
        code = self._test_bucketing(init_process_group=False, loop=2)
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for _ in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        for _ in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        fc.run(code)

    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_concat_op",
            "schedule_comm_wait",
        ]
    )
    # todo: This pass mucks things up since Inductor thinks its inference
    # and can apply this. Should turn off these passes in compiled autograd
    @torch._inductor.config.patch(
        reorder_for_locality=False,
        reorder_for_peak_memory=False,
        # The correctness of this test relies on the pointless permute ops
        # in the joint graph does not get eliminated..
        pattern_matcher=False,
    )
    def test_bucketing_concat_op(self):
        # [FUNCTION DOC] With the concat-op fusion pass, checks generated code twice (fresh
        # run, then two iterations): three groups of flatten, fused kernel and plain
        # all-reduce, then three wait ops. Skips below world size 2.
        if self.world_size < 2:
            self.skipTest("DDP bucketing requires world_size >= 2")
        # Gradient is None
        code = self._test_bucketing()
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for _ in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        for _ in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        fc.run(code)

        # Gradient is not None
        code = self._test_bucketing(init_process_group=False, loop=2)
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for _ in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        for _ in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        fc.run(code)


class DDP_TP_Test(InductorTestCase):
    # [CLASS DOC] Single-process test case that fakes a 4-rank world with the fake backend
    # to test replicate over a data-parallel mesh combined with tensor parallelism.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets rank 0, world size 4, selects device 0 of the current
        # device type, and initializes a fake-backend process group with a fake store.
        super().setUp()
        self.rank = 0
        self.world_size = 4
        torch.get_device_module(device_type).set_device(self.rank)

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the process group (does not call the parent
        # tearDown).
        dist.destroy_process_group()

    @unittest.skip(
        "Temporarily disabled due to SymInt error: `unhashable type: non-nested SymInt`"
    )
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_ddp_tp(self):
        # [FUNCTION DOC] Currently skipped (SymInt hashing error). Builds a 2x2 dp/tp mesh,
        # applies column/row parallel plans to two Net copies, replicates both on the dp
        # mesh, compiles one, and expects compiled-autograd backward to raise an
        # AssertionError about getting a DTensor instead of a ProxyTensor (known
        # limitation; the eager reference comparison is commented out).
        ref_model = Net()
        compiled_replicate_model = deepcopy(ref_model)
        mesh_2d = init_device_mesh(
            device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
            "fc3": ColwiseParallel(),
            "fc4": RowwiseParallel(),
        }
        ref_model = parallelize_module(ref_model, tp_mesh, parallelize_plan)
        ref_model = replicate(ref_model, device_mesh=dp_mesh)
        compiled_replicate_model = parallelize_module(
            compiled_replicate_model, tp_mesh, parallelize_plan
        )
        compiled_replicate_model = replicate(
            compiled_replicate_model, device_mesh=dp_mesh
        )
        compiled_replicate_model = torch.compile(compiled_replicate_model)
        data = torch.randn([1, DIM])
        with compiled_autograd._enable(compiler_fn()):
            loss = compiled_replicate_model(data).sum()
            # TODO: We need "pre-dispatch tracing of backward graph" to make this work:
            # https://github.com/pytorch/pytorch/issues/127797#issuecomment-2291695474
            with self.assertRaisesRegex(
                AssertionError,
                "Expected ProxyTensor, got <class 'torch.distributed.tensor.DTensor'>",
            ):
                loss.backward()

        # ref_loss = ref_model(data).sum()
        # ref_loss.backward()
        # for p1, p2 in zip(
        #     ref_model.parameters(), compiled_replicate_model.parameters()
        # ):
        #     self.assertEqual(p1.grad, p2.grad)


if __name__ == "__main__":
    run_tests()
