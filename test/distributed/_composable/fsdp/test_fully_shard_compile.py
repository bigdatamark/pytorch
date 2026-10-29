# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for fully_shard and legacy FSDP interacting with torch.compile, dynamo and
# compiled autograd. Contents: a small encoder module, tests that dynamo does not trace FSDP hook
# code, compiled autograd graph breaks on FSDP backward hooks, a compiled optimizer step with
# uneven shards, DTensor spec dtype under TP plus mixed precision, tracing of the
# use_training_state context manager and the fsdp copy_ op, and recompiling per-layer under FSDP.
# Most tests are skipped without a GPU backend (HAS_GPU). Import-time side effect: defines the
# module-level device_type from get_devtype().


import copy
import unittest

import torch
import torch._dynamo.compiled_autograd as compiled_autograd
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.utils import counters
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    fully_shard,
    FullyShardedDataParallel as FSDP,
    MixedPrecisionPolicy,
    ShardingStrategy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


device_type = torch.device(get_devtype())


class Mod(torch.nn.Module):
    # [CLASS DOC] HELPER: Small MLP-like model with an encoder of three Linear layers (784 to
    # 1024, 1024 to 1024, 1024 to 4096) created directly on device_type.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Builds the encoder sequential of three Linear
        # layers on device_type.
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 1024, device=device_type),
            torch.nn.Linear(1024, 1024, device=device_type),
            torch.nn.Linear(1024, 4096, device=device_type),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the input through the encoder and returns the result.
        return self.encoder(x)


class TestFullyShardCompileCompute(FSDPTest):
    # [CLASS DOC] Multi-process (GPU) tests that compile or trace modules wrapped by fully_shard
    # and check dynamo, compiled autograd and inductor behavior with FSDP2 hooks and DTensors.
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_disable_compiling_hooks(self):
        # [FUNCTION DOC] Verifies dynamo never traces into the FSDP hook source file. Temporarily
        # replaces the dynamo trace_rules check function with a wrapper that counts calls whose
        # code object comes from _fsdp_state.py, compiles a fully_shard MLP, runs forward and
        # backward, restores the original check (barriers around the swap) and asserts the count
        # is 0. Needs GPU backend and at least 2 GPUs.
        """Verify that dynamo never traces into FSDP hooks in forward or backward."""
        torch._dynamo.reset()
        trace_rules_check_count = 0
        HOOKS_FILE_NAME = "torch/distributed/fsdp/_fully_shard/_fsdp_state.py"

        def patched_trace_rules_check(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for the dynamo trace_rules check: increments the
            # enclosing counter when the first argument's code file name ends with the FSDP state
            # hooks file, then delegates to the original check and returns its result.
            nonlocal trace_rules_check_count
            f_code = args[0]
            if hasattr(f_code, "co_filename") and f_code.co_filename.endswith(
                HOOKS_FILE_NAME
            ):
                trace_rules_check_count += 1
            return orig_trace_rules_check(*args, **kwargs)

        orig_trace_rules_check = torch._dynamo.trace_rules.check
        torch.distributed.barrier()
        torch._dynamo.trace_rules.check = patched_trace_rules_check
        model = MLP(4).to(device_type)
        fully_shard(model)
        model.compile()
        out = model(torch.randn((4, 4), device=device_type))
        out.sum().backward()
        torch.distributed.barrier()
        torch._dynamo.trace_rules.check = orig_trace_rules_check
        self.assertEqual(trace_rules_check_count, 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compiled_autograd_fsdp2_backward(self):
        # [FUNCTION DOC] Verifies compiled autograd works with FSDP2 backward and that FSDP hooks
        # cause graph breaks. Runs an eager reference on a copy, then runs forward and backward
        # on the fully_shard model under compiled autograd with a counting aot_eager backend.
        # Asserts output equals the reference, exactly one compiled autograd capture and 2
        # backend compilations (two graph breaks from disabled pre-backward and post-backward
        # hooks). Needs GPU backend and at least 2 GPUs.
        """
        Verify that compiled autograd with FSDP2 backward does not crash and
        that FSDP hooks cause graph breaks (due to _dynamo_disable).

        For MLP(4) with a single fully_shard, the compiled autograd backward
        graph has 2 graph breaks from _dynamo_disable on FSDP hooks:
          1. _pre_backward tensor hook (registered on FSDP module output)
          2. post_backward (called from RegisterPostBackwardFunction.backward)
        _root_post_backward_final_callback runs outside the compiled autograd
        scope as a queued engine callback.
        """
        torch._dynamo.reset()
        counters.clear()
        model = MLP(4).to(device_type)
        ref_model = copy.deepcopy(model)
        fully_shard(model)
        fully_shard(ref_model)
        inp = torch.randn((4, 4), device=device_type)

        # Eager reference
        ref_out = ref_model(inp)
        ref_out.sum().backward()

        backend_count = 0
        orig_aot_eager = torch._dynamo.lookup_backend("aot_eager")

        def counting_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend wrapper that counts invocations in the enclosing
            # counter and delegates compilation to the aot_eager backend.
            nonlocal backend_count
            backend_count += 1
            return orig_aot_eager(gm, example_inputs)

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiler hook for compiled autograd: compiles the captured
            # graph with torch.compile using the counting backend.
            return torch.compile(gm, backend=counting_backend)

        with compiled_autograd._enable(compiler_fn):
            out = model(inp)
            out.sum().backward()

        self.assertEqual(out, ref_out)
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # 2 graph breaks from _dynamo_disable on FSDP hooks split the
        # compiled autograd graph into 2 sub-graphs compiled by the backend:
        #   graph break 1: _pre_backward tensor hook
        #   graph break 2: post_backward (from RegisterPostBackwardFunction)
        self.assertEqual(backend_count, 2)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compile_optimizer_uneven_shard(self):
        # [FUNCTION DOC] Regression test (issue 176667) for compiling the optimizer step on
        # unevenly sharded parameters. Shards a Conv2d with 47 output channels (meta init then
        # to_empty, bfloat16 params, float32 reduce, reshard_after_forward off), runs one eager
        # training step with Adam, then a forward/backward followed by a torch.compile'd
        # optimizer step; passes if nothing raises. Needs GPU backend and at least 2 GPUs.
        # Regression test for https://github.com/pytorch/pytorch/issues/176667
        # When a param dim is not divisible by world_size, FSDP2 creates
        # param._local_tensor as a narrow view of an N-D padded tensor, while
        # grad._local_tensor is a view of a 1-D flat gradient buffer. Dynamo
        # must not reuse param's symbolic context for the grad.
        torch._dynamo.reset()
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # 47 out_channels not divisible by world_size=2, forcing uneven sharding
        with torch.device("meta"):
            model = nn.Conv2d(3, 47, 3, padding=1)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
        model.to_empty(device=device_type)
        with torch.no_grad():
            for p in model.parameters():
                p.uniform_(-0.01, 0.01)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 3, 8, 8, device=device_type, dtype=torch.bfloat16)

        # Eager warmup
        model(x).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Compiled optimizer step
        model(x).sum().backward()
        torch.compile(opt.step)()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(4)
    def test_tp_mixed_precision_dtensor_spec_dtype(self):
        # [FUNCTION DOC] Checks that the DTensor spec dtype of a TP plus FSDP weight matches the
        # bfloat16 mixed precision param dtype, both eager and with the model compiled (eager
        # backend). Skips when world size is odd. Builds a (dp, tp=2) mesh, applies row-wise TP
        # to a one-layer module, fully_shards with a placement function that flips the shard dim,
        # runs a forward, asserts the weight local tensor and spec tensor_meta dtype are
        # bfloat16, then backward and barrier. Needs GPU backend and at least 4 GPUs.
        torch._dynamo.reset()
        if self.world_size % 2 != 0:
            self.skipTest(f"Expects even world size but got {self.world_size}")

        dim, tp_size = 16, 2
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size // tp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = mesh["dp"], mesh["tp"]
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )

        class RowwiseLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one bias-free projection Linear whose forward
            # returns a clone of the projection output; used with row-wise tensor parallelism.
            def __init__(self, dim: int):
                # [FUNCTION DOC] HELPER: Creates a single bias-free Linear projection of size
                # dim on device_type.
                super().__init__()
                self.proj = nn.Linear(dim, dim, bias=False, device=device_type)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the projection and returns a clone
                # of the result.
                return self.proj(x).clone()

        def shard_placement_fn(param: nn.Parameter) -> Shard:
            # [FUNCTION DOC] HELPER: Shard placement function: for a DTensor param returns a
            # Shard on the dim mirrored from the first existing Shard placement (last dim index
            # minus that dim); otherwise returns Shard(0).
            if isinstance(param, DTensor):
                for placement in param.placements:
                    if isinstance(placement, Shard):
                        return Shard(param.ndim - 1 - placement.dim)
            return Shard(0)

        for compile_model in (False, True):
            model = RowwiseLinear(dim)
            parallelize_module(model, tp_mesh, {"proj": RowwiseParallel()})
            if compile_model:
                model.compile(backend="eager")
            fully_shard(
                model,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                shard_placement_fn=shard_placement_fn,
                reshard_after_forward=False,
            )

            x = torch.randn(
                2,
                dim // tp_size,
                device=device_type,
                dtype=torch.bfloat16,
            )
            y = model(x)
            weight = model.proj.weight
            self.assertIsInstance(weight, DTensor)
            self.assertEqual(weight._local_tensor.dtype, torch.bfloat16)
            self.assertIsNotNone(weight._spec.tensor_meta)
            self.assertEqual(weight._spec.tensor_meta.dtype, torch.bfloat16)
            y.sum().backward()
            dist.barrier()


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFullyShardCompile(FSDPTest):
    # [CLASS DOC] Tests (skipped without GPU backend) of tracing FSDP
    # internals and legacy FSDP wrapping with dynamo and aot_eager.
    def test_dynamo_trace_use_training_state(self):
        # [FUNCTION DOC] Checks that dynamo can trace the use_training_state context manager of
        # an FSDPParamGroup. Builds a dummy param group with no parameters, runs a function that
        # sets the state to IDLE, enters the context with FORWARD and returns x plus 1 only if
        # the state is FORWARD. Asserts the state is restored to IDLE, the eager and
        # fullgraph-compiled outputs match, and the compile used one frame, one op and one graph.
        torch._dynamo.reset()
        # Construct a dummy FSDPParamGroup, since we just want to test the `use_training_state` ctx manager.
        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            (torch.nn.Linear(1, 1),),  # module: Tuple[nn.Module, ...],
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            device_type,  # device: torch.device,
            None,  # shard_placement_fn: Optional[Callable],
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def f(x):
            # [FUNCTION DOC] HELPER: Resets the param group training state to IDLE, enters the
            # FORWARD training state context, and returns x plus 1 if the state is FORWARD inside
            # it, else x.
            param_group._training_state = TrainingState.IDLE
            with param_group.use_training_state(TrainingState.FORWARD):
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1
                else:
                    return x

        inp = torch.zeros(1)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)

        eager_out = f(inp)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        self.assertEqual(eager_out, inp + 1)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_out = torch.compile(f, backend=cnt, fullgraph=True)(inp)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)
        self.assertEqual(len(cnt.graphs), 1)

    def test_trace_fsdp_copy_(self):
        # [FUNCTION DOC] Checks the fsdp copy_ op can be traced by aot_eager. Registers a custom
        # op add_one_out, defines a function that writes x plus 1 into a buffer view and then
        # copies another view of that buffer into x with the fsdp copy_ op, and asserts the
        # compiled run yields the same x as the eager run on a copy.
        @torch.library.custom_op("mylib::add_one_out", mutates_args={"out"})
        def add_one_out(x: torch.Tensor, out: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op body: writes x plus 1 into out in place (out is
            # declared as mutated).
            torch.add(x, 1, out=out)

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a zero buffer via the add_one_out op on one view,
            # then copies a second view of the buffer into x via the fsdp copy_ op; mutates x
            # and returns None.
            buf = torch.zeros(2)
            buf_view = buf.view(-1)
            torch.ops.mylib.add_one_out(x, out=buf_view)
            buf_view2 = buf.view(-1)
            torch.ops.fsdp.copy_(x, buf_view2)

        ref_x = torch.zeros(2)
        x = copy.deepcopy(ref_x)
        f(ref_x)
        torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(x, ref_x)

    def test_dynamo_recompiles_on_fsdp_layers(self):
        # [FUNCTION DOC] Smoke test: wraps each Linear in the Mod encoder with torch.compile,
        # wraps the whole module in legacy FSDP (FULL_SHARD, use_orig_params), and runs one
        # forward; passes if nothing raises (no assertions).
        m = Mod()
        for name, child in m.encoder.named_children():
            if isinstance(child, torch.nn.Linear):
                new_child = torch.compile(child)
                setattr(m.encoder, name, new_child)
        m = FSDP(m, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
        inp = torch.randn(32, 784, device=device_type)
        m(inp)


if __name__ == "__main__":
    run_tests()
