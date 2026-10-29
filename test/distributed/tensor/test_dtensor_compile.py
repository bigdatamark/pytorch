# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Compile tests for DTensor: checks that torch.compile (Dynamo, AOTAutograd,
# Inductor), export and make_fx can trace DTensor, DeviceMesh and placement objects. Defines a
# PytreeTuple opaque test class, helper graph-capture compilers, and three test classes:
# TestDTensorCompile (single-process, fake process group), TestDTensorCompileE2E (multi-process,
# real tensor/FSDP parallel compile) and TestDTensorACCompile. Import-time side effects: sets
# dev_type/device_type from the current accelerator, registers PytreeTuple as a constant opaque
# custom class, and builds aot_eager_graph.

import contextlib
import copy
import functools
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.compiler.config
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C import FileCheck
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._functorch._aot_autograd.autograd_cache import check_cacheable
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import skipXPUIf
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    skipIfHpu,
    skipIfTorchDynamo,
    skipIfXpu,
    TEST_WITH_SLOW,
    TEST_XPU,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import checkpoint


dev_type = torch.device(get_devtype())


class PytreeTuple(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Test-only tuple-like container (custom class base) that wraps a tuple
    # of values and mimics tuple behavior: indexing, iteration, length, equality with itself or a
    # plain tuple, hashing and concatenation. Registered below as an opaque constant class so
    # Dynamo can handle instances created while tracing.
    """
    Tuple-like values that are treated as leaves of a PyTree.
    """

    def __init__(self, *values):
        # [FUNCTION DOC] HELPER: Stores the positional arguments as a tuple in _values.
        self._values = tuple(values)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns the class name followed by the values repr with the
        # surrounding tuple parentheses stripped.
        pr = repr(self._values)[1:-1]
        return f"{type(self).__name__}({pr})"

    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Returns the element (or slice) of the stored tuple at index i.
        return self._values[i]

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the stored values.
        return iter(self._values)

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of stored values.
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equal to another PytreeTuple with equal values, or to a plain
        # tuple with equal contents; returns False for any other type.
        if isinstance(other, self.__class__):
            return self._values == other._values
        elif isinstance(other, tuple):
            return self._values == other
        return False

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hashes the stored tuple of values.
        return hash(self._values)

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Concatenation with a PytreeTuple or tuple, returning a new
        # PytreeTuple; raises NotImplementedError for other operand types.
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*self, *other)
        raise NotImplementedError(type(other))

    def __radd__(self, other):
        # [FUNCTION DOC] HELPER: Reflected concatenation (tuple on the left); returns a new
        # PytreeTuple with the other values first. Raises NotImplementedError for other types.
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*other, *self)
        raise NotImplementedError(type(other))

    def index(self, value):
        # [FUNCTION DOC] HELPER: Delegates to tuple index lookup on the stored values.
        return self._values.index(value)

    def count(self, value):
        # [FUNCTION DOC] HELPER: Delegates to tuple count on the stored values.
        return self._values.count(value)

    def __fx_repr__(self) -> tuple[str, dict[str, type]]:
        # [FUNCTION DOC] HELPER: Returns a pair of the repr string and a mapping from type name
        # to type, covering PytreeTuple, each value's type and the types of items inside iterable
        # (non-string) values, so FX codegen can rebuild the object from the string.
        # Return a repr string that can reconstruct this object and the types needed
        # Collect all types used in the values
        types_dict: dict[str, type] = {"PytreeTuple": PytreeTuple}
        for val in self._values:
            val_type = type(val)
            types_dict[val_type.__name__] = val_type
            # Also handle nested types (e.g., Shard inside a list)
            if hasattr(val, "__iter__") and not isinstance(val, str):
                for item in val:
                    item_type = type(item)
                    types_dict[item_type.__name__] = item_type
        return repr(self), types_dict


# Register PytreeTuple as an opaque value type to enable Dynamo to handle
# instances created during tracing
from torch._library.opaque_object import MemberType, register_custom_class


register_custom_class(
    PytreeTuple,
    typ="constant",
    members={
        "__getitem__": MemberType.USE_REAL,
        "__iter__": MemberType.USE_REAL,
        "__len__": MemberType.USE_REAL,
        "__eq__": MemberType.USE_REAL,
        "__hash__": MemberType.USE_REAL,
    },
)


class SimpleModel(nn.Module):
    # [CLASS DOC] HELPER: Small test module with two MLPModule submodules (mlp_0, mlp_1) created
    # on the given device and applied in sequence.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Builds the two MLPModule layers on the given device.
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Runs the input through mlp_0 then mlp_1 and returns the result.
        return self.mlp_1(self.mlp_0(input))


def extract_graph(fx_g, _, graph_cell):
    # [FUNCTION DOC] HELPER: Compiler callback for aot_autograd: stores the generated code of the
    # FX graph into slot 0 of the supplied one-element list graph_cell (side effect) and returns
    # the graph module unchanged. The second argument (example inputs) is unused.
    graph_cell[0] = fx_g.code
    return fx_g


# Make a custom compiler that runs aot autograd but extracts the fw graph
fw_graph_cell = [None]
bw_graph_cell = [None]
fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)
bw_compiler = functools.partial(extract_graph, graph_cell=bw_graph_cell)

from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd


aot_eager_graph = aot_autograd(
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=min_cut_rematerialization_partition,
)

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def _apply_sharding(mod: nn.Module, shard_dim: int, device_mesh: DeviceMesh):
    # [FUNCTION DOC] HELPER: Distributes a module over a 1D mesh through distribute_module. A
    # partition function shards each parameter on shard_dim when the parameter has that many
    # dimensions, otherwise replicates it, and re-registers it as a DTensor Parameter. Returns
    # the module with distributed parameters.
    """
    Shards on the given dimension if possible, else replicate
    Args:
        mod: (nn.Module) Module to shard or replicate
        shard_dim: (int) Dimension to shard on if possible
        device_mesh: (DeviceMesh) 1D Device Mesh

    Returns:
        Sharded DTensor
    """

    def shard_module_params(name, module, device_mesh):
        # [FUNCTION DOC] HELPER: Partition function for distribute_module: for every parameter of
        # the module picks Shard(shard_dim) or Replicate, wraps the distributed tensor as an
        # nn.Parameter and registers it under the last component of its name (mutates module).
        for name, param in module.named_parameters():
            placement = Replicate()
            if shard_dim < len(param.size()):
                placement = Shard(shard_dim)
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [placement])
            )
            name = name.split(".")[-1]
            module.register_parameter(name, dist_param)

    sharded_mod = distribute_module(mod, device_mesh, shard_module_params)
    return sharded_mod


class TestDTensorCompile(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Single-process tests of torch.compile on DTensor. setUp creates a fake process
    # group (world size 2, rank 0) so no real communication occurs; tests compile functions that
    # build or consume DTensor, DeviceMesh and placements and check results, recompilation counts
    # and generated graphs.
    def setUp(self):
        # [FUNCTION DOC] Disables canonicalize_output_graph_node_order in Dynamo config and
        # initializes a fake-backend process group (rank 0, world_size from the property).
        super().setUp()
        torch._dynamo.config.canonicalize_output_graph_node_order = False
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        # [FUNCTION DOC] Restores canonicalize_output_graph_node_order to True, runs the base
        # teardown and destroys the process group.
        torch._dynamo.config.canonicalize_output_graph_node_order = True
        super().tearDown()
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] Returns the module-level device_type string (current accelerator or
        # cpu).
        return device_type

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fake world size, 2.
        return 2

    def test_dtensor_basic(self):
        # [FUNCTION DOC] Compiles a function doing arithmetic on a Shard(0) DTensor with
        # aot_eager (fullgraph) and checks the forward and backward through to_local run.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x * x + 2.
            return x * x + 2

        param = torch.randn(4, 4, requires_grad=True)
        x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)

        res = fn(x)
        res.to_local().sum().backward()

    def test_to_local_backward_unbacked_symbolic_stride(self):
        # [FUNCTION DOC] Marks both dims of a DTensor as unbacked (with hint overrides), compiles
        # a function that calls to_local, multiplies by 2 and sums, and checks backward gives a
        # gradient of 2 everywhere on the local tensor.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local = torch.randn(8, 8, device=self.device_type, requires_grad=True)
        x = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
        torch._dynamo.decorators.mark_unbacked(x, 0, hint_override=x.shape[0])
        torch._dynamo.decorators.mark_unbacked(x, 1, hint_override=x.shape[1])

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the doubled local tensor of x.
            return (x.to_local() * 2).sum()

        fn(x).backward()
        self.assertEqual(local.grad, torch.full_like(local, 2))

    def test_compile_waits_act_nested_in_dtensor_local_tensor(self):
        # [FUNCTION DOC] Regression test: a DTensor whose local tensor is an
        # AsyncCollectiveTensor must be waited on by the compiled runtime path. Counts
        # wait_tensor calls (patched) after warmup, asserts one frame compiled, at least one
        # wait, and the ACT marked completed.
        # Regression test for https://github.com/pytorch/pytorch/issues/180614.
        # AOTAutograd must resolve AsyncCollectiveTensors (ACTs) nested in
        # ``DTensor._local_tensor`` (e.g. from ``redistribute(async_op=True)``
        # or FSDP2's async all-gather) before tracing and again before runtime
        # execution. Otherwise inductor can read an in-flight collective.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x + 1.
            return x + 1

        def make_dtensor_with_act_local():
            # [FUNCTION DOC] HELPER: Builds a Replicate DTensor and replaces its local tensor
            # with an AsyncCollectiveTensor wrapping a clone of the local data.
            dt = DTensor.from_local(
                torch.randn(4, 4, device=self.device_type),
                mesh,
                [Replicate()],
                run_check=False,
            )
            dt._local_tensor = AsyncCollectiveTensor(dt._local_tensor.clone())
            return dt

        wait_calls = []
        orig_wait_tensor = funcol.wait_tensor

        def counting_wait_tensor(t):
            # [FUNCTION DOC] HELPER: Records the call in wait_calls and delegates to the original
            # wait_tensor.
            wait_calls.append(1)
            return orig_wait_tensor(t)

        with patch.object(funcol, "wait_tensor", counting_wait_tensor):
            # Warmup compiles through Python dispatch (which waits); clear the
            # counter so we measure only the compiled runtime path.
            fn(make_dtensor_with_act_local())
            wait_calls.clear()

            dt = make_dtensor_with_act_local()
            self.assertFalse(dt._local_tensor.completed)
            fn(dt)

        self.assertEqual(cnt.frame_count, 1)
        self.assertGreaterEqual(
            len(wait_calls),
            1,
            "compiled graph must wait the AsyncCollectiveTensor nested in "
            "DTensor._local_tensor",
        )
        self.assertTrue(dt._local_tensor.completed)

    def test_async_collective_tensor_nested_in_dtensor_reuses(self):
        # [FUNCTION DOC] Checks that DTensors alternating between ACT-wrapped and plain local
        # tensors reuse a single compiled graph (frame_count stays 1) and always produce the
        # expected result (all twos).
        # FSDP+TP motivating case for the ACT guard relaxation: a DTensor whose
        # _local_tensor alternates between an AsyncCollectiveTensor (async
        # all-gather output) and the resolved plain Tensor must reuse the
        # compiled graph. The recursion into _local_tensor re-enters
        # VariableBuilder.wrap_tensor, hits is_polymorphic_act, and relaxes the
        # inner guard via UnwrapCollectiveTensorSource, while the outer DTensor
        # guards (TYPE_MATCH, DTENSOR_SPEC_MATCH, requires_grad) do not
        # discriminate on the local's class -- so no recompile on the change.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x + 1.
            return x + 1

        def make_dt(act_local):
            # [FUNCTION DOC] HELPER: Builds a ones DTensor; if act_local is true, wraps its local
            # tensor in an AsyncCollectiveTensor.
            dt = DTensor.from_local(
                torch.ones(4, 4, device=self.device_type),
                mesh,
                [Replicate()],
                run_check=False,
            )
            if act_local:
                dt._local_tensor = AsyncCollectiveTensor(dt._local_tensor.clone())
            return dt

        expected = torch.full((4, 4), 2.0, device=self.device_type)
        for i in range(4):
            # Step 0 traces with the ACT local; odd steps resolve to a plain
            # Tensor local before the compiled region.
            out = fn(make_dt(act_local=(i % 2 == 0)))
            self.assertEqual(out.to_local(), expected)
        # ACT and plain-Tensor locals share one compiled graph.
        self.assertEqual(cnt.frame_count, 1)

    def test_direct_aot_dtensor_local_tensor_act_to_plain_no_crash(self):
        # [FUNCTION DOC] Uses aot_function directly: traces with an ACT-local DTensor, then calls
        # with a plain-local DTensor; asserts correct output (input plus one) and that the
        # forward compiler ran only once.
        # Companion to the wait test above: the crash half of
        # https://github.com/pytorch/pytorch/issues/180614. Direct AOTAutograd
        # entry points do not have Dynamo guards to recompile when a nested ACT
        # input is later a plain tensor. The compiled graph must run on the
        # plain local instead of trying to wait or unwrap it as ACT.
        from torch._functorch.aot_autograd import aot_function
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        compile_calls = 0

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Counts compiler invocations (nonlocal compile_calls) and
            # returns the graph module unchanged.
            nonlocal compile_calls
            compile_calls += 1
            return gm

        fn = aot_function(lambda x: x + 1, fw_compiler=fw_compiler)

        # Trace with an ACT-wrapped local so the codegen specializes on it.
        dt_act = DTensor.from_local(
            torch.randn(4, 4, device=self.device_type),
            mesh,
            [Replicate()],
            run_check=False,
        )
        dt_act._local_tensor = AsyncCollectiveTensor(dt_act._local_tensor.clone())
        fn(dt_act)

        # Invoke the same compiled graph with a plain-local DTensor.
        dt_plain = DTensor.from_local(
            torch.randn(4, 4, device=self.device_type),
            mesh,
            [Replicate()],
            run_check=False,
        )
        self.assertNotIsInstance(dt_plain._local_tensor, AsyncCollectiveTensor)
        out = fn(dt_plain)
        self.assertEqual(out.to_local(), dt_plain.to_local() + 1)
        self.assertEqual(compile_calls, 1)

    def test_compile_dtensor_local_tensor_act_backward_passthrough(self):
        # [FUNCTION DOC] Compiles an identity function over a DTensor with ACT local tensor and
        # checks backward still works, i.e. the plain-tensor cotangent is accepted.
        # Passthrough forwards preserve ACT wrappers. If nested ACTs are
        # allowed into traced metadata, backward expects an ACT tangent and
        # rejects the plain-local cotangent autograd supplies.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x unchanged.
            return x

        local = torch.randn(4, 4, device=self.device_type, requires_grad=True)
        dt = DTensor.from_local(local, mesh, [Replicate()], run_check=False)
        dt._local_tensor = AsyncCollectiveTensor(dt._local_tensor.clone())
        out = fn(dt)
        out.to_local().sum().backward()
        self.assertIsNotNone(local.grad)

    @unittest.skipIf(not HAS_GPU, "standalone_compile requires GPU and triton")
    @skipIfXpu(msg="standalone_compile coverage is CUDA-only")
    @skip_if_lt_x_gpu(1)
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_aot_standalone_compile_dtensor_to_dtype_layout(self):
        # [FUNCTION DOC] GPU-only test: compiles with a backend using standalone_compile (aot
        # mode) a function that indexes a DTensor rope cache, moves it with to(device) and calls
        # to_local, via AOT compile under inference mode; asserts an artifact is produced.
        from torch._inductor import standalone_compile

        device = torch.device(self.device_type, 0)
        mesh = DeviceMesh(self.device_type, torch.arange(1))
        index = DTensor.from_local(
            torch.arange(2, device=device, dtype=torch.int64),
            mesh,
            [Replicate()],
            run_check=False,
        )
        rope_cache = DTensor.from_local(
            torch.randn(4, 4, device=device),
            mesh,
            [Replicate()],
            run_check=False,
        )

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend calling standalone_compile on the graph with dynamic
            # shapes from the graph and aot=True; returns its result.
            return standalone_compile(
                gm,
                example_inputs,
                dynamic_shapes="from_graph",
                options={"config_patches": {}},
                aot=True,
                donate_graph_module=True,
            )

        def index_to(index, rope_cache):
            # [FUNCTION DOC] HELPER: Indexes the rope cache with index, moves it to device,
            # returns the local tensor.
            return rope_cache[index].to(device=device).to_local()

        with torch._dynamo.config.patch(enable_aot_compile=True):
            compiled = torch.compile(index_to, backend=backend, fullgraph=True)
        with torch.inference_mode():
            artifact = compiled.aot_compile(((index, rope_cache), {}))

        self.assertIsNotNone(artifact)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW or TEST_XPU,
        "https://github.com/pytorch/pytorch/issues/178745",
    )
    @unittest.skipIf(not torch.accelerator.is_available(), "accelerator not available")
    def test_dtensor_basic_compile(self):
        # [FUNCTION DOC] Compiles a module holding a DTensor buffer that adds a DTensor built
        # from local input and returns to_local; checks the result equals eager and compares the
        # Dynamo graph and the AOT forward graph text against expected strings.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        param = torch.randn(4, 4)
        param_x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)

        class Foo(torch.nn.Module):
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the captured DTensor as buffer.
                super().__init__()
                self.buffer = torch.nn.Buffer(param_x)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the buffer to a DTensor made from x with the buffer
                # mesh and returns its local tensor.
                inter = self.buffer + DTensor.from_local(
                    x, self.buffer.device_mesh, [Shard(0)], run_check=False
                )
                return inter.to_local()

        m = Foo()
        inp = torch.randn(4, 4, dtype=torch.float64)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(m, backend=backend, fullgraph=True)(inp)
        self.assertEqual(res, m(inp))
        self.assertExpectedInline(
            str(backend.graphs[0].code).strip(),
            """\
def forward(self, L_self_buffers_buffer_ : torch.distributed.tensor.DTensor, L_self_buffers_buffer_device_mesh : torch.distributed.device_mesh.DeviceMesh, L_x_ : torch.Tensor):
    l_self_buffers_buffer_ = L_self_buffers_buffer_
    l_self_buffers_buffer_device_mesh = L_self_buffers_buffer_device_mesh
    l_x_ = L_x_
    from_local = torch.distributed.tensor._api.from_local(l_x_, l_self_buffers_buffer_device_mesh, [torch.distributed.tensor.placement_types.Shard(dim=0)], run_check = False);  l_x_ = l_self_buffers_buffer_device_mesh = None
    inter = l_self_buffers_buffer_ + from_local;  l_self_buffers_buffer_ = from_local = None
    to_local = inter.to_local();  inter = None
    return (to_local,)""",
        )
        self.assertExpectedInline(
            str(backend.fw_graphs[0].code).strip(),
            f"""\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    _to_copy = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float64, layout = torch.strided, device = device(type='{self.device_type}', index=0));  arg3_1 = None
    view = torch.ops.aten.view.default(_to_copy, [4, 4]);  _to_copy = None
    add = torch.ops.aten.add.Tensor(arg0_1, view);  arg0_1 = view = None
    view_1 = torch.ops.aten.view.default(add, [4, 4]);  add = None
    return (view_1,)""",
        )

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/179690")
    @skipIfXpu(msg="AssertionError: torch-xpu-ops: 2958")
    @unittest.skipIf(not torch.accelerator.is_available(), "accelerator not available")
    def test_dtensor_basic_export(self):
        # [FUNCTION DOC] Exports a module with a DTensor buffer via
        # dynamo_graph_capture_for_export, checks the captured code, then builds a joint graph
        # with aot_export_joint_with_descriptors under the tracing context and checks the joint
        # graph code against expected text.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        param = torch.randn(4, 4)
        param_x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)

        class Foo(torch.nn.Module):
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the captured DTensor as buffer.
                super().__init__()
                self.buffer = torch.nn.Buffer(param_x)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the buffer to a DTensor made from x and returns its
                # local tensor.
                inter = self.buffer + DTensor.from_local(
                    x, mesh, [Shard(0)], run_check=False
                )
                return inter.to_local()

        m = Foo()
        inp = (torch.randn(4, 4, dtype=torch.float64),)

        gm = dynamo_graph_capture_for_export(m)(*inp)
        tracing_context = gm.meta.get("tracing_context", None)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_self_buffers_buffer_ , L_self_buffers_buffer_device_mesh , L_x_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_self_buffers_buffer_ = L_self_buffers_buffer_
    l_self_buffers_buffer_device_mesh = L_self_buffers_buffer_device_mesh
    l_x_ = L_x_
    from_local = torch.distributed.tensor._api.from_local(l_x_, l_self_buffers_buffer_device_mesh, [torch.distributed.tensor.placement_types.Shard(dim=0)], run_check = False);  l_x_ = l_self_buffers_buffer_device_mesh = None
    inter = l_self_buffers_buffer_ + from_local;  l_self_buffers_buffer_ = from_local = None
    to_local = inter.to_local();  inter = None
    return self._dynamo_bytecode_unflatten((to_local,), _fn_args)""",
        )

        with tracing(tracing_context):
            with ExitStack() as stack:
                joint_with_descriptors = aot_export_joint_with_descriptors(
                    stack,
                    gm,
                    inp,
                )
                joint_gm = joint_with_descriptors.graph_module

        # During tracing, sharding propagation cache is skipped, so an extra dry run for
        # add is performed in _propagate_tensor_meta_non_cached, hence add_1 instead of add
        self.assertExpectedInline(
            str(joint_gm.code).strip(),
            f"""\
def forward(self, arg0_1, arg1_1, arg2_1):
    _to_copy = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float64, layout = torch.strided, device = device(type='{self.device_type}', index=0));  arg2_1 = None
    view = torch.ops.aten.view.default(_to_copy, [4, 4]);  _to_copy = None
    add = torch.ops.aten.add.Tensor(arg0_1, view);  arg0_1 = view = None
    view_1 = torch.ops.aten.view.default(add, [4, 4]);  add = None
    return (view_1,)""",
        )

    def test_placement_compile(self):
        # [FUNCTION DOC] Checks that predicate methods of placements (is_replicate, is_shard with
        # positional/keyword/None dim, is_partial) compile fullgraph and agree with eager for
        # Shard, Replicate, Partial and _StridedShard instances.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sums constants depending on which placement predicates hold;
            # raises RuntimeError if a Shard has a negative dim.
            a = 0
            if x.is_replicate():
                a += 1
            if x.is_shard():
                a += 2
                if x.dim < 0:
                    raise RuntimeError("dim < 0")
            if x.is_shard(0):
                a += 2
            if x.is_shard(dim=0):
                a += 2
            if x.is_shard(dim=None):
                a += 2
            if x.is_partial():
                a += 3
            return a

        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)
        split_factors = [2, 3, 4]
        for x in [Shard(0), Replicate(), Partial()] + [
            _StridedShard(0, split_factor=s) for s in split_factors
        ]:
            opt_fn = fn(x)
            compiled_out = compiled_fn(x)
            self.assertEqual(opt_fn, compiled_out)

    def test_strided_shard_view_unbacked_local(self):
        # [FUNCTION DOC] Regression test: a view on a _StridedShard DTensor whose local tensor
        # has an unbacked dim (from nonzero) must compile without leaking unbacked symbols;
        # checks the output placements stay the strided shard and the shape is flattened as
        # expected.
        # Regression for torchtitan #3409: during compile-time sharding
        # propagation for any _StridedShard view op,
        # _StridedShard.local_shard_size_and_offset used to materialize
        # offsets via .tolist() on a fake offsets tensor, allocating one
        # unbacked SymInt per element. These symbols never reached the
        # returned DTensor's tensor_meta, tripping the downstream
        # PendingUnbackedSymbolNotFound check. The bug triggers for any
        # _StridedShard view under compile regardless of whether the local
        # tensor has backed or unbacked dims; torch.nonzero (producing an
        # unbacked dim 0) is simply how torchtitan #3409 surfaced it.
        #
        # The bug lives in pure sharding-propagation arithmetic on rank 0
        # alone (no collectives are issued during fake-tensor view tracing),
        # so the FakeStore + world_size=2 setup used by this test class is
        # equivalent to the multi-rank scenario for the no-offset path
        # being exercised here.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        global_bs = 16
        slen = 4
        dim = 4
        split_factor = 8

        def fn(sentinel):
            # [FUNCTION DOC] HELPER: Builds a local tensor with unbacked first dim bounded by the
            # global batch size, wraps it as a _StridedShard DTensor and returns it viewed to 2D.
            nz = torch.nonzero(sentinel).flatten()
            n_unbacked = nz.size(0)
            torch._check(n_unbacked >= 1)
            torch._check(n_unbacked <= global_bs)
            local = torch.randn(n_unbacked, slen, dim)
            dt = DTensor.from_local(
                local,
                mesh,
                (_StridedShard(dim=0, split_factor=split_factor),),
                shape=(global_bs, slen, dim),
                stride=(slen * dim, dim, 1),
                run_check=False,
            )
            return dt.view(-1, dim)

        sentinel = torch.ones(global_bs, dtype=torch.int64)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(sentinel)
        self.assertEqual(
            out.placements,
            (_StridedShard(dim=0, split_factor=split_factor),),
        )
        self.assertEqual(out.shape, (global_bs * slen, dim))

    def test_device_mesh_compile(self):
        # [FUNCTION DOC] Compiles a function calling DeviceMesh size, get_coordinate and
        # get_group (positional and keyword) on a mesh passed as input; compares with eager
        # output.
        def fn(x: DeviceMesh):
            # [FUNCTION DOC] HELPER: Returns the sum of mesh size variants, the coordinate and
            # two process groups.
            # test size()
            a = x.size()
            b = x.size(0)
            c = x.size(mesh_dim=0)
            size = a + b + c
            # test get_coordinate()
            coord = x.get_coordinate()
            # test get_group()
            group0 = x.get_group(0)
            group1 = x.get_group(mesh_dim=1)
            return size, coord, group0, group1

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).unsqueeze(1))
        opt_fn = fn(mesh)

        compiled_out = torch.compile(fn, backend="aot_eager", fullgraph=True)(mesh)
        self.assertEqual(opt_fn, compiled_out)

    def test_device_mesh_input_to_dtensor_from_local(self):
        # [FUNCTION DOC] Checks that a DeviceMesh passed as an input is used by
        # DTensor.from_local inside compiled code: output device_mesh is the runtime mesh,
        # including for a second, different mesh object on a later call.
        """Test that DeviceMesh passed as input is correctly used in DTensor.from_local.

        This tests the opaque object remapping: when a DeviceMesh is passed as a
        function input and used inside DTensor.from_local, the output DTensor
        should use the runtime mesh, not a compile-time baked-in mesh.
        """

        def fn(local_tensor, mesh):
            # [FUNCTION DOC] HELPER: Builds a Shard(0) DTensor from the local tensor and given
            # mesh.
            # Use the passed-in mesh to create a DTensor
            return DTensor.from_local(local_tensor, mesh, [Shard(0)], run_check=False)

        # Create two different meshes to verify the output uses the runtime mesh
        mesh1 = DeviceMesh(self.device_type, torch.arange(self.world_size))

        local_tensor = torch.randn(4, 4)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        # First call with mesh1
        result1 = compiled_fn(local_tensor, mesh1)
        self.assertIsInstance(result1, DTensor)
        self.assertIs(result1.device_mesh, mesh1)

        # Create a second mesh (same shape but different object)
        mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size))
        self.assertIsNot(mesh1, mesh2)  # Different objects

        # Second call with mesh2 - the output should use mesh2, not mesh1
        result2 = compiled_fn(local_tensor, mesh2)
        self.assertIsInstance(result2, DTensor)
        self.assertIs(result2.device_mesh, mesh2)

    def test_get_local_rank_compile(self):
        # [FUNCTION DOC] Compiles functions using device_mesh.get_local_rank with a dimension
        # name, an integer dim and no argument, and compares each against eager.
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        def fn_with_str_arg(x):
            # [FUNCTION DOC] HELPER: Multiplies x by the local rank looked up by mesh dim name.
            local_rank = x.device_mesh.get_local_rank("dp")
            return x * local_rank

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        ref = fn_with_str_arg(x)

        opt_fn = torch.compile(fn_with_str_arg, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def fn_with_int_arg(x):
            # [FUNCTION DOC] HELPER: Multiplies x by the local rank looked up by integer mesh
            # dim.
            local_rank = x.device_mesh.get_local_rank(0)
            return x * local_rank

        ref2 = fn_with_int_arg(x)
        opt_fn2 = torch.compile(fn_with_int_arg, backend="aot_eager", fullgraph=True)
        res2 = opt_fn2(x)
        self.assertEqual(res2, ref2)

        def fn_without_arg(x):
            # [FUNCTION DOC] HELPER: Adds the local rank (no dim argument) to x.
            # will fail if device_mesh.ndim > 1
            local_rank = x.device_mesh.get_local_rank()
            return x + local_rank

        ref3 = fn_without_arg(x)
        opt_fn3 = torch.compile(fn_without_arg, backend="aot_eager", fullgraph=True)
        res3 = opt_fn3(x)
        self.assertEqual(res3, ref3)

    def test_dtensor_from_local_with_extracted_device_mesh_no_dce(self):
        # [FUNCTION DOC] Compiles a function that reads device_mesh from an input DTensor, builds
        # a new DTensor from it, and also returns an add on the input; with two distinct meshes,
        # the outputs must carry the matching input mesh (case where the add is kept).
        """Test that DTensor.from_local() works in compiled code when using
        a DeviceMesh extracted from another DTensor passed as input.

        This exercises:
        1. Accessing dtensor.device_mesh inside compiled code
        2. Passing that DeviceMesh to DTensor.from_local() with a different tensor
        3. Returning the new DTensor with correct DeviceMesh
        """
        mesh1 = DeviceMesh(self.device_type, torch.arange(self.world_size))
        mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(dt, local_tensor):
            # [FUNCTION DOC] HELPER: Returns a new Replicate DTensor using dt's mesh plus dt +
            # dt.
            # Extract DeviceMesh from input DTensor
            extracted_mesh = dt.device_mesh
            # Create new DTensor with the extracted mesh
            new_dt = DTensor.from_local(
                local_tensor,
                extracted_mesh,
                [Replicate()],
                run_check=False,
            )
            return new_dt, torch.add(dt, dt)

        # Create input DTensor
        input_local = torch.randn(4, 4)
        input_dt1 = DTensor.from_local(input_local, mesh1, [Shard(0)], run_check=False)
        input_dt2 = DTensor.from_local(input_local, mesh2, [Shard(0)], run_check=False)

        # Create the normal tensor to pass
        other_tensor = torch.randn(4, 4)

        # Compile and run
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        result1, _ = opt_fn(input_dt1, other_tensor)
        result2, _ = opt_fn(input_dt2, other_tensor)

        # Verify the new DTensor has the correct DeviceMesh
        self.assertIs(result1.device_mesh, mesh1)
        self.assertIs(result2.device_mesh, mesh2)

    def test_dtensor_from_local_with_extracted_device_mesh_dce(self):
        # [FUNCTION DOC] Same as the no_dce variant but the function returns only the new DTensor
        # (other work is dead code); output mesh must still be the matching input mesh.
        """Test that DTensor.from_local() works in compiled code when using
        a DeviceMesh extracted from another DTensor passed as input.

        This exercises:
        1. Accessing dtensor.device_mesh inside compiled code
        2. Passing that DeviceMesh to DTensor.from_local() with a different tensor
        3. Returning the new DTensor with correct DeviceMesh
        """
        mesh1 = DeviceMesh(self.device_type, torch.arange(self.world_size))
        mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(dt, local_tensor):
            # [FUNCTION DOC] HELPER: Returns a new Replicate DTensor using dt's mesh.
            # Extract DeviceMesh from input DTensor
            extracted_mesh = dt.device_mesh
            # Create new DTensor with the extracted mesh
            new_dt = DTensor.from_local(
                local_tensor,
                extracted_mesh,
                [Replicate()],
                run_check=False,
            )
            return new_dt

        # Create input DTensor
        input_local = torch.randn(4, 4)
        input_dt1 = DTensor.from_local(input_local, mesh1, [Shard(0)], run_check=False)
        input_dt2 = DTensor.from_local(input_local, mesh2, [Shard(0)], run_check=False)

        # Create the normal tensor to pass
        other_tensor = torch.randn(4, 4)

        # Compile and run
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        result1 = opt_fn(input_dt1, other_tensor)
        result2 = opt_fn(input_dt2, other_tensor)

        # Verify the new DTensor has the correct DeviceMesh
        self.assertIs(result1.device_mesh, mesh1)
        self.assertIs(result2.device_mesh, mesh2)

    def test_fakify_dtensor(self):
        # [FUNCTION DOC] Compiles an identity function over a DTensor input and output and
        # compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in DTensor as inputs/outputs to the function
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x unchanged.
            return x

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor(self):
        # [FUNCTION DOC] Compiles x * x + 2 over a Shard(0) DTensor input/output (fullgraph) and
        # compares against eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x * x + 2.
            return x * x + 2

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @unittest.skip("https://github.com/pytorch/pytorch/issues/171934")
    def test_dtensor_input_mutations(self):
        # [FUNCTION DOC] Skipped test (known issue link in decorator): compiles a function that
        # applies sin to x and mutates y in place, comparing outputs and x.grad with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x.sin() and adds 2 to y in place.
            out = x.sin()
            y.add_(2)
            return out

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        x_ref = DTensor.from_local(
            torch.randn(4), mesh, [Shard(0)], run_check=False
        ).requires_grad_(True)
        y_ref = DTensor.from_local(
            torch.randn(4), mesh, [Shard(0)], run_check=False
        ).requires_grad_(False)

        x = x_ref.clone().detach().requires_grad_(True)
        y = y_ref.clone().detach().requires_grad_(False)

        ref = fn(x_ref.clone(), y_ref)
        res = opt_fn(x.clone(), y)
        self.assertEqual(res, ref)

        ref.sum().backward()
        res.sum().backward()
        self.assertEqual(x.grad, x_ref.grad)

    @skipIfHpu
    def test_dtensor_dynamic(self):
        # [FUNCTION DOC] Marks dim 0 dynamic on a Shard(0) DTensor, compiles a multiply followed
        # by redistribute to Replicate and to_local indexing, and compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, redistributes to Replicate and returns row 0 of
            # local.
            return (
                torch.mul(x, x)
                .redistribute(device_mesh=x.device_mesh, placements=[Replicate()])
                .to_local()[0]
            )

        x = DTensor.from_local(
            torch.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_slice(self):
        # [FUNCTION DOC] Compiles with dynamic=True a function that splits a DTensor with
        # tensor_split into two, redistributes each to Replicate and takes row 0; compares eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a list with row 0 of each Replicate-redistributed
            # chunk.
            return [
                t.redistribute(
                    device_mesh=x.device_mesh, placements=[Replicate()]
                ).to_local()[0]
                for t in torch.tensor_split(x, 2)
            ]

        x = DTensor.from_local(
            torch.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1981")
    def test_dtensor_dynamic_loss_parallel_log_softmax(self):
        # [FUNCTION DOC] Under loss_parallel, compiles (dynamic) a log_softmax over the last dim
        # of a Shard(1) DTensor followed by redistribute to Replicate; compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Float32 log_softmax on last dim, redistribute to Replicate,
            # row 0.
            t = torch.nn.functional.log_softmax(x, x.ndim - 1, dtype=torch.float32)
            return t.redistribute(
                device_mesh=x.device_mesh, placements=[Replicate()]
            ).to_local()[0]

        with loss_parallel():
            x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(1)], run_check=False)
            ref = fn(x)

            opt_fn = torch.compile(
                fn, backend="aot_eager", fullgraph=True, dynamic=True
            )
            res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dynamic_cat(self):
        # [FUNCTION DOC] Compiles a function that concatenates two DTensors on dim 0 (one marked
        # dynamic), redistributes to Replicate and indexes row 0; compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in tuple of DTensors as
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Concatenates x and y, redistributes to Replicate, returns
            # row 0.
            return (
                torch.cat((x, y), dim=0)
                .redistribute(device_mesh=x.device_mesh, placements=[Replicate()])
                .to_local()[0]
            )

        x = DTensor.from_local(
            torch.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        y = DTensor.from_local(
            torch.rand(4, 4, requires_grad=True), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    def test_dtensor_dynamic_recompiles(self):
        # [FUNCTION DOC] Checks recompilation behavior with dynamic DTensor shapes: different
        # sizes reuse one graph (frame count 1), while a function branching on size recompiles
        # once when crossing the guarded threshold (count 2 after a large size).
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def inp(*shape):
            # [FUNCTION DOC] HELPER: Builds a Shard(0) DTensor of the given shape with both dims
            # marked dynamic.
            param = torch.randn(*shape, requires_grad=True)
            x = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        def run(func, *shape):
            # [FUNCTION DOC] HELPER: Runs func on a new DTensor and calls backward on the summed
            # result.
            res = func(inp(*shape))
            res.sum().backward()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Squares x and returns the local tensor.
            y = x * x
            return y.to_local()

        run(f, 4, 4)
        run(f, 6, 8)
        run(f, 10, 10)
        self.assertEqual(cnt.frame_count, 1)

        # sanity check that shape guard recompiles are still handled
        @torch.compile(backend=cnt, fullgraph=True)
        def g(x):
            # [FUNCTION DOC] HELPER: Squares x if size(0) <= 16, else doubles it; returns local
            # tensor.
            if x.size(0) <= 16:
                y = x * x
            else:
                y = x + x
            return y.to_local()

        cnt.clear()
        run(g, 4, 4)
        run(g, 8, 8)
        self.assertEqual(cnt.frame_count, 1)
        run(g, 64, 8)
        self.assertEqual(cnt.frame_count, 2)

    @unittest.skipIf(not HAS_GPU, "requires GPU for RNG support")
    def test_dtensor_unbacked_matmuls(self):
        # [FUNCTION DOC] GPU-only. Replaces the process group with a fake world of 4 and a 2x2
        # mesh, then for many placement combinations compiles torch.mm on random DTensors with
        # unbacked dims, checks output placements where given, and re-runs with uneven sizes
        # asserting one compiled frame and eager-matching output shape.
        from torch.distributed.tensor import randn as d_randn

        # use 2x2 mesh for testing
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        device_mesh = init_device_mesh(self.device_type, (2, 2))

        def test_placements(x_placements, y_placements, out_placements=None):
            # [FUNCTION DOC] HELPER: Builds 64x64 DTensors with the given placements and unbacked
            # dims, compiles mm (fullgraph), optionally asserts output placements, then runs
            # uneven shapes (20x17 @ 17x5) checking no recompile and matching output shape.
            # create DTensors with unbacked outer/inner sizes
            x_dt = d_randn(64, 64, device_mesh=device_mesh, placements=x_placements)
            y_dt = d_randn(64, 64, device_mesh=device_mesh, placements=y_placements)
            for i in range(2):
                torch._dynamo.decorators.mark_unbacked(x_dt, i)
                torch._dynamo.decorators.mark_unbacked(y_dt, i)

            # full-graph capture
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            fn = torch.compile(torch.mm, backend=cnt, fullgraph=True)
            out = fn(x_dt, y_dt)

            # check output placements
            if out_placements is not None:
                self.assertEqual(out.placements, out_placements)

            # test on uneven shardings
            dx = d_randn(20, 17, device_mesh=device_mesh, placements=x_placements)
            dy = d_randn(17, 5, device_mesh=device_mesh, placements=y_placements)
            out, eager_out = fn(dx, dy), torch.mm(dx, dy)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(out.shape, eager_out.shape)

        test_placements(
            (Replicate(), Replicate()),
            (Replicate(), Replicate()),
            (Replicate(), Replicate()),
        )
        test_placements(
            (Replicate(), Shard(1)), (Replicate(), Shard(0)), (Replicate(), Partial())
        )
        test_placements(
            (Replicate(), Shard(0)), (Replicate(), Replicate()), (Replicate(), Shard(0))
        )

        # note: just check that these run; it's reasonable for the output placements to change,
        # depending on how tie-breaking is done.
        test_placements((Partial(), Partial()), (Partial(), Partial()))
        test_placements((Partial(), Partial()), (Shard(0), Replicate()))
        test_placements((Replicate(), Shard(0)), (Replicate(), Shard(0)))
        test_placements((Shard(0), Shard(1)), (Shard(1), Shard(0)))
        test_placements((Partial(), Replicate()), (Shard(0), Shard(0)))

    @unittest.skipIf(not HAS_GPU, "requires GPU for RNG support")
    def test_dtensor_matmul_zero_size_shards(self):
        # [FUNCTION DOC] GPU-only. On a fake 2x2 mesh compiles mm, addmm, bmm and baddbmm with
        # unbacked dims, then runs with a size (3) smaller than the world size so local shards
        # are empty, asserting shapes match eager and no recompilation occurs.
        from torch.distributed.tensor import randn as d_randn

        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        device_mesh = init_device_mesh(self.device_type, (2, 2))

        # Placements that shard on the contracting/output dims so that
        # zero-size local shards can arise at runtime.
        px, py = (Replicate(), Shard(1)), (Replicate(), Shard(0))
        # batched variants: shard the non-batch dims the same way
        bpx, bpy = (Replicate(), Shard(2)), (Replicate(), Shard(1))

        def _run(torch_fn, make_args, unbacked_dims=None):
            # [FUNCTION DOC] HELPER: Compiles torch_fn once with args from make_args(64) (marking
            # dims unbacked, all dims or those listed per argument index), then runs with
            # make_args(3), asserting the output shape equals eager and frame_count stays 1.
            """Compile once on large tensors, then re-run on small/zero-size."""
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            fn = torch.compile(torch_fn, backend=cnt, fullgraph=True)

            # initial compile with large shapes
            args0 = make_args(64)
            for arg_idx, a in enumerate(args0):
                if unbacked_dims is not None:
                    for i in unbacked_dims.get(arg_idx, []):
                        torch._dynamo.decorators.mark_unbacked(a, i)
                else:
                    for i in range(a.dim()):
                        torch._dynamo.decorators.mark_unbacked(a, i)
            fn(*args0)

            # re-run with m=3 which produces zero-size local shards
            # (3 < world_size=4), verifying no recompilation
            for m in [3]:
                args = make_args(m)
                c_out = fn(*args)
                eager_out = torch_fn(*args)
                self.assertEqual(c_out.shape, eager_out.shape)
                self.assertEqual(cnt.frame_count, 1)

        # mm: (m,k) @ (k,n) -> (m,n)
        _run(
            torch.mm,
            lambda m: (
                d_randn(m, 64, device_mesh=device_mesh, placements=px),
                d_randn(64, 64, device_mesh=device_mesh, placements=py),
            ),
        )

        # addmm: bias(n) + (m,k) @ (k,n) -> (m,n)
        # Only mark mat1 dim 0 (m) as unbacked; bias dims must be backed
        # because sharding prop checks bias_shape == 1 for broadcast.
        _run(
            torch.addmm,
            lambda m: (
                d_randn(
                    64, device_mesh=device_mesh, placements=(Replicate(), Replicate())
                ),
                d_randn(m, 64, device_mesh=device_mesh, placements=px),
                d_randn(64, 64, device_mesh=device_mesh, placements=py),
            ),
            unbacked_dims={1: [0]},
        )

        # bmm: (b,m,k) @ (b,k,n) -> (b,m,n)
        _run(
            torch.bmm,
            lambda m: (
                d_randn(2, m, 64, device_mesh=device_mesh, placements=bpx),
                d_randn(2, 64, 64, device_mesh=device_mesh, placements=bpy),
            ),
            unbacked_dims={0: [1]},
        )

        # baddbmm: bias(b,1,n) + (b,m,k) @ (b,k,n) -> (b,m,n)
        # Use bias with m=1 (broadcast) so the bias shape doesn't change
        # across m values, avoiding recompilation and unbacked broadcast guards.
        _run(
            torch.baddbmm,
            lambda m: (
                d_randn(
                    2,
                    1,
                    64,
                    device_mesh=device_mesh,
                    placements=(Replicate(), Replicate()),
                ),
                d_randn(2, m, 64, device_mesh=device_mesh, placements=bpx),
                d_randn(2, 64, 64, device_mesh=device_mesh, placements=bpy),
            ),
            unbacked_dims={1: [1]},
        )

    def test_dtensor_matmul_cost_hint_or_upper_bound(self):
        # [FUNCTION DOC] Uses a fake world of 4 and Shard(0) @ Shard(1) DTensors with unbacked
        # dims. Checks the matmul strategy choice follows size upper bounds set with _check
        # (output Shard(1) or Shard(0)) and, without bounds, follows hint_override values.
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        device_mesh = init_device_mesh(self.device_type, (4,))

        x_dt = DTensor.from_local(
            torch.randn(8, 8),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        y_dt = DTensor.from_local(
            torch.randn(8, 8),
            device_mesh=device_mesh,
            placements=[Shard(1)],
        )
        torch._dynamo.decorators.mark_unbacked(x_dt, 0)
        torch._dynamo.decorators.mark_unbacked(y_dt, 1)

        # large k dim tells compiler it's cheaper to all-gather on x
        def f1(x, y):
            # [FUNCTION DOC] HELPER: Asserts upper bounds on x rows and y cols, returns x @ y.
            torch._check(x.size(0) <= 32)
            torch._check(y.size(1) <= 16384)
            return x @ y

        out = torch.compile(f1, backend="aot_eager", fullgraph=True)(x_dt, y_dt)
        self.assertEqual(out.placements, (Shard(1),))

        # for the reverse, all-gather on y
        def f2(x, y):
            # [FUNCTION DOC] HELPER: Asserts the reverse upper bounds, returns x @ y.
            torch._check(x.size(0) <= 16384)
            torch._check(y.size(1) <= 32)
            return x @ y

        out = torch.compile(f2, backend="aot_eager", fullgraph=True)(x_dt, y_dt)
        self.assertEqual(out.placements, (Shard(0),))

        # specifying hint_override also determines the strategy
        def f3(x, y):
            # [FUNCTION DOC] HELPER: Returns x @ y.
            return x @ y

        x_dt = x_dt + 1
        y_dt = y_dt + 1
        torch._dynamo.decorators.mark_unbacked(x_dt, 0, hint_override=16)
        torch._dynamo.decorators.mark_unbacked(x_dt, 1, hint_override=1024)

        out = torch.compile(f3, backend="aot_eager", fullgraph=True)(x_dt, y_dt)
        self.assertEqual(out.placements, (Shard(1),))

    def test_dtensor_requires_grad_recompile(self):
        # [FUNCTION DOC] Checks that calling a compiled function with a DTensor without and then
        # with requires_grad triggers a recompile (frame_count 2).
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Squares x and returns the local tensor.
            y = x * x
            return y.to_local()

        full_x = torch.randn(8, 8, requires_grad=False)
        x = distribute_tensor(full_x, mesh, [Shard(0)])
        f(x)

        full_x = torch.randn(8, 8, requires_grad=True)
        x = distribute_tensor(full_x, mesh, [Shard(0)])
        f(x)

        self.assertEqual(cnt.frame_count, 2)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/177751")
    @with_comms
    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_dtensor_requires_grad_intermediate_backward(self):
        # [FUNCTION DOC] Skipped on Linux (issue link in decorator). Compiles, with autograd op
        # tracing, a function that runs backward on an intermediate requiring grad and returns
        # its grad; compares the full tensor with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, marks it requires_grad, backprops a scaled sum
            # and returns the intermediate's grad.
            y = x * 2
            y.requires_grad_()
            loss = (y * 3).sum()
            loss.backward()
            return y.grad

        full_x = torch.randn(8, 8)
        x = distribute_tensor(full_x, mesh, [Shard(0)])

        ref = fn(x.full_tensor())
        result = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, result.full_tensor())

    def test_dtensor_attribute_access_on_intermediate(self):
        # [FUNCTION DOC] Compiles a function that branches on the placements of an intermediate
        # DTensor and adds to its _local_tensor; compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 2 to the local tensor if the first placement is a
            # shard, else 3.
            tmp = x * 2
            if tmp.placements[0].is_shard():
                return tmp._local_tensor + 2
            else:
                return tmp._local_tensor + 3

        x = DTensor.from_local(torch.ones(4), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_constructor_w_graph_break(self):
        # [FUNCTION DOC] Builds a DTensor via the constructor from a DTensorSpec inside a
        # function that prints (graph break) and runs it eagerly and with the eager backend; no
        # assertions beyond no error.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        x = torch.randn(64, 32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(), Shard(0)),
            tensor_meta=TensorMeta(
                shape=torch.Size([128, 32]), stride=(32, 1), dtype=x.dtype
            ),
        )

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            # [FUNCTION DOC] HELPER: Prints a message then constructs a DTensor from x and the
            # spec.
            print("graph break!")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        fn(x)
        torch.compile(fn, backend="eager")(x)

    def test_dtensor_constructor_w_dynamo_disable(self):
        # [FUNCTION DOC] Constructs a DTensor from a spec inside a function decorated with dynamo
        # disable (non-recursive); eager and compiled calls must give equal results.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        x = torch.randn(32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(),),
            tensor_meta=TensorMeta(shape=torch.Size([32]), stride=(1,), dtype=x.dtype),
        )

        @torch._dynamo.disable(recursive=False)
        def fn(x):
            # [FUNCTION DOC] HELPER: Prints a message then constructs a DTensor from x and the
            # spec.
            print("foo")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        out = fn(x)
        out2 = torch.compile(fn, backend="eager")(x)
        self.assertEqual(out, out2)

    def test_dtensor_noncontiguous_output(self):
        # [FUNCTION DOC] Compiles a linear on permuted DTensors whose output is a non-contiguous
        # permutation, then runs backward on the contiguous output; checks no error.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Permutes x, applies the linear op with y and z and permutes
            # back.
            x_transposed = x.permute(0, 2, 1).contiguous()
            tmp = torch._C._nn.linear(x_transposed, y, z)
            return tmp.permute(0, 2, 1)

        x_inner = torch.randn(4, 16, 4, requires_grad=True)
        y_inner = torch.randn(4, 16, requires_grad=True)
        z_inner = torch.randn(4, requires_grad=True)
        x = DTensor.from_local(x_inner, mesh, [Shard(1)], run_check=False)
        y = DTensor.from_local(y_inner, mesh, [Shard(1)], run_check=False)
        z = DTensor.from_local(z_inner, mesh, [Replicate()], run_check=False)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y, z)
        out.contiguous().sum().backward()

    def test_dynamo_dtensor_from_local(self):
        # [FUNCTION DOC] Compiles functions that create a DTensor from local (mesh and placements
        # positional, then as keywords) and return to_local plus 2; checks backward works,
        # results equal eager and frame counts are 1 and then 2.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # create DTensor inside fn and run some compute
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x into a Replicate DTensor, returns local tensor plus
            # 2.
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            return dt.to_local() + 2

        # below is the op approach for reference
        # from torch.distributed._tensor.api import _FromTorchTensor
        # def from_local_tensor(x):
        #     return _FromTorchTensor.apply(x, mesh, [Replicate()], False)

        # _dt_lib_def = torch.library.Library("dtensor", "DEF")
        # _dt_lib_def.define("from_local(Tensor self) -> Tensor")

        # _dt_lib_impl = torch.library.Library("dtensor", "IMPL")
        # _dt_lib_impl.impl("from_local", from_local_tensor, "Autograd")

        x = torch.ones(1, requires_grad=True)
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        # backward should work as well
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # test if user calls from_local with mesh/placements as kwargs and that should still work
        def from_local_kwargs_fn(x):
            # [FUNCTION DOC] HELPER: Same as fn but passes device_mesh and placements as
            # keywords.
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            return dt.to_local() + 2

        ref = from_local_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(from_local_kwargs_fn, backend=cnt, fullgraph=True)
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamo_dtensor_from_local_dynamic_shapes(self):
        # [FUNCTION DOC] Checks from_local with explicit shape and stride under three dynamic
        # cases (all dims dynamic, only sizes dynamic with static stride, mixed dynamic/static
        # sizes and strides): results equal eager, backward works, one frame each.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # Case 1: all dims dynamic
        def fn(x):
            # [FUNCTION DOC] HELPER: Case 1: from_local with shape and stride of x; returns local
            # + 2.
            dt = DTensor.from_local(
                x,
                mesh,
                [Replicate()],
                run_check=False,
                shape=x.shape,
                stride=x.stride(),
            )
            return dt.to_local() + 2

        inp = torch.randn(4, 6, requires_grad=True)
        ref = fn(inp)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = torch.compile(fn, backend=cnt, fullgraph=True, dynamic=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # Case 2: only sizes are dynamic, strides are static
        def fn(x):
            # [FUNCTION DOC] HELPER: Case 2: from_local with x.shape and static stride (1,);
            # returns local + 2.
            dt = DTensor.from_local(
                x, mesh, [Replicate()], run_check=False, shape=x.shape, stride=(1,)
            )
            return dt.to_local() + 2

        inp = torch.randn(4, requires_grad=True)
        torch._dynamo.mark_dynamic(inp, 0)
        ref = fn(inp)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = torch.compile(fn, backend=cnt, fullgraph=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # Case 3: both sizes and strides have a mix of dynamic and static dims
        def fn(x):
            # [FUNCTION DOC] HELPER: Case 3: from_local with mixed shape/stride entries; returns
            # local + 2.
            dt = DTensor.from_local(
                x,
                mesh,
                [Replicate()],
                run_check=False,
                shape=(x.shape[0], x.shape[1], 2),
                stride=(x.stride()[0], 2, 1),
            )
            return dt.to_local() + 2

        inp = torch.randn(4, 6, 2, requires_grad=True)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)
        ref = fn(inp)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        res = torch.compile(fn, backend=cnt, fullgraph=True)(inp)
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

    def test_dynamo_dtensor_recompile(self):
        # [FUNCTION DOC] Checks guards for DTensor inputs with dynamic=False: a second DTensor
        # with the same placements reuses the graph, while different placements (Shard(1))
        # recompile (frame_count 2).
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x * x.
            return torch.mul(x, x)

        x = DTensor.from_local(torch.rand(2, 2), mesh, [Shard(0)], run_check=False)
        x2 = DTensor.from_local(torch.rand(2, 2), mesh, [Shard(0)], run_check=False)
        x3 = DTensor.from_local(torch.rand(2, 2), mesh, [Shard(1)], run_check=False)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True, dynamic=False)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(fn(x3), opt_fn(x3))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfHpu
    def test_dtensor_partial_placement_redistribute_unbalanced_correct_strides(self):
        # [FUNCTION DOC] Replaces the process group with a fake rank 3 / world 4 group and a mesh
        # of ranks 1 and 3; redistributes a Partial DTensor to Shard(1) and checks that a fake
        # tensor made from the result has the same shape and stride as the DTensor and its local.
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        placement = Shard(1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Redistributes x to the enclosing placement on the enclosing
            # mesh.
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=4)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = torch.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        # tmp_dt has an inner, non-contiguous tensor, and is an autograd non-leaf
        tmp_dt = fn(x_dt)
        fake_mode = torch._subclasses.FakeTensorMode()
        tmp_dt_fake = fake_mode.from_tensor(tmp_dt)
        self.assertEqual(tmp_dt.shape, tmp_dt_fake.shape)
        self.assertEqual(tmp_dt.stride(), tmp_dt_fake.stride())
        self.assertEqual(tmp_dt._local_tensor.shape, tmp_dt_fake._local_tensor.shape)
        self.assertEqual(
            tmp_dt._local_tensor.stride(), tmp_dt_fake._local_tensor.stride()
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent(self):
        # [FUNCTION DOC] GPU-only. On a fake rank-3 / world-4 group and mesh of ranks 1 and 3,
        # compiles with the default backend a redistribute of a Partial DTensor to Shard(1)
        # (whose local tensor ends non-contiguous), then runs backward; checks tangents are made
        # contiguous so Inductor stride asserts do not fail.
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        # When this tensor is a fwd graph output,
        # AOTAutograd needs to make sure we trace the backward
        # with a contiguous tangent
        placement = Shard(1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Redistributes x to the enclosing placement on the enclosing
            # mesh.
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=4)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = torch.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        out_dt = torch.compile(fn)(x_dt)
        # If we don't properly contiguify our traced tangents,
        # this fails with an inductor stride assert
        out_dt.to_local().sum().backward()

    def test_dynamo_to_local_grad_placements_sequence(self):
        # [FUNCTION DOC] Compiles to_local with grad_placements given as a PytreeTuple sequence
        # of placements (closure-captured DTensor); output must equal eager.
        placements = PytreeTuple([Shard(0)])

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the closure DTensor's local tensor plus 2, using the
            # PytreeTuple grad placements. The argument x is unused.
            return dt.to_local(grad_placements=placements) + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_grad_placements_sequence_intermediate(self):
        # [FUNCTION DOC] Like the previous test but the PytreeTuple is created inside the
        # compiled function; compares with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a PytreeTuple of placements and returns the closure
            # DTensor's local tensor plus 2. The argument x is unused.
            placements = PytreeTuple([Shard(0)])
            return dt.to_local(grad_placements=placements) + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_from_local_grad_placements_sequence_intermediate(self):
        # [FUNCTION DOC] Compiles DTensor.from_local with the placements passed as a keyword that
        # is a PytreeTuple (not a list); compares to_local plus 2 with eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        placements = PytreeTuple(Shard(0))

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x with the closure PytreeTuple placements, returns
            # local + 2.
            dt = DTensor.from_local(
                x,
                mesh,
                placements=placements,
                run_check=False,
            )
            return dt.to_local() + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)

        out_ref = fn(x)
        out_test = fn_opt(x)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_from_local_grad_placements_sequence_intermediate_as_args(self):
        # [FUNCTION DOC] Same as the previous test but placements is passed positionally.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        placements = PytreeTuple(Shard(0))

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x with the closure PytreeTuple placements
            # (positional), returns local + 2.
            dt = DTensor.from_local(
                x,
                mesh,
                placements,
                run_check=False,
            )
            return dt.to_local() + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)

        out_ref = fn(x)
        out_test = fn_opt(x)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_kwargs(self):
        # [FUNCTION DOC] Compiles to_local with grad_placements passed as a list keyword; output
        # must equal eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the closure DTensor's local tensor (grad placement
            # Shard 0) plus 2. The argument x is unused.
            return dt.to_local(grad_placements=[Shard(0)]) + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_custom_partial_placement(self):
        # [FUNCTION DOC] Checks that to_local compiles when grad_placements contains a custom
        # Partial subclass (which cannot be turned into a Python constant); output equals eager.
        """Test that to_local works with custom Partial subclasses in grad_placements.

        This tests the fix for custom placement objects like _ScaledPartial that
        cannot be converted to Python constants via as_python_constant().
        """

        # Define a custom Partial subclass similar to _ScaledPartial in torchtitan
        class ScaledPartial(Partial):
            # [CLASS DOC] HELPER: Partial subclass with sum reduction that also stores a divide
            # factor and divides the tensor by it in place before delegating the reduction to
            # Partial.
            def __init__(self, reduction_divide_factor: float):
                # [FUNCTION DOC] HELPER: Stores reduction_divide_factor and initializes Partial
                # with sum.
                self.reduction_divide_factor = reduction_divide_factor
                super().__init__(reduce_op="sum")

            def _reduce_value(
                self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Divides the tensor in place by the stored factor, then
                # returns the parent class reduction of the value over the mesh dim.
                tensor.div_(self.reduction_divide_factor)
                return super()._reduce_value(tensor, mesh, mesh_dim)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        scaled_partial = ScaledPartial(reduction_divide_factor=2.0)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.to_local with the custom Partial as grad
            # placement, plus 2.
            return x.to_local(grad_placements=[scaled_partial]) + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_kwargs_forward_hook(self):
        # [FUNCTION DOC] Registers a forward hook on a Linear with DTensor parameters; the hook
        # calls to_local with grad_placements and rebuilds a DTensor. Compiled module output must
        # equal eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fw_hook(module, inp, out):
            # [FUNCTION DOC] HELPER: Forward hook: takes the output's local tensor (grad
            # placements set to its placements), adds 2, and re-wraps as a DTensor on the mesh.
            tmp = out.to_local(grad_placements=out.placements) + 2
            return DTensor.from_local(tmp, mesh, out.placements, run_check=False)

        mod = torch.nn.Linear(4, 4)
        mod.register_forward_hook(fw_hook)
        mod.weight = torch.nn.Parameter(
            DTensor.from_local(mod.weight, mesh, [Replicate()], run_check=False)
        )
        mod.bias = torch.nn.Parameter(
            DTensor.from_local(mod.bias, mesh, [Replicate()], run_check=False)
        )
        opt_mod = torch.compile(mod, backend="aot_eager", fullgraph=True)

        x = torch.ones(4, 4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = mod(dt)
        out_test = opt_mod(dt)
        self.assertEqual(out_ref, out_test)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_different_gradient_placement(self):
        # [FUNCTION DOC] GPU-only. Compiles with Inductor a permute / layer_norm / permute on
        # DTensors, then multiplies the output with the input and runs backward; checks that
        # differing gradient placements do not break compile (no error expected).
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Permutes x, makes it contiguous, applies layer_norm with
            # weight y and bias z over the last size-4 dim, permutes back.
            permute = x.permute(0, 2, 1)
            x.device_mesh
            permute2 = permute.contiguous()
            layer_norm = torch.nn.functional.layer_norm(permute2, (4,), y, z, 1e-05)
            out = layer_norm.permute(0, 2, 1)
            return out

        x = torch.randn(4, 2, 4, requires_grad=True, device=self.device_type)
        x_dt = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        y = torch.randn(4, requires_grad=True, device=self.device_type)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        z = torch.randn(4, requires_grad=True, device=self.device_type)
        z_dt = DTensor.from_local(z, mesh, [Replicate()], run_check=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt, y_dt, z_dt)
        out_dt = torch.matmul(tmp_dt, x_dt).permute(0, 2, 1)
        out_dt.sum().backward()

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/180656"
    )
    def test_dynamo_dtensor_from_local_redistribute(self):
        # [FUNCTION DOC] Skipped on Linux or slow runs (issue link). Compiles a function that
        # builds a DTensor, redistributes to Replicate and returns local + 2, for positional and
        # keyword arguments; checks the Dynamo graph text, equality with eager and that the graph
        # passes check_cacheable.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x as Shard(0) DTensor, redistributes to Replicate
            # (positional), returns local + 2.
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)
        self.assertExpectedInline(
            str(backend.graphs[0].code).strip(),
            """\
def forward(self, L_x_ : torch.Tensor, L_mesh_ : torch.distributed.device_mesh.DeviceMesh):
    l_x_ = L_x_
    l_mesh_ = L_mesh_
    dt = torch.distributed.tensor._api.from_local(l_x_, l_mesh_, [torch.distributed.tensor.placement_types.Shard(dim=0)], run_check = False);  l_x_ = None
    redistribute = dt.redistribute(l_mesh_, [torch.distributed.tensor.placement_types.Replicate()]);  dt = l_mesh_ = None
    to_local = redistribute.to_local();  redistribute = None
    add = to_local + 2;  to_local = None
    return (add,)""",
        )
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def redistribute_kwargs_fn(x):
            # [FUNCTION DOC] HELPER: Same as fn but redistribute is given device_mesh and
            # placements as keywords.
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return (
                dt.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
                + 2
            )

        x = torch.ones(1)
        ref = redistribute_kwargs_fn(x)
        backend = AotEagerAndRecordGraphs()
        opt_kwargs_fn = torch.compile(
            redistribute_kwargs_fn, backend=backend, fullgraph=True
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)
        self.assertExpectedInline(
            str(backend.graphs[0].code).strip(),
            """\
def forward(self, L_x_ : torch.Tensor, L_mesh_ : torch.distributed.device_mesh.DeviceMesh):
    l_x_ = L_x_
    l_mesh_ = L_mesh_
    dt = torch.distributed.tensor._api.from_local(l_x_, l_mesh_, [torch.distributed.tensor.placement_types.Shard(dim=0)], run_check = False);  l_x_ = None
    redistribute = dt.redistribute(device_mesh = l_mesh_, placements = [torch.distributed.tensor.placement_types.Replicate()]);  dt = l_mesh_ = None
    to_local = redistribute.to_local();  redistribute = None
    add = to_local + 2;  to_local = None
    return (add,)""",
        )

        # This should not throw a BypassAOTAutogradCache error
        check_cacheable(backend.graphs[0])

    @skipIfHpu
    def test_dynamo_dtensor_from_local_redistribute_async(self):
        # [FUNCTION DOC] Compiles a function using redistribute with async_op=True, waiting on
        # the result when it is an AsyncCollectiveTensor; output equals eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps x as DTensor, async redistributes to Replicate, waits
            # if the local result is an AsyncCollectiveTensor and returns it.
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            out = dt.redistribute(mesh, [Replicate()], async_op=True).to_local()
            if isinstance(out, AsyncCollectiveTensor):
                return out.wait()
            else:
                return out

        x = torch.ones(1)
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dont_recompile_on_same_placement_devicemesh(self):
        # [FUNCTION DOC] Checks that rebuilding an equal mesh and placement between calls does
        # not recompile, while a different placement does: two total frames for the Shard and
        # Partial placements.
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a DTensor from x using closure mesh and placement;
            # returns nothing.
            DTensor.from_local(x, mesh, [placement], run_check=False)

        x = torch.ones(4, 4, requires_grad=True)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        placement = Shard(1)
        fn(x)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        placement = Shard(1)
        # no recompile, placement is unchanged
        fn(x)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        placement = Partial()
        # recompile since placement is different
        fn(x)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        placement = Partial()
        # no recompile, placement is unchanged
        fn(x)

        # 2 total frames (one for Partial(), one for Shard())
        self.assertEqual(cnt.frame_count, 2)

    def test_dtensor_dynamo_device_mesh_attrs(self):
        # [FUNCTION DOC] Compiles with the eager backend a function branching on the DTensor
        # device mesh device_type attribute; output equals eager.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x_dt):
            # [FUNCTION DOC] HELPER: Adds 1 if the mesh device type matches the test device, else
            # 2.
            if x_dt.device_mesh.device_type == f"{self.device_type}":
                return x_dt + 1
            else:
                return x_dt + 2

        x = torch.ones(4, 4)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        ref = fn(x_dt)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_dt)
        self.assertEqual(ref, res)

    @skipIfHpu
    def test_graph_input_is_async(self):
        # [FUNCTION DOC] Feeds an AsyncCollectiveTensor (from async redistribute) to a function
        # compiled with the recording aot_autograd backend and checks the forward graph has a
        # plain input and no wait_tensor op.
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().sin().
            return x.sin().sin()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x = torch.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        x2 = x_dt.redistribute(mesh, [Replicate()], async_op=True)
        x2 = x2.to_local()
        self.assertTrue(isinstance(x2, AsyncCollectiveTensor))
        opt_fn(x2)
        # ACTs are unwrapped before AOT autograd tracing in
        # process_inputs, so the graph receives a plain tensor with
        # no wait_tensor op.
        self.assertExpectedInline(
            str(fw_graph_cell[0]).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    sin_1 = torch.ops.aten.sin.default(sin);  sin = None
    return (sin_1,)""",
        )

    @skipIfTorchDynamo()
    def test_unwrap_async_collective_tensor_tangent(self):
        # [FUNCTION DOC] Compiles a clone of a TwoTensor and runs backward with plain TwoTensor
        # gradients and with gradients whose inner tensors are AsyncCollectiveTensor; checks only
        # that no error is raised (skipped under Dynamo).
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.clone().
            return x.clone()

        ref_x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        ref_y = fn(ref_x)

        ref_y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        fn_comp = torch.compile(fn, fullgraph=True)

        x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        x2 = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        y2.backward(
            gradient=TwoTensor(
                AsyncCollectiveTensor(torch.randn(2, 3)),
                AsyncCollectiveTensor(torch.randn(2, 3)),
            )
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_partial_placement_graph_output(self):
        # [FUNCTION DOC] GPU-only. Compiles with Inductor an add of a Partial DTensor returning a
        # Partial graph output, then matmul with a Replicate DTensor and backward; no error
        # expected.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x + x.
            return x + x

        x = torch.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Partial()], run_check=False)

        y = torch.randn(4, 4, requires_grad=True)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt)
        out_dt = torch.matmul(tmp_dt, y_dt)
        out_dt.sum().backward()

    @unittest.skipIf(
        torch._inductor.config.triton.native_matmul, "Matmul is now generated"
    )
    def _test_tp_compile_comm_reordering(self):
        # [FUNCTION DOC] HELPER: Shared body for the comm reordering tests (not itself collected;
        # skipped when native matmul is on). Builds a fake attention/transformer, applies
        # sequence/tensor parallel on a 2-way mesh, compiles with Inductor, runs backward,
        # asserts one frame and that generated code has all_gather, wait_tensor then a matmul in
        # order.
        class FakeAttention(nn.Module):
            # [CLASS DOC] HELPER: Fake attention with four 16x16 linear layers (wq, wk, wv, wo).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the four linear layers.
                super().__init__()
                self.wq = nn.Linear(16, 16)
                self.wk = nn.Linear(16, 16)
                self.wv = nn.Linear(16, 16)
                self.wo = nn.Linear(16, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the q, k and v projections as a stand-in for
                # attention, then applies the output projection.
                xq = self.wq(x)
                xk = self.wk(x)
                xv = self.wv(x)
                # fake attention:
                xo = xq + xk + xv
                return self.wo(xo)

        class FakeTransformerBlock(nn.Module):
            # [CLASS DOC] HELPER: Holds a single FakeAttention module as attn.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the attention submodule.
                super().__init__()
                self.attn = FakeAttention()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the attention module to x.
                return self.attn(x)

        class FakeTransformer(nn.Module):
            # [CLASS DOC] HELPER: Holds a single FakeTransformerBlock as block.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the block submodule.
                super().__init__()
                self.block = FakeTransformerBlock()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the block to the input.
                return self.block(input)

        model = FakeTransformer().to(self.device_type)

        tp_mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("tp",))

        # apply sequence parallel
        parallel_plan = {
            "attn": PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
        }

        parallelize_module(
            module=model.block,
            device_mesh=tp_mesh,
            parallelize_plan=parallel_plan,
        )

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(model, backend=cnt, fullgraph=True)
        inp = torch.rand(20, 16).to(self.device_type)
        out = compiled_model(inp)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, 1)

        code = run_and_get_triton_code(compiled_model, inp)
        FileCheck().check(
            "buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(primal"
        ).check("torch.ops._c10d_functional.wait_tensor.default(buf0").check(
            "extern_kernels.mm(buf0,"
        ).run(code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(1)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    def test_tp_compile_comm_reordering(self):
        # [FUNCTION DOC] GPU test with compute/comm reordering enabled and one compile thread:
        # runs the shared tensor-parallel comm reordering check.
        self._test_tp_compile_comm_reordering()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(1)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @torch._inductor.config.patch("graph_partition", True)
    def test_tp_compile_comm_reordering_graph_partition(self):
        # [FUNCTION DOC] Same as the reordering test but with Inductor graph_partition enabled.
        self._test_tp_compile_comm_reordering()

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_make_fx_with_invoke_subgraph_dtensor(self):
        # [FUNCTION DOC] With force_compile_during_fx_trace, traces with make_fx (fake mode) a
        # function that wraps a tensor as a Shard(0) DTensor and calls a function compiled with
        # the invoke_subgraph backend that redistributes to Replicate. Asserts the printed graph
        # contains an invoke_subgraph higher-order op whose subgraph has all_gather and
        # wait_tensor.
        """Test that make_fx can trace over torch.compile with invoke_subgraph backend and DTensor.

        This tests the scenario where:
        1. An outer make_fx traces a function
        2. Inside that function, a torch.compile'd function with invoke_subgraph backend is called
        3. The compiled function operates on DTensor inputs
        4. The invoke_subgraph HOP should appear in the traced graph
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        torch._dynamo.reset()

        def inner_fn(dt):
            # [FUNCTION DOC] HELPER: Redistributes the DTensor to Replicate on the enclosing
            # mesh.
            # Redistribute DTensor
            return dt.redistribute(mesh, [Replicate()])

        compiled_fn = torch.compile(inner_fn, backend="invoke_subgraph", fullgraph=True)

        def outer_fn(x):
            # [FUNCTION DOC] HELPER: Adds 1, builds a Shard(0) DTensor, calls the compiled inner
            # function and returns the local tensor of its result.
            # Convert plain tensor to DTensor
            dt = DTensor.from_local(x + 1, mesh, [Shard(0)], run_check=False)
            # Call compiled function with DTensor
            dt_out = compiled_fn(dt)
            # Convert back to plain tensor
            return dt_out.to_local()

        x = torch.randn(4, 4, device="cpu")

        # Trace with make_fx
        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        # Verify the full graph structure including invoke_subgraph
        graph_str = "\n".join(
            line.rstrip()
            for line in traced.print_readable(print_output=False).strip().split("\n")
        )
        self.assertExpectedInline(
            graph_str,
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[4, 4]"):
        # No stacktrace found for following nodes
        add: "f32[4, 4]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        view: "f32[4, 4]" = torch.ops.aten.view.default(add, [4, 4]);  add = None
        repeated_subgraph0 = self.repeated_subgraph0
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj0_1 = self._opaque_obj0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', view, _opaque_obj0, _opaque_obj0_1);  repeated_subgraph0 = view = _opaque_obj0 = _opaque_obj0_1 = None
        getitem: "f32[8, 4]" = invoke_subgraph[0]
        getitem_1 = invoke_subgraph[1];  invoke_subgraph = getitem_1 = None
        view_1: "f32[8, 4]" = torch.ops.aten.view.default(getitem, [8, 4]);  getitem = None
        return view_1

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 4]", arg1_1, arg2_1):
            # No stacktrace found for following nodes
            all_gather_into_tensor: "f32[8, 4]" = torch.ops._c10d_functional.all_gather_into_tensor.default(arg0_1, 2, '0');  arg0_1 = None
            wait_tensor: "f32[8, 4]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
            return (wait_tensor, arg2_1)""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_dtensor_requires_grad(self):
        # [FUNCTION DOC] With force_compile_during_fx_trace, runs aot_function over an outer
        # function that calls a Dynamo-compiled (invoke_subgraph backend) inner function on
        # DTensors with requires_grad. Runs forward and backward and asserts a forward graph was
        # produced and Dynamo compiled the inner function exactly once.
        """Test AOTAutograd over Dynamo with DTensor inputs/outputs and requires_grad.

        This tests the scenario where:
        1. An outer aot_function traces a function with DTensor requires_grad inputs
        2. Inside that function, a torch.compile'd function with invoke_subgraph backend is called
        3. Both inner and outer operate on DTensors
        4. The inner Dynamo region should only be compiled once
        """
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._functorch.aot_autograd import aot_function
        from torch._functorch.compilers import nop

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        torch._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        def inner_fn(dt):
            # [FUNCTION DOC] HELPER: Returns dt * 2 + 1.
            # Simple operation on DTensor
            return dt * 2 + 1

        compiled_fn = torch.compile(inner_fn, backend=compile_counter)

        def outer_fn(dt):
            # [FUNCTION DOC] HELPER: Adds 1, calls the compiled inner function and returns the
            # sum.
            # Outer function also operates on DTensor
            dt2 = dt + 1
            dt3 = compiled_fn(dt2)
            return dt3.sum()

        # Create DTensor with requires_grad
        local_tensor = torch.randn(4, 4, requires_grad=True)
        dt_input = DTensor.from_local(local_tensor, mesh, [Shard(0)], run_check=False)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Saves the forward graph module in the nonlocal fw_graph and
            # returns it unchanged.
            nonlocal fw_graph
            fw_graph = gm
            return gm

        aot_fn = aot_function(
            outer_fn,
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            _disable_torch_fn_metadata_mode=True,
        )

        # Run forward and backward
        result = aot_fn(dt_input)
        result.backward()

        # Check that we got a forward graph
        self.assertIsNotNone(fw_graph, "Expected a forward graph")

        # Check compile count - should be exactly 1 compilation
        self.assertEqual(
            compile_counter.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 compilation, got {compile_counter.frame_count}",
        )

    def test_device_mesh_slice(self):
        # [FUNCTION DOC] Switches to a fake 8-rank group and a 2x4 mesh (dp, tp); checks that a
        # function slicing the "tp" submesh and building a DTensor on it gives the same result in
        # eager, under make_fx fake tracing and under aot_eager compile.
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        mesh = init_device_mesh(self.device_type, (2, 4), mesh_dim_names=("dp", "tp"))

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a Shard(0) DTensor on the tp submesh; returns the
            # input if tp is not a mesh dim name, else the local tensor.
            tp_mesh = mesh["tp"]
            dt = DTensor.from_local(x, tp_mesh, [Shard(0)], run_check=False)
            if "tp" not in dt.device_mesh.mesh_dim_names:
                return x
            return dt.to_local()

        x = torch.randn(4, 4)
        res = fn(x)
        traced = make_fx(fn, tracing_mode="fake")(x)
        self.assertEqual(traced(x), res)

        compiled = torch.compile(fn, backend="aot_eager")(x)
        self.assertEqual(compiled, res)

    def test_compile_redistribute_flattened_mesh(self):
        # [FUNCTION DOC] Switches to a fake 8-rank group and a 3D mesh (fsdp, cp, tp), pre-
        # creates the flattened fsdp+cp mesh, then compiles a redistribute from [Shard, Shard,
        # Replicate] to all Replicate; checks the output placements and that backward runs.
        """
        Test that redistribute works with pre-flattened meshes during compile.

        When redistributing from [Shard, Shard, Replicate] to [Replicate, Replicate, Replicate]
        on a 3D mesh, DTensor can optimize by using a single all-gather on the flattened
        mesh instead of 2 sequential all-gathers. This requires the flattened mesh to be
        pre-created before compile to avoid tracing issues.
        """
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)

        mesh = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("fsdp", "cp", "tp")
        )
        # Pre-create flattened mesh for fsdp+cp before compile
        mesh["fsdp", "cp"]._flatten()

        def redistribute_fn(x):
            # [FUNCTION DOC] HELPER: Redistributes x to Replicate on all three mesh dims.
            return x.redistribute(placements=[Replicate(), Replicate(), Replicate()])

        input_dt = DTensor.from_local(
            torch.randn(4, 8, device=self.device_type, requires_grad=True),
            device_mesh=mesh,
            placements=[Shard(0), Shard(0), Replicate()],
            run_check=False,
        )

        compiled_fn = torch.compile(
            redistribute_fn, fullgraph=True, backend="aot_eager"
        )
        result = compiled_fn(input_dt)
        self.assertEqual(result.placements, (Replicate(), Replicate(), Replicate()))

        # Test backward pass
        result.sum().backward()

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_flattened_submesh_no_getattr_compile_on_one_rank(self):
        # [FUNCTION DOC] With compile_on_one_rank enabled on a fake 4-rank 2x2 mesh with a
        # flattened dp_tp submesh, patches the AOT stage-2a partition function to count
        # DeviceMesh get_attr nodes in the partitioned forward/backward graphs; asserts the count
        # is 0. Restores the patched function and recreates the default fake process group in a
        # finally block.
        """When compile_on_one_rank=True, the flattened submesh should appear as a
        custom op call_function node (derived from the mesh graph input) rather than
        as a get_attr constant holding an unpicklable ProcessGroup."""
        from torch._library.fake_class_registry import FakeScriptObject

        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)

        mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=("dp", "tp"))
        mesh["dp", "tp"]._flatten("dp_tp")

        import torch._functorch._aot_autograd.graph_compile as gc

        orig = gc._aot_stage2a_partition
        device_mesh_getattr_count = [0]

        def check_partition(fx_g, joint_inputs, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper over the original partition function: calls it,
            # counts get_attr nodes in the forward and backward modules whose value (unwrapping
            # fake script objects) is a DeviceMesh, stores the count in a one-element list and
            # returns the result.
            result = orig(fx_g, joint_inputs, *args, **kwargs)
            fw_module, bw_module = result[0], result[1]
            count = 0
            for gm in (fw_module, bw_module):
                for node in gm.graph.nodes:
                    if node.op != "get_attr":
                        continue
                    val = getattr(gm, node.target, None)
                    if isinstance(val, FakeScriptObject):
                        val = getattr(val, "real_obj", val)
                    if isinstance(val, DeviceMesh):
                        count += 1
            device_mesh_getattr_count[0] = count
            return result

        gc._aot_stage2a_partition = check_partition
        torch._functorch._aot_autograd.graph_compile._aot_stage2a_partition = (
            check_partition
        )
        try:
            torch._dynamo.reset()
            with torch.compiler.config.patch(compile_on_one_rank=True):

                def fn(x, mesh):
                    # [FUNCTION DOC] HELPER: Builds a [Shard(0), Shard(0)] DTensor on the mesh
                    # input, redistributes to Replicate on both dims and returns the sum of its
                    # local tensor.
                    dt = DTensor.from_local(
                        x, mesh, [Shard(0), Shard(0)], run_check=False
                    )
                    return (
                        dt.redistribute(mesh, [Replicate(), Replicate()])
                        .to_local()
                        .sum()
                    )

                compiled = torch.compile(fn)
                x = torch.randn(4, 4, requires_grad=True)
                loss = compiled(x, mesh)
                loss.backward()

            self.assertEqual(
                device_mesh_getattr_count[0],
                0,
                "DeviceMesh get_attr nodes survived into partitioned graphs; "
                "the flattened submesh should use a custom op instead",
            )
        finally:
            gc._aot_stage2a_partition = orig
            torch._functorch._aot_autograd.graph_compile._aot_stage2a_partition = orig
            dist.destroy_process_group()
            dist.init_process_group(
                "fake", store=FakeStore(), rank=0, world_size=self.world_size
            )

    def test_sibling_submesh_reconstruct_from_ancestor(self):
        # [FUNCTION DOC] With compile_on_one_rank, builds a mesh with fsdp and tp dims by
        # unflatten of a world mesh and a concatenated mesh, traces with make_fx a function whose
        # closure captures the fsdp submesh. Asserts no DeviceMesh get_attr nodes exist and that
        # a get_submesh call node derives the closure mesh from the concatenated ancestor. Resets
        # the process group in a finally block.
        """When a closure captures an FSDP submesh while a sibling mesh (the
        concatenated fsdp+tp mesh) is a graph input, the closure-captured mesh
        must be reconstructed via _get_submesh — not baked as a get_attr
        constant that breaks serialization.

        This reproduces the SimpleFSDP + TP pattern: ReplicateComputation
        captures the FSDP submesh as self.device_mesh, while parameters are
        DTensors on DeviceMesh._concatenate([fsdp_mesh, tp_mesh]). With a
        1D world mesh unflattened into (fsdp, tp), the root mesh has dim
        name 'world' — so the reconstruct_fn must find the concatenated
        ancestor (which contains 'fsdp') rather than looking only at root.
        """
        from torch._library.fake_class_registry import FakeScriptObject

        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)

        try:
            # Build 1D world mesh → unflatten → (fsdp=4, tp=2)
            world_mesh = init_device_mesh(
                self.device_type, (8,), mesh_dim_names=("world",)
            )
            dense_mesh = world_mesh._unflatten(0, (4, 2), ("fsdp", "tp"))
            fsdp_mesh = dense_mesh["fsdp"]
            tp_mesh = dense_mesh["tp"]

            # Concatenate like SimpleFSDP's _distribute_dtensor does
            concat_mesh = DeviceMesh._concatenate([fsdp_mesh, tp_mesh])

            with torch.compiler.config.patch(compile_on_one_rank=True):

                def fn(local_tensor, concat_mesh_input):
                    # [FUNCTION DOC] HELPER: Wraps local data on the concatenated mesh input,
                    # takes the local tensor, re-wraps it on the closure fsdp mesh, redistributes
                    # to Replicate and sums.
                    # concat_mesh_input is a graph input (placeholder).
                    # Simulate SimpleFSDP: create DTensor on concat mesh,
                    # extract local, re-wrap on the closure-captured
                    # fsdp_mesh, then all-gather.
                    dt = DTensor.from_local(
                        local_tensor,
                        concat_mesh_input,
                        [Shard(0), Shard(1)],
                        run_check=False,
                    )
                    local = dt.to_local()
                    fsdp_dt = DTensor.from_local(
                        local, fsdp_mesh, [Shard(0)], run_check=False
                    )
                    return (
                        fsdp_dt.redistribute(fsdp_mesh, [Replicate()]).to_local().sum()
                    )

                local_t = torch.randn(4, 4)
                gm = make_fx(fn, tracing_mode="fake")(local_t, concat_mesh)

            device_mesh_getattr_count = 0
            for node in gm.graph.nodes:
                if node.op != "get_attr":
                    continue
                val = getattr(gm, node.target, None)
                if isinstance(val, FakeScriptObject):
                    val = getattr(val, "real_obj", val)
                if isinstance(val, DeviceMesh):
                    device_mesh_getattr_count += 1

            self.assertEqual(
                device_mesh_getattr_count,
                0,
                "DeviceMesh get_attr nodes found; the closure-captured FSDP "
                "submesh should be reconstructed from the tracked concatenated "
                "ancestor via _get_submesh, not baked as a constant",
            )

            submesh_ops = [
                node
                for node in gm.graph.nodes
                if node.op == "call_function" and "get_submesh" in str(node.target)
            ]
            self.assertGreater(
                len(submesh_ops),
                0,
                "Expected _get_submesh call to derive the FSDP submesh from "
                "the tracked concatenated ancestor mesh",
            )
        finally:
            dist.destroy_process_group()
            dist.init_process_group(
                "fake", store=FakeStore(), rank=0, world_size=self.world_size
            )

    def test_to_local_symbolic_sizes_non_sharded_dims(self):
        # [FUNCTION DOC] Compiles a function that narrows a dynamic dim, re-wraps as a Shard(1)
        # DTensor with the full shape, redistributes to Replicate and views; asserts the result
        # equals eager and that no view line of the forward graph has a floor-division in its
        # symbolic shape.
        # Checks that symbolic sizes are not polluted by the sharding
        # in to_local - for instance we could incorrectly have (16 * (s27//2))
        # as opposed to just (8 * s27) which enables correct generation
        # https://github.com/pytorch/pytorch/issues/175690
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Narrows dim 1 to its share, builds a Shard(1) DTensor with
            # the original shape and stride, redistributes to Replicate and flattens trailing
            # dims.
            local = x.narrow(1, 0, x.shape[1] // self.world_size)
            dt = DTensor.from_local(
                local,
                mesh,
                [Shard(1)],
                run_check=False,
                shape=x.shape,
                stride=x.stride(),
            )
            r = dt.redistribute(placements=[Replicate()])
            out = r.to_local()
            return out.view(out.shape[0], -1)

        inp = torch.randn(1, 10, 8)
        torch._dynamo.mark_dynamic(inp, 1)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        result = opt_fn(inp)
        self.assertEqual(result, fn(inp))

        # The view shape should use the clean symbol (s0), not 2*((s0//2)).
        fw_code = backend.fw_graphs[0].print_readable(print_output=False)
        for line in fw_code.splitlines():
            if "view" in line and "-1" in line:
                self.assertNotIn(
                    "//", line, lambda msg: f"{msg}\nPolluted symbolic shape: {line}"
                )

    def test_to_local_symbolic_sizes_uneven_shard(self):
        # [FUNCTION DOC] Same function as the previous test compiled once; runs it with a dynamic
        # dim of size 10 (divisible by world size) and 9 (not divisible) and compares each with
        # eager.
        # Regression test to ensure our narrow changes does not cause any
        # unintentional incorrectness
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            # [FUNCTION DOC] HELPER: Narrows dim 1 to its share, builds a Shard(1) DTensor with
            # the original shape and stride, redistributes to Replicate and flattens trailing
            # dims.
            local = x.narrow(1, 0, x.shape[1] // self.world_size)
            dt = DTensor.from_local(
                local,
                mesh,
                [Shard(1)],
                run_check=False,
                shape=x.shape,
                stride=x.stride(),
            )
            r = dt.redistribute(placements=[Replicate()])
            out = r.to_local()
            return out.view(out.shape[0], -1)

        opt_fn = torch.compile(fn, backend="aot_eager")

        # Divisible: 10 % 2 == 0
        inp_even = torch.randn(1, 10, 8)
        torch._dynamo.mark_dynamic(inp_even, 1)
        result = opt_fn(inp_even)
        self.assertEqual(result, fn(inp_even))

        # Non-divisible: 9 % 2 != 0, exercises padding/unpadding + recompile
        # - most importantly ensures correctness
        inp_odd = torch.randn(1, 9, 8)
        torch._dynamo.mark_dynamic(inp_odd, 1)
        result = opt_fn(inp_odd)
        self.assertEqual(result, fn(inp_odd))

    def test_mesh_get_process_group_1d(self):
        # [FUNCTION DOC] Checks the _dtensor.mesh_get_process_group op on a 1D mesh returns a
        # ProcessGroup equal to the mesh group for dim 0.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        pg = torch.ops._dtensor.mesh_get_process_group(mesh, 0)
        self.assertIsInstance(pg, dist.ProcessGroup)
        self.assertEqual(pg, mesh.get_group(0))

    def test_mesh_get_process_group_2d(self):
        # [FUNCTION DOC] Switches to a fake 4-rank group and a 2x2 mesh; checks the op returns a
        # ProcessGroup equal to mesh.get_group for each of the two dims.
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)

        mesh = DeviceMesh(self.device_type, torch.arange(4).reshape(2, 2))
        for dim in range(2):
            pg = torch.ops._dtensor.mesh_get_process_group(mesh, dim)
            self.assertIsInstance(pg, dist.ProcessGroup)
            self.assertEqual(pg, mesh.get_group(dim))

    def test_mesh_get_process_group_fake(self):
        # [FUNCTION DOC] Under FakeTensorMode converts the mesh to a fake script object and
        # checks the mesh_get_process_group op still returns a ProcessGroup equal to the real
        # group.
        from torch._library.fake_class_registry import (
            FakeScriptObject,
            maybe_to_fake_obj,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        with FakeTensorMode() as fake_mode:
            fake_mesh = maybe_to_fake_obj(fake_mode, mesh)
            self.assertIsInstance(fake_mesh, FakeScriptObject)

            pg = torch.ops._dtensor.mesh_get_process_group(fake_mesh, 0)
            self.assertIsInstance(pg, dist.ProcessGroup)
            self.assertEqual(pg, mesh.get_group(0))

    def test_resolve_group_uses_op_with_compile_on_one_rank(self):
        # [FUNCTION DOC] With compile_on_one_rank, checks _resolve_group returns a ProcessGroup
        # for a 1D mesh and for (mesh, dim) tuples on a 2x2 mesh (fake 4-rank group), and that
        # _group_or_group_name passes a ProcessGroup through unchanged; without the flag
        # _resolve_group returns a string.
        """Test that _resolve_group returns a ProcessGroup (not a string name)
        when compile_on_one_rank is enabled, for both 1D mesh and (mesh, dim)
        tuple inputs. Also tests that _group_or_group_name passes through the
        ProcessGroup instead of extracting .group_name."""
        from torch.distributed._functional_collectives import (
            _group_or_group_name,
            _resolve_group,
        )

        mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))

        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        mesh_2d = DeviceMesh(self.device_type, torch.arange(4).reshape(2, 2))

        with torch.compiler.config.patch(compile_on_one_rank=True):
            # 1D mesh
            result = _resolve_group(mesh_1d)
            self.assertIsInstance(result, dist.ProcessGroup)
            self.assertEqual(result, mesh_1d.get_group(0))

            # (mesh, dim) tuple
            for dim in range(2):
                result = _resolve_group((mesh_2d, dim))
                self.assertIsInstance(result, dist.ProcessGroup)
                self.assertEqual(result, mesh_2d.get_group(dim))

            # _group_or_group_name passes ProcessGroup through
            pg = mesh_1d.get_group(0)
            result = _group_or_group_name(pg)
            self.assertIs(result, pg)

        # Without compile_on_one_rank, _resolve_group returns a string name
        result = _resolve_group(mesh_1d)
        self.assertIsInstance(result, str)

    @unittest.skipIf(not torch.accelerator.is_available(), "accelerator not available")
    def test_no_device_mesh_getattr_in_tp_dp_joint_graph(self):
        # [FUNCTION DOC] Builds a SimpleFSDP-style setup on a fake 4-rank 2x2 (fsdp, tp) mesh:
        # tensor-parallelizes a small model on meta device, wraps every parameter with a DP
        # DTensor and a parametrization class, materializes and fills the parameters, captures
        # the graph with dynamo_graph_capture_for_export and exports the joint graph. Asserts no
        # DeviceMesh appears as a get_attr constant in the joint graph. Registers generated
        # classes in their module (global side effect).
        """Backward closures of DTensor ops should not capture DeviceMesh as
        get_attr constants in the joint graph (they break AOTAutograd cache
        serialization because ProcessGroups are unpicklable)."""
        from torch._library.fake_class_registry import FakeScriptObject

        # Need world_size=4 for a 2x2 mesh; re-init the fake PG.
        dist.destroy_process_group()
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)

        def _distribute_dtensor(
            tensor: DTensor,
            device_mesh: DeviceMesh,
            dp_placements: tuple[Placement, ...],
        ) -> DTensor:
            # [FUNCTION DOC] HELPER: Redistributes a tensor-parallel DTensor's local data from
            # replicated to the DP placements on device_mesh, then returns a DTensor on the mesh
            # concatenated from device_mesh and the original mesh, with a strided shard placement
            # (when the original is sharded on that dim) followed by the original placements.
            inner_spec = tensor._spec
            spanned_mesh = DeviceMesh._concatenate([device_mesh, inner_spec.mesh])
            shard_dim = dp_placements[0].dim
            split_factor = inner_spec.num_shards_map[shard_dim]
            tensor_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else dp_placements[0]
                ),
            ) + inner_spec.placements
            current_spec = DTensorSpec(
                mesh=device_mesh,
                placements=(Replicate(),) * len(dp_placements),
                tensor_meta=inner_spec.tensor_meta,
            )
            target_spec = DTensorSpec(
                mesh=device_mesh,
                placements=tuple(dp_placements),
                tensor_meta=inner_spec.tensor_meta,
            )
            result_tensor = redistribute_local_tensor(
                tensor._local_tensor,
                current_spec=current_spec,
                target_spec=target_spec,
            )
            return DTensor(
                result_tensor.requires_grad_(tensor.requires_grad),
                DTensorSpec(
                    mesh=spanned_mesh,
                    placements=tensor_placement,
                    tensor_meta=inner_spec.tensor_meta,
                ),
                requires_grad=tensor.requires_grad,
            )

        class ReplicateComputation(nn.Module):
            def __init__(self, device_mesh, param_sharding):
                # [FUNCTION DOC] HELPER: Stores the DP mesh and param sharding and builds
                # replicate compute placements and sum-Partial gradient placements for each mesh
                # dim.
                super().__init__()
                self.device_mesh = device_mesh
                self.param_sharding = param_sharding
                self.compute_placements = [Replicate()] * device_mesh.ndim
                self.grad_placements = [Partial(reduce_op="sum")] * device_mesh.ndim

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Rewraps the DTensor's local tensor on the DP mesh,
                # redistributes to Replicate with Partial gradient placements on to_local, then
                # wraps the result as a DTensor on the non-DP submesh with the remaining
                # placements.
                non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
                sharded_dtensor = DTensor.from_local(
                    x.to_local(), self.device_mesh, self.param_sharding
                )
                replicated_local_tensor = sharded_dtensor.redistribute(
                    placements=self.compute_placements,
                ).to_local(grad_placements=self.grad_placements)
                non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
                non_dp_mesh_dim_names = tuple(
                    x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
                )
                non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]
                return DTensor.from_local(
                    replicated_local_tensor, non_dp_mesh, non_dp_placements
                )

        param_sharding = (Shard(0),)
        wrap_idx = [0]

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two bias-free 16x16 linear layers with a relu between them;
            # forward returns the sum of the output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.linear1 = nn.Linear(16, 16, bias=False)
                self.linear2 = nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1, relu, linear2 and returns the sum.
                return self.linear2(torch.relu(self.linear1(x))).sum()

        with torch.compiler.config.patch("compile_on_one_rank", True):
            device = torch.device(f"{self.device_type}:0")
            torch.accelerator.set_device_index(0)
            mesh_2d = init_device_mesh(
                self.device_type, (2, 2), mesh_dim_names=("fsdp", "tp")
            )
            dp_mesh = mesh_2d["fsdp"]

            with torch.device("meta"):
                model = Model()
            parallelize_module(
                model,
                mesh_2d["tp"],
                {
                    "linear1": ColwiseParallel(),
                    "linear2": RowwiseParallel(),
                },
            )
            # Apply data parallelism with parametrization
            for mod in model.modules():
                params_dict = dict(mod.named_parameters(recurse=False))
                for p_name, p in params_dict.items():
                    mod.register_parameter(
                        p_name,
                        nn.Parameter(_distribute_dtensor(p, dp_mesh, param_sharding)),
                    )
                parametrization = ReplicateComputation(dp_mesh, param_sharding)
                props = {
                    pn: property(
                        lambda self, _pn=pn: parametrization(self._parameters[_pn])
                    )
                    for pn in params_dict
                }
                idx = wrap_idx[0]
                wrap_idx[0] += 1
                cls = type(
                    f"SimpleFSDP{mod.__class__.__name__}_{idx}",
                    (mod.__class__,),
                    props,
                )
                mod.__class__ = cls
                import sys

                sys.modules[cls.__module__].__dict__[cls.__name__] = cls

            model.to_empty(device=device)
            with (
                torch.compiler.config.patch("compile_on_one_rank", False),
                torch.no_grad(),
            ):
                for p in model.parameters():
                    p.fill_(0.01)
            model.train()

            x = torch.randn(4, 16, device=device)
            with (
                torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
                torch.fx.traceback.preserve_node_meta(),
            ):
                gm = dynamo_graph_capture_for_export(model)(x)
                tracing_context = gm.meta["tracing_context"]

            with tracing(tracing_context), ExitStack() as stack:
                jwd = aot_export_joint_with_descriptors(stack, gm, (x,))

            for node in jwd.graph_module.graph.nodes:
                if node.op != "get_attr":
                    continue
                val = getattr(jwd.graph_module, node.target, None)
                if isinstance(val, FakeScriptObject):
                    val = getattr(val, "real_obj", val)
                self.assertNotIsInstance(
                    val,
                    DeviceMesh,
                    "DeviceMesh should not appear as get_attr in the joint graph",
                )

    def test_compile_optimizer_grad_view_base_dim_mismatch(self):
        # [FUNCTION DOC] Regression test: compiles an Adam optimizer step for a DTensor param
        # whose local tensor is a view of a 2D base while its gradient's local tensor is a view
        # of a 1D base; must not error from reusing the param symbolic context for the grad.
        # Regression test for https://github.com/pytorch/pytorch/issues/176667
        # When param._local_tensor is a view of an N-D base but
        # grad._local_tensor is a view of a 1-D base (as in FSDP2's flat
        # gradient buffer), dynamo's meta tensor creation must not reuse the
        # param's symbolic context for the grad.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # param: view of 2D base (simulates FSDP2 reshard padding)
        param_local = torch.randn(32, 256)[:16]
        param = DTensor.from_local(param_local, mesh, [Replicate()], run_check=False)
        param.requires_grad_(True)
        param.retain_grad()

        # grad: view of 1D base (simulates FSDP2 flat gradient buffer)
        grad_local = torch.randn(16 * 256 + 1000)[: 16 * 256].view(16, 256)
        param.grad = DTensor.from_local(
            grad_local, mesh, [Replicate()], run_check=False
        )

        opt = torch.optim.Adam([param], lr=1e-3)
        compiled_step = torch.compile(opt.step, backend="aot_eager")
        compiled_step()

    def test_pad_tensor_no_guard_on_symbolic_pad_size(self):
        # [FUNCTION DOC] Builds a fake tensor with a symbolic second dim, traces pad_tensor with
        # make_fx in symbolic mode on a symbolic pad size, and asserts the symbol is not
        # concretized by a guard and the placeholder dim stays symbolic.
        """pad_tensor must not create a guard that concretizes symbolic pad sizes.

        When tracing with make_fx in symbolic mode, pad_size may be a SymInt
        (e.g., for uneven DTensor sharding where local shard sizes vary by
        rank). guard_or_false(pad_size == 0) must not be evaluated during
        tracing, otherwise it creates a guard that collapses the SymInt to its
        hint value, making the graph rank-specific instead of rank-independent.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily
        from torch.distributed.tensor._collective_utils import pad_tensor
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with unset_fake_temporarily():
            real = torch.empty(400, 15, device="meta")
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC, DimDynamic.DYNAMIC]
        )
        x = fake_mode.from_tensor(real, symbolic_context=sym_ctx)
        with fake_mode:
            x = x.to("cpu")

        # Verify the symbol is alive before tracing
        self.assertTrue(isinstance(x.shape[1], torch.SymInt))
        self.assertFalse(x.shape[1].node.expr.is_number)

        def fn(t):
            # [FUNCTION DOC] HELPER: Pads dim 1 of t up to size 15 using pad_tensor.
            pad_size = 15 - t.size(1)
            return pad_tensor(t, 1, pad_size)

        with fake_mode:
            gm = make_fx(fn, tracing_mode="symbolic")(x)

        # The symbol must survive — not be guarded to a concrete value
        self.assertFalse(
            x.shape[1].node.expr.is_number,
            lambda msg: f"{msg}\npad_tensor created a guard that concretized the symbolic dim: "
            f"expr={x.shape[1].node.expr}",
        )

        # The traced graph should have a symbolic pad size, not concrete 0
        placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
        val = placeholder.meta["val"]
        self.assertTrue(
            isinstance(val.shape[1], torch.SymInt),
            "Placeholder dim should be symbolic",
        )

    def test_make_fx_tp_embedding_no_shadow_nodes(self):
        # [FUNCTION DOC] Traces an embedding lookup of a Shard(0) DTensor weight with make_fx
        # (fake mode) and asserts the graph has no empty_strided nodes (no shadow nodes leaked
        # from sharding propagation).
        """make_fx through a TP-sharded embedding must not produce dead shadow nodes.

        DTensor's ShardingPropagator runs each op on fake global-shape tensors
        to derive output metadata. Without disable_proxy_modes_tracing, make_fx
        traces those internal metadata ops, leaving dead "shadow" nodes backed
        by empty_strided placeholders. For aten.embedding, the shadow node reads
        uninitialized indices and crashes with out-of-bounds access at runtime.
        """
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        vocab_size, embed_dim = 32, 16
        weight = distribute_tensor(torch.randn(vocab_size, embed_dim), mesh, [Shard(0)])

        def fn(weight, indices):
            # [FUNCTION DOC] HELPER: Wraps indices as a Replicate DTensor and returns the local
            # tensor of the embedding lookup with the weight.
            dt_indices = DTensor.from_local(
                indices, mesh, [Replicate()], run_check=False
            )
            return torch.nn.functional.embedding(dt_indices, weight).to_local()

        indices = torch.randint(0, vocab_size, (4,))

        traced = make_fx(fn, tracing_mode="fake")(weight, indices)

        empty_strided_nodes = [
            n
            for n in traced.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.empty_strided.default
        ]
        self.assertEqual(
            len(empty_strided_nodes),
            0,
            "Shadow empty_strided nodes from ShardingPropagator leaked into "
            "the make_fx graph; disable_proxy_modes_tracing is not active",
        )

    def test_stable_hash_for_caching_is_rank_specific(self):
        # [FUNCTION DOC] Regression test: the DTensor stable cache hash must depend on the local
        # tensor device; hashes before and after moving the local tensor to meta must differ.
        # Regression test for https://github.com/pytorch/pytorch/issues/188390.
        # _stable_hash_for_caching must include the local tensor's device so
        # that each rank produces a unique AOTAutograd cache key.  Before the
        # fix, the device was omitted and rank 1 incorrectly reused rank 0's
        # compiled kernel, causing CUDA errors at runtime.
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        local = torch.empty(2, 4)
        dt = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
        h0 = dt._stable_hash_for_caching()
        # Swap to a different device type to verify the device string participates
        # in the hash.  This does not model two real CPU ranks (both would be plain
        # "cpu" with no index); see test_stable_hash_for_caching_cuda_ranks for the
        # actual multi-rank scenario.  _spec is intentionally left inconsistent with
        # _local_tensor; this is a focused unit test of the hash only.
        dt._local_tensor = dt._local_tensor.to("meta")
        h1 = dt._stable_hash_for_caching()
        self.assertNotEqual(h0, h1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 CUDA devices")
    def test_stable_hash_for_caching_cuda_ranks(self):
        # [FUNCTION DOC] Needs 2 CUDA devices: two DTensors with equal specs but local tensors on
        # cuda:0 and cuda:1 must have different stable cache hashes.
        # Exercise the exact scenario from #188390: two DTensors with identical
        # global specs but local tensors on cuda:0 vs cuda:1 must produce
        # different AOTAutograd cache keys.
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        local0 = torch.empty(2, 4, device="cuda:0")
        local1 = torch.empty(2, 4, device="cuda:1")
        dt0 = DTensor.from_local(local0, mesh, [Shard(0)], run_check=False)
        dt1 = DTensor.from_local(local1, mesh, [Shard(0)], run_check=False)
        h0, h1 = dt0._stable_hash_for_caching(), dt1._stable_hash_for_caching()
        self.assertNotEqual(h0, h1)


@instantiate_parametrized_tests
class TestDTensorCompileE2E(DTensorTestBase):
    # [CLASS DOC] Multi-process (4 ranks) end-to-end tests that compile tensor-parallel and
    # FSDP+TP models and DTensor redistribution with real process groups via DTensorTestBase;
    # _bwd_ctx optionally enables compiled autograd for the backward pass. Parametrized over
    # sequence parallel and compiled-autograd flags.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the world size, 4.
        return 4

    # multiprocess relies on pickling the source code
    # so compiled autograd tests can't dynamically wrap this class
    def _bwd_ctx(self, use_ca):
        # [FUNCTION DOC] HELPER: Returns a no-op context manager if use_ca is false, otherwise
        # the compiled autograd enable context using torch.compile.
        if not use_ca:
            return contextlib.nullcontext()
        return torch._dynamo.compiled_autograd._enable(torch.compile)

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    @parametrize("use_ca", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel, use_ca):
        # [FUNCTION DOC] Parallelizes a two-MLP model over a 1D mesh with colwise/rowwise styles
        # (with input/output preparation when sequence parallel), compiles it with fullgraph and
        # the aot_eager counter backend, runs backward (optionally under compiled autograd), and
        # asserts the output equals eager and a single frame was compiled.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = SimpleModel(self.device_type)

        colwise_style = (
            ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel()
        )
        rowwise_style = (
            RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel()
        )

        if is_seq_parallel:
            # use input preparation to test out the compile of it
            prepare_module_input = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
            )
            prepare_module_out = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(0),
            )
            plan = {
                "mlp_0": prepare_module_input,
                "mlp_0.net1": ColwiseParallel(),
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": RowwiseParallel(),
                "mlp_1": prepare_module_out,
            }
        else:
            plan = {
                "mlp_0.net1": colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            }

        model = parallelize_module(
            model,
            mesh,
            parallelize_plan=plan,
        )
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_mod = torch.compile(model, backend=cnt, fullgraph=True)
        compiled_out = compiled_mod(inp)
        with self._bwd_ctx(use_ca):
            compiled_out.sum().backward()
        self.assertEqual(compiled_out, out)
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_2d_fsdp_tp_compile(self, use_ca):
        # [FUNCTION DOC] Needs 4 GPUs. Applies tensor parallel on the tp dim of a (dp=2, tp) mesh
        # and FSDP on the dp dim to two copies of a model; compiles one with the counting
        # aot_eager backend, checks forward output equals eager, runs backward and asserts one
        # frame.
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        twod_mesh = init_device_mesh(
            self.device_type,
            (data_parallel_size, self.world_size // data_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, twod_mesh["tp"], parallelize_plan)
        eager_2d = FSDP(
            tp_model,
            device_id=dev_type.type,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )
        out = eager_2d(inp)
        tp_model2 = parallelize_module(
            model_copy,
            twod_mesh["tp"],
            parallelize_plan,
        )
        fsdp_2d = FSDP(
            tp_model2,
            device_id=dev_type.type,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )

        # TODO: once aot autograd support is ready we can just use default backend
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_2d = torch.compile(fsdp_2d, backend=cnt)
        compiled_output = compiled_2d(inp)
        with self._bwd_ctx(use_ca):
            compiled_output.sum().backward()

        self.assertEqual(out, compiled_output)
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_2d_fsdp_tp_ac_compile(self, use_ca):
        # [FUNCTION DOC] Needs 4 GPUs. Same as the 2D FSDP+TP test but wraps the tensor-parallel
        # model with non-reentrant activation checkpointing; compares forward outputs and then
        # the parameter gradients of eager and compiled FSDP models.
        dp_degree = 2
        tp_degree = self.world_size // dp_degree
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=("dp", "tp"),
        )

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
        tp_model = checkpoint_wrapper(
            tp_model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        eager_2d = FSDP(tp_model, device_mesh=mesh_2d["dp"], use_orig_params=True)

        tp_model2 = parallelize_module(model_copy, mesh_2d["tp"], parallelize_plan)
        fsdp_2d = FSDP(
            tp_model2,
            device_mesh=mesh_2d["dp"],
            use_orig_params=True,
        )
        # TODO: once aot autograd support is ready we can just use default backend
        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager")

        # forward pass
        out = eager_2d(inp)
        compiled_output = compiled_2d(inp)
        self.assertEqual(out, compiled_output)

        # backward pass
        out.sum().backward()
        with self._bwd_ctx(use_ca):
            compiled_output.sum().backward()

        # compare the gradients:
        for n, p in zip(fsdp_2d.parameters(), compiled_2d.parameters()):
            self.assertEqual(n.grad, p.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("use_ca", [True, False])
    def test_compile_dtensor_redistribute_backward(self, use_ca):
        # [FUNCTION DOC] Needs 4 GPUs. Compiles a function that builds Shard(0)/Shard(1) DTensors
        # from reshaped inputs, matmuls them, redistributes to Replicate and returns the local
        # tensor; checks output and input gradients against eager.
        mesh = DeviceMesh(
            device_type=self.device_type, mesh=torch.arange(self.world_size)
        )

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds two DTensors from x and y, multiplies them,
            # redistributes the product to Replicate and returns its local tensor.
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        ref = fn(x_ref, y_ref)

        x = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y = torch.arange(8, requires_grad=True, dtype=torch.float32)
        res = opt_fn(x, y)

        self.assertEqual(res, ref)

        # Now run and assert the backward + gradients
        ref.sum().backward()
        with self._bwd_ctx(use_ca):
            res.sum().backward()

        self.assertEqual(x_ref.grad, x.grad)
        self.assertEqual(y_ref.grad, y.grad)

    @with_comms
    def test_compile_embedding_redistribute(self):
        # [FUNCTION DOC] Compiles a network whose embedding is sharded on dim 0 and whose forward
        # redistributes the result to Shard(1); applied to a replicated input, the full tensor
        # must equal the eager embedding output.
        mesh = self.build_device_mesh()

        class Network(nn.Module):
            # [CLASS DOC] HELPER: Module with a mesh and an embedding sharded via _apply_sharding
            # on dim 0; forward embeds the DTensor input and redistributes to Shard(1).
            def __init__(self, embedding, mesh):
                # [FUNCTION DOC] HELPER: Stores the mesh and shards the embedding over it.
                super().__init__()
                self.mesh = mesh
                self.embedding = _apply_sharding(embedding, 0, self.mesh)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Embeds x and redistributes the result to Shard(1).
                x = self.embedding(x)
                x = x.redistribute(self.mesh, [Shard(1)])
                return x

        embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        inp = torch.randint(0, 10, (8,), device=self.device_type)
        ref_out = embedding(inp)
        sharded_net = torch.compile(Network(embedding, mesh))
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        output = sharded_net(replicated_inp)
        self.assertEqual(output.full_tensor(), ref_out)

    @with_comms
    def test_unbacked_illegal_views(self):
        # [FUNCTION DOC] Checks view(-1) on DTensors with unbacked dims: legal when the dim 0 is
        # sharded and known divisible; raises RuntimeError "not evenly divisible by mesh
        # dimension" when sharded on dim 1 or when divisibility is not given to the compiler.
        """Test that views with unbacked shapes match eager behavior"""
        device_mesh = self.build_device_mesh()

        def create_dt(shard_dim):
            # [FUNCTION DOC] HELPER: Makes an 8x8x8 distributed random tensor sharded on
            # shard_dim with all three dims marked unbacked.
            tensor = torch.randn(8, 8, 8)
            dt = distribute_tensor(tensor, device_mesh, [Shard(shard_dim)])
            for i in range(3):
                torch._dynamo.decorators.mark_unbacked(dt, i)
            return dt

        # aot_eager backend decomposes to as_strided, will fail as unsupported
        @torch.compile(backend="eager", fullgraph=True)
        def flatten_on_even_mesh(x):
            # [FUNCTION DOC] HELPER: Checks size(0) is divisible by the world size, then flattens
            # x.
            torch._check(x.size(0) % self.world_size == 0)
            return x.view(-1)

        # view should be legal, since sharding is even and flatten is on first dim
        dt = create_dt(0)
        flatten_on_even_mesh(dt)

        # flattening on non-first dimension should fail
        dt = create_dt(1)
        with self.assertRaisesRegex(
            RuntimeError, "is not evenly divisible by mesh dimension"
        ):
            flatten_on_even_mesh(dt)

        @torch.compile(backend="eager", fullgraph=True)
        def flatten(x):
            # [FUNCTION DOC] HELPER: Flattens x with view(-1).
            return x.view(-1)

        # uneven case: not informing compiler of divisibility will crash
        dt = create_dt(0)
        with self.assertRaisesRegex(
            RuntimeError, "is not evenly divisible by mesh dimension"
        ):
            flatten(dt)

    @with_comms
    def test_split_with_symint_split_size(self):
        # [FUNCTION DOC] Compiles with dynamic=True a split of a replicated DTensor by an integer
        # split size (2, 3, 4) and checks each chunk's full tensor matches the plain split.
        """
        Test that split works with symbolic integer split_size when using
        torch.compile with dynamic=True.
        """
        mesh = self.build_device_mesh()
        placements = [Replicate()]

        global_tensor = torch.randn(8, 8, device=self.device_type)
        input_dt = distribute_tensor(global_tensor, mesh, placements)

        def split_fn(x, split_size):
            # [FUNCTION DOC] HELPER: Splits x along dim 0 with the given split size.
            return torch.split(x, split_size, dim=0)

        compiled_split_fn = torch.compile(split_fn, dynamic=True)

        # Test with different split sizes: evenly divisible and not evenly divisible
        for split_size in [2, 3, 4]:
            expected = split_fn(global_tensor, split_size)
            result = compiled_split_fn(input_dt, split_size)

            self.assertEqual(len(result), len(expected))
            for dt_chunk, tensor_chunk in zip(result, expected):
                self.assertEqual(dt_chunk.full_tensor(), tensor_chunk)

    @with_comms
    def test_dtensor_processgroup_backward(self):
        # [FUNCTION DOC] With compile_on_one_rank enabled, runs aot_function on a redistribute
        # plus sum with backward and checks neither the forward nor backward graph code contains
        # opaque objects.
        """Test that ProcessGroups are correctly handled in backward graph."""
        from torch._functorch.aot_autograd import aot_function

        with patch("torch.compiler.config.compile_on_one_rank", True):
            mesh = self.build_device_mesh()

            def fn(dt):
                # [FUNCTION DOC] HELPER: Redistributes dt to Replicate on the mesh and returns
                # the sum.
                out = dt.redistribute(mesh, [Replicate()])
                return out.sum()

            local_tensor = torch.randn(
                4, 8, device=self.device_type, requires_grad=True
            )
            dt_input = DTensor.from_local(
                local_tensor, mesh, [Shard(0)], run_check=False
            )

            fw_graph_cell = [None]
            bw_graph_cell = [None]

            def extract_fw_graph(fx_g, _):
                # [FUNCTION DOC] HELPER: Stores the forward graph module in the cell and returns
                # it.
                fw_graph_cell[0] = fx_g
                return fx_g

            def extract_bw_graph(fx_g, _):
                # [FUNCTION DOC] HELPER: Stores the backward graph module in the cell and returns
                # it.
                bw_graph_cell[0] = fx_g
                return fx_g

            compiled_fn = aot_function(
                fn,
                fw_compiler=extract_fw_graph,
                bw_compiler=extract_bw_graph,
            )

            output = compiled_fn(dt_input)
            output.backward()

            fw_graph = fw_graph_cell[0]
            if fw_graph is not None:
                fw_code = fw_graph.code
                self.assertNotIn(
                    "_opaque_obj",
                    fw_code,
                    lambda msg: f"{msg}\nForward graph should not contain opaque objects. Graph:\n{fw_code}",
                )

            bw_graph = bw_graph_cell[0]
            if bw_graph is not None:
                bw_code = bw_graph.code
                self.assertNotIn(
                    "_opaque_obj",
                    bw_code,
                    lambda msg: f"{msg}\nBackward graph should not contain opaque objects. Graph:\n{bw_code}",
                )

    @with_comms
    def test_dtensor_processgroup_extraction(self):
        # [FUNCTION DOC] With compile_on_one_rank enabled, captures the forward graph of a
        # redistribute via aot_function; asserts no opaque objects in the code and that the graph
        # has more than one placeholder (process groups extracted as inputs).
        """Test that with compile_on_one_rank, DeviceMesh becomes a graph input
        and ProcessGroups are extracted in-graph as placeholders."""
        from torch._functorch.aot_autograd import aot_function

        with patch("torch.compiler.config.compile_on_one_rank", True):
            mesh = self.build_device_mesh()

            def fn(dt):
                # [FUNCTION DOC] HELPER: Redistributes dt to Replicate on the mesh.
                return dt.redistribute(mesh, [Replicate()])

            local_tensor = torch.randn(4, 8, device=self.device_type)
            dt_input = DTensor.from_local(
                local_tensor, mesh, [Shard(0)], run_check=False
            )

            fw_graph_cell = [None]

            def extract_graph(fx_g, _):
                # [FUNCTION DOC] HELPER: Stores the graph module in the cell and returns it.
                fw_graph_cell[0] = fx_g
                return fx_g

            compiled_fn = aot_function(
                fn,
                fw_compiler=extract_graph,
                bw_compiler=lambda fx_g, _: fx_g,
            )

            compiled_fn(dt_input)

            fw_graph = fw_graph_cell[0]
            self.assertIsNotNone(fw_graph)

            graph_code = fw_graph.code

            self.assertNotIn(
                "_opaque_obj",
                graph_code,
                lambda msg: f"{msg}\nGraph should not contain opaque objects. Graph:\n{graph_code}",
            )

            placeholders = [n for n in fw_graph.graph.nodes if n.op == "placeholder"]
            self.assertGreater(
                len(placeholders),
                1,
                lambda msg: f"{msg}\nExpected ProcessGroup placeholders but only got {len(placeholders)} placeholder(s)",
            )

    @with_comms
    def test_dtensor_processgroup_deduplication(self):
        # [FUNCTION DOC] With compile_on_one_rank enabled, captures the forward graph of two
        # redistributes sharing a mesh; asserts no opaque objects and more than two placeholders
        # (the two DTensor inputs plus the shared process group).
        """Test that multiple DTensors sharing the same ProcessGroup only pass
        it once as a graph placeholder."""
        from torch._functorch.aot_autograd import aot_function

        with patch("torch.compiler.config.compile_on_one_rank", True):
            mesh = self.build_device_mesh()

            def fn(dt1, dt2):
                # [FUNCTION DOC] HELPER: Redistributes both DTensors to Replicate and returns
                # their sum.
                out1 = dt1.redistribute(mesh, [Replicate()])
                out2 = dt2.redistribute(mesh, [Replicate()])
                return out1 + out2

            local1 = torch.randn(4, 8, device=self.device_type)
            local2 = torch.randn(4, 8, device=self.device_type)
            dt1 = DTensor.from_local(local1, mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(local2, mesh, [Shard(0)], run_check=False)

            fw_graph_cell = [None]

            def extract_graph(fx_g, _):
                # [FUNCTION DOC] HELPER: Stores the graph module in the cell and returns it.
                fw_graph_cell[0] = fx_g
                return fx_g

            compiled_fn = aot_function(
                fn,
                fw_compiler=extract_graph,
                bw_compiler=lambda fx_g, _: fx_g,
            )

            compiled_fn(dt1, dt2)

            fw_graph = fw_graph_cell[0]
            self.assertIsNotNone(fw_graph)

            graph_code = fw_graph.code

            self.assertNotIn(
                "_opaque_obj",
                graph_code,
                lambda msg: f"{msg}\nGraph should not contain opaque objects. Graph:\n{graph_code}",
            )

            placeholders = [n for n in fw_graph.graph.nodes if n.op == "placeholder"]
            self.assertGreater(
                len(placeholders),
                2,
                lambda msg: f"{msg}\nExpected ProcessGroup placeholders but only got {len(placeholders)} placeholder(s)",
            )


class TestDTensorACCompile(DTensorTestBase):
    # [CLASS DOC] Two-rank regression test class for DTensor combined with activation
    # checkpointing and compile (the docstring describes a crash when lifting an unbound inner
    # symbol into the checkpoint subgraph).
    """Regression test for DTensor + AC + compile interaction.

    When a DTensor with independent inner/outer symbolic shapes is lifted
    as a freevar into a checkpoint HOP subgraph, ``_lift_basic_symbols``
    recurses into the inner tensor with ``source=None`` and tries to lift
    an unbound inner symbol to the root tracer, crashing with::

        AssertionError: Source of '<sym>' is None when lifting it
        to input of top-level.
    """

    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the world size, 2.
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_tp_ac_compile_dtensor_inner_symbol(self):
        # [FUNCTION DOC] Needs 2 GPUs. Tensor-parallelizes an embedding plus MLP model with
        # DTensor outputs, wraps the MLP in non-reentrant checkpointing, compiles it with
        # aot_eager, and runs forward/backward with two sequence lengths so dynamic recompile
        # occurs; no assertions beyond no error.
        # KNOWN ISSUE: sets a global dynamo config flag
        # (skip_fwd_side_effects_in_bwd_under_checkpoint) without restoring it, leaking into
        # other tests.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # Embedding produces a DTensor whose inner/outer sizes get
        # independent symbols (outer = global seq_len, inner = local
        # seq_len). This is the pattern from TP with variable-length seqs.
        class EmbedBlock(nn.Module):
            # [CLASS DOC] HELPER: Embedding followed by an MLPModule block, mapping ids to
            # features.
            def __init__(self, vocab, dim, device):
                # [FUNCTION DOC] HELPER: Creates the embedding (vocab by dim) and the MLP block
                # on the device.
                super().__init__()
                self.embed = nn.Embedding(vocab, dim, device=device)
                self.block = MLPModule(device)

            def forward(self, ids):
                # [FUNCTION DOC] HELPER: Embeds ids and runs the MLP block.
                return self.block(self.embed(ids))

        model = EmbedBlock(32, 10, self.device_type)

        parallelize_module(
            model,
            mesh,
            {
                "embed": RowwiseParallel(
                    output_layouts=Shard(1), use_local_output=False
                ),
                "block.net1": ColwiseParallel(
                    input_layouts=Shard(1), use_local_output=False
                ),
                "block.net2": RowwiseParallel(use_local_output=False),
            },
        )
        model.block = checkpoint_wrapper(
            model.block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

        model.block.compile(backend="aot_eager", fullgraph=True)

        ids1 = torch.randint(0, 32, (4, 20), device=self.device_type)
        out1 = model(ids1)
        out1.sum().backward()
        model.zero_grad()

        # Different seq_len triggers dynamic-shape recompilation where
        # the DTensor inner symbol gets lifted through the AC HOP.
        ids2 = torch.randint(0, 32, (4, 25), device=self.device_type)
        out2 = model(ids2)
        out2.sum().backward()


if __name__ == "__main__":
    run_tests()
