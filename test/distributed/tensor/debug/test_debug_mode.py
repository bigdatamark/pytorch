# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DebugMode (torch.utils._debug_mode), which records dispatched ops,
# DTensor redistributions, compiled-region nodes, Triton kernels and optional tensor hashes. Has
# four test classes: JSON log save/load, DTensor debug-string output (on a fake process group
# with 8 ranks, CUDA required), utility hashing, and an NCCL multi-process collective logging
# test. Import-time side effects: none beyond imports.

import contextlib
import json
import math
import os
import tempfile
import unittest

import torch
import torch._dynamo.config
import torch.distributed as dist
import torch.distributed._functional_collectives as _functional_collectives
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._debug_mode import (
    _OpCall,
    _RedistributeCall,
    _TritonKernelCall,
    DebugMode,
    hash_tensor_fn,
    norm_hash_fn,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._triton import has_triton_package


class TestDebugModeLogSerialization(TestCase):
    # [CLASS DOC] Tests that DebugMode logs survive save_logs / load_logs through JSON files,
    # including hashes, recorded outputs and non-finite hash values. Stateless apart from
    # per-test temporary directories.
    def _run_hashed_debug_mode(self, x):
        # [FUNCTION DOC] HELPER: Runs sin then sum on x under DebugMode with tensor hash logging
        # (inputs included) and returns the recorded log list.
        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes(hash_inputs=True):
            x.sin().sum()
        return debug_mode.logs

    def test_save_load_empty_logs(self):
        # [FUNCTION DOC] Saving an empty log list to a temp JSON file and loading it back
        # returns an empty list.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.json")
            DebugMode.save_logs([], path)
            self.assertEqual(DebugMode.load_logs(path), [])

    def test_save_load_redistribute_outer_call(self):
        # [FUNCTION DOC] Round-trips a single hand-built outer redistribute call record through
        # JSON. The loaded record must be a redistribute call, report as an outer call, and
        # render to the same text as the original.
        call = _RedistributeCall(1, "S(0)", "R", None, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "redistribute.json")
            DebugMode.save_logs([call], path)
            loaded_call = DebugMode.load_logs(path)[0]

        self.assertIsInstance(loaded_call, _RedistributeCall)
        self.assertTrue(loaded_call.is_outer_call)
        self.assertEqual(loaded_call.render([]), call.render([]))

    def test_save_load_logs_for_hash_mismatch(self):
        # [FUNCTION DOC] Records hashed logs for two different inputs and saves both. Loaded
        # logs must render like the originals, show no mismatches against the in-memory logs,
        # and keep tuple input hashes on op records. Comparing the two loaded runs must yield
        # mismatches that include the sin op.
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        x_different = x + 1

        logs1 = self._run_hashed_debug_mode(x)
        logs2 = self._run_hashed_debug_mode(x_different)

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "run1.json")
            path2 = os.path.join(tmpdir, "run2.json")
            DebugMode.save_logs(logs1, path1)
            DebugMode.save_logs(logs2, path2)

            loaded1 = DebugMode.load_logs(path1)
            loaded2 = DebugMode.load_logs(path2)

        self.assertEqual(
            [log.render([]) for log in loaded1],
            [log.render([]) for log in logs1],
        )
        self.assertEqual(
            DebugMode.check_hash_mismatches(logs1, loaded1, compare_inputs=True),
            [],
        )
        self.assertTrue(
            any(
                isinstance(log, _OpCall)
                and log.log is not None
                and isinstance(log.log.get("input_hash"), tuple)
                for log in loaded1
            )
        )

        mismatches = DebugMode.check_hash_mismatches(
            loaded1, loaded2, compare_inputs=True
        )
        self.assertGreater(len(mismatches), 0)
        self.assertIn("aten::sin", {mismatch["call"] for mismatch in mismatches})

    def test_save_load_logs_with_record_outputs(self):
        # [FUNCTION DOC] Records logs with output recording and hashing enabled, checks some op
        # record holds an output, then saves and reloads. Loaded records carry no raw recorded
        # data (the record field is None) yet hash comparison against the originals finds no
        # mismatches.
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        with (
            DebugMode() as debug_mode,
            DebugMode.record_outputs(),
            DebugMode.log_tensor_hashes(),
        ):
            x.sin().sum()
        logs = debug_mode.logs

        self.assertTrue(
            any(
                isinstance(log, _OpCall)
                and log.record is not None
                and "output" in log.record
                for log in logs
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "logs.json")
            DebugMode.save_logs(logs, path)
            loaded_logs = DebugMode.load_logs(path)

        self.assertTrue(all(log.record is None for log in loaded_logs))
        self.assertEqual(DebugMode.check_hash_mismatches(logs, loaded_logs), [])

    def test_save_load_logs_with_nonfinite_hashes(self):
        # [FUNCTION DOC] Uses a custom hash function that returns NaN, then infinity, and saves
        # each log. The files must be strictly valid JSON (no NaN or Infinity literals), and
        # loaded logs must restore NaN and infinite hashes. Comparing them reports mismatches
        # whose hash fields are NaN and infinite respectively.
        def reject_nonstandard_json_constant(value):
            # [FUNCTION DOC] HELPER: JSON parse_constant hook that fails the test whenever the
            # parser meets a nonstandard constant such as NaN or Infinity.
            raise AssertionError(f"nonstandard JSON constant: {value}")

        def run_with_hash(hash_value):
            # [FUNCTION DOC] HELPER: Runs sin and sum on a 4x4 float tensor under DebugMode with
            # a hash function that always returns hash_value; returns the recorded logs.
            x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            with (
                DebugMode() as debug_mode,
                DebugMode.log_tensor_hashes(
                    hash_fn=lambda t: hash_value, hash_inputs=True
                ),
            ):
                x.sin().sum()
            return debug_mode.logs

        logs_nan = run_with_hash(float("nan"))
        logs_inf = run_with_hash(float("inf"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path_nan = os.path.join(tmpdir, "nan.json")
            path_inf = os.path.join(tmpdir, "inf.json")
            DebugMode.save_logs(logs_nan, path_nan)
            DebugMode.save_logs(logs_inf, path_inf)

            with open(path_nan, encoding="utf-8") as f:
                nan_json = f.read()
            with open(path_inf, encoding="utf-8") as f:
                inf_json = f.read()

            json.loads(nan_json, parse_constant=reject_nonstandard_json_constant)
            json.loads(inf_json, parse_constant=reject_nonstandard_json_constant)
            self.assertNotIn("NaN", nan_json)
            self.assertNotIn("Infinity", inf_json)

            loaded_nan = DebugMode.load_logs(path_nan)
            loaded_inf = DebugMode.load_logs(path_inf)

        self.assertTrue(
            any(
                isinstance(log, _OpCall)
                and log.log is not None
                and math.isnan(log.log["hash"])
                for log in loaded_nan
            )
        )
        self.assertTrue(
            any(
                isinstance(log, _OpCall)
                and log.log is not None
                and math.isinf(log.log["hash"])
                for log in loaded_inf
            )
        )

        mismatches = DebugMode.check_hash_mismatches(
            loaded_nan, loaded_inf, compare_inputs=True
        )
        self.assertGreater(len(mismatches), 0)
        self.assertTrue(any(math.isnan(mismatch["hash1"]) for mismatch in mismatches))
        self.assertTrue(any(math.isinf(mismatch["hash2"]) for mismatch in mismatches))


@requires_cuda
class TestDTensorDebugMode(TestCase):
    # [CLASS DOC] CUDA-only tests of DebugMode string output for DTensor ops, redistributions,
    # compiled graphs, nn.Module annotations and hash checks. Each test uses a fake process group
    # of world size 8 set up in setUp (rank 0) and torn down in tearDown.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Turns on the dynamo option for canonical output-graph node order for
        # the whole class (global config mutation, undone in tearDownClass).
        super().setUpClass()
        torch._dynamo.config.canonicalize_output_graph_node_order = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Restores the canonical output-graph node order dynamo option to off.
        torch._dynamo.config.canonicalize_output_graph_node_order = False
        super().tearDownClass()

    def tearDown(self):
        # [FUNCTION DOC] Runs base teardown then destroys the default process group.
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        # [FUNCTION DOC] Initializes a fake-backend default process group (rank 0, world size 8,
        # in-memory store) and sets the device type to cuda. Global state: process group.
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
        self.device_type = "cuda"

    def test_debug_mode_mm(self):
        # [FUNCTION DOC] Multiplies two DTensors sharded on dim 0 under DebugMode with torch
        # function and id recording, and compares the debug string to an expected dump showing
        # the implicit all-gather redistribution. Also checks record types, args_str-only
        # storage, that a compiled aot_eager run records outputs and hashes matching eager,
        # tuple hash functions (norm, hash_tensor) against direct computation, and edge cases
        # (NaN, inf, complex, float8 and int8 hashes).
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=False)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with DebugMode(record_torchfunction=True, record_ids=True) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))  ->  dt$7: f32[8, 32]| S(0)
    aten::mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))
      -> output: S(0)
      redistribute_input [implicit] (1, S(0) -> R)
        redistribute_input(t$2: f32[1, 32], trace: S(0)->R)
          _c10d_functional::all_gather_into_tensor(t$2: f32[1, 32], 8, 0)  ->  t$3: f32[8, 32]
          _c10d_functional::_wrap_tensor_autograd(t$3: f32[8, 32])  ->  t$4: f32[8, 32]
          _c10d_functional::wait_tensor(t$3: f32[8, 32])  ->  t$3: f32[8, 32]
      aten::mm(t$5: f32[1, 8], t$3: f32[8, 32])  ->  t$6: f32[1, 32]
  <method 'sum' of 'torch._C.TensorBase' objects>(dt$7: f32[8, 32]| S(0))  ->  dt$9: f32[]| P(sum)
    aten::sum(dt$7: f32[8, 32]| S(0))
      aten::sum(t$6: f32[1, 32])  ->  t$8: f32[]""",
        )

        self.assertTrue(isinstance(debug_mode.operators[0], _OpCall))
        self.assertTrue(isinstance(debug_mode.operators[3], _RedistributeCall))
        self.assertEqual(next(iter(debug_mode.operators[1])), torch.ops.aten.mm.default)

        # check stringification
        self.assertTrue(hasattr(debug_mode.operators[0], "args_str"))
        self.assertFalse(hasattr(debug_mode.operators[0], "args"))

        # check recording hook
        def mm(x, y):
            # [FUNCTION DOC] HELPER: Matrix-multiplies x and y and returns the sum of the result.
            return (x @ y).sum()

        eager_out = mm(x_dtensor, y_dtensor)

        # check recording hook for compiled variant
        with (
            DebugMode() as debug_mode,
            DebugMode.record_outputs(),
            DebugMode.log_tensor_hashes(),
        ):
            compiled_out = torch.compile(mm, backend="aot_eager")(x_dtensor, y_dtensor)

        # check numerical equivalence
        self.assertTrue(torch.equal(eager_out, compiled_out))
        sum_op = next(
            iter(
                op
                for op in debug_mode.operators
                if isinstance(op, _OpCall) and str(op.op) == "aten.sum.default"
            )
        )
        self.assertTrue(torch.equal(sum_op.record["output"], eager_out.to_local()))
        self.assertTrue(
            "aten::sum(t: f32[1, 32])  ->  t: f32[]  # {'hash': "
            in debug_mode.debug_string()
        )

        # check tuple hash functions
        with (
            DebugMode() as debug_mode,
            DebugMode.log_tensor_hashes(hash_fn=["norm", "hash_tensor"]),
        ):
            mm(x_dtensor, y_dtensor)

        output_hash = debug_mode.operators[-1].log["hash"]
        norm_ = lambda x: norm_hash_fn(x, use_scalar=True)  # noqa: E731
        hash_ = lambda x: hash_tensor_fn(x, use_scalar=True)  # noqa: E731

        self.assertEqual(output_hash[0], norm_(eager_out))
        self.assertEqual(output_hash[1], hash_(eager_out))

        # some edge cases
        self.assertEqual(norm_(torch.tensor(torch.nan)), torch.nan)
        self.assertEqual(norm_(torch.tensor(torch.inf)), torch.inf)
        self.assertEqual(norm_(torch.complex(torch.ones(4), torch.zeros(4))), 4)
        self.assertEqual(hash_(torch.ones(4, dtype=torch.float8_e5m2)), 0)
        self.assertEqual(hash_(torch.ones(4, dtype=torch.int8)), 0)
        self.assertEqual(hash_(torch.ones(5, dtype=torch.int8)), 1)

    def test_debug_string_inside_context(self):
        # [FUNCTION DOC] Checks that calling debug_string while the DebugMode context is still
        # active gives the same text as calling it after the context exits.
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=False)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with DebugMode() as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()
            s0 = debug_mode.debug_string()
        s1 = debug_mode.debug_string()
        self.assertEqual(s0, s1)

    def test_debug_mode_backward(self):
        # [FUNCTION DOC] Adds DTensors sharded on different dims, sums, and runs backward under
        # DebugMode with stack traces. Uses substring checks (order of clone and detach is not
        # deterministic) for forward ops, implicit redistribution with all-to-all, backward ops,
        # split, _to_copy, and that the last op's stack trace mentions the backward call.
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(8, 1, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(1)], run_check=False)

        with DebugMode(
            record_torchfunction=True, record_stack_trace=True, record_output=False
        ) as debug_mode:
            z = x_dtensor + y_dtensor
            z.sum().backward()

        debug_string = debug_mode.debug_string(show_stack_trace=False)

        # Certain operations (clone, detach) appear in non-deterministic order during
        # backward pass redistribution.
        # For operations that appear multiple times, we include full parameters to
        # disambiguate (e.g., two different aten::add operations). For unique operations,
        # we use partial matching to avoid brittleness if the exact signature format changes.

        # Check for key forward operations
        self.assertIn(
            "<method 'add' of 'torch._C.TensorBase' objects>", debug_string
        )  # Unique - no params needed
        self.assertIn(
            "aten::add.Tensor(dt: f32[8, 8]| S(0), dt: f32[8, 8]| S(1))", debug_string
        )
        self.assertIn("redistribute_input [implicit] (1, S(1) -> S(0))", debug_string)
        self.assertIn(
            "redistribute_input(t: f32[8, 1], trace: S(1)->S(0))", debug_string
        )
        self.assertIn(
            "_dtensor::shard_dim_alltoall", debug_string
        )  # Unique - no params needed
        self.assertIn("aten::add.Tensor(t: f32[1, 8], t: f32[1, 8])", debug_string)

        # Check for sum operations
        self.assertIn(
            "<method 'sum' of 'torch._C.TensorBase' objects>", debug_string
        )  # Unique - no params needed
        self.assertIn("aten::sum(dt: f32[8, 8]| S(0))", debug_string)
        self.assertIn("aten::sum(t: f32[1, 8])", debug_string)

        # Check for backward operations
        self.assertIn(
            "torch._tensor.backward", debug_string
        )  # Unique - no params needed
        self.assertIn("aten::ones_like(dt: f32[]| P(sum)", debug_string)
        self.assertIn("aten::ones_like(t: f32[]", debug_string)
        self.assertIn("aten::expand(dt: f32[]| R, [8, 8])", debug_string)
        self.assertIn("aten::expand(t: f32[], [8, 8])", debug_string)

        # Check for redistribute operations in backward pass
        self.assertIn("redistribute_input(t: f32[8, 8], trace: R->S(1))", debug_string)
        self.assertIn("redistribute_input(t: f32[8, 8], trace: R->S(0))", debug_string)

        # Check for split operations in backward pass
        self.assertIn("aten::split.Tensor(t: f32[8, 8], 1, 1)", debug_string)
        self.assertIn("aten::split.Tensor(t: f32[8, 8], 1)", debug_string)

        # Check for _to_copy operations
        self.assertIn(
            "aten::_to_copy(t: f32[8, 1], dtype=torch.float32, layout=torch.strided, device=cpu)",
            debug_string,
        )
        self.assertIn(
            "aten::_to_copy(t: f32[1, 8], dtype=torch.float32, layout=torch.strided, device=cpu)",
            debug_string,
        )

        # Check that both clone and detach operations appear (order-independent)
        self.assertIn("aten::clone(t: f32[8, 1])", debug_string)
        self.assertIn("aten::clone(t: f32[1, 8])", debug_string)
        self.assertIn("aten::detach(t: f32[8, 1])", debug_string)
        self.assertIn("aten::detach(t: f32[1, 8])", debug_string)

        # check stack trace
        self.assertTrue("z.sum().backward()" in debug_mode.operators[-1].stack_trace)

    def test_stack_trace_in_compiled_region(self):
        # [FUNCTION DOC] Compiles a small module with aot_eager (full graph) and records forward
        # and backward ops under DebugMode with the interpreter. Asserts that source lines of the
        # model code appear in the stack traces of specific recorded op positions.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Two linears (8 to 4, 4 to 8) with an add, relu and sum.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear layers l1 and l2.
                super().__init__()
                self.l1 = torch.nn.Linear(8, 4)
                self.l2 = torch.nn.Linear(4, 8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 2, applies l1, relu, l2, then returns the sum.
                x = x + 2
                x = self.l1(x)
                x = x.relu()
                x = self.l2(x)
                x = x.sum()
                return x

        x = torch.randn(16, 8)
        model = torch.compile(Foo(), backend="aot_eager", fullgraph=True)

        # test forward nodes
        with DebugMode(
            record_stack_trace=True, run_compile_with_interpreter=True
        ) as debug_mode:
            out = model(x)

        op_calls = [op for op in debug_mode.operators if isinstance(op, _OpCall)]
        self.assertTrue("x = x + 2" in op_calls[0].stack_trace)
        self.assertTrue("x = self.l1(x)" in op_calls[1].stack_trace)
        self.assertTrue("x = x.relu()" in op_calls[3].stack_trace)
        self.assertTrue("x = x.sum()" in op_calls[-1].stack_trace)

        # test backward nodes
        with DebugMode(
            record_stack_trace=True, run_compile_with_interpreter=True
        ) as debug_mode:
            out.backward()

        op_calls = [op for op in debug_mode.operators if isinstance(op, _OpCall)]
        self.assertTrue("out.backward()" in op_calls[0].stack_trace)
        self.assertTrue("x = x.sum()" in op_calls[1].stack_trace)
        self.assertTrue("x = self.l2(x)" in op_calls[2].stack_trace)
        self.assertTrue("x = x.relu()" in op_calls[12].stack_trace)

    def test_debug_mode_densor_redistribution_trace(self):
        # [FUNCTION DOC] Uses a 4x2 mesh with explicit multi-mesh-dim shard order and multiplies
        # DTensors sharded on both mesh dims. Compares the debug string with the expected dump
        # of a two-step redistribution trace (all-gathers with chunk and cat) and sum.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).view(4, 2))

        x = torch.randn(16, 8, requires_grad=True)
        y = torch.randn(8, 16, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0), Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(1), Shard(1)], run_check=False)
        x_dtensor._spec.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),)
        y_dtensor._spec.shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 1)),)
        with DebugMode(record_torchfunction=False, record_output=False) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  aten::mm(dt: f32[128, 8]| S(0)[0]S(0)[1], dt: f32[8, 128]| S(1)[0]S(1)[1])
    redistribute_input [implicit] (1, S(1)[0]S(1)[1] -> RR)
      redistribute_input(t: f32[8, 16], trace: S(1)[0]S(1)[1]->S(1)R->RR)
        _c10d_functional::all_gather_into_tensor(t: f32[8, 16], 2, 3)
        _c10d_functional::_wrap_tensor_autograd(t: f32[16, 16])
        _c10d_functional::wait_tensor(t: f32[16, 16])
        aten::chunk(t: f32[16, 16], 2)
        aten::cat(['t: f32[8, 16]', 't: f32[8, 16]'], 1)
        _c10d_functional::all_gather_into_tensor(t: f32[8, 32], 4, 1)
        _c10d_functional::_wrap_tensor_autograd(t: f32[32, 32])
        _c10d_functional::wait_tensor(t: f32[32, 32])
        aten::chunk(t: f32[32, 32], 4)
        aten::cat(['t: f32[8, 32]', 't: f32[8, 32]', 't: f32[8, 32]', 't: f32[8, 32]'], 1)
    aten::mm(t: f32[16, 8], t: f32[8, 128])
  aten::sum(dt: f32[128, 128]| S(0)[0]S(0)[1])
    aten::sum(t: f32[16, 128])""",
        )

    def test_debug_mode_explicit_redistribute(self):
        # [FUNCTION DOC] Checks that user-called redistribute and full_tensor are logged with the
        # explicit annotation (not implicit), via expected debug-string dumps; full_tensor also
        # adds a view.
        """Test that explicit user-called redistribute shows [explicit] annotation."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)

        with DebugMode() as debug_mode:
            x_dtensor.redistribute(mesh, [Replicate()])

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    redistribute_input [explicit] (t: f32[1, 8], trace: S(0)->R)
      _c10d_functional::all_gather_into_tensor(t: f32[1, 8], 8, 0)  ->  t: f32[8, 8]
      _c10d_functional::_wrap_tensor_autograd(t: f32[8, 8])  ->  t: f32[8, 8]
      _c10d_functional::wait_tensor(t: f32[8, 8])  ->  t: f32[8, 8]""",
        )

        with DebugMode() as debug_mode:
            x_dtensor.full_tensor()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    redistribute_input [explicit] (t: f32[1, 8], trace: S(0)->R)
      _c10d_functional::all_gather_into_tensor(t: f32[1, 8], 8, 0)  ->  t: f32[8, 8]
      _c10d_functional::_wrap_tensor_autograd(t: f32[8, 8])  ->  t: f32[8, 8]
      _c10d_functional::wait_tensor(t: f32[8, 8])  ->  t: f32[8, 8]
    aten::view(t: f32[8, 8], [8, 8])  ->  t: f32[8, 8]""",
        )

    def test_output_placements(self):
        # [FUNCTION DOC] Runs topk on a DTensor sharded on the topk dim so the input must be
        # redistributed, with output recording on. The expected dump must show the output
        # placements for both outputs of the multi-output op.
        """Test that output placements are recorded for multi-output DTensor ops."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # Use topk which returns multiple outputs (values, indices)
        # Shard on dim=1 (the topk dimension) to trigger redistribution
        x = torch.randn(8, 2, requires_grad=False)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        # Test with redistribution (slow path) - output placements should be recorded
        with DebugMode(record_output=True) as debug_mode:
            torch.topk(x_dtensor, k=4, dim=1)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  aten::topk(dt: f32[8, 16]| S(1), 4, 1)
    -> output: ('R', 'R')
    redistribute_input [implicit] (0, S(1) -> R)
      redistribute_input(t: f32[8, 2], trace: S(1)->R)
        _c10d_functional::all_gather_into_tensor(t: f32[8, 2], 8, 0)  ->  t: f32[64, 2]
        _c10d_functional::_wrap_tensor_autograd(t: f32[64, 2])  ->  t: f32[64, 2]
        _c10d_functional::wait_tensor(t: f32[64, 2])  ->  t: f32[64, 2]
        aten::chunk(t: f32[64, 2], 8)  ->  ['t: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]']
        aten::cat(['t: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]', 't: f32[8, 2]'], 1)  ->  t: f32[8, 16]
    aten::topk(t: f32[8, 16], 4, 1)  ->  ('t: f32[8, 4]', 't: i64[8, 4]')""",
        )

    def test_debug_mode_einsum(self):
        # [FUNCTION DOC] Runs einsum on two DTensors with partial/replicate placements on a 4x2
        # mesh. Expected dump shows the decomposition (unsqueeze, permute, view, bmm) running
        # directly on partial inputs, ending with a doubly partial-sum output and no
        # redistribution.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).view(4, 2))

        # Create test tensors with mixed Partial placements: P(sum)R and RP(sum).
        # Per-input linearity allows bmm to operate directly on Partial inputs
        # without redistribution, producing P(sum)P(sum) output.
        # Numerics are verified in test_matrix_ops.py::test_mm_partial_inputs.
        a = torch.randn(16, 6, 8)
        b = torch.randn(8, 4, 4)

        a_dt = DTensor.from_local(a, mesh, [Partial(), Replicate()], run_check=False)
        b_dt = DTensor.from_local(b, mesh, [Replicate(), Partial()], run_check=False)

        # Capture the operator decomposition
        with DebugMode(record_torchfunction=True, record_output=False) as debug_mode:
            torch.einsum("bld,dnh->blnh", a_dt, b_dt)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.functional.einsum(bld,dnh->blnh, dt: f32[16, 6, 8]| P(sum)R, dt: f32[8, 4, 4]| RP(sum))
    aten::unsqueeze(dt: f32[16, 6, 8]| P(sum)R, 3)
      aten::unsqueeze(t: f32[16, 6, 8], 3)
    aten::unsqueeze(dt: f32[16, 6, 8, 1]| P(sum)R, 4)
      aten::unsqueeze(t: f32[16, 6, 8, 1], 4)
    aten::permute(dt: f32[16, 6, 8, 1, 1]| P(sum)R, [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 8, 1, 1], [0, 1, 3, 4, 2])
    aten::unsqueeze(dt: f32[8, 4, 4]| RP(sum), 3)
      aten::unsqueeze(t: f32[8, 4, 4], 3)
    aten::unsqueeze(dt: f32[8, 4, 4, 1]| RP(sum), 4)
      aten::unsqueeze(t: f32[8, 4, 4, 1], 4)
    aten::permute(dt: f32[8, 4, 4, 1, 1]| RP(sum), [3, 4, 1, 2, 0])
      aten::permute(t: f32[8, 4, 4, 1, 1], [3, 4, 1, 2, 0])
    aten::permute(dt: f32[16, 6, 1, 1, 8]| P(sum)R, [0, 1, 4, 2, 3])
      aten::permute(t: f32[16, 6, 1, 1, 8], [0, 1, 4, 2, 3])
    aten::view(dt: f32[16, 6, 8, 1, 1]| P(sum)R, [1, 96, 8])
      aten::view(t: f32[16, 6, 8, 1, 1], [1, 96, 8])
    aten::permute(dt: f32[1, 1, 4, 4, 8]| RP(sum), [4, 2, 3, 0, 1])
      aten::permute(t: f32[1, 1, 4, 4, 8], [4, 2, 3, 0, 1])
    aten::view(dt: f32[8, 4, 4, 1, 1]| RP(sum), [1, 8, 16])
      aten::view(t: f32[8, 4, 4, 1, 1], [1, 8, 16])
    aten::bmm(dt: f32[1, 96, 8]| P(sum)R, dt: f32[1, 8, 16]| RP(sum))
      aten::bmm(t: f32[1, 96, 8], t: f32[1, 8, 16])
    aten::view(dt: f32[1, 96, 16]| P(sum)P(sum), [16, 6, 1, 4, 4])
      aten::view(t: f32[1, 96, 16], [16, 6, 1, 4, 4])
    aten::permute(dt: f32[16, 6, 1, 4, 4]| P(sum)P(sum), [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 1, 4, 4], [0, 1, 3, 4, 2])
    aten::view(dt: f32[16, 6, 4, 4, 1]| P(sum)P(sum), [16, 6, 4, 4])
      aten::view(t: f32[16, 6, 4, 4, 1], [16, 6, 4, 4])""",
        )

    def test_real_tensor(self):
        # [FUNCTION DOC] Records a plain linear plus sum on a regular tensor with torch-function
        # logging and checks the expected dump (nested aten calls under each torch-level call).
        x = torch.randn(8, 8, 8)
        linear = torch.nn.Linear(8, 8)

        with DebugMode(record_torchfunction=True, record_output=False) as debug_mode:
            linear(x).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch._C._nn.linear(t: f32[8, 8, 8], t: f32[8, 8], t: f32[8])
      aten::view(t: f32[8, 8, 8], [64, 8])
      aten::t(t: f32[8, 8])
      aten::addmm(t: f32[8], t: f32[64, 8], t: f32[8, 8])
      aten::view(t: f32[64, 8], [8, 8, 8])
  <method 'sum' of 'torch._C.TensorBase' objects>(t: f32[8, 8, 8])
      aten::sum(t: f32[8, 8, 8])""",
        )

    def test_fake_tensor(self):
        # [FUNCTION DOC] Creates fake tensors, runs matmul under DebugMode with fake-tensor
        # recording and checks the dump labels them as fake and shows the view, mm and
        # unsafe-view decomposition.
        with FakeTensorMode():
            x = torch.randn(8, 8)
            y = torch.randn(8, 8, 8)

        with DebugMode(
            record_torchfunction=True, record_faketensor=True, record_output=False
        ) as debug_mode:
            torch.matmul(y, x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.matmul(ft: f32[8, 8, 8], ft: f32[8, 8])
      aten::view(ft: f32[8, 8, 8], [64, 8])
      aten::mm(ft: f32[64, 8], ft: f32[8, 8])
      aten::_unsafe_view(ft: f32[64, 8], [8, 8, 8])""",
        )

    def test_tensor_attributes(self):
        # [FUNCTION DOC] Sets custom Python attributes on input tensors and records matmul with
        # attribute recording for two names and original-args storage. The dump must show the
        # attributes only on the tensors carrying them, and the stored first arg must be the
        # very same object as the input.
        x = torch.randn(8, 8)
        x.a1 = "x1"
        x.a2 = "x2"
        y = torch.randn(8, 8, 8)
        y.a1 = "y"

        with DebugMode(
            record_torchfunction=True,
            record_faketensor=True,
            record_tensor_attributes=["a1", "a2"],
            store_original_args=True,
            record_output=False,
        ) as debug_mode:
            torch.matmul(y, x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.matmul(t: f32[8, 8, 8]{a1=y}, t: f32[8, 8]{a1=x1, a2=x2})
      aten::view(t: f32[8, 8, 8]{a1=y}, [64, 8])
      aten::mm(t: f32[64, 8], t: f32[8, 8]{a1=x1, a2=x2})
      aten::_unsafe_view(t: f32[64, 8], [8, 8, 8])""",
        )

        self.assertTrue(hasattr(debug_mode.operators[0], "args"))
        self.assertEqual(id(debug_mode.operators[0].args[0]), id(y))

    @parametrize("has_inner_mode", [True, False])
    @parametrize("has_outer_mode", [True, False])
    def test_nested_debug_mode(self, has_inner_mode, has_outer_mode):
        # [FUNCTION DOC] Parametrized over presence of an extra dispatch mode inside and outside
        # DebugMode. Multiplies sharded DTensors and expects the implicit redistribution to
        # still appear in the debug string regardless of mode nesting.
        class DummyTorchDispatchMode1(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode used as the inner mode.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls the op unchanged and returns its result.
                return func(*args, **kwargs)

        class DummyTorchDispatchMode2(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode used as the outer mode.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls the op unchanged and returns its result.
                return func(*args, **kwargs)

        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        inner_mode = (
            DummyTorchDispatchMode1() if has_inner_mode else contextlib.nullcontext()
        )
        outer_mode = (
            DummyTorchDispatchMode2() if has_outer_mode else contextlib.nullcontext()
        )

        with outer_mode:
            with DebugMode() as debug_mode:
                with inner_mode:
                    torch.mm(x_dtensor, y_dtensor)

        self.assertTrue(
            "redistribute_input [implicit] (1, S(0) -> R)" in debug_mode.debug_string()
        )

    def test_debug_mode_higher_order_cond(self):
        # [FUNCTION DOC] Calls the cond higher-order op directly (avoids compilation) under
        # DebugMode with torch-function recording and checks the call name is in the debug string.
        """Test DebugMode with higher order operation."""
        x = torch.randn(1, 8, requires_grad=True)

        with DebugMode(record_torchfunction=True) as debug_mode:
            # rewrite torch.conda as torch.ops.higher_order.cond to avoid compilation
            torch.ops.higher_order.cond(
                torch.tensor(True), lambda x: x + 1, lambda x: x - 1, (x,)
            )

        # Verify that cond operations are captured in debug mode
        self.assertIn("torch.ops.higher_order.cond", debug_mode.debug_string())

    def test_compile(self):
        # [FUNCTION DOC] Compiles a sin-cos function, then calls it inside DebugMode. The
        # already-compiled calls record nothing (empty debug string) and the compile frame count
        # stays at 1, so DebugMode causes no recompilation.
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns cos of sin of x.
            return x.sin().cos()

        x = torch.randn(8)
        f(x)
        with DebugMode() as debug_mode:
            f(x)
            self.assertEqual(len(debug_mode.debug_string()), 0)
            f(x)
        f(x)
        self.assertEqual(
            cnt.frame_count, 1
        )  # check DebugMode doesn't trigger additional recompilations

    def test_annotate(self):
        # [FUNCTION DOC] Checks DebugMode annotate markers and nn.Module recording in eager mode
        # (expected dump shows module and annotation nesting), then compiles the module with
        # aot_eager and inductor. Annotations survive inside the aot_eager region; the inductor
        # run shows only the lowered addmm out call.
        x = torch.randn(8, 8)

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Single linear (8 to 8) module that emits an annotation.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear layer l1.
                super().__init__()
                self.l1 = torch.nn.Linear(8, 8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds a Foo annotation to DebugMode, then applies l1.
                DebugMode._annotate("Foo")
                return self.l1(x)

        mod = Foo()
        with DebugMode(record_nn_module=True) as debug_mode:
            DebugMode._annotate("forward")
            mod(x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  [annotate] forward
    [nn.Mod] Foo
    [annotate] Foo
      [nn.Mod] Foo.l1
        aten::t(t: f32[8, 8])  ->  t: f32[8, 8]
        aten::addmm(t: f32[8], t: f32[8, 8], t: f32[8, 8])  ->  t: f32[8, 8]""",
        )

        for backend in ["aot_eager", "inductor"]:
            with DebugMode(run_compile_with_interpreter=True) as debug_mode:
                torch.compile(mod, backend=backend, fullgraph=True)(x)

            if backend == "inductor":
                self.assertExpectedInline(
                    debug_mode.debug_string(),
                    """    aten::addmm.out(t: f32[8], t: f32[8, 8], t: f32[8, 8], out=t: f32[8, 8])  ->  t: f32[8, 8]""",
                )
            else:
                self.assertExpectedInline(
                    debug_mode.debug_string(),
                    """\
  [aot_eager region (compile)] enter
  [annotate] Foo
    aten::t(t: f32[8, 8])  ->  t: f32[8, 8]
    aten::addmm(t: f32[8], t: f32[8, 8], t: f32[8, 8])  ->  t: f32[8, 8]
  [aot_eager region (compile)] exit""",
                )

    def test_nn_module_in_eager(self):
        # [FUNCTION DOC] Runs a nested module (Bar holding Foo and a linear) eagerly with module
        # recording and checks the expected hierarchy of nn.Mod entries. Then runs forward and
        # backward with stack traces and checks the forward stack trace of the last sum op
        # (also in debug_string) mentions the source line. Finally checks that an eager-backend
        # full-graph compile does not break on nn.Module recording.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Two chained linear layers (4 to 4), l1 then l2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear layers l1 and l2.
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4)
                self.l2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies l1 then l2 to x.
                return self.l2(self.l1(x))

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps a Foo (abc) followed by a linear layer (xyz).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the Foo submodule abc and linear xyz.
                super().__init__()
                self.abc = Foo()
                self.xyz = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies abc then xyz to x.
                return self.xyz(self.abc(x))

        mod = Bar()
        inp = torch.randn(4, 4)
        with DebugMode(record_nn_module=True, record_output=False) as debug_mode:
            _ = mod(inp)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    [nn.Mod] Bar
      [nn.Mod] Bar.abc
        [nn.Mod] Bar.abc.l1
          aten::t(t: f32[4, 4])
          aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])
        [nn.Mod] Bar.abc.l2
          aten::t(t: f32[4, 4])
          aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])
      [nn.Mod] Bar.xyz
        aten::t(t: f32[4, 4])
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])""",
        )

        with DebugMode(record_stack_trace=True) as debug_mode:
            out = mod(inp).sum()
            out.backward()

        sum_op = [
            op
            for op in debug_mode.operators
            if isinstance(op, _OpCall) and str(op.op) == "aten.sum.dim_IntList"
        ][-1]
        self.assertTrue("self.l2(self.l1(x))" in sum_op.fwd_stack_trace)
        self.assertTrue(
            "self.l2(self.l1(x))" in debug_mode.debug_string(show_stack_trace=True)
        )

        # check that nn_module doesn't graph break in compiled regions
        fn = torch.compile(mod, backend="eager", fullgraph=True)
        with DebugMode(record_nn_module=True) as debug_mode:
            fn(inp)

    @torch._functorch.config.patch(guess_tangent_strides_as_outputs=True)
    def test_nn_module_in_compiled_regions(self):
        # [FUNCTION DOC] Checks module-call hierarchy in dumps when only a submodule is compiled
        # with aot_eager, and when the whole model is compiled including backward (forward and
        # backward compile regions, bwd-labelled modules). Compares against expected dumps.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear l1, relu, then linear l2 (all 4 to 4).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear layers l1 and l2.
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4)
                self.l2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies l1, relu, then l2.
                return self.l2(self.l1(x).relu())

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Adds 2.0 to the input then applies linear l3.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear layer l3.
                super().__init__()
                self.l3 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns l3 applied to x plus 2.0.
                return self.l3(x + 2.0)

        class Baz(torch.nn.Module):
            # [CLASS DOC] HELPER: Composes a Foo (foo) followed by a Bar (bar).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates submodules foo and bar.
                super().__init__()
                self.foo = Foo()
                self.bar = Bar()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies foo then bar to x.
                return self.bar(self.foo(x))

        # Only region of module is compiled, test nn.Mod call hierarchy
        mod = Baz()
        mod.foo = torch.compile(mod.foo, backend="aot_eager", fullgraph=True)
        inp = torch.randn(4, 4)
        with DebugMode(
            record_nn_module=True, run_compile_with_interpreter=True
        ) as debug_mode:
            mod(inp).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    [nn.Mod] Baz
    [aot_eager region (compile)] enter
      [nn.Mod (compile)] L['self'].l1
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
      aten::relu(t: f32[4, 4])  ->  t: f32[4, 4]
      [nn.Mod (compile)] L['self'].l2
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
    [aot_eager region (compile)] exit
      [nn.Mod] Baz.bar
        aten::add.Tensor(t: f32[4, 4], 2.0)  ->  t: f32[4, 4]
        [nn.Mod] Baz.bar.l3
          aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
          aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
    aten::sum(t: f32[4, 4])  ->  t: f32[]""",
        )

        # Entire region is aot-eager compiled, with backwards
        mod = torch.compile(Baz(), backend="aot_eager", fullgraph=True)
        inp = torch.randn(4, 4)
        with DebugMode(
            record_nn_module=True, run_compile_with_interpreter=True
        ) as debug_mode:
            out = mod(inp).sum()
            out.backward()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  [aot_eager region (compile)] enter
    [nn.Mod (compile)] L['self'].foo
      [nn.Mod (compile)] L['self'].foo.l1
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
      aten::relu(t: f32[4, 4])  ->  t: f32[4, 4]
      [nn.Mod (compile)] L['self'].foo.l2
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
    [nn.Mod (compile)] L['self'].bar
      aten::add.Tensor(t: f32[4, 4], 2.0)  ->  t: f32[4, 4]
      [nn.Mod (compile)] L['self'].bar.l3
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
  [aot_eager region (compile)] exit
    aten::sum(t: f32[4, 4])  ->  t: f32[]
    aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)  ->  t: f32[]
    aten::expand(t: f32[], [4, 4])  ->  t: f32[4, 4]
    aten::empty_strided([4, 4], [4, 1], dtype=torch.float32, layout=torch.strided, device=cpu, pin_memory=False)  ->  t: f32[4, 4]
    aten::copy_(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
  [aot_eager region (compile)] enter
    [nn.Mod (compile)] L['self'].bar
      [nn.Mod (compile)] L['self'].bar.l3
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::sum.dim_IntList(t: f32[4, 4], [0], True)  ->  t: f32[1, 4]
        aten::view(t: f32[1, 4], [4])  ->  t: f32[4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
    [nn.Mod (compile)] L['self'].foo
      [nn.Mod (compile)] L['self'].foo.l2
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::sum.dim_IntList(t: f32[4, 4], [0], True)  ->  t: f32[1, 4]
        aten::view(t: f32[1, 4], [4])  ->  t: f32[4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
      aten::detach(t: f32[4, 4])  ->  t: f32[4, 4]
      aten::detach(t: f32[4, 4])  ->  t: f32[4, 4]
      aten::threshold_backward(t: f32[4, 4], t: f32[4, 4], 0)  ->  t: f32[4, 4]
      [nn.Mod (compile bwd)] L['self'].foo.l1
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
        aten::sum.dim_IntList(t: f32[4, 4], [0], True)  ->  t: f32[1, 4]
        aten::view(t: f32[1, 4], [4])  ->  t: f32[4]
        aten::t(t: f32[4, 4])  ->  t: f32[4, 4]
  [aot_eager region (compile)] exit
    aten::detach(t: f32[4])  ->  t: f32[4]
    aten::detach(t: f32[4])  ->  t: f32[4]
    aten::detach(t: f32[4, 4])  ->  t: f32[4, 4]
    aten::detach(t: f32[4])  ->  t: f32[4]
    aten::detach(t: f32[4, 4])  ->  t: f32[4, 4]
    aten::detach(t: f32[4, 4])  ->  t: f32[4, 4]""",
        )

    def test_record_function(self):
        # [FUNCTION DOC] Compiles a matmul-add-sum function with aot_eager inside a profiler
        # record_function scope named FWD, then runs backward. The expected dump shows the
        # outer record-function entry wrapping the forward compile region, while a record_function
        # inside the compiled function is ignored.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Computes x @ y, adds 1 inside a profiler scope, returns sum.
            z = x @ y
            with torch.profiler.record_function("this_is_ignored"):
                z = z + 1
            return z.sum()

        x = torch.randn(8, 4, requires_grad=True)
        y = torch.randn(4, 2, requires_grad=True)
        with DebugMode(run_compile_with_interpreter=True) as debug_mode:
            with torch.profiler.record_function("FWD"):
                out = torch.compile(fn, backend="aot_eager")(x, y)
            out.backward()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  [record function] FWD
    [aot_eager region (compile)] enter
      aten::mm(t: f32[8, 4], t: f32[4, 2])  ->  t: f32[8, 2]
      aten::add.Tensor(t: f32[8, 2], 1)  ->  t: f32[8, 2]
      aten::sum(t: f32[8, 2])  ->  t: f32[]
      aten::t(t: f32[8, 4])  ->  t: f32[4, 8]
      aten::t(t: f32[4, 2])  ->  t: f32[2, 4]
    [aot_eager region (compile)] exit
      aten::detach(t: f32[4, 8])  ->  t: f32[4, 8]
      aten::detach(t: f32[2, 4])  ->  t: f32[2, 4]
    aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)  ->  t: f32[]
  [aot_eager region (compile)] enter
    aten::expand(t: f32[], [8, 2])  ->  t: f32[8, 2]
    aten::mm(t: f32[4, 8], t: f32[8, 2])  ->  t: f32[4, 2]
    aten::mm(t: f32[8, 2], t: f32[2, 4])  ->  t: f32[8, 4]
  [aot_eager region (compile)] exit
    aten::detach(t: f32[8, 4])  ->  t: f32[8, 4]
    aten::detach(t: f32[4, 2])  ->  t: f32[4, 2]""",
        )

    def test_in_place_mutation(self):
        # [FUNCTION DOC] Runs a module doing an add and then an in-place add on an integer
        # tensor with output recording and input hashing. Expected dump shows input hash taken
        # before each op, so the in-place op sees the already incremented value.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with only a forward method (no state).
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 1, then adds 1 in place to the result; returns it.
                y = x + 1
                y.add_(1)
                return y

        mod = Foo()
        inp = torch.tensor(1)
        with (
            DebugMode(record_output=True) as debug_mode,
            DebugMode.log_tensor_hashes(hash_inputs=True),
        ):
            _ = mod(inp)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    aten::add.Tensor(t: i64[], 1)  ->  t: i64[]  # {'input_hash': ((1.0, None), {}), 'hash': 2.0}
    aten::add_.Tensor(t: i64[], 1)  ->  t: i64[]  # {'input_hash': ((2.0, None), {}), 'hash': 3.0}""",
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_triton_kernel_logs(self):
        # [FUNCTION DOC] GPU and triton only. Compiles a function launching an autotuned Triton
        # add kernel under DebugMode and checks at least one Triton kernel call was recorded and
        # its rendering contains the triton marker.
        import triton

        from torch.testing._internal.triton_utils import add_kernel_autotuned

        def call_triton(x, y):
            # [FUNCTION DOC] HELPER: Allocates a zero output like x and launches the autotuned
            # add kernel over its elements (grid from the block size); returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        x = torch.randn(128, device=GPU_TYPE)
        y = torch.randn(128, device=GPU_TYPE)

        with DebugMode() as debug_mode:
            torch.compile(call_triton)(x, y)

        triton_calls = [
            op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)
        ]
        self.assertGreater(len(triton_calls), 0)
        self.assertIn("[triton]", triton_calls[0].render([]))

    def test_check_hash_mismatches(self):
        # [FUNCTION DOC] Two identical hashed runs of sin and sum give no mismatches. A run on
        # different data gives exactly two mismatches, for the sin and the sum ops in order.
        x = torch.randn(64, 64, device=GPU_TYPE)
        x_different = torch.randn(64, 64, device=GPU_TYPE)

        # Identical runs should have no mismatches
        with DebugMode() as dm1, DebugMode.log_tensor_hashes():
            x.sin().sum()
        with DebugMode() as dm2, DebugMode.log_tensor_hashes():
            x.sin().sum()
        mismatches = DebugMode.check_hash_mismatches(dm1.logs, dm2.logs)
        self.assertEqual(len(mismatches), 0)

        # Different inputs should produce hash mismatches
        with DebugMode() as dm3, DebugMode.log_tensor_hashes():
            x_different.sin().sum()

        # Check that mismatches are detected
        mismatches = DebugMode.check_hash_mismatches(dm1.logs, dm3.logs)
        self.assertEqual(len(mismatches), 2)
        self.assertEqual(
            [call["call"] for call in mismatches], ["aten::sin", "aten::sum"]
        )

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 2**26,
        "Being conservative, test peak memory is 25MB?",
    )
    def test_tensor_hash_redistribute(self):
        # [FUNCTION DOC] Skipped without CUDA or with under 64MB device memory. Redistributes a
        # sharded all-ones DTensor to replicated with hashing and checks the single all-gather
        # record's hash equals the total element count across the world (hashing a collective
        # gives the correct value).
        # test that hashing collectives gives correct results
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        local_tensor = torch.ones(2**18, device=self.device_type)
        dt = DTensor.from_local(local_tensor, mesh, [Shard(0)], run_check=False)

        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
            dt.redistribute(mesh, [Replicate()])

        # Find all_gather hash
        all_gather_logs = [
            op
            for op in debug_mode.logs
            if isinstance(op, _OpCall)
            and op.op == torch.ops._c10d_functional.all_gather_into_tensor.default
        ]
        self.assertEqual(len(all_gather_logs), 1)
        actual_hash = all_gather_logs[0].log["hash"]
        self.assertEqual(actual_hash, float(local_tensor.numel() * self.world_size))

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_check_triton_hash_mismatches(self):
        # [FUNCTION DOC] GPU and triton only. Runs the same Triton add with hashed inputs twice
        # with a differing second operand. Comparing logs with inputs must report mismatches of
        # the triton kernel type, covering both input-hash and output-hash mismatches.
        import triton

        from torch.testing._internal.triton_utils import add_kernel_autotuned

        def call_triton(x, y):
            # [FUNCTION DOC] HELPER: Allocates a zero output like x and launches the autotuned
            # add kernel over its elements (grid from the block size); returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        a = torch.randn(128, device=GPU_TYPE)
        b = torch.randn(128, device=GPU_TYPE)
        c = torch.randn(128, device=GPU_TYPE)

        # Run with hash logging to verify triton kernels can be hashed
        with DebugMode() as dm_t1, DebugMode.log_tensor_hashes(hash_inputs=True):
            torch.compile(call_triton)(a, b)

        # Different inputs should have different hashes in triton kernels
        with DebugMode() as dm_t2, DebugMode.log_tensor_hashes(hash_inputs=True):
            torch.compile(call_triton)(a, c)

        # Compare triton kernel hashes
        mismatches = DebugMode.check_hash_mismatches(
            dm_t1.logs, dm_t2.logs, compare_inputs=True
        )
        triton_mismatches = [m for m in mismatches if m["call_type"] == "triton kernel"]
        self.assertGreater(len(triton_mismatches), 0)

        # check both input & output hash mismatches are detected
        self.assertGreater(len([m for m in triton_mismatches if m["is_input_hash"]]), 0)
        self.assertGreater(
            len([m for m in triton_mismatches if not m["is_input_hash"]]), 0
        )

    def test_check_structure_mismatches(self):
        # [FUNCTION DOC] Compares logs with different operators (sin vs cos) and with different
        # lengths (sin vs sin then cos); expects check_hash_mismatches to raise ValueError with
        # "Operators don't match" and "Log lengths don't match" respectively.
        x = torch.randn(32, 32, device=self.device_type)

        with DebugMode() as dm1, DebugMode.log_tensor_hashes():
            x.sin()
        with DebugMode() as dm2, DebugMode.log_tensor_hashes():
            x.cos()
        with DebugMode() as dm3, DebugMode.log_tensor_hashes():
            x.sin().cos()

        with self.assertRaisesRegex(ValueError, "Operators don't match"):
            DebugMode.check_hash_mismatches(dm1.logs, dm2.logs)

        with self.assertRaisesRegex(ValueError, "Log lengths don't match"):
            DebugMode.check_hash_mismatches(dm1.logs, dm3.logs)

    def test_pretty_print_dtensor_make_fx(self):
        # [FUNCTION DOC] Traces a forward and backward matmul of DTensors with make_fx in fake
        # mode, removes dead nodes, recompiles and prints the graph as text. The text must show a
        # DTensor-typed mm node with shape and placement annotation.
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        A = torch.randn(8, 32)
        B = torch.randn(32, 32)
        dA = distribute_tensor(A, mesh, [Shard(0)]).requires_grad_()
        dB = distribute_tensor(B, mesh, [Replicate()]).requires_grad_()

        def f(dA, dB):
            # [FUNCTION DOC] HELPER: Multiplies the matrices, sums, runs backward and returns the
            # gradients of both inputs.
            dy = dA @ dB
            loss = dy.sum()
            loss.backward()
            return dA.grad, dB.grad

        # We actually need the tracing_mode='fake' here, or to trace under a FakeTensorMode.
        # make_fx has some logic to ensure we don't accidentally stash real tensors in the graph
        # so we won't stash our DTensors properly if they don't hold Fake inner tensors
        gm = make_fx(f, tracing_mode="fake")(dA, dB)
        # DCE isn't necessary here, there were just a lot of dead detach() nodes that spammed the graph
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # Colored is nice for actual viewing, not using in this test though
        gm_str = gm.print_readable(colored=False, print_output=False)
        self.assertTrue('"DTensor(f32[8, 32], S(0))" = torch.ops.aten.mm' in gm_str)

    @torch._functorch.config.patch(guess_tangent_strides_as_outputs=True)
    def test_invoke_subgraph(self):
        # [FUNCTION DOC] Compiles a function calling a nested compile region twice and compares
        # forward and backward results with eager. The expected dump shows invoke_subgraph calls
        # with enter and exit annotations for the forward and backward subgraphs; gradients match.
        # Test that DebugMode can trace the operations inside
        # invoke_subgraph HOP

        @torch.compiler.nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Nested compile region; returns sin of x.
            a = torch.sin(x)
            return a

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies gn, multiplies by 2, applies gn again, returns it.
            foo1 = gn(x)
            y = foo1 * 2
            foo2 = gn(y)
            return foo2

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        x.grad = None
        x_clone.grad = None

        ref = fn(x)
        ref.sum().backward()

        with DebugMode() as debug_mode:
            res = opt_fn(x_clone)
            res.sum().backward()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, t: f32[8, 8])  ->  ('t: f32[8, 8]', 't: f32[8, 8]')
    [annotate] [enter InvokeSubgraph HOP] partitioned_fw_subgraph_0_0
      aten::sin(t: f32[8, 8])  ->  t: f32[8, 8]
    [annotate] [exit InvokeSubgraph HOP] partitioned_fw_subgraph_0_0
    aten::mul.Tensor(t: f32[8, 8], 2)  ->  t: f32[8, 8]
    torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, t: f32[8, 8])  ->  ('t: f32[8, 8]', 't: f32[8, 8]')
    [annotate] [enter InvokeSubgraph HOP] partitioned_fw_subgraph_0_0
      aten::sin(t: f32[8, 8])  ->  t: f32[8, 8]
    [annotate] [exit InvokeSubgraph HOP] partitioned_fw_subgraph_0_0
    aten::sum(t: f32[8, 8])  ->  t: f32[]
    aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)  ->  t: f32[]
    aten::expand(t: f32[], [8, 8])  ->  t: f32[8, 8]
    aten::empty_strided([8, 8], [8, 1], dtype=torch.float32, layout=torch.strided, device=cpu, pin_memory=False)  ->  t: f32[8, 8]
    aten::copy_(t: f32[8, 8], t: f32[8, 8])  ->  t: f32[8, 8]
    torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, t: f32[8, 8], t: f32[8, 8])  ->  ('t: f32[8, 8]',)
    [annotate] [enter InvokeSubgraph HOP] partitioned_bw_subgraph_0_0
      aten::cos(t: f32[8, 8])  ->  t: f32[8, 8]
      aten::mul.Tensor(t: f32[8, 8], t: f32[8, 8])  ->  t: f32[8, 8]
    [annotate] [exit InvokeSubgraph HOP] partitioned_bw_subgraph_0_0
    aten::mul.Tensor(t: f32[8, 8], 2)  ->  t: f32[8, 8]
    torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, t: f32[8, 8], t: f32[8, 8])  ->  ('t: f32[8, 8]',)
    [annotate] [enter InvokeSubgraph HOP] partitioned_bw_subgraph_0_0
      aten::cos(t: f32[8, 8])  ->  t: f32[8, 8]
      aten::mul.Tensor(t: f32[8, 8], t: f32[8, 8])  ->  t: f32[8, 8]
    [annotate] [exit InvokeSubgraph HOP] partitioned_bw_subgraph_0_0
    aten::detach(t: f32[8, 8])  ->  t: f32[8, 8]""",
            ignore_comments=True,
        )

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_nested_invoke_subgraph(self):
        # [FUNCTION DOC] With single-use subgraph inlining disabled, compiles a function whose
        # region calls an inner region. Output equals eager, and the expected dump shows the inner
        # invoke_subgraph nested within the outer one.
        # Test that DebugMode can trace the operations inside
        # invoke_subgraph HOP
        @torch.compiler.nested_compile_region
        def ggn(x):
            # [FUNCTION DOC] HELPER: Inner compile region; returns cos of x.
            a = torch.cos(x)
            return a

        @torch.compiler.nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Outer region; adds 1, applies ggn, returns sin of the result.
            x = x + 1
            x = ggn(x)
            a = torch.sin(x)
            return a

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the outer region gn and doubles the result.
            foo1 = gn(x)
            y = foo1 * 2
            return y

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        x = torch.randn(8, 8)

        ref = fn(x)

        with DebugMode() as debug_mode:
            res = opt_fn(x)
        self.assertEqual(ref, res)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    torch.ops.higher_order.invoke_subgraph(subgraph_1, t: f32[8, 8])  ->  ('t: f32[8, 8]',)
    [annotate] [enter InvokeSubgraph HOP] subgraph_1
      aten::add.Tensor(t: f32[8, 8], 1)  ->  t: f32[8, 8]
      torch.ops.higher_order.invoke_subgraph(subgraph_0, t: f32[8, 8])  ->  ('t: f32[8, 8]',)
      [annotate] [enter InvokeSubgraph HOP] subgraph_0
        aten::cos(t: f32[8, 8])  ->  t: f32[8, 8]
      [annotate] [exit InvokeSubgraph HOP] subgraph_0
      aten::sin(t: f32[8, 8])  ->  t: f32[8, 8]
    [annotate] [exit InvokeSubgraph HOP] subgraph_1
    aten::mul.Tensor(t: f32[8, 8], 2)  ->  t: f32[8, 8]""",
            ignore_comments=True,
        )


class TestDebugModeUtils(TestCase):
    # [CLASS DOC] Tests for standalone hashing utilities of DebugMode; stateless.
    # KNOWN ISSUE: the docstring mentions the NCCL backend but this class only tests hashing.
    """Test DebugMode with NCCL backend without using DTensor."""

    def test_hash_empty_tensor(self):
        # [FUNCTION DOC] Hashes an empty tensor with hash_tensor_fn; must not raise, returning a
        # tensor by default and a Python integer when scalar output is requested.
        t = torch.tensor([])
        # hash tensor fn should not error out with empty tensor
        out = torch.utils._debug_mode.hash_tensor_fn(t)
        self.assertTrue(isinstance(out, torch.Tensor))
        out = torch.utils._debug_mode.hash_tensor_fn(t, use_scalar=True)
        self.assertTrue(isinstance(out, int))


class TestDTensorDebugModeNCCLBackend(MultiProcessTestCase):
    # [CLASS DOC] Multi-process (2 ranks) NCCL tests that DebugMode logs and hashes plain and
    # functional all-gather collectives. Each test creates and destroys its own process group.
    @property
    def world_size(self):
        # [FUNCTION DOC] Number of spawned ranks: always 2 (collectives need at least two).
        return 2  # Need at least 2 ranks for collectives

    def setUp(self):
        # [FUNCTION DOC] Runs base setup then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def _init_process_group(self):
        # [FUNCTION DOC] HELPER: In each worker, selects the CUDA device equal to the rank and
        # creates an NCCL process group using a file store; stores the device string.
        """Initialize NCCL process group for each spawned process."""
        torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self.device = f"cuda:{self.rank}"

    def _destroy_process_group(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group.
        """Destroy the process group."""
        dist.destroy_process_group()

    def tearDown(self):
        # [FUNCTION DOC] Runs base teardown, then deletes the file-store file, ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_allgather_base(self):
        # [FUNCTION DOC] Needs NCCL and 2 GPUs. Each rank all-gathers a tensor filled with rank+1
        # under DebugMode with input hashing. The debug string must name the all-gather-base op,
        # the last log's output hash must equal the gathered tensor's norm hash, and each rank's
        # slice must hold its contribution.
        self._init_process_group()
        tensor = torch.ones(10, 10, device=torch.device(self.device)) * (self.rank + 1)
        # Output size must be world_size * input_size
        output_tensor = torch.zeros(
            10 * self.world_size, 10, device=torch.device(self.device)
        )

        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes(hash_inputs=True):
            dist.all_gather_single(output_tensor, tensor)

        self.assertTrue("c10d::_allgather_base_" in debug_mode.debug_string())

        hash_ = lambda x: norm_hash_fn(x, use_scalar=True)  # noqa: E731

        self.assertEqual(debug_mode.operators[-1].log["hash"][0], hash_(output_tensor))

        # Verify each rank's contribution
        for i in range(self.world_size):
            expected_slice = torch.ones(10, 10, device=self.device) * (i + 1)
            self.assertEqual(output_tensor[i * 10 : (i + 1) * 10], expected_slice)

        self._destroy_process_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_allgather_base_async_op(self):
        # [FUNCTION DOC] Same as the all-gather test but with the async flag: waits on the work
        # handle, checks the op is logged and the input hash matches. The output hash check is
        # disabled because the hash is taken before the async op completes.
        # KNOWN ISSUE: output hash is incorrect for async collectives (see the TODO in the code).
        """Test all_gather_into_tensor with async_op=True."""
        self._init_process_group()
        tensor = torch.ones(10, 10, device=torch.device(self.device)) * (self.rank + 1)
        # Output size must be world_size * input_size
        output_tensor = torch.zeros(
            10 * self.world_size, 10, device=torch.device(self.device)
        )

        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes(hash_inputs=True):
            # Call with async_op=True returns a work handle
            work = dist.all_gather_single(output_tensor, tensor, async_op=True)
            # Wait for the async operation to complete
            work.wait()

        self.assertTrue("c10d::_allgather_base_" in debug_mode.debug_string())
        hash_ = lambda x: norm_hash_fn(x, use_scalar=True)  # noqa: E731

        # TODO: the output hash not correct for async op, we should either not log the hash
        # or wait before logging.
        # https://github.com/pytorch/pytorch/pull/169295
        # self.assertEqual(debug_mode.operators[-1].log["hash"][0], hash_(output_tensor))
        self.assertEqual(
            debug_mode.operators[-1].log["input_hash"][0][1], hash_(tensor)
        )

        # Verify each rank's contribution
        for i in range(self.world_size):
            expected_slice = torch.ones(10, 10, device=self.device) * (i + 1)
            self.assertEqual(output_tensor[i * 10 : (i + 1) * 10], expected_slice)

        self._destroy_process_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_allgather_functional_with_async_collective_tensor(self):
        # [FUNCTION DOC] Uses the functional all-gather (returns an async collective tensor) under
        # DebugMode with hashing, waits for the result, and checks the last log's hash equals the
        # result's norm hash, the functional op name is logged, and shape and per-rank slices
        # are right.
        self._init_process_group()
        tensor = torch.ones(10, 10, device=torch.device(self.device)) * (self.rank + 1)

        # Use functional collectives which return AsyncCollectiveTensor
        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
            result = _functional_collectives.all_gather_single(
                tensor, gather_dim=0, group=dist.group.WORLD
            )

        result = result.wait()
        hash_ = lambda x: norm_hash_fn(x, use_scalar=True)  # noqa: E731

        self.assertEqual(debug_mode.operators[-1].log["hash"], hash_(result))

        self.assertTrue(
            "_c10d_functional::all_gather_into_tensor" in debug_mode.debug_string()
        )

        # Verify the result shape - should be world_size times bigger
        self.assertEqual(result.shape[0], tensor.shape[0] * self.world_size)
        # Verify each rank's contribution
        for i in range(self.world_size):
            expected_slice = torch.ones(10, 10, device=self.device) * (i + 1)
            self.assertEqual(result[i * 10 : (i + 1) * 10], expected_slice)

        self._destroy_process_group()


instantiate_parametrized_tests(TestDTensorDebugMode)


if __name__ == "__main__":
    run_tests()
