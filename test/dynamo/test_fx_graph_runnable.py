# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests checking that the fx_graph_runnable structured-trace artifact
# emitted when compiling is a standalone Python script that runs successfully in a fresh
# interpreter. Covers plain ops, dynamic shapes, toy modules, user-defined Triton kernels
# (nested, autotuned, global constants), fake-process-group collectives, DTensor, and
# multi-process-group repro generation. At import time, when Triton is available, it
# defines several Triton kernels and a global constexpr; when distributed is available it
# imports fake-store and DTensor helpers; and it creates the trace logger.
import io
import logging
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch._logging.structured
import torch.distributed as dist
from torch._inductor.codecache import WritableTempFile
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE
from torch.utils._triton import has_triton


if torch.distributed.is_available():
    from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

if has_triton():
    import triton
    import triton.language as tl

    GLOBAL_SCALE = 2
    GLOBAL_SCALE_CONSTEXPR: tl.constexpr = tl.constexpr(GLOBAL_SCALE)

    def init_to_zero(name):
        # [FUNCTION DOC] HELPER: Returns a pre-hook callable that zeroes the tensor stored under
        # the given argument name in the kernel argument mapping.
        return lambda nargs: nargs[name].zero_()

    # [FUNCTION DOC] HELPER: Triton kernel; each program handles one block of BLOCK_SIZE
    # elements, loads x and y with a bounds mask, and stores x minus y to the output.
    @triton.jit
    def subtract_kernel_inner(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x - y
        tl.store(output_ptr + offsets, output, mask=mask)

    # [FUNCTION DOC] HELPER: Triton kernel that only forwards its arguments to the inner
    # subtract kernel, used to test nested kernel calls.
    @triton.jit
    def nested_kernel_with_inner_call(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        subtract_kernel_inner(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE)

    # [FUNCTION DOC] HELPER: Triton kernel; per block, loads x and y with a bounds mask and
    # atomically adds their sum into the output (so the output must be pre-initialised).
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)

    # [FUNCTION DOC] HELPER: Autotuned (single config of block size 1024) variant of the
    # atomic-add kernel; pre and post hooks zero the output, keyed on the element count.
    # Per block, loads x and y with a bounds mask and atomically adds their sum to the output.
    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE": 1024},
                num_warps=4,
                num_stages=2,
                pre_hook=init_to_zero("output_ptr"),
            )
        ],
        pre_hook=init_to_zero("output_ptr"),
        post_hook=init_to_zero("output_ptr"),
        key=["n_elements"],
    )
    @triton.jit
    def add_kernel_autotune(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)

    autotuned_subtract = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        ],
        key=["n_elements"],
    )(subtract_kernel_inner)

    # [FUNCTION DOC] HELPER: Triton kernel that multiplies each element of x by the module
    # level GLOBAL_SCALE constant (bounds-masked, block-wise) and stores to the output.
    @triton.jit
    def ref_global(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = x * GLOBAL_SCALE
        tl.store(output_ptr + offsets, output, mask=mask)

    # [FUNCTION DOC] HELPER: Triton kernel that statically prints the global constexpr scale
    # at compile time and then calls the global-scale kernel on the same arguments.
    @triton.jit
    def nested_ref_global_constexpr(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        tl.static_print(f"SCALE={GLOBAL_SCALE_CONSTEXPR}")
        ref_global(x_ptr, output_ptr, n_elements, BLOCK_SIZE)


from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


class FxGraphRunnableArtifactFilter(logging.Filter):
    # [CLASS DOC] HELPER: Logging filter that accepts only trace records whose metadata
    # holds an artifact entry named fx_graph_runnable.
    def filter(self, record):
        # [FUNCTION DOC] HELPER: Returns True when the record metadata contains an artifact whose
        # name is fx_graph_runnable; raises AttributeError if the record has no metadata.
        return (
            "artifact" in record.metadata
            and record.metadata["artifact"]["name"] == "fx_graph_runnable"
        )


class StructuredTracePayloadFormatter(logging.Formatter):
    # [CLASS DOC] HELPER: Log formatter that outputs just the stripped structured-trace
    # payload text of a record.
    def format(self, record):
        # [FUNCTION DOC] HELPER: Returns the record payload with surrounding whitespace removed.
        return record.payload.strip()


trace_log = logging.getLogger("torch.__trace")


class ToyModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Small MLP (linear, relu, dropout 0.1, linear) used as a compile
    # target; sizes are constructor arguments.
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        # [FUNCTION DOC] HELPER: Builds two linear layers (input to hidden, hidden to output),
        # a ReLU and a dropout layer with probability 0.1.
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the first linear, relu, dropout, then second linear.
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@mock.patch.dict(os.environ, {"TRITON_ALLOW_NON_CONSTEXPR_GLOBALS": "1"})
class FxGraphRunnableTest(TestCase):
    # [CLASS DOC] Tests that the fx_graph_runnable artifact captured during compilation runs
    # standalone. The class patches the environment to allow non-constexpr Triton globals.
    # Each test compiles something, then checks the captured payload via the shared helper.
    def setUp(self):
        # [FUNCTION DOC] Resets dynamo, clears the structured-trace intern table, sets the trace
        # logger to debug level (saving the old level), and attaches a string-buffer handler with
        # the artifact filter and payload formatter so only fx_graph_runnable output is captured.
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Create a custom filter specifically for fx_graph_runnable entries
        self.filter = FxGraphRunnableArtifactFilter()

        # Create a separate buffer and handler for capturing fx_graph_runnable entries
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTracePayloadFormatter())
        self.handler.addFilter(self.filter)
        trace_log.addHandler(self.handler)

    def tearDown(self):
        # [FUNCTION DOC] Removes the capture handler and restores the trace logger level.
        # KNOWN ISSUE: does not call the parent tearDown.
        trace_log.removeHandler(self.handler)
        trace_log.setLevel(self.old_level)

    def _exec_and_verify_payload(self):
        # [FUNCTION DOC] HELPER: Takes the captured payload text, asserts it is non-empty and
        # contains a forward definition, writes it to a temporary script and runs it with the
        # current interpreter (captured output, 45 second timeout). Asserts exit code zero, with
        # stderr in the failure message. Raises on timeout.
        # Write captured payload & run it in a fresh Python process
        payload = self.buffer.getvalue().strip()
        self.assertTrue(payload, "Expected fx_graph_runnable payload but got nothing")
        self.assertIn("def forward", payload)  # sanity-check for actual FX code

        with WritableTempFile("w", suffix=".py") as tmp:
            tmp.write(payload)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name], capture_output=True, text=True, timeout=45
            )

            self.assertEqual(
                res.returncode,
                0,
                lambda msg: f"{msg}\nStandalone fx_graph_runnable failed:\nSTDERR:\n{res.stderr}",
            )

    # basic tests
    def test_basic_tensor_add(self):
        # [FUNCTION DOC] Compiles a function adding one to a random vector and verifies the
        # captured payload runs standalone.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        torch.compile(f)(torch.randn(4))  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_triton_kernel_autotune(self):
        # [FUNCTION DOC] Compiles a function that launches the autotuned atomic-add Triton kernel
        # on GPU tensors and verifies the payload runs. Skipped without Triton or a GPU.
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates a ones tensor like x, launches the autotuned add
            # kernel over a grid derived from the chosen block size, and returns the output.
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()

            def grid(
                meta,
            ):
                # [FUNCTION DOC] HELPER: Grid function: one-element tuple with the ceiling of
                # the element count over the config block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel_autotune[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        torch.compile(add)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_triton_kernel(self):
        # [FUNCTION DOC] Compiles a function that launches the plain atomic-add Triton kernel
        # with a fixed block size on GPU tensors and verifies the payload runs. Skipped without
        # Triton or a GPU.
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates a ones output and launches the add kernel over one
            # program per element, returning the output.
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = x.numel()
            add_kernel[n_elements,](x, y, output, n_elements, BLOCK_SIZE=4)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        torch.compile(add)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_nested_triton_kernel(self):
        # [FUNCTION DOC] Compiles a function launching a Triton kernel that calls another Triton
        # kernel, and verifies the payload runs. Skipped without Triton or a GPU.
        def subtract_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Launches the nested subtract kernel into an empty output and
            # returns it.
            output = torch.empty_like(x)
            n_elements = x.numel()
            nested_kernel_with_inner_call[(n_elements,)](
                x, y, output, n_elements, BLOCK_SIZE=1024
            )
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16) * 0.5

        torch.compile(subtract_nested)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_nested_and_autotuned_same_kernel(self):
        # [FUNCTION DOC] Compiles a function using both an autotuned wrapper and the plain
        # version of the same subtract kernel, and verifies the payload runs. Skipped without
        # Triton or a GPU.
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Runs the autotuned and the plain subtract kernels into two
            # outputs and returns their sum.
            output1 = torch.empty_like(x)
            output2 = torch.empty_like(x)
            n_elements = x.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Grid function: one-element tuple with the ceiling of
                # the element count over the block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            autotuned_subtract[grid](x, y, output2, n_elements)
            subtract_kernel_inner[(n_elements,)](
                x, y, output1, n_elements, BLOCK_SIZE=1024
            )
            return output1 + output2

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16) * 0.5

        torch.compile(f)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_multi_kernel_nesting_and_global_constexpr(self):
        # [FUNCTION DOC] Compiles a function launching four Triton kernels (global-constant
        # scaling, nested global-constexpr kernel, nested subtract, plain subtract) and verifies
        # the payload runs. Skipped without Triton or a GPU.
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Launches the four kernels into separate outputs and returns
            # the sum of the outputs.
            n_elements = x.numel()

            output1 = torch.empty_like(x)
            ref_global[(n_elements,)](x, output1, n_elements, BLOCK_SIZE=1024)

            output2 = torch.empty_like(x)
            nested_ref_global_constexpr[(n_elements,)](
                x, output2, n_elements, BLOCK_SIZE=1024
            )

            output3 = torch.empty_like(x)
            nested_kernel_with_inner_call[(n_elements,)](
                x, y, output3, n_elements, BLOCK_SIZE=1024
            )

            output4 = torch.empty_like(x)
            subtract_kernel_inner[(n_elements,)](
                x, y, output4, n_elements, BLOCK_SIZE=1024
            )

            return output1 + output2 + output3 + output4

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16) * 0.5

        torch.compile(f)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_two_inputs_matmul(self):
        # [FUNCTION DOC] Compiles a matrix multiply followed by relu on random inputs and
        # verifies the payload runs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns relu of the matrix product of a and b.
            return (a @ b).relu()

        a, b = torch.randn(2, 3), torch.randn(3, 4)
        torch.compile(f)(a, b)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_scalar_multiply(self):
        # [FUNCTION DOC] Compiles a function multiplying by two and verifies the payload runs.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input times two.
            return x * 2

        torch.compile(f)(torch.randn(5))  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    # testing dynamic shapes
    def test_dynamic_shapes_run(self):
        # [FUNCTION DOC] Marks both dimensions of a matrix as dynamic, compiles a function doing
        # a Gram-style product with relu, and verifies the payload runs.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu of x times its transpose.
            return (x @ x.transpose(0, 1)).relu()

        a = torch.randn(10, 12)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)

        torch.compile(f)(a)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_broadcast_add_dynamic(self):
        # [FUNCTION DOC] Marks one dimension of each broadcastable input as dynamic, compiles an
        # addition, and verifies the payload runs.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus twice y.
            return x + y * 2

        x = torch.randn(5, 1)
        y = torch.randn(1, 8)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)

        torch.compile(f)(x, y)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_toy_model_basic(self):
        # [FUNCTION DOC] Compiles a small ToyModel in eval mode (no dropout randomness) and
        # verifies the payload runs.
        model = ToyModel(input_size=8, hidden_size=16, output_size=4)
        model.eval()  # Set to eval mode to avoid dropout randomness

        x = torch.randn(3, 8)
        torch.compile(model)(x)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_toy_model_batch_processing(self):
        # [FUNCTION DOC] Compiles an eval-mode ToyModel on a larger batch and verifies the
        # payload runs.
        model = ToyModel(input_size=12, hidden_size=24, output_size=6)
        model.eval()

        x = torch.randn(16, 12)
        torch.compile(model)(x)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    def test_toy_model_dynamic_batch(self):
        # [FUNCTION DOC] Compiles an eval-mode ToyModel with a dynamic batch dimension and
        # verifies the payload runs.
        model = ToyModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()

        x = torch.randn(7, 10)
        torch._dynamo.mark_dynamic(x, 0)

        torch.compile(model)(x)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    # Distributed collectives tests with FakeProcessGroup
    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_reduce_collective(self):
        # [FUNCTION DOC] Initialises a fake two-rank process group, compiles a function doing an
        # all-reduce then doubling, always destroys the group, and verifies the payload runs.
        # Skipped without distributed support or in fbcode/sandcastle.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            # [FUNCTION DOC] HELPER: All-reduces x in place and returns it times two.
            dist.all_reduce(x)
            return x * 2

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_gather_collective(self):
        # [FUNCTION DOC] Same as the all-reduce test but compiles an all-gather into a list of
        # two buffers and sums them; verifies the payload runs. Skipped like the other
        # collective tests.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            # [FUNCTION DOC] HELPER: Gathers x into two empty buffers and returns their sum.
            output_tensors = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(output_tensors, x)
            return output_tensors[0] + output_tensors[1]

        try:
            x = torch.randn(3, 3)
            torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_broadcast_collective(self):
        # [FUNCTION DOC] Same pattern with a broadcast from rank zero followed by a sum;
        # verifies the payload runs. Skipped like the other collective tests.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            # [FUNCTION DOC] HELPER: Broadcasts x from source rank zero and returns its sum.
            dist.broadcast(x, src=0)
            return x.sum()

        try:
            x = torch.randn(5, 5)
            torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_reduce_scatter_collective(self):
        # [FUNCTION DOC] Same pattern with a reduce-scatter of a two-element input list into an
        # output buffer; verifies the payload runs. Skipped like the other collective tests.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            # [FUNCTION DOC] HELPER: Reduce-scatters a list of x and its clone into a new
            # buffer and returns the buffer.
            input_list = [x, x.clone()]
            output = torch.empty_like(x)
            dist.reduce_scatter(output, input_list)
            return output

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available"
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_dtensor_compile_redistribute(self):
        # [FUNCTION DOC] Under a fake two-rank group and a CPU device mesh, compiles a function
        # that builds two sharded DTensors from local tensors, multiplies them and redistributes
        # to replicated, then verifies the payload runs. Group is always destroyed afterward.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        mesh = DeviceMesh("cpu", list(range(2)))

        def f(x, y):
            # [FUNCTION DOC] HELPER: Wraps reshaped x and y as sharded DTensors, matmuls them,
            # redistributes the result to replicate, and returns the local tensor.
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        try:
            x = torch.arange(8, dtype=torch.float32)
            y = torch.arange(8, dtype=torch.float32)
            torch.compile(f)(x, y)  # noqa: UNSPECIFIED_BACKEND
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    def test_metrics_context(self):
        # [FUNCTION DOC] Regression test: with inductor trace and provenance tracking enabled,
        # compiles a simple function and verifies the payload runs (it previously failed due to
        # adding provenance metrics outside a metrics context).
        """
        When TORCH_COMPILE_DEBUG is set, provenance_tracking_level is set to 1, and
        the generated fx_graph_runnable crashed with,
        RuntimeError: Cannot add inductor_provenance outside of a MetricsContext
        """
        import torch._inductor.config as inductor_config

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input times two plus one.
            return x * 2 + 1

        # Enable provenance tracking to trigger the code path that adds metrics
        with inductor_config.patch(
            {"trace.enabled": True, "trace.provenance_tracking_level": 1}
        ):
            x = torch.randn(4, 4)
            torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
            self._exec_and_verify_payload()

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_expression(self):
        # [FUNCTION DOC] With dynamic shapes by default, compiles two adaptive average pooling
        # calls and verifies the payload runs (guards against emitting an invalid symbolic
        # size expression).
        """
        Test not emitting something like "s27*s53**2 = 36"
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns two adaptive average pool results (input, and input
            # plus one) with different output sizes.
            return torch.ops.aten._adaptive_avg_pool2d(
                x, (6, 6)
            ), torch.ops.aten._adaptive_avg_pool2d(x + 1, (2, 5))

        x = torch.randn(2, 4, 16, 16)
        torch.compile(f)(x)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_storage_nbytes_symint_mismatch(self):
        # [FUNCTION DOC] With dynamic shapes by default, builds a strided view whose storage
        # size symbols differ from the shape symbols, compiles a function on it, and verifies the
        # payload runs (all symbols must be defined in the script).
        """
        Test that symbols in storage nbytes are extracted when they differ from shape.

        When a tensor is a view created via as_strided where the storage comes from
        one tensor but the shape dimension comes from another tensor, the storage
        nbytes expression uses different symbols than the tensor shape. All symbols
        must be defined at the top of the generated repro script.
        """

        def f(view, weights):
            # [FUNCTION DOC] HELPER: Returns the sum of the view times weights with an added axis.
            return (view * weights.unsqueeze(1)).sum()

        # Create storage_src with dynamic shape (s0, s1)
        storage_src = torch.randn(32, 64)
        torch._dynamo.mark_dynamic(storage_src, 0)
        torch._dynamo.mark_dynamic(storage_src, 1)

        # Create weights with independent dynamic shape (s2)
        weights = torch.randn(100)
        torch._dynamo.mark_dynamic(weights, 0)

        # Create as_strided view: storage from storage_src, shape from weights
        flat = storage_src.flatten()
        s2 = weights.shape[0]
        view = flat.as_strided((s2, 16), (16, 1))

        torch.compile(f)(view, weights)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_repeat_interleave_with_output_size(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a function using repeat-interleave with a
        # data-dependent output size; verifies the payload runs, contains the forward and
        # repeat-interleave code, and includes the fixup comment and fill of the repeats.
        def f(data, repeats, output_size):
            # [FUNCTION DOC] HELPER: Expands repeats to indices with the item value as output size
            # and returns the data rows selected by those indices.
            indices = torch.repeat_interleave(repeats, output_size=output_size.item())
            return data[indices]

        num_segments = 128
        data = torch.randn(1000, 16)
        repeats = torch.randint(5, 15, (num_segments,), dtype=torch.int64)
        output_size = repeats.sum()

        torch.compile(f, dynamic=True)(data, repeats, output_size)  # noqa: UNSPECIFIED_BACKEND

        self._exec_and_verify_payload()

        # Verify the payload contains the repeat_interleave fixup
        payload = self.buffer.getvalue().strip()
        self.assertIn("def forward", payload)
        self.assertIn("repeat_interleave", payload)
        # Verify the fixup code is present
        self.assertIn("# Fixup: ensure sum(repeats) == output_size", payload)
        self.assertIn("_repeats.fill_", payload)

    def test_repeat_interleave_with_constant_output_size(self):
        # [FUNCTION DOC] Compiles a function using repeat-interleave with a constant output size
        # and verifies the payload runs and does not contain the repeats fixup comment.
        def f(data, repeats):
            # [FUNCTION DOC] HELPER: Repeat-interleaves with constant output size 1280 and returns
            # the selected data rows.
            # output_size is a constant, not a dynamic input
            indices = torch.repeat_interleave(repeats, output_size=1280)
            return data[indices]

        num_segments = 128
        data = torch.randn(1000, 16)
        repeats = torch.full((num_segments,), 10, dtype=torch.int64)

        torch.compile(f)(data, repeats)  # noqa: UNSPECIFIED_BACKEND
        self._exec_and_verify_payload()
        payload = self.buffer.getvalue().strip()
        self.assertNotIn("# Fixup: ensure sum(repeats) == output_size", payload)


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
class TestFxGraphRunnableMultiProcessGroup(TestCase):
    # [CLASS DOC] Tests standalone repro generation for graphs using several process groups.
    # Skipped in fbcode/sandcastle.
    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    def test_multiple_process_groups(self):
        # [FUNCTION DOC] Creates a fake four-rank world with two registered groups named tp and
        # dp, traces a function of functional collectives with make_fx, generates a standalone
        # repro, asserts it mentions fake group setup, both names and size two, then runs it in a
        # subprocess (60 second timeout) and asserts exit code zero. Destroys the default group.
        # KNOWN ISSUE: the temporary script file is created without deletion and never removed.
        import tempfile

        from torch._dynamo.repro.after_aot import generate_standalone_repro
        from torch.fx.experimental.proxy_tensor import make_fx

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=4, store=store)

        try:
            tp_pg = dist.new_group([0, 1])
            dp_pg = dist.new_group([0, 2])

            torch._C._distributed_c10d._register_process_group("tp", tp_pg)
            torch._C._distributed_c10d._register_process_group("dp", dp_pg)

            def f(x):
            # [FUNCTION DOC] HELPER: All-gathers over group tp, waits, all-reduces over group dp,
            # waits, and returns a one-element tuple with the result doubled.
                y = torch.ops._c10d_functional.all_gather_into_tensor(x, 2, "tp")
                z = torch.ops._c10d_functional.wait_tensor(y)
                w = torch.ops._c10d_functional.all_reduce(z, "sum", "dp")
                v = torch.ops._c10d_functional.wait_tensor(w)
                return (v * 2,)

            args = [torch.randn(4, 4)]
            gm = make_fx(f)(*args)
            repro = generate_standalone_repro(gm, args)

            self.assertIn("setup_fake_process_groups", repro)
            self.assertIn("'tp'", repro)
            self.assertIn("'dp'", repro)
            self.assertIn("'size': 2", repro)

            # Write to temp file and execute
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(repro)
                tmp.flush()
                result = subprocess.run(
                    [sys.executable, tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            self.assertEqual(
                result.returncode,
                0,
                lambda msg: f"{msg}\nGenerated repro failed to execute:\nSTDERR:\n{result.stderr}",
            )

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not (IS_FBCODE or IS_SANDCASTLE):
        # fbcode complains about not being able to find torch in subprocess
        run_tests()
