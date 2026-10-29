# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for PipelineStage construction and runtime behavior. Contains a CPU
# single-process test of metadata inference restoring module buffers, two forward-hook
# factories used to corrupt stage outputs, a multi-process GPU test class (tracer, tracer with
# kwargs, manual stage building, custom weight-gradient builder, output chunk memory) and a
# negative test class (shape/dtype mismatch errors, dw_builder misuse). Module constants: d_hid
# 512, batch_size 256, chunks 8; device_type and backend come from the current accelerator
# (cpu if none). Import-time side effect: seeds torch with 0 and registers the parametrized
# StageTest tests.
import os
import tempfile

from model_registry import ExampleCode, ModelWithKwargs, MultiMLP

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    build_stage,
    pipeline,
    PipelineStage,
    ScheduleGPipe,
)
from torch.distributed.pipelining._utils import PipeliningMetadataError
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
    TestCase,
)
from torch.utils._pytree import tree_map_only


d_hid = 512
batch_size = 256
chunks = 8

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)

torch.manual_seed(0)


class PipelineStageMetadataInferenceTest(TestCase):
    # [CLASS DOC] Single-process (gloo, world size 1) test of dynamic metadata inference.
    def test_dynamic_metadata_inference_restores_module_buffers(self):
        # [FUNCTION DOC] Builds a one-stage GPipe schedule over a module whose forward bumps a
        # counter buffer and whose output hook bumps it again in backward. Initializes a file
        # based gloo group if none exists (destroyed afterwards), calls the schedule's private
        # _initialize_stage with an input requiring grad and a target, then asserts the counter
        # and scale buffers equal their initial values (inference must restore buffers).
        class BufferMutatingModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer and counter and scale buffers;
            # forward mutates counter without grad and, when output requires grad, registers a
            # backward hook that also mutates counter. Used to detect un-restored buffers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 4x4 Linear and buffers counter (zeros) and
                # scale (ones), both of size 4.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.register_buffer("counter", torch.zeros(4))
                self.register_buffer("scale", torch.ones(4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 1 to counter without grad, returns linear(x)
                # times scale, and attaches the backward hook if the output requires grad.
                with torch.no_grad():
                    self.counter.add_(1)
                out = self.linear(x) * self.scale
                if out.requires_grad:
                    out.register_hook(self._backward_hook)
                return out

            def _backward_hook(self, grad):
                # [FUNCTION DOC] HELPER: Gradient hook that adds 10 to counter without grad and
                # returns the gradient unchanged.
                with torch.no_grad():
                    self.counter.add_(10)
                return grad

        device = torch.device("cpu")
        init_pg = not dist.is_initialized()
        with tempfile.TemporaryDirectory() as tmpdir:
            if init_pg:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{os.path.join(tmpdir, 'pg')}",
                    rank=0,
                    world_size=1,
                )
            try:
                mod = BufferMutatingModule().to(device)
                stage = PipelineStage(
                    mod,
                    stage_index=0,
                    num_stages=1,
                    device=device,
                )
                schedule = ScheduleGPipe(
                    stage,
                    n_microbatches=1,
                    loss_fn=lambda out, target: out.sum() + target.sum() * 0,
                )

                initial_counter = mod.counter.clone()
                initial_scale = mod.scale.clone()
                x = torch.randn(2, 4, device=device, requires_grad=True)
                target = torch.zeros((), device=device)

                # This exercises the full metadata-inference lifecycle. The
                # scale buffer is saved by autograd, so restoring buffers before
                # backward metadata inference would bump its version counter.
                schedule._initialize_stage((x,), {}, target=target)

                self.assertEqual(mod.counter, initial_counter)
                self.assertEqual(mod.scale, initial_scale)
            finally:
                if init_pg:
                    dist.destroy_process_group()


def get_dtype_change_hook(new_dtype):
    # [FUNCTION DOC] HELPER: Returns a forward hook that casts every tensor in the module
    # output (any pytree) to new_dtype, simulating mixed precision.
    """A simple hook for simulating mixed precision"""

    def dtype_change_hook(module, input, output):
        # [FUNCTION DOC] HELPER: Forward hook returning output with all tensors cast to
        # new_dtype; module and input are ignored.
        def f(x):
            # [FUNCTION DOC] HELPER: Casts tensor x to new_dtype.
            return x.to(new_dtype)

        return tree_map_only(torch.Tensor, f, output)

    return dtype_change_hook


def get_flatten_hook():
    # [FUNCTION DOC] HELPER: Returns a forward hook that flattens every tensor in the module
    # output to one dimension, simulating a wrong output shape.
    """A simple hook for simulating wrong model output shape"""

    def flatten_hook(module, input, output):
        # [FUNCTION DOC] HELPER: Forward hook returning output with all tensors flattened;
        # module and input are ignored.
        def f(x):
            # [FUNCTION DOC] HELPER: Flattens tensor x.
            return x.flatten()

        return tree_map_only(torch.Tensor, f, output)

    return flatten_hook


class StageTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process continuous test (one rank per accelerator, world size from the
    # harness) for PipelineStage with a GPipe schedule. Provides backend and device hooks
    # and tests that need NCCL/XCCL and 2+ accelerators.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level distributed backend name.
        # Testing with NCCL backend
        return backend

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level device type string.
        return device_type

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns the device for this rank (device type with the rank
        # as index).
        return torch.device(device_type, self.rank)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ModelClass", [ExampleCode, MultiMLP])
    def test_tracer(self, ModelClass):
        # [FUNCTION DOC] Parametrized over ExampleCode and MultiMLP. Traces the model with
        # pipeline, builds this rank's stage via pipe.build_stage, runs a GPipe schedule
        # (rank 0 feeds the input) and on the last rank compares output to eager (loose
        # tolerances). Raises AssertionError if stage state_dict keys are not a subset of the
        # original model keys.
        mod = ModelClass(d_hid, self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        x_mb = x.chunk(chunks)[0]

        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            # [FUNCTION DOC] HELPER: Rank 0 steps the schedule with x; other ranks step with
            # no input. Returns the step result.
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        out = _run_step(x)
        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x)
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # Test qualname mapping
        submod_keys = stage.submod.state_dict().keys()
        # Confirm keys are consistent with original model
        old_keys = mod.state_dict().keys()
        if not all(k in old_keys for k in submod_keys):
            raise AssertionError(
                f"Some keys not found in old_keys: {[k for k in submod_keys if k not in old_keys]}"
            )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ModelClass", [ModelWithKwargs])
    def test_tracer_kwargs(self, ModelClass):
        # [FUNCTION DOC] Like test_tracer but for ModelWithKwargs: traces with a keyword
        # microbatch argument y, builds the stage with build_stage from the stage module and
        # pipe info, runs the schedule (rank 0 passes x and y), compares the last rank output
        # to eager, and checks stage state_dict keys are a subset of the original keys.
        mod = ModelClass(d_hid, self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)

        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        stage_mod = pipe.get_stage_module(self.rank)

        # Test build_stage
        stage = build_stage(
            stage_mod,
            self.rank,
            pipe.info(),
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        if self.rank == 0:
            out = schedule.step(x, y=y)
        else:
            out = schedule.step()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # Test qualname mapping
        submod_keys = stage.submod.state_dict().keys()
        # Confirm keys are consistent with original model
        old_keys = mod.state_dict().keys()
        if not all(k in old_keys for k in submod_keys):
            raise AssertionError(
                f"Some keys not found in old_keys: {[k for k in submod_keys if k not in old_keys]}"
            )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_manual(self):
        # [FUNCTION DOC] Builds a PipelineStage manually from the rank-th layer of a MultiMLP,
        # runs a GPipe schedule over chunks and compares the last rank output to eager.
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            # [FUNCTION DOC] HELPER: Rank 0 steps the schedule with x; others step with none.
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        out = _run_step(x)
        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_custom_dw_with_fb_schedule(self):
        # [FUNCTION DOC] Gives the stage a dw_builder from a counting helper object and runs
        # GPipe with MSE loss (sum). Asserts the weight-gradient runner was invoked once per
        # chunk (the counter equals chunks) and that the last rank output matches eager.
        """Tests that separate weight grad function 'dw_runner' gets run under a schedule that's only aware of F/B."""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)

        class CustomState:
            # [CLASS DOC] HELPER: Holds a call counter i that dw_runner increments.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the counter i to 0.
                self.i = 0

            def dw_builder(self):
                # [FUNCTION DOC] HELPER: Returns a new dw_runner closure that prints and
                # increments the call counter each time it is called.
                """This simulates a function attached to a model with a custom backward.
                Each call to builder gives a new dw_runner that has some updated state to compute the latest dw.
                """

                def dw_runner():
                    # [FUNCTION DOC] HELPER: Prints the current call count and increments it.
                    # This inner function would be called by PipelineStage during `backward_weight_one_chunk`
                    print(f"dw called {self.i}th time")
                    self.i += 1

                return dw_runner

        cs = CustomState()

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            dw_builder=cs.dw_builder,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(
            stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
        )

        # Run
        def _run_step(x):
            # [FUNCTION DOC] HELPER: Rank 0 passes x, the last rank passes the target, middle
            # ranks pass nothing to the schedule step. Returns the step result.
            if self.rank == 0:
                return schedule.step(x)
            elif self.rank == self.world_size - 1:
                return schedule.step(target=target)
            else:
                return schedule.step()

        out = _run_step(x)

        self.assertEqual(cs.i, chunks)

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_output_chunks_memory_usage(self):
        # [FUNCTION DOC] Runs a GPipe schedule with MSE loss and checks stage.output_chunks
        # is empty initially, fwd_cache is empty after the step, only the last rank holds
        # output chunks, and clear_runtime_states empties output_chunks.
        # KNOWN ISSUE: the final assertion message says the last stage should store output
        # chunks, but it asserts the length is zero after clearing. The non-last-rank
        # assertion passes a function as the message, which unittest does not call.
        """Test that output_chunks doesn't store memory for non-first stages."""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")
        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )
        self.assertEqual(
            len(stage.output_chunks), 0, "output_chunks should be empty initially"
        )

        schedule = ScheduleGPipe(
            stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
        )

        def _run_step(x):
            # [FUNCTION DOC] HELPER: Rank 0 passes x, the last rank passes the target, middle
            # ranks pass nothing to the schedule step. Returns the step result.
            if self.rank == 0:
                return schedule.step(x)
            elif self.rank == self.world_size - 1:
                return schedule.step(target=target)
            else:
                return schedule.step()

        _run_step(x)

        # Verify fwd_cache is empty
        self.assertEqual(len(stage.fwd_cache), 0, "fwd_cache should be cleared")

        # Check output_chunks state after step
        if self.rank == self.world_size - 1:
            self.assertEqual(
                len(stage.output_chunks),
                chunks,
                "Last stage should store output chunks",
            )
        else:
            self.assertEqual(
                len(stage.output_chunks),
                0,
                lambda msg: f"{msg}\nNon-last stage (rank {self.rank}) should not store output chunks",
            )

        # Clear the schedule and stage caches
        stage.clear_runtime_states()
        if self.rank == self.world_size - 1:
            # Last stage should have output_chunks populated
            self.assertEqual(
                len(stage.output_chunks), 0, "Last stage should store output chunks"
            )


instantiate_parametrized_tests(StageTest)


class StageNegativeTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process continuous test class for error paths of PipelineStage
    # (shape/dtype validation and dw_builder misuse); needs NCCL/XCCL and 2+ accelerators.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level distributed backend name.
        return backend

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the module-level device type string.
        return device_type

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns the device for this rank (device type with the rank
        # as index).
        return torch.device(device_type, self.rank)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_shape_prop_mismatch(self):
        # [FUNCTION DOC] Builds a manual stage with runtime validation enabled and runs a
        # valid step first. On rank 0 then expects PipeliningMetadataError for a wrong batch
        # size (shape mismatch), an int32 input (dtype mismatch), a forward hook that
        # flattens the stage output (shape mismatch) and a hook casting output to bfloat16
        # (dtype mismatch). The second hook is left registered.
        """Tests shape prop errors are raised"""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )
        stage._runtime_validate = True

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            # [FUNCTION DOC] HELPER: Rank 0 steps the schedule with x; others step with none.
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        _run_step(x)

        if self.rank == 0:
            with self.assertRaisesRegex(PipeliningMetadataError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            with self.assertRaisesRegex(PipeliningMetadataError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # output of stage's mlp layer will be flattened by this hook, the stage should err
            handle = stage_mod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningMetadataError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            stage_mod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            with self.assertRaisesRegex(PipeliningMetadataError, "dtype mismatch"):
                _run_step(x)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_custom_dw_errors(self):
        # [FUNCTION DOC] Creates a stage with a dw_builder, forces its internal has-backward
        # flag on, and asserts calling backward_weight_one_chunk for chunk 0 raises
        # AssertionError mentioning backward_one_chunk (no backward was run first).
        """Tests expected errors are raised"""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        stage_with_dw_builder = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            dw_builder=lambda: None,
        )
        stage_with_dw_builder._has_backward = True
        with self.assertRaisesRegex(AssertionError, "backward_one_chunk"):
            stage_with_dw_builder.backward_weight_one_chunk(bwd_chunk_id=0)


if __name__ == "__main__":
    run_tests()
