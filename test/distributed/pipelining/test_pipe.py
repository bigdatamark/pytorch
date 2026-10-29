# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for torch.distributed.pipelining.pipeline, which traces a model containing
# pipe_split markers into a Pipe of sequential stage modules. Defines two example models
# (ExampleCode with multi-use parameters and skip connections, MultiMLP) and PipeTests, which
# check stage counts, output equivalence with the original model, and that stage state_dict
# names are a subset of the original names. Seeds the RNG at import time.
from model_registry import MLPModule, ModelWithParamAlias

import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


d_hid = 512
microbatch_size = 16

torch.manual_seed(0)


# Basic example
class ExampleCode(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model with two shared matmul parameters (mm_param1 used twice),
    # two linear layers and a skip connection, split by three pipe_split calls into 4 stages.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two d_hid x d_hid random matmul parameters and two
        # Linear layers.
        super().__init__()
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Runs matmul, add of y, relu, then (after a split) matmul with the
        # same parameter and lin1, (split) relu plus the skip connection and a second matmul,
        # (split) lin2 and relu. Returns the final activation tensor.
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


class MultiMLP(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model of four sequential MLPModule blocks separated by pipe_split
    # markers (4 stages); the last stage subtracts a second input y.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates four MLPModule submodules of width d_hid.
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Applies the four MLPs in order with a pipe_split between each
        # and returns the result minus y.
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x - y


EXPECTED_N_STAGES = {
    ExampleCode: 4,
    MultiMLP: 4,
    ModelWithParamAlias: 2,
}

# Currently, we don't enforce full set equality on the FQNs between the original
# and pipelined models, because in the multi-use param case, PP will deduplicate
# the FQNs from the state_dict.
# TODO
CHECK_FQN_SET_EQUALITY = False


class PipeTests(TestCase):
    # [CLASS DOC] Test case for pipeline() tracing and splitting; test methods build models,
    # trace them with microbatch example inputs and compare against eager execution.
    @parametrize("ModelClass", [ExampleCode, MultiMLP, ModelWithParamAlias])
    def test_model_split(self, ModelClass):
        # [FUNCTION DOC] Parametrized over three model classes. Traces the model with pipeline,
        # raises AssertionError if the stage count differs from EXPECTED_N_STAGES, if the pipe
        # output differs from eager output, or if any stage state_dict key is not in the
        # original model's state_dict. Full key-set equality is only checked when
        # CHECK_FQN_SET_EQUALITY is enabled (it is off). Prints progress messages.
        mod = ModelClass()
        x = torch.randn(microbatch_size, d_hid)
        y = torch.randn(microbatch_size, d_hid)

        pipe = pipeline(
            mod,
            mb_args=(x, y),
        )

        if pipe.num_stages != EXPECTED_N_STAGES[ModelClass]:
            raise AssertionError(
                f"nstages = {pipe.num_stages}, expect {EXPECTED_N_STAGES[ModelClass]}"
            )

        ref_out = mod(x, y)
        out = pipe(x, y)[0]
        torch.testing.assert_close(out, ref_out)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}")

        # Check qualname
        # state_dict.keys include both parameters and persistent buffers
        old_names = set(mod.state_dict().keys())
        new_names = set()
        for idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(idx)
            stage_fqns = set(stage_mod.state_dict().keys())
            if not stage_fqns.issubset(old_names):
                raise AssertionError(
                    f"stage_fqns {stage_fqns} is not a subset of old_names {old_names}"
                )
            new_names.update(stage_fqns)

        if CHECK_FQN_SET_EQUALITY:
            if old_names != new_names:
                raise AssertionError(f"""
            old names {old_names}
            new names {new_names}
            """)
        print("Qualname check passed")

    def test_pipeline_with_non_float_inputs(self):
        # [FUNCTION DOC] Traces a two-stage embedding model whose boolean mask and integer ids
        # are non-float tensors that cross or enter stages; pipeline must not try to set
        # requires_grad on them. Asserts 2 stages and that the pipe output matches eager.
        """Test that pipeline() handles non-float tensors crossing stage boundaries."""

        class EmbeddingModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Embedding (stage 0) followed by Linear scaled by a mask
            # (stage 1), split by one pipe_split.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates Embedding(100, 64) and Linear(64, 64).
                super().__init__()
                self.embed = torch.nn.Embedding(100, 64)
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, input_ids, mask):
                # [FUNCTION DOC] HELPER: Embeds ids, splits, applies linear and multiplies by
                # the mask cast to float; returns the resulting tensor.
                x = self.embed(input_ids)
                pipe_split()
                x = self.linear(x)
                x = x * mask.unsqueeze(-1).float()
                return x

        model = EmbeddingModel()
        input_ids = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        # This should not raise "only Tensors of floating point dtype can require gradients"
        pipe = pipeline(model, mb_args=(input_ids, mask))
        self.assertEqual(pipe.num_stages, 2)

        # Verify non-float tensors don't have requires_grad in metadata
        ref_out = model(input_ids, mask)
        out = pipe(input_ids, mask)[0]
        torch.testing.assert_close(out, ref_out)


instantiate_parametrized_tests(PipeTests)

if __name__ == "__main__":
    run_tests()
