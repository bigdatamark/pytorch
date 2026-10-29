# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Test that torch.distributed.pipelining.pipeline can split a Transformer-like
# stack of 8 MLP layers into 2 stages at a SplitPoint and that the resulting pipe is
# numerically equal to the original. Defines tiny model classes and module constants (hidden
# size 16, 8 layers, microbatch size 4) and instantiates device-generic tests (including XPU)
# at import time.
import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


d_hid = 16
n_layers = 8
microbatch_size = 4


class MLPModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Two Linear layers (d_hid to d_hid) with a ReLU between them.
    def __init__(self, d_hid):
        # [FUNCTION DOC] HELPER: Creates net1, relu and net2 for hidden size d_hid.
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1, relu, net2 in order and returns the result.
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TransformerLike(torch.nn.Module):
    # [CLASS DOC] HELPER: Model holding n_layers MLPModules in a Sequential called layers.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the Sequential of n_layers MLPModule(d_hid) blocks.
        super().__init__()
        self.layers = torch.nn.Sequential(*[MLPModule(d_hid) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs the input tensor through all layers and returns it.
        return self.layers(x)


class TransformerTests(TestCase):
    # [CLASS DOC] Device-generic test case; hw_classification marks it as needing an
    # accelerator. Holds the pipeline IR test.
    hw_classification = HardwareClassification.ACCELERATOR

    def test_ir(self, device):
        # [FUNCTION DOC] Builds TransformerLike on the device and traces it with pipeline,
        # splitting at the start of the middle layer into 2 stages. Raises AssertionError
        # if the stage count differs or the union of layer names across stage modules does
        # not equal the original. Finally asserts the pipe output matches the original
        # model output (assert_close). Prints progress messages.
        transformer = TransformerLike().to(device)
        x = torch.randn(microbatch_size, d_hid, device=device)

        # Split into 2 stages
        num_stages = 2
        split_spec = {f"layers.{n_layers // num_stages}": SplitPoint.BEGINNING}

        pipe = pipeline(
            transformer,
            (x,),
            split_spec=split_spec,
        )
        if pipe.num_stages != num_stages:
            raise AssertionError(f"{pipe.num_stages=}, expect {num_stages}")

        def get_layers(module):
            # [FUNCTION DOC] HELPER: Returns the child names of module.layers as a list.
            layers = [name for name, _ in module.layers.named_children()]
            return layers

        # Collect all layers in pipe
        layers = []
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            layers += get_layers(stage_mod)

        # Check layer completeness
        orig_layers = get_layers(transformer)
        if sorted(layers) != sorted(orig_layers):
            raise AssertionError(f"{layers} != {orig_layers}")
        print("Layers matched!")

        # Check equivalence
        ref = transformer(x)
        out = pipe(x)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


instantiate_device_type_tests(TransformerTests, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
