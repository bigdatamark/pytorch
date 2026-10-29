# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the pipelining IR unflattening. One test splits a two-block model
# (with pipe_split calls inside and between blocks) into 4 stages, checks stage parameter
# names exist in the original state dict and that the pipe output matches the model. A second
# class checks the helper that moves placeholder nodes in an fx graph ahead of get_attr nodes
# (regression for issue 162898). Device-generic tests are instantiated at import time for
# cpu, cuda, hpu and xpu.
import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


# Building block for model
class Block(torch.nn.Module):
    # [CLASS DOC] HELPER: Conv2d then two Linear layers with a ReLU; forward calls pipe_split
    # after lin0 (a stage boundary) and takes an optional constant tensor to add.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates conv (16 to 16 channels, 3x3, padding 1), lin0, relu
        # and lin1 (256 to 256).
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(256, 256)

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies conv and lin0, marks a stage split, adds constant to
        # x (the result is discarded, so the add has no effect on the output), then applies
        # lin1 and relu and returns the result.
        # KNOWN ISSUE: the add result is not assigned, so constant never affects the output.
        x = self.conv(x)
        x = self.lin0(x)
        pipe_split()
        x.add(constant)
        x = self.lin1(x)
        return self.relu(x)


# Full model
class M(torch.nn.Module):
    # [CLASS DOC] HELPER: Two Blocks in sequence with a pipe_split between them; with the
    # split inside each Block this yields 4 pipeline stages.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates block0 and block1.
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs block0, marks a stage split, runs block1; the constant
        # keyword is forwarded to both blocks. Returns the output tensor.
        x = self.block0(x, constant=constant)
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


class UnflattenTests(TestCase):
    # [CLASS DOC] Device-generic test case holding the unflatten end-to-end test.
    def test_unflatten(self, device):
        # [FUNCTION DOC] Traces M with pipeline using a 1x16x256x256 input and a constant
        # keyword tensor. Raises AssertionError unless there are 4 stages or if any stage
        # parameter name is missing from the original state dict. Then asserts the pipe
        # output equals the eager output. Prints progress messages.
        x = torch.randn(1, 16, 256, 256, device=device)
        constant = torch.ones(1, 16, 256, 256, device=device)

        mod = M().to(device)

        pipe = pipeline(
            mod,
            (x,),
            {"constant": constant},
        )

        if pipe.num_stages != 4:
            raise AssertionError(
                f"Expected pipe.num_stages == 4, got {pipe.num_stages}"
            )
        orig_state_dict = mod.state_dict()

        # Check qualnames
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            for param_name, _ in stage_mod.named_parameters():
                if param_name not in orig_state_dict:
                    raise AssertionError(f"{param_name} not in original state dict")
        print("Param qualname test passed")

        # Check equivalence
        ref = mod(x, constant)
        out = pipe(x, constant)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


class UnflattenPlaceholderOrderingTests(TestCase):
    # [CLASS DOC] Regression tests for the placeholder-ordering fix (issue 162898) of
    # _move_placeholders_to_front on hand-built fx graphs.
    """regression tests for https://github.com/pytorch/pytorch/issues/162898"""

    def _check(self, node_ops, expected_ops):
        # [FUNCTION DOC] HELPER: Builds a graph with one node per entry of node_ops
        # (placeholder or get_attr), adds an output, calls _move_placeholders_to_front,
        # lints the graph, and asserts the non-output node op sequence equals expected_ops.
        from torch.distributed.pipelining._IR import _move_placeholders_to_front

        g = torch.fx.Graph()
        for i, op in enumerate(node_ops):
            if op == "placeholder":
                g.placeholder(f"n{i}")
            elif op == "get_attr":
                g.create_node("get_attr", f"submod_{i}")
        g.output(None)

        _move_placeholders_to_front(g)
        g.lint()

        result = [n.op for n in g.nodes if n.op != "output"]
        self.assertEqual(result, expected_ops)

    def test_get_attr_between_placeholders(self):
        # [FUNCTION DOC] get_attr nodes sit between placeholders; all placeholders must end
        # up first, with the get_attr last.
        self._check(
            ["placeholder", "placeholder", "get_attr", "placeholder", "placeholder"],
            ["placeholder", "placeholder", "placeholder", "placeholder", "get_attr"],
        )

    def test_get_attr_before_all_placeholders(self):
        # [FUNCTION DOC] A get_attr precedes all placeholders; it must be moved after them.
        self._check(
            ["get_attr", "placeholder", "placeholder", "placeholder"],
            ["placeholder", "placeholder", "placeholder", "get_attr"],
        )

    def test_multiple_get_attrs(self):
        # [FUNCTION DOC] Several interleaved get_attr nodes; all move after the placeholders
        # while keeping their relative order.
        self._check(
            ["placeholder", "get_attr", "placeholder", "get_attr", "placeholder"],
            ["placeholder", "placeholder", "placeholder", "get_attr", "get_attr"],
        )

    def test_no_interleaving(self):
        # [FUNCTION DOC] Graph already ordered placeholders then get_attr; stays unchanged.
        self._check(
            ["placeholder", "placeholder", "placeholder", "get_attr"],
            ["placeholder", "placeholder", "placeholder", "get_attr"],
        )

    def test_only_placeholders(self):
        # [FUNCTION DOC] Graph with only placeholders; stays unchanged.
        self._check(
            ["placeholder", "placeholder", "placeholder"],
            ["placeholder", "placeholder", "placeholder"],
        )


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    UnflattenTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
