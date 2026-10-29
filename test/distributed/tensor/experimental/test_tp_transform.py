# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the experimental tensor_parallel_transformation, which rewrites an
# exported program so selected modules run tensor-parallel. Contains small dummy models and a
# multi-process DTensor test class that compares transformed output with the original model and
# counts the c10d functional collectives inserted in the graph. No import-time side effects.
from collections import defaultdict

import torch
from torch.distributed.tensor.experimental._tp_transform import (
    tensor_parallel_transformation,
)
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPListModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Model holding a ModuleList of num_mlps MLPs, each Linear(6, 18),
    # ReLU, Linear(18, 6) with optional bias on the last linear.
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3, bias=True):
        # [FUNCTION DOC] HELPER: Creates num_mlps Sequential MLPs; bias applies only to the
        # second linear of each MLP.
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        for _ in range(num_mlps):
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(6, 18),
                    torch.nn.ReLU(),
                    torch.nn.Linear(18, 6, bias=bias),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Keeps the first half of the input along dim 1 (chunk into 2),
        # runs it through every MLP in sequence, and adds a tensor of ones to the result.
        x = torch.chunk(x, 2, dim=1)[0]
        for mlp in self.mlps:
            x = mlp(x)
        return x + torch.ones_like(x)


class DummyModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Model with a Linear(3, 5) named fc and a constant buffer of bucket
    # boundaries; its forward uses searchsorted, an op the transform has no sharding rule for.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates fc and registers the boundaries buffer [-1, 0, 1].
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)
        self.register_buffer("boundaries", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the bucket index of every fc output value within the
        # boundaries buffer.
        return torch.searchsorted(self.boundaries, self.fc(x))


class TensorParallelTest(DTensorTestBase):
    # [CLASS DOC] Multi-process DTensor test case for tensor_parallel_transformation. Each test
    # exports a model, transforms it, and compares outputs and collective counts.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Only delegates to the base class setup.
        super().setUp()

    def assert_has_c10d_ops(
        self, gm: torch.fx.GraphModule, expected_ops_count: dict[str, int]
    ) -> None:
        # [FUNCTION DOC] HELPER: Walks the graph, counts call_function nodes whose target name
        # contains c10d_functional keyed by target string, and asserts the counts mapping equals
        # expected_ops_count exactly (so unexpected collectives also fail).
        actual_ops_count: dict[str, int] = defaultdict(int)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "c10d_functional" in str(node.target):
                    actual_ops_count[str(node.target)] += 1
        self.assertDictEqual(expected_ops_count, actual_ops_count)

    @with_comms
    def test_tp_transform_with_uncovered_op(self):
        # [FUNCTION DOC] Exports DummyModel, applies column-wise parallelism to fc, and checks
        # the transformed module matches the original output. Expects exactly one all_gather
        # plus one wait_tensor, gathering the sharded fc result before the uncovered op.
        model = DummyModel().to(device=self.device_type)
        inputs = (torch.randn(7, 3, requires_grad=False).to(device=self.device_type),)
        with torch.no_grad():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            {"fc": ColwiseParallel},
        )
        tp_model = tp_exported_program.module()
        with torch.no_grad():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_gather to be inserted to distributed sharded fc results
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_gather_into_tensor.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )

    @with_comms
    def test_tp_transform_e2e(self):
        # [FUNCTION DOC] Seeds RNG, exports a two-MLP model, shards each MLP column-wise then
        # row-wise, runs under inference mode and checks output equality. Expects two
        # all_reduce and two wait_tensor ops, one per MLP.
        torch.manual_seed(0)
        model = MLPListModule(2).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
            "mlps.1.0": ColwiseParallel,
            "mlps.1.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_reduce to be inserted at the end of each MLP
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 2,
                "_c10d_functional.wait_tensor.default": 2,
            },
        )

    @with_comms
    def test_tp_transform_no_bias(self):
        # [FUNCTION DOC] Same as the end-to-end test but with one MLP whose last linear has no
        # bias. Checks output equality and expects one all_reduce and one wait_tensor.
        torch.manual_seed(0)
        model = MLPListModule(1, bias=False).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )


if __name__ == "__main__":
    run_tests()
