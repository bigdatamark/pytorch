# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo test module checking that the inductor helper that runs an FX pass
# and saves the result invokes shape propagation. Defines one TestCase; no import-time
# side effects beyond imports.
from unittest import mock

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._inductor.utils import pass_execution_and_save


class FxPassesPreGradTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case holding the pre-grad FX pass utility test.
    @mock.patch("torch._inductor.utils.ShapeProp.propagate")
    def test_pass_execution_and_save(self, mock_shape_prop):
        # [FUNCTION DOC] Exports a tiny module (parameter plus input) in strict mode, runs a
        # no-op pass on its graph module through pass_execution_and_save with shape propagation
        # mocked out, and asserts the mocked propagate was invoked exactly once.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 4x4 ones parameter whose forward adds it to the
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the 4x4 ones parameter named param.
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(4, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the parameter plus the input tensor.
                return self.param + x

        def fx_pass(graph: torch.fx.GraphModule) -> None:
            # [FUNCTION DOC] HELPER: No-op FX pass that takes a graph module and returns None.
            return

        sample_input = torch.randn(4, 4)
        m = TestModule()
        m(sample_input)
        exported_program = torch.export.export(m, (sample_input,), strict=True)
        gm = exported_program.graph_module

        pass_execution_and_save(fx_pass, gm, sample_input, "Apply testing pass")
        mock_shape_prop.assert_called_once()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
