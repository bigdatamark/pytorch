# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Unit test of FSDP's execution-order tracer used with torch.fx symbolic tracing:
# traces a small model with two conditional paths and checks the recorded module forward
# order, per-module parameter usage records, parameter forward order and visited parameter
# set, and that the tracer's patched methods are restored afterward. Import-time side effect:
# instantiates the device-generic test class for cuda/hpu/xpu; running the file calls
# run_tests().
import torch
from torch.distributed.fsdp._trace_utils import _ExecOrderTracer
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class Model(torch.nn.Module):
    # [CLASS DOC] HELPER: Model for tracing tests with direct parameters (weight1, weight2 and
    # an unused weight_unused) and several submodules (layer0, layer1, a three-part layer2,
    # shared relu). forward takes a flag choosing whether to also run layer1 and weight2.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the three random direct parameters (6x6, 6x6, 2x2),
        # linear layers layer0 (6 to 6), layer1 (6 to 6, no bias), layer2 (Linear 6 to 3,
        # ReLU, Linear 3 to 6, no biases) and a shared ReLU.
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight2 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight_unused = torch.nn.Parameter(torch.randn(2, 2))
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, run_all_layers: bool) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies layer0 with ReLU, layer2 with ReLU, then multiplies by
        # weight1. If run_all_layers, then also layer1 with ReLU, multiplication by weight2, and
        # finally recomputes the result from x through layer0 again (to test repeated module use).
        # weight_unused is never used. Returns the result tensor.
        # KNOWN ISSUE: in the run_all_layers branch the last assignment discards the previous
        # value of z and uses x, so layer1 and weight2 do not affect the output; this is
        # intentional for tracing coverage only.
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = z @ self.weight1
        if run_all_layers:
            z = self.relu(self.layer1(z))
            z = z @ self.weight2
            # Use `layer0` twice to check the handling of multiplicity in the
            # saved data structures
            z = self.relu(self.layer0(x))
        return z


class TestSymbolicTracing(TestCase):
    # [CLASS DOC] Single-process test case for the execution-order tracer's recorded data.
    def test_symbolic_tracing_outputs(self):
        # [FUNCTION DOC] Traces the model with run_all_layers fixed to true inside the
        # execution-order tracer's patch context and asserts: the tracer's call_module and
        # create_proxy are restored on exit; the recorded module forward order is the expected
        # twelve-entry sequence (with repeats for reused modules); each module's parameter
        # usage list matches expectations (including the root's interleaved direct parameters
        # and no entries for relu); the parameter forward order has seven parameters; and the
        # visited set has the same size and members.
        """
        Tests running ``tracer.trace()`` inside ``patch_tracer()`` by checking
        the saved data structures.
        """
        model = Model()
        tracer = torch.fx.Tracer()
        orig_call_module = tracer.call_module
        orig_create_proxy = tracer.create_proxy
        exec_order_tracer = _ExecOrderTracer()
        with exec_order_tracer.patch_tracer(tracer=tracer, root_module=model):
            concrete_args = {"run_all_layers": True}
            tracer.trace(model, concrete_args)
        # Check that the tracer methods are unchanged after exiting the context
        self.assertEqual(orig_call_module, tracer.call_module)
        self.assertEqual(orig_create_proxy, tracer.create_proxy)
        # Check `module_forward_order`
        correct_module_forward_order = [
            model,
            model.layer0,
            model.relu,
            model.layer2,
            model.layer2[0],
            model.layer2[1],
            model.layer2[2],
            model.relu,
            model.layer1,
            model.relu,
            model.layer0,
            model.relu,
        ]
        exec_info = exec_order_tracer.exec_info
        self.assertEqual(exec_info.module_forward_order, correct_module_forward_order)
        # Check `module_to_param_usage_infos`
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model],
            [
                (model.layer0, list(model.layer0.named_parameters())),
                (model.layer2, list(model.layer2.named_parameters())),
                (model, [("weight1", model.weight1)]),
                (model.layer1, list(model.layer1.named_parameters())),
                (model, [("weight2", model.weight2)]),
                (model.layer0, list(model.layer0.named_parameters())),
            ],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer0],
            [(model.layer0, list(model.layer0.named_parameters()))],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer1],
            [(model.layer1, list(model.layer1.named_parameters()))],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer2],
            [
                (model.layer2[0], list(model.layer2[0].named_parameters())),
                (model.layer2[2], list(model.layer2[2].named_parameters())),
            ],
        )
        self.assertEqual(exec_info.module_to_param_usage_infos[model.relu], [])
        # Check `param_forward_order`
        correct_param_order = [
            model.layer0.weight,
            model.layer0.bias,
            model.layer2[0].weight,
            model.layer2[2].weight,
            model.weight1,
            model.layer1.weight,
            model.weight2,
        ]
        self.assertEqual(exec_info.param_forward_order, correct_param_order)
        # Check `visited_params`
        self.assertEqual(
            len(exec_info.visited_params), len(exec_info.param_forward_order)
        )
        self.assertEqual(exec_info.visited_params, set(exec_info.param_forward_order))


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestSymbolicTracing, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
