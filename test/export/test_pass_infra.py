# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests of the export pass infrastructure: running a no-op pass base class
# transformation on exported programs and checking graph structure, node names and graph signature
# updates, plus the graph module replace hook used to keep the signature in sync with renames.
import copy
import unittest

import torch
from functorch.experimental import control_flow
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch.export import export
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    # [CLASS DOC] Pass infrastructure tests, skipped when Dynamo is unsupported.
    def test_export_pass_base(self) -> None:
        # [FUNCTION DOC] Exports a model with concatenation and tensor split (strict), applies a
        # pass base subclass that changes nothing, and checks every call_function node keeps a
        # non-empty stack trace and the node count, ops and targets match the original graph.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating the input with itself and splitting it into
            # two sections.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Concatenates the input with itself and returns the aten
                # split-by-sections result with two sections.
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            # [CLASS DOC] HELPER: Pass subclass with no overrides, so it should reproduce the graph
            # unchanged.
            pass

        ep = export(f, (torch.ones(3, 2),), strict=True)
        old_nodes = ep.graph.nodes

        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        # [FUNCTION DOC] Exports (strict) a model using a conditional with data-dependent checks on
        # scalar values in both branches and runs the do-nothing pass base transform on it; passes
        # if no error is raised. Skipped on Windows.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward runs a cond over two tensors with branch
            # functions that read a scalar from a tensor, check it lies between two and five, and
            # subtract or add.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Just calls the base constructor.
                super().__init__()

            def forward(self, pred, x, y):
                # [FUNCTION DOC] HELPER: Defines two branch functions and returns the cond result
                # for the predicate and the two tensors.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: True branch: reads the scalar from the first tensor,
                    # checks it is within two to five, returns the difference of the tensors.
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - y

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: False branch: reads the scalar from the second tensor,
                    # checks it is within two to five, returns the sum of the tensors.
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        _ = export(mod, (torch.tensor(True), x, y), strict=True)._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )

    def test_node_name_stability(self) -> None:
        # [FUNCTION DOC] Checks node names are unchanged for a model with a parameter, two buffers
        # (one mutated in forward) and two inputs after a do-nothing pass transform: names are
        # compared pairwise over before and after graphs.
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter and two buffers; forward computes an
            # affine-like combination of inputs and mutates the second buffer in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a parameter and two buffers with fixed scalar
                # values.
                super().__init__()

                # Define a parameter
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Computes input one plus the parameter times buffer one
                # plus input two times buffer two, adds one to buffer two in place, and returns the
                # combination.
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (torch.rand(1), torch.rand(1))
        m = CustomModule()

        ep_before = export(m, inps, strict=True)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # [FUNCTION DOC] Applies a function pass that renames every call_function node by appending
        # a suffix, via the program transform helper, and checks that all user output names in the
        # new graph signature contain the suffix and differ from the old signature's.
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and two buffers combined with two inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and two buffers.
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the first input plus the parameter times buffer
                # one plus the second input times buffer two.
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()

        # Test the custom module with two input tensors
        input_tensor1 = torch.tensor(5.0)
        input_tensor2 = torch.tensor(6.0)

        ep_before = torch.export.export(
            my_module, (input_tensor1, input_tensor2), strict=True
        )
        from torch.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            # [FUNCTION DOC] HELPER: Pass function: renames each call_function node by appending a
            # suffix, recompiles the graph module and returns a pass result marked as modified.
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)

    def test_replace_hook_basic(self) -> None:
        # [FUNCTION DOC] Renames nodes with a pass run on a deep copy of the graph module, while a
        # replace hook taken from a deep-copied graph signature is installed on the module. Checks
        # that user outputs of the copied signature get the suffix and differ from the original
        # signature.
        class CustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and two buffers combined with two inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and two buffers.
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the first input plus the parameter times buffer
                # one plus the second input times buffer two.
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()
        inputs = (torch.tensor(6.0), torch.tensor(7.0))
        ep_before = export(my_module, inputs, strict=True)

        def replace_pass(gm):
            # [FUNCTION DOC] HELPER: Pass function: appends a suffix to the name of every
            # call_function node, recompiles and returns a modified pass result.
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)


if __name__ == "__main__":
    run_tests()
