# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for get_source_partitions and check_subgraphs_connected from
# torch.fx.passes.utils.source_matcher_utils, plus legalize_graph from
# torch.fx.passes.tools_common. Models are exported through dynamo export or torch.export, then
# the graph is partitioned by source module or function type (or by name string when the
# torch_fn metadata is used), and partition counts and connectivity are asserted. Import time
# adds the parent test directory to sys.path. The parametrized tests are instantiated at the
# bottom of the file.

import os
import sys
import unittest

import torch


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    raise_on_run_directly,
    skipIfTorchDynamo,
)
from torch.testing._internal.jit_utils import JitTestCase


def _get_node_names(nodes: list[torch.fx.Node]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the name attribute of each node in the given list,
    # preserving order. Used to build assertion messages.
    return [n.name for n in nodes]


class TestSourceMatcher(JitTestCase):
    # [CLASS DOC] JitTestCase based tests of source partitioning. Tests mostly skip when
    # dynamo is not supported. The torch_fn export variants clear the source_fn_stack
    # metadata so the partitioner relies on torch_fn information, and are parametrized over
    # strict true and false.
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_linear_relu_linear(self):
        # [FUNCTION DOC] Exports a linear, linear, relu, linear model through dynamo export
        # with aten graph, removes dead code and partitions by Linear and ReLU module types.
        # Asserts two partition keys, three linear partitions (the first linear module is
        # called twice) and one relu partition, and that only the second linear partition is
        # connected to the relu partition.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear1 (applied twice), relu, and linear2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear1 (3 to 3), relu and linear2 (3 to 5).
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1 twice, then relu, then linear2.
                x = self.linear1(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.ReLU]
        )

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.Linear]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][1],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][2],
                module_partitions[torch.nn.ReLU][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_conv_relu_maxpool(self):
        # [FUNCTION DOC] Exports a three-convolution model with a constant tensor addition, relu
        # and max pool, and partitions by Conv2d, ReLU and MaxPool2d. Asserts three conv
        # partitions and one each of relu and maxpool, that only the third conv is connected to
        # relu, and that relu is connected to maxpool but not the reverse.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three Conv2d layers, relu, max pool and a stored
            # constant tensor added to the first conv output.
            def __init__(self, constant_tensor: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Stores the constant tensor and creates conv1, conv2 and
                # conv3, a ReLU and a MaxPool2d with kernel size three.
                super().__init__()
                self.constant_tensor = constant_tensor
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Computes conv1, conv2, adds the constant to the first
                # output, applies conv3 to the sum of the second conv output and that value,
                # then relu and max pool, and returns the result.
                a = self.conv1(x)
                b = self.conv2(a)
                c = a + self.constant_tensor
                z = self.conv3(b + c)
                return self.maxpool(self.relu(z))

        inputs = (torch.randn(1, 3, 256, 256),)
        gm, _ = torch._dynamo.export(M(torch.ones(1, 16, 256, 256)), aten_graph=True)(
            *inputs
        )
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d]
        )

        self.assertEqual(len(module_partitions), 3)
        self.assertEqual(len(module_partitions[torch.nn.Conv2d]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)
        self.assertEqual(len(module_partitions[torch.nn.MaxPool2d]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][1],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][2],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.MaxPool2d][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.ReLU][0],
                module_partitions[torch.nn.MaxPool2d][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_conv_relu_conv(self):
        # [FUNCTION DOC] Exports a model using a functional convolution wrapper module twice
        # around a relu, and partitions by the functional conv2d. Asserts one partition key
        # with two partitions.
        class FunctionalConv2d(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the functional conv2d with stored stride,
            # padding, dilation and groups settings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets stride, padding, dilation and groups
                # attributes to defaults.
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Calls the functional conv2d on the input, weight and
                # bias with the stored settings and returns the result.
                return torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module containing two FunctionalConv2d submodules with a
            # functional relu between them.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1 and conv2 as FunctionalConv2d instances.
                super().__init__()
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Applies conv1, functional relu, then conv2, passing
                # weight and bias each time.
                x = self.conv1(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        inputs = (torch.randn(1, 3, 5, 5), torch.rand(3, 3, 3, 3), torch.rand(3))
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.functional.conv2d]
        )

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[torch.nn.functional.conv2d]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_linear_relu_linear(self):
        # [FUNCTION DOC] Exports a model with four functional linear calls and two functional
        # relu calls and partitions by those functions. Asserts two keys, four linear partitions
        # and two relu partitions.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional linear twice, relu, linear twice,
            # relu with shared weight and bias inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base Module constructor only.
                super().__init__()

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Applies the functional linear twice, relu, linear
                # twice, relu and returns the result.
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                return x

        inputs = (torch.randn(1, 5), torch.rand((5, 5)), torch.zeros(5))
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.functional.linear, torch.nn.functional.relu]
        )

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.functional.linear]), 4)
        self.assertEqual(len(module_partitions[torch.nn.functional.relu]), 2)

    @skipIfTorchDynamo(
        "unexplained 3.13 failure: weakref inlining raises dynamic shape error only in 3.13"
    )
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_legalize_slice(self):
        # [FUNCTION DOC] Exports a module that reads an integer from a scalar tensor, adds
        # runtime checks on it and slices another tensor with it. Runs the exported module under
        # the fake mode of the example inputs with an Interpreter, legalizes the graph, then
        # runs it again the same way. Passes if neither run raises. Skipped under dynamo and
        # when dynamo is unsupported.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a scalar tensor x and a vector y; converts x to
            # a Python integer, checks it is nonnegative and less than the size of y minus one,
            # and returns the leading b plus one elements of y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Extracts the integer from x, asserts bounds with
                # torch._check, and returns the slice of y up to that value plus one.
                b = x.item()
                torch._check(b >= 0)
                torch._check(b + 1 < y.size(0))
                return y[: b + 1]

        ep = torch.export.export(M(), (torch.tensor(4), torch.randn(10)), strict=True)
        fake_inputs = [
            node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder"
        ]
        gm = ep.module()
        with fake_inputs[0].fake_mode:
            torch.fx.Interpreter(gm).run(*fake_inputs)
        legalized_gm = legalize_graph(gm)
        with fake_inputs[0].fake_mode:
            torch.fx.Interpreter(legalized_gm).run(*fake_inputs)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_linear_relu_linear_torch_fn_export(self, strict: bool):
        # [FUNCTION DOC] Same model and expectations as the linear, relu, linear module test,
        # but exported with torch.export (strict flag parametrized), the source_fn_stack
        # metadata of every node set to none, and partitioning by the name strings linear and
        # relu. Asserts the same counts and connectivity.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear1 (applied twice), relu and linear2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear1 (3 to 3), relu and linear2 (3 to 5).
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1 twice, relu, then linear2.
                x = self.linear1(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        gm = torch.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "torch_fn" only.
        # TODO: remove this after we fix "torch_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["linear", "relu"])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions["linear"]), 3)
        self.assertEqual(len(module_partitions["relu"]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["linear"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["linear"][1],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["linear"][2],
                module_partitions["relu"][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_conv_relu_maxpool_torch_fn_export(self, strict: bool):
        # [FUNCTION DOC] Export (parametrized strict) variant of the conv, relu, maxpool
        # test, with source_fn_stack cleared and partitioning by the name strings conv2d,
        # relu and max_pool2d. Asserts three conv partitions, one relu, one max pool, and the
        # same connectivity pattern.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three Conv2d layers, relu, max pool and a
            # stored constant tensor.
            def __init__(self, constant_tensor: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Stores the constant tensor and creates the three
                # convolutions, ReLU and MaxPool2d.
                super().__init__()
                self.constant_tensor = constant_tensor
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Runs the convolutions, adds the constant to the first
                # output, applies the third conv to the sum, then relu and max pool.
                a = self.conv1(x)
                b = self.conv2(a)
                c = a + self.constant_tensor
                z = self.conv3(b + c)
                return self.maxpool(self.relu(z))

        inputs = (torch.randn(1, 3, 256, 256),)
        gm = torch.export.export(
            M(torch.ones(1, 16, 256, 256)), inputs, strict=strict
        ).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "torch_fn" only.
        # TODO: remove this after we fix "torch_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(
            gm.graph, ["conv2d", "relu", "max_pool2d"]
        )

        self.assertEqual(len(module_partitions), 3)
        self.assertEqual(len(module_partitions["conv2d"]), 3)
        self.assertEqual(len(module_partitions["relu"]), 1)
        self.assertEqual(len(module_partitions["max_pool2d"]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["conv2d"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["conv2d"][1],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["conv2d"][2],
                module_partitions["relu"][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions["max_pool2d"][0],
                module_partitions["relu"][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions["relu"][0],
                module_partitions["max_pool2d"][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_functional_conv_relu_conv_torch_fn_export(
        self, strict: bool
    ):
        # [FUNCTION DOC] Export (parametrized strict) variant of the functional conv test
        # with source_fn_stack cleared, partitioning by the name conv2d. Asserts one key
        # with two partitions.
        class FunctionalConv2d(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the functional conv2d with stored settings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets stride, padding, dilation and groups
                # attributes to defaults.
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Calls the functional conv2d with the stored settings
                # and returns the result.
                return torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module containing two FunctionalConv2d submodules with a
            # functional relu between them.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1 and conv2 as FunctionalConv2d instances.
                super().__init__()
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Applies conv1, functional relu, then conv2.
                x = self.conv1(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        inputs = (torch.randn(1, 3, 5, 5), torch.rand(3, 3, 3, 3), torch.rand(3))
        gm = torch.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "torch_fn" only.
        # TODO: remove this after we fix "torch_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["conv2d"])

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions["conv2d"]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_functional_linear_relu_linear_torch_fn_export(
        self, strict: bool
    ):
        # [FUNCTION DOC] Export (parametrized strict) variant of the functional linear and relu
        # test with source_fn_stack cleared, partitioning by the names linear and relu. Asserts
        # four linear and two relu partitions.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional linear twice, relu,
            # linear twice, relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base Module constructor only.
                super().__init__()

            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Applies the functional linear twice, relu, linear
                # twice, relu and returns the result.
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                return x

        inputs = (torch.randn(1, 5), torch.rand((5, 5)), torch.zeros(5))
        gm = torch.export.export(M(), inputs, strict=strict).module()
        gm.graph.eliminate_dead_code()

        # Remove "source_fn_stack" meta to let partitioner use "torch_fn" only.
        # TODO: remove this after we fix "torch_fn". T199561090
        for node in gm.graph.nodes:
            node.meta["source_fn_stack"] = None

        module_partitions = get_source_partitions(gm.graph, ["linear", "relu"])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions["linear"]), 4)
        self.assertEqual(len(module_partitions["relu"]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @parametrize("strict", (True, False))
    def test_module_partitioner_weight_tied(self, strict: bool):
        # [FUNCTION DOC] Regression test for a model whose linear weight is also stored as a
        # separate attribute and used in an addition. Exports with torch.export (strict
        # parametrized), partitions by the Linear class when strict and by the name linear
        # otherwise, and asserts a single partition whose only output node is named linear and
        # whose input nodes are only the model input x.
        # real-world example: https://github.com/pytorch/pytorch/issues/142035
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer and a second attribute
            # referring to the same weight parameter; forward returns the linear output and
            # the weight plus one.
            def __init__(self, input_size, output_size):
                # [FUNCTION DOC] HELPER: Creates the linear layer of the given sizes and stores
                # its weight as tied_weight.
                super().__init__()
                # Define a linear layer
                self.linear = torch.nn.Linear(input_size, output_size)
                self.tied_weight = self.linear.weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear output of the input together with
                # the tied weight plus one.
                # Forward pass through the linear layer
                b = self.tied_weight + 1
                return self.linear(x), b

        inputs = (torch.randn(1, 10),)
        gm = torch.export.export(
            M(input_size=10, output_size=1), inputs, strict=strict
        ).module()
        gm.graph.eliminate_dead_code()

        k = torch.nn.Linear if strict else "linear"
        module_partitions = get_source_partitions(gm.graph, [k])

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[k]), 1)
        self.assertEqual(len(module_partitions[k][0].output_nodes), 1)
        self.assertEqual(module_partitions[k][0].output_nodes[0].name, "linear")
        input_node_names = {node.name for node in module_partitions[k][0].input_nodes}
        self.assertEqual(input_node_names, {"x"})

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_linear_relu_linear_nn_module_stack(self):
        # [FUNCTION DOC] Exports with strict false, which does not record source function
        # stacks, so the partitioner falls back to nn_module_stack information. Partitions by
        # Linear and ReLU and asserts three linear partitions, one relu and the same
        # connectivity as the other linear, relu, linear tests.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear1, linear2, relu, linear3
            # applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three Linear layers (3 to 3, 3 to 3, 3 to 5)
                # and a ReLU.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear3 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1, linear2, relu, linear3 in order.
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.relu(x)
                x = self.linear3(x)
                return x

        inputs = (torch.randn(3, 3),)
        # strict=False does not populate source_fn_stack, so
        # get_source_partitions should fall back to nn_module_stack.
        gm = torch.export.export(M(), inputs, strict=False).module()
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.ReLU]
        )

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.Linear]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)

        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][1],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][2],
                module_partitions[torch.nn.ReLU][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_get_source_partitions_deterministic_ordering(self):
        # [FUNCTION DOC] Checks that each Linear partition lists its input nodes, output nodes
        # and params in the order they occur in the graph. Builds a model where a third linear
        # consumes the sum of two linear outputs, exports with dynamo export, builds a map from
        # node to its graph position, and asserts each of the three lists equals its sorted
        # version, with the node names in the failure message.
        """
        Verifies that get_source_partitions() returns input_nodes, output_nodes,
        and params in deterministic order (sorted by graph position).
        """

        # Create a model where a single linear takes input from multiple sources
        # and produces outputs consumed by multiple downstream nodes.
        # This ensures multiple input_nodes, output_nodes, and params per partition.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear1 and linear2 both reading x and a linear3
            # reading their sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three Linear layers of size 3 by 3.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1 and linear2 to x, adds the results,
                # applies linear3 and returns it.
                a = self.linear1(x)
                b = self.linear2(x)
                # linear3 takes both a and b as inputs, so both are input_nodes
                c = self.linear3(a + b)
                return c

        inputs = (torch.randn(3, 3),)
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()

        partitions = get_source_partitions(gm.graph, [torch.nn.Linear])

        # Build a position map to verify nodes are sorted by graph order
        node_positions = {n: i for i, n in enumerate(gm.graph.nodes)}

        for partition in partitions[torch.nn.Linear]:
            # Verify input_nodes are sorted by graph position
            input_positions = [node_positions[n] for n in partition.input_nodes]
            self.assertEqual(
                input_positions,
                sorted(input_positions),
                f"input_nodes not in graph order: {_get_node_names(partition.input_nodes)}",
            )

            # Verify output_nodes are sorted by graph position
            output_positions = [node_positions[n] for n in partition.output_nodes]
            self.assertEqual(
                output_positions,
                sorted(output_positions),
                f"output_nodes not in graph order: {_get_node_names(partition.output_nodes)}",
            )

            # Verify params are sorted by graph position
            params_positions = [node_positions[n] for n in partition.params]
            self.assertEqual(
                params_positions,
                sorted(params_positions),
                f"params not in graph order: {_get_node_names(partition.params)}",
            )


instantiate_parametrized_tests(TestSourceMatcher)

if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
