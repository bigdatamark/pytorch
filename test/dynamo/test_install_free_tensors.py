# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests the dynamo config option install_free_tensors, which installs parameters
# and buffers (and other free tensors) on the compiled graph module as attributes instead of
# lifting them to graph inputs. Provides helpers that compile or export a function and count
# tensor placeholder nodes in the resulting graph, small sample modules, and two test classes:
# one comparing input counts with the option off and on, one checking exported graphs with the
# option on. No import-time side effects.
import unittest
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.graph_module import GraphModule


def compile_and_extract_graph(
    fn, *args, **kwargs
) -> tuple[Callable, list[torch.fx.GraphModule]]:
    # [FUNCTION DOC] HELPER: Compiles fn with a graph-recording eager backend requiring a full
    # graph, runs it once on the given arguments to trigger compilation, and returns the
    # compiled callable together with the list of recorded graph modules.
    backend = EagerAndRecordGraphs()
    result_fn = torch.compile(backend=backend, fullgraph=True)(fn)
    # Run fn to capture graph
    _ = result_fn(*args, **kwargs)
    return result_fn, backend.graphs


def get_num_input_nodes(graph: GraphModule) -> int:
    # [FUNCTION DOC] HELPER: Returns how many placeholder nodes in the graph carry a tensor
    # example value in their metadata. Placeholders without example value metadata (possible
    # in some export tests) are not counted.
    """Returns the number of input nodes in the input GraphModule
    by counting the number of placeholder tensors
    """
    placeholder_cnt = 0
    for node in graph.graph.nodes:
        # Missing in some export tests so check manually
        placeholder_is_tensor = "example_value" in node.meta and isinstance(
            node.meta["example_value"], torch.Tensor
        )
        if node.op == "placeholder" and placeholder_is_tensor:
            placeholder_cnt += 1
    return placeholder_cnt


class SimpleLinearModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample module wrapping a single Linear layer (5 inputs, 1 output),
    # i.e. one weight parameter and one bias parameter. Forward applies the layer.
    """
    Simple linear model with 1 parameter and 1 buffer
    for basic testing purposes
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the Linear layer attribute named fwd.
        super().__init__()
        self.fwd = torch.nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.fwd(x)


class ResBlock(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample residual block: a sequential of convolution, batch norm and
    # ReLU, then a sequential of convolution and batch norm, plus a ReLU activation applied
    # after adding the skip connection. Both convolutions are 3x3 with padding one so the
    # shape is preserved.
    """
    Basic resnet building block - used for testing structure
    more typical of real models (i.e sequential, activations,
    and batchnorm)
    """

    def __init__(self, in_: int, out_: int):
        # [FUNCTION DOC] HELPER: Builds the two convolution/batch-norm sequentials and the
        # activation for in_ input and out_ output channels.
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_, out_, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_),
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs both sequentials, adds the original input in place to
        # the result (skip connection), applies the ReLU and returns it.
        skip = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += skip
        out = self.activation(out)
        return out


class InstallParamsAsGraphAttrTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that count graph inputs for several function and module shapes, once
    # with install_free_tensors off (parameters and buffers lifted to inputs) and once on
    # (installed as graph attributes, only real inputs remain). Uses two check helpers that
    # run the compile and comparison.
    @torch._dynamo.config.patch(install_free_tensors=False)
    def check_num_inputs_and_equality_no_install(
        self,
        fn_to_compile: torch.nn.Module | Callable,
        expected_num_inline_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs with install_free_tensors patched off. Compiles the
        # function with the example inputs, asserts exactly one graph was produced, asserts
        # the number of tensor placeholders equals the expected count, and asserts the
        # compiled and original results are equal.
        # KNOWN ISSUE: the docstring names the count argument expected_num_inputs but the
        # parameter is expected_num_inline_inputs.
        """Compiles the original fn, then:
        * Checks that the number of inputs in the graph is expected_num_inputs
        * Checks that the compiled fn and original fn are equal
        """
        # inlined ex
        opt_fn, graphs = compile_and_extract_graph(fn_to_compile, *example_inputs)
        self.assertEqual(len(graphs), 1, msg="Expected 1 graph (no breaks)")
        actual_num_inputs = get_num_input_nodes(graphs[0])
        self.assertEqual(actual_num_inputs, expected_num_inline_inputs)
        self.assertEqual(opt_fn(*example_inputs), fn_to_compile(*example_inputs))

    @torch._dynamo.config.patch(install_free_tensors=True)
    def check_num_inputs_and_equality_install(
        self,
        fn_to_compile: torch.nn.Module | Callable,
        expected_num_installed_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs with install_free_tensors patched on. Same checks as the
        # non-install variant: one graph with no breaks, tensor placeholder count equal to the
        # expected count, and compiled output equal to the original output.
        # KNOWN ISSUE: the second bullet in the docstring starts with a hash sign instead of
        # an asterisk (cosmetic).
        """Compiles the original fn, then:
        * Checks the number of inputs when installed is consistent with original_fn
        # Checks that the compiled fn when installed and original fn are equal
        """
        opt_installed_fn, graphs = compile_and_extract_graph(
            fn_to_compile, *example_inputs
        )
        self.assertEqual(len(graphs), 1, msg="Expected 1 graph (no breaks)")
        actual_num_inputs = get_num_input_nodes(graphs[0])
        self.assertEqual(actual_num_inputs, expected_num_installed_inputs)
        self.assertEqual(
            opt_installed_fn(*example_inputs), fn_to_compile(*example_inputs)
        )

    # ==================== Test Params and Buffer from NN Module ====================
    def test_optimizing_linear(self) -> None:
        # [FUNCTION DOC] Single linear module: expects three graph inputs without install
        # (input, weight, bias) and one with install.
        net = SimpleLinearModule()
        input1 = torch.randn((1, 5))
        # Expected: 1 + 1 * 2 = 3
        self.check_num_inputs_and_equality_no_install(net, 3, (input1,))
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_breadth_linear(self) -> None:
        # [FUNCTION DOC] Module with five parallel linear layers summed: expects eleven inputs
        # without install (one input plus two per layer) and one with install.
        class BreadthModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding five 1-to-1 linear layers named fwd through
            # fwd5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the five linear layers.
                super().__init__()
                self.fwd = torch.nn.Linear(1, 1)
                self.fwd2 = torch.nn.Linear(1, 1)
                self.fwd3 = torch.nn.Linear(1, 1)
                self.fwd4 = torch.nn.Linear(1, 1)
                self.fwd5 = torch.nn.Linear(1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of all five linear layers applied to
                # the same input.
                return (
                    self.fwd(x)
                    + self.fwd2(x)
                    + self.fwd3(x)
                    + self.fwd4(x)
                    + self.fwd5(x)
                )

        net = BreadthModel()
        input1 = torch.randn((1, 1))
        # Expected: 1 + 5 * 2 = 11
        self.check_num_inputs_and_equality_no_install(net, 11, (input1,))
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_nested_linear(self) -> None:
        # [FUNCTION DOC] Wraps the simple linear module in four levels of linear-then-inner
        # nesting; expects one plus two per level (five levels of parameters) inputs without
        # install and one with install.
        class NestedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own linear layer that applies an inner
            # module first and then its linear layer to the result.
            def __init__(self, inner_module: torch.nn.Module) -> None:
                # [FUNCTION DOC] HELPER: Stores a new linear layer and the given inner module.
                super().__init__()
                self.fwd = torch.nn.Linear(1, 1)
                self.inner_module = inner_module

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the inner module to the input and then the
                # linear layer.
                return self.fwd(self.inner_module(x))

        # Nest 5x
        kDepth = 4
        net = SimpleLinearModule()
        for _ in range(kDepth):
            net = NestedModel(net)
        input1 = torch.randn((1, 5))
        self.check_num_inputs_and_equality_no_install(
            net, 1 + 2 * (kDepth + 1), (input1,)
        )
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_simple_batchnorm(self) -> None:
        # [FUNCTION DOC] Batch norm layer over three channels: expects six inputs without
        # install (input, two parameters, three buffers) and one with install.
        net = torch.nn.BatchNorm2d(3)
        tensor = torch.randn((1, 3, 3, 3))
        # BatchNorm2d has 2 params, and 3 buffers
        self.check_num_inputs_and_equality_no_install(net, 6, (tensor,))
        self.check_num_inputs_and_equality_install(net, 1, (tensor,))

    def test_nets_as_input(self) -> None:
        # [FUNCTION DOC] Modules passed as function arguments, captured, or in a list: checks
        # four functions (one module argument, two module arguments, one argument plus a
        # captured module, a list of two modules). Without install the expected counts are
        # three, five, five and five; with install every case expects one. Each case compares
        # compiled to original output.
        """
        Tests when the nn.Module is an input to the fn we are optimizing

        In this case, we should treat it as regular input, which means we
        can lift parameters/buffers, but should not install them
        """
        # Test nn model as input
        net = SimpleLinearModule()
        net2 = SimpleLinearModule()
        x = torch.randn(1, 5)

        def test_fn(x: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the module argument to the tensor.
            return net(x)

        # When nn is in input, we don't install the params
        self.check_num_inputs_and_equality_no_install(test_fn, 3, (x, net))
        self.check_num_inputs_and_equality_install(test_fn, 1, (x, net))

        def test_fn2(
            # [FUNCTION DOC] HELPER: Sums the outputs of two module arguments applied to the
            # tensor.
            x: torch.Tensor, net: torch.nn.Module, net2: torch.nn.Module
        ) -> torch.Tensor:
            return net(x) + net2(x)

        self.check_num_inputs_and_equality_no_install(test_fn2, 5, (x, net, net2))
        self.check_num_inputs_and_equality_install(test_fn2, 1, (x, net, net2))

        def test_fn3(x: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Sums the output of the module argument and a captured
            # second module applied to the tensor.
            return net(x) + net2(x)

        # In case of local scope (net2 here), we can install
        self.check_num_inputs_and_equality_no_install(test_fn3, 5, (x, net))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x, net))

        def test_fn_list(x: torch.Tensor, nets: list[torch.nn.Module]):
            # [FUNCTION DOC] HELPER: Sums the outputs of every module in the list argument
            # applied to the tensor.
            return sum([net(x) for net in nets])

        self.check_num_inputs_and_equality_no_install(test_fn_list, 5, (x, [net, net2]))
        self.check_num_inputs_and_equality_install(test_fn_list, 1, (x, [net, net2]))

    def test_resnet_structure(self) -> None:
        # [FUNCTION DOC] Residual block on a small image: expects fifteen inputs without
        # install (input plus parameters and buffers) and one with install.
        net = ResBlock(3, 3)
        tensor = torch.randn(1, 3, 3, 3)
        # Conv2d has 2 params, BatchNorm2d has 3 buffers + 2 params, and Relu has 0 params
        # So expected = 2 + 5 + 5 + 2 = 14 + 1 for input
        self.check_num_inputs_and_equality_no_install(net, 15, (tensor,))
        self.check_num_inputs_and_equality_install(net, 1, (tensor,))

    def test_transformer(self) -> None:
        # [FUNCTION DOC] Evaluation-mode transformer with model dimension 32 run on source and
        # target tensors: expects 186 inputs without install and 2 with install.
        # needs eval mode - must disable dropout
        transformer = torch.nn.Transformer(d_model=32).eval()
        src = torch.rand(10, 32, 32)
        tgt = torch.rand(20, 32, 32)

        self.check_num_inputs_and_equality_no_install(transformer, 186, (src, tgt))
        self.check_num_inputs_and_equality_install(transformer, 2, (src, tgt))

    # ==================== Test Parameters and Buffers as input ====================
    def test_optimizing_params_in_input(self) -> None:
        # [FUNCTION DOC] Parameters used as inputs or captured in various ways: a parameter as
        # the only argument with a captured module, a parameter as second argument, a
        # parameter stored in a module global, and a list of parameters. Checks counts with
        # and without install; argument parameters stay inputs, captured ones are installed.
        # Sets the global variable global_param as a side effect.
        param = torch.nn.Parameter(torch.randn(1, 5))
        net = SimpleLinearModule()

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_num_inputs_and_equality_no_install(test_fn, 3, (param,))
        self.check_num_inputs_and_equality_install(test_fn, 1, (param,))

        x = torch.randn(1, 5)

        def test_fn2(x: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the parameter
            # argument.
            return net(x) + param

        # net gets installed, param does not here
        self.check_num_inputs_and_equality_no_install(test_fn2, 4, (x, param))
        self.check_num_inputs_and_equality_install(test_fn2, 2, (x, param))

        global global_param
        global_param = torch.nn.Parameter(torch.randn(1, 5))

        def test_fn3(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the global
            # parameter.
            return net(x) + global_param

        # net and global does too
        self.check_num_inputs_and_equality_no_install(test_fn3, 4, (x,))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x,))

        def test_fn4(
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the sum of the
            # parameters in the list argument.
            x: torch.Tensor, list_params: list[torch.nn.Parameter]
        ) -> torch.Tensor:
            return net(x) + sum(list_params)

        # list_params should not be installed
        self.check_num_inputs_and_equality_no_install(test_fn4, 4, (x, [param, param]))
        self.check_num_inputs_and_equality_install(test_fn4, 2, (x, [param, param]))

    def test_optimizing_buffer_in_input(self) -> None:
        # [FUNCTION DOC] Same scenarios as the parameter test but with a buffer: buffer as
        # sole argument, as second argument, and as a module global. Checks counts with and
        # without install. Sets the global variable global_buf as a side effect.
        buf = torch.nn.Buffer(data=torch.ones((1, 5)))
        net = SimpleLinearModule()

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_num_inputs_and_equality_no_install(test_fn, 3, (buf,))
        self.check_num_inputs_and_equality_install(test_fn, 1, (buf,))

        x = torch.randn(1, 5)

        def test_fn2(x: torch.Tensor, buf: torch.nn.Buffer):
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the buffer argument.
            return net(x) + buf

        # net gets installed, buf does not here
        self.check_num_inputs_and_equality_no_install(test_fn2, 4, (x, buf))
        self.check_num_inputs_and_equality_install(test_fn2, 2, (x, buf))

        global global_buf
        global_buf = torch.nn.Buffer(torch.randn(1, 5))

        def test_fn3(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the global buffer.
            return net(x) + global_buf

        # net and global does too
        self.check_num_inputs_and_equality_no_install(test_fn3, 4, (x,))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x,))

    def test_optimizing_buffer_and_param_in_input(self) -> None:
        # [FUNCTION DOC] Functions combining a captured parameter and captured buffer (expects
        # three inputs without install, one with), and a function receiving the parameter and
        # buffer as explicit arguments (three inputs in both modes).
        param = torch.nn.Parameter(torch.randn(5, 1))
        buf = torch.nn.Buffer(data=torch.ones((1, 1)))
        x = torch.randn(1, 5)

        def test_linear(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the captured parameter times the input plus the
            # captured buffer.
            return param * x + buf

        self.check_num_inputs_and_equality_no_install(test_linear, 3, (x,))
        self.check_num_inputs_and_equality_install(test_linear, 1, (x,))

        def test_linear_explicit(
            x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the first argument times the second plus the
            # third.
            return a * x + b

        # Now, param and buf are input so should not be inlined
        self.check_num_inputs_and_equality_no_install(
            test_linear_explicit, 3, (x, param, buf)
        )
        self.check_num_inputs_and_equality_install(
            test_linear_explicit, 3, (x, param, buf)
        )


class InstallParamsWhenExport(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that exporting with install_free_tensors on yields graphs with only
    # the expected tensor inputs, using one check helper.
    @torch._dynamo.config.patch(install_free_tensors=True)
    def check_export_matches_expectation(
        self,
        fn_to_export: Callable,
        expected_num_exported_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs with install_free_tensors patched on. Exports the
        # function with the dynamo export API, calls it on the example inputs to get the graph
        # module, asserts the tensor placeholder count equals the expected count, and asserts
        # the graph output equals the original function output.
        """Exports the original fn, then:
        * Checks that the number of inputs in the exported is expected_num_exported_inputs
        * Checks that the exported fn and original fn are equal
        """
        exported_fn = torch._dynamo.export(fn_to_export)
        out_graph = exported_fn(*example_inputs)[0]
        actual_num_inputs = get_num_input_nodes(out_graph)
        self.assertEqual(actual_num_inputs, expected_num_exported_inputs)
        self.assertEqual(out_graph(*example_inputs), fn_to_export(*example_inputs))

    def test_simple_linear(self) -> None:
        # [FUNCTION DOC] Exports a linear module directly, a function calling it, and a
        # function calling it twice on two inputs; expects one, one and two inputs.
        net = SimpleLinearModule()
        input1 = torch.randn((1, 5))
        self.check_export_matches_expectation(net, 1, (input1,))

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (input1,))

        # Check multiple inputs
        def test_fn_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Sums the captured module applied to each of two
            # arguments.
            return net(x) + net(y)

        input2 = torch.randn((1, 5))
        self.check_export_matches_expectation(test_fn_2, 2, (input1, input2))

    def test_simple_batchnorm(self) -> None:
        # [FUNCTION DOC] Exports a batch norm layer directly and via a wrapping function; both
        # expect a single input.
        net = torch.nn.BatchNorm2d(3)
        tensor = torch.randn((1, 3, 3, 3))
        self.check_export_matches_expectation(net, 1, (tensor,))

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (tensor,))

    def test_resnet_structure(self) -> None:
        # [FUNCTION DOC] Exports a residual block directly and via a wrapping function; both
        # expect a single input.
        net = ResBlock(3, 3)
        tensor = torch.randn(1, 3, 3, 3)
        self.check_export_matches_expectation(net, 1, (tensor,))

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (tensor,))

    def test_transformer(self) -> None:
        # [FUNCTION DOC] Exports a transformer directly and via a wrapping function; both
        # expect two inputs (source and target).
        transformer = torch.nn.Transformer(d_model=32).eval()
        src = torch.rand(10, 32, 32)
        tgt = torch.rand(20, 32, 32)

        self.check_export_matches_expectation(transformer, 2, (src, tgt))

        def test_fn(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured transformer to source and target.
            return transformer(src, tgt)

        self.check_export_matches_expectation(test_fn, 2, (src, tgt))

    def test_optimizing_params_in_input(self) -> None:
        # [FUNCTION DOC] Exports with a parameter passed as input: the captured module cases
        # expect one input, a function adding the parameter argument expects two, and one
        # taking a list of parameters expects two.
        param = torch.nn.Parameter(torch.randn(1, 5))
        net = SimpleLinearModule()

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_export_matches_expectation(net, 1, (param,))
        self.check_export_matches_expectation(test_fn, 1, (param,))

        x = torch.randn(1, 5)

        def test_fn2(x: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the parameter
            # argument.
            return net(x) + param

        # net gets installed, param does not here
        self.check_export_matches_expectation(test_fn2, 2, (x, param))

        def test_fn3(
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the sum of the list
            # of parameters.
            x: torch.Tensor, list_params: list[torch.nn.Parameter]
        ) -> torch.Tensor:
            return net(x) + sum(list_params)

        # list_params should not be installed or inlined here
        self.check_export_matches_expectation(test_fn3, 2, (x, [param, param]))

    def test_optimizing_buffer_in_input(self) -> None:
        # [FUNCTION DOC] Exports with a buffer passed as input: captured module cases expect
        # one input, a function adding the buffer argument expects two.
        buf = torch.nn.Buffer(data=torch.ones((1, 5)))
        net = SimpleLinearModule()

        def test_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module to the argument.
            return net(x)

        self.check_export_matches_expectation(net, 1, (buf,))
        self.check_export_matches_expectation(test_fn, 1, (buf,))

        x = torch.randn(1, 5)

        def test_fn2(x: torch.Tensor, buf: torch.nn.Buffer) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the captured module and adds the buffer argument.
            return net(x) + buf

        # net gets installed, buf does not here
        self.check_export_matches_expectation(test_fn2, 2, (x, buf))

    def test_optimizing_buffer_and_param_in_input(self) -> None:
        # [FUNCTION DOC] Exports a function taking a tensor, a parameter and a buffer as
        # explicit arguments; expects three inputs.
        param = torch.nn.Parameter(torch.randn(5, 1))
        buf = torch.nn.Buffer(data=torch.ones((1, 1)))
        x = torch.randn(1, 5)

        def test_linear_explicit(
            # [FUNCTION DOC] HELPER: Returns the second argument times the first plus the
            # third.
            x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return a * x + b

        # Now, param and buf are input so should not be inlined
        self.check_export_matches_expectation(test_linear_explicit, 3, (x, param, buf))

    def test_global_tensor_export(self) -> None:
        # [FUNCTION DOC] Exports a function that adds a global tensor (assigned to a module-
        # level variable in this test, a side effect) to its argument; expects one input.
        global x
        x = torch.randn((5, 5))

        def fn(a: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds the global tensor to the argument.
            return a + x

        inp = torch.randn(5, 5)
        self.check_export_matches_expectation(fn, 1, (inp,))

    def test_nonlocal_closure(self) -> None:
        # [FUNCTION DOC] Exports a function that adds a closure-captured tensor to its
        # argument; expects one input.
        x = torch.randn((5, 5))

        def fn(a: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds the captured tensor to the argument.
            return a + x

        inp = torch.randn((5, 5))
        self.check_export_matches_expectation(fn, 1, (inp,))

    @torch._dynamo.config.patch(install_free_tensors=True)
    def test_modify_net_state(self) -> None:
        # [FUNCTION DOC] Exports a module whose forward lazily creates an attribute (ones like
        # the input) on first call. Computes the eager reference first, resets the attribute
        # to none, exports, and asserts the exported graph output equals the reference.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and an attribute that starts as
            # none and is set to a ones tensor on the first forward call.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer and the unset attribute.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.a = None

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates the attribute if unset (mutating the module),
                # then returns the linear output plus the attribute.
                if self.a is None:
                    self.a = torch.ones_like(x)
                return self.linear(x) + self.a

        mod = Mod()
        inp = torch.randn(5, 5)
        # NOTE: since this fn modifies original class,
        # need to get reference value before tracing
        res = mod(inp)
        mod.a = None
        ep = torch._dynamo.export(mod)
        graph, _ = ep(inp)
        self.assertEqual(graph(inp), res)

    def test_list_of_tensor(self) -> None:
        # [FUNCTION DOC] Exports a function that adds the first two tensors of a list input;
        # expects two inputs.
        def fn(x: list[torch.Tensor]):
            # [FUNCTION DOC] HELPER: Returns the sum of the first two list elements.
            return x[0] + x[1]

        inp = [torch.tensor([1.3, 3.77, 0.1]), torch.tensor([8.7, 6.23, 9.9])]
        self.check_export_matches_expectation(fn, 2, (inp,))

    def test_nested_list_of_tensor(self) -> None:
        # [FUNCTION DOC] Exports a function over a list whose first item is a list of one
        # tensor and second item a tensor; expects two inputs.
        def fn(x: list[list[torch.Tensor] | torch.Tensor]):
            # [FUNCTION DOC] HELPER: Returns the first nested tensor plus the second item.
            return x[0][0] + x[1]  # type: ignore[index]

        inp = [[torch.tensor([1.3, 3.77, 0.1])], torch.tensor([8.7, 6.23, 9.9])]
        self.check_export_matches_expectation(fn, 2, (inp,))

    def test_dict_of_tensor(self) -> None:
        # [FUNCTION DOC] Exports a function that ignores its dictionary argument and adds five
        # to a captured dictionary's tensor; expects one input.
        inp_dict = {"temp": torch.tensor(12)}

        def fn(inp: dict[str, torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the captured dictionary tensor plus five,
            # ignoring the argument.
            return inp_dict["temp"] + 5

        self.check_export_matches_expectation(fn, 1, (inp_dict,))

    # TODO[lucaskabela]: register the flatten/unflatten function so we can evaluate this test
    @unittest.expectedFailure
    def test_user_defined_object(self) -> None:
        # [FUNCTION DOC] Expected failure: exports a function taking a user-defined object
        # holding two tensors plus a tensor, expecting two inputs. Fails because the object
        # type is not registered for flattening (per the TODO).
        class UserDefinedTestClass:
            # [CLASS DOC] HELPER: Plain object storing two tensors in attributes x and y.
            def __init__(self, x, y) -> None:
                # [FUNCTION DOC] HELPER: Stores the two tensors.
                self.x = x
                self.y = y

        x = torch.randn((3, 3))
        y = torch.randn((3, 3))

        def fn(obj: UserDefinedTestClass, inp: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the sum of the object's two tensors and the
            # input.
            return obj.x + obj.y + inp

        z = torch.randn((3, 1))

        self.check_export_matches_expectation(fn, 2, (UserDefinedTestClass(x, y), z))

    def test_tensors_as_nn_attr(self) -> None:
        # [FUNCTION DOC] Exports a module that stores plain (non-parameter, non-buffer)
        # tensors as attributes; expects one input.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding two plain ones tensors as attributes and
            # adding them to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two plain tensor attributes.
                super().__init__()
                self.a = torch.ones((5, 5))
                self.b = torch.ones((5, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the two attributes and the input.
                return self.a + self.b + x

        mod = Mod()
        inp = torch.randn(5, 5)
        self.check_export_matches_expectation(mod, 1, (inp,))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
