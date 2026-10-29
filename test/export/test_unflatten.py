# [MODULE DOC] Tests for torch.export.unflatten, which rebuilds the original module hierarchy
# from a flat exported program. Covers nested modules, parameters, buffers and buffer
# mutation, shared and aliased submodules, preserved module call signatures and flat argument
# adapters, container parameters, control flow and higher order operators inside submodules,
# torchbind objects, constants, pytree specs, module stack metadata, graph pass interactions
# (set grad, autocast, stack traces), reparametrization, duck typing of inputs, and state dict
# handling. Import-time effect: the flake8 file wide opt-out is set and tests are skipped when
# dynamo is unsupported.
# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import unittest
from re import escape
from typing import Any, List, Optional

import torch
import torch._dynamo as torchdynamo
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import export, FlatArgsAdapter, unflatten
from torch.export.unflatten import _assign_attr, _AttrKind, _disable_interpreter
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
from torch.utils._pytree import TreeSpec


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestUnflatten(TestCase):
    # [CLASS DOC] Main test case for unflatten. Typical test defines eager modules in the test
    # body, exports them (usually strict), unflattens the exported program, then compares
    # outputs or inspects module structure, parameters, buffers and graphs of the unflattened
    # result against the eager original.
    def compare_outputs(self, eager, unflattened, args):
        # [FUNCTION DOC] HELPER: Calls the eager module and the unflattened module with the
        # same arguments and asserts the two outputs are close (allclose); the outputs are
        # expected to be single tensors.
        orig_output = eager(*args)
        unflattened_output = unflattened(*args)
        self.assertTrue(torch.allclose(orig_output, unflattened_output))

    def test_unflatten_nested(self):
        # [FUNCTION DOC] Exports a module with nested submodules (a child with a nested child,
        # a second child with a buffer, a root parameter and a child parameter), unflattens
        # it, and compares outputs of the root and each submodule against the eager ones. Also
        # checks every entry of the eager state dict matches the unflattened state dict by
        # close comparison.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost submodule; forward divides its input by itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input divided by itself.
                return x / x

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule owning a nested child and a parameter of ones,
            # applying the nested child then adding the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested child and registers a two by three
                # parameter of ones.
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the nested child to the input and adds the
                # parameter.
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a buffer of ones that subtracts the buffer
            # from its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a two by three buffer of ones.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input minus the buffer.
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module that multiplies by a root parameter then applies
            # the two children in order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children and registers a two by three
                # parameter of ones.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the root parameter, then applies foo
                # and bar in sequence and returns the result.
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = export(orig_eager, (torch.rand(2, 3),), {}, strict=True)
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)

        # Compare the root modules and all submodules
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)

        # Check state dicts are equal
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for name, value in orig_state_dict.items():
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))

    def test_unflatten_buffer_mutation(self):
        # [FUNCTION DOC] Exports a module whose submodule mutates its own buffer in place
        # during forward, unflattens it, and checks that the unflattened buffer stays equal to
        # the eager buffer both before and after running both modules on the same random
        # input.
        class Child(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a buffer that is incremented in place by the
            # input in forward, returning the input minus the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a two by three buffer of ones.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the input to the buffer in place and returns the
                # input minus the buffer.
                self.child2buffer.add_(x)
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module that applies the child then multiplies by a root
            # parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child and registers a two by three
                # parameter of ones.
                super().__init__()
                self.foo = Child()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the child and multiplies the result by the
                # root parameter.
                x = self.foo(x)
                return x * self.rootparam

        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {}, strict=True)
        unflattened_module = unflatten(export_module)

        # Buffer should look the same before and after one run
        eager_buffer = eager_module.foo.child2buffer
        unflattened_buffer = unflattened_module.foo.child2buffer
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

        inputs = (torch.rand(2, 3),)
        eager_module(*inputs)
        unflattened_module(*inputs)
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

    def test_unflatten_nested_access(self):
        # [FUNCTION DOC] Exports a module whose forward reads a child buffer through its
        # parent by attribute path and also calls the child, unflattens it, and checks outputs
        # match eager.
        class Child(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module with a buffer of ones, subtracting it from its
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a two by three buffer of ones.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns input minus the buffer.
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module that adds a child's buffer to the input directly
            # and then calls the child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child and registers a two by three
                # parameter of ones.
                super().__init__()
                self.foo = Child()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the child's buffer to the input, then applies
                # the child.
                x = x + self.foo.child2buffer
                x = self.foo(x)
                return x

        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {}, strict=True)
        unflattened_module = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(eager_module, unflattened_module, inputs)

    def test_unflatten_shared_submodule(self):
        # [FUNCTION DOC] Exports a module in which a single normalization layer instance
        # appears twice inside a sequential container, unflattens it, compares outputs, checks
        # that each sequential index exists in the result and that entries zero and two are
        # the very same object (identity preserved).
        class Shared(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a sequential of one shared layer norm, a
            # relu, the same layer norm again and a relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a layer norm over ten features and places it
                # twice in the sequential container.
                super().__init__()
                layernorm = torch.nn.LayerNorm(10)
                self.sub_net = torch.nn.Sequential(
                    layernorm,
                    torch.nn.ReLU(),
                    layernorm,
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the sequential container to the input.
                return self.sub_net(x)

        eager_module = Shared()
        inps = (torch.rand(10),)
        export_module = export(eager_module, inps, {}, strict=True)
        unflattened_module = unflatten(export_module)
        self.compare_outputs(eager_module, unflattened_module, inps)
        self.assertTrue(hasattr(unflattened_module, "sub_net"))
        for i in range(len(eager_module.sub_net)):
            self.assertTrue(hasattr(unflattened_module.sub_net, str(i)))
        self.assertEqual(
            id(getattr(unflattened_module.sub_net, "0")),
            id(getattr(unflattened_module.sub_net, "2")),
        )

    def test_unflatten_shared_submodule_reorder(self):
        # [FUNCTION DOC] Regression test for reordering submodules when sharing creates alias
        # keys with a numeric suffix: exports a model whose sequential container reuses one
        # block twice, unflattens, compares outputs and checks the first and third entries are
        # the same object.
        """Test that _reorder_submodules handles @N-suffixed FQNs correctly.

        When modules are shared (aliased), PyTorch assigns @N suffixes to
        duplicate entries in _modules. The fqn_order dict (built from
        fqn_list) filters out @N entries, so _reorder_submodules must fall
        back to the base FQN for ordering.
        """

        class Block(torch.nn.Module):
            # [CLASS DOC] HELPER: Block wrapping a linear layer of width ten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ten by ten linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a sequential container reusing one block instance
            # at two positions separated by relus.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one block and puts it twice in the sequential
                # container with relus between.
                super().__init__()
                shared_block = Block()
                self.blocks = torch.nn.Sequential(
                    shared_block,
                    torch.nn.ReLU(),
                    shared_block,
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the sequential container.
                return self.blocks(x)

        eager_module = Model()
        inps = (torch.rand(2, 10),)
        export_module = export(eager_module, inps, {}, strict=True)
        unflattened_module = unflatten(export_module)
        self.compare_outputs(eager_module, unflattened_module, inps)
        # Verify shared identity is preserved
        self.assertEqual(
            id(getattr(unflattened_module.blocks, "0")),
            id(getattr(unflattened_module.blocks, "2")),
        )

    def test_assign_attr_noncontiguous_call_indices(self):
        # [FUNCTION DOC] Directly tests the attribute assignment utility on a bare root module
        # with a base submodule and one suffixed call-copy submodule, skipping the
        # intermediate copies. For both a parameter kind and a buffer kind, assigns a freshly
        # made object at a nested path and asserts that both the base submodule and the
        # suffixed one hold that identical object.
        """_assign_attr must populate every @N call-name variant present in
        _modules, even when the indices are non-contiguous.

        When a multi-called submodule has some calls inside a torch.no_grad() /
        set_grad_enabled region, replace_set_grad_with_hop_pass relocates those
        intermediate call copies (e.g. @1, @2) into a wrap_with_set_grad_enabled
        HOP subgraph, leaving non-contiguous indices at the top level (e.g. base
        + @3). A contiguous scan starting at @1 stops at the first gap and never
        reaches the surviving @3 copy, leaving its parameters unassigned; that
        copy later fails in _sink_params with a missing-attribute error.
        """
        for attr_kind, make_obj in (
            (_AttrKind.PARAMETER, lambda: torch.nn.Parameter(torch.randn(4))),
            (_AttrKind.BUFFER, lambda: torch.ones(4)),
        ):
            root = torch.nn.Module()
            # base + @3, with @1/@2 absent (relocated into a HOP subgraph).
            root._modules["leaf"] = torch.nn.Module()
            root._modules["leaf@3"] = torch.nn.Module()

            obj = make_obj()
            _assign_attr(obj, root, "leaf.bias", attr_kind)

            self.assertIs(root._modules["leaf"].bias, obj)
            self.assertIs(root._modules["leaf@3"].bias, obj)

    def test_assert_tensor_metadata_stack(self):
        # [FUNCTION DOC] Exports (non strict default) a model where an inner module calls
        # convert to integer dtype operations, checks every dtype metadata assertion node
        # carries a module stack of length two, then unflattens and checks outputs are close
        # to eager.
        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module with a random tensor attribute that casts both
            # inputs to int32, adds the attribute to the first and returns the sum with the
            # second.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a random vector of three as plain attribute a.
                super().__init__()
                self.a = torch.randn(3)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Casts x and y to int32, adds the attribute to x and
                # returns x plus y.
                x = x.to(dtype=torch.int32)
                y = y.to(dtype=torch.int32)
                x = x + self.a
                return x + y

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module squaring both inputs and passing them to the
            # inner module.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the inner module as attribute n.
                super().__init__()
                self.n = N()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Squares x and y elementwise and returns the inner
                # module applied to them.
                x = x * x
                y = y * y
                return self.n(x, y)

        m = M()
        ep = torch.export.export(m, (torch.randn(3), torch.randn(3)))
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten._assert_tensor_metadata.default:
                self.assertEqual(len(node.meta.get("nn_module_stack")), 2)

        uep = torch.export.unflatten(ep)

        inp = (torch.randn(3), torch.randn(3))
        self.assertTrue(torch.allclose(uep(*inp), m(*inp)))

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    @skipIfTorchDynamo("Non strict mode is not meant to run with dynamo")
    def test_unflatten_preserve_signature(self):
        # [FUNCTION DOC] Exports a module with a preserved call signature for a nested path,
        # in strict and non strict mode. Unflattens, compares outputs, replaces the preserved
        # submodule with a fresh eager instance and checks outputs again. Passing inputs whose
        # tree structure differs without an adapter must raise TypeError with a message about
        # no flat args adapter. With an adapter that drops extra trailing arguments, the extra
        # input call must give the original output. Skipped on Windows and under dynamo.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost module taking a pair and a dict, returning a dict
            # of sum and product of the dict value and the pair second element.
            def forward(self, zx, y):
                # [FUNCTION DOC] HELPER: Returns a mapping with sum and product entries
                # computed from the keyed tensor and second tuple element.
                return {"x": y["key"] + zx[1], "w": y["key"] * zx[1]}

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that calls the nested child with a tuple and a dict,
            # combining results.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested child.
                super().__init__()
                self.nested = NestedChild()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Makes a ones tensor like x, calls the nested child
                # with pair and dict keyword, returns the product plus ones minus the sum.
                z = torch.ones_like(x)
                xw = self.nested((z, x), y={"key": y})
                return xw["w"] + z - xw["x"]

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that subtracts one from its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input minus one.
                return x - 1

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module applying foo then bar.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies foo to both inputs then bar to the result.
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)
        for strict in [True, False]:
            export_module = export(
                orig_eager,
                inps,
                {},
                preserve_module_call_signature=("foo.nested",),
                strict=strict,
            )
            unflattened = unflatten(export_module)
            self.compare_outputs(export_module.module(), unflattened, inps)
            unflattened.foo.nested = NestedChild()
            self.compare_outputs(export_module.module(), unflattened, inps)

            # Test tree spec mismatched input
            orig_outs = export_module.module()(*inps)
            new_inps = *inps, torch.rand(2, 3)
            with self.assertRaisesRegex(
                TypeError,
                "There is no flat args adapter specified. Are you sure you are calling this with the right arguments?",
            ):
                unflattened(new_inps)

            # With flat args adapter
            class KeepTwoFlatArgsAdapter(FlatArgsAdapter):
                # [CLASS DOC] HELPER: Flat arguments adapter that truncates the flattened
                # input list to its first two items.
                def adapt(
                    self,
                    target_spec: TreeSpec,
                    input_spec: TreeSpec,
                    input_args: List[Any],
                    metadata: dict[str, Any],
                    obj: Optional[Any] = None,
                ) -> List[Any]:
                    # [FUNCTION DOC] HELPER: Adapter hook; repeatedly removes the last element
                    # while the list has more than two, mutating and returning the list. Other
                    # arguments are ignored.
                    while len(input_args) > 2:
                        input_args.pop(-1)
                    return input_args

            unflattened = unflatten(export_module, KeepTwoFlatArgsAdapter())
            new_outs = unflattened(*new_inps)
            self.assertTrue(torch.allclose(orig_outs, new_outs))

    def test_unflatten_param_list_dict(self):
        # [FUNCTION DOC] Exports a module with a list of parameters and a dict of parameters,
        # each used in forward, unflattens it and compares outputs with the exported module
        # run on new random input.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter list and parameter dict of two
            # random items each, summed into the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Fills a parameter list and a parameter dict with two
                # random two by three parameters each.
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds each parameter from the list and the dict to the
                # input in turn.
                for i in range(2):
                    x = x + self.param_list[i]
                    x = x + self.param_dict[f"key_{i}"]
                return x

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_unflatten_preserve_with_unused_input(self):
        # [FUNCTION DOC] Non strict export of a model whose preserved child receives a top-k
        # output where one of the child inputs is not needed by its result path; dead code is
        # eliminated from the graph before unflattening. Checks the unflattened module output
        # matches the exported module on random inputs. Skipped on Windows.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Child returning the sum of two inputs and the third.
            def forward(self, x, a, b):
                # [FUNCTION DOC] HELPER: Returns x plus a and b as a pair.
                return x + a, b

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent computing a top-k of its second input and feeding
            # values and indices to a child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child.
                super().__init__()
                self.m1 = M1()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Takes the top two of y and returns the first output
                # of the child applied to x, values and indices.
                a, b = torch.topk(y, 2)
                return self.m1(x, a, b)[0]

        ep = torch.export.export(
            M(),
            (torch.randn(2), torch.randn(5)),
            preserve_module_call_signature=("m1",),
            strict=False,
        )
        ep.graph.eliminate_dead_code()
        unflattened = unflatten(ep)
        self.compare_outputs(ep.module(), unflattened, (torch.randn(2), torch.randn(5)))

    def test_unflatten_wrong_input(self):
        # [FUNCTION DOC] Exports a module that sums its input and all its parameters from a
        # list and a dict, specialized to a two by three input. The exported module call with
        # a different shape must raise AssertionError mentioning a failed size guard; the
        # unflattened module with the wrong shape must raise RuntimeError mentioning the
        # expected input shape of two versus six.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter list and parameter dict of two
            # random items each; forward returns the sum of the input and all parameter sums.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Fills a parameter list and parameter dict with two
                # random two by three parameters each.
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input sum plus the sum of every parameter
                # in the list and the dict.
                a = x.sum()
                for i in range(2):
                    a = a + self.param_list[i].sum()
                    a = a + self.param_dict[f"key_{i}"].sum()
                return a

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] == 2"),
        ):
            # expected 2, but got 6
            export_module.module()(torch.randn(6, 6))

        unflattened = unflatten(export_module)
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be equal to 2, but got 6"),
        ):
            unflattened(torch.randn(6, 6))

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_unflatten_with_inplace_compile(self):
        # [FUNCTION DOC] Exports a nested module with a parameter, buffer and nested child,
        # unflattens it, compiles a submodule in place with an explain backend wrapping
        # inductor requiring a full graph, compares outputs to eager, asserts exactly one
        # graph was captured, then compiles the whole unflattened module in place and compares
        # again. Skipped on Windows.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost module dividing its input by itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input divided by itself.
                return x / x

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a nested child and a parameter of ones, adding
            # the parameter after the child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested child and registers the parameter.
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the nested child and adds the parameter.
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of ones, subtracted from the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a two by three buffer of ones.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input minus the buffer.
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root with a parameter, multiplying then applying two
            # children in order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children and registers the root
                # parameter.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the root parameter then applies foo and
                # bar in turn.
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        # in-place compilation should work. Pass fullgraph to ensure no graph breaks.
        from torch._dynamo.backends.debugging import ExplainWithBackend

        eb = ExplainWithBackend("inductor")
        unflattened.foo.compile(backend=eb, fullgraph=True)
        inputs = (torch.randn(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.assertEqual(len(eb.graphs), 1)

        unflattened.compile()
        self.compare_outputs(orig_eager, unflattened, inputs)

    def test_fx_trace(self):
        # [FUNCTION DOC] Exports a module that takes a tuple and a dict of tensors, unflattens
        # it, and checks the unflattened module can be symbolically traced by the fx tracer
        # when all three flat arguments are given as placeholders. Passes if no exception is
        # raised.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing the two tuple elements and a dict value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Adds the two elements of x, then adds the entry foo
                # of y.
                x = x[0] + x[1]
                x = x + y["foo"]
                return x

        orig_eager = MyModule()
        inputs = ((torch.rand(2, 3), torch.rand(2, 3)), {"foo": torch.rand(2, 3)})
        export_module = export(orig_eager, inputs, {}, strict=True)

        unflattened = unflatten(export_module)
        torch.fx.symbolic_trace(
            unflattened, concrete_args=(torch.fx.PH, torch.fx.PH, torch.fx.PH)
        )

    def test_double_nested_submodule(self):
        # [FUNCTION DOC] Exports a module that calls a doubly nested submodule through an
        # attribute chain from the root, unflattens it and compares outputs against eager.
        class SubSubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost module squaring its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the elementwise square.
                return x * x

        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module holding the innermost one; its own forward
            # returns zero.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the innermost child.
                super().__init__()
                self.subsubmod = SubSubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input minus itself.
                return x - x

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module that reaches into the doubly nested child and
            # adds its result to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the middle child.
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the innermost child applied to
                # it.
                return x + self.submod.subsubmod(x)

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)

    def test_unflatten_container_type(self):
        # [FUNCTION DOC] Exports a module (strict and non strict) whose children use a list
        # input, a buffer and a linear layer, unflattens the non strict export, re-exports the
        # unflattened module non strict, and checks the re-exported module gives results close
        # to the original on the same inputs.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module wrapping a four by four linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a leaf child and a random four by four buffer;
            # forward sums the buffer, leaf output and the two list entries.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf and registers the buffer.
                super().__init__()
                self.leaf = Leaf()
                self.buffer = torch.nn.Buffer(torch.randn(4, 4))

            def forward(self, x, z):
                # [FUNCTION DOC] HELPER: Returns the sum of the buffer, leaf result and both
                # list items, each reduced to a scalar.
                return self.buffer.sum() + self.leaf(x).sum() + z[0].sum() + z[1].sum()

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module that reads the child buffer directly and
            # combines it with the child output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child.
                super().__init__()
                self.bar = Bar()

            def forward(self, x, z):
                # [FUNCTION DOC] HELPER: Adds the child buffer to the input and list entries,
                # calls the child, and adds the sum.
                y = self.bar.buffer + x + z[0] + z[1]
                return self.bar(x, z) + y.sum()

        inp = (torch.randn(4, 4), [torch.randn(4, 4), torch.randn(4, 4)])
        mod = Foo()

        ep_strict = torch.export.export(mod, inp, strict=True)  # noqa: F841
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        ep = torch.export.export(gm_unflat_non_strict, inp, strict=False)
        self.assertTrue(torch.allclose(ep.module()(*inp), mod(*inp)))

    def test_unflattened_module_nodes_has_meta_val(self):
        # [FUNCTION DOC] Exports and unflattens a module whose submodule returns a pair,
        # compares outputs, and then checks every node in every module of the unflattened
        # result, except output nodes, has a recorded value in its metadata.
        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule returning the sum and the product of its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of x plus x and x times x.
                return x + x, x * x

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module adding the sum of the submodule results to the
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule.
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the sum over the submodule
                # outputs.
                return x + sum(self.submod(x))

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)

        def check_meta(gm):
            # [FUNCTION DOC] HELPER: Asserts that every non output node of a graph module has
            # a non empty value in its meta data.
            for n in gm.graph.nodes:
                if n.op == "output":
                    continue
                self.assertTrue(n.meta.get("val") is not None)

        for m in unflattened.modules():
            check_meta(m)

    def test_unflatten_requires_grad_param(self):
        # [FUNCTION DOC] Creates a module on the meta device with a parameter that does not
        # require gradients, exports and unflattens it, and checks that the parameter in the
        # unflattened state dict and as an attribute still has requires_grad false.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a three by three parameter of ones without
            # gradients, added to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter without gradient tracking.
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter plus the input.
                return self.p + x

        with torch.device("meta"):
            mod = M()

        inputs = (torch.randn(3, 3, device="meta"),)
        ep = export(mod, inputs, strict=True)
        unflattened = unflatten(ep)
        self.assertTrue(unflattened.state_dict()["p"].requires_grad is False)
        self.assertTrue(unflattened.p.requires_grad is False)

    def test_placeholder_and_get_attr_ordering_after_unflattened(self):
        # [FUNCTION DOC] Unflattens a convolution followed by a transpose and checks the graph
        # of the created submodule has its inputs in order: the placeholder first and then two
        # attribute fetches (weight and bias).
        class TransposeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a strided convolution followed by a
            # transposition of the first two dims.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution with three input channels,
                # one output channel, kernel three and stride two.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution and swaps the first two
                # dimensions.
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        exported_program = export(TransposeModule(), args=(x,), strict=True)
        unflattened_module = unflatten(exported_program)

        # Check the inputs of the created call_module node are in order
        call_module_input_order = []
        for node in unflattened_module.graph.nodes:
            if node.op == "call_module":
                transpose_module = unflattened_module.get_submodule(node.target)
                for sub_node in transpose_module.graph.nodes:
                    if sub_node.op == "placeholder" or sub_node.op == "get_attr":
                        call_module_input_order.append(sub_node.op)
        self.assertEqual(
            call_module_input_order, ["placeholder", "get_attr", "get_attr"]
        )

    def test_unflatten_constant_tensor(self):
        # [FUNCTION DOC] Exports a module whose child builds a tensor from a Python float
        # attribute inside forward, unflattens it and compares outputs with the exported
        # module.
        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule storing a float attribute and adding a tensor made
            # from it to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the float attribute 0.1.
                super().__init__()
                self.initializer = 0.1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds a tensor constructed from the float attribute to
                # the input.
                return x + torch.tensor(self.initializer)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Root adding the submodule result to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule.
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the submodule applied to it.
                return x + self.submod(x)

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    @skipIfTorchDynamo("custom objects not supported in dynamo yet")
    def test_unflatten_constant_obj(self):
        # [FUNCTION DOC] Registers a fake class for a TorchScript test object, builds a model
        # whose submodule holds an instance of that object, exports non strict under torchbind
        # tracing, unflattens and compares outputs with the exported module. Skipped under
        # dynamo.
        init_torchbind_implementations()

        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:  # noqa: F841
            # [CLASS DOC] HELPER: Fake implementation of the torchbind test class storing two
            # ints, used to trace calls; reconstructable from flattened context.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: Stores x and y.
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flat_ctx):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from a flattened
                # context of key value pairs by passing them as keyword arguments.
                return cls(**dict(flat_ctx))

            def add_tensor(self, z):
                # [FUNCTION DOC] HELPER: Returns the sum of the two stored values times z.
                return (self.x + self.y) * z

        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule holding a real torchbind object created with
            # values ten and twenty and adding its add_tensor result to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the torchbind object.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the object's add_tensor on x.
                return x + self.attr.add_tensor(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Root adding the submodule result to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule.
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the submodule output.
                return x + self.submod(x)

        with enable_torchbind_tracing():
            export_module = torch.export.export(
                Mod(), (torch.randn((2, 3)),), strict=False
            )
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    def test_unflatten_skipped_call_module(self):
        # [FUNCTION DOC] Exports (non strict) a call chain in which module C calls a sibling
        # module D that is held by the root rather than by C, then unflattens and compares the
        # generated forward code of the root, B and C with expected strings. The root calls B,
        # B calls C and adds the input, and the call of D is inlined as sine into C code, with
        # D not appearing as a submodule call.
        class C(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes the cosine of its input and
            # passes it to the module D of the enclosing A instance via a global reference.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies cosine then calls the d child of the closed
                # over module a.
                return a.d(x.cos())

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding C and adding the input to its result.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates C.
                super().__init__()
                self.c = C()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the C result plus the input.
                return self.c(x) + x

        class D(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing sine of its input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return x.sin()

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding B and D and only calling B.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates B and D.
                super().__init__()
                self.b = B()
                self.d = D()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns B applied to the input.
                return self.b(x)

        a = A()

        # The call chain looks like this:
        # A -> B -> C -> A.d
        ep = torch.export.export(a, (torch.randn(3),), strict=False)
        ufm = unflatten(ep)
        self.assertExpectedInline(
            str(ufm.graph_module.code).strip(),
            """\
def forward(self, x):
    b = self.b(x);  x = None
    return (b,)""",
        )
        self.assertExpectedInline(
            str(ufm.b.graph_module.code).strip(),
            """\
def forward(self, x):
    c = self.c(x)
    add = torch.ops.aten.add.Tensor(c, x);  c = x = None
    return add""",
        )
        self.assertExpectedInline(
            str(ufm.b.c.graph_module.code).strip(),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x);  x = None
    sin = torch.ops.aten.sin.default(cos);  cos = None
    return sin""",
        )

    def test_nested_leaf_non_strict(self):
        # [FUNCTION DOC] Non strict export of a three level model with a preserved call
        # signature for the middle module, followed by unflatten. Passes if unflatten does not
        # raise; nothing else is checked.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf adding one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        class Nested(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module holding the leaf and adding two to its result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf.
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns leaf output plus two.
                return self.leaf(x) + 2

        class TopLevel(torch.nn.Module):
            # [CLASS DOC] HELPER: Top module holding the middle module and adding three.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested module.
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns nested output plus three.
                return self.nested(x) + 3

        ep = torch.export.export(
            TopLevel(),
            (torch.randn(3),),
            strict=False,
            preserve_module_call_signature=("nested",),
        )

        torch.export.unflatten(ep)

    def test_unflatten_submodule_ordering(self):
        # [FUNCTION DOC] Exports a model where one submodule instance is registered under two
        # names and another module is registered between them, and checks the unflattened
        # module lists exactly four module names, including duplicates, in the same order as
        # the eager model.
        class Module2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random buffer and parameter, added to the
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a three by four buffer and parameter.
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.rand(3, 4))
                self.register_parameter("param", torch.nn.Parameter(torch.rand(3, 4)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus buffer plus parameter.
                return x + self.buffer + self.param

        class Module1(torch.nn.Module):
            # [CLASS DOC] HELPER: Same shape as the previous helper, a separate class with its
            # own buffer and parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a three by four buffer and parameter.
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.rand(3, 4))
                self.register_parameter("param", torch.nn.Parameter(torch.rand(3, 4)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus buffer plus parameter.
                return x + self.buffer + self.param

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Root with mod2, an alias mod3 pointing to the same module,
            # and mod1; forward chains mod1, mod2 and mod3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the modules and the alias.
                super().__init__()
                self.mod2 = Module2()
                self.mod3 = self.mod2
                self.mod1 = Module1()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies mod1 then mod2 then mod3 to the input.
                return self.mod3(self.mod2(self.mod1(x)))

        mod = Module()

        ep = torch.export.export(mod, (torch.randn(3, 4),), strict=True)

        unflattened = torch.export.unflatten(ep)
        fqn_list = [x for x, _ in unflattened.named_modules(remove_duplicate=False)]
        self.assertEqual(len(fqn_list), 4)
        self.assertEqual(
            [x for x, _ in mod.named_modules(remove_duplicate=False)],
            fqn_list,
        )

    def test_duplicate_placeholder(self):
        # [FUNCTION DOC] Exports, strict and non strict, a model that applies one shared layer
        # norm three times in a module list, unflattens each export and checks outputs match
        # eager.
        N, C, H, W = 1, 2, 2, 3

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a module list holding the same layer norm
            # instance three times, applied in a loop.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one layer norm over the last three dimensions
                # and repeats it three times.
                super().__init__()
                layer = torch.nn.LayerNorm([C, H, W])
                self.norms = torch.nn.ModuleList(
                    [
                        layer,  # reuse layer norm
                        layer,
                        layer,
                    ]
                )

            def forward(self, input_):
                # [FUNCTION DOC] HELPER: Applies each normalization in turn, feeding each
                # output to the next, and returns the last output.
                for i in range(len(self.norms)):
                    output = self.norms[i](input_)
                    input_ = output
                return output

        mod = MyModule()
        input_ = torch.randn(N, C, H, W)

        ep_strict = export(copy.deepcopy(mod), (input_,), strict=True)
        umod = unflatten(ep_strict)
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))

        ep_non_strict = export(copy.deepcopy(mod), (input_,), strict=False)
        umod = unflatten(ep_non_strict)
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))

    def test_simple_alias(self):
        # [FUNCTION DOC] Tests unflatten with shared parameters. First model shares a
        # parameter between a top level attribute and a linear layer bias, and the test checks
        # the unflattened attributes are the same object. Second model aliases a bias that is
        # then unused at top level, and the test checks outputs equal eager. Note both models
        # use the same class name; the second shadows the first.
        # handle weight sharing, check tensor ids after unflattening
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose linear layer bias is the parameter also stored
            # at top level; forward adds the shared bias to the linear output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and linear layer and assigns
                # the parameter as the layer bias.
                super().__init__()
                # alias param
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.m = torch.nn.Linear(4, 4)
                self.m.bias = self.bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear output plus the shared bias.
                return self.m(x) + self.bias

        m = Foo()
        inps = (torch.randn(4, 4),)
        ep = export(m, inps, strict=True)
        unep = unflatten(ep)
        self.assertTrue(id(unep.m.bias) == id(unep.bias))

        # handle aliasing where one alias is unused
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module aliasing the layer bias to a top level parameter that
            # forward never uses directly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and linear layer and assigns
                # the parameter as the layer bias.
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.m = torch.nn.Linear(4, 4)
                self.m.bias = (
                    self.bias
                )  # self.bias is unused, aliasing should be handled

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer output.
                return self.m(x)

        m = Foo()
        inps = (torch.randn(4, 4),)
        ep = export(m, inps, strict=True)
        unep = unflatten(ep)
        self.assertTrue(torch.allclose(unep(*inps), m(*inps)))

    def test_attr_as_submod_input(self):
        # [FUNCTION DOC] Exports a model that passes a root buffer as an argument to
        # submodules in a module list and checks the unflattened module output is close to
        # eager using the assert close utility.
        class layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its second input to its first.
            def forward(self, x, const) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x plus const.
                return x + const

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Root with a buffer of ones and two submodules, both applied
            # in a loop with the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the buffer and builds a module list of two
                # layers.
                super().__init__()
                self.const = torch.nn.Buffer(torch.ones(4, 8))
                self.layers = torch.nn.ModuleList([layer() for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies each layer to the running value with the
                # buffer.
                for layer in self.layers:
                    x = layer(x, self.const)
                return x

        mod = M()
        x = torch.randn(4, 8)
        ep = export(mod, (x,), strict=True)
        unflattened = unflatten(ep)
        torch.testing.assert_close(unflattened(x), mod(x))

    def test_dedup_sym_size(self):
        # [FUNCTION DOC] Exports a model with dynamic first dimension and preserved call
        # signature for one submodule, where size and half-size computations are repeated in
        # the root and two submodules. Checks the unflattened graph counts of the symbolic
        # size operation: one in the root, one in the signature preserved submodule m1 and
        # none in m2, whose half-size value arrives as an input. Also runs the unflattened
        # module.
        # Here, sym_size & floor div are used in 3 subgraphs (top-level, m1, m2),
        # but only one copy of sym_size is created in the initial export graph.
        # For m1, sym_size & floordiv should be copied as recompute since we preserve the call signature,
        # but for m2 floordiv should be passed in as a placeholder.
        # Test that this is preserved, and the unflattened module runs correctly.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule slicing y to half the length of x.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first half of y based on x size.
                d = x.size(0) // 2
                return y[:d]

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Second submodule identical in behavior to the first.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first half of y based on x size.
                d = x.size(0) // 2
                return y[:d]

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Root holding the two submodules and adding the remaining
            # half of y to their results.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates m1 and m2.
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes half the size of x, calls both children and
                # returns the second half of y plus both child results.
                d = x.size(0) // 2
                m1_res = self.m1(x, y)
                m2_res = self.m2(x, y)
                return y[d:] + m1_res + m2_res

        inputs = (torch.ones(10), torch.ones(10))
        d_ = torch.export.Dim("foo", max=2048)
        d = 2 * d_
        ep = torch.export.export(
            M(),
            inputs,
            dynamic_shapes=((d,), (d,)),
            strict=False,
            preserve_module_call_signature=("m1",),
        )
        unflat = unflatten(ep)
        unflat(*inputs)

        fn_count_sym_size = lambda graph: [node.target for node in graph.nodes].count(
            torch.ops.aten.sym_size.int
        )
        self.assertEqual(fn_count_sym_size(unflat.graph), 1)
        self.assertEqual(fn_count_sym_size(unflat.m1.graph), 1)
        self.assertEqual(fn_count_sym_size(unflat.m2.graph), 0)

    def test_unflatten_eager(self):
        # [FUNCTION DOC] Exports a nested module, unflattens it with the interpreter disabled,
        # checks the root and a child report that they do not use the interpreter, compares
        # outputs and state dicts with eager, checks symbolic tracing works with concrete
        # arguments, and finally compiles one submodule with full graph compile and compares
        # again.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost module dividing its input by itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x divided by itself.
                return x / x

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a nested child and parameter of ones.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child and registers the parameter.
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the nested child then adds the parameter.
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of ones subtracted from the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the buffer.
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus the buffer.
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root with a parameter, applying a multiplication and then
            # two children.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates children and registers the root parameter.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the parameter and applies foo then bar.
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = export(orig_eager, (torch.rand(2, 3),), {}, strict=True)
        with _disable_interpreter():
            unflattened = unflatten(export_module)

        self.assertEqual(unflattened._run_with_interpreter, False)
        self.assertEqual(unflattened.foo._run_with_interpreter, False)

        inputs = (torch.rand(2, 3),)

        # Compare the root modules and all submodules
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)

        # Check state dicts are equal
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for name, value in orig_state_dict.items():
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))

        # Check composability with symbolic trace, as torchrec ddp uses symbolic
        # tracer
        symbolic_traced = torch.fx.symbolic_trace(unflattened, concrete_args=inputs)
        self.assertTrue(torch.allclose(orig_eager(*inputs), symbolic_traced(*inputs)))

        # torch.compile submodule
        unflattened.foo = torch.compile(unflattened.foo, fullgraph=True)
        self.compare_outputs(orig_eager, unflattened, inputs)

    def test_unflatten_none(self):
        # [FUNCTION DOC] Exports a model whose preserved submodule returns a pair containing
        # None, with a None input as well, unflattens, and checks the first output is close to
        # the eager one for a fresh instance.
        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the doubled input and None.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair of x plus x and None, ignoring y.
                return x + x, None

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Root doubling the input and returning the child result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child m2.
                super().__init__()
                self.m2 = M2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Doubles x and returns the child output pair.
                x = x + x
                return self.m2(x, y)

        ep = export(
            M(), (torch.rand(2, 3), None), preserve_module_call_signature=("m2",)
        )
        unflattened = unflatten(ep)
        inp = (torch.randn(2, 3), None)
        self.assertTrue(torch.allclose(M()(*inp)[0], unflattened(*inp)[0]))

    def test_unflatten_empty_branch(self):
        # [FUNCTION DOC] Exports a model that calls one submodule twice, once with a tensor
        # and once with None, (default export, then with a preserved call signature given as a
        # plain string), unflattens each and compares results with a fresh eager instance. The
        # submodule takes different branches in the two calls.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two ones tensors when the input is None and
            # the doubled and squared input otherwise.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Chooses a branch on whether the input is None and
                # returns a pair.
                if x is None:
                    return torch.ones(3), torch.ones(3)
                else:
                    return x + x, x * x

        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Root calling the branching module twice and summing the four
            # outputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the child module m.
                super().__init__()
                self.m = M()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the child on x and on y, and returns the sum of
                # the results.
                a, b = self.m(x)
                c, d = self.m(y)
                return a + b + c + d

        ep = torch.export.export(M1(), (torch.randn(3), None))
        unf = torch.export.unflatten(ep)
        inp = (torch.randn(3), None)
        self.assertTrue(torch.allclose(unf(*inp), M1()(*inp)))

        ep = torch.export.export(
            M1(), (torch.randn(3), None), preserve_module_call_signature="m"
        )
        unf = torch.export.unflatten(ep)
        inp = (torch.randn(3), None)
        self.assertTrue(torch.allclose(unf(*inp), M1()(*inp)))

    def test_unflatten_root_module_type(self) -> None:
        # [FUNCTION DOC] Exports a nested model, unflattens, and checks the root and a child
        # report a type name whose last dotted part equals the original class name, then
        # checks the outputs match eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doubling its input.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Root holding the doubling child and calling it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child.
                super().__init__()
                self.m = M()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the child applied to the input.
                return self.m(x)

        inp = (torch.randn(3),)
        ep = torch.export.export(M1(), inp)
        unf = torch.export.unflatten(ep)
        self.assertIsNotNone(unf.type_name())
        self.assertEqual(unf.type_name().split(".")[-1], "M1")
        self.assertEqual(unf.m.type_name().split(".")[-1], "M")
        self.assertTrue(torch.allclose(unf(*inp), M1()(*inp)))


if __name__ == "__main__":
    run_tests()
