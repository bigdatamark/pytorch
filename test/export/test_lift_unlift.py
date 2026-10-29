# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for the constant-lifting pass and unlifting of exported program states.
# Contains a small FX graph builder used to hand-construct graphs with parameters, buffers,
# constants and torchbind calls, tests that run the lift constants pass and inspect the lifted
# placeholders and mutated signature, a test of unlifting non-persistent buffers, and tests of the
# constant attribute map.
from collections import OrderedDict
from typing import Any

import torch
from torch._export.passes.lift_constants_pass import (
    ConstantAttrMap,
    lift_constants_pass,
)
from torch.export._unlift import _unlift_exported_program_lifted_states
from torch.export.exported_program import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.export.graph_signature import CustomObjArgument
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib


class GraphBuilder:
    # [CLASS DOC] HELPER: Test utility that incrementally builds an FX graph together with example
    # values. State: the graph, a mapping from names to nodes, a mapping from names to example
    # values, a per-module-path id table and counter for building nn_module_stack metadata, and a
    # mapping from placeholder nodes to input kinds.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty graph, empty name-to-node and name-to-value
        # tables, an empty module stack key table, id counter zero and an empty placeholder-kind
        # table.
        self.graph = torch.fx.Graph()
        self.nodes = {}
        self.values = {}
        self.nn_module_stack_key: dict[str, int] = {}
        self.latest_id = 0
        self.input_to_kind: dict[torch.fx.Node, InputKind] = {}

    def input(self, name: str, value: torch.Tensor, kind: InputKind):
        # [FUNCTION DOC] HELPER: Adds a placeholder with the given name, stores the example value
        # as its val metadata, registers node and value under the name, and records the input kind
        # for the node. Returns nothing.
        node = self.graph.placeholder(name)
        node.meta["val"] = value
        self.nodes[name] = node
        self.values[name] = value
        self.input_to_kind[node] = kind

    def add(self, x: str, y: str, out: str, module_fqn: str = ""):
        # [FUNCTION DOC] HELPER: Adds an aten add node over two previously registered names, naming
        # it out. Computes the example value by actually adding the stored values, stores it as val
        # metadata, attaches an nn_module_stack built from the module path (empty by default), and
        # registers node and value under out.
        node = self.graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            (self.nodes[x], self.nodes[y]),
            name=out,
        )
        self.values[out] = self.values[x] + self.values[y]
        node.meta["val"] = self.values[out]
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[out] = node

    def call_function(self, target, args, out: str, module_fqn: str = ""):
        # [FUNCTION DOC] HELPER: Adds a call_function node for an arbitrary target taking
        # registered names as arguments, named out. The example value is obtained by really calling
        # the target on the stored values; val metadata and nn_module_stack (from the module path)
        # are set and node and value are registered.
        arg_nodes = tuple(self.nodes[arg] for arg in args)
        arg_values = tuple(self.values[arg] for arg in args)
        node = self.graph.create_node(
            "call_function",
            target,
            arg_nodes,
            name=out,
        )
        self.values[out] = target(*arg_values)
        node.meta["val"] = self.values[out]
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[out] = node

    def constant(
        self, name: str, value: Any, target: str | None = None, module_fqn: str = ""
    ):
        # [FUNCTION DOC] HELPER: Adds a get_attr node fetching the given target attribute
        # (defaulting to the name itself), sets val and nn_module_stack metadata, and registers
        # node and value under name. The target may differ from the name so several nodes can read
        # one attribute.
        if target is None:
            target = name
        node = self.graph.get_attr(target)
        node.meta["val"] = value
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[name] = node
        self.values[name] = value

    def output(self, out: str):
        # [FUNCTION DOC] HELPER: Marks the node registered under the given name as the graph
        # output.
        self.graph.output(self.nodes[out])

    def create_nn_module_stack(
        self, module_fqn: str
    ) -> OrderedDict[int, tuple[str, type]]:
        # [FUNCTION DOC] HELPER: Builds an ordered mapping from integer ids to pairs of cumulative
        # dotted path and the generic module class, one entry per prefix of the dotted module path.
        # Ids are allocated from a counter the first time a prefix is seen and reused afterwards,
        # so prefixes are consistent across nodes.
        cur_name = ""
        nn_module_stack = OrderedDict()
        for atom in module_fqn.split("."):
            if cur_name == "":
                cur_name = atom
            else:
                cur_name = cur_name + "." + atom

            if cur_name not in self.nn_module_stack_key:
                id_counter = self.latest_id
                self.latest_id += 1
                self.nn_module_stack_key[cur_name] = id_counter
            else:
                id_counter = self.nn_module_stack_key[cur_name]

            nn_module_stack[id_counter] = (cur_name, torch.nn.Module)
        return nn_module_stack

    def create_input_specs(self):
        # [FUNCTION DOC] HELPER: Builds one input spec per placeholder in graph order, using the
        # recorded input kind, a tensor argument carrying the placeholder name, no target, and
        # persistent set to true for buffers and unset for other kinds. Returns the list.
        input_specs = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                input_specs.append(
                    InputSpec(
                        kind=self.input_to_kind[node],
                        arg=TensorArgument(name=node.name),
                        target=None,
                        persistent=(
                            True
                            if self.input_to_kind[node] == InputKind.BUFFER
                            else None
                        ),
                    )
                )
        return input_specs

    # NOTE: does not handle non-user-outputs atm
    def gen_graph_signature(self) -> ExportGraphSignature:
        # [FUNCTION DOC] HELPER: Builds an export graph signature from the graph. Requires exactly
        # one output node with exactly one argument, otherwise raises AssertionError. Input specs
        # come from the placeholders and each output arg becomes a user-output spec.
        # Parameter/buffer mutation outputs are not handled.
        output = [n for n in self.graph.nodes if n.op == "output"]
        if len(output) != 1:
            raise AssertionError(f"Expected 1 output node, got {len(output)}")
        output = output[0]
        if len(output.args) != 1:
            raise AssertionError(f"multiple outputs NYI, got {len(output.args)}")

        return ExportGraphSignature(
            input_specs=self.create_input_specs(),
            output_specs=[
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=n.name),
                    target=None,
                )
                for n in output.args
            ],
        )


class TestLift(TestCase):
    # [CLASS DOC] Tests of the lift constants pass over hand-built graphs, plus unlifting of
    # non-persistent buffers.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Standard setup, then loads the torchbind test library so the test
        # custom class and ops exist.
        super().setUp()
        load_torchbind_test_lib()

    def test_lift_basic(self):
        # [FUNCTION DOC] Builds a graph with a parameter, buffer, two user inputs, a tensor
        # constant and a torchbind object constant used by a custom op, runs the lift pass, and
        # checks: two constants returned under the generated lifted names, no get_attr nodes
        # remain, six placeholders, the lifted tensor and object placed right after params and
        # buffers with matching val metadata (a custom object argument for the object), and that
        # the graph signature gained a constant-tensor spec and a custom-object spec at those
        # positions.
        builder = GraphBuilder()

        builder.input("param", torch.rand(2, 3), InputKind.PARAMETER)
        builder.input("buffer", torch.rand(2, 3), InputKind.BUFFER)
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("y", torch.rand(2, 3), InputKind.USER_INPUT)

        builder.add("x", "y", out="foo")
        builder.add("foo", "param", out="bar")
        builder.add("bar", "buffer", out="baz")
        builder.constant("const_tensor", torch.rand(2, 3))
        builder.constant("const_obj", torch.classes._TorchScriptTesting._Foo(10, 20))
        builder.add("baz", "const_tensor", out="out")
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj", "x"),
            out="torchbind_out",
        )
        builder.add("out", "torchbind_out", out="final_out")
        builder.output("final_out")

        builder.graph.lint()
        graph = builder.graph
        const_tensor = builder.values["const_tensor"]
        const_obj = builder.values["const_obj"]

        root = {"const_tensor": const_tensor, "const_obj": const_obj}
        gm = torch.fx.GraphModule(root, graph)
        graph_signature = builder.gen_graph_signature()
        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 2)

        # The key of the constants table should match the fqn of the constant.
        # In this case, it's just the name of the constant, since the constant
        # is at the root submodule.
        # TODO(suo): we shouldn't hardcode these names in the test, this is an
        # internal detail of the pass.
        self.assertIn("lifted_tensor_0", constants)
        self.assertEqual(constants["lifted_tensor_0"], const_tensor)
        self.assertIn("lifted_custom_0", constants)
        self.assertEqual(constants["lifted_custom_0"], const_obj)

        # The constant node should be removed.
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # The constant should be lifted to a placeholder node.
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 6)

        # The lifted constant should be placed before user inputs but after params/buffers
        lifted_tensor_placeholder = placeholder_nodes[2]
        self.assertEqual(lifted_tensor_placeholder.target, "lifted_tensor_0")
        # It should have a val equivalent to the constant
        self.assertEqual(lifted_tensor_placeholder.meta["val"], const_tensor)

        lifted_obj_placeholder = placeholder_nodes[3]
        self.assertEqual(lifted_obj_placeholder.target, "lifted_custom_0")
        # It should have a val equivalent to the constant
        self.assertEqual(
            lifted_obj_placeholder.meta["val"],
            CustomObjArgument(
                name="lifted_custom_0",
                class_fqn="__torch__.torch.classes._TorchScriptTesting._Foo",
            ),
        )

        # Graph signature should have been mutated a way that reflects the placeholders.
        tensor_constant_input_spec = graph_signature.input_specs[2]
        self.assertEqual(tensor_constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(tensor_constant_input_spec.arg, TensorArgument)
        self.assertEqual(
            tensor_constant_input_spec.arg.name, lifted_tensor_placeholder.name
        )

        obj_constant_input_spec = graph_signature.input_specs[3]
        self.assertEqual(obj_constant_input_spec.kind, InputKind.CUSTOM_OBJ)
        self.assertIsInstance(obj_constant_input_spec.arg, CustomObjArgument)
        self.assertEqual(obj_constant_input_spec.arg.name, lifted_obj_placeholder.name)

    def test_lift_nested(self):
        # [FUNCTION DOC] Lifts a tensor constant that lives in a nested submodule path; checks one
        # constant is returned under the key prefixed with the submodule path, the get_attr node is
        # removed, four placeholders exist, the lifted placeholder comes first, and the signature
        # has a matching constant-tensor spec at the front.
        builder = GraphBuilder()
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("y", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("z", torch.rand(2, 3), InputKind.USER_INPUT)

        builder.add("x", "y", out="foo")
        builder.add("foo", "z", out="bar", module_fqn="foo")
        builder.constant("const_tensor", torch.rand(2, 3), module_fqn="foo")
        builder.add("bar", "const_tensor", "out")
        builder.output("out")

        graph = builder.graph
        graph.lint()

        const_tensor = builder.values["const_tensor"]
        root = {"const_tensor": builder.values["const_tensor"]}

        graph_signature = builder.gen_graph_signature()
        gm = torch.fx.GraphModule(root, graph)

        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 1)

        # The key of the constants table should match the fqn of the constant.
        self.assertIn("foo.lifted_tensor_0", constants)
        self.assertEqual(constants["foo.lifted_tensor_0"], const_tensor)

        # The constant node should be removed.
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # The constant should be lifted to a placeholder node.
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 4)

        # The lifted constant should be placed before user inputs but after params/buffers
        lifted_constant_placeholder = placeholder_nodes[0]
        self.assertEqual(lifted_constant_placeholder.target, "lifted_tensor_0")

        # Graph signature should have been mutated a way that reflects the placeholders.
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)
        self.assertEqual(constant_input_spec.arg.name, lifted_constant_placeholder.name)

    def test_duplicate_constant_access(self):
        # [FUNCTION DOC] Uses graphs where one tensor attribute is read by two get_attr nodes and
        # one torchbind object is read through two attributes, then lifts. Checks that only two
        # constants are produced, no get_attr nodes remain, there are three placeholders and three
        # input specs, and the first spec is a constant tensor.
        const = torch.rand(2, 3)
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)

        builder = GraphBuilder()
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.constant("const_tensor", const, target="const_tensor")
        # loading the same target twice
        builder.constant("const_tensor2", const, target="const_tensor")

        # loading the same object twice with different targets
        builder.constant("const_obj", const_obj)
        builder.constant("const_obj2", const_obj)
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj", "x"),
            out="torchbind_out",
        )
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj2", "x"),
            out="torchbind_out2",
        )
        builder.add("x", "const_tensor", out="foo")
        builder.add("foo", "const_tensor2", out="tensor_out")
        builder.add("torchbind_out", "torchbind_out2", out="obj_out")
        builder.add("tensor_out", "obj_out", out="out")
        builder.output("out")
        graph = builder.graph
        graph.lint()

        input_specs = builder.create_input_specs()
        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=builder.nodes["out"].name),
                target=None,
            )
        ]
        graph_signature = ExportGraphSignature(input_specs, output_specs)

        root = {"const_tensor": const, "const_obj": const_obj, "const_obj2": const_obj}
        gm = torch.fx.GraphModule(root, graph)

        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 2)

        # All get_attr nodes should be removed
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # There should only be two additional inputs (plus the existing user input)
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 3)

        # Graph signature should have been mutated a way that reflects the placeholders.
        self.assertEqual(len(graph_signature.input_specs), 3)
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)

    def test_unlift_nonpersistent_buffer(self):
        # [FUNCTION DOC] Exports a module with a non-persistent buffer that is mutated in forward
        # (non-strict), unlifts the program states, and checks the stateful module still holds the
        # buffer with equal value, lists it among non-persistent buffers, and omits it from the
        # state dict.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one non-persistent zero buffer whose forward adds one
            # to the buffer in place and returns the sum of the input plus the sum of the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the non-persistent buffer holding a single zero.
                super().__init__()
                self.register_buffer(
                    "non_persistent_buf", torch.zeros(1), persistent=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Increments the buffer in place and returns the input sum
                # plus the buffer sum.
                self.non_persistent_buf.add_(1)
                return x.sum() + self.non_persistent_buf.sum()

        foo = Foo()
        exported = torch.export.export(foo, (torch.ones(5, 5),), strict=False)
        stateful_gm = _unlift_exported_program_lifted_states(exported)

        # Check the unlifted stateful_gm contains the original non-persistent buffer
        self.assertTrue(hasattr(stateful_gm, "non_persistent_buf"))
        non_persistent_buf = stateful_gm.get_buffer("non_persistent_buf")
        self.assertEqual(non_persistent_buf, foo.get_buffer("non_persistent_buf"))
        self.assertIn("non_persistent_buf", stateful_gm._non_persistent_buffers_set)
        self.assertNotIn("non_persistent_buf", stateful_gm.state_dict())


class ConstantAttrMapTest(TestCase):
    # [CLASS DOC] Tests for the constant attribute map container, including torchbind object keys.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Standard setup, then loads the torchbind test library.
        super().setUp()
        load_torchbind_test_lib()

    def test_dict_api(self):
        # [FUNCTION DOC] Exercises the mapping behavior: after adding an object and a tensor with
        # names, checks length, iteration and keys give the object and tensor, values give lists of
        # names, lookups return lists, membership works, adding an unsupported key type raises
        # TypeError, and deleting an entry shrinks the map.
        constant_attr_map = ConstantAttrMap()
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)
        const_tensor = torch.ones(2, 3)
        constant_attr_map.add(const_obj, "foo.bar")
        constant_attr_map.add(const_tensor, "foo.bar.baz")
        self.assertEqual(len(constant_attr_map), 2)
        self.assertEqual(list(constant_attr_map), [const_obj, const_tensor])
        self.assertEqual(list(constant_attr_map.keys()), [const_obj, const_tensor])
        self.assertEqual(
            list(constant_attr_map.values()), [["foo.bar"], ["foo.bar.baz"]]
        )
        self.assertEqual(constant_attr_map[const_obj], ["foo.bar"])
        self.assertEqual(constant_attr_map[const_tensor], ["foo.bar.baz"])
        self.assertTrue(const_obj in constant_attr_map)
        with self.assertRaises(TypeError):
            constant_attr_map.add(1, "foo.bar")

        del constant_attr_map[const_obj]
        self.assertEqual(len(constant_attr_map), 1)


if __name__ == "__main__":
    run_tests()
