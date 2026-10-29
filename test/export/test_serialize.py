"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_sym_bool)
"""

# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for export serialization: round trips of exported programs through
# serialize/deserialize and save/load, covering metadata, constants, aliasing, symbolic shapes,
# custom ops and objects, triton kernels, version checks and predispatch wrappers. Import time:
# registers two opaque custom classes with the library and builds the list of export-db cases.
import copy
import io
import math
import tempfile
import unittest
import zipfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


if HAS_GPU:
    import triton
    import triton.language as tl

    from torch.library import wrap_triton
    from torch.utils._triton import has_triton
else:

    def has_triton():
        # [FUNCTION DOC] HELPER: Fallback used when no GPU support is present; reports that triton
        # is unavailable by always returning False.
        return False


import torch
import torch._dynamo as torchdynamo
import torch._export.serde.schema as schema
import torch.export._trace
import torch.utils._pytree as pytree
from torch._export.db.case import ExportCase, SupportLevel
from torch._export.db.examples import all_examples
from torch._export.serde.schema import ArgumentKind
from torch._export.serde.serialize import (
    _dict_to_dataclass,
    _to_json_bytes,
    canonicalize,
    deserialize,
    deserialize_torch_artifact,
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    GraphModuleSerializer,
    serialize,
    SerializeError,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._library.opaque_object import get_opaque_type_name, register_custom_class
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim, export, load, save, unflatten
from torch.export.pt2_archive.constants import ARCHIVE_VERSION_PATH
from torch.fx.experimental.symbolic_shapes import is_concrete_int, ValueRanges
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


@dataclass(frozen=True)
class _OpaqueConfig(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Frozen dataclass with one integer field, scale, used as an opaque
    # value type constant (registered with the constant kind) that serialization must carry
    # through.
    scale: int

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Returns the pair FX codegen needs to rebuild the object: a repr
        # string that reconstructs it from its scale, and a mapping of the class name to the class.
        return (
            f"_OpaqueConfig(scale={self.scale!r})",
            {"_OpaqueConfig": _OpaqueConfig},
        )


register_custom_class(_OpaqueConfig, typ="constant")


class _OpaqueEngine(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Opaque reference-type object (registered with the symbolic kind) holding
    # an integer multiplier attribute.
    def __init__(self, multiplier: int):
        # [FUNCTION DOC] HELPER: Runs the base initializer and stores the integer multiplier on
        # the instance.
        super().__init__()
        self.multiplier = multiplier


register_custom_class(_OpaqueEngine, typ="symbolic")


def get_filtered_export_db_tests():
    # [FUNCTION DOC] HELPER: Returns a list of name and case pairs from all export-db
    # examples whose support level is SUPPORTED. Used to parametrize the export-db round-trip
    # test; no side effects.
    return [
        (name, case)
        for name, case in all_examples().items()
        if case.support_level == SupportLevel.SUPPORTED
    ]


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    # [CLASS DOC] Tests of serialization (export, then serialize or save, then deserialize or load)
    # of exported programs: extension ops, metadata, constants, aliasing, symbolic shapes,
    # ranges and triton kernels. Skipped when dynamo is unsupported.
    def test_export_with_extension_op_serialization(self):
        # [FUNCTION DOC] Registers a custom extension operator type with the serializer, swaps it
        # into an exported add node, and checks serialize followed by deserialize restores exactly
        # one call to that operator. Side effect: registers a global extension handler.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds the input to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the input doubled
                # by addition.
                return x + x

        class FooExtensionOp:
            # [CLASS DOC] HELPER: Fake extension operator type: constant hash, equal to any
            # instance of its own type, callable by forwarding to the aten add Tensor overload, and
            # named foo.my_op.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns a constant hash so all instances collide.
                return 0

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type.
                return type(other) is type(self)

            def __call__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the aten add Tensor overload.
                return torch.ops.aten.add.Tensor(*args, **kwargs)

            @property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property giving the operator name foo.my_op.
                return "foo.my_op"

        class ExtensionVerifier(torch._export.verifier.Verifier):
            # [CLASS DOC] HELPER: Verifier for the made-up FOO dialect that additionally allows the
            # fake extension operator type.
            dialect = "FOO"

            def allowed_op_types(self):
                # [FUNCTION DOC] HELPER: Returns the base verifier's allowed op types with the fake
                # extension operator type appended.
                return super().allowed_op_types() + (FooExtensionOp,)

        class FooExtensionHandler(torch._export.serde.serialize.ExtensionHandler):
            # [CLASS DOC] HELPER: Serializer extension handler for the foo namespace: maps the fake
            # operator to the name my_op and back, and reports the aten add Tensor schema for it.
            @classmethod
            def namespace(cls):
                # [FUNCTION DOC] HELPER: Returns the namespace string foo.
                return "foo"

            @classmethod
            def to_op_name(cls, op):
                # [FUNCTION DOC] HELPER: Returns the fixed op name my_op for any operator.
                return "my_op"

            @classmethod
            def from_op_name(cls, name: str):
                # [FUNCTION DOC] HELPER: Checks (through the enclosing test's assertEqual) that the
                # name is my_op and returns a new fake extension operator instance.
                self.assertEqual(name, "my_op")
                return FooExtensionOp()

            @classmethod
            def op_schema(cls, op):
                # [FUNCTION DOC] HELPER: Returns the schema of the aten add Tensor overload
                # regardless of the operator.
                return torch.ops.aten.add.Tensor._schema

        inp = (torch.ones(10),)
        ep = export(TestModule(), inp, strict=True)

        # Register the custom op handler.
        foo_custom_op = FooExtensionOp()
        torch._export.serde.serialize.register_extension(
            FooExtensionOp, FooExtensionHandler
        )

        new_gm = copy.deepcopy(ep.graph_module)
        # Inject the custom operator.
        for node in new_gm.graph.nodes:
            if node.name == "add":
                node.target = foo_custom_op

        new_ep = ep._update(new_gm, ep.graph_signature, verifiers=[ExtensionVerifier])
        serialized = serialize(new_ep)
        deserialized = deserialize(serialized)
        self.assertEqual(
            len(
                deserialized.graph.find_nodes(op="call_function", target=foo_custom_op)
            ),
            1,
        )

    def test_predispatch_export_with_autograd_op(self):
        # [FUNCTION DOC] Exports with pre-dispatch under no-grad a module that re-enables grad
        # internally, saves and loads it through a byte buffer, and checks outputs and their
        # requires_grad flag match the original program.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds the input to itself inside a grad-
            # enabled region.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Enables grad and returns
                # the input added to itself.
                with torch.enable_grad():
                    return x + x

        inp = (torch.ones(10),)
        with torch.no_grad():
            from torch.export._trace import _export

            ep = _export(Foo(), inp, pre_dispatch=True)

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)
        self.assertEqual(exp_out.requires_grad, actual_out.requires_grad)

    def test_export_example_inputs_preserved(self):
        # [FUNCTION DOC] Checks that example inputs (positional and keyword) stored on an exported
        # program survive save and load: runs both programs on the stored example inputs and
        # compares outputs.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Has a parameter p; forward sums two inputs and
            # optionally adds p.
            """A test module with that has multiple args and uses kwargs"""

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameter p, a 2x3 tensor of ones.
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x, y, use_p=False):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the sum of x and
                # y, plus the parameter p when use_p is true. Unused flag defaults to false.
                out = x + y
                if use_p:
                    out += self.p
                return out

        model = MyModule().eval()
        random_inputs = (torch.rand([2, 3]), torch.rand([2, 3]))
        exp_program = export(model, random_inputs, {"use_p": True}, strict=True)

        output_buffer = io.BytesIO()
        # Tests that example inputs are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)
        # Extract the example inputs from before and after saving.
        orig_args, orig_kwargs = exp_program.example_inputs
        loaded_args, loaded_kwargs = loaded_model.example_inputs
        # Run both modules and confirm that outputs match.
        orig_out = exp_program.module()(*orig_args, **orig_kwargs)
        loaded_out = loaded_model.module()(*loaded_args, **loaded_kwargs)
        self.assertEqual(orig_out, loaded_out)

    def test_metadata_run_decomp_serder(self):
        # [FUNCTION DOC] Saves and loads a program of a sine op, runs decompositions on the loaded
        # program, and checks the resulting graph code matches an expected text containing a single
        # aten sin call.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Applies sine to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the sine of x.
                return x.sin()

        exp_program = export(M(), (torch.randn(4, 4),), strict=True)

        output_buffer = io.BytesIO()
        # Tests that example forward arg names are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)

        ep = loaded_model.run_decompositions({})
        # We should preserve the original module name
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x);  x = None
    return (sin,)""",
        )

    def test_metadata_parsing_with_layer_split(self):
        # [FUNCTION DOC] Round-trips a model whose forward splits a sequential container
        # (introducing commas and parentheses in module-stack metadata) through save and load, and
        # checks outputs are equal.
        # Tests that modules with more complicated layer patterns can be serialized
        # and deserialized correctly.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a sequential stack of three SiLU layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a sequential container of three SiLU activations.
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the first layer,
                # then the rest of the sequential slice, to the input.
                # Splitting layers of a sequential stack introduces commas and parens
                # into metadata trace.
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h)
                return h

        inp = (torch.ones(10),)
        # Module will only be able to roundtrip if metadata
        # can be correctly parsed.
        ep = export(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_nested_layer_split(self):
        # [FUNCTION DOC] Round-trips a model whose submodules and buffer have unusual names
        # (brackets, parentheses, colon) through save and load and checks the outputs match.
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Sequential stack of three SiLU layers; forward adds
            # 2 to the output of the layer slice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a sequential container of three SiLU activations.
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the first layer
                # then the remaining slice of layers, and adds 2.
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h) + 2
                return h

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Registers two Bar submodules and a buffer under
            # names containing special characters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers submodules and a random buffer under the
                # special names.
                super().__init__()
                self.register_module("a[(1)]", Bar())
                self.register_module("b[(2)]", Bar())
                self.register_buffer("c:[22]", torch.randn(1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Fetches the two
                # submodules and buffer by name, chains the two submodules on x and adds
                # the buffer.
                out_a, out_b = getattr(self, "a[(1)]"), getattr(self, "b[(2)]")
                out_c = getattr(self, "c:[22]")
                h = out_a(x)
                h = out_b(h)
                return h + out_c

        inp = (torch.ones(10),)
        ep = export(Foo(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_param_mutation(self):
        # [FUNCTION DOC] Exports a module that mutates a parameter under no-grad, runs
        # decompositions, saves and loads it, and checks the loaded signature records the mutated
        # parameter mapping from the div node to the parameter.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Has a 4x4 parameter that forward halves in place
            # under no-grad before adding it to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 4x4 parameter of ones.
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Divides the parameter by
                # 2 in place without gradient tracking and returns x plus the parameter.
                with torch.no_grad():
                    self.parameter.div_(2)
                return x + self.parameter

        foo = Foo()
        ep = torch.export.export(foo, (torch.rand(4, 4),)).run_decompositions()
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)
        val = loaded_ep.graph_signature.parameters_to_mutate
        self.assertEqual({"div": "parameter"}, val)

    def test_serialize_constant_outputs(self):
        # [FUNCTION DOC] Round-trips a program whose forward returns a tensor, None and an integer
        # constant, and checks the loaded program produces the same outputs.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns x plus one, None and the constant 1024.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns a tuple of x plus
                # one, None, and 1024.
                # Along with tensor output, return Nonetype
                # and constant. Although these outputs aren't
                # very useful, they do show up in graphs.
                return x + 1, None, 1024

        # Check that module can be roundtripped, thereby confirming proper deserialization.
        inp = (torch.ones(10),)
        ep = export(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_multiple_returns_from_node(self) -> None:
        # [FUNCTION DOC] Serializes a program using layer norm (multiple outputs from one node) and
        # checks the last node targets native layer norm with three outputs whose tensor names are
        # all unique.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Applies functional layer norm with given weight
            # and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x, w, b):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns layer norm of x
                # over its trailing dimensions with weight w, bias b and epsilon 1e-5.
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
            strict=True,
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.native_layer_norm.default")
        # aten::native_layer_norm returns 3 tensors
        self.assertEqual(len(node.outputs), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_sym_int(self) -> None:
        # [FUNCTION DOC] Exports with dynamic shapes, serializes, and checks every sym_size node
        # has input argument names self and dim.
        class DynamicShapeSimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Matmul plus bias, halved, flattened by a view of the
            # product of its two sizes, then concatenated with itself.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes the matmul of a
                # and b plus c, halves it, views it flat using its symbolic sizes and returns the
                # flat tensor concatenated with itself.
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        sym_size_nodes = [
            node
            for node in serialized.exported_program.graph_module.graph.nodes
            if node.target == "torch.ops.aten.sym_size.int"
        ]
        for node in sym_size_nodes:
            self.assertEqual(node.inputs[0].name, "self")
            self.assertEqual(node.inputs[1].name, "dim")

    def test_serialize_sym_float(self) -> None:
        # [FUNCTION DOC] Builds a model with a symbolic float multiplier and some dimension objects
        # but never exports or asserts anything (an existing TODO says it tests nothing).
        # KNOWN ISSUE: the test body performs no export, serialization or assertion, so it
        # verifies nothing.
        # TODO(rec): This doesn't seem to test anything!

        class DynamicFloatSimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Scales the matmul result by a stored symbolic float
            # multiplier then flattens and concatenates.
            def __init__(self, multiplier: torch.SymFloat):
                # [FUNCTION DOC] HELPER: Stores the given symbolic float as the
                # multiplier attribute.
                super().__init__()
                self.multiplier = multiplier

            def forward(self, a, b, c) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes the multiplied
                # result, views it flat via its symbolic sizes and concatenates it with itself.
                d = (torch.matmul(a, b) + c) / 2
                e = d * self.multiplier
                e_s0 = e.shape[0]
                e_s1 = e.shape[1]
                e_s3 = e_s0 * e_s1
                f = e.view(e_s3)
                return torch.cat([f, f])

        multiplier_sym = torch.SymFloat("multiplier_sym")
        _model = DynamicFloatSimpleModel(multiplier_sym)
        _inputs = (
            torch.randn(2, 4),
            torch.randn(4, 7),
            torch.randn(2, 7),
        )
        _dim0_ac = Dim("dim0_ac")
        _dim1_bc = Dim("dim1_b")

    def test_serialize_infinite_sym_int(self) -> None:
        # [FUNCTION DOC] Same dynamic-shape model as the sym int test; after serialization checks
        # every range constraint has an unbounded (None) maximum value.
        class DynamicShapeSimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Matmul plus bias, halved, flattened and concatenated
            # with itself.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes the halved
                # sum of the matmul of a and b with c, views it flat and returns it concatenated
                # with itself.
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        for v in serialized.exported_program.range_constraints.values():
            self.assertEqual(v.max_val, None)

    def test_symint_list(self):
        # [FUNCTION DOC] Creates an unbacked symbolic integer and serializes the inputs of an ones
        # op with a size list mixing ints and that symbol, checking one input results and its
        # argument type is a symbolic int list.
        # This reflects the behavior from inductor's ExternFallbackNode
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        symint = shape_env.create_unbacked_symint()
        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        res = serializer.serialize_inputs(
            torch.ops.aten.ones.default, ([1, symint, 3],), {}
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].arg._type, "as_sym_ints")

    def test_serialize_list_returns(self) -> None:
        # [FUNCTION DOC] Serializes a program returning the split of a tensor and checks the
        # last node is the split_with_sizes op with a single list output of three tensors with
        # unique names.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Splits the input into chunks of size 2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the split of x
                # into pieces of 2 rows.
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        exported_module = export(MyModule(), (input,), strict=True).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        # split.Tensor gets decomposed to split_with_sizes by the core ATen decomposition table
        self.assertEqual(node.target, "torch.ops.aten.split_with_sizes.default")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_nonfinite_inputs(self) -> None:
        # [FUNCTION DOC] Serializes a program adding infinity, negative infinity and NaN scalars
        # and checks the JSON bytes parse without ever hitting a nonstandard float constant (the
        # parse hook raises an error).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds infinity, negative infinity then NaN scalars to
            # the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Chains three scalar adds
                # with infinite, negative infinite and NaN values and returns the result.
                x = torch.ops.aten.add.Scalar(x, math.inf)
                x = torch.ops.aten.add.Scalar(x, -math.inf)
                return torch.ops.aten.add.Scalar(x, math.nan)

        fn = Module()
        ep = torch.export.export(
            fn,
            (torch.randn(3, 2),),
        )
        json_bytes = _to_json_bytes(
            ExportedProgramSerializer().serialize(ep).exported_program
        )
        import json

        def parse_constant(x):
            # [FUNCTION DOC] HELPER: Parse hook for the JSON loader that raises a runtime error
            # naming any nonstandard float constant encountered.
            raise RuntimeError(f"Invalid JSON float: {x}")

        json.loads(json_bytes, parse_constant=parse_constant)

    def test_multi_return_some_unused(self) -> None:
        # [FUNCTION DOC] Checks that for an op with several outputs where only the first is
        # used, the serialized node still lists both outputs matching the op schema, with unique
        # tensor names.
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the first output of the var_mean op over
            # dimension one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Calls var_mean with
                # correction on dimension 1 and returns only its first result.
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export(
            MyModule(), (torch.ones([512, 512], requires_grad=True),), strict=True
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_multi_return_list_tensor_unused(self) -> None:
        # [FUNCTION DOC] Defines a scoped custom op returning a tensor and a list of tensors,
        # exports a module using only the tensor, removes dead code, and checks serialization
        # synthesizes tensor names containing the word unused for the dropped list output, then
        # that a deserialize round trip gives allclose results.
        """
        Make sure serialization handles unused List[Tensor] outputs in
        multi-return ops. The getitem node for the unused list output is
        removed by DCE, so the serializer must synthesize names.
        """
        with torch.library._scoped_library("mylib", "DEF") as lib:
            lib.define(
                "tensor_and_list(Tensor x) -> (Tensor, Tensor[])",
            )

            @torch.library.impl(lib, "tensor_and_list", "CPU")
            def tensor_and_list_impl(x):
                # [FUNCTION DOC] HELPER: CPU kernel of the custom op: returns x doubled and a list
                # of x and x plus one.
                return x * 2, [x, x + 1]

            @torch.library.impl(lib, "tensor_and_list", "Meta")
            def tensor_and_list_meta(x):
                # [FUNCTION DOC] HELPER: Meta kernel of the custom op with the same body as the
                # CPU kernel.
                return x * 2, [x, x + 1]

            class MyModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Uses the custom op and returns only its first
                # (tensor) output.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Calls the custom op
                    # and returns its first output, ignoring the list.
                    # Only use the first (Tensor) output; the List[Tensor] is unused
                    return torch.ops.mylib.tensor_and_list(x)[0]

            exported_module = export(MyModule(), (torch.ones(3),), strict=True)
            # Simulate the scenario where DCE removes the getitem for the
            # unused List[Tensor] output (as happens in real serialization
            # pipelines like package_sigmoid_model).
            exported_module.graph.eliminate_dead_code()
            exported_module.graph_module.recompile()

            serialized = ExportedProgramSerializer().serialize(exported_module)
            node = serialized.exported_program.graph_module.graph.nodes[-1]
            self.assertEqual(node.target, "torch.ops.mylib.tensor_and_list.default")
            self.assertEqual(len(node.outputs), 2)

            # First output is a single tensor
            self.assertTrue(node.outputs[0].as_tensor.name != "")
            # Second output is the unused list -- should still have tensor entries
            self.assertTrue(len(node.outputs[1].as_tensors) > 0)
            for t in node.outputs[1].as_tensors:
                self.assertIn("_unused_", t.name)

            # Roundtrip: deserialize and verify functional correctness
            deserialized_ep = deserialize(serialize(exported_module))
            inp = torch.randn(3)
            self.assertTrue(
                torch.allclose(
                    exported_module.module()(inp),
                    deserialized_ep.module()(inp),
                )
            )

    def test_rational_ranges(self) -> None:
        # [FUNCTION DOC] Sets a rational-valued range constraint on a dynamic symbol (lower 5/3,
        # upper 10/3), serializes, and checks the serialized min and max are rounded inward to 2
        # and 3. Raises an error if there is not exactly one range constraint.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        ep = export(
            M(), (torch.randn(4),), dynamic_shapes=({0: Dim("temp")},), strict=True
        )

        range_constraints = list(ep.range_constraints.keys())
        if len(range_constraints) != 1:
            raise AssertionError(
                f"Expected 1 range constraint, got {len(range_constraints)}"
            )
        symint = range_constraints[0]

        import sympy

        upper_range = sympy.Rational(10, 3)
        lower_range = sympy.Rational(10, 6)
        ep.range_constraints[symint] = ValueRanges(lower=lower_range, upper=upper_range)

        serialized = ExportedProgramSerializer().serialize(ep)
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].min_val, 2
        )
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].max_val, 3
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not has_triton(), "requires cuda and triton"
    )
    def test_triton_hop(self) -> None:
        # [FUNCTION DOC] Needs CUDA and triton. Exports models calling a wrapped triton kernel
        # (default and eight-warp variants), decomposes without decomposing the custom triton op,
        # serializes, and checks the functional triton wrapper node: positional kernel argument
        # names and values, keyword fields (name, grid, output indices, num warps, parameter names
        # and types), one tensor-list output named getitem, and that deserializing it raises a
        # SerializeError saying deserialize is not implemented.
        # [FUNCTION DOC] HELPER: Triton kernel adding two input pointers elementwise plus a float
        # and an int scalar, with masking by block size, storing to the output pointer.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            fval,
            ival,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y + fval + ival
            tl.store(out_ptr + offsets, output, mask=mask)

        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates an output like x, launches the wrapped add kernel
            # over a grid derived from the block size with float 3.14 and int 42 constants, and
            # returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Launch grid function: number of blocks needed to cover the
                # element count for the given block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrap_triton(add_kernel)[grid](
                x, y, output, n_elements, 3.14, 42, BLOCK_SIZE=16
            )

            return output

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls the add helper on its inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the result of the
                # triton add helper.
                return custom_add(x, y)

        def custom_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Same as the plain add helper but launches the kernel with
            # eight warps.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Launch grid function: number of blocks needed to cover the
                # element count for the given block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrap_triton(add_kernel)[grid](
                x, y, output, n_elements, 3.14, 42, BLOCK_SIZE=16, num_warps=8
            )

            return output

        class MyModelAutotune(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls the eight-warp add helper on its inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the result of the
                # autotune add helper.
                return custom_add_autotune(x, y)

        device = "cuda"

        for m in [MyModel().to(device), MyModelAutotune().to(device)]:
            args = (torch.randn(3, device=device), torch.randn(3, device=device))
            ep = torch.export.export(m, args=args)
            ep = ep.run_decompositions(decompose_custom_triton_ops=False)
            if not torch.allclose(m(*args), ep.module()(*args)):
                raise AssertionError(
                    "Exported model output does not match eager output"
                )

            serialized = ExportedProgramSerializer().serialize(ep)

            for node in serialized.exported_program.graph_module.graph.nodes:
                if (
                    node.target
                    == "torch.ops.higher_order.triton_kernel_wrapper_functional"
                ):
                    triton_node = node

            self.assertIsNotNone(triton_node)

            args = []
            arg_names = []
            kwargs = {}

            for arg in triton_node.inputs:
                if arg.kind == ArgumentKind.POSITIONAL:
                    args.append(arg.arg)
                    arg_names.append(arg.name)
                elif arg.kind == ArgumentKind.KEYWORD:
                    kwargs[arg.name] = arg.arg

            self.assertEqual(len(args), 6)
            # Positional args carry kernel parameter names
            expected_param_names = [
                "in_ptr0",
                "in_ptr1",
                "out_ptr",
                "n_elements",
                "fval",
                "ival",
            ]
            self.assertEqual(arg_names, expected_param_names)
            # Always: name, grid, output_indices, num_warps,
            # kernel_param_names, kernel_param_types
            # Triton version dependent: num_cpu_threads, shared_memory_bytes
            self.assertTrue(len(kwargs) >= 6)

            for i in range(3):
                self.assertIsNotNone(args[i].as_tensor)

            self.assertEqual(args[3].as_int, 3)
            self.assertAlmostEqual(args[4].as_float, 3.14, places=2)
            self.assertEqual(args[5].as_int, 42)
            kernel_name = kwargs["name"].as_string
            symbol_name = kernel_name.rpartition("_")[0]
            self.assertEqual(symbol_name, "add_kernel")
            self.assertEqual(kwargs["grid"].as_ints, [1, 1, 1])
            self.assertEqual(kwargs["output_indices"].as_ints, [2])
            self.assertEqual(
                kwargs["num_warps"].as_int, 8 if isinstance(m, MyModelAutotune) else 4
            )

            if "num_cpu_threads" in kwargs:
                self.assertEqual(kwargs["num_cpu_threads"].as_int, 0)
            if "shared_memory_bytes" in kwargs:
                self.assertEqual(kwargs["shared_memory_bytes"].as_int, 0)

            self.assertIn("kernel_param_names", kwargs)
            self.assertEqual(
                kwargs["kernel_param_names"].as_strings, expected_param_names
            )
            self.assertIn("kernel_param_types", kwargs)
            self.assertEqual(
                len(kwargs["kernel_param_types"].as_strings),
                len(expected_param_names),
            )

            self.assertEqual(len(triton_node.outputs), 1)
            self.assertIsNotNone(triton_node.outputs[0].as_tensors)
            self.assertEqual(
                len(triton_node.outputs[0].as_tensors),
                len(kwargs["output_indices"].as_ints),
            )
            self.assertEqual(triton_node.outputs[0].as_tensors[0].name, "getitem")

            with self.assertRaisesRegex(
                SerializeError,
                "deserialize nyi for torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional",
            ):
                ExportedProgramDeserializer().deserialize(
                    serialized.exported_program,
                    serialized.state_dict,
                    serialized.constants,
                    serialized.example_inputs,
                )

    @unittest.skipIf(
        not torch.cuda.is_available() or not has_triton(), "requires cuda and triton"
    )
    def test_triton_constexpr_matching(self) -> None:
        # [FUNCTION DOC] Needs CUDA and triton. Exports a model using a wrapped triton kernel that
        # has a bool and an int compile-time constant, and checks serialization succeeds and
        # produces a functional triton wrapper node (exercises matching of constant values to
        # kernel cache entries).
        """Test that constexpr values are properly matched during serialization.

        This tests the normalization logic that handles various constexpr types
        (bool, int, float, string) when matching kernel cache entries. The kernel
        signature stores constexprs as strings which are parsed back to Python types.
        """

        # [FUNCTION DOC] HELPER: Triton kernel with block size and fast path compile-time constants
        # that copies masked elements of the input to the output.
        @triton.jit
        def kernel_with_constexprs(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            USE_FAST_PATH: "tl.constexpr",  # bool constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x, mask=mask)

        def custom_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocates an output like x and launches the wrapped constexpr
            # kernel with block size 128 and the fast path flag true, returning the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Launch grid function: number of blocks needed to cover the
                # element count for the given block size.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrap_triton(kernel_with_constexprs)[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=128,
                USE_FAST_PATH=True,  # bool constexpr
            )
            return output

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls the constexpr kernel helper on its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the result of the
                # custom op helper.
                return custom_op(x)

        device = "cuda"
        m = Model().to(device)
        args = (torch.randn(1024, device=device),)

        # Run the model in eager mode first to warm up the Triton cache
        eager_result = m(*args)

        ep = torch.export.export(m, args=args)
        ep = ep.run_decompositions(decompose_custom_triton_ops=False)
        if not torch.allclose(eager_result, ep.module()(*args)):
            raise AssertionError("Exported model output does not match eager result")

        # This should not raise - constexpr matching should work for bool values
        serialized = ExportedProgramSerializer().serialize(ep)

        # Verify the triton node was serialized
        triton_node = None
        for node in serialized.exported_program.graph_module.graph.nodes:
            if node.target == "torch.ops.higher_order.triton_kernel_wrapper_functional":
                triton_node = node
                break
        self.assertIsNotNone(triton_node)

    def test_kwargs_default(self) -> None:
        # [FUNCTION DOC] Checks that default keyword arguments of searchsorted are serialized even
        # when not given: node has four inputs, with right equal to true and side equal to the
        # string right.
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls searchsorted of x against random values with
            # side right and the right flag true.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the searchsorted
                # result of x against a random 3x2 tensor.
                values = torch.randn(3, 2)
                return torch.searchsorted(x, values, side="right", right=True)

        f = Foo()

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export(f, (x,), strict=True).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)

        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(node.inputs[2].name, "right")
        self.assertEqual(node.inputs[2].arg.as_bool, True)
        self.assertEqual(node.inputs[3].name, "side")
        self.assertEqual(node.inputs[3].arg.as_string, "right")

    def test_canonicalize(self) -> None:
        # [FUNCTION DOC] Serializes a program with two equivalent adds in different operand order,
        # canonicalizes it, and checks the tensor names of the first two nodes' first inputs are
        # sorted in order.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Computes y plus x and x plus y and sums them.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns b plus a where a
                # is y plus x and b is x plus y.
                a = y + x
                b = x + y
                return b + a

        ep = export(Module(), (torch.randn(3, 2), torch.randn(3, 2)), strict=True)
        s = ExportedProgramSerializer().serialize(ep)
        c = canonicalize(s.exported_program)
        g = c.graph_module.graph
        self.assertLess(
            g.nodes[0].inputs[0].arg.as_tensor.name,
            g.nodes[1].inputs[0].arg.as_tensor.name,
        )

    def test_int_list(self) -> None:
        # [FUNCTION DOC] Serializes a sum over an empty dimension list and checks the dimension
        # argument is serialized with the ints argument type.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls the sum dim_IntList op with an empty
            # dimension list.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns sum over an empty
                # list of dimensions.
                return torch.ops.aten.sum.dim_IntList(x, [])

        ep = torch.export.export(M(), (torch.randn(3, 2),), strict=True)
        serialized = ExportedProgramSerializer().serialize(ep)
        for node in serialized.exported_program.graph_module.graph.nodes:
            if "aten.sum.dim_IntList" in node.target:
                self.assertEqual(node.inputs[1].arg.type, "as_ints")

    def test_empty_constant(self) -> None:
        # [FUNCTION DOC] Exports a linear model, saves and loads it, and checks eager and loaded
        # outputs are allclose and that the loaded program has no constants.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Wraps a 4x4 linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the linear layer.
                return self.linear(x)

        m = M()
        sample_inputs = (torch.randn(1, 4),)
        eager_out = m(*sample_inputs)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        ep_out = loaded_ep.module()(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))
        self.assertEqual(len(loaded_ep.constants), 0)

    def test_empty_state_dict(self) -> None:
        # [FUNCTION DOC] Exports a model with a plain tensor attribute, round-trips it, and checks
        # outputs match and the loaded state dict is empty.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a stored random constant tensor to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a 4x4 random tensor as a plain attribute.
                super().__init__()
                self.const = torch.randn(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus the
                # stored constant.
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(4, 4),)
        eager_out = m(*sample_inputs)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        ep_out = loaded_ep.module()(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))
        self.assertEqual(len(loaded_ep.state_dict), 0)

    def test_preserve_aliasing(self) -> None:
        # [FUNCTION DOC] Round-trips a model with aliased submodules, buffers and views, and checks
        # outputs match and that shared storage is preserved after load: aliased linear weights and
        # biases share storage, and a constant and its diagonal view share storage in both the
        # program and the unlifted module (writes through one show in the other).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Has two names for one linear layer, persistent and
            # non-persistent buffers, and a constant plus a view of its diagonal.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the aliased layers, buffers and the constants.
                super().__init__()
                self.linear1 = torch.nn.Linear(8, 8)
                self.linear2 = self.linear1  # alias of linear1
                self.register_buffer("buffer1", torch.randn(8, 8))
                self.register_buffer("buffer2", torch.randn(8, 8), persistent=False)
                self.const1 = torch.ones(8, 8)
                self.const2 = self.const1.diagonal()  # a partial view of const1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the sum of both
                # linear applications, both buffers and both constants.
                return (
                    self.linear1(x)
                    + self.linear2(x)
                    + self.buffer1
                    + self.buffer2
                    + self.const1
                    + self.const2
                )

        m = M()
        sample_inputs = (torch.randn(1, 8),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        eager_out = m(*sample_inputs)
        epm = loaded_ep.module()
        ep_out = epm(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))

        # loaded_ep should preserve the aliasing info
        self.assertEqual(
            loaded_ep.state_dict["linear1.weight"].untyped_storage(),
            loaded_ep.state_dict["linear2.weight"].untyped_storage(),
        )
        self.assertEqual(
            loaded_ep.state_dict["linear1.bias"].untyped_storage(),
            loaded_ep.state_dict["linear2.bias"].untyped_storage(),
        )
        self.assertEqual(
            loaded_ep.constants["const1"].untyped_storage(),
            loaded_ep.constants["const2"].untyped_storage(),
        )
        # verify const1 and const2 share the same storage
        loaded_ep.constants["const1"][0][0] = 123
        self.assertEqual(loaded_ep.constants["const2"][0], 123)
        loaded_ep.constants["const2"][-1] = 321
        self.assertEqual(loaded_ep.constants["const1"][-1][-1], 321)

        # unlifted module should also preserve the aliasing info
        epm = loaded_ep.module()
        epm_state_dict = epm.state_dict()
        self.assertEqual(
            epm_state_dict["linear1.weight"].untyped_storage(),
            epm_state_dict["linear2.weight"].untyped_storage(),
        )
        self.assertEqual(
            epm_state_dict["linear1.bias"].untyped_storage(),
            epm_state_dict["linear2.bias"].untyped_storage(),
        )
        self.assertEqual(
            epm.const1.untyped_storage(),
            epm.const2.untyped_storage(),
        )
        # verify const1 and const2 share the same storage
        epm.const1[0][0] = 123
        self.assertEqual(epm.const2[0], 123)
        epm.const2[-1] = 321
        self.assertEqual(epm.const1[-1][-1], 321)

    def test_storage_offset(self) -> None:
        # [FUNCTION DOC] Round-trips a model whose constant is a slice of a larger tensor and
        # checks outputs equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a linear layer output to a constant that is the
            # first four elements of a range.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the constant slice and a 4x4 linear layer.
                super().__init__()
                self.const = torch.arange(8)[:4]
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the linear output
                # plus the constant.
                return self.linear(x) + self.const

        m = M()
        sample_inputs = (torch.randn(1, 4),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_1D_tensor_slicing(self) -> None:
        # [FUNCTION DOC] Round-trips a model whose constant is a strided 1D slice and checks
        # outputs equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a stride-two slice constant to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores every second element of a range of eight
                # as a constant.
                super().__init__()
                self.const = torch.arange(8)[::2]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus
                # the constant.
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(4),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_2D_tensor_slicing(self) -> None:
        # [FUNCTION DOC] Round-trips a model whose constant is a 2x2 corner of a 4x4 tensor and
        # checks outputs equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a 2x2 sliced constant to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the top-left 2x2 slice of a random 4x4 tensor.
                super().__init__()
                self.const = torch.randn(4, 4)[:2, :2]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus
                # the constant.
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_non_float_weight(self) -> None:
        # [FUNCTION DOC] Round-trips a model with a non-trainable int8 parameter and checks
        # outputs equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds an int8 non-grad parameter to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 2x2 int8 parameter of ones with
                # requires_grad off.
                super().__init__()
                self.p = torch.nn.Parameter(
                    torch.ones(2, 2, dtype=torch.int8), requires_grad=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus
                # the parameter.
                return x + self.p

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    @requires_gpu
    def test_weight_sharing_gpu(self) -> None:
        # [FUNCTION DOC] Needs a GPU. Checks that two constants where one is a view of the other
        # share storage in the exported program and after save and load, and outputs match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a GPU constant matrix, a row view of it, and a
            # linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the constant, its first-row view and a 4x4
                # linear layer.
                super().__init__()
                self.c2 = torch.ones(2, 4, device=GPU_TYPE)
                self.c1 = self.c2[0, :]
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns linear output
                # plus both constants.
                return self.linear(x) + self.c1 + self.c2

        m = M().to(GPU_TYPE)
        sample_inputs = (torch.randn(2, 4, device=GPU_TYPE),)
        ep = torch.export.export(m, sample_inputs)
        # Check that c1 and c2 share the same storage
        self.assertEqual(
            ep.constants["c1"].untyped_storage(), ep.constants["c2"].untyped_storage()
        )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        # Check that c1 and c2 share the same storage after serdes
        self.assertEqual(
            loaded_ep.constants["c1"].untyped_storage(),
            loaded_ep.constants["c2"].untyped_storage(),
        )
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_complex_constant(self) -> None:
        # [FUNCTION DOC] Round-trips a model that multiplies by complex constants and checks
        # outputs equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Multiplies sin of the input by complex constants and
            # returns two results.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns sin of x times
                # the complex constant one plus one imaginary, and sin of x times one imaginary.
                s = torch.sin(x)
                y = (1 + 1j) * s
                z = 1j * s
                return y, z

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestDeserialize(TestCase):
    # [CLASS DOC] Tests of exported programs round-tripping through serialize and deserialize,
    # comparing outputs and graph node metadata, covering custom ops, control flow, symbolic
    # values, custom objects and opaque types. Skipped on Windows and when dynamo is unsupported.
    def setUp(self):
        # [FUNCTION DOC] Initializes the base test and registers torchbind test implementations.
        super().setUp()
        init_torchbind_implementations()

    def _check_graph_nodes(self, gm1, gm2, _check_meta=True):
        # [FUNCTION DOC] HELPER: Recursively compares two graph modules: equal node counts and ops,
        # and for call nodes equal argument counts and keyword names, value metadata (same shape
        # and dtype or same string form), stack trace, and nested cond and map subgraphs. Unless
        # disabled, also compares module stack and source function stack metadata of non-
        # placeholder, non-output, non-attribute nodes. Fails the test on any mismatch.
        # TODO: The _check_meta flag bypasses checking for
        # source_fn/nn_module_stack as there is an issue with
        # roundtripping the source_fn value on torch.ops.map nodes
        # original source_fn: <functorch.experimental._map.MapWrapper object at 0x7f80a0549930>
        # deserialized source_fn: 'functorch.experimental._map.map'

        self.assertEqual(len(gm1.graph.nodes), len(gm2.graph.nodes))

        for node1, node2 in zip(gm1.graph.nodes, gm2.graph.nodes):
            self.assertEqual(node1.op, node2.op)
            if node1.op == "call_function":
                # Check "val" metadata
                val1 = node1.meta.get("val", None)
                val2 = node2.meta.get("val", None)
                self.assertEqual(len(node1.args), len(node2.args))
                self.assertEqual(set(node1.kwargs.keys()), set(node2.kwargs.keys()))
                if val1 is None or val2 is None:
                    # Either both are None
                    self.assertEqual(val1, val2)
                elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                    # Or both are fake tensors with the same shape/dtype
                    self.assertEqual(len(val1.shape), len(val2.shape))
                    for s1, s2 in zip(val1.shape, val2.shape):
                        if is_concrete_int(s1) and is_concrete_int(s2):
                            self.assertEqual(s1, s2)
                        else:
                            self.assertEqual(str(s1), str(s2))
                    self.assertEqual(val1.dtype, val2.dtype)
                elif isinstance(val1, (list, tuple)) and isinstance(
                    val2, (list, tuple)
                ):
                    # Or both are fake tensors lists with one element and with the
                    # same shape/dtype
                    for v1, v2 in zip(
                        pytree.tree_leaves(val1), pytree.tree_leaves(val2)
                    ):
                        if isinstance(v1, FakeTensor):
                            self.assertEqual(v1.shape, v2.shape)
                            self.assertEqual(v1.dtype, v2.dtype)
                else:
                    # For expressions like 's0 < 10' can only compare through string
                    self.assertEqual(str(val1), str(val2))

                # Check "stack_trace" metadata
                self.assertEqual(
                    node1.meta.get("stack_trace", None),
                    node2.meta.get("stack_trace", None),
                )

                if node1.target == torch.ops.higher_order.cond:
                    true_graph1 = getattr(gm1, node1.args[1].target)
                    true_graph2 = getattr(gm2, node2.args[1].target)
                    self._check_graph_nodes(true_graph1, true_graph2)

                    false_graph1 = getattr(gm1, node1.args[2].target)
                    false_graph2 = getattr(gm2, node2.args[2].target)
                    self._check_graph_nodes(false_graph1, false_graph2)
                elif node1.target == torch.ops.higher_order.map_impl:
                    map_graph1 = getattr(gm1, node1.args[0].target)
                    map_graph2 = getattr(gm2, node2.args[0].target)
                    self._check_graph_nodes(map_graph1, map_graph2, False)

            if _check_meta and node1.op not in ("get_attr", "placeholder", "output"):
                # Check "nn_module_stack" metadata
                self.assertEqual(
                    node1.meta.get("nn_module_stack", None),
                    node2.meta.get("nn_module_stack", None),
                )
                # Check "source_fn_stack" metadata
                self.assertEqual(
                    node1.meta.get("source_fn_stack", None),
                    node2.meta.get("source_fn_stack", None),
                )

    def check_graph(
        self,
        fn,
        inputs,
        dynamic_shapes=None,
        _check_meta=True,
        use_pre_dispatch=True,
        strict=True,
    ) -> None:
        # [FUNCTION DOC] HELPER: Exports fn with the given inputs, serializes with an opset
        # version, deserializes, drops dead code, and checks outputs are same type and allclose
        # (equal for meta or float8 outputs) and graph nodes match. Runs with pre-dispatch and also
        # without when pre-dispatch is requested; otherwise only without. Arguments: fn module,
        # inputs, optional dynamic shapes, metadata check flag, pre-dispatch flag, strict flag.
        """Export a graph, serialize it, deserialize it, and compare the results."""

        def _deepcopy_inputs(inputs):
            # [FUNCTION DOC] HELPER: Returns a deep copy of the inputs, temporarily clearing
            # instance dictionaries of tensors that have one so the copy does not fail, and
            # restoring them on both original and copy.
            # copy.deepcopy(deepcopy) can fail if tensor inputs have attribute (i.e. __dict__).
            # we remove __dict__ when deepcopying.
            dict_mapping = dict()
            inputs_clone = ()
            for idx, i in enumerate(inputs):
                if isinstance(i, torch.Tensor) and hasattr(inputs[0], "__dict__"):
                    dict_mapping[idx] = i.__dict__
                    i.__dict__ = {}
                inputs_clone += (copy.deepcopy(i),)

            # Add __dict__ back.
            for k, v in dict_mapping.items():
                inputs[k].__dict__ = v
                inputs_clone[k].__dict__ = v
            return inputs_clone

        def _check_graph(pre_dispatch):
            # [FUNCTION DOC] HELPER: One round trip check: exports (public export if pre-dispatch,
            # private export without pre-dispatch otherwise), serializes, deserializes, then
            # compares outputs and graph nodes as described for the enclosing helper.
            if pre_dispatch:
                ep = torch.export.export(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            else:
                # We should have this branch because
                # PT2 Inference goes through this private
                # export API.
                ep = torch.export._trace._export(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    pre_dispatch=False,
                )
            ep.graph.eliminate_dead_code()

            serialized_artifact = serialize(ep, opset_version={"aten": 0})
            deserialized_ep = deserialize(
                serialized_artifact, expected_opset_version={"aten": 0}
            )
            deserialized_ep.graph.eliminate_dead_code()

            orig_outputs = ep.module()(*_deepcopy_inputs(inputs))
            loaded_outputs = deserialized_ep.module()(*_deepcopy_inputs(inputs))

            flat_orig_outputs = pytree.tree_leaves(orig_outputs)
            flat_loaded_outputs = pytree.tree_leaves(loaded_outputs)

            for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
                self.assertEqual(type(orig), type(loaded))
                # torch.allclose doesn't work for float8
                if isinstance(orig, torch.Tensor) and orig.dtype not in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.float8_e8m0fnu,
                ]:
                    if orig.is_meta:
                        self.assertEqual(orig, loaded)
                    else:
                        self.assertTrue(torch.allclose(orig, loaded))
                else:
                    self.assertEqual(orig, loaded)
            self._check_graph_nodes(
                ep.graph_module, deserialized_ep.graph_module, _check_meta
            )

        if use_pre_dispatch:
            _check_graph(pre_dispatch=True)
            _check_graph(pre_dispatch=False)
        else:
            _check_graph(pre_dispatch=False)

    def test_optional_tuple(self):
        # [FUNCTION DOC] Defines a custom op taking an optional tensor and returning a tensor and
        # optional tensor, and round-trips a module calling it.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b, Tensor? c) -> (Tensor, Tensor?)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo")
            def foo_impl(a, b, c):
                # [FUNCTION DOC] HELPER: Implementation and fake kernel: returns a plus b and, when
                # c is given, c plus a plus b, else None.
                res2 = None
                if c is not None:
                    res2 = c + a + b
                return a + b, res2

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Calls the custom op with its three inputs.
                def forward(self, a, b, c):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the result of
                    # the custom op.
                    return torch.ops.mylib.foo(a, b, c)

            self.check_graph(M(), (torch.randn(3), torch.randn(3), torch.randn(3)))

    def test_list_of_int_and_float_lists_custom_op(self):
        # [FUNCTION DOC] Defines a custom op taking lists of int lists and float lists, round-trips
        # modules calling it with non-empty and empty groups, and checks the serialized arguments
        # have the int-lists and float-lists argument types with the exact values.
        # Example program: a custom op that takes List[List[int]] and
        # List[List[float]] arguments. Round-tripping it (export -> serialize ->
        # deserialize) exercises the serde nested-list path end to end: these
        # args must serialize as as_int_lists / as_float_lists, for both
        # non-empty and empty lists.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::apply_groups",
                "(Tensor x, int[][] int_groups, float[][] float_groups) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::apply_groups", "cpu", lib=lib)
            @torch.library.register_fake("mylib::apply_groups")
            def apply_groups_impl(x, int_groups, float_groups):
                # [FUNCTION DOC] HELPER: Implementation and fake kernel: returns x plus the sum of
                # all numbers in both group lists.
                bias = sum(v for group in int_groups for v in group) + sum(
                    v for group in float_groups for v in group
                )
                return x + bias

            class NonEmpty(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Calls the op with fixed non-empty int and
                # float groups.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the op
                    # applied with nested groups of ints and floats.
                    return torch.ops.mylib.apply_groups(
                        x, [[1, 2], [3, 4, 5]], [[1.5], [2.5, 3.5]]
                    )

            class Empty(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Calls the op with empty groups.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the op
                    # applied with two empty lists.
                    return torch.ops.mylib.apply_groups(x, [], [])

            # The round-trips must succeed; before nested-list support was added,
            # serialization raised SerializeError on List[List[int/float]] args.
            self.check_graph(NonEmpty(), (torch.randn(3),))
            self.check_graph(Empty(), (torch.randn(3),))

            # Pin down the exact serialized argument kinds and values.
            def _serialized_op_args(mod):
                # [FUNCTION DOC] HELPER: Exports the module, serializes it, finds the single
                # apply_groups node (asserting there is exactly one) and returns a mapping of input
                # name to serialized argument.
                ep = torch.export.export(mod, (torch.randn(3),), strict=True)
                serialized = ExportedProgramSerializer().serialize(ep)
                nodes = [
                    n
                    for n in serialized.exported_program.graph_module.graph.nodes
                    if n.target and "apply_groups" in n.target
                ]
                self.assertEqual(len(nodes), 1)
                return {i.name: i.arg for i in nodes[0].inputs}

            args = _serialized_op_args(NonEmpty())
            self.assertEqual(args["int_groups"].type, "as_int_lists")
            self.assertEqual(args["int_groups"].as_int_lists, [[1, 2], [3, 4, 5]])
            self.assertEqual(args["float_groups"].type, "as_float_lists")
            self.assertEqual(args["float_groups"].as_float_lists, [[1.5], [2.5, 3.5]])

            empty_args = _serialized_op_args(Empty())
            self.assertEqual(empty_args["int_groups"].type, "as_int_lists")
            self.assertEqual(empty_args["int_groups"].as_int_lists, [])
            self.assertEqual(empty_args["float_groups"].type, "as_float_lists")
            self.assertEqual(empty_args["float_groups"].as_float_lists, [])

    def test_unbacked_bindings_serialize(self):
        # [FUNCTION DOC] Exports a module with unbacked ints in both strict and non-strict
        # mode, round-trips it, and checks unbacked bindings metadata is present on the same
        # nodes and that every bound symbol is an unbacked int with an index below the shape
        # environment's counter.
        from torch._export.utils import _get_shape_env_from_gm
        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls item on both inputs, doubles one and adds,
            # then adds two.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Mutates x by adding 2,
                # converts x and y to Python numbers and returns the combined number.
                x += 2
                n = x.item()
                n = n * 2 + y.item()
                return n + 2

        inps = (
            torch.tensor(4),
            torch.tensor(5),
        )
        for _strict in [True, False]:
            ep = torch.export.export(M(), inps, strict=_strict).run_decompositions()

            # check bindings after deserialization
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            loaded_ep = load(buffer)
            bound = set()
            for old_node, new_node in zip(ep.graph.nodes, loaded_ep.graph.nodes):
                self.assertEqual(
                    "unbacked_bindings" in old_node.meta,
                    "unbacked_bindings" in new_node.meta,
                )
                bound.update(new_node.meta.get("unbacked_bindings", {}))

            # check ShapeEnv counters
            shape_env = _get_shape_env_from_gm(loaded_ep.graph_module)
            next_index = shape_env.unbacked_symint_counter
            shape_env.unbacked_symint_counter += 1
            for symbol in bound:
                self.assertTrue(symbol_is_type(symbol, SymT.UNBACKED_INT))
                self.assertTrue(
                    int(str(symbol)[len(prefix_str[SymT.UNBACKED_INT]) :]) < next_index
                )

    def test_sym_bool_dynamic_shapes(self) -> None:
        # [FUNCTION DOC] Round-trips a model with dynamic shapes involving shifts and negation in a
        # slice, with the sequence dimension bounded to 4.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Slices x using a computed size derived from
            # y's shape.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Slices the second
                # dimension of x by a negative amount derived from half of y's size and returns
                # the slice.
                z = x[:, -((y.shape[0] >> 1) << 1) :, :]
                return z

        inputs = (torch.ones(4, 5, 10), torch.ones(3))
        dynamic_shapes = {"x": {}, "y": {0: Dim("seqlen", max=4)}}
        # Compile with dynamic_shapes set to get operator.neg/shifts involved
        self.check_graph(MyModule(), inputs, dynamic_shapes=dynamic_shapes)

    def test_auto_functionalize(self):
        # [FUNCTION DOC] Defines three mutating custom ops (one tensor return, two returns, no
        # return) and round-trips a module calling all three without pre-dispatch since auto
        # functionalize is not supported there.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo2",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo3",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo1")
            def foo1_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: Implementation and fake kernel: adds to x and z in place
                # and returns n doubled by addition.
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return n + n

            @torch.library.impl("mylib::foo2", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo2")
            def foo2_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: Implementation and fake kernel: adds to x and z in place
                # and returns the pair n plus n and n times n.
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return (n + n, n * n)

            @torch.library.impl("mylib::foo3", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo3")
            def foo3_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: Implementation and fake kernel: adds to x and z in place
                # and returns nothing.
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Calls the three mutating ops in sequence.
                def forward(self, x, y, z, n):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Calls the first op,
                    # then the third op, and returns the output of the second op.
                    n = torch.ops.mylib.foo1(x, y, z, 2, n)
                    torch.ops.mylib.foo3(x, y, z, 2, n)
                    return torch.ops.mylib.foo2(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # TODO Auto_functionalize is not supported on pre_dispatch IR
            self.check_graph(M(), orig_args, use_pre_dispatch=False)

    def test_hoo_symint_input(self):
        # [FUNCTION DOC] Round-trips a module whose conditional branches capture a number from
        # item, without pre-dispatch.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Runs a conditional on a constant true predicate with
            # branches that add or subtract a captured number.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Gets a number from c and
                # returns the conditional of adding or subtracting b and that number from a.
                num = c.item()
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda a, b: a + b + num,
                    false_fn=lambda a, b: a - b - num,
                    operands=(a, b),
                )

        inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(2))
        self.check_graph(Mod(), inp, use_pre_dispatch=False)

    def test_none_input(self):
        # [FUNCTION DOC] Tests loading old-style programs lacking an input spec name: replaces the
        # second input spec by a none argument, deserializes, unflattens, and checks outputs equal
        # the original module's.
        """
        Testing a backwards-compatibility breakage where old models do not have
        an input spec with the node name.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns x plus z; y is unused.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus z.
                return x + z

        ep = torch.export.export(M(), (torch.ones(3, 3), None, torch.ones(3, 3)))

        serialized_program = ExportedProgramSerializer(None, 2).serialize(ep)
        serialized_program.exported_program.graph_module.signature.input_specs[1] = (
            schema.InputSpec.create(
                user_input=schema.UserInputSpec(
                    arg=schema.Argument.create(as_none=True)
                )
            )
        )
        ep = ExportedProgramDeserializer(None).deserialize(
            serialized_program.exported_program, {}, {}, {}
        )
        ep.graph_module.recompile()
        unflattened = torch.export.unflatten(ep)
        inp = (torch.rand(3, 3), None, torch.rand(3, 3))
        self.assertEqual(unflattened(*inp), M()(*inp))

    def test_multi_return(self) -> None:
        # [FUNCTION DOC] Round-trips a layer norm model (multiple outputs from one node).
        """
        Test multiple return from a single node (ex. layer_norm has 2 outputs)
        """

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Applies functional layer norm with weight and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x, w, b):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns layer norm of x
                # with weight w, bias b and epsilon 1e-5.
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        inputs = (
            torch.ones([512, 512], requires_grad=True),
            torch.ones([512]),
            torch.ones([512]),
        )
        self.check_graph(MyModule(), inputs)

    def test_basic(self) -> None:
        # [FUNCTION DOC] Round-trips a simple arithmetic model returning two tensors.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds, multiplies and divides the input by itself and
            # returns the result and a clone.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Adds x to itself, squares
                # it, divides it by itself, and returns that result and a clone of it.
                x = x + x
                x = x * x
                x = x / x
                return x, x.clone()

        inputs = (torch.ones([512], requires_grad=True),)
        self.check_graph(MyModule(), inputs)

    def test_dynamic(self) -> None:
        # [FUNCTION DOC] Round-trips the dynamic shape model with a shared dynamic first dimension.
        class DynamicShapeSimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Matmul plus bias, halved, flattened and concatenated
            # with itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes the halved
                # sum of the matmul of a and b with c and returns its flat view concatenated
                # with itself.
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dynamic_shapes = {"a": {0: dim0_ac}, "b": None, "c": {0: dim0_ac}}
        self.check_graph(DynamicShapeSimpleModel(), inputs, dynamic_shapes)

    def test_sym_bool(self):
        # [FUNCTION DOC] Round-trips a model asserting x's size is contained in y, which creates a
        # symbolic bool.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Asserts a size membership then adds the inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Asserts x's first size is
                # in y then returns x plus y.
                assert x.size(0) in y  # noqa: S101
                return x + y

        f = Module()
        self.check_graph(f, (torch.ones(1), torch.ones(3)))

    def test_sym_bool_torch_check_equal(self):
        # [FUNCTION DOC] Round-trips a model that checks the size of a nonzero result equals 2.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the nonzero indices after constraining their
            # count to 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes nonzero of x,
                # constrains its size to equal 2 and returns it.
                y = x.nonzero()
                z = y.size(0)
                torch._check(z == 2)
                return y

        self.check_graph(Module(), (torch.Tensor([1, 0, 1, 0]),))

    def test_sym_int_torch_check_equal(self):
        # [FUNCTION DOC] Round-trips a model that constrains a symbolic size to be divisible by 3
        # and equal to 3.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns nonzero indices after two size constraints.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Computes nonzero of x,
                # requires size divisible by 3 and equal to 3, and returns it.
                y = x.nonzero()
                z = y.size(0)
                torch._check(z % 3 == 0)
                torch._check(z == 3)
                return y

        self.check_graph(Module(), (torch.Tensor([1, 0, 1, 0, 1, 0]),))

    def test_shape(self):
        # [FUNCTION DOC] Round-trips a model using dynamic sizes of both dimensions.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Unpacks the size of x and combines the sizes with
            # the first row.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the first size
                # plus the second size plus the first row of x, and also the first size.
                z, y = x.size()
                return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        self.check_graph(Foo(), inputs, dynamic_shapes)

    def test_module(self):
        # [FUNCTION DOC] Round-trips a small two-linear network with activations.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Two linear layers with relu; the first
            # applied twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the first linear
                # twice, relu, then the second linear.
                x = self.linear1(x)
                x = self.linear1(x)
                x = torch.nn.functional.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        self.check_graph(M(), inputs)

    def test_module_meta(self):
        # [FUNCTION DOC] Round-trips a module built on the meta device with meta inputs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a 3x3 parameter to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the parameter
                # plus x.
                return self.p + x

        with torch.device("meta"):
            mod = M()

        inputs = (torch.randn(3, 3, device="meta"),)
        self.check_graph(mod, inputs)

    def test_pytree_namedtuple(self):
        # [FUNCTION DOC] Registers a namedtuple and a NamedTuple class with pytree under serialized
        # names, exports a module taking and returning them, serializes, and checks two namedtuple
        # field entries on the serialized graph, that the deserialized program, its unlifted module
        # and the unflattened module all keep the field mapping in their metadata. Clears the
        # program's example inputs since the classes are local.
        N1 = namedtuple("N1", ["a", "b"])

        class N2(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with two tensor fields a and b.
            a: torch.Tensor
            b: torch.Tensor

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Combines two namedtuple inputs and returns the other
            # namedtuple type.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns a namedtuple of
                # x.a plus y.a and x.b times y.b.
                return N2(x.a + y.a, x.b * y.b)

        pytree._register_namedtuple(
            N1,
            serialized_type_name="test.export.test_serialize.test_pytree_namedtuple.N1",
        )
        pytree._register_namedtuple(
            N2,
            serialized_type_name="test.export.test_serialize.test_pytree_namedtuple.N2",
        )

        inp = (N1(torch.randn(3), torch.randn(3)), N1(torch.randn(3), torch.randn(3)))
        ep = torch.export.export(M(), inp)
        ep.example_inputs = None  # Can't pickle the input since the namedtuple class is not at a global namespace
        serialized = ExportedProgramSerializer().serialize(ep)
        self.assertEqual(
            len(serialized.exported_program.graph_module.treespec_namedtuple_fields), 2
        )
        deserialized = ExportedProgramDeserializer().deserialize(
            serialized.exported_program,
            serialized.state_dict,
            serialized.constants,
        )
        self.assertTrue("treespec_namedtuple_fields" in deserialized.graph_module.meta)
        self.assertEqual(
            deserialized.graph_module.meta["treespec_namedtuple_fields"],
            {
                "test.export.test_serialize.test_pytree_namedtuple.N1": ["a", "b"],
                "test.export.test_serialize.test_pytree_namedtuple.N2": ["a", "b"],
            },
        )

        unlifted = deserialized.module()
        self.assertTrue("treespec_namedtuple_fields" in unlifted.meta)
        self.assertEqual(len(unlifted.meta["treespec_namedtuple_fields"]), 2)

        unflattened = unflatten(deserialized)
        self.assertTrue("treespec_namedtuple_fields" in unflattened.meta)
        self.assertEqual(len(unflattened.meta["treespec_namedtuple_fields"]), 2)

    def test_cond(self):
        # [FUNCTION DOC] Round-trips a program with a conditional on a value from x.
        from functorch.experimental.control_flow import cond

        inputs = torch.ones(4, 3), torch.zeros(4, 3)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Runs a conditional that adds or subtracts its
            # two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Defines both branch
                # functions and returns the conditional of x's first element greater than 4.
                def t(x, y):
                    # [FUNCTION DOC] HELPER: True branch: returns the sum of its inputs.
                    return x + y

                def f(x, y):
                    # [FUNCTION DOC] HELPER: False branch: returns the difference of its inputs.
                    return x - y

                return cond(x[0][0] > 4, t, f, [x, y])

        self.check_graph(M(), inputs)

    def test_sym_float(self):
        # [FUNCTION DOC] Round-trips a model multiplying a number from item by 0.1.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Scales a scalar read from the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the item of x
                # times 0.1.
                b = x.item()
                return b * 0.1

        self.check_graph(M(), (torch.tensor(1.0),))

    def test_arg_from(self):
        # [FUNCTION DOC] Round-trips a no-input model that re-initializes buffers in place,
        # under no-grad.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Two buffers re-initialized in the forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers a 10x10 weight buffer and a bias buffer of ones.
                super().__init__()
                self.register_buffer("compress_weight", torch.ones((10, 10)))
                self.register_buffer("compress_bias", torch.ones(10))

            def forward(self) -> None:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns early if a buffer
                # is None; else Kaiming-initializes the weight, computes fan in and uniformly
                # initializes the bias in range plus or minus its inverse square root.
                if self.compress_weight is None or self.compress_bias is None:
                    return
                torch.nn.init.kaiming_uniform_(self.compress_weight, a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.compress_weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.compress_bias, -bound, bound)

        with torch.no_grad():
            self.check_graph(M(), ())

    def test_map(self):
        # [FUNCTION DOC] Round-trips a map higher-order op over the first dimension of a tensor,
        # skipping source metadata checks.
        from functorch.experimental import control_flow

        def f(x, y):
            # [FUNCTION DOC] HELPER: Mapped function: returns the sum of its two arguments.
            return x + y

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Applies the map op with the adder over a
            # stacked input.
            def forward(self, xs, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns map of the add
                # function over xs with extra argument y.
                return control_flow.map(f, xs, y)

        g = Module()
        inputs = (torch.ones(3, 2, 2), torch.ones(2))
        self.check_graph(g, inputs, _check_meta=False)

    def test_positional_argument_with_default_value(self):
        # [FUNCTION DOC] Checks that a default-valued bias argument of a linear op stays a
        # positional argument across the round trip.
        class MyLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Calls the aten linear op with fixed random weight
            # and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.weight = torch.randn(10, 10)
                self.bias = torch.randn(10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the linear op of
                # x with the stored weight and bias.
                # bias has an default value here but it should be preserved
                # as a positional argument.
                return torch.ops.aten.linear.default(x, self.weight, self.bias)

        self.check_graph(MyLinear(), (torch.randn(10, 10),))

    def test_tensor_tensor_list(self):
        # [FUNCTION DOC] Defines a custom op returning a tensor and a tensor list with CPU and Meta
        # kernels and round-trips a module that uses both outputs.
        with torch.library._scoped_library("_export", "FRAGMENT") as lib:
            lib.define(
                "_test_tensor_tensor_list_output(Tensor x, Tensor y) -> (Tensor, Tensor[])",
                tags=torch.Tag.pt2_compliant_tag,
            )

            def _test_tensor_tensor_list_output(x, y):
                # [FUNCTION DOC] HELPER: Kernel: returns y and a list holding x.
                return y, [x]

            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "CPU",
            )
            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "Meta",
            )

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Calls the custom op and adds its tensor to the
                # first listed tensor.
                def forward(self, x, y):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns a plus the
                    # first element of b.
                    a, b = torch.ops._export._test_tensor_tensor_list_output.default(
                        x, y
                    )
                    return a + b[0]

            self.check_graph(M(), (torch.rand(3, 2), torch.rand(3, 2)))

    def test_list_of_optional_tensors(self) -> None:
        # [FUNCTION DOC] Round-trips an index op with a list containing None entries and a tensor.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Indexes the sum of two inputs with a list of two
            # None and a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the index of x
                # plus y by indices and adds z.
                indices = [None, None, torch.tensor([1, 3, 5, 7])]
                indexed = torch.ops.aten.index.Tensor(x + y, indices)
                return indexed + z

        inputs = (torch.rand(8, 8, 8), torch.rand(8, 8, 8), torch.rand(8, 8, 4))
        self.check_graph(MyModule(), inputs)

    def test_sym_ite(self):
        # [FUNCTION DOC] Round-trips a symbolic if-else on dynamic dimensions.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Selects one of two sizes depending on whether the
            # first size equals 5.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the sym_ite of
                # the first-size test between the two sizes.
                b = x.shape[0] == 5
                ret = torch.sym_ite(b, x.shape[0], x.shape[1])
                return ret

        dynamic_shapes = {"x": {0: Dim("dim0"), 1: Dim("dim1")}}
        self.check_graph(Foo(), (torch.ones(4, 5),), dynamic_shapes=dynamic_shapes)

    def test_multiple_getitem(self):
        # [FUNCTION DOC] Inserts a duplicate getitem node on a top-k multiplication, round-trips
        # with serialize and deserialize, checks outputs equal and that the deserialized code has
        # deduplicated getitem calls matching an expected string.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the doubled top two values and
            # their indices.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Takes top 2 of x, doubles
                # the values and returns values and indices.
                a, b = torch.topk(x, 2)
                a = a * 2
                return a, b

        ep = torch.export.export(M(), (torch.ones(3),), strict=True)

        # insert another getitem node
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor:
                getitem_0 = node.args[0]
                with ep.graph.inserting_before(getitem_0):
                    getitem_copy = ep.graph.node_copy(getitem_0)
                    mul_node = ep.graph.call_function(
                        torch.ops.aten.mul.Tensor, (getitem_copy, 2)
                    )
                    mul_node.meta = copy.copy(getitem_copy.meta)
                    node.args = (getitem_0, mul_node)

        deserialized_ep = deserialize(serialize(ep))

        inp = (torch.randn(3),)
        orig_res = ep.module()(*inp)
        res = deserialized_ep.module()(*inp)
        self.assertTrue(torch.allclose(orig_res[0], res[0]))
        self.assertTrue(torch.allclose(orig_res[1], res[1]))

        # The deserialized graph should have deduped getitem calls
        self.assertExpectedInline(
            deserialized_ep.graph_module.code.strip("\n"),
            """\
def forward(self, x):
    topk = torch.ops.aten.topk.default(x, 2);  x = None
    getitem = topk[0]
    getitem_1 = topk[1];  topk = None
    mul_tensor = torch.ops.aten.mul.Tensor(getitem, 2)
    mul = torch.ops.aten.mul.Tensor(getitem, mul_tensor);  getitem = mul_tensor = None
    return (mul, getitem_1)
    """,
        )

    @parametrize(
        "name,case",
        get_filtered_export_db_tests(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        # [FUNCTION DOC] Parametrized per supported export-db case: with the new experimental
        # tracer enabled, round-trips the case's model with its example args, skipping metadata
        # checks for map cases.
        model = case.model
        _check_meta = "map" not in name
        with torch._export.config.patch(use_new_tracer_experimental=True):
            self.check_graph(model, case.example_args, _check_meta=_check_meta)

    def test_constraints(self):
        # [FUNCTION DOC] Round-trips a model with a data-dependent size and a check on it.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Creates a ones matrix sized by a number read from x.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Reads a number from x,
                # requires it nonnegative and returns the sum of y plus the sum of a ones tensor
                # with that many rows.
                n = x.item()
                torch._check(n >= 0)
                return y.sum() + torch.ones(n, 5).sum()

        f = Module()
        self.check_graph(f, (torch.tensor(3), torch.randn(4, 5)))

    def test_get_attr(self) -> None:
        # [FUNCTION DOC] Round-trips a model adding a scalar tensor constant, so a constant
        # attribute is serialized.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a constant scalar tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus the
                # tensor 3.
                return x + torch.tensor(3)

        f = Module()
        self.check_graph(f, (torch.tensor(3),))

    def test_get_attr_list(self) -> None:
        # [FUNCTION DOC] Round-trips a model concatenating with a constant tensor made in forward.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Concatenates the input with a constant tensor of
            # two ones.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the concatenation
                # of x and a two-element constant.
                return torch.cat([x, torch.tensor([1, 1])])

        f = Module()
        self.check_graph(f, (torch.tensor([1, 1]),))

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_device(self) -> None:
        # [FUNCTION DOC] Needs CUDA. Round-trips a convolution, relu and scale model on the GPU.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Conv layer then relu then multiply by 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3-to-16 channel convolution and a relu.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns half the relu of
                # the convolution of x.
                conv = self.conv(x)
                relu = self.relu(conv)
                mul = relu * 0.5
                return mul

        inp = torch.randn((1, 3, 224, 224), dtype=torch.float).to("cuda")
        model = MyModule().eval().cuda()
        self.check_graph(model, (inp,))

    def test_custom_obj_tuple_out(self):
        # [FUNCTION DOC] Round-trips a model calling torchbind ops on a custom object attribute,
        # including one returning a tuple, in non-strict mode.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a custom Foo object and calls its ops.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Calls the tuple-return
                # op, sums the two results, passes through the takes_foo op and adds x.
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_custom_obj(self):
        # [FUNCTION DOC] Round-trips a model calling the takes_foo torchbind op twice on a custom
        # object, non-strict.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a custom Foo object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Calls takes_foo on x then
                # on that result, and adds x.
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_custom_obj_list_out(self):
        # [FUNCTION DOC] Round-trips a model whose torchbind op returns a list, non-strict.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a custom Foo object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Calls the list-return op,
                # sums the three results, applies takes_foo and adds x.
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_export_no_inputs(self):
        # [FUNCTION DOC] Serializes and deserializes a program with no inputs and example inputs
        # cleared, comparing outputs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Squares a stored constant tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.p = torch.ones(3, 3)

            def forward(self):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the constant
                # times itself.
                return self.p * self.p

        ep = torch.export.export(M(), (), strict=True)
        ep._example_inputs = None
        roundtrip_ep = deserialize(serialize(ep))
        self.assertTrue(torch.allclose(ep.module()(), roundtrip_ep.module()()))

    def test_serialize_dtypes(self):
        # [FUNCTION DOC] For each of several unusual dtypes (float8 variants, uint32, uint64),
        # round-trips a module casting its input to that dtype, in non-strict mode.
        for dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            torch.uint32,
            torch.uint64,
        ]:

            class MyModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Casts the input to the loop's dtype.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x converted
                    # to the current dtype.
                    return x.to(dtype)

            m = MyModule()
            inputs = (torch.ones(2, 3),)
            self.check_graph(m, inputs, strict=False)

    def test_forward_compatibility(self):
        # [FUNCTION DOC] Checks forward compatibility: converting a dictionary with an unknown
        # extra field to the tensor argument dataclass ignores the extra field and equals the
        # plain object.
        self.assertEqual(
            schema.TensorArgument(
                name="x",
            ),
            _dict_to_dataclass(
                schema.TensorArgument,
                {
                    "shiny_new_field": "hello world",
                    "name": "x",
                },
            ),
        )

    def test_opaque_value_type_constant(self):
        # [FUNCTION DOC] Defines a custom op with an opaque value-type constant argument, with fake
        # kernel, and round-trips a module holding that config object in non-strict mode.
        opaque_type_name = get_opaque_type_name(_OpaqueConfig)
        with torch.library._scoped_library("test_opaque_val", "FRAGMENT") as lib:
            lib.define(
                f"scale(Tensor x, {opaque_type_name} config) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
            )

            @torch.library.impl(lib, "scale", "CompositeExplicitAutograd")
            def scale_impl(x, config):
                # [FUNCTION DOC] HELPER: Implementation: returns x times the config scale.
                return x * config.scale

            @torch.library.register_fake("test_opaque_val::scale")
            def scale_fake(x, config):
                # [FUNCTION DOC] HELPER: Fake kernel: returns an empty tensor like x.
                return torch.empty_like(x)

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Stores a config object and passes it to the op.
                def __init__(self, config):
                    # [FUNCTION DOC] HELPER: Constructor that only runs the base module
                    # initializer; the module has no state.
                    super().__init__()
                    self.config = config

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the scale op
                    # applied to x and the stored config.
                    return torch.ops.test_opaque_val.scale(x, self.config)

            self.check_graph(M(_OpaqueConfig(scale=3)), (torch.randn(4),), strict=False)

    def test_opaque_reference_type_constant(self):
        # [FUNCTION DOC] Defines a custom op with an opaque reference-type argument, with fake
        # kernel, and round-trips a module holding that engine in non-strict mode.
        opaque_type_name = get_opaque_type_name(_OpaqueEngine)
        with torch.library._scoped_library("test_opaque_ref", "FRAGMENT") as lib:
            lib.define(
                f"transform(Tensor x, {opaque_type_name} engine) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
            )

            @torch.library.impl(lib, "transform", "CompositeExplicitAutograd")
            def transform_impl(x, engine):
                # [FUNCTION DOC] HELPER: Implementation: returns x times the engine's multiplier.
                return x * engine.multiplier

            @torch.library.register_fake("test_opaque_ref::transform")
            def transform_fake(x, engine):
                # [FUNCTION DOC] HELPER: Fake kernel: returns an empty tensor like x.
                return torch.empty_like(x)

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Test module. Stores an engine object and passes it to the op.
                def __init__(self, engine):
                    # [FUNCTION DOC] HELPER: Constructor that only runs the base module
                    # initializer; the module has no state.
                    super().__init__()
                    self.engine = engine

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the transform
                    # op applied to x and the stored engine.
                    return torch.ops.test_opaque_ref.transform(x, self.engine)

            self.check_graph(
                M(_OpaqueEngine(multiplier=5)), (torch.randn(4),), strict=False
            )


instantiate_parametrized_tests(TestDeserialize)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSchemaVersioning(TestCase):
    # [CLASS DOC] Tests that deserialization rejects a mismatched schema version. Skipped when
    # dynamo is unsupported.
    def test_error(self):
        # [FUNCTION DOC] Serializes a program, sets the major schema version to minus one, and
        # checks deserialization raises SerializeError saying the version does not match.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds the input to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        f = Module()
        ep = export(f, (torch.randn(1, 3),), strict=True)

        serialized_program = ExportedProgramSerializer().serialize(ep)
        serialized_program.exported_program.schema_version.major = -1
        with self.assertRaisesRegex(
            SerializeError, r"Serialized schema version .* does not match our current"
        ):
            ExportedProgramDeserializer().deserialize(
                serialized_program.exported_program,
                serialized_program.state_dict,
                serialized_program.constants,
                serialized_program.example_inputs,
            )


# We didn't set up kwargs input yet
unittest.expectedFailure(TestDeserialize.test_exportdb_supported_case_fn_with_kwargs)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSaveLoad(TestCase):
    # [CLASS DOC] Tests of saving and loading exported programs to buffers, files and paths,
    # including extra files, version checks, constants, dtypes, gradient flags, large and GPU
    # tensors, and metadata. Skipped when dynamo is unsupported.
    def test_save_buffer(self):
        # [FUNCTION DOC] Saves a program of a linear model to a byte buffer, loads it back and
        # checks allclose outputs.
        inp = (torch.tensor([0.1, 0.1]),)

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds one, transposes, relu, then a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies add one,
                # transpose, relu, then linear and returns the result.
                x = x + 1
                y = x.t()
                y = y.relu()
                y = self.linear(y)
                return y

        ep = export(Module(), inp, strict=True)

        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_deserialize_torch_artifact_dict(self):
        # [FUNCTION DOC] Saves a dictionary with one tensor using plain tensor saving, passes
        # the bytes to the artifact deserializer and checks a dictionary with the same tensor
        # is returned.
        data = {"key": torch.tensor([1, 2, 3])}
        buf = io.BytesIO()
        torch.save(data, buf)
        serialized = buf.getvalue()
        result = deserialize_torch_artifact(serialized)

        self.assertIsInstance(result, dict)
        self.assertTrue(torch.equal(result["key"], torch.tensor([1, 2, 3])))

    @unittest.skipIf(IS_WINDOWS, "Cannot modify file in windows")
    def test_save_file(self):
        # [FUNCTION DOC] Saves a program to a named temporary file and loads it by file name, then
        # compares outputs. Skipped on Windows.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input squared.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x times x.
                return x * x

        f = Foo()

        inp = (torch.randn(2, 2),)
        ep = export(f, inp, strict=True)

        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            save(ep, f.name)
            f.seek(0)
            loaded_ep = load(f.name)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_path(self):
        # [FUNCTION DOC] Saves a program to a path object in a temporary file name and loads it
        # from the path, comparing outputs.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the sum of two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus y.
                return x + y

        f = Foo()

        inp = (torch.tensor([6]), torch.tensor([7]))
        ep = export(f, inp, strict=True)

        with TemporaryFileName(suffix=".pt2") as fname:
            path = Path(fname)
            save(ep, path)
            loaded_ep = load(path)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_extra(self):
        # [FUNCTION DOC] Saves with an extra file and loads with a dictionary asking for it; checks
        # outputs equal and the dictionary now holds the saved content moo.
        inp = (torch.tensor([0.1, 0.1]),)

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns x squared plus x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x times x plus x.
                return x * x + x

        f = Foo()

        ep = export(f, inp, strict=True)

        buffer = io.BytesIO()
        save(ep, buffer, extra_files={"extra.txt": "moo"})
        buffer.seek(0)
        extra_files = {"extra.txt": ""}
        loaded_ep = load(buffer, extra_files=extra_files)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))
        self.assertEqual(extra_files["extra.txt"], "moo")

    @unittest.skipIf(
        IS_FBCODE or IS_MACOS or IS_WINDOWS, "The file path is different in fbcode CI"
    )
    def test_version_error(self):
        # [FUNCTION DOC] Saves a program, copies the archive replacing its archive version entry by
        # minus one, and checks loading raises ValueError saying the version does not match.
        # Skipped on fbcode, macOS and Windows since it relies on the temp file path layout.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input doubled by addition.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        f = Foo()

        ep = export(f, (torch.randn(1, 3),), strict=True)

        with self.assertRaisesRegex(
            ValueError, r"Saved archive version -1 does not match our current"
        ):
            with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
                save(ep, f.name)
                f.seek(0)
                file_prefix = f.name.split("/")[2].split(".")[0]

                # Create a new file and copy things over, but modify the
                # archive version
                with tempfile.NamedTemporaryFile(suffix=".pt2") as fnew:
                    with zipfile.ZipFile(f, "r") as zin:
                        with zipfile.ZipFile(fnew, "w") as zout:
                            for item in zin.infolist():
                                if (
                                    item.filename
                                    != f"{file_prefix}/{ARCHIVE_VERSION_PATH}"
                                ):
                                    zout.writestr(item, zin.read(item.filename))
                            zout.writestr(f"{file_prefix}/{ARCHIVE_VERSION_PATH}", "-1")

                    f.seek(0)
                    load(fnew.name)

    def test_save_constants(self):
        # [FUNCTION DOC] Round-trips a model with a constant attribute and constants created in
        # forward, comparing outputs.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds a stored scalar and two scalar tensors made
            # in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor that only runs the base module initializer;
                # the module has no state.
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus the stored
                # tensor and two new scalar tensors 3 and 4.
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),), strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        inp = (torch.tensor(1),)
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_load_with_multiple_empty_tensors(self) -> None:
        # [FUNCTION DOC] Saves a model with three zero-length buffers, two of the same dtype
        # (duplicates) and one different, and checks the archive bytes contain two weight entries
        # (weight 0 and 1) but not a third, and that loaded dtypes are correct.
        # Test scenario where models have multiple empty tensors
        # but with different data types.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Registers three empty buffers: two uint8 and
            # one float32.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the three zero-element buffers.
                super().__init__()
                self.register_buffer(
                    "int_buffer",
                    torch.zeros([0], dtype=torch.uint8),
                )
                self.register_buffer(
                    "int_buffer2",
                    torch.zeros([0], dtype=torch.uint8),
                )
                self.register_buffer(
                    "float_buffer",
                    torch.zeros([0], dtype=torch.float32),
                )

            def forward(self, t: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns t plus
                # the buffers.
                return t + self.int_buffer + self.float_buffer + self.int_buffer2

        m = M()
        inp = torch.rand([0])

        ep = torch.export.export(m, (inp,))

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        model_bytes = buffer.getvalue()

        # First two buffers are duplicates, but not the third one.
        # So in the serialized model, there will be two physical tensors.
        self.assertTrue(b"weight_0" in model_bytes)
        self.assertTrue(b"weight_1" in model_bytes)
        self.assertFalse(b"weight_2" in model_bytes)

        buffer = io.BytesIO(model_bytes)
        buffer.seek(0)
        dep = torch.export.load(buffer)
        unf = torch.export.unflatten(dep)
        self.assertEqual(unf.int_buffer.dtype, torch.uint8)
        self.assertEqual(unf.int_buffer2.dtype, torch.uint8)
        self.assertEqual(unf.float_buffer.dtype, torch.float32)

    def test_save_load_multiple_dtypes(self) -> None:
        # [FUNCTION DOC] Round-trips a model with float32 and float16 parameters and int8 and
        # float64 buffers, checking outputs and that the dtypes are preserved after unflatten.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds mixed-dtype parameters and buffers.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates parameters and buffers of several dtypes.
                super().__init__()
                self.p_f32 = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
                self.p_f16 = torch.nn.Parameter(
                    torch.randn(4, 4, dtype=torch.float16), requires_grad=False
                )
                self.register_buffer("b_i8", torch.ones(4, 4, dtype=torch.int8))
                self.register_buffer("b_f64", torch.randn(4, 4, dtype=torch.float64))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus all state
                # converted to float.
                return (
                    x
                    + self.p_f32
                    + self.p_f16.float()
                    + self.b_i8.float()
                    + self.b_f64.float()
                )

        m = M()
        inp = (torch.randn(4, 4),)
        ep = torch.export.export(m, inp)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*inp), loaded_ep.module()(*inp))

        unf = unflatten(loaded_ep)
        self.assertEqual(unf.p_f32.dtype, torch.float32)
        self.assertEqual(unf.p_f16.dtype, torch.float16)
        self.assertEqual(unf.b_i8.dtype, torch.int8)
        self.assertEqual(unf.b_f64.dtype, torch.float64)

    def test_save_load_requires_grad_preserved(self) -> None:
        # [FUNCTION DOC] Round-trips a model and checks the loaded state dict keeps requires_grad:
        # true for the trainable parameter and false for the frozen parameter and buffer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Has a trainable parameter, a frozen parameter and
            # a buffer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the three 8x8 state tensors.
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(8, 8))
                self.frozen = torch.nn.Parameter(torch.randn(8, 8), requires_grad=False)
                self.register_buffer("buf", torch.randn(8, 8))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the matrix
                # product of x and w plus the frozen parameter plus the buffer.
                return x @ self.w + self.frozen + self.buf

        m = M()
        inp = (torch.randn(1, 8),)
        ep = torch.export.export(m, inp)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*inp), loaded_ep.module()(*inp))

        loaded_sd = loaded_ep.state_dict
        self.assertTrue(loaded_sd["w"].requires_grad)
        self.assertFalse(loaded_sd["frozen"].requires_grad)
        self.assertFalse(loaded_sd["buf"].requires_grad)

    def test_save_load_large_tensor(self) -> None:
        # [FUNCTION DOC] Round-trips a model with a 1024 by 1024 linear layer and compares outputs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Wraps a large linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(1024, 1024)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the linear layer.
                return self.linear(x)

        m = M()
        inp = (torch.randn(1, 1024),)
        ep = torch.export.export(m, inp)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*inp), loaded_ep.module()(*inp))

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_save_load_cuda_tensor(self) -> None:
        # [FUNCTION DOC] Needs CUDA. Round-trips a GPU linear model and checks all loaded state
        # dict entries are on cuda and outputs match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Wraps a 64 by 64 linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the linear layer.
                return self.linear(x)

        m = M().cuda()
        inp = (torch.randn(1, 64, device="cuda"),)
        ep = torch.export.export(m, inp)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        loaded_sd = loaded_ep.state_dict
        for name, param in loaded_sd.items():
            self.assertEqual(
                param.device.type, "cuda", lambda msg: f"{msg}\n{name} not on cuda"
            )
        self.assertEqual(m(*inp), loaded_ep.module()(*inp))

    def test_from_node_metadata_serialization(self):
        # [FUNCTION DOC] Re-exports an exported convolution module to get nested from_node
        # metadata, round-trips, and checks node count, names, ops, targets and per-node
        # from_node metadata (compared via dictionary form) are preserved for non-placeholder,
        # non-output nodes.
        """Test that from_node metadata is properly serialized and deserialized."""

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Wraps a convolution layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a one-to-64 channel convolution.
                super().__init__()
                self.conv_layer = torch.nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=3, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the convolution.
                return self.conv_layer(x)

        m = Model()
        inputs = (torch.randn(1, 1, 32, 32),)

        ep = export(m, inputs, strict=True)
        reexported_ep = export(
            ep.module(), inputs, strict=True
        )  # for generating more complex from_node metadata
        serialized = serialize(reexported_ep)
        reexported_ep_loaded = deserialize(serialized)

        # Verify that node names and from_node metadata are preserved after serialization/deserialization
        self.assertEqual(
            len(list(reexported_ep.graph_module.graph.nodes)),
            len(list(reexported_ep_loaded.graph_module.graph.nodes)),
        )
        for node, node_loaded in zip(
            reexported_ep.graph_module.graph.nodes,
            reexported_ep_loaded.graph_module.graph.nodes,
        ):
            # Verify node name consistency
            self.assertEqual(node.name, node_loaded.name)
            self.assertEqual(node.op, node_loaded.op)
            self.assertEqual(node.target, node_loaded.target)

            if node.op not in {"placeholder", "output"}:
                from_node_orig = node.meta.get("from_node")
                from_node_loaded = node_loaded.meta.get("from_node")

                if from_node_orig is None:
                    self.assertIsNone(from_node_loaded)
                else:
                    self.assertIsNotNone(from_node_loaded)
                    self.assertEqual(len(from_node_orig), len(from_node_loaded))
                    for node_source_orig, node_source_loaded in zip(
                        from_node_orig, from_node_loaded
                    ):
                        self.assertEqual(
                            node_source_orig.to_dict(), node_source_loaded.to_dict()
                        )


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerializeCustomClass(TestCase):
    # [CLASS DOC] Tests serialization of programs with torchbind script objects as constants.
    # Skipped when dynamo is unsupported.
    def setUp(self):
        # [FUNCTION DOC] Initializes the base test and registers torchbind test implementations.
        super().setUp()
        init_torchbind_implementations()

    def test_custom_class(self):
        # [FUNCTION DOC] Inserts a node calling an op on a pickle-testing custom object,
        # serializes, checks the output text mentions class_fqn and the object's qualified name
        # (raising AssertionError otherwise), deserializes, and checks the node argument is a
        # script object with the same type and state and top value 7.
        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input doubled.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs, strict=True)

        # Replace one of the values with an instance of our custom class
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (custom_obj,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    custom_node.meta["torch_fn"] = (
                        "take_an_instance",
                        "take_an_instance",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        serialized_vals = serialize(ep)

        ep_str = serialized_vals.exported_program.decode("utf-8")
        if "class_fqn" not in ep_str:
            raise AssertionError("'class_fqn' not found in serialized output")
        if custom_obj._type().qualified_name() not in ep_str:
            raise AssertionError(
                f"'{custom_obj._type().qualified_name()}' not found in serialized output"
            )

        deserialized_ep = deserialize(serialized_vals)

        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(isinstance(arg, torch._C.ScriptObject))
                self.assertEqual(arg._type(), custom_obj._type())
                self.assertEqual(arg.__getstate__(), custom_obj.__getstate__())
                self.assertEqual(arg.top(), 7)

    def test_custom_class_containing_fake_tensor(self):
        # [FUNCTION DOC] Builds a module under fake mode holding a custom object containing a fake
        # tensor, exports non-strict with torchbind tracing, round-trips, and checks the loaded
        # constant's content is a fake tensor.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a custom object that contains a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the contained-tensor custom object from
                # random data.
                super().__init__()
                self.custom_obj = torch.classes._TorchScriptTesting._ContainsTensor(
                    torch.rand(2, 3)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus the tensor
                # from the object.
                return x + self.custom_obj.get()

        with FakeTensorMode():
            f = Foo()

        inputs = (torch.zeros(2, 3),)
        with enable_torchbind_tracing():
            ep = export(f, inputs, strict=False)

        serialized_vals = serialize(ep)
        ep = deserialize(serialized_vals)
        self.assertTrue(isinstance(ep.constants["custom_obj"].get(), FakeTensor))

    def test_custom_class_input_to_function(self):
        # [FUNCTION DOC] Builds a module with a torchbind attribute under fake mode, exports non-
        # strict, round-trips, and checks the program code and the unlifted module code match
        # expected strings, with the attribute as a script object in both.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Holds a Foo custom object passed to takes_foo.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the custom object.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus the
                # takes_foo op on the object and x.
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        with FakeTensorMode():
            f = Foo()

        inputs = (torch.zeros(2, 3),)
        with enable_torchbind_tracing():
            ep = export(f, inputs, strict=False)

        serialized_vals = serialize(ep)
        ep = deserialize(serialized_vals)
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, obj_attr, x):
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(obj_attr, x);  obj_attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo);  x = takes_foo = None
    return (add,)""",
        )
        self.assertTrue(isinstance(ep.constants["attr"], torch.ScriptObject))
        gm = ep.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo);  x = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertTrue(isinstance(gm.attr, torch.ScriptObject))

    def test_custom_tag_metadata_serialization(self):
        # [FUNCTION DOC] Attaches custom metadata to the graph module and to the add node,
        # serializes, deserializes, and checks both survive and the add node is seen exactly once.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input doubled.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_ep = ep._update(new_gm, ep.graph_signature)
        serialized_vals = serialize(new_ep)
        new_ep = deserialize(serialized_vals)

        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertEqual(counter, 1)

    def test_custom_tag_metadata_decomp(self):
        # [FUNCTION DOC] Attaches custom metadata to a linear node and the graph, runs
        # decompositions, and checks all decomposed call nodes carry the tag and graph metadata is
        # retained, with more than one node.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Wraps a 2 by 2 linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Applies the linear layer.
                return self.linear(x)

        f = Foo()

        inputs = (torch.ones(2, 2),)
        ep = export(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        counter = 0
        for node in new_gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                counter += 1
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"
        self.assertEqual(counter, 1)

        new_ep = ep._update(new_gm, ep.graph_signature)
        new_ep = new_ep.run_decompositions()

        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_ep.graph.nodes:
            if node.op == "call_function":
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertTrue(counter > 1)

    def test_custom_tag_metadata_copy(self):
        # [FUNCTION DOC] Attaches custom metadata to a graph module and its add node, deep copies
        # the module, and checks the metadata is retained exactly once.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Returns the input doubled.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus x.
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_gm = copy.deepcopy(new_gm)

        self.assertEqual(new_gm.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertEqual(counter, 1)

    def test_unbacked_range_serdes(self):
        # [FUNCTION DOC] Round-trips a program with unbacked integer constraints and checks the
        # loaded shape environment's variable ranges equal the original's.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Reads a number from x, bounds it by y's size, and
            # returns an empty tensor and an element of y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Constrains the item to be
                # nonnegative and below y's first size and returns an empty tensor of that size and
                # the indexed element.
                n = x.item()
                torch._check(n >= 0)
                torch._check(n < y.size(0))
                return torch.empty(n), y[n]

        ep = torch.export.export(
            Foo(),
            (torch.tensor([5]), torch.randn(10)),
            dynamic_shapes={
                "x": None,
                "y": (Dim.DYNAMIC,),
            },
        )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        # pre-serialize ep
        pre_shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in ep.graph.nodes]
        ).shape_env
        post_shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in loaded_ep.graph.nodes]
        ).shape_env
        self.assertEqual(pre_shape_env.var_to_range, post_shape_env.var_to_range)

    def test_backed_size_oblivious_serdes(self):
        # [FUNCTION DOC] Exports with backed size oblivious mode, round-trips, and checks the
        # loaded shape environment gives the first symbol a lower bound of 0.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Adds two dynamic inputs and an item from a third.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns x plus y plus the
                # item of z.
                return x + y + z.item()

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                Foo(),
                (torch.randn(1), torch.randn(1), torch.tensor([5])),
                dynamic_shapes={
                    "x": (Dim.DYNAMIC,),
                    "y": (Dim.DYNAMIC,),
                    "z": None,
                },
            )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in loaded_ep.graph.nodes]
        ).shape_env
        s0 = next(iter(ep.graph.nodes)).meta["val"].size(0)
        self.assertEqual(shape_env.var_to_range[s0.node.expr].lower, 0)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestPredispatchSerialization(TestCase):
    # [CLASS DOC] Tests that predispatch wrapper functions of function transforms survive
    # serialization. Skipped when dynamo is unsupported.
    def test_predispatch_jvp_serialize_roundtrip(self):
        # [FUNCTION DOC] Exports a forward-mode derivative model non-strict, checks the increment
        # and decrement nesting calls appear before and after round-trip, that deserialized targets
        # are the real functions, and outputs match.
        """Test that JVP predispatch wrapper functions survive serialization round-trip."""
        from torch._functorch.predispatch import (
            _jvp_decrement_nesting,
            _jvp_increment_nesting,
        )

        class JVP(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Computes a forward-mode derivative of a helper.
            def foo(self, x, r, t) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Method used as the differentiated function: returns
                # x minus 0.1 times r plus 0.1 times t.
                return x - 0.1 * r + 0.1 * t

            def forward(self, x, y, r, t, z, o) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: forward pass of the test module. Returns the jvp of the
                # helper at primals x, r, t with tangents y, z, o.
                return torch.func.jvp(
                    self.foo,
                    (x, r, t),
                    (y, z, o),
                )

        inp = (
            torch.rand(2, 4),
            torch.rand(2, 4),
            torch.rand(2, 1),
            torch.rand(2, 1),
            torch.zeros(2, 1),
            torch.ones(2, 1),
        )

        ep = export(JVP(), inp, strict=False)

        graph_str = str(ep.graph)
        self.assertIn("_jvp_increment_nesting", graph_str)
        self.assertIn("_jvp_decrement_nesting", graph_str)

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)

        loaded_graph_str = str(loaded_ep.graph)
        self.assertIn("_jvp_increment_nesting", loaded_graph_str)
        self.assertIn("_jvp_decrement_nesting", loaded_graph_str)

        # Verify the deserialized targets are the actual functions
        for node in loaded_ep.graph.nodes:
            if node.op == "call_function":
                if "increment" in node.name:
                    self.assertIs(node.target, _jvp_increment_nesting)
                elif "decrement" in node.name:
                    self.assertIs(node.target, _jvp_decrement_nesting)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertTrue(torch.allclose(exp_out[0], actual_out[0]))
        self.assertTrue(torch.allclose(exp_out[1], actual_out[1]))

    def test_predispatch_vmap_serialize_roundtrip(self):
        # [FUNCTION DOC] Exports a vmap model with pre-dispatch, checks vmap increment and
        # decrement nesting calls appear before and after round-trip, and outputs equal.
        """Test that vmap predispatch wrapper functions survive serialization round-trip."""
        from torch.export._trace import _export

        class Vmap(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module. Applies vmap of a small function.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: forward pass of the test module. Maps a function that
                # multiplies x and y, adds one and sums over the first dimension, then sums the
                # mapped result over dimension zero.
                f = lambda x, y: (x * y + 1).sum(dim=0)  # noqa: E731
                return torch.vmap(f)(x, y).sum(dim=0)

        inp = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.2, 0.3]))

        ep = _export(Vmap(), inp, pre_dispatch=True)

        # Verify predispatch vmap nodes are in the graph
        graph_str = str(ep.graph)
        self.assertIn("_vmap_increment_nesting", graph_str)
        self.assertIn("_vmap_decrement_nesting", graph_str)

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)

        # Verify predispatch nodes survive round-trip
        loaded_graph_str = str(loaded_ep.graph)
        self.assertIn("_vmap_increment_nesting", loaded_graph_str)
        self.assertIn("_vmap_decrement_nesting", loaded_graph_str)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertTrue(torch.allclose(exp_out, actual_out))


if __name__ == "__main__":
    run_tests()
