# Owner(s): ["oncall: export"]
# flake8: noqa
# [MODULE DOC] Tests for swapping preserved submodules of an exported program with eager modules
# via the private module-swapping utility, plus pytree handling of custom dataclass inputs and
# outputs in swapped graphs. Import time: only imports and the TestSwap class definition, which is
# expanded into strict and non-strict variants by parameterized_class.
import unittest
from dataclasses import dataclass
from typing import Any, List

from parameterized import parameterized_class

import torch
import torch._dynamo as torchdynamo
from torch import Tensor
from torch._export import config
from torch._export.utils import register_dataclass_as_pytree_node
from torch.export import export, register_dataclass
from torch.export._swap import _swap_modules
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
@parameterized_class(
    [
        {"strict": False},
        {"strict": True},
    ],
    class_name_func=lambda cls,
    _,
    params: f"{cls.__name__}_{'strict' if params['strict'] else 'nonstrict'}",
)
class TestSwap(TestCase):
    # [CLASS DOC] Test case run twice (class attribute strict False and True, class names suffixed
    # with nonstrict and strict); skipped on Windows or when dynamo is unsupported. Each test
    # exports a small module with self.strict, swaps modules and compares numerics or
    # generated code.
    def test_unflatten_preserve_signature(self):
        # [FUNCTION DOC] Exports a two-child model with the call signatures of a nested grandchild
        # and a sibling preserved, swaps both with fresh eager instances, and checks the swapped
        # module output is numerically close to the exported module output.
        class NestedChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Innermost module; takes a tuple and a mapping with one key.
            def forward(self, zx, y):
                # [FUNCTION DOC] HELPER: Returns a mapping with the sum and the product of the
                # mapped tensor and the tuple's second element.
                return {"x": y["key"] + zx[1], "w": y["key"] * zx[1]}

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a nested child, called with a tuple and a mapping.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested child module.
                super().__init__()
                self.nested = NestedChild()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds a tensor of ones, calls the nested child with it
                # and the first input, and combines the child's results with it by add and
                # subtract.
                z = torch.ones_like(x)
                xw = self.nested((z, x), y={"key": y})
                return xw["w"] + z - xw["x"]

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module that subtracts one from its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input minus one.
                return x - 1

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root model chaining the first child then the second child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two child modules as attributes.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Feeds the inputs through the first child, then the second
                # child, and returns the result.
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo.nested", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo.nested": NestedChild(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_unflatten_preserve_with_unused_input(self):
        # [FUNCTION DOC] Exports a model whose preserved child receives an input computed from a
        # top-k result (including an input the child does not really use), swaps in a fresh child,
        # and checks the outputs match numerically.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Child taking three inputs and returning the sum of the first two
            # and the third unchanged.
            def forward(self, x, a, b):
                # [FUNCTION DOC] HELPER: Returns a tuple of the sum of the first two inputs and the
                # third input.
                return x + a, b

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Root model that computes top-two values and indices of its second
            # input and passes them to the child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child module.
                super().__init__()
                self.m1 = M1()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs the top-two selection on the second input, calls the
                # child and returns the first element of its result.
                a, b = torch.topk(y, 2)
                return self.m1(x, a, b)[0]

        ep = torch.export.export(
            M(),
            (torch.randn(2), torch.randn(5)),
            preserve_module_call_signature=("m1",),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"m1": M1()},
        )

        inps = (torch.randn(2), torch.randn(5))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_nested_leaf(self):
        # [FUNCTION DOC] Exports a three-level model preserving only the middle module call
        # signature, swaps in a fresh middle module (which contains a leaf), and checks numerics
        # against the exported module.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module adding one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        class Nested(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module wrapping a leaf.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf module.
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf's output plus two.
                return self.leaf(x) + 2

        class TopLevel(torch.nn.Module):
            # [CLASS DOC] HELPER: Top module wrapping the middle module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the middle module.
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the middle module's output plus three.
                return self.nested(x) + 3

        ep = torch.export.export(
            TopLevel(),
            (torch.randn(3),),
            strict=self.strict,
            preserve_module_call_signature=("nested",),
        )

        swapped_gm = _swap_modules(
            ep,
            {"nested": Nested()},
        )

        inps = (torch.randn(3),)
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_dedup_sym_size(self):
        # [FUNCTION DOC] Exports a model with a dynamic dimension (twice a symbolic dimension)
        # where a size-derived value is used by the top level and two children, preserving only one
        # child's signature. Swaps that child and checks the outputs match for sizes ten and
        # twenty, ensuring the shared symbolic size computation is handled in the swapped graph.
        # Here, sym_size & floor div are used in 3 subgraphs (top-level, m1, m2),
        # but only one copy of sym_size is created in the initial export graph.
        # For m1, sym_size & floordiv should be copied as recompute since we preserve the call signature,
        # but for m2 floordiv should be passed in as a placeholder.
        # Test that this is preserved, and the unflattened module runs correctly.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Child that slices its second input to half the first input's
            # leading size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the leading half-length slice of the second input.
                d = x.size(0) // 2
                return y[:d]

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Identical to the other child; not preserved or swapped.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the leading half-length slice of the second input.
                d = x.size(0) // 2
                return y[:d]

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Root model with two children that also uses the half size itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children.
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls both children and returns the tail slice of the
                # second input plus both child results.
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
            strict=self.strict,
            preserve_module_call_signature=("m1",),
        )

        swapped_gm = _swap_modules(
            ep,
            {"m1": M1()},
        )

        inps = (torch.randn(10), torch.randn(10))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

        inps = (torch.randn(20), torch.randn(20))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_remove_duplicate_pytree_simple(self):
        # [FUNCTION DOC] Exports a two-child model preserving both signatures, swaps both with
        # fresh instances, checks numerics, and compares the generated code of the swapped module
        # to an expected string showing the first child's dict output flowing directly to the
        # second child with redundant pytree flatten and unflatten steps removed.
        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Child returning a mapping of two tensors.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds ones like the first input, computes some unused
                # intermediates, and returns a mapping with the sum and the product of the inputs.
                z = torch.ones_like(x)
                w = y + z[1]
                x = y * z[1]
                return {"res1": x + y, "res2": x * y}

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Child consuming the mapping produced by the first child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product entry plus the sum entry minus one.
                return x["res2"] + x["res1"] - 1

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root model piping the first child's output to the second child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs the first child, then the second on its output.
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo": Child1(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))
        self.assertExpectedInline(
            swapped_gm.code.strip(),
            """\
def forward(self, x, y):
    x_1 = x
    y_1 = y
    _spec_0 = self._spec_0
    _spec_1 = self._spec_1
    _spec_4 = self._spec_4
    tree_flatten = torch.utils._pytree.tree_flatten((x_1, y_1));  x_1 = y_1 = None
    getitem = tree_flatten[0];  tree_flatten = None
    x = getitem[0]
    y = getitem[1];  getitem = None
    tree_unflatten_1 = torch.utils._pytree.tree_unflatten([x, y], _spec_1);  x = y = _spec_1 = None
    getitem_1 = tree_unflatten_1[0];  tree_unflatten_1 = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1];  getitem_1 = None
    foo = self.foo(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    bar = self.bar(foo);  foo = None
    tree_flatten_spec_1 = torch.fx._pytree.tree_flatten_spec(bar, _spec_4);  bar = _spec_4 = None
    getitem_10 = tree_flatten_spec_1[0];  tree_flatten_spec_1 = None
    tree_unflatten = torch.utils._pytree.tree_unflatten((getitem_10,), _spec_0);  getitem_10 = _spec_0 = None
    return tree_unflatten""",
        )

    @unittest.expectedFailure
    def test_remove_duplicate_pytree_different_order(self):
        # [FUNCTION DOC] Expected-failure test: a first child returns two mappings and the second
        # child consumes them in swapped order, which the duplicate-pytree removal does not
        # support. Body exports, swaps, checks numerics and compares generated code.
        """
        This is not supported yet because module `foo`s outputs are not all
        directly used in as inputs to `bar` in the same order as outputted from
        `foo`. To support this, we would have to do some sort of ordering.
        """

        class Child1(torch.nn.Module):
            # [CLASS DOC] HELPER: Child returning two mappings in a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns one mapping holding a sum and another holding a
                # product and a square.
                return {"res1": x + y}, {"res2": x * y, "res3": x * x}

        class Child2(torch.nn.Module):
            # [CLASS DOC] HELPER: Child taking the two mappings in swapped positional order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, y, x):
                # [FUNCTION DOC] HELPER: Combines entries of the two mappings and returns the
                # product term minus the sum term.
                y = y["res2"] * y["res3"]
                x = x["res1"] + x["res1"]
                return y - x

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Root model passing the first child's outputs to the second in
            # reversed order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two children.
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Unpacks the first child's two outputs and passes them to
                # the second child in reversed order.
                x, y = self.foo(x, y)
                x = self.bar(y, x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo": Child1(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))
        self.assertExpectedInline(
            swapped_gm.code.strip(),
            """\
def forward(self, x, y):
    x, y, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    _spec_0 = self._spec_0
    _spec_3 = self._spec_3
    tree_unflatten = torch.utils._pytree.tree_unflatten([x, y], _spec_0);  x = y = _spec_0 = None
    getitem = tree_unflatten[0];  tree_unflatten = None
    getitem_1 = getitem[0]
    getitem_2 = getitem[1];  getitem = None
    foo = self.foo(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    getitem_3 = foo[0]
    getitem_4 = foo[1];
    bar = self.bar(getitem_4, getitem_3);  foo = None
    tree_flatten_spec_1 = torch.fx._pytree.tree_flatten_spec(bar, _spec_3);  bar = _spec_3 = None
    getitem_9 = tree_flatten_spec_1[0];  tree_flatten_spec_1 = None
    return pytree.tree_unflatten((getitem_9,), self._out_spec)""",
        )

    def test_custom_input_args(self):
        # [FUNCTION DOC] Registers a dataclass of two tensors as a pytree node with the private
        # helper, exports a model that multiplies them as a matrix product (strict per
        # self.strict), swaps with an empty mapping, and checks running the result through the
        # fx interpreter and directly gives close results.
        @dataclass
        class CustomInput:
            # [CLASS DOC] HELPER: Dataclass holding two tensor fields, used as a custom input type.
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomInput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model taking the dataclass and returning the matrix product of
            # its fields.
            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the two fields.
                return torch.matmul(inputs.a, inputs.b)

        ep = export(
            Foo(),
            (CustomInput(torch.randn(2, 3), torch.randn(3, 2)),),
            strict=self.strict,
        )
        swapped = _swap_modules(ep, {})
        inp = (CustomInput(torch.randn(2, 3), torch.randn(3, 2)),)
        res1 = torch.fx.Interpreter(swapped).run(*inp)
        res2 = swapped(*inp)
        self.assertTrue(torch.allclose(res1, res2))

    def test_custom_input_kwargs(self):
        # [FUNCTION DOC] Registers the dataclass with the public register function, exports a model
        # taking a tensor and a keyword-only dataclass, in a loop over the new-tracer flag with the
        # config patched, then swaps with an empty mapping and compares interpreter and direct
        # runs. Only the program from the last loop iteration is used for the checks.
        @dataclass
        class CustomInput:
            # [CLASS DOC] HELPER: Dataclass holding two tensor fields, passed as a keyword input.
            a: Tensor
            b: Tensor

        register_dataclass(
            CustomInput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the matrix product of the keyword dataclass's
            # fields to a tensor input.
            def forward(self, x, *, inputs):
                # [FUNCTION DOC] HELPER: Returns the first input plus the matrix product of the
                # two fields of the keyword-only argument.
                return x + torch.matmul(inputs.a, inputs.b)

        for use_new_tracer in [True, False]:
            with config.patch(use_new_tracer_experimental=use_new_tracer):
                ep = export(
                    Foo(),
                    (torch.randn(2, 2),),
                    {"inputs": CustomInput(torch.randn(2, 3), torch.randn(3, 2))},
                    strict=self.strict,
                )
        swapped = _swap_modules(ep, {})
        inp_args = (torch.randn(2, 2),)
        inp_kwargs = {"inputs": CustomInput(torch.randn(2, 3), torch.randn(3, 2))}
        res1 = torch.fx.Interpreter(swapped).run(*(*inp_args, *inp_kwargs.values()))
        res2 = swapped(*inp_args, **inp_kwargs)
        self.assertTrue(torch.allclose(res1, res2))

    def test_custom_input_kwargs_use_private(self):
        # [FUNCTION DOC] Registers the dataclass through the private helper and exports a model
        # taking a keyword-only dataclass with the experimental new tracer enabled; passes if no
        # exception is raised and makes no further assertions.
        @dataclass
        class CustomInput:
            # [CLASS DOC] HELPER: Dataclass holding two tensor fields, passed as a keyword input.
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomInput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the matrix product of the keyword dataclass's
            # fields to a tensor input.
            def forward(self, x, *, inputs):
                # [FUNCTION DOC] HELPER: Returns the first input plus the matrix product of the
                # two fields of the keyword-only argument.
                return x + torch.matmul(inputs.a, inputs.b)

        # shouldn't error
        with config.patch(use_new_tracer_experimental=True):
            _ = export(
                Foo(),
                (torch.randn(2, 2),),
                {"inputs": CustomInput(torch.randn(2, 3), torch.randn(3, 2))},
                strict=self.strict,
            )

    def test_custom_output(self):
        # [FUNCTION DOC] Registers an output dataclass as a pytree node, always exports with strict
        # mode on (ignoring self.strict), swaps with an empty mapping, and checks both fields of
        # both returned dataclass instances match between interpreter and direct execution.
        @dataclass
        class CustomOutput:
            # [CLASS DOC] HELPER: Dataclass holding two tensor fields, returned from the model.
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomOutput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning two output dataclass instances built from
            # products and sums of its inputs and a transpose.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a tuple of two output dataclass instances: squares
                # of the inputs, and the first input times the second's transpose with their sum.
                return (CustomOutput(a * a, b * b), CustomOutput(a * b.T, a + b.T))

        ep = export(Foo(), (torch.randn(2, 3), torch.randn(3, 2)), strict=True)
        swapped = _swap_modules(ep, {})
        inp = (torch.randn(2, 3), torch.randn(3, 2))
        res1 = torch.fx.Interpreter(swapped).run(*inp)
        res2 = swapped(*inp)
        self.assertTrue(torch.allclose(res1[0].a, res2[0].a))
        self.assertTrue(torch.allclose(res1[0].b, res2[0].b))
        self.assertTrue(torch.allclose(res1[1].a, res2[1].a))
        self.assertTrue(torch.allclose(res1[1].b, res2[1].b))


if __name__ == "__main__":
    run_tests()
