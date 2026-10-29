# Owner(s): ["module: dynamo", "module: higher order operators"]
# [MODULE DOC] Tests for the flat_apply higher-order operator and for Dynamo's handling of
# functions marked with nonstrict_trace. Defines sample dataclasses (point, input, output
# variants, some registered as pytree dataclasses and some deliberately not), a constant-like
# custom class Norm, and a distance function. Import-time side effects: registers Norm as an
# opaque custom class, and registers Point, InputData and OutputData as pytree dataclasses
# globally. Includes graph-text helpers print_graph and remove_file_comment.
import re
from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    extract_graph,
    normalize_gm,
    remove_trailing_space,
)
from torch._higher_order_ops.flat_apply import (
    flat_apply,
    func_to_graphable,
    is_graphable,
    to_graphable,
)
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def distance(a, b, norm):
    # [FUNCTION DOC] HELPER: Computes the Euclidean (norm type l2) or Manhattan (norm type l1)
    # distance between two points given as objects with tensor fields x and y. Returns a tensor.
    # For any other norm type it falls through and returns None.
    if norm.typ == "l2":
        return torch.sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    elif norm.typ == "l1":
        return (a.x - b.x).abs() + (a.y - b.y).abs()


@dataclass(frozen=True)
class Norm(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Frozen dataclass with one string field typ, used as an opaque
    # constant-like argument. Registered below as a custom class with constant semantics.
    typ: str

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Returns a pair of the source expression text that rebuilds this
        # Norm (including its typ value) and a mapping from the name Norm to the class, for FX
        # code printing.
        return f"Norm(typ={self.typ!r})", {"Norm": Norm}


torch._library.opaque_object.register_custom_class(Norm, typ="constant")


@dataclass
class Point:
    # [CLASS DOC] HELPER: Dataclass holding two tensors x and y. Registered as a pytree
    # dataclass so it flattens into its tensor fields.
    x: Tensor
    y: Tensor


pytree.register_dataclass(Point)


@dataclass
class InputData:
    # [CLASS DOC] HELPER: Dataclass with an integer count and a tensor values, registered as a
    # pytree dataclass; a valid input type for nonstrict_trace functions.
    count: int
    values: Tensor


pytree.register_dataclass(InputData)


@dataclass
class InputInvalid:
    # [CLASS DOC] HELPER: Dataclass with the same fields as InputData but intentionally never
    # registered with pytree, used to check the invalid-input error.
    count: int
    values: Tensor


# No pytree for InputInvalid


@dataclass
class OutputData:
    # [CLASS DOC] HELPER: Dataclass with two tensor fields result1 and result2, registered as a
    # pytree dataclass; a valid output type for nonstrict_trace functions.
    result1: Tensor
    result2: Tensor


pytree.register_dataclass(OutputData)


@dataclass
class OutputInvalid:
    # [CLASS DOC] HELPER: Dataclass with the same fields as OutputData but intentionally never
    # registered with pytree, used to check the invalid-output error.
    result1: Tensor
    result2: Tensor


# No pytree for OutputInvalid


class FlatApplyTests(PytreeRegisteringTestCase):
    # [CLASS DOC] TestCase (with automatic pytree registration cleanup) for the flat_apply
    # operator directly and for the graph Dynamo emits for nonstrict_trace functions. Stateless
    # itself.
    def test_simple(self):
        # [FUNCTION DOC] Builds two Point objects and an l2 Norm, converts the distance function
        # and the arguments to graphable form, checks the function yields no extra leaves and
        # that every flattened leaf is graphable, then asserts flat_apply gives the same value
        # as calling the function directly.
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))
        norm = Norm("l2")

        args = (a, b)
        kwargs = {"norm": norm}

        empty_list, func_spec = func_to_graphable(distance)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, distance(*args, **kwargs))

    def test_non_tensor_output(self):
        # [FUNCTION DOC] Same flat_apply round trip as the simple test but with a function
        # returning a nested structure mixing tensors, a float, and a string. Asserts the
        # flat_apply result equals the direct result, showing non-tensor outputs are preserved.
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))

        args = (a, b)
        kwargs = {}

        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns a list holding a tensor, a tuple with a tensor, a
            # list of a tensor and a float, and a string, and a final tensor; built from the
            # fields of the two points.
            return [a.x + 1, (b.x + 2, [a.y + 3, 4.0], "5"), 6 + b.y]

        empty_list, func_spec = func_to_graphable(f)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, f(*args, **kwargs))

    def test_nonstrict_trace_dynamo_graph(self):
        # [FUNCTION DOC] Defines local Point and PointTensor classes (shadowing the module-level
        # Point), registers both as pytree nodes through the helper that undoes registration at
        # teardown, and compiles a function calling a nonstrict_trace function with a
        # PointTensor. Asserts the recorded Dynamo graph text has the flat apply capture call
        # with the callable and input spec and returns the single tensor result.
        class Point:
            # [CLASS DOC] HELPER: Plain class with tensor attributes x and y, local to this
            # test.
            x: Tensor
            y: Tensor

            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores x and y as attributes.
                self.x = x
                self.y = y

        class PointTensor:
            # [CLASS DOC] HELPER: Plain class pairing a Point with a tensor t, local to this
            # test.
            p: Point
            t: Tensor

            def __init__(self, p, t):
                # [FUNCTION DOC] HELPER: Stores p and t as attributes.
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        def trace_point(p):
            # [FUNCTION DOC] HELPER: Graph-breaks, then returns the product of the point's x and
            # y.
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Graph-breaks, then returns the
            # tensor field plus the result of the point helper on the nested point.
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds a Point from x and y, a tensor sum t, wraps them in
            # a PointTensor, and returns the traced function's result.
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10]", L_y_: "f32[10]"):
        l_x_ = L_x_
        l_y_ = L_y_

        add: "f32[10]" = l_x_ + l_y_

        trace_point_tensor_callable : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.trace_point_tensor_callable
        trace_point_tensor_input_spec : torch.utils._pytree.TreeSpec = self.trace_point_tensor_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(trace_point_tensor_callable, trace_point_tensor_input_spec, l_x_, l_y_, add);  trace_point_tensor_callable = trace_point_tensor_input_spec = l_x_ = l_y_ = add = None
        getitem: "f32[10]" = flat_apply_capture[0];  flat_apply_capture = None
        return (getitem,)
""",
        )

    def test_nonstrict_trace_captured_tensor_post_aot_graph(self):
        # [FUNCTION DOC] Compiles with an AOT recording backend a function calling a
        # nonstrict_trace function that uses a captured global tensor. Asserts the forward graph
        # text multiplies the inputs and adds the captured constant tensor as a graph attribute.
        cst = torch.ones(1)

        @torch._dynamo.nonstrict_trace
        def trace_me(x, y):
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Graph-breaks, then returns x times
            # y plus the captured tensor cst.
            torch._dynamo.graph_break()
            return x * y + cst

        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the traced function on x and y and returns its
            # result.
            return trace_me(x, y)

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        _tensor_constant0: "f32[1]" = self._tensor_constant0
        add: "f32[10]" = torch.ops.aten.add.Tensor(mul, _tensor_constant0);  mul = _tensor_constant0 = None
        return (add,)
""",
        )


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
class TestInputOutput(PytreeRegisteringTestCase):
    # [CLASS DOC] TestCase (skipped under Dynamo-wrapped runs) for argument and result handling
    # of nonstrict_trace functions: valid and invalid dataclass inputs and outputs. Stateless.
    def test_simple(self):
        # [FUNCTION DOC] Compares eager and aot_eager full-graph compiled results for a function
        # calling a nonstrict_trace function with an int and a tensor, then checks the extracted
        # Dynamo graph text matches the expected flat apply capture graph.
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Uses closure variables a and b to
            # compute an output tensor and two derived tensors from the count and values;
            # returns all three.
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sin to the values, calls the traced function with
            # count and values, and returns the sum of all four tensors.
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i.count, i.values)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: y, z_result1, z_result2 = gn(i.count, i.values)
        gn_callable : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.gn_callable
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_callable, gn_input_spec, 5, l_i_values);  gn_callable = gn_input_spec = None
        getitem: "f32[4, 4]" = flat_apply_capture[0]
        getitem_1: "f32[4, 4]" = flat_apply_capture[1]
        getitem_2: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: x = torch.sin(i.values)
        sin: "f32[4, 4]" = torch.sin(l_i_values);  l_i_values = None

        # code: return x + y + z_result1 + z_result2
        add: "f32[4, 4]" = sin + getitem;  sin = getitem = None
        add_1: "f32[4, 4]" = add + getitem_1;  add = getitem_1 = None
        add_2: "f32[4, 4]" = add_1 + getitem_2;  add_1 = getitem_2 = None
        return (add_2,)
""",
        )

    def test_dataclass_input(self):
        # [FUNCTION DOC] Same as the simple test but the traced function takes the InputData
        # dataclass directly; checks aot_eager results equal eager and the extracted graph text
        # matches.
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputData) -> tuple[Tensor, Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Reads count and values from the
            # dataclass and returns three tensors computed with closure variables a and b.
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sin to the values, calls the traced function with
            # the dataclass, and returns the sum of the four tensors.
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: y, z_result1, z_result2 = gn(i)
        gn_callable : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.gn_callable
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_callable, gn_input_spec, 5, l_i_values);  gn_callable = gn_input_spec = None
        getitem: "f32[4, 4]" = flat_apply_capture[0]
        getitem_1: "f32[4, 4]" = flat_apply_capture[1]
        getitem_2: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: x = torch.sin(i.values)
        sin: "f32[4, 4]" = torch.sin(l_i_values);  l_i_values = None

        # code: return x + y + z_result1 + z_result2
        add: "f32[4, 4]" = sin + getitem;  sin = getitem = None
        add_1: "f32[4, 4]" = add + getitem_1;  add = getitem_1 = None
        add_2: "f32[4, 4]" = add_1 + getitem_2;  add_1 = getitem_2 = None
        return (add_2,)
""",
        )

    def test_invalid_input(self):
        # [FUNCTION DOC] Passing an InputInvalid (not registered with pytree) to a
        # nonstrict_trace function under full-graph compilation must raise
        # torch._dynamo.exc.Unsupported whose message mentions an invalid input type.
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputInvalid) -> tuple[Tensor, Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Computes three tensors from the
            # unregistered dataclass's count and values and closure variables a and b.
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputInvalid) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sin to the values, calls the traced function, and
            # sums the four tensors.
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputInvalid(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Invalid input type for nonstrict_trace-ed function",
        ):
            opt_fn(i)

    def test_dataclass_output(self):
        # [FUNCTION DOC] Traced function returns a tensor and an OutputData dataclass; checks
        # aot_eager results equal eager and the extracted graph text matches the expected
        # capture with three outputs.
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputData]:
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Returns an output tensor and an
            # OutputData holding two derived tensors, using closure variables a and b.
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputData(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sin, calls the traced function, and sums sin, the
            # first output, and both dataclass fields.
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(lambda i: fn(i), backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

        _, gms, _, _ = extract_graph(lambda i: fn(i), i)
        self.assertExpectedInline(
            print_graph(gms[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_i_values: "f32[4, 4]"):
        l_i_values = L_i_values

        # code: y, z = gn(i.count, i.values)
        gn_callable : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.gn_callable
        gn_input_spec : torch.utils._pytree.TreeSpec = self.gn_input_spec
        flat_apply_capture = torch__dynamo_variables_torch_flat_apply_capture(gn_callable, gn_input_spec, 5, l_i_values);  gn_callable = gn_input_spec = None
        getitem: "f32[4, 4]" = flat_apply_capture[0]
        getitem_1: "f32[4, 4]" = flat_apply_capture[1]
        getitem_2: "f32[4, 4]" = flat_apply_capture[2];  flat_apply_capture = None

        # code: x = torch.sin(i.values)
        sin: "f32[4, 4]" = torch.sin(l_i_values);  l_i_values = None

        # code: return x + y + z.result1 + z.result2
        add: "f32[4, 4]" = sin + getitem;  sin = getitem = None
        add_1: "f32[4, 4]" = add + getitem_1;  add = getitem_1 = None
        add_2: "f32[4, 4]" = add_1 + getitem_2;  add_1 = getitem_2 = None
        return (add_2,)
""",
        )

    def test_invalid_output(self):
        # [FUNCTION DOC] Traced function returns an OutputInvalid (not registered with pytree);
        # under full-graph compilation this must raise torch._dynamo.exc.Unsupported whose
        # message mentions an unsupported output type.
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputInvalid]:
            # [FUNCTION DOC] HELPER: Marked nonstrict_trace. Returns an output tensor and an
            # unregistered OutputInvalid dataclass of two derived tensors.
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputInvalid(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            # [FUNCTION DOC] HELPER: Applies sin, calls the traced function, and sums the
            # outputs.
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Unsupported output type for nonstrict_trace-ed function",
        ):
            opt_fn(i)


def remove_file_comment(gm_str: str) -> str:
    # [FUNCTION DOC] HELPER: Rewrites the file and code marker comments inside a printed graph
    # so that the file path part is dropped and only the code marker remains, then removes
    # trailing whitespace. Returns the resulting string.
    return remove_trailing_space(re.sub(r"# File.*, code:", "# code:", gm_str))


def print_graph(graph: torch.fx.GraphModule) -> str:
    # [FUNCTION DOC] HELPER: Prints the GraphModule as readable source text (not to stdout) and
    # returns it with file comments removed via remove_file_comment.
    return remove_file_comment(graph.print_readable(print_output=False))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
