# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests dynamo's tracking of attributes set or read on input tensors and on user
# objects. Covers tensor-valued and constant attributes attached to input tensors (whether
# they are lifted to graph inputs), guard behavior when an assigned attribute changes type,
# many attribute reads and writes with and without graph breaks, reconstruction of attributes
# after inlining, and assigning to the data of tensors. No import-time side effects.
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)


class TestInputAttrTracking(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case with one test per attribute scenario. Tests compile small
    # functions with a capturing backend, a compile counter or the eager backend and compare
    # against eager results, graph placeholder counts, frame counts and operation counts.
    def test_tensor_property_on_tensor(self):
        # [FUNCTION DOC] Reads a tensor-valued attribute that was set on an input tensor
        # outside the compiled function; checks the compiled result matches eager and that the
        # captured graph has two placeholders, i.e. the attribute tensor is lifted to a graph
        # input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by its attached attribute.
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            # [FUNCTION DOC] HELPER: Backend that stores the graph module in the enclosing
            # variable and returns it unchanged for execution.
            nonlocal graph
            graph = gm
            return gm

        fn = torch.compile(fn, backend=grab_graph_backend, fullgraph=True)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this lifts y to inputs!
        self.assertEqual(placeholder_cnt, 2)

    def test_tensor_property_assigned_on_tensor(self):
        # [FUNCTION DOC] Assigns a tensor argument as an attribute on another input tensor
        # inside the compiled function and multiplies them; checks results match eager and the
        # graph has two placeholders (the attribute value is already an input).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the attribute on the first argument to the second
            # argument and returns their product.
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = torch.randn([2, 2])

        eager_result = fn(x_, y_)

        graph = None

        def grab_graph_backend(gm, inps):
            # [FUNCTION DOC] HELPER: Backend that stores the graph module in the enclosing
            # variable and returns it unchanged for execution.
            nonlocal graph
            graph = gm
            return gm

        fn = torch.compile(fn, backend=grab_graph_backend, fullgraph=True)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # y is already an input
        self.assertEqual(placeholder_cnt, 2)

    def test_const_property_on_tensor(self):
        # [FUNCTION DOC] Reads an integer attribute attached to an input tensor outside the
        # compiled function; checks the result matches eager and the graph has only one
        # placeholder, so the constant is not lifted to an input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by its attached attribute.
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4
        x_.y = y_

        eager_result = fn(x_)

        graph = None

        def grab_graph_backend(gm, inps):
            # [FUNCTION DOC] HELPER: Backend that stores the graph module in the enclosing
            # variable and returns it unchanged for execution.
            nonlocal graph
            graph = gm
            return gm

        fn = torch.compile(fn, backend=grab_graph_backend, fullgraph=True)
        compile_result = fn(x_)
        self.assertEqual(eager_result, compile_result)

        placeholder_cnt = 0
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                placeholder_cnt += 1

        # We want to be very sure that this does not lifts y to inputs, as its a const
        self.assertEqual(placeholder_cnt, 1)

    def test_const_property_assigned_on_tensor(self):
        # [FUNCTION DOC] Assigns an integer argument as an attribute on an input tensor inside
        # the compiled function (eager backend, full graph) and checks the result matches
        # eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the attribute on the first argument to the second
            # argument and returns the tensor times that attribute.
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])
        y_ = 4

        eager_result = fn(x_, y_)

        fn = torch.compile(fn, backend="eager", fullgraph=True)
        compile_result = fn(x_, y_)
        self.assertEqual(eager_result, compile_result)

    def test_guards_correctly_property_assigned_on_tensor_type_change(self):
        # [FUNCTION DOC] Compiles with the eager backend and a full graph, calls with an
        # integer then with a tensor as the assigned attribute value, and checks both results
        # equal the input times that value, showing guards handle the type change.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the attribute on the first argument to the second
            # argument and returns their product.
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch.compile(fn, backend="eager", fullgraph=True)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_guards_correctly_property_assigned_on_tensor_type_change_inductor(self):
        # [FUNCTION DOC] Same type-change scenario as the eager-backend variant but compiled
        # with the inductor backend: an integer then a tensor assigned as an attribute, each
        # result checked against the input times the value.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the attribute on the first argument to the second
            # argument and returns their product.
            x.y = y
            return x * x.y

        x_ = torch.randn([2, 2])

        fn = torch.compile(fn, backend="inductor", fullgraph=True)
        compile_result_const = fn(x_, 4)
        self.assertEqual(compile_result_const, x_ * 4)

        y = torch.randn([2, 2])
        compile_result_tensor = fn(x_, y)
        self.assertEqual(compile_result_tensor, x_ * y)

    def test_complex_attr_access_without_graph_breaks(self):
        # [FUNCTION DOC] Sets two attributes on each tensor in a list, then reads them back
        # and accumulates products, compiled with full graph. Asserts the result matches
        # eager, one frame, and nine operations.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: For each tensor in the list sets attribute y to the
            # tensor y and attribute z to y times z; then multiplies the attributes of all
            # tensors together and returns both products.
            for t in x:
                t.y = y
                t.z = y * z

            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z

            return new_y, new_z

        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]

        y = torch.randn([2, 2])
        z = 5

        eager_result = fn(x, y, z)

        counter = CompileCounter()
        fn = torch.compile(fn, backend=counter, fullgraph=True)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)
        # Graph for reference
        #         -------------  ------  -----------------------  ------------------------------------  --------
        # placeholder    l_y_    L_y_                     ()                                    {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)                             {}
        # call_function  mul_3   <built-in function mul>  (l_y_, 1)                             {}
        # call_function  mul_4   <built-in function mul>  (mul, 1)                              {}
        # call_function  mul_5   <built-in function mul>  (l_y_, mul_3)                         {}
        # call_function  mul_6   <built-in function mul>  (mul_1, mul_4)                        {}
        # call_function  mul_7   <built-in function mul>  (l_y_, mul_5)                         {}
        # call_function  mul_8   <built-in function mul>  (mul_2, mul_6)                        {}
        # output         output  output                   ((mul_7, mul_8, mul, mul_1, mul_2),)  {}

    def test_complex_attr_access_with_graph_breaks(self):
        # [FUNCTION DOC] Same as the full-graph variant but with a print call between the
        # attribute writes and reads, compiled without full graph. Asserts the result matches
        # eager, two frames (a graph break), and nine operations in total.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Sets attributes on each tensor in the list, prints a
            # message (forcing a graph break), then multiplies the attributes together and
            # returns both products.
            for t in x:
                t.y = y
                t.z = y * z

            print("Break!")

            new_y = 1
            new_z = 1
            for t in x:
                new_y = t.y * new_y
                new_z = t.z * new_z

            return new_y, new_z

        x_0 = torch.randn([2, 2])
        x_1 = torch.randn([2, 2])
        x_2 = torch.randn([2, 2])
        x = [x_0, x_1, x_2]

        y = torch.randn([2, 2])
        z = 5

        eager_result = fn(x, y, z)

        counter = CompileCounter()
        fn = torch.compile(fn, backend=counter, fullgraph=False)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 9)
        # Graph for reference
        # -------------  ------  -----------------------  ----------------------  --------
        # placeholder    l_y_    L_y_                     ()                      {}
        # call_function  mul     <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_1   <built-in function mul>  (l_y_, 5)               {}
        # call_function  mul_2   <built-in function mul>  (l_y_, 5)               {}
        # output         output  output                   ((mul, mul_1, mul_2),)  {}
        # [GRAPH BREAK!]
        # -------------  -------  -----------------------  -----------------  --------
        # placeholder    l_x_0_y  L_x_0_y                  ()                 {}
        # placeholder    l_x_0_z  L_x_0_z                  ()                 {}
        # placeholder    l_x_1_y  L_x_1_y                  ()                 {}
        # placeholder    l_x_1_z  L_x_1_z                  ()                 {}
        # placeholder    l_x_2_y  L_x_2_y                  ()                 {}
        # placeholder    l_x_2_z  L_x_2_z                  ()                 {}
        # call_function  mul      <built-in function mul>  (l_x_0_y, 1)       {}
        # call_function  mul_1    <built-in function mul>  (l_x_0_z, 1)       {}
        # call_function  mul_2    <built-in function mul>  (l_x_1_y, mul)     {}
        # call_function  mul_3    <built-in function mul>  (l_x_1_z, mul_1)   {}
        # call_function  mul_4    <built-in function mul>  (l_x_2_y, mul_2)   {}
        # call_function  mul_5    <built-in function mul>  (l_x_2_z, mul_3)   {}
        # output         output   output                   ((mul_4, mul_5),)  {}

    def test_complex_attr_access_with_inline_reconstruct(self):
        # [FUNCTION DOC] Sets an integer attribute on three input tensors, calls an inlined
        # function that prints and sums the attributes, and scales two tensors by the sum;
        # compiled without full graph. Asserts matching results, one frame and two operations.
        def inline_test_fn(x, y, z):
            # [FUNCTION DOC] HELPER: Prints a message and returns the sum of attribute a on
            # the three arguments.
            print("f")
            return x.a + y.a + z.a

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Sets attribute a to 1, 2 and 3 on the three tensors,
            # calls the inlined helper to get their sum, then returns the first two tensors
            # multiplied by it.
            x.a = 1
            y.a = 2
            z.a = 3

            mult = inline_test_fn(x, y, z)
            y = y * mult
            x = x * mult
            return x, y

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        z = torch.randn([2, 2])

        eager_result = fn(x, y, z)

        counter = CompileCounter()

        fn = torch.compile(fn, backend=counter, fullgraph=False)

        compile_result = fn(x, y, z)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)
        # Graph for reference
        # __compiled_fn_2 <eval_with_key>.0 opcode         name    target                   args             kwargs
        # -------------  ------  -----------------------  ---------------  --------
        # placeholder    l_x_    L_x_                     ()               {}
        # placeholder    l_y_    L_y_                     ()               {}
        # call_function  mul     <built-in function mul>  (l_y_, 6)        {}
        # call_function  mul_1   <built-in function mul>  (l_x_, 6)        {}
        # output         output  output                   ((mul_1, mul),)  {}

    def test_set_data_on_input_tensor(self):
        # [FUNCTION DOC] Assigns the data of one input tensor (different shape) to another
        # input inside a fully compiled function and multiplies them. Checks the result
        # matches eager, one frame, five operations, and that the normalized printed graph
        # matches the expected inline text, which uses the get-data-attribute call and the
        # shallow-copy-data op with grad disabled around it.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the data of the first tensor to the data of the
            # second; returns their product if their sizes are equal (always true after the
            # assignment), otherwise the square of the second.
            x.data = y.data
            if x.size() == y.size():
                return x * y
            else:
                return y * y

        x = torch.randn([5, 5])
        y = torch.randn([2, 2])

        eager_result = fn(x, y)

        eager_and_record = EagerAndRecordGraphs()

        counter = CompileCounterWithBackend(eager_and_record)

        fn = torch.compile(fn, backend=counter, fullgraph=True)

        compile_result = fn(x, y)

        graph = eager_and_record.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 5)
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 2]", L_y_: "f32[2, 2]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _get_data_attr: "f32[2, 2]" = torch._C._autograd._get_data_attr(l_y_)

        _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

        shallow_copy_data_: "f32[2, 2]" = torch.ops.aten.shallow_copy_data_(l_x_, _get_data_attr);  _get_data_attr = shallow_copy_data_ = None

        _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None

        mul: "f32[2, 2]" = l_x_ * l_y_;  l_x_ = l_y_ = None
        return (mul,)
""",
        )

    # See [Note: set_data_on_scoped_tensor] This does not actually get captured in the graph yet.
    def test_set_data_on_scoped_tensor(self):
        # [FUNCTION DOC] Creates a local tensor, assigns an input tensor's data to it and
        # multiplies them, compiled without full graph. Asserts the result matches eager, two
        # frames and three operations (this data assignment on a scoped tensor currently
        # causes a graph break per the referenced note).
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a zero tensor, sets its data to the input's data,
            # then returns their product if sizes match, else the input.
            z = torch.zeros([4, 4])
            z.data = x.data
            if x.size() == z.size():
                return z * x
            else:
                return x

        x = torch.randn([5, 5])

        eager_result = fn(x)

        counter = CompileCounter()

        fn = torch.compile(fn, backend=counter, fullgraph=False)

        compile_result = fn(x)
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 3)

    def test_set_data_on_user_defined_class_input_tensor(self):
        # [FUNCTION DOC] Compiles a bound method of a user-defined object that sets attributes
        # derived from enclosing-scope tensors, with a full graph, and compares to the same
        # method run eagerly on another object. Asserts one frame and three operations.
        class MyUserDefinedClass:
            # [CLASS DOC] HELPER: Simple object storing two tensors in attributes x and y and
            # offering a method that sets two more attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the two arguments on the instance as x and y.
                self.x = x
                self.y = y

            def do_some_setattr_stuff(self):
                # [FUNCTION DOC] HELPER: Sets attribute z to the product and attribute a to
                # the sum of the enclosing-scope tensors x and y, and returns their product.
                # KNOWN ISSUE: uses the enclosing variables x and y, not the instance
                # attributes, though the values are the same tensors in this test.
                self.z = x * y
                self.a = x + x
                return self.z * self.a

        x = torch.randn([5, 5])
        y = torch.randn([5, 5])
        mudc_1 = MyUserDefinedClass(x, y)

        eager_result = mudc_1.do_some_setattr_stuff()

        counter = CompileCounter()

        mudc_2 = MyUserDefinedClass(x, y)
        do_some_setattr_stuff = torch.compile(
            mudc_2.do_some_setattr_stuff, backend=counter, fullgraph=True
        )

        compile_result = do_some_setattr_stuff()
        self.assertEqual(compile_result, eager_result)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)
        # Graph for reference
        #  __compiled_fn_0 <eval_with_key>.0 opcode         name    target                   args                  kwargs
        # -------------  ------  -----------------------  --------------------  --------
        # placeholder    l_x_    L_x_                     ()                    {}
        # placeholder    l_y_    L_y_                     ()                    {}
        # call_function  mul     <built-in function mul>  (l_x_, l_y_)          {}
        # call_function  add     <built-in function add>  (l_x_, l_x_)          {}
        # call_function  mul_1   <built-in function mul>  (mul, add)            {}
        # output         output  output                   ((mul_1, mul, add),)  {}


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
