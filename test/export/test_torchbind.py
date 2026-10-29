# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# [MODULE DOC] Tests exporting modules that hold or receive TorchBind (script class) objects such
# as the testing Foo class and tensor queue: attributes, inputs, aliases, custom op arguments,
# list/tuple returns, queue mutation, effects tokens, retracing, aot_export_module, make_fx, dynamo
# graph capture and wrapping higher order ops. Defines assertion helpers, and the
# TestExportTorchbind case with fake class registration in setUp. Import time: imports only; the
# parametrized tests are instantiated at the bottom of the module.

import copy

import torch
import torch.utils._pytree as pytree
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._higher_order_ops.wrap import wrap
from torch._library.fake_class_registry import FakeScriptObject
from torch.export._trace import _export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _assertEqualSkipScriptObject(test_case, exp, actual):
    # [FUNCTION DOC] HELPER: Flattens both nested structures to leaves, checks the leaf counts
    # match, then compares leaves with the test case equality assertion, skipping any pair where
    # both leaves are script objects. Raises assertion failures through the test case.
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    for a, b in zip(flat_exp, flat_actual):
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            continue
        test_case.assertEqual(a, b)


def _check_script_obj_equal(test_case, a: torch.ScriptObject, b: torch.ScriptObject):
    # [FUNCTION DOC] HELPER: Default object comparator. Asserts two script objects have the same
    # qualified type name and equal flattened state (via the object flatten method). Returns the
    # result of the last assertion (None on success); failures raise through the test case.
    return test_case.assertEqual(
        a._type().qualified_name(), b._type().qualified_name()
    ) and test_case.assertEqual(a.__obj_flatten__(), b.__obj_flatten__())


def _assertEqualScriptObject(
    test_case, exp, actual, check_obj_eq=_check_script_obj_equal
):
    # [FUNCTION DOC] HELPER: Like the skip variant but compares script object leaf pairs with the
    # check_obj_eq callable (default compares type name and flattened state), and all other
    # leaf pairs with the test case equality assertion. Also checks the leaf counts match.
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    for a, b in zip(flat_exp, flat_actual):
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            check_obj_eq(test_case, a, b)
        else:
            test_case.assertEqual(a, b)


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    # [CLASS DOC] Test case for exporting programs involving TorchBind objects; skipped under
    # dynamo. Per-test state: counters (kept on the test instance) of fake queue push, pop, size
    # calls and fake Foo add_tensor calls, plus the list of torchbind ops. Most tests export with
    # strict off, via a shared helper that also checks eager equivalence, kwargs reordering and
    # retracing, and then assert the generated code of the unlifted module and the graph module.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Initializes the torchbind test implementations, zeroes the four
        # call counters on the test instance, registers fake classes for the Foo and tensor queue
        # script classes that increment those counters when called, and stores the list of
        # torchbind custom ops used by the tests.
        super().setUp()
        init_torchbind_implementations()

        test = self
        test.tq_push_counter = 0
        test.tq_pop_counter = 0
        test.tq_size_counter = 0
        test.foo_add_tensor_counter = 0

        # We need different fake classes, which update the counters
        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            # [CLASS DOC] HELPER: Fake stand-in for the Foo script class. Stores two integers; its
            # add tensor method bumps the foo add_tensor counter of the enclosing test.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: Stores the two integer fields.
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flattend_foo):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from the flattened
                # state, a sequence of name and value pairs turned into keyword arguments.
                return cls(**dict(flattend_foo))

            def add_tensor(self, z):
                # [FUNCTION DOC] HELPER: Increments the add_tensor counter and returns the
                # tensor argument scaled by the sum of the two stored integers.
                test.foo_add_tensor_counter += 1
                return (self.x + self.y) * z

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            # [CLASS DOC] HELPER: Fake stand-in for the tensor queue script class holding a plain
            # Python list as the queue; its push, pop and size methods bump the matching counters
            # of the enclosing test.
            def __init__(self, queue):
                # [FUNCTION DOC] HELPER: Stores the given list as the queue.
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from the flattened
                # state, a sequence of name and value pairs turned into keyword arguments.
                return cls(**dict(flattened_ctx))

            def push(self, x):
                # [FUNCTION DOC] HELPER: Increments the push counter and appends the tensor to
                # the end of the queue.
                test.tq_push_counter += 1
                self.queue.append(x)

            def pop(self):
                # [FUNCTION DOC] HELPER: Increments the pop counter and removes and returns the
                # first element of the queue.
                test.tq_pop_counter += 1
                return self.queue.pop(0)

            def size(self):
                # [FUNCTION DOC] HELPER: Increments the size counter and returns the queue length.
                test.tq_size_counter += 1
                return len(self.queue)

            def is_empty(self):
                # [FUNCTION DOC] HELPER: Returns whether the queue has no elements (no counter).
                return len(self.queue) == 0

            def float_size(self):
                # [FUNCTION DOC] HELPER: Returns the queue length as a float (no counter).
                return float(len(self.queue))

        self.torch_bind_ops = [
            torch.ops._TorchScriptTesting.takes_foo,
            torch.ops._TorchScriptTesting.takes_foo_python_meta,
            torch.ops._TorchScriptTesting.takes_foo_list_return,
            torch.ops._TorchScriptTesting.takes_foo_tuple_return,
            torch.ops._TorchScriptTesting.take_an_instance,
            torch.ops._TorchScriptTesting.take_an_instance_inferred,
            torch.ops._TorchScriptTesting.takes_foo_cia,
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_size,
        ]

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Deregisters the fake classes registered in setUp for the Foo and
        # tensor queue script classes so each test starts clean.
        # KNOWN ISSUE: does not call the base class tearDown.
        torch._library.fake_class_registry.deregister_fake_class(
            "_TorchScriptTesting::_Foo"
        )
        torch._library.fake_class_registry.deregister_fake_class(
            "_TorchScriptTesting::_TensorQueue"
        )

    def _test_export_same_as_eager(
        self, f, args, kwargs=None, strict=True, pre_dispatch=False
    ):
        # [FUNCTION DOC] HELPER: Exports the function or module with the torchbind tracing
        # context enabled, either through the public export followed by running decompositions
        # (pre-dispatch) or through the private export (non pre-dispatch). Checks that the
        # unlifted module matches eager for the given keyword order and for reversed keyword
        # order, then re-exports the unlifted module and checks it also matches. Script objects
        # are compared by type name and flattened state. Returns the exported program.
        kwargs = kwargs or {}

        def export_wrapper(f, args, kwargs, strict, pre_dispatch):
            # [FUNCTION DOC] HELPER: Exports under the torchbind tracing context using the path
            # selected by the pre-dispatch flag, and returns the exported program.
            with enable_torchbind_tracing():
                if pre_dispatch:
                    exported_program = torch.export.export(
                        f, args, kwargs, strict=strict
                    ).run_decompositions({})
                else:
                    exported_program = _export(
                        f, args, kwargs, strict=strict, pre_dispatch=False
                    )
            return exported_program

        exported_program = export_wrapper(f, args, kwargs, strict, pre_dispatch)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        unlifted = exported_program.module()
        exp = f(*args, **kwargs)
        _assertEqualScriptObject(self, unlifted(*args, **kwargs), exp)
        _assertEqualScriptObject(
            self,
            unlifted(*args, **reversed_kwargs),
            exp,
        )

        # check re-tracing
        retraced_ep = export_wrapper(unlifted, args, kwargs, strict, pre_dispatch)
        _assertEqualScriptObject(self, retraced_ep.module()(*args, **kwargs), exp)
        return exported_program

    @parametrize("pre_dispatch", [True, False])
    def test_none(self, pre_dispatch):
        # [FUNCTION DOC] Exports (non strict, parametrized pre-dispatch) a module that calls a
        # method on a Foo attribute and has an extra input that is None. Checks the unlifted code
        # uses the call torchbind higher order op and that the graph module threads an effect
        # token through it.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x, n):
                # [FUNCTION DOC] HELPER: Returns the input plus the attribute's add tensor result;
                # the second argument is unused.
                return x + self.attr.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), None),
            strict=False,
            pre_dispatch=pre_dispatch,
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x, n):
    x, n, = fx_pytree.tree_flatten_spec(([x, n], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x, n);  n = _guards_fn = None
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x, n):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj_attr, 'add_tensor', x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",
        )

    def test_method_schema(self):
        # [FUNCTION DOC] Builds an empty tensor queue, converts it to a fake object under a fake
        # tensor mode, and checks the string form of the schemas of its push and pop methods
        # (the queue type plus a tensor argument returning none; and the queue type returning a
        # tensor).
        tq = _empty_tensor_queue()
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_obj = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, tq)
        self.assertExpectedInline(
            str(fake_obj.push.schema),
            """push(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, Tensor _1) -> NoneType _0""",
        )
        self.assertExpectedInline(
            str(fake_obj.pop.schema),
            """pop(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0) -> Tensor _0""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module that calls a method of a Foo attribute (non strict,
        # parametrized pre-dispatch) and checks the generated unlifted and graph module code, in
        # which the object is lifted to an input and calls are sequenced with effect tokens.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the attribute's add tensor result.
                return x + self.attr.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj_attr, 'add_tensor', x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute_as_custom_op_argument(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module that passes a Foo attribute as an argument to a
        # torchbind custom op (non strict, parametrized pre-dispatch) and checks the generated
        # unlifted and graph module code, where the op call is wrapped with an effect token.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the custom op result for the
                # attribute and the input.
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_input(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module taking a Foo script object as a forward input (non
        # strict, parametrized pre-dispatch), checks generated code for the unlifted and graph
        # module, and checks the fake add_tensor call counter is six for pre-dispatch and four
        # otherwise (the export runs the program several times, and retracing doubles it).
        cc = torch.classes._TorchScriptTesting._Foo(10, 20)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module taking a tensor and a Foo object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x, cc):
                # [FUNCTION DOC] HELPER: Returns the input plus the object's add tensor result.
                return x + cc.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x, cc):
    x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
    _guards_fn = self._guards_fn(x, cc);  _guards_fn = None
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, x, cc):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, cc, 'add_tensor', x);  token = cc = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",
        )
        # aot_export_function runs the program twice
        # in run_functionalized_fw_and_collect_metadata and create_aot_dispatcher_function
        # We also have a re-tracing test, which doubles the count.
        if pre_dispatch:
            self.assertEqual(self.foo_add_tensor_counter, 6)
        else:
            self.assertEqual(self.foo_add_tensor_counter, 4)

    @parametrize("pre_dispatch", [True, False])
    def test_input_as_custom_op_argument(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module passing a Foo input to a torchbind custom op. First
        # removes the registered Meta kernel of the op and clears its dispatch cache, and checks
        # export raises a RuntimeError saying no python implementation is found. Then registers
        # a Meta implementation calling the object's method, re-exports and checks generated
        # code. Side effect: leaves the new Meta kernel registered on the global op.
        cc = torch.classes._TorchScriptTesting._Foo(10, 20)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module taking a tensor and a Foo object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, x, cc):
                # [FUNCTION DOC] HELPER: Returns the input plus the custom op result for the
                # object and the input.
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        del torch.ops._TorchScriptTesting.takes_foo.default.py_kernels[
            torch._C.DispatchKey.Meta
        ]
        torch.ops._TorchScriptTesting.takes_foo.default._dispatch_cache.clear()
        # Even though a C++ implementation for takes_foo.default is registered,
        # we still need the python implementation for takes_foo.default to trace with FakeFoo.
        with self.assertRaisesRegex(RuntimeError, "no python implementation is found"):
            self._test_export_same_as_eager(
                MyModule(),
                (torch.ones(2, 3), cc),
                strict=False,
                pre_dispatch=pre_dispatch,
            )

        torch.ops._TorchScriptTesting.takes_foo.default.py_impl(
            torch._C.DispatchKey.Meta
        )(lambda cc, x: cc.add_tensor(x))
        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), cc),
            strict=False,
            pre_dispatch=pre_dispatch,
        )

        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x, cc):
    x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
    _guards_fn = self._guards_fn(x, cc);  _guards_fn = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(cc, x);  cc = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, x, cc):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, cc, x);  token = cc = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_torchbind_alias(self, pre_dispatch):
        # [FUNCTION DOC] Exports (non strict) a module whose Foo attribute is aliased under
        # several attribute names and also held by a child module, with calls through the alias
        # and the child; only checks eager equivalence via the shared helper.
        class F2(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module storing the given script object.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores the given script object as an attribute.
                super().__init__()
                self.foo = foo

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the custom op result for the
                # stored object and the input.
                return x + torch.ops._TorchScriptTesting.takes_foo(self.foo, x)

        class F1(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module with one Foo object under three attribute names
            # and a child module sharing it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo object, aliases it as two more
                # attributes and builds the child with the alias.
                super().__init__()
                self.alpha = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.beta = self.alpha
                self.gamma = self.alpha
                self.foo = F2(self.gamma)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the custom op result through the
                # alias plus the child's output.
                return (
                    x
                    + torch.ops._TorchScriptTesting.takes_foo(self.gamma, x)
                    + self.foo(x)
                )

        self._test_export_same_as_eager(
            F1(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )

    def test_torchbind_register_attr_at_runtime_get_restored(self):
        # [FUNCTION DOC] Exports (non strict) a module whose forward assigns its Foo input as a
        # new module attribute, then checks the original script object does not keep an
        # attribute with that name (the runtime attribute registration is undone).
        # alias as model attribute
        class F3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that registers its object argument as an attribute
            # during forward.
            def forward(self, x, foo):
                # [FUNCTION DOC] HELPER: Stores the object on the module and returns the input
                # plus its add tensor result.
                self.foo = foo
                return x + self.foo.add_tensor(x)

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        torch.export.export(F3(), (torch.ones(2, 3), foo), strict=False)
        self.assertFalse(hasattr(foo, "foo"))

    @parametrize("pre_dispatch", [True, False])
    def test_torchbind_input_and_alias(self, pre_dispatch):
        # [FUNCTION DOC] Exports (non strict, parametrized pre-dispatch) a module constructed with
        # a Foo object that is also stored as an attribute, checking eager equivalence only.
        # alias as model attribute
        class F3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a script object given at construction.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores the given script object as an attribute.
                super().__init__()
                self.foo = foo

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the object's add tensor result.
                return x + self.foo.add_tensor(x)

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        self._test_export_same_as_eager(
            F3(foo), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )

    @parametrize("pre_dispatch", [True, False])
    def test_unlift_custom_obj(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module calling a torchbind custom op twice on the same Foo
        # attribute (non strict, parametrized pre-dispatch) and checks generated code for the
        # unlifted and graph modules, where the two calls are chained through effect tokens.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the custom op on the input, then on that result,
                # and returns the input plus the second result.
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, x)
    takes_foo_default_1 = torch.ops._TorchScriptTesting.takes_foo.default(attr, takes_foo_default);  attr = takes_foo_default = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default_1);  x = takes_foo_default_1 = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, getitem_1);  getitem = obj_attr = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(x, getitem_3);  x = getitem_3 = None
    return (getitem_2, add)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_list_out(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module using a torchbind op that returns a list of three
        # tensors, whose sum feeds a second op (non strict, parametrized pre-dispatch), and checks
        # generated code for the unlifted and graph modules.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the list-returning op, sums the three results,
                # feeds the sum to the single-return op and adds that to the input.
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(attr, x)
    getitem_2 = takes_foo_list_return_default[0]
    getitem_3 = takes_foo_list_return_default[1]
    getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add_1);  attr = add_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add_2,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_list_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1]
    getitem_4 = getitem_1[2];  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add_1);  getitem = obj_attr = add_1 = None
    getitem_5 = with_effects_1[0]
    getitem_6 = with_effects_1[1];  with_effects_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, getitem_6);  x = getitem_6 = None
    return (getitem_5, add_2)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_tuple_out(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module using a torchbind op that returns a tuple of two
        # tensors, whose sum feeds a second op (non strict, parametrized pre-dispatch), and checks
        # generated code for the unlifted module and the graph module.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values ten and twenty.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the tuple-returning op, sums its two results, feeds
                # the sum to the single-return op and adds that to the input.
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(attr, x)
    getitem_1 = takes_foo_tuple_return_default[0]
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add);  attr = add = None
    add_1 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add_1,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_tuple_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1]
    getitem_2 = with_effects[2];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add);  getitem = obj_attr = add = None
    getitem_3 = with_effects_1[0]
    getitem_4 = with_effects_1[1];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(x, getitem_4);  x = getitem_4 = None
    return (getitem_3, add_1)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_unbacked_symint(self, pre_dispatch):
        # [FUNCTION DOC] Exports a module whose torchbind op returns a tensor with data dependent
        # shape (non strict, parametrized pre-dispatch). Finds the op node in the unlifted graph
        # and checks its unbacked bindings has two entries; the first key is a plain symbol and
        # its path has two elements: a method-call key named size and a sequence key with index
        # zero (the effect token is not part of the path).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Foo script object attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo attribute with values two and three.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the result of the tensor-returning torchbind op
                # applied to the attribute and the input.
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                return a

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        gm = ep.module()
        foo_node = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops._TorchScriptTesting.takes_foo_tensor_return.default
        )
        unbacked_bindings = foo_node.meta["unbacked_bindings"]
        self.assertEqual(len(unbacked_bindings), 2)
        u = next(iter(unbacked_bindings.keys()))
        path = unbacked_bindings[u]
        # the unbacked bindings should be CallMethodKey(name='size'), SequenceKey(idx=0)
        # it should not include the effect token in the path
        self.assertEqual(
            type(u).__name__, "Symbol"
        )  # check binding is symbol, not expr
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0].name, "size")
        self.assertEqual(path[1].idx, 0)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods(self, make_fx_tracing_mode):
        # [FUNCTION DOC] Traces a model that pushes, pops and sizes a tensor queue input with the
        # make_fx tracer in the parametrized mode (fake or symbolic). Checks the model saw a fake
        # script object while tracing, the fake queue counters (two pushes, two pops, two size
        # calls), the real queue untouched, and the traced code (calls through the call torchbind
        # op, with size results specialized to constants). Then turns off the fake check and
        # compares the traced graph against eager on separate real queues.
        test = self

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an unused linear layer and a flag controlling
            # whether forward asserts the queue argument is a fake script object (the enclosing
            # test case does the asserting).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer and sets the fake check flag on.
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Optionally asserts the queue is a fake script object,
                # pushes cosine and sine of the input, pops twice combining each popped value with
                # the queue size, and returns the two results and the queue.
                if self.check_tq_is_fake:
                    test.assertTrue(isinstance(tq, FakeScriptObject))
                tq.push(x.cos())
                tq.push(x.sin())
                x_cos = tq.pop() + tq.size()
                x_sin = tq.pop() - tq.size()
                return x_sin, x_cos, tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)
        gm = make_fx(mod, tracing_mode=make_fx_tracing_mode)(tq, x)
        self.assertEqual(self.tq_push_counter, 2)
        self.assertEqual(self.tq_pop_counter, 2)
        self.assertEqual(self.tq_size_counter, 2)
        self.assertEqual(tq.size(), 0)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'push', cos);  cos = call_torchbind = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'push', sin);  sin = call_torchbind_1 = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_3 = None
    add = torch.ops.aten.add.Tensor(call_torchbind_2, 1);  call_torchbind_2 = None
    call_torchbind_4 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_5 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_5 = None
    sub = torch.ops.aten.sub.Tensor(call_torchbind_4, 0);  call_torchbind_4 = None
    return (sub, add, arg0_1)
    """,
        )
        mod.check_tq_is_fake = False
        _assertEqualSkipScriptObject(self, gm(tq, x), mod(tq1, x))

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods_fakify_internal_states(
        self, make_fx_tracing_mode
    ):
        # [FUNCTION DOC] Like the previous make_fx queue test, but the queue starts with two
        # tensors so pops are valid at trace time. Checks the traced graph does not push, pops
        # twice and sizes twice on the fake object (counters), that the real queue is unchanged
        # by tracing, and the traced code. Then runs the graph and eager on separate real queues
        # and checks both queues end empty.
        test = self

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an unused linear layer, a flag controlling the fake
            # script object check, and a reference to the running test case that does the check.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer, sets the fake check flag on and
                # keeps the test case reference.
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.check_tq_is_fake = True
                self.current_test = test

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Optionally asserts the queue is a fake script object,
                # pops twice, combining each popped value with the queue size and the input, and
                # returns the two results and the queue.
                if self.check_tq_is_fake:
                    self.current_test.assertTrue(isinstance(tq, FakeScriptObject))
                x_cos = tq.pop() + tq.size() + x
                x_sin = tq.pop() - tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        for _ in range(2):
            tq.push(torch.ones(2, 3))
            tq1.push(torch.ones(2, 3))
        x = torch.ones(2, 3)
        prev_size = tq.size()
        gm = make_fx(mod, tracing_mode=make_fx_tracing_mode)(tq, x)
        self.assertEqual(self.tq_push_counter, 0)
        self.assertEqual(self.tq_pop_counter, 2)
        self.assertEqual(self.tq_size_counter, 2)
        self.assertEqual(tq.size(), prev_size)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_1 = None
    add = torch.ops.aten.add.Tensor(call_torchbind, 1);  call_torchbind = None
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_3 = None
    sub = torch.ops.aten.sub.Tensor(call_torchbind_2, 0);  call_torchbind_2 = None
    add_2 = torch.ops.aten.add.Tensor(sub, arg1_1);  sub = arg1_1 = None
    return (add_2, add_1, arg0_1)
    """,
        )
        # turn off tq type checking in eager execution
        mod.check_tq_is_fake = False
        _assertEqualSkipScriptObject(self, gm(tq, x), mod(tq1, x))
        self.assertEqual(tq.size(), 0)
        self.assertEqual(tq1.size(), 0)

    def test_non_strict_export_methods(self):
        # [FUNCTION DOC] Exports (non strict, then decomposed) a model that calls pop, float size,
        # is empty and size methods on a real tensor queue input holding two tensors, and checks
        # the graph module code with effect tokens threaded through each queue method. Checks the
        # real queue still has both items afterwards, in the original order and identity.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer using a tensor queue input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a two by two linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Pops a tensor and uses the float size and the linear layer
                # for a first result; branches on whether the queue is empty to compute a second
                # result from another pop (with the linear layer) or from a pop plus size; returns
                # both results and the queue.
                x_cos = tq.pop() + tq.float_size() + self.linear(x)
                if tq.is_empty():
                    x_sin = self.linear(tq.pop()) - tq.size() + x
                else:
                    x_sin = tq.pop() + tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = _empty_tensor_queue()
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        tq.push(a)
        tq.push(b)
        ep = torch.export.export(
            mod, (tq, torch.randn(2, 2)), strict=False
        ).run_decompositions({})
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, p_linear_weight, p_linear_bias, tq, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, tq, 'pop');  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.call_torchbind, tq, 'float_size');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1.0);  getitem_1 = None
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  p_linear_weight = p_linear_bias = None
    add_1 = torch.ops.aten.add.Tensor(add, linear);  add = linear = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops.higher_order.call_torchbind, tq, 'is_empty');  getitem_2 = None
    getitem_4 = with_effects_2[0];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops.higher_order.call_torchbind, tq, 'pop');  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops.higher_order.call_torchbind, tq, 'size');  getitem_6 = None
    getitem_8 = with_effects_4[0];  with_effects_4 = None
    add_2 = torch.ops.aten.add.Tensor(getitem_7, 0);  getitem_7 = None
    add_3 = torch.ops.aten.add.Tensor(add_2, x);  add_2 = x = None
    return (getitem_8, add_3, add_1, tq)""",
        )
        self.assertEqual(tq.size(), 2)
        self.assertTrue(tq.pop() is a)
        self.assertTrue(tq.pop() is b)

    @skipIfCrossRef  # arg names change with torch function mode
    def test_safe_to_trace_with_real(self):
        # [FUNCTION DOC] Checks that a script object marked safe for tracing (a constant tensor
        # container) is traced with its real value: compiles a module with full graph under a
        # recording backend and compares code and results with eager, then exports non strictly
        # under torchbind tracing and checks the decomposed graph code. Skipped under cross
        # ref mode because argument names change.
        x = torch.randn(3, 3)
        safe_obj = torch.classes._TorchScriptTesting._ConstantTensorContainer(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reads a tensor from a script object input.
            def forward(self, safe_obj: torch.ScriptObject) -> None:
                # [FUNCTION DOC] HELPER: Returns the sine of the tensor obtained from the object.
                return safe_obj.get().sin()

        mod = Mod()
        backend = EagerAndRecordGraphs()
        out = torch.compile(mod, backend=backend, fullgraph=True)(safe_obj)
        self.assertEqual(out, mod(safe_obj))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_safe_obj_ : torch.ScriptObject):
    l_safe_obj_ = L_safe_obj_
    call_torchbind = torch.ops.higher_order.call_torchbind(l_safe_obj_, 'get');  l_safe_obj_ = None
    sin = call_torchbind.sin();  call_torchbind = None
    return (sin,)""",
        )

        with enable_torchbind_tracing():
            ep = torch.export.export(mod, (safe_obj,), strict=False).run_decompositions(
                {}
            )
            self.assertExpectedInline(
                ep.graph_module.code.strip(),
                """\
def forward(self, token, safe_obj):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, safe_obj, 'get');  token = safe_obj = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    sin = torch.ops.aten.sin.default(getitem_1);  getitem_1 = None
    return (getitem, sin)""",
            )

    def test_identifying_torchbind_ops(self):
        # [FUNCTION DOC] Checks every op in the test's list of torchbind ops reports having a
        # torchbind overload, and that ordinary aten ops (add, cos) do not.
        for op in self.torch_bind_ops:
            self.assertTrue(op._has_torchbind_op_overload)

        for op in [
            torch.ops.aten.add,
            torch.ops.aten.cos,
        ]:
            self.assertFalse(op._has_torchbind_op_overload)

    def test_torchbind_op_register_fallthrough(self):
        # [FUNCTION DOC] For each torchbind op, registers a fallthrough kernel for an autograd
        # dispatch key in a scoped library fragment and checks the dispatcher reports that key as
        # a fallthrough for the op. The fragment is removed after each iteration.
        TEST_DISPATCH_KEY = torch._C.DispatchKey.AutogradCPU
        TEST_DISPATCH_KEY_STR = "AutogradCPU"

        for op_packet in self.torch_bind_ops:
            op = op_packet.default
            ns, _ = torch._library.utils.parse_namespace(op_packet._qualified_op_name)
            with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                lib.impl(
                    op.name(), torch.library.fallthrough_kernel, TEST_DISPATCH_KEY_STR
                )
                self.assertTrue(
                    torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                        op.name(), TEST_DISPATCH_KEY
                    )
                )

    def test_torchbind_op_fallthrough_keys_respects_lib_impl(self):
        # [FUNCTION DOC] For each torchbind op without a native or Python kernel for an autograd
        # dispatch key, registers first a regular kernel and checks the key is not among the
        # op's fallthrough keys, then (in a new scoped fragment) a fallthrough kernel and checks
        # the key is among them. Asserts that at least one op was exercised.
        TEST_DISPATCH_KEY = torch._C.DispatchKey.AutogradCPU
        TEST_DISPATCH_KEY_STR = "AutogradCPU"

        tested = 0
        for op_packet in self.torch_bind_ops:
            op = op_packet.default
            ns, _ = torch._library.utils.parse_namespace(op_packet._qualified_op_name)
            if (
                not torch._C._dispatch_has_kernel_for_dispatch_key(
                    op.name(), TEST_DISPATCH_KEY
                )
                and TEST_DISPATCH_KEY not in op.py_kernels
            ):
                tested += 1
                with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                    lib.impl(
                        op.name(), lambda *args, **kwargs: args, TEST_DISPATCH_KEY_STR
                    )
                    self.assertTrue(TEST_DISPATCH_KEY not in op._fallthrough_keys())

                with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                    lib.impl(
                        op.name(),
                        torch.library.fallthrough_kernel,
                        TEST_DISPATCH_KEY_STR,
                    )
                    self.assertTrue(TEST_DISPATCH_KEY in op._fallthrough_keys())
        self.assertTrue(tested > 0)

    def test_make_fx_schema_checking_script_object(self):
        # [FUNCTION DOC] Checks that make_fx in fake mode rejects a real script object passed to
        # the queue push torchbind op, both for positional and keyword calls, raising a
        # RuntimeError saying a fake script object is expected. Registers fallthrough kernels for
        # three dispatch keys in a scoped library for the duration of the check.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the queue push op positionally.
            def forward(self, tq, x, foo):
                # [FUNCTION DOC] HELPER: Pushes the cosine of the input onto the object using the
                # op with positional arguments and returns the queue.
                torch.ops._TorchScriptTesting.queue_push(foo, x.cos())
                return tq

        class ModelCallByKW(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the queue push op with keyword arguments.
            def forward(self, tq, x, foo):
                # [FUNCTION DOC] HELPER: Pushes the cosine of the input onto the object using the
                # op with keyword arguments and returns the queue.
                torch.ops._TorchScriptTesting.queue_push(x=x.cos(), foo=foo)
                return tq

        mod = Model()
        modkw = ModelCallByKW()

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        x = torch.ones(3, 3)
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        ns = "_TorchScriptTesting"
        with torch.library._scoped_library(ns, "FRAGMENT") as lib:
            op = torch.ops._TorchScriptTesting.queue_push
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "AutogradCPU")
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "ADInplaceOrView")
            lib.impl(
                op.__name__,
                torch.library.fallthrough_kernel,
                "PythonTLSSnapshot",
            )

            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(mod, tracing_mode="fake")(tq, x, foo)

            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(modkw, tracing_mode="fake")(tq, x, foo)

    @parametrize("fallthrough_via", ["lib_impl", "py_impl"])
    def test_make_fx_tensor_queue_operators(self, fallthrough_via):
        # [FUNCTION DOC] Traces a model, which uses the queue push, pop and size ops under a cuda
        # autocast, with make_fx in fake mode, after making the autograd CPU dispatch key a
        # fallthrough either by a scoped library kernel or by a Python implementation
        # (parametrized; the Python variant removes the kernel and clears the dispatch caches
        # afterwards). Runs the model eagerly once beforehand, checks the traced code and
        # compares the graph with eager on a second real queue.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using the queue operators inside a cuda autocast block.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Under autocast, pushes cosine and sine of the input,
                # pops twice and combines each popped value with the queue size by subtraction and
                # addition, and returns both results and the queue.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                    torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                    x_sin = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) - torch.ops._TorchScriptTesting.queue_size(tq)
                    x_cos = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) + torch.ops._TorchScriptTesting.queue_size(tq)
                    return x_sin, x_cos, tq

        mod = Model()

        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq2 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)

        mod(tq1, x)

        ops = [
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_size,
        ]
        if fallthrough_via == "lib_impl":
            ns = "_TorchScriptTesting"
            with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                for op in ops:
                    lib.impl(
                        op.__name__, torch.library.fallthrough_kernel, "AutogradCPU"
                    )

                gm = make_fx(mod, tracing_mode="fake")(tq1, x)
        else:
            for op in ops:
                op.default.py_impl(torch._C.DispatchKey.AutogradCPU)(
                    torch.library.fallthrough_kernel
                )
            gm = make_fx(mod, tracing_mode="fake")(tq1, x)
            for op in ops:
                op.default._dispatch_cache.clear()
                del op.default.py_kernels[torch._C.DispatchKey.AutogradCPU]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, cos);  cos = queue_push = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_1 = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, sin);  sin = queue_push_1 = None
    queue_pop = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size = torch.ops._TorchScriptTesting.queue_size.default(arg0_1);  queue_size = None
    sub = torch.ops.aten.sub.Tensor(queue_pop, 1);  queue_pop = None
    queue_pop_1 = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TorchScriptTesting.queue_size.default(arg0_1);  queue_size_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop_1, 0);  queue_pop_1 = None
    return (sub, add, arg0_1)""",
        )
        _assertEqualSkipScriptObject(self, gm(tq1, x), mod(tq2, x))

    def test_aot_export_tensor_queue_operators(self):
        # [FUNCTION DOC] Exports a model using queue push, pop and size ops with the functorch
        # module export (no joint graph) on a fake queue and fake tensor, and checks the graph
        # code: an effect token is the first input and output and each queue op is wrapped with
        # effects.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using the queue operators.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the base module initializer.
                super().__init__()

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Pushes cosine and sine of the input, pops twice combining
                # each popped value with the queue size, and returns both results and the queue.
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                x_sin = torch.ops._TorchScriptTesting.queue_pop(
                    tq
                ) - torch.ops._TorchScriptTesting.queue_size(tq)
                x_cos = torch.ops._TorchScriptTesting.queue_pop(
                    tq
                ) + torch.ops._TorchScriptTesting.queue_size(tq)
                return x_sin, x_cos, tq

        mod = Model()

        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_tq1 = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, tq1)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_tq1, fake_x), trace_joint=False)[0]

        # inputs: token, tq, x
        # return: token, x_sin, x_cos, tq
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_6 = None
    getitem_8 = with_effects_4[0]
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    with_effects_5 = torch.ops.higher_order.with_effects(getitem_8, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_8 = None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    return (getitem_10, sub, add, arg1_1)""",
        )

    def test_export_inplace_custom_op(self):
        # [FUNCTION DOC] Exports (non strict, pre-dispatch) a model that calls the queue push op,
        # which mutates the input queue, and returns the queue. Checks the unlifted code, the
        # graph module code and the printed graph, where the push call is wrapped with effects
        # and the queue is returned as is.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that pushes a tensor into a queue input.
            def forward(self, tq: torch.ScriptObject, x: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Pushes the input tensor onto the queue with the op and
                # returns the queue.
                torch.ops._TorchScriptTesting.queue_push(tq, x)
                return tq

        mod = Model()
        ep = self._test_export_same_as_eager(
            mod,
            (_empty_tensor_queue(), torch.randn(3, 3)),
            strict=False,
            pre_dispatch=True,
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, tq, x):
    tq, x, = fx_pytree.tree_flatten_spec(([tq, x], {}), self._in_spec)
    _guards_fn = self._guards_fn(tq, x);  _guards_fn = None
    queue_push_default = torch.ops._TorchScriptTesting.queue_push.default(tq, x);  x = queue_push_default = None
    return pytree.tree_unflatten((tq,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, tq, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.queue_push.default, tq, x);  token = x = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, tq)""",
        )
        self.assertExpectedInline(
            str(ep.graph_module.graph).strip(),
            """\
graph():
    %token : [num_users=1] = placeholder[target=token]
    %tq : [num_users=2] = placeholder[target=tq]
    %x : [num_users=1] = placeholder[target=x]
    %with_effects : [num_users=1] = call_function[target=torch.ops.higher_order.with_effects](args = (%token, _TorchScriptTesting.queue_push.default, %tq, %x), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%with_effects, 0), kwargs = {})
    return (getitem, tq)""",
        )

    def test_deepcopy(self):
        # [FUNCTION DOC] Deep copies a tensor queue script object at three points while pushing
        # tensors and checks each copy is independent (sizes zero, two and three). Does the same
        # for a Foo object with an increment method and checks each copy keeps its own state
        # (add results three, five, seven).
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq_0 = copy.deepcopy(tq)
        tq.push(torch.zeros(2, 2))
        tq.push(torch.ones(2, 2))
        tq_1 = copy.deepcopy(tq)
        tq.push(torch.ones(2, 2) * 2)
        self.assertEqual(tq_0.size(), 0)
        self.assertEqual(tq_1.size(), 2)
        self.assertEqual(tq.size(), 3)

        foo = torch.classes._TorchScriptTesting._Foo(1, 2)
        foo_0 = copy.deepcopy(foo)
        foo.increment(1)
        foo_1 = copy.deepcopy(foo)
        foo.increment(1)
        self.assertEqual(foo_0.add(1), 3)
        self.assertEqual(foo_1.add(1), 5)
        self.assertEqual(foo.add(1), 7)


class TestCompileTorchbind(TestCase):
    # [CLASS DOC] Test case for compiling (dynamo with eager, aot_eager and inductor backends)
    # functions that take, hold or close over TorchBind script objects. setUp registers fake
    # classes for the tensor queue and for a flatten-with-tensor-op class and resets dynamo;
    # tearDown only resets dynamo (the fake classes are not deregistered).
    def setUp(self):
        # [FUNCTION DOC] HELPER: Initializes torchbind test implementations, registers fake
        # classes for the tensor queue (list based push, pop, size) and for the
        # flatten-with-tensor-op class (get returning a stored tensor), then resets dynamo state.
        super().setUp()
        init_torchbind_implementations()

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            # [CLASS DOC] HELPER: Fake tensor queue backed by a Python list (no counters).
            def __init__(self, queue):
                # [FUNCTION DOC] HELPER: Stores the given list as the queue.
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from the flattened
                # state, a sequence of name and value pairs turned into keyword arguments.
                return cls(**dict(flattened_ctx))

            def push(self, x):
                # [FUNCTION DOC] HELPER: Appends the tensor to the end of the queue.
                self.queue.append(x)

            def pop(self):
                # [FUNCTION DOC] HELPER: Removes and returns the first element of the queue.
                return self.queue.pop(0)

            def size(self):
                # [FUNCTION DOC] HELPER: Returns the number of queued elements.
                return len(self.queue)

        @torch._library.register_fake_class("_TorchScriptTesting::_FlattenWithTensorOp")
        class FakeFlatten:
            # [CLASS DOC] HELPER: Fake of the class that wraps one tensor and exposes it through a
            # get method.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Stores the wrapped tensor.
                self.t = t

            def get(self):
                # [FUNCTION DOC] HELPER: Returns the stored tensor.
                return self.t

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from the flattened
                # state, a sequence of name and value pairs turned into keyword arguments.
                return cls(**dict(flattened_ctx))

        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state after each test; does not call the base class
        # tearDown.
        torch._dynamo.reset()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_script_object_input(self, backend):
        # [FUNCTION DOC] Compiles (parametrized backend; eager uses a graph recording backend) a
        # model that pushes, pops and sizes a tensor queue input, runs eager on a second queue and
        # compares results, queue sizes and next popped tensors. For the eager backend checks the
        # captured graph code uses the call torchbind op and does not return the queue as an
        # output, since dynamo knows inputs need not be returned.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a flag attribute (unused) that pushes and pops a
            # tensor queue input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the unused fake check flag.
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Pushes cosine and sine of the input, pops one tensor,
                # subtracts the queue size from it and returns that result and the queue.
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        mod = Model()
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq2 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq3 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq4 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.randn(2, 3)
        ret = torch.compile(mod, backend=backend)(tq1, x)
        eager_ret = mod(tq2, x)
        _assertEqualSkipScriptObject(self, ret, eager_ret)
        self.assertEqual(ret[1].size(), eager_ret[1].size())
        self.assertEqual(ret[1].pop(), eager_ret[1].pop())
        # Note that dynamo captured graph
        # does not return L_tq_ as output. This is because it's able
        # to detect that L_tq_ is an input therefore don't return
        # it as graph output. Related logic is in dynamo/codegen.py
        if backend == "eager":
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_tq_ : torch.ScriptObject, L_x_ : torch.Tensor):
        l_tq_ = L_tq_
        l_x_ = L_x_
        cos = l_x_.cos()
        call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', cos);  cos = None
        sin = l_x_.sin();  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', sin);  sin = None
        call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'pop')
        call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
        x_sin = call_torchbind_2 - 1;  call_torchbind_2 = None
        return (x_sin,)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_script_object_input_guards(self, backend):
        # [FUNCTION DOC] Compiles the same queue model repeatedly with one counting backend and
        # checks recompilation counts: an empty queue compiles once; a longer queue recompiles;
        # a one-tensor queue recompiles; another queue with the same tensor shape does not;
        # a tensor requiring grad recompiles; a different dtype recompiles (and the result then
        # matches eager). Verifies guards cover queue length and queued tensor properties.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that pushes and pops a tensor queue input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the unused fake check flag.
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Pushes cosine and sine of the input, pops one tensor,
                # subtracts the queue size from it and returns that result and the queue.
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        mod = Model()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
        x = torch.randn(2, 3)

        tq1 = _empty_tensor_queue()
        torch.compile(mod, backend=cnt)(tq1, x)
        self.assertEqual(cnt.frame_count, 1)

        tq2 = _empty_tensor_queue()
        for _ in range(10):
            tq2.push(torch.randn(4, 5, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq2, x)
        # Queue length change causes re-compile
        self.assertEqual(cnt.frame_count, 2)

        tq3 = _empty_tensor_queue()
        tq3.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq3, x)
        # Tensor in queue changes shape causes re-compile
        self.assertEqual(cnt.frame_count, 3)

        tq4 = _empty_tensor_queue()
        tq4.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq4, x)
        # No recompile
        self.assertEqual(cnt.frame_count, 3)

        tq5 = _empty_tensor_queue()
        tq5.push(torch.randn(2, 3, requires_grad=True))
        torch.compile(mod, backend=cnt)(tq5, x)
        # Tensor in queue changes dispatch key causes re-compile
        self.assertEqual(cnt.frame_count, 4)

        tq6 = _empty_tensor_queue()
        tq6_ref = _empty_tensor_queue()
        queued_tensor = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        tq6.push(queued_tensor)
        tq6_ref.push(queued_tensor.detach().clone().requires_grad_(True))
        compiled_out, _ = torch.compile(mod, backend=cnt)(tq6, x)
        eager_out, _ = mod(tq6_ref, x)
        # Tensor in queue changes dtype causes re-compile
        self.assertEqual(cnt.frame_count, 5)
        self.assertEqual(compiled_out, eager_out)

    def test_compile_script_object_input_automatic_dynamic_shape(self):
        # [FUNCTION DOC] Compiles the queue model with a counting backend for queues holding one
        # tensor of differing second dimension: the second shape triggers a recompile (making the
        # dimension dynamic) and the third shape reuses it, so the frame count ends at two.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that pushes and pops a tensor queue input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the unused fake check flag.
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Pushes cosine and sine of the input, pops one tensor,
                # subtracts the queue size from it and returns that result and the queue.
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        mod = Model()
        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 3)

        tq1 = _empty_tensor_queue()
        tq1.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq1, x)
        self.assertEqual(cnt.frame_count, 1)

        tq2 = _empty_tensor_queue()
        # make first tensor's second dim dynamic
        tq2.push(torch.randn(2, 4, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq2, x)
        self.assertEqual(cnt.frame_count, 2)

        tq3 = _empty_tensor_queue()
        tq3.push(torch.randn(2, 5, requires_grad=False))
        # should have no-recompilation
        torch.compile(mod, backend=cnt)(tq3, x)
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_error_on_input_aliasing_contents(self, backend):
        # [FUNCTION DOC] Checks compiling a model whose queue already contains the same tensor
        # that is also passed as the input raises a RuntimeError mentioning aliasing. Parametrized
        # over backends; the eager one uses a graph recording backend.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns sine of the input and cosine of a tensor
            # popped from the queue.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the unused fake check flag.
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Returns the input's sine and the cosine of the popped
                # tensor.
                return x.sin(), tq.pop().cos()

        x = torch.randn(2, 3)
        mod = Model()

        tq1 = _empty_tensor_queue()
        tq1.push(x)
        with self.assertRaisesRegex(RuntimeError, "is aliasing"):
            torch.compile(mod, backend=backend)(tq1, x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_setattr(self, backend):
        # [FUNCTION DOC] Checks compiling a function that sets a new attribute on a script object
        # raises a RuntimeError about a weird method call on the TorchScript object.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def setattr_f(tq):
            # [FUNCTION DOC] HELPER: Sets an attribute on the object and returns it.
            tq.a = 1
            return tq

        with self.assertRaisesRegex(
            RuntimeError, "Weird method call on TorchScript object"
        ):
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_missing_attr(self, backend):
        # [FUNCTION DOC] Checks compiling a function that reads an attribute not defined on the
        # script object raises a RuntimeError saying the fake script object is missing a method
        # implementation.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def setattr_f(tq):
            # [FUNCTION DOC] HELPER: Returns an undefined attribute of the object.
            return tq._not_defined_attr

        with self.assertRaisesRegex(
            RuntimeError, "FakeScriptObject missing method implementation"
        ):
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_body_aliasing_contents(self, backend):
        # [FUNCTION DOC] Compiles a function that pushes a view and a permuted version of the
        # input onto a queue and returns values depending on the queue size, and compares the
        # result with eager. For the eager backend (when not already compiling) also checks the
        # captured graph code, where the size is constant folded.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def f(tq, x):
            # [FUNCTION DOC] HELPER: Pushes a flattened view and a transposed view of the input to
            # the queue, then returns the first view minus the queue size, the second plus the
            # queue size, and the queue.
            x1 = x.view(-1)
            x2 = x.permute(1, 0)
            tq.push(x1)
            tq.push(x2)
            return x1 - tq.size(), x2 + tq.size(), tq

        x = torch.randn(2, 3)
        _assertEqualScriptObject(
            self,
            f(_empty_tensor_queue(), x),
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x),
        )
        if not torch._dynamo.is_compiling() and backend == "eager":
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_x_ : torch.Tensor, L_tq_ : torch.ScriptObject):
    l_x_ = L_x_
    l_tq_ = L_tq_
    x1 = l_x_.view(-1)
    x2 = l_x_.permute(1, 0);  l_x_ = None
    call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x1)
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x2)
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'size')
    sub = x1 - 2;  x1 = None
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
    add = x2 + 2;  x2 = None
    return (sub, add)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_tensor_op_in_tensor_flatten(self, backend):
        # [FUNCTION DOC] Uses a script object whose get method returns a stored tensor: runs eager,
        # compiles with full graph on the parametrized backend, and exports non strictly with
        # decompositions. Checks the exported graph code and that eager, compiled and exported
        # outputs all agree.
        test_obj = torch.classes._TorchScriptTesting._FlattenWithTensorOp(
            torch.randn(3, 2)
        )

        class TestMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module combining a tensor from the object with the input and the
            # first dimension size of that tensor.
            def forward(self, obj, x):
                # [FUNCTION DOC] HELPER: Returns the object's tensor plus the input plus the
                # first dimension size of the object's tensor.
                return obj.get() + x + obj.get().size(0)

        mod = TestMod()

        x = torch.randn(3, 1)
        eager_out = mod(test_obj, x)
        compiled_out = torch.compile(mod, backend=backend, fullgraph=True)(test_obj, x)
        ep = torch.export.export(mod, (test_obj, x), strict=False).run_decompositions(
            {}
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj, 'get');  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, x);  getitem_1 = x = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.call_torchbind, obj, 'get');  getitem = obj = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(add, 3);  add = None
    return (getitem_2, add_1)""",
        )
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out, ep.module()(test_obj, x))

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_error_on_non_fakified_method(self, backend):
        # [FUNCTION DOC] Checks compiling a function that calls a queue method present on the real
        # object but absent from the fake class raises a RuntimeError about a missing method
        # implementation on the fake script object.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def f(tq, x):
            # [FUNCTION DOC] HELPER: Pushes two views of the input to the queue and then calls the
            # clone queue method, which the fake does not implement.
            x1 = x.view(-1)
            x2 = x.permute(1, 0)
            tq.push(x1)
            tq.push(x2)
            # though real tensor queue implemented a method clone_queue,
            # The fakified version doesn't.
            flat_obj = tq.clone_queue()
            return flat_obj

        x = torch.randn(2, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeScriptObject missing method implementation"
        ):
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_as_hop_input(self, backend):
        # [FUNCTION DOC] Checks that a script object can be passed as an input to the wrap higher
        # order operator inside a compiled function, comparing the compiled result with eager.
        def f(tq, x):
            # [FUNCTION DOC] HELPER: Calls the inner function through the wrap higher order op and
            # returns its result.
            def fn(tq, x):
                # [FUNCTION DOC] HELPER: Pushes the tensor onto the queue and returns its sine.
                tq.push(x)
                return x.sin()

            return wrap(fn, tq, x)

        x = torch.randn(2, 3)
        _assertEqualScriptObject(
            self,
            f(_empty_tensor_queue(), x),
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x),
        )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_closure(self, backend):
        # [FUNCTION DOC] Compiles a function that uses a queue captured from the enclosing scope
        # through a nested function and compares with eager.
        # KNOWN ISSUE: the backend parameter is ignored; the function is always compiled with the
        # eager backend, so the three parametrizations test the same thing.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the nested function, then pops from the captured queue
            # and returns the popped tensor and the queue.
            def inner_f(x):
                # [FUNCTION DOC] HELPER: Pushes the sine of the input onto the captured queue.
                tq.push(x.sin())

            inner_f(x)
            return tq.pop(), tq

        opt_f = torch.compile(f, backend="eager")

        tq = _empty_tensor_queue()
        x = torch.randn(3, 2)
        _assertEqualScriptObject(self, f(x), opt_f(x))

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_global_obj(self, backend):
        # [FUNCTION DOC] Sets a module level global to a new empty tensor queue (side effect:
        # the global remains defined after the test), compiles a function that pushes to and pops
        # from it, and compares the compiled result with eager.
        global _TENSOR_QUEUE_GLOBAL_TEST
        _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()

        def f(x):
            # [FUNCTION DOC] HELPER: Pushes the input's sine onto the global queue and returns the
            # popped tensor and the queue.
            _TENSOR_QUEUE_GLOBAL_TEST.push(x.sin())
            return _TENSOR_QUEUE_GLOBAL_TEST.pop(), _TENSOR_QUEUE_GLOBAL_TEST

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(3, 2)
        eager_ret = f(x)
        opt_ret = opt_f(x)
        _assertEqualScriptObject(self, eager_ret, opt_ret)

    def test_compile_obj_graph_breaks(self):
        # [FUNCTION DOC] Compiles a function with explicit graph breaks between queue operations
        # using a counting backend, compares with eager, and checks four compiled frames result.
        cnt = torch._dynamo.testing.CompileCounter()

        def f(tq, x):
            # [FUNCTION DOC] HELPER: Pushes twice, pops, pushes a value using the queue size, then
            # pushes another, with a graph break between each step; returns the input, a popped
            # tensor and the queue.
            tq.push(x.sin())
            tq.push(x.sin())
            torch._dynamo.graph_break()
            tq.pop()
            torch._dynamo.graph_break()
            tq.push(x.cos() + tq.size())
            torch._dynamo.graph_break()
            tq.push(x.cos() - tq.size())
            return x, tq.pop(), tq

        opt_f = torch.compile(f, backend=cnt)
        x = torch.randn(3, 2)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )
        self.assertEqual(cnt.frame_count, 4)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_attributes(self, backend):
        # [FUNCTION DOC] Compiles a module that stores a queue as an attribute, pushes and pops
        # in forward, and compares with a fresh eager module. For the eager backend checks a
        # single graph was captured and the queue is lifted as a graph input rather than kept as
        # an attribute.
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning an empty tensor queue attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates an empty tensor queue attribute.
                super().__init__()
                self.tq = _empty_tensor_queue()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Pushes the input onto the queue and returns the popped
                # tensor.
                self.tq.push(x)
                return self.tq.pop()

        x = torch.randn(2, 3)
        opt_f = torch.compile(Model(), backend=backend)
        _assertEqualScriptObject(self, Model()(x), opt_f(x))
        if backend == "eager":
            self.assertEqual(len(backend.graphs), 1)
            # lifted as input. In the future, we would want to cosolidate this
            # with non-strict behavior, where they're set as attributes.
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_self_tq : torch.ScriptObject, L_x_ : torch.Tensor):
        l_self_tq = L_self_tq
        l_x_ = L_x_
        call_torchbind = torch.ops.higher_order.call_torchbind(l_self_tq, 'push', l_x_);  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_self_tq, 'pop');  l_self_tq = None
        return (call_torchbind_1,)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_torchbind_op(self, backend):
        # [FUNCTION DOC] Compiles a function calling the queue push and pop torchbind ops with
        # the queue input, then queue methods, and compares the compiled result with eager on
        # fresh queues. Parametrized over backends.
        def f(tq, x):
            # [FUNCTION DOC] HELPER: Calls push op twice (cosine, cosine plus one), pop op, push op
            # with sine; then returns a popped tensor, another popped tensor plus the size, and
            # the queue.
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
            torch.ops._TorchScriptTesting.queue_pop(tq)
            torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
            return tq.pop(), tq.pop() + tq.size(), tq

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(2)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )

    @requires_cuda_and_triton
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_torchbind_op_with_autocast(self, backend, device):
        # [FUNCTION DOC] Same as the torchbind op compile test but the function runs under an
        # autocast context for the parametrized device (cpu or cuda); requires cuda and triton.
        # Compares compiled and eager results.
        def f(tq, x):
            # [FUNCTION DOC] HELPER: Under autocast, calls push op twice, pop op and push op, then
            # returns a popped tensor, another popped tensor plus the size, and the queue.
            with torch.autocast(device_type=device):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
                torch.ops._TorchScriptTesting.queue_pop(tq)
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
            return tq.pop(), tq.pop() + tq.size(), tq

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(2, device=device)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )

    @requires_cuda_and_triton
    @parametrize("device", ["cpu", "cuda"])
    def test_export_obj_torchbind_op_with_autocast(self, device):
        # [FUNCTION DOC] Exports (default settings) a module that calls queue ops under autocast
        # for the parametrized device (cpu or cuda; requires cuda and triton) and compares the
        # exported module with eager on fresh queues.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using queue ops under an autocast context.
            def forward(self, x, tq):
                # [FUNCTION DOC] HELPER: Under autocast, calls push op twice, pop op and push op,
                # then returns a popped tensor, another popped tensor plus the size, and the queue.
                with torch.autocast(device_type=device):
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
                    torch.ops._TorchScriptTesting.queue_pop(tq)
                    torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                return tq.pop(), tq.pop() + tq.size(), tq

        x = torch.randn(2, device=device)
        args = (x,)
        mod = Mod()
        ep = torch.export.export(mod, (x, _empty_tensor_queue()))
        _assertEqualScriptObject(
            self, ep.module()(x, _empty_tensor_queue()), mod(x, _empty_tensor_queue())
        )


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestRegisterFakeClass(TestCase):
    # [CLASS DOC] Test case for the fake class registration API and its error checks; skipped under
    # dynamo. setUp initializes torchbind test implementations; tearDown clears the global fake
    # class registry.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Initializes torchbind test implementations.
        super().setUp()
        init_torchbind_implementations()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Clears the global fake class registry (does not call the base
        # class tearDown).
        torch._library.fake_class_registry.global_fake_class_registry.clear()

    def test_register_fake_class_no_torch_bind_class(self):
        # [FUNCTION DOC] Checks registering a fake class under a name that is not a real
        # torchbind class raises a RuntimeError about failing to instantiate the class.
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate class"):

            @torch._library.register_fake_class("_TorchScriptTesting::NOT_A_VALID_NAME")
            class Invalid:
                # [CLASS DOC] HELPER: Empty class used only as an invalid registration target.
                pass

    def test_register_fake_class_no_from_real(self):
        # [FUNCTION DOC] Checks registering a fake class lacking the unflatten class method raises
        # a RuntimeError asking to define it.
        with self.assertRaisesRegex(
            RuntimeError, "define a classmethod __obj_unflatten__"
        ):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class InvalidFakeFoo:
                # [CLASS DOC] HELPER: Fake class missing the unflatten class method.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Does nothing.
                    pass

    def test_register_fake_class_from_real_not_classmethod(self):
        # [FUNCTION DOC] Checks registering a fake class whose unflatten method is a plain method
        # rather than a class method raises a RuntimeError saying it is not a classmethod.
        with self.assertRaisesRegex(RuntimeError, "is not a classmethod"):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                # [CLASS DOC] HELPER: Fake Foo with a non-class-method unflatten (the error
                # under test).
                def __init__(self, x, y):
                    # [FUNCTION DOC] HELPER: Stores the two integer fields.
                    self.x = x
                    self.y = y

                def __obj_unflatten__(cls, flattend_foo):
                    # [FUNCTION DOC] HELPER: Intended unflatten; builds an instance from the
                    # flattened state (a sequence of pairs as keyword arguments) but lacks the
                    # class method decorator, which is the error under test.
                    return cls(**dict(flattend_foo))

    def test_register_fake_class_valid(self):
        # [FUNCTION DOC] Checks registering a valid fake class (stores two integers, has a class
        # method that unflattens) under the Foo name by passing the class directly succeeds
        # without raising.
        class FakeFoo:
            # [CLASS DOC] HELPER: Valid fake Foo with a class method unflatten.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the two integer fields.
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flattend_foo):
                # [FUNCTION DOC] HELPER: Class method rebuilding an instance from the flattened
                # state, a sequence of name and value pairs turned into keyword arguments.
                return cls(**dict(flattend_foo))

        torch._library.register_fake_class("_TorchScriptTesting::_Foo", FakeFoo)


instantiate_parametrized_tests(TestExportTorchbind)
instantiate_parametrized_tests(TestCompileTorchbind)

if __name__ == "__main__":
    run_tests()
