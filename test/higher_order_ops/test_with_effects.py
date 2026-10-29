# Owner(s): ["module: functorch"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the with_effects higher order op and effectful operator support. Shows how
# AOTAutograd threads ordering tokens through effectful ops (the aten print op, torchbind custom
# ops, user custom ops registered with an effect type) in forward and backward graphs, with and
# without unlifting of effect tokens, plus Dynamo/Inductor compilation, FX dead code elimination,
# torch.export with invoke_subgraph, and functional tensor mode. Helper functions build
# forward/backward graphs. Import-time effects: none beyond imports (torchbind implementations are
# initialized in setUp). Many tests need CUDA, SM80+ and triton.
import unittest
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import (
    aot_function,
    default_decompositions,
    min_cut_rematerialization_partition,
    nop,
)
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_export_module
from torch._guards import tracing, TracingContext
from torch._higher_order_ops.effects import (
    _EffectType,
    _get_effect,
    _register_effectful_op,
    with_effects,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import has_side_effect
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM70OrLater, SM80OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

from torch.testing._internal.two_tensor import TwoTensor


def extract_graph(fx_g, _, graph_cell):
    # [FUNCTION DOC] HELPER: Compiler callback for aot_function. Stores the received graph module
    # into the first slot of the given single-element list and returns the graph module unchanged;
    # the second positional argument (example inputs) is ignored.
    graph_cell[0] = fx_g
    return fx_g


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    # [FUNCTION DOC] HELPER: Runs the function through AOT compilation and returns the forward and
    # backward graph modules. Records graphs via callbacks into one-element lists. If any input
    # tensor requires grad, the output is summed and backward is run so the backward graph is
    # captured; otherwise the backward compiler is a no-op and the backward entry is None. Uses the
    # given partitioner (default min-cut rematerialization), the default decompositions, and the
    # dynamic flag. Returns a pair (forward graph, backward graph or None).
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    requires_grad = False

    def fn_req_grad(t):
        # [FUNCTION DOC] HELPER: Tree-map callback that sets the enclosing flag when any tensor
        # requires grad, and returns the tensor unchanged.
        nonlocal requires_grad
        requires_grad = requires_grad or t.requires_grad
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, fn_req_grad, inps)

    out = aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=(
            partial(extract_graph, graph_cell=bw_graph_cell) if requires_grad else nop
        ),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps)

    if requires_grad:
        out.sum().backward()

    return (fw_graph_cell[0], bw_graph_cell[0])


def make_inputs_non_leaves(inps):
    # [FUNCTION DOC] HELPER: Returns the same nested structure with every tensor replaced by that
    # tensor plus one, which makes leaf inputs into non-leaf tensors in the autograd graph.
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.add(1), inps)


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    # [CLASS DOC] Tests for the with_effects higher order op. The class is skipped when Dynamo is
    # unsupported. setUp initializes the torchbind test implementations. Most tests export or
    # compile small functions with effectful ops and compare exact generated graph text.
    def setUp(self):
        # [FUNCTION DOC] Runs base setup and initializes the torchbind test implementations.
        super().setUp()
        init_torchbind_implementations()

    def test_print(self):
        # [FUNCTION DOC] Traces a module with two aten print calls: with plain tracing both stay
        # direct (checked with FileCheck counting exactly two); with AOTAutograd export each is
        # wrapped in with_effects chained through a token (exact code checked; one input and one
        # output token). With the unlift tokens config enabled, tokens are created inside the graph
        # and sunk at the end.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that prints a message, doubles the input, prints again,
            # returns a tuple.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Print, double the input, print, and return the result
                # in a one-tuple.
                torch.ops.aten._print("moo")
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        FileCheck().check_count("torch.ops.aten._print.default", 2, exactly=True).run(
            gm.code
        )

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        with torch._functorch.config.patch(unlift_effect_tokens=True):
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            self.assertExpectedInline(
                str(gm.code).strip(),
                """\
def forward(self, arg1_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.aten._print.default, 'moo');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem_2]);  getitem_2 = _sink_tokens_default = None
    return (add,)""",
            )

    def test_torchbind_custom_op(self):
        # [FUNCTION DOC] Exports under torchbind tracing a module that calls a custom op taking
        # a torchbind object and a tensor. Asserts the exact graph code, where the call is
        # wrapped in with_effects with the torchbind object as an attribute, and one input and
        # one output token.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a torchbind Foo object (10, 20) and adding the
            # custom op result on the input and object to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the torchbind Foo object attribute.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a one-tuple of the input plus the output of the
                # torchbind custom op.
                return (x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x),)

        with enable_torchbind_tracing():
            gm, gs = aot_export_module(M(), (torch.ones(2, 3),), trace_joint=False)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _torchbind_obj0 = self._torchbind_obj0
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, _torchbind_obj0, arg1_1);  arg0_1 = _torchbind_obj0 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_print_with_buffer_mutations(self):
        # [FUNCTION DOC] Exports a module that prints, mutates a buffer in place and prints again.
        # Asserts exact graph code (effect-wrapped prints, buffer mutation as functional add), plus
        # one input token, one output token and one mutated buffer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of ones; forward prints, adds the doubled
            # input into the buffer in place, adds the input to the buffer and prints again;
            # returns a one-tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the buffer of three ones.
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the print, buffer mutation, sum and print sequence
                # from the class.
                torch.ops.aten._print("moo")
                res = x + x
                self.buf.add_(res)
                res = self.buf + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg2_1, arg2_1)
    add_1 = torch.ops.aten.add.Tensor(arg1_1, add);  arg1_1 = add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add_1, add_2)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.buffers_to_mutate), 1)

    def test_print_with_input_mutations(self):
        # [FUNCTION DOC] Exports a module that prints, mutates its input in place and prints again.
        # Asserts one input token, one output token and one mutated user input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that prints, mutates its input in place with its
            # double, prints again.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the parent constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Prints, mutates the input in place, recomputes a double,
                # prints, and returns it in a one-tuple.
                torch.ops.aten._print("moo")
                res = x + x
                x.add_(res)
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.user_inputs_to_mutate), 1)

    def test_alias_op(self):
        # [FUNCTION DOC] Checks that with_effects rejects an aliasing operator: tracing with
        # make_fx a function that applies the in-place absolute op through with_effects must raise
        # AssertionError whose message says ops with aliasing are not supported.
        def f(token, x):
            # [FUNCTION DOC] HELPER: Calls with_effects with a token and the in-place absolute op
            # on x and returns the new token and the output.
            token, out = with_effects(token, torch.ops.aten.absolute_.default, x)
            return token, out

        with self.assertRaisesRegex(
            AssertionError, r"Ops with aliasing is not supported"
        ):
            make_fx(f)(torch.tensor([]), torch.tensor(4))

    def test_compile_aot_eager(self):
        # [FUNCTION DOC] Compiles a function with two prints using the aot_eager backend and
        # asserts the result is close to the eager result.
        def f(x):
            # [FUNCTION DOC] HELPER: Print, double the input, print, return the doubled tensor.
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM70OrLater, "triton")
    def test_compile_inductor(self):
        # [FUNCTION DOC] Same as the aot_eager compile test but with the inductor backend; skipped
        # on Windows and when GPU compute capability is below 7.0.
        def f(x):
            # [FUNCTION DOC] HELPER: Print, double the input, print, return the doubled tensor.
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipIfNoDynamoSupport
    def test_compile_inductor_external_op_return_none(self):
        # [FUNCTION DOC] Registers a custom op in a scoped library that mutates its output argument
        # and returns nothing, then compiles a function calling it with inductor and checks the
        # result is close to eager. Skipped on Windows or without Dynamo support.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::inplace_add",
                "(Tensor input, Tensor(a!) output) -> ()",
                lib=lib,
            )

            def inplace_add(input: torch.Tensor, output: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Custom op implementation: raises AssertionError if the two
                # tensors are on different devices, otherwise adds the input into the output in
                # place; returns nothing.
                if input.device != output.device:
                    raise AssertionError(
                        f"Expected input.device == output.device, "
                        f"got {input.device} vs {output.device}"
                    )
                output.add_(input)

            lib.impl("inplace_add", inplace_add, "CompositeExplicitAutograd")

            def f(x):
                # [FUNCTION DOC] HELPER: Makes a zero tensor of size three, calls the in-place
                # custom op with the input and that tensor, and returns it.
                out = torch.empty(3)
                out = torch.zeros_like(out)
                torch.ops.mylib.inplace_add(x, out)
                return out

            inputs = (torch.randn(3),)

            res = torch.compile(f, backend="inductor")(*inputs)
            self.assertTrue(torch.allclose(res, f(*inputs)))

    def test_compile_aot_eager_requires_grad(self):
        # [FUNCTION DOC] Compiles a printing function with aot_eager on an input requiring grad,
        # checks the result against eager, and runs backward on the sum.
        def f(x):
            # [FUNCTION DOC] HELPER: Print, double the input, print, return the doubled tensor.
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3, requires_grad=True),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

        res.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "triton")
    @skipIfNoDynamoSupport
    def test_register_effectful_custom_op(self):
        # [FUNCTION DOC] Registers a custom op that records tensors into a dictionary by prefix,
        # marks it as an ordered effect, and attaches forward hooks to every submodule of a small
        # model via a breadth traversal. Compiles with inductor on CUDA, checks outputs, backward
        # and gradient, and that exactly two records (for the model and its linear layer) were
        # made. Needs CUDA and SM80.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            # global variable to store the recorded tensor and prefix.
            recorded_dict = {}

            # Pytorch custom op implementation
            @torch.library.custom_op("mylib::record_scalar_tensor", mutates_args=())
            def record_scalar_tensor(x: torch.Tensor, prefix: str) -> None:
                # [FUNCTION DOC] HELPER: Custom op body: stores a clone of the tensor in the shared
                # dictionary under the given prefix; returns nothing.
                recorded_dict[prefix] = x.clone()
                return

            # Meta function of the custom op
            @record_scalar_tensor.register_fake
            def record_scalar_tensor_meta(x, prefix):
                # [FUNCTION DOC] HELPER: Fake implementation of the recording op; returns nothing.
                return

            record_scalar_tensor.register_effect(_EffectType.ORDERED)

            self.assertEqual(_get_effect(record_scalar_tensor), _EffectType.ORDERED)

            my_config = {}
            my_config["MockModule"] = "mean"
            my_config["MockModule.linear"] = "mean"
            my_config["MockModule.relu"] = "mean"

            class MyLinear(torch.nn.Module):
                # [CLASS DOC] HELPER: Linear layer with randomly initialized weight
                # and bias parameters.
                def __init__(self, in_features, out_features):
                    # [FUNCTION DOC] HELPER: Creates the weight and bias parameters of
                    # the requested sizes.
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.randn(out_features, in_features), requires_grad=True
                    )
                    self.bias = torch.nn.Parameter(
                        torch.randn(out_features), requires_grad=True
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the functional linear transform with
                    # the stored parameters.
                    return torch.nn.functional.linear(x, self.weight, self.bias)

            class MockModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Model with a 10 by 10 custom linear layer and a trainable 10
                # by 10 buffer; forward applies ReLU to the linear output plus the buffer.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer and registers
                    # the random buffer.
                    super().__init__()
                    self.linear = MyLinear(10, 10)
                    self.register_buffer(
                        "buf0", torch.randn(10, 10, requires_grad=True)
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the ReLU of the linear output plus the buffer.
                    return torch.nn.functional.relu(self.linear(x) + self.buf0)

            def forward_hook(
                module: torch.nn.Module,
                inputs: torch.Tensor,
                output: torch.Tensor,
                prefix: str,
                aggregate_method: str,
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Forward hook: records an aggregate of the output (mean,
                # max, or sum for any other method name) through the effectful recording op under
                # the given prefix, then returns the output unchanged.
                if aggregate_method == "mean":
                    torch.ops.mylib.record_scalar_tensor(output.mean(), prefix)
                elif aggregate_method == "max":
                    torch.ops.mylib.record_scalar_tensor(output.max(), prefix)
                else:
                    # demo purpose, using "min"
                    torch.ops.mylib.record_scalar_tensor(output.sum(), prefix)
                return output

            def add_hooks(module, config):
                # [FUNCTION DOC] HELPER: Walks the module tree breadth-wise from the root; for each
                # module picks the aggregate method from the config (default mean), builds a prefix
                # from the dotted name and method, registers the forward hook, and collects
                # handles. Returns the list of handles.
                handles: list[RemovableHandle] = []
                q = deque([(module.__class__.__name__, module)])
                while q:
                    name, m = q.pop()
                    children = [(name + "." + n, y) for (n, y) in m.named_children()]
                    q.extend(children)
                    aggregate_method = config.get(name, "mean")
                    prefix = name + ":" + aggregate_method
                    handle = m.register_forward_hook(
                        partial(
                            forward_hook,
                            prefix=prefix,
                            aggregate_method=aggregate_method,
                        )
                    )
                    if handle:
                        handles.append(handle)
                return handles

            x = torch.randn(10, 10, device="cuda")
            mod = MockModule().to("cuda")

            add_hooks(mod, my_config)

            opt_mod = torch.compile(backend="inductor")(mod)
            y = opt_mod(x)

            self.assertTrue(torch.allclose(y, mod(x)))
            # Ensure it works well with backward
            y.sum().backward()
            # Ensure the grad is existing
            self.assertTrue(isinstance(opt_mod.linear.weight.grad, torch.Tensor))

            self.assertEqual(len(recorded_dict), 2)
            self.assertTrue("MockModule.linear:mean" in recorded_dict)
            self.assertTrue("MockModule:mean" in recorded_dict)

    @skipIfNoDynamoSupport
    def test_effectful_custom_op_with_subclasses(self):
        # [FUNCTION DOC] Defines two ordered effectful ops with CPU/CUDA/Meta implementations that
        # count forward and backward calls, with an autograd rule whose backward calls the second
        # op. Checks call counts and results against eager for dense and TwoTensor subclass inputs,
        # without and with grad, using aot_eager and the default backend. Also checks exact forward
        # and backward graph text with tokens for subclass inputs. Skipped without Dynamo support.
        with torch.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("zoo(Tensor x) -> Tensor")
            lib.define("zoo2(Tensor x) -> Tensor")

            d = {"fw": 0, "bw": 0}

            def reset_counter():
                # [FUNCTION DOC] HELPER: Sets both the forward and backward call counters to zero.
                d["fw"] = 0
                d["bw"] = 0

            def assert_counter(fw, bw):
                # [FUNCTION DOC] HELPER: Asserts the forward and backward call counters equal
                # the expected values.
                self.assertEqual(d["fw"], fw)
                self.assertEqual(d["bw"], bw)

            def foo_impl(a):
                # [FUNCTION DOC] HELPER: Implementation of the first op: increments the forward
                # counter and returns twice a clone of the input.
                d["fw"] = d["fw"] + 1
                return 2 * a.clone()

            def foo_meta(a):
                # [FUNCTION DOC] HELPER: Meta implementation for the first op: returns a clone
                # of the input.
                return a.clone()

            def foo2_impl(x):
                # [FUNCTION DOC] HELPER: Implementation of the second op: increments the backward
                # counter and returns a clone of the input.
                d["bw"] = d["bw"] + 1
                return x.clone()

            def foo2_meta(a):
                # [FUNCTION DOC] HELPER: Meta implementation for the second op: returns a clone
                # of the input.
                return a.clone()

            for backend in ["CPU", "CUDA"]:
                lib.impl("zoo", foo_impl, backend)
                lib.impl("zoo2", foo2_impl, backend)
            lib.impl("zoo", foo_meta, "Meta")
            lib.impl("zoo2", foo2_meta, "Meta")

            def foo_bwd(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward rule for the first op: calls the second op on the
                # gradient (for its counting side effect) and returns a clone of the gradient.
                torch.ops._mylib.zoo2(grad)
                return grad.clone()

            torch.library.register_autograd("_mylib::zoo", foo_bwd, lib=lib)

            torch.library._register_effectful_op(
                torch.ops._mylib.zoo.default, _EffectType.ORDERED
            )
            torch.library._register_effectful_op(
                torch.ops._mylib.zoo2.default, _EffectType.ORDERED
            )

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Applies the first custom op to x and adds y.
                return torch.ops._mylib.zoo(x) + y

            def ins_sc():
                # [FUNCTION DOC] HELPER: Returns a pair of inputs: a TwoTensor wrapping two equal
                # float tensors, and a plain float tensor.
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
                    ),
                    torch.tensor([4.0, 5.0, 6.0]),
                )

            def ins_dense():
                # [FUNCTION DOC] HELPER: Returns a pair of plain float tensors.
                return torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])

            for ins_fn, expected_fw_count in zip([ins_sc, ins_dense], [2, 1]):
                reset_counter()
                ref_out = fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                compiled_fn = torch.compile(fn, backend="aot_eager")
                out = compiled_fn(*ins_fn())
                reset_counter()
                out = compiled_fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                self.assertEqual(ref_out, out)

            def ins_dense_req_grad():
                # [FUNCTION DOC] HELPER: Returns a pair of plain float tensors that require grad.
                return (
                    torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                    torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                )

            def ins_sc_req_grad():
                # [FUNCTION DOC] HELPER: Returns a pair of TwoTensor objects, each wrapping two
                # float tensors that require grad.
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                    TwoTensor(
                        torch.tensor([7.0, 8.0, 9.0], requires_grad=True),
                        torch.tensor([10.0, 11.0, 12.0], requires_grad=True),
                    ),
                )

            for (
                ins_fn_req_grad,
                (
                    expected_fw_count,
                    expected_fw_count_after_bw,
                    expected_bw_count_after_bw,
                ),
            ) in zip([ins_dense_req_grad, ins_sc_req_grad], [(1, 1, 1), (2, 2, 2)]):
                ref_ins = ins_fn_req_grad()
                reset_counter()
                ref_out = fn(*ref_ins)
                assert_counter(expected_fw_count, 0)
                ref_out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)

                compiled_fn = torch.compile(fn, fullgraph=True)

                ins = ins_fn_req_grad()
                out = compiled_fn(*ins)
                reset_counter()
                out = compiled_fn(*ins)
                assert_counter(expected_fw_count, 0)
                self.assertEqual(ref_out, out)
                out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)
                self.assertEqual(ref_ins[1].grad, ins[1].grad)
                self.assertEqual(ref_ins[0].grad, ins[0].grad)

            fw_graph, bw_graph = get_fw_bw_graph(fn, ins_sc_req_grad())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.zoo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._mylib.zoo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = primals_4 = None
    add_1 = torch.ops.aten.add.Tensor(getitem_3, primals_5);  getitem_3 = primals_5 = None
    return (getitem_2, add, add_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.zoo2.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._mylib.zoo2.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    clone = torch.ops.aten.clone.default(tangents_1)
    clone_1 = torch.ops.aten.clone.default(tangents_2)
    return (clone, clone_1, tangents_1, tangents_2, getitem_6)""",
            )

    def test_dce(self):
        # [FUNCTION DOC] Checks that an op marked side-effectful is not removed by FX dead code
        # elimination. Registers a logging custom op (appending to a list), registers it as an
        # ordered effect, traces a function using it symbolically, eliminates dead code,
        # recompiles, runs the graph and checks the op still ran.
        # KNOWN ISSUE: the final assertion passes the list length as an assertTrue condition with
        #   1 as the message, so it only checks the log is non-empty and does not compare the
        #   length to one.
        # If an operator is marked as side effectful, it should not get DCEd by
        # FX's eliminate_dead_code

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            log3 = []

            @torch.library.custom_op(
                "mylib::my_logger3",
                mutates_args=(),
            )
            def my_logger3(s: str, t: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op body: appends the string to the log list and
                # returns a one-element zero tensor.
                log3.append(s)
                return torch.zeros(1)

            @my_logger3.register_fake
            def my_logger3(s, t) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation of the logging op: returns a
                # one-element zero tensor.
                return torch.zeros(1)

            # Registering an op as being effectful should also prevent FX DCE
            from torch._library.effects import EffectType

            torch.library._register_effectful_op(
                "mylib::my_logger3", EffectType.ORDERED
            )

            def foo(x):
                # [FUNCTION DOC] HELPER: Makes a scalar tensor from the first dimension size, calls
                # the logging op on it, and returns the input doubled.
                b = torch.scalar_tensor(x.shape[0])
                torch.ops.mylib.my_logger3("moo", b)
                return x + x

            gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 3))
            gm.graph.eliminate_dead_code()
            gm.recompile()
            gm(torch.ones(3, 3))
            self.assertTrue(len(log3), 1)

    def test_effects_and_input_mutation_return(self):
        # [FUNCTION DOC] Checks a print combined with an op writing to an output argument (sin with
        # an out tensor): compiled full graph output equals eager, and the forward graph text shows
        # the print wrapped in with_effects and the mutated input returned as an extra output.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Prints, then computes sin of the first argument into the
            # second and returns the result.
            torch.ops.aten._print("effect")
            return torch.sin(a, out=b)

        inp = [torch.randn(3, 3), torch.ones(3, 3)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    return (getitem, sin, sin)""",
        )

    def test_effects_and_input_output_view_simple(self):
        # [FUNCTION DOC] Checks a function that only returns a view of its input: compiled outputs
        # equal eager for inputs without and with grad, and the forward graph has just the view
        # with no effects.
        def fn(a):
            # [FUNCTION DOC] HELPER: Returns the input flattened by a view.
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_effects_and_aliased_outputs(self):
        # [FUNCTION DOC] Checks that aliasing between outputs (an output and a view of it) is
        # preserved by AOT compilation in the presence of a print, for inputs with and without
        # grad: after mutating one output in place, both outputs still match the eager ones.
        def fn(a):
            # [FUNCTION DOC] HELPER: Doubles the input, prints, creates a flattened
            # view, returns both.
            b = a.mul(2)
            torch.ops.aten._print("effect")
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(fn, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = fn(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_effects_and_input_mutation_is_output(self):
        # [FUNCTION DOC] Checks a function that mutates its input in place, prints, and returns the
        # input. Compares aot_eager full graph results to eager for non-leaf grad-requiring input
        # and plain input, and checks the exact forward graph (multiplication, wrapped print,
        # mutated output returned twice).
        def fn(a):
            # [FUNCTION DOC] HELPER: Multiplies the input by two in place, prints, and
            # returns the input.
            a.mul_(2)
            torch.ops.aten._print("effect")
            return a

        inp = make_inputs_non_leaves([torch.ones(3, 3, requires_grad=True)])
        ref_out = fn(*inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [torch.ones(3, 3, requires_grad=False)]
        ref_out = fn(*inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, mul, mul)""",
        )

    @skipIfTorchDynamo()
    def test_effectful_op_in_backward(self):
        # [FUNCTION DOC] Registers an ordered effect for a custom op that is used in its own
        # backward rule, compiles with inductor for dense and TwoTensor inputs and compares results
        # and gradients to eager, and checks exact forward/backward graph text (the backward graph
        # gets the token as an input). Unregisters the effect at the end and asserts it is gone.
        # Skipped under Dynamo.
        with torch.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")

            def foo_impl(a):
                # [FUNCTION DOC] HELPER: Implementation of the op: returns a clone of the input.
                return a.clone()

            def foo_bwd(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward rule: calls the same op on the gradient and
                # returns the result.
                return torch.ops._mylib.foo(grad)

            for backend in ["CPU", "CUDA", "Meta"]:
                lib.impl("foo", foo_impl, backend)

            torch.library.register_autograd("_mylib::foo", foo_bwd, lib=lib)

            handle = _register_effectful_op(
                torch.ops._mylib.foo.default, _EffectType.ORDERED
            )
            self.assertEqual(
                _get_effect(torch.ops._mylib.foo.default), _EffectType.ORDERED
            )

            try:

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Applies the custom op to x and adds y.
                    return torch.ops._mylib.foo(x) + y

                def ins_dense_req_grad():
                    # [FUNCTION DOC] HELPER: Returns a pair of plain float tensors
                    # that require grad.
                    return (
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                def ins_sc_req_grad():
                    # [FUNCTION DOC] HELPER: Returns a TwoTensor of two grad-requiring tensors plus
                    # a plain tensor that requires grad.
                    return (
                        TwoTensor(
                            torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                            torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                        ),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                for i, ins_fn in enumerate([ins_dense_req_grad, ins_sc_req_grad]):
                    ref_ins = ins_fn()

                    ref_out = fn(*ref_ins)
                    ref_out.sum().backward()

                    compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
                    ins = ins_fn()
                    out = compiled_fn(*ins)
                    self.assertEqual(ref_out, out)
                    out.sum().backward()
                    self.assertEqual(ref_ins[1].grad, ins[1].grad)
                    self.assertEqual(ref_ins[0].grad, ins[0].grad)

                    fw_graph, bw_graph = get_fw_bw_graph(fn, ins)
                    if i == 0:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_3);  getitem_1 = primals_3 = None
    return (getitem, add)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_token):
    with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    return (getitem_3, tangents_1, getitem_2)""",
                        )
                    elif i == 1:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._mylib.foo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = None
    add_1 = torch.ops.aten.add.Tensor(getitem_3, primals_4);  getitem_3 = primals_4 = None
    return (getitem_2, add, add_1)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._mylib.foo.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    return (getitem_5, getitem_7, tangents_1, tangents_2, getitem_6)""",
                        )
                    else:
                        raise NotImplementedError
            finally:
                handle.destroy()

            self.assertEqual(_get_effect(torch.ops._mylib.foo.default), None)

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_only_in_backward(self):
        # [FUNCTION DOC] Registers an ordered effect for the cosine op, which only appears in
        # the sine backward. Compiles with inductor and checks the exact graphs: the forward
        # has no effects and the backward wraps the cosine in with_effects using a token
        # input. Repeats for a TwoTensor input (two wrapped calls). The registration is
        # destroyed in a finally clause.
        handle = _register_effectful_op(torch.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return x.sin()

            def inps_fn():
                # [FUNCTION DOC] HELPER: Returns a one-tuple with a float vector
                # that requires grad.
                return (torch.tensor([1.0, 2.0, 3.0], requires_grad=True),)

            torch.compile(fn, backend="inductor", fullgraph=True)(*inps_fn())

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1):
    sin = torch.ops.aten.sin.default(primals_1)
    return (sin, primals_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, tangents_1, tangents_token):
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    return (mul, getitem)""",
            )

            def inps_fn_sc():
                # [FUNCTION DOC] HELPER: Returns a one-tuple with a TwoTensor of two
                # grad-requiring float vectors.
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                )

            torch.compile(fn, backend="inductor", fullgraph=True)(*inps_fn_sc())
            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn_sc())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    sin = torch.ops.aten.sin.default(primals_1)
    sin_1 = torch.ops.aten.sin.default(primals_2)
    return (sin, sin_1, primals_1, primals_2)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, tangents_1, tangents_2, tangents_token):
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten.cos.default, primals_2);  getitem = primals_2 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_2, getitem_3);  tangents_2 = getitem_3 = None
    return (mul, mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_in_forward_and_backward(self):
        # [FUNCTION DOC] Registers an ordered effect for cosine, which now appears in the forward
        # and in the sine backward. Compiles with inductor and checks exact forward and backward
        # graph text with tokens. The registration is destroyed in a finally clause.
        handle = _register_effectful_op(torch.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                # [FUNCTION DOC] HELPER: Applies cosine then sine to the input.
                x = x.cos()
                return x.sin()

            inps = (torch.tensor([1.0, 2.0, 3.0], requires_grad=True),)
            torch.compile(fn, backend="inductor", fullgraph=True)(*inps)

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps)
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.aten.cos.default, primals_2);  primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    sin = torch.ops.aten.sin.default(getitem_1)
    return (getitem, sin, primals_2, getitem_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_2, getitem_1, tangents_1, tangents_token):
    with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, getitem_1);  tangents_token = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_3);  tangents_1 = getitem_3 = None
    sin_1 = torch.ops.aten.sin.default(primals_2);  primals_2 = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg);  mul = neg = None
    return (mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not TEST_CUDA, "triton")
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_export_invoke_subgraph(self):
        # [FUNCTION DOC] Needs CUDA and triton (expected to fail without an accelerator triton).
        # Registers a memory-recording ordered-effect op, builds a model with three repeated nested
        # compile regions, exports it and runs decompositions, then checks the subgraph structure
        # and the exact graph text, the number of records, numerical match, and the unlifted-token
        # graph form; finally compiles the model and checks record count and closeness. Inline of
        # single-use subgraphs is off.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            recorded_list = []

            @torch.library.custom_op("mylib::record_memory", mutates_args=())
            def record_memory(prefix: str, module_name: str) -> None:
                # [FUNCTION DOC] HELPER: Custom op body: synchronizes CUDA, formats allocated and
                # reserved memory in megabytes with the prefix and module name, and appends the
                # string to the shared list.
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                memory_str = f"[{prefix}] {module_name}: allocated={mem_alloc:.2f} MB, reserved={mem_reserved:.2f} MB"
                recorded_list.append(memory_str)

            @record_memory.register_fake
            def record_memory_fake(prefix, module_name):
                # [FUNCTION DOC] HELPER: Fake implementation of the memory-recording
                # op; returns nothing.
                return

            record_memory.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib.record_memory.default)

            class N(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with two 1024 by 1024 linear layers and a
                # ReLU; forward is a nested compile region that records memory and applies
                # linear, ReLU, linear.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Creates the two linear layers and the ReLU.
                    super().__init__()
                    self.linear1 = torch.nn.Linear(1024, 1024)
                    self.relu = torch.nn.ReLU()
                    self.linear2 = torch.nn.Linear(1024, 1024)

                @torch.compiler.nested_compile_region
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Records memory with the forward label, then
                    # applies the layers.
                    torch.ops.mylib.record_memory("forward", "N")
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.linear2(x)
                    return x

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding three of the above submodules in a
                # list; forward applies them in sequence, records memory once more and
                # returns a one-tuple.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Creates the module list of three submodules.
                    super().__init__()
                    self.mod_list = torch.nn.ModuleList(N() for _ in range(3))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Runs the submodules in order, records memory, and
                    # returns a one-tuple.
                    for m in self.mod_list:
                        x = m(x)
                    torch.ops.mylib.record_memory("forward", "N")
                    return (x,)

            model = M().to("cuda")
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(32, 1024, requires_grad=True, device="cuda")

            # Test torch.export
            ep = torch.export.export(model, (x,))
            decomp = ep.run_decompositions()
            self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

            self.assertExpectedInline(
                decomp.graph_module.code.strip(),
                """\
def forward(self, token, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias, x):
    repeated_subgraph0 = self.repeated_subgraph0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', token, x, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias);  repeated_subgraph0 = token = x = p_mod_list_0_linear1_weight = p_mod_list_0_linear1_bias = p_mod_list_0_linear2_weight = p_mod_list_0_linear2_bias = None
    getitem = invoke_subgraph[0]
    getitem_1 = invoke_subgraph[1];  invoke_subgraph = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', getitem, getitem_1, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias);  repeated_subgraph0_1 = getitem = getitem_1 = p_mod_list_1_linear1_weight = p_mod_list_1_linear1_bias = p_mod_list_1_linear2_weight = p_mod_list_1_linear2_bias = None
    getitem_2 = invoke_subgraph_1[0]
    getitem_3 = invoke_subgraph_1[1];  invoke_subgraph_1 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_2, 'subgraph_0', getitem_2, getitem_3, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias);  repeated_subgraph0_2 = getitem_2 = getitem_3 = p_mod_list_2_linear1_weight = p_mod_list_2_linear1_bias = p_mod_list_2_linear2_weight = p_mod_list_2_linear2_bias = None
    getitem_4 = invoke_subgraph_2[0]
    getitem_5 = invoke_subgraph_2[1];  invoke_subgraph_2 = None
    with_effects = torch.ops.higher_order.with_effects(getitem_4, torch.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    return (getitem_6, getitem_5)""",
            )

            self.assertExpectedInline(
                decomp.graph_module.repeated_subgraph0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.mylib.record_memory.default, 'forward', 'N');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    addmm = torch.ops.aten.addmm.default(arg3_1, arg1_1, permute);  arg3_1 = arg1_1 = permute = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
    return (getitem, addmm_1)""",
            )

            recorded_list.clear()
            out2 = ep.module()(x)
            self.assertEqual(len(recorded_list), 4)
            self.assertTrue(torch.allclose(model(x)[0], out2[0]))

            # Test when we unlift the tokens from the graph. This is used in the inductor path.
            with (
                tracing(TracingContext(None)),
                torch._functorch.config.patch(unlift_effect_tokens=True),
            ):
                gm, gs = aot_export_module(ep.module(), (x,), trace_joint=False)
                self.assertExpectedInline(
                    str(gm.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
    _make_token_default = torch.ops.prims._make_token.default()
    repeated_subgraph0 = self.repeated_subgraph0
    with_effects_1 = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0, 'subgraph_0', arg13_1, arg1_1, arg2_1, arg3_1, arg4_1);  _make_token_default = repeated_subgraph0 = arg13_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = None
    getitem = with_effects_1[0]
    getitem_1 = with_effects_1[1];  with_effects_1 = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    with_effects_2 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0_1, 'subgraph_0', getitem_1, arg5_1, arg6_1, arg7_1, arg8_1);  getitem = repeated_subgraph0_1 = getitem_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = None
    getitem_2 = with_effects_2[0]
    getitem_3 = with_effects_2[1];  with_effects_2 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_2, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0_2, 'subgraph_0', getitem_3, arg9_1, arg10_1, arg11_1, arg12_1);  getitem_2 = repeated_subgraph0_2 = getitem_3 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = None
    getitem_4 = with_effects_3[0]
    getitem_5 = with_effects_3[1];  with_effects_3 = None
    with_effects = torch.ops.higher_order.with_effects(getitem_4, torch.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem_6]);  getitem_6 = _sink_tokens_default = None
    return (getitem_5,)""",
                )
                self.assertExpectedInline(
                    str(gm.repeated_subgraph0.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.mylib.record_memory.default, 'forward', 'N');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    t = torch.ops.aten.t.default(arg2_1);  arg2_1 = None
    addmm = torch.ops.aten.addmm.default(arg3_1, arg1_1, t);  arg3_1 = arg1_1 = t = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    t_1 = torch.ops.aten.t.default(arg4_1);  arg4_1 = None
    addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, t_1);  arg5_1 = relu = t_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem]);  getitem = _sink_tokens_default = None
    return (addmm_1,)""",
                )

        recorded_list.clear()
        out2 = torch.compile(model)(x)
        self.assertEqual(len(recorded_list), 4)
        self.assertTrue(torch.allclose(model(x)[0], out2[0], atol=1e-7, rtol=1e-4))

    @skipIfTorchDynamo()
    def test_effect_autograd_function(self):
        # [FUNCTION DOC] Registers an ordered effect for a gradient-logging custom op called inside
        # a custom autograd function backward. Compiles with the AOT eager recording backend,
        # compares loss with eager, and checks the exact forward graph (no effects) and backward
        # graph (token input, effect-wrapped op). Skipped under Dynamo.
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            @torch.library.custom_op("mylib::log_grad", mutates_args=())
            def log_grad(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op body: returns a clone of the input.
                return x.clone()

            @torch.library.register_fake("mylib::log_grad")
            def log_grad_fake(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation: returns a clone of the input.
                return x.clone()

            log_grad.register_effect(_EffectType.ORDERED)

            class NoOpWithLoggingBackward(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd function whose forward squares the input elementwise
                # and whose backward passes the gradient through the logging op and returns it.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Returns the input multiplied by itself; the
                    # context is unused.
                    return x * x

                @staticmethod
                def backward(ctx, grad_output):
                    # [FUNCTION DOC] HELPER: Passes the gradient through the logging op and
                    # returns the result.
                    logged_grad = torch.ops.mylib.log_grad(grad_output)
                    return logged_grad

            def fn(x):
                # [FUNCTION DOC] HELPER: Applies the autograd function and returns the sum
                # of the result.
                y = NoOpWithLoggingBackward.apply(x)
                return y.sum()

            x = torch.randn(3, 4, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)

            backend = AotEagerAndRecordGraphs()
            compiled_fn = torch.compile(fn, backend=backend)
            loss = compiled_fn(x)
            loss.backward()

            loss_ref = fn(x_clone)
            loss_ref.backward()
            self.assertEqual(loss, loss_ref)

            self.assertExpectedInline(
                backend.fw_graphs[0].code.strip(),
                """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_1);  primals_1 = None
    sum_1 = torch.ops.aten.sum.default(mul);  mul = None
    return (sum_1,)""",
            )

            self.assertExpectedInline(
                backend.bw_graphs[0].code.strip(),
                """\
def forward(self, tangents_1, tangents_token):
    expand = torch.ops.aten.expand.default(tangents_1, [3, 4]);  tangents_1 = None
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.mylib.log_grad.default, expand);  tangents_token = expand = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    return (getitem_1, getitem)""",
            )

    def test_with_effects_through_functional_tensor_mode(self):
        # [FUNCTION DOC] Calls with_effects on an add op under functional tensor mode (with the
        # functionalize dispatch key excluded) using functional tensor inputs, unwraps the result
        # if it is a functional tensor, and asserts it equals the plain sum.
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def fn_with_effects(x, y):
            # [FUNCTION DOC] HELPER: Makes a token, calls with_effects on the add op with both
            # inputs and returns only the computed result, ignoring the new token.
            token = torch.ops.prims._make_token()
            new_token, result = with_effects(
                token,
                torch.ops.aten.add.Tensor,
                x,
                y,
            )
            return result

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with (
            torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            ),
            FunctionalTensorMode(),
        ):
            x_func = FunctionalTensor.to_functional(x)
            y_func = FunctionalTensor.to_functional(y)
            result = fn_with_effects(x_func, y_func)

        expected = x + y
        if isinstance(result, FunctionalTensor):
            result = torch._from_functional_tensor(result.elem)
        self.assertEqual(result, expected)

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_effectful_op_with_flex_attention(self):
        # [FUNCTION DOC] Needs CUDA and SM80. Registers an ordered-effect noop custom op, uses it
        # on the query before flex attention with a pass-through score function, compiles the
        # function and checks the output shape matches the query shape (2, 4, 128, 64).
        """Test that effectful custom ops work with flex_attention."""
        from torch._library.effects import EffectType
        from torch.nn.attention.flex_attention import flex_attention

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:

            @torch.library.custom_op("mylib::noop", mutates_args=())
            def noop(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op body: returns a clone of the input.
                return x.clone()

            @noop.register_fake
            def noop_fake(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation: returns a clone of the input.
                return x.clone()

            noop.register_effect(EffectType.ORDERED)

            def score_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Score modification that returns the score unchanged.
                return score

            def fn(q, k, v):
                # [FUNCTION DOC] HELPER: Applies the noop custom op to q, then flex attention with
                # the score mod.
                q = torch.ops.mylib.noop(q)
                out = flex_attention(q, k, v, score_mod=score_mod)
                return out

            batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
            q = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )

            compiled_fn = torch.compile(fn)
            out = compiled_fn(q, k, v)
            self.assertEqual(out.shape, (batch_size, num_heads, seq_len, head_dim))


if __name__ == "__main__":
    run_tests()
