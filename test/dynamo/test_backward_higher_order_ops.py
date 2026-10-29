# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the trace_wrapped higher-order op, which wraps a tensor hook function
# so that a gradient hook can be traced into backward graphs. Covers eager use, torch.compile
# with several backends, make_fx tracing, and compiled autograd (including a stateful hook and
# a hook that graph-breaks). Defines two module-level hook helpers; no import-time side effects.

import functools
import itertools
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm
from torch.fx.experimental.proxy_tensor import make_fx


def _multiply(x):
    # [FUNCTION DOC] HELPER: Returns x multiplied elementwise by itself (the squared tensor).
    return x * x


def _multiply_invoke(grad):
    # [FUNCTION DOC] HELPER: Gradient hook; returns trace_wrapped applied to grad with _multiply
    # as the wrapped function, so the hook result is grad squared.
    return trace_wrapped(grad, fn=_multiply)


class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for the trace_wrapped backward hook op across eager, make_fx,
    # compile backends and compiled autograd. Tests keep no shared state; each resets Dynamo
    # where it loops over backends.
    def test_invoke_in_eager(self):
        # [FUNCTION DOC] Eager (uncompiled) run: registers _multiply_invoke as a hook on x,
        # backpropagates a gradient of 2s through x * y with both inputs 0.5, and asserts
        # x.grad equals y * grad_out. The hook squares the incoming gradient (which is 1.0),
        # so the squaring does not change the value here.
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x * y

        out = fn(x, y)
        grad_out = torch.tensor([2.0, 2.0])
        out.backward(grad_out)
        self.assertEqual(x.grad, y * grad_out)

    def test_invoke_in_pt2(self):
        # [FUNCTION DOC] For each of the eager, aot_eager and inductor backends: resets Dynamo,
        # compiles a function that registers the _multiply_invoke hook on x and returns x * y,
        # runs backward with a gradient of 2s, and asserts x.grad equals grad_out * y.
        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Registers the _multiply_invoke hook on x; returns x * y.
                x.register_hook(_multiply_invoke)
                return x * y

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            out.backward(grad_out)
            self.assertEqual(x.grad, grad_out * y)

    def test_invoke_make_fx_forward_contrived(self):
        # [FUNCTION DOC] Traces _multiply_invoke with make_fx on a requires-grad input, checks
        # the traced module returns the squared values (0.25) and that its normalized readable
        # code equals an expected string holding a single trace_wrapped call node.
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

    def test_invoke_make_bw(self):
        # [FUNCTION DOC] Runs a small eager forward/backward (z = x*x; z + z) to get x.grad,
        # then make_fx traces _multiply_invoke on that gradient. Asserts the traced module
        # returns the squared gradient (4.0 each) and matches the expected readable code.
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        def fwd(x):
            # [FUNCTION DOC] HELPER: Returns x*x added to itself.
            z = x * x
            return z + z

        res = fwd(x)
        res.backward(torch.tensor([1.0, 1.0]))
        out = make_fx(_multiply_invoke)(x.grad)
        self.assertEqual(out(x.grad), torch.tensor([4.0, 4.0]))
        actual = normalize_gm(out.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

    @torch._functorch.config.patch(guess_tangent_strides_as_outputs=True)
    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd(self, _):
        # [FUNCTION DOC] The patched AOT_COUNTER (second argument, unused) makes graph names
        # deterministic. For each backend, compiles a forward registering the hook on x and
        # returning x + y, then runs backward under compiled autograd with a compiler that
        # captures the single backward graph via inductor. Asserts x.grad equals grad_out
        # squared and, for aot_eager and inductor, that the captured graph text matches.
        graph = None

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiled-autograd compiler; returns gm compiled
            # fullgraph and dynamic with inner_compiler as the backend.
            def inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Backend; asserts no graph captured yet, stores gm_
                # in the enclosing graph variable, returns inductor.compile of it.
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Registers _multiply_invoke on x; returns x + y.
                x.register_hook(_multiply_invoke)
                return x + y

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(x.grad, grad_out * grad_out)
            if backend == "aot_eager":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, L_sizes_0_: "f32[0, 2]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[2]" = l_inputs_[0]
        getitem_1: "f32[2]" = l_inputs_[1]
        getitem_2: "f32[2]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(2)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[2]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], (getitem_9,));  getitem_9 = None
        aot1_tangents_1: "f32[2]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot1_tangents_1], getitem_1, None, False);  getitem_1 = None
        call_accumulate_grad: "f32[2]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[2]" = aot1_tangents_1 * aot1_tangents_1;  aot1_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        call_accumulate_grad_1: "f32[2]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (call_accumulate_grad, call_accumulate_grad_1)
""",
                )
            elif backend == "inductor":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, L_sizes_0_: "f32[0, 2]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[2]" = l_inputs_[0]
        getitem_1: "f32[2]" = l_inputs_[1]
        getitem_2: "f32[2]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(2)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[2]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], (getitem_9,));  getitem_9 = None
        aot3_tangents_1: "f32[2]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot3_tangents_1], getitem_1, None, False);  getitem_1 = None
        call_accumulate_grad: "f32[2]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[2]" = aot3_tangents_1 * aot3_tangents_1;  aot3_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        call_accumulate_grad_1: "f32[2]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (call_accumulate_grad, call_accumulate_grad_1)
""",
                )

            graph = None

    @torch._functorch.config.patch(guess_tangent_strides_as_outputs=True)
    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd_side_effect(self, _):
        # [FUNCTION DOC] The patched AOT_COUNTER (second argument, unused) fixes graph names.
        # Registers on x a hook (built with functools.partial) that increments counter on a
        # plain object and squares the gradient, compiles x + y fullgraph on inductor, and runs
        # backward under compiled autograd. Asserts the counter is 1 and x.grad is 4.0, the
        # graph matches the expected text, and two further backward runs bring the counter to 3.
        def _side_effect_stateful_fn2(x, obj):
            # [FUNCTION DOC] HELPER: Increments obj.counter by one and returns x squared.
            obj.counter = obj.counter + 1
            return _multiply(x)

        def _side_effectful_invoke2(grad, fn):
            # [FUNCTION DOC] HELPER: Returns trace_wrapped of grad with fn as the wrapped function.
            return trace_wrapped(grad, fn=fn)

        graph = None

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiled-autograd compiler; compiles gm fullgraph and
            # dynamic using inner_compiler as backend.
            def inner_compiler(gm_, example_inputs_):
                # [FUNCTION DOC] HELPER: Backend; asserts no graph captured yet, stores gm_
                # in the enclosing graph variable, returns inductor.compile of it.
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        for backend in ["inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            class MyObj:
                # [CLASS DOC] HELPER: Mutable holder with an integer counter attribute.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Sets counter to 0.
                    self.counter = 0

            obj = MyObj()
            inner_fn = functools.partial(_side_effect_stateful_fn2, obj=obj)
            hook_fn = functools.partial(_side_effectful_invoke2, fn=inner_fn)
            x.register_hook(hook_fn)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns x + y.
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(obj.counter, 1)
            self.assertEqual(x.grad, grad_out + grad_out)
            if backend in ["aot_eager", "inductor"]:
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, L_sizes_0_: "f32[0, 2]", L_hooks_1_keywords_fn_keywords_obj_counter: "Sym(s45)"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_
        l_hooks_1_keywords_fn_keywords_obj_counter = L_hooks_1_keywords_fn_keywords_obj_counter

        getitem: "f32[2]" = l_inputs_[0]
        getitem_1: "f32[2]" = l_inputs_[1]
        getitem_2: "f32[2]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(2)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[2]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], (getitem_9,));  getitem_9 = None
        aot0_tangents_1: "f32[2]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot0_tangents_1], getitem_1, None, False);  getitem_1 = None
        call_accumulate_grad: "f32[2]" = accumulate_grad[0];  accumulate_grad = None

        add: "Sym(s45 + 1)" = l_hooks_1_keywords_fn_keywords_obj_counter + 1;  l_hooks_1_keywords_fn_keywords_obj_counter = None

        result: "f32[2]" = aot0_tangents_1 * aot0_tangents_1;  aot0_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        call_accumulate_grad_1: "f32[2]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (call_accumulate_grad, call_accumulate_grad_1, add)
""",
                )

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 2)

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 3)
            graph = None

    def test_invoke_in_pt2_compiled_autograd_graph_breaks(self):
        # [FUNCTION DOC] For each backend, registers a hook whose wrapped function calls print
        # (a graph break), compiles x + y with fullgraph, and asserts that running backward
        # under compiled autograd (inductor, fullgraph, dynamic) raises
        # torch._dynamo.exc.Unsupported with a message matching "print".
        def _graph_breaking_fn(x):
            # [FUNCTION DOC] HELPER: Prints a message (unsupported in a fullgraph trace), then
            # returns x squared.
            print("Boo!")
            return _multiply(x)

        def _graph_break_invoke(grad):
            # [FUNCTION DOC] HELPER: Returns trace_wrapped of grad wrapping _graph_breaking_fn.
            return trace_wrapped(grad, fn=_graph_breaking_fn)

        def compiler_fn(gm):
            # [FUNCTION DOC] HELPER: Compiled-autograd compiler; compiles gm with inductor,
            # fullgraph and dynamic.
            return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Registers _graph_break_invoke on x; returns x + y.
                x.register_hook(_graph_break_invoke)
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "print",
            ):
                with compiled_autograd._enable(compiler_fn):
                    out.backward(grad_out)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
