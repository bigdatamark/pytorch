# Owner(s): ["module: functionalization"]
# [MODULE DOC] Tests for auto-functionalization of mutating custom ops under torch.compile (the
# auto_functionalized and auto_functionalized_v2 higher-order ops) with the aot_eager and inductor
# backends. Each test registers a throwaway op in a scoped "mylib" fragment library, compiles a
# caller, compares mutated arguments and results with eager execution, and in many cases asserts
# the exact post-grad / AOT graph text captured through logging. Graph expectations only hold under
# static-by-default shapes and so are guarded by the dynamo config flag. Also covers slice
# inference, views and aliasing, SymInt args, tensor lists, optional tensors, returns, inplace/out
# variants, and the effect of the v2 config flag. Import time has no side effects beyond imports.

import unittest

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.config as inductor_config
import torch._inductor.test_case
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._higher_order_ops.auto_functionalize import try_use_slice
from torch.testing._internal.logging_utils import logs_to_string


class AutoFunctionalizeTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Main test case (inductor test base) for auto-functionalization. Holds the shared
    # runners run_aot_eager and run_inductor that compile with a backend while capturing the logged
    # graph, plus tests that cover v1 and v2 behavior. Most tests define and tear down their own
    # library ops inside scoped libraries so no op registrations leak between tests.
    def test_auto_functionalize_can_with_default(self):
        # [FUNCTION DOC] Smoke test that a custom op whose schema has an optional mutated tensor
        # and defaulted trailing arguments can be auto-functionalized when called with only the
        # first two arguments. Defines the op, a CPU implementation that performs a no-op addition,
        # and compiles a caller. Passes if compilation and the call raise nothing; no outputs are
        # compared.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, int b, Tensor(d!)? c=None, Tensor? d=None, int e=-1) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b, c=None, d=None, e=-1):
                # [FUNCTION DOC] HELPER: CPU implementation of the test op; computes a sum of the
                # first two arguments, discards it, and returns nothing. Ignores the optional
                # arguments.
                a + b
                return

            def f(a, mode):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op with the input tensor
                # and the integer zero only, so every defaulted argument is left unset. The mode
                # argument is unused.
                return torch.ops.mylib.foo(
                    a,
                    0,
                )

            a = torch.tensor([10, 10, 10], dtype=torch.int64)

            torch.compile(f)(a, 0)

    def test_auto_functionalize_can_with_none_return(self):
        # [FUNCTION DOC] Smoke test that an op declared with a mutated out tensor and no return
        # value (None return type) compiles. Registers an eager implementation that copies the
        # input into out, compiles a caller and runs it. Passes if no exception is raised; values
        # are not checked.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, Tensor(a!) out) -> None")

            def foo_impl(x, out):
                # [FUNCTION DOC] HELPER: Eager implementation of the op; copies x into the out
                # tensor.
                out.copy_(x)

            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
            x = torch.randn(3)
            out = torch.zeros(3)

            @torch.compile
            def f(x, out):
                # [FUNCTION DOC] HELPER: Compiled caller; invokes the op on the input and out
                # tensors and returns nothing.
                torch.ops.mylib.foo(x, out)

            f(x, out)

    def test_auto_functionalize_self_as_mutate_arg(self):
        # [FUNCTION DOC] Smoke test that an op whose only argument, named self, is mutated can be
        # auto-functionalized with the inductor backend and fullgraph compilation. Passes if the
        # compiled call raises nothing.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) self) -> None")

            def foo_impl(self: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Eager implementation of the op; applies sine in place to
                # its argument.
                self.sin_()

            x = torch.randn(3)
            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @torch.compile(backend="inductor", fullgraph=True)
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled caller; invokes the mutating op on its input.
                torch.ops.mylib.foo(x)

            f(x)

    def test_auto_functionalize_tensorlist(self):
        # [FUNCTION DOC] Checks auto-functionalization of an op mutating a list of tensors (out)
        # with a SymInt list argument. Compiles with inductor (fullgraph) on cloned arguments, runs
        # the same function eagerly on other clones, and asserts the argument structures are equal,
        # i.e. each output tensor was filled with the input.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim, Tensor(a!)[] out) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(all_gather_output, all_gather_input_split_sizes, dim, out):
                # [FUNCTION DOC] HELPER: CPU implementation (excluded from dynamo tracing); copies
                # the gathered input into every tensor of the out list.
                for o in out:
                    o.copy_(all_gather_output)

            def f(all_gather_output, all_gather_input_split_sizes, dim, out):
                # [FUNCTION DOC] HELPER: Caller under test; forwards all four arguments to the
                # mutating op.
                torch.ops.mylib.foo(
                    all_gather_output, all_gather_input_split_sizes, dim, out
                )

            a = torch.ones(4)
            b = [2, 3]
            c = 0
            d = [torch.empty(4) for _ in range(2)]
            orig_args = (a, b, c, d)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def test_can_auto_functionalize(self):
        # [FUNCTION DOC] Table-driven check of the can_auto_functionalize predicate. For each
        # schema in an expected-true list (mutating tensor, optional tensor, tensor list arguments,
        # with no return, a tensor return or a tuple of tensors) and an expected-false list (no
        # mutation, aliasing returns, returning a mutated alias or a tensor list), defines the op
        # in a scoped library and asserts the predicate on the default overload gives the expected
        # answer. In all cases the predicate on the overload packet is asserted false. Failure
        # message includes the schema.
        from torch._higher_order_ops.auto_functionalize import can_auto_functionalize

        expected_true = [
            "(Tensor(a!) x) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!)[] z, SymInt w) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor)",
        ]
        expected_false = [
            "(Tensor x) -> ()",
            "(Tensor(a) x) -> Tensor(a)",
            "(Tensor(a!) x) -> Tensor(a!)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor(a)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor[])",
        ]
        for schema in expected_true:
            with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
                torch.library.define("mylib::a", schema, lib=lib)

                self.assertTrue(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))

        for schema in expected_false:
            with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
                torch.library.define("mylib::a", schema, lib=lib)
                self.assertFalse(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_old(self):
        # [FUNCTION DOC] Checks the original (v1) auto-functionalization with the v2 flag patched
        # off. Defines an op mutating a tensor and a tensor-list-element, compiles a caller with
        # inductor, captures the post-grad graph via logging, and when shapes are static asserts
        # the graph text (a single call to the op, copy-free) and that the source line of the call
        # appears in the graph stack trace comment. Finally compares mutated arguments against
        # eager run.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); adds the first
                # list element plus the scalar to x and the second element plus n to z, in place.
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op with the scalar two.
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # Check the graph under static shapes
            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg0_1 = foo_default = None
        return ()""",
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: torch.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_with_returns_old(self):
        # [FUNCTION DOC] Same as the v1 test above but for an op with two tensor returns. Registers
        # an implementation, a fake kernel returning two new tensors, compiles with the v2 flag
        # off, and for static shapes asserts the post-grad graph (op call followed by two indexed
        # outputs). Asserts mutated args and returned outputs match an eager run on clones.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation; mutates x and z in place and returns
                # two new tensors, the first list element plus the scalar and the second list
                # element plus n.
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: Fake (abstract) implementation returning the same two
                # non-mutating results as the real kernel.
                return y[0] + w, y[1] + n

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; returns the two results of the op
                # called with the scalar two.
                return torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )

            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        return (getitem_4, getitem_5)""",
                    ignore_comments=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    def test_auto_functionalize_on_view(self):
        # [FUNCTION DOC] For each value of the v2 flag, checks auto-functionalization when the
        # mutated argument is a view. The op sine-es its input in place through a numpy view. First
        # verifies eager behavior, then compiles (aot_eager_decomp_partition, fullgraph) a function
        # that clones its input, takes a full slice, mutates the slice and returns the clone; the
        # result must equal sine of the input.
        for value in [True, False]:
            with (
                torch.library._scoped_library("mylib", "FRAGMENT") as lib,
                inductor_config.patch({"enable_auto_functionalized_v2": value}),
            ):
                torch.library.define(
                    "mylib::foo",
                    "(Tensor(a!) x) -> ()",
                    tags=torch.Tag.pt2_compliant_tag,
                    lib=lib,
                )

                @torch.library.impl("mylib::foo", "cpu", lib=lib)
                @torch._dynamo.disable
                def foo_impl(x):
                    # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies
                    # numpy sine in place over a numpy view of the tensor, so the tensor itself is
                    # mutated.
                    x_np = x.detach().numpy()  # view
                    np.sin(x_np, out=x_np)
                    return

                x = torch.randn(3)
                expected = x.sin()
                torch.ops.mylib.foo(x)
                if not torch.allclose(x, expected):
                    raise AssertionError

                @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
                def f(x):
                    # [FUNCTION DOC] HELPER: Compiled function; clones the input, mutates a full
                    # slice of the clone with the op and returns the clone.
                    x = x.clone()
                    y = x[:]
                    torch.ops.mylib.foo(y)
                    return x

                y = f(x)
                self.assertEqual(y, x.sin())

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_optional_old(self):
        # [FUNCTION DOC] Same as the v1 test but with optional mutated arguments, where x is None
        # and the others are tensors (v2 flag off). Asserts the static-shape post-grad graph keeps
        # None for the missing argument, that the call-site source text is in the graph, and that
        # mutated arguments match eager.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); adds into x and
                # z in place, each only when it is not None.
                if x is not None:
                    x.add_(y[0] + w)
                if z is not None:
                    z.add_(y[1] + n)

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op with the scalar two.
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = None
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)
            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(None, [arg1_1, arg2_1], arg3_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg0_1 = foo_default = None
        return ()""",
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: torch.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_auto_functionalize_op(self):
        # [FUNCTION DOC] Smoke test for a mutating custom op whose fake kernel returns a tensor
        # with four unbacked dynamic sizes. Config enables scalar output capture and dynamic output
        # shape ops. Defines the op (mutates its decoder argument, CPU only), its fake, compiles a
        # caller with fullgraph and runs it once. Passes if no exception is raised.
        @torch.library.custom_op(
            "mylib::mk_image", mutates_args=("decoder",), device_types=["cpu"]
        )
        def mk_image(decoder: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation; ignores its argument and returns a
            # random tensor of shape 2 by 3 by 4 by 5.
            return torch.randn(2, 3, 4, 5)

        @torch.library.register_fake("mylib::mk_image")
        def _(decoder: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Fake kernel for the custom op; builds four new dynamic sizes
            # from the fake context and returns an empty tensor of that unbacked shape.
            image_size = [torch.library.get_ctx().new_dynamic_size() for _ in range(4)]
            return torch.empty(image_size)

        @torch.compile(fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled caller; invokes the default overload of the custom
            # op.
            return torch.ops.mylib.mk_image.default(x)

        x = torch.zeros(100, dtype=torch.int64)
        f(x)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_v2(self, _dynamic=False):
        # [FUNCTION DOC] Checks the v2 auto-functionalization (flag patched on). The dynamic
        # parameter chooses the dynamic or static compile mode (not a pytest argument; defaults to
        # static). Defines a mutating op, compiles a caller with inductor, captures the post-grad
        # graph and, under static-by-default config, asserts the expected graph text for the chosen
        # mode (the op call followed by copy-back nodes for mutated inputs). Compares mutated args
        # against eager.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); adds the first
                # list element plus the scalar to x and the second element plus n to z, in place.
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op with the scalar two.
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", dynamic=_dynamic, fullgraph=True)(
                    *compiled_args
                )

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "f32[s77][1]cpu", arg2_1: "Sym(s77)", arg3_1: "f32[s77][1]cpu", arg4_1: "f32[s77][1]cpu", arg5_1: "f32[s77][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg3_1, arg4_1], arg5_1, 2, arg0_1);  arg3_1 = arg4_1 = arg0_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg5_1, arg5_1);  arg5_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg2_1 = arg3_1 = arg0_1 = foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def run_aot_eager(self, f, orig_args, _dynamic=False):
        # [FUNCTION DOC] HELPER: Runs a function under the aot_eager backend (fullgraph, optional
        # dynamic shapes) on cloned inputs while capturing the logged AOT graphs. Returns a three
        # item list: the cloned arguments after the call (to observe mutation), the call result,
        # and the captured graph text with its first four log lines dropped.
        aot_eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )

        result = None
        with ctx():
            result = torch.compile(
                f, backend="aot_eager", fullgraph=True, dynamic=_dynamic
            )(*aot_eager_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        return [aot_eager_args, result, graph]

    def run_inductor(
        self,
        f,
        orig_args,
        _dynamic=False,
        log_module="torch._inductor.compile_fx",
        log_function="post_grad_graphs",
    ):
        # [FUNCTION DOC] HELPER: Same as the aot_eager runner but for the inductor backend. The
        # logger module and artifact name default to the post-grad graphs but can be overridden.
        # Returns the cloned arguments after the call, the result, and the captured graph text with
        # the first three log lines dropped.
        compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(log_module, log_function)
        result = None
        with ctx():
            result = torch.compile(
                f, backend="inductor", fullgraph=True, dynamic=_dynamic
            )(*compiled_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()

        return [compiled_args, result, graph]

    def test_cumulative_out_preserves_out_dtype_under_compile(self):
        # [FUNCTION DOC] Checks that the cumulative sum and cumulative product ops with an integer
        # out tensor of float input keep the out dtype under compile. For each op, runs eager,
        # aot_eager and inductor (fullgraph) on separate copies of an int32 out tensor, then
        # asserts the out tensors and returned values match eager results.
        test_cases = [
            (torch.cumsum, torch.tensor([0.9201, 0.1166], dtype=torch.float32)),
            (torch.cumprod, torch.tensor([2.0, 0.6], dtype=torch.float32)),
        ]

        for op, x in test_cases:
            with self.subTest(op=op.__name__):

                def f(out, x):
                # [FUNCTION DOC] HELPER: Calls the current op over dimension zero, writing into
                # out, and returns the result.
                    return op(x, 0, out=out)

                eager_out = torch.tensor([0, 0], dtype=torch.int32)
                aot_eager_out = eager_out.clone()
                inductor_out = eager_out.clone()

                eager_ret = f(eager_out, x)
                aot_eager_ret = torch.compile(f, backend="aot_eager", fullgraph=True)(
                    aot_eager_out, x
                )
                inductor_ret = torch.compile(f, backend="inductor", fullgraph=True)(
                    inductor_out, x
                )

                self.assertEqual(aot_eager_out, eager_out)
                self.assertEqual(inductor_out, eager_out)
                self.assertEqual(aot_eager_ret, eager_ret)
                self.assertEqual(inductor_ret, eager_ret)

    def test_out_return_preserves_view_after_graph_break(self):
        # [FUNCTION DOC] Checks that out-variant ops keep returning a view of the out argument
        # after a graph break. First compiles (aot_eager) a function taking two views of a larger
        # buffer, calling addcmul with out twice, and checks shapes, that the result shares storage
        # with the output, and values. Second, with views of identical metadata, checks add with
        # out keeps the result and output as views of the original base tensor.
        sz = 9

        def f(data, other1, other2):
            # [FUNCTION DOC] HELPER: Takes two views of the first sz elements, forces a graph
            # break, runs addcmul into a fresh buffer for expected values, then into the output
            # view, and returns the output, result and expected buffer.
            inp = data[:sz]
            output = data[:sz]
            torch._dynamo.graph_break()
            expected = torch.empty_like(output)
            torch.addcmul(
                inp,
                other1.view(inp.shape),
                other2.view(inp.shape),
                out=expected,
            )
            result = torch.addcmul(
                inp,
                other1.view(inp.shape),
                other2.view(inp.shape),
                out=output,
            )
            return output, result, expected

        data = torch.randn(2 * sz, dtype=torch.float64)
        other1 = torch.randn(sz, dtype=torch.float64)
        other2 = torch.randn(sz, dtype=torch.float64)

        output, result, expected = torch.compile(f, backend="aot_eager")(
            data, other1, other2
        )

        self.assertEqual(output.shape, (sz,))
        self.assertEqual(result.shape, (sz,))
        self.assertEqual(expected.shape, (sz,))
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output, expected)
        self.assertEqual(result, expected)

        def same_metadata_view(data, other):
            # [FUNCTION DOC] HELPER: Takes two full views of the data, forces a graph break, runs
            # add into a new buffer and into the output view, and returns the base, output, result
            # and expected.
            inp = data.view(data.shape)
            output = data.view(data.shape)
            torch._dynamo.graph_break()
            expected = torch.empty_like(output)
            torch.add(inp, other, out=expected)
            result = torch.add(inp, other, out=output)
            return data, output, result, expected

        base, output, result, expected = torch.compile(
            same_metadata_view, backend="aot_eager"
        )(torch.randn(sz), torch.randn(sz))

        self.assertIs(output._base, base)
        self.assertIs(result._base, base)
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output, expected)
        self.assertEqual(result, expected)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_with_returns_v2(self):
        # [FUNCTION DOC] Checks v2 auto-functionalization (flag patched on) for an op with two
        # tensor returns. Compiles with inductor, and for static shapes asserts the post-grad graph
        # (op call, two indexed outputs and copy-back nodes for mutated inputs). Asserts mutated
        # args and outputs equal an eager run on clones.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); mutates x and z
                # in place and returns two new tensors.
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: Fake kernel returning the same two non-mutating results.
                return y[0] + w, y[1] + n

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; returns the two results of the op
                # called with the scalar two.
                return torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)
            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )
            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg2_1 = arg3_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return (getitem_4, getitem_5)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    # foo takes two inputs that are not views.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra1(self, _dynamic=False):
        # [FUNCTION DOC] Checks v2 for an op that mutates two separate non-view tensors in place
        # with sine (dynamic parameter picks the dynamic or static mode). Runs aot_eager, inductor
        # and eager, and asserts that mutated arguments and results agree across all three. Under
        # static-by-default config, asserts the expected AOT and inductor graphs (a single
        # higher-order call over the two base tensors followed by copy-backs).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x, y):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op on x and y, then returns
                # their sum.
                torch.ops.mylib.foo(x, y)
                return x + y

            orig_args = (torch.randn(2), torch.randn(2))

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)", arg2_1: "f32[s77][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg2_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[s77][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[s77][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)", arg2_1: "f32[s77][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg2_1);  foo_default = None
        add: "f32[s77][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, arg2_1)
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function does not have return.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra2(self, _dynamic=False):
        # [FUNCTION DOC] V2 test where the op mutates two scalar views (first and second element)
        # of the same input and the function returns nothing (dynamic parameter picks dynamic or
        # static mode). Runs aot_eager, inductor and eager; asserts mutated args and results agree.
        # Under static-by-default config asserts the AOT graph (one base with offsets for each view
        # and a single copy-back) and the inductor graph (as_strided views feeding the op, then a
        # copy-back).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; takes the first and second elements of
                # the input as views, calls the op on them and returns nothing.
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)
                return

            orig_args = [torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function returns both views and the input
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra3(self):
        # [FUNCTION DOC] V2 test where the op mutates two scalar views of one input and the
        # function returns both views and the input. Compares aot_eager, inductor and eager results
        # and mutated args, and under static shapes asserts the AOT and inductor graphs, where the
        # returned values are re-derived from the functionalized output by selection.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; creates two element views of the input,
                # mutates them with the op and returns both views plus the input.
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 1);  getitem_1 = None
        return (select_2, select_3)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        return (select_2, select_3)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes x, y both being graph inputs and views of the same shared base but do not overlap.
    # In this special case functionlization will have none as base for x and y. so they will be assumed
    # to have unique bases during functionalizations. During inplace, we notice that they both share storage
    # but because their memory does not overlap we can inplace both. see github issue #139628
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra5(self):
        # [FUNCTION DOC] V2 regression test (github issue 139628) for two graph inputs that are
        # non-overlapping views of the same underlying buffer (rows of a 2 by 2 tensor).
        # Functionalization treats them as having separate bases, and during in-place conversion
        # inductor can still mutate both in place because their memory does not overlap. Compares
        # aot_eager, inductor and eager and, for static shapes, asserts the AOT graph and the
        # inductor graph.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x, y):
                # [FUNCTION DOC] HELPER: Caller under test; returns the result of calling the op on
                # both inputs.
                return torch.ops.mylib.foo(x, y)

            base = torch.randn(2, 2)
            orig_args = [base[0], base[1]]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes a mutable list with views in addition to other args.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra4(self):
        # [FUNCTION DOC] V2 test with an op taking a mutated tensor plus a mutated tensor list,
        # called on views of two inputs and a plain third input. Compares aot_eager, inductor and
        # eager results; for mutated args it checks the third argument against eager and the
        # inductor args against aot_eager. Under static shapes asserts the AOT graph (bases, list
        # length and per-element metadata) and inductor graph (as_strided views, op call,
        # copy-backs).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!)[] y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and to the first element of the list y.
                x.sin_()
                y[0].sin_()

            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Caller under test; takes element zero views of x and z and
                # calls the op with the first view and a list of the second view and y. Returns
                # nothing.
                a = x[0]
                b = z[0]
                torch.ops.mylib.foo(a, [b, y])

            orig_args = [torch.randn(2), torch.randn(2), torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args[2], eager_args[2])
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_length = 2, _y_0_base_index = 1, _y_0_size = (), _y_0_stride = (), _y_0_storage_offset = 0, _y_1_base_index = 2, _all_bases = [arg0_1, arg2_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2]
        getitem_3: "f32[2][1]cpu" = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__2 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg2_1, [], [], 0)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, [as_strided_default_1, arg1_1]);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__2 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_optional_v2(self):
        # [FUNCTION DOC] V2 test with optional mutated arguments where x is None. Compiles with
        # inductor, captures the post-grad graph and, for static shapes, asserts that the op
        # receives None for the missing argument and that copy-back nodes exist only for mutated
        # inputs actually used. Checks mutated args against an eager run.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); adds into x and
                # z in place, each only when not None.
                if x is not None:
                    x.add_(y[0] + w)
                if z is not None:
                    z.add_(y[1] + n)

            def f(x, y, z, n):
                # [FUNCTION DOC] HELPER: Caller under test; invokes the op with the scalar two.
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = None
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(None, [arg1_1, arg2_1], arg3_1, 2, arg0_1);  arg1_1 = arg2_1 = arg0_1 = foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg3_1, arg3_1);  arg3_1 = copy_ = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_inference_mode1_v2(self):
        # [FUNCTION DOC] Runs the first extra v2 test under inference mode.
        # KNOWN ISSUE: the name says v2 and this method patches the v2 flag to false, but the
        # called test is itself decorated to patch the flag to true, so the inner flag wins while
        # it runs.
        with torch.inference_mode():
            self.test_auto_functionalize_extra1()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode2_v2(self):
        # [FUNCTION DOC] Runs the second extra v2 test under inference mode, with the v2 flag on.
        with torch.inference_mode():
            self.test_auto_functionalize_extra2()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode3_v2(self):
        # [FUNCTION DOC] Runs the third extra v2 test under inference mode, with the v2 flag on.
        with torch.inference_mode():
            self.test_auto_functionalize_extra3()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode4_v2(self):
        # [FUNCTION DOC] Runs the fourth extra v2 test under inference mode, with the v2 flag on.
        with torch.inference_mode():
            self.test_auto_functionalize_extra4()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic_v2(self):
        # [FUNCTION DOC] Runs the base v2 test in dynamic-shape mode, with the v2 flag on.
        self.test_auto_functionalize_v2(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic2_v2(self):
        # [FUNCTION DOC] Runs the first extra v2 test in dynamic-shape mode, with the v2 flag on.
        self.test_auto_functionalize_extra1(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic3_v2(self):
        # [FUNCTION DOC] Runs the second extra v2 test in dynamic-shape mode, with the v2 flag on.
        self.test_auto_functionalize_extra2(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_graph_input_is_view(self):
        # [FUNCTION DOC] Regression check that v2 handles a graph input that is a view: compiles
        # with aot_eager (static shapes, fullgraph) a function that takes the first element view of
        # its input and mutates it with a no-op op, then calls it with the second row of a matrix,
        # so the graph input is itself a view. Would fail if not-in-place args were cloned without
        # preserving strides. Passes if nothing raises.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); does nothing.
                pass

            @torch.compile(fullgraph=True, dynamic=False, backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function; takes the first element view of its
                # input and calls the op on it, returning nothing.
                a = x[0]
                torch.ops.mylib.foo(a)
                return

            x = torch.tensor([[1, 2], [3, 4]])
            # This would fail if auto_functionalized_v2 uses clone and not clone_preserve_strides
            # to clone not-inplaced args.
            f(x[1])

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_graph_input_view_storage_offset(self):
        # [FUNCTION DOC] Checks v2 with graph inputs that are views with nonzero storage offset.
        # First verifies that an as_strided based function compiled with inductor matches eager on
        # a slice of a base. Then defines a mutating op that adds ten, compiles a function that
        # calls it on a row view of its input and returns the input, runs it on the second row of a
        # matrix for both eager and inductor, and asserts the base tensors and results match.
        def f_as_strided(x):
            # [FUNCTION DOC] HELPER: Returns a clone of a two element strided view of x with stride
            # one and offset zero.
            return torch.as_strided(x, (2,), (1,), 0).clone()

        opt_f_as_strided = torch.compile(
            f_as_strided, fullgraph=True, dynamic=False, backend="inductor"
        )
        base = torch.arange(8, dtype=torch.float32)
        self.assertEqual(opt_f_as_strided(base[2:4]), f_as_strided(base[2:4]))

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); adds ten to x
                # in place.
                x.add_(10)

            def f(x):
                # [FUNCTION DOC] HELPER: Function under test; calls the op on the first element
                # view of x and returns x.
                a = x[0]
                torch.ops.mylib.foo(a)
                return x

            opt_f = torch.compile(f, fullgraph=True, dynamic=False, backend="inductor")

            eager_base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            result_eager = f(eager_base[1])

            inductor_base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            result_inductor = opt_f(inductor_base[1])

            self.assertEqual(inductor_base, eager_base)
            self.assertEqual(result_inductor, result_eager)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias(self, _dynamic=False):
        # [FUNCTION DOC] V2 test where the two mutated args are both plain alias views of one
        # input, and the function returns both aliases and the input (dynamic parameter picks the
        # mode). Compares aot_eager, inductor and eager; for static-by-default shapes asserts the
        # AOT graph (base with alias flags and one copy-back) and the inductor graph (two alias
        # nodes feeding the op).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; creates two alias views of the input,
                # calls the op on them, returns both aliases and the input.
                a = torch.ops.aten.alias.default(x)
                b = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(2)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg1_1])
        getitem_1: "f32[s0][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = copy_ = None
        alias_2: "f32[s0][1]cpu" = torch.ops.aten.alias.default(getitem_1)
        alias_3: "f32[s0][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_2: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1)
        alias_3: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        alias_default: "f32[s0][1]cpu" = torch.ops.aten.alias.default(arg1_1)
        alias_default_1: "f32[s0][1]cpu" = torch.ops.aten.alias.default(arg1_1)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  copy_ = None
        return (arg1_1, arg1_1)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        alias_default: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        return (arg0_1, arg0_1)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Test that slice view is generated instead of as_strided when split is used.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split(self, _dynamic=False):
        # [FUNCTION DOC] V2 test that splitting a tensor into two parts and mutating both pieces is
        # encoded with slice metadata (dimension, start, end) instead of strided views. Dynamic
        # parameter chooses the mode, though split specializes sizes so graphs are identical.
        # Compares aot_eager, inductor and eager; for static shapes asserts the AOT graph and the
        # inductor graph (slice nodes feeding the op).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; splits the 10 by 10 input into column
                # parts of width four and six, mutates both with the op and returns both parts and
                # the input.
                splits = x.split([4, 6], dim=1)
                a = splits[0]
                b = splits[1]
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(10, 10)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1);  getitem_3 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(getitem_3, [4, 6], 1);  getitem_3 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1);  arg0_1 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1)
        getitem_4: "f32[10, 4][10, 1]cpu" = split_with_sizes_1[0];  split_with_sizes_1 = None
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(arg0_1, [4, 6], 1);  arg0_1 = None
        getitem_7: "f32[10, 6][10, 1]cpu" = split_with_sizes_2[1];  split_with_sizes_2 = None
        return (getitem_4, getitem_7)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split_dynamic(self):
        # [FUNCTION DOC] Runs the split test in dynamic-shape mode; split specializes the input
        # sizes so no symbolic size shows up.
        self.test_split(_dynamic=True)

    # Test that slice view is generated instead of as_strided when slice is used.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice(self, _dynamic=False):
        # [FUNCTION DOC] V2 test that taking two slices of one input (rows zero to two, and column
        # three) and mutating both is encoded with slice metadata rather than strided views.
        # Dynamic parameter chooses the mode. Compares aot_eager, inductor and eager; for
        # static-by-default shapes asserts the AOT graph and the inductor graph.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; slices the input along dimension zero
                # (first two rows) and along dimension one (column three), mutates both slices with
                # the op, returns both and the input.
                a = torch.ops.aten.slice.Tensor(x, 0, 0, 2)
                b = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(10, 10)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[s77, s77][s77, 1]cpu", arg1_1: "Sym(s77)"):
        floordiv: "Sym(0)" = 0 // arg1_1;  arg1_1 = None
        add_6: "Sym(2)" = floordiv + 2
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = floordiv, _x_slice_end = add_6, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg0_1]);  floordiv = add_6 = None
        getitem_1: "f32[s77, s77][s77, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = 0, _x_slice_end = 2, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg0_1])
        getitem_1: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77, s77][s77, 1]cpu", arg1_1: "Sym(s77)"):
        floordiv: "Sym(0)" = 0 // arg1_1;  arg1_1 = None
        add_6: "Sym(2)" = floordiv + 2;  floordiv = add_6 = None
        slice_tensor: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_tensor_1: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4);  arg0_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_tensor_1: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4);  arg0_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice_dynamic(self):
        # [FUNCTION DOC] Runs the slice test in dynamic-shape mode, with the v2 flag on.
        self.test_slice(_dynamic=True)

    def test_try_use_slice(self):
        # [FUNCTION DOC] Unit test of try_use_slice from the auto_functionalize module, which tries
        # to express a tensor as a slice of its base and returns a triple of dimension, start and
        # end, or None. Round-trip checks: for full tensors and for pieces from splitting 2D, 3D
        # and cubic tensors along every dimension (negative and positive), re-slicing the base with
        # the returned triple must reproduce the piece. Negative checks: selecting a single element
        # or a row (rank change) and multi-dimension crops give None. Simple row slices round-trip.
        def test_round_trip(base, tensor):
            # [FUNCTION DOC] HELPER: Calls try_use_slice for the base and tensor, unpacks the
            # triple, slices the base with the aten slice op using it and asserts the result equals
            # the tensor.
            (dim, start, end) = try_use_slice(base, tensor)
            sliced = torch.ops.aten.slice.Tensor(base, dim, start, end)
            self.assertEqual(sliced, tensor)

        t = torch.tensor([[2, 2], [3, 4]])
        test_round_trip(t, t)

        for dim in range(-1, 1):
            f = t.split(2, dim)
            test_round_trip(t, f[0])

        for dim in range(-1, 1):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        t = torch.randint(1, 10, (3, 3, 3))
        test_round_trip(t, t)

        for dim in range(-3, 3):
            f = t.split([1, 2], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        for dim in range(-3, 3):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        t = torch.rand(10, 10, 10)
        test_round_trip(t, t)
        for dim in range(-3, 3):
            f = t.split([2, 2, 6], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        # example where slice won't work

        # selection
        t = torch.ones(10)
        b = t[0]
        self.assertEqual(try_use_slice(t, b), None)

        t = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(try_use_slice(t, t[0]), None)
        self.assertEqual(try_use_slice(t, t[1]), None)

        t = torch.tensor(
            [
                [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17]],
                [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]],
            ]
        )

        self.assertEqual(try_use_slice(t, t[0:1, 0:1, :7]), None)
        self.assertEqual(try_use_slice(t, t[0:1, 0:2, :3]), None)
        self.assertEqual(try_use_slice(t, t[0:2, 1, 0:8]), None)

        # simple slice operations are supported
        test_round_trip(t, t[0:2])
        test_round_trip(t, t[3:4])

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2(self, _dynamic=False):
        # [FUNCTION DOC] V2 test with one mutated arg an alias of the input and the other a tensor
        # whose shape has an unbacked symbolic size (clone, nonzero, float cast, slice). Dynamo
        # config allows dynamic output shape ops. Compares aot_eager, inductor and eager; for
        # static-by-default shapes asserts the AOT graph (runtime asserts on the unbacked size,
        # alias flags on both bases) and the inductor graph.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); applies sine in
                # place to x and y.
                x.sin_()
                y.sin_()

            def f(x):
                # [FUNCTION DOC] HELPER: Caller under test; aliases the input, builds a tensor with
                # an unbacked size from the nonzero indices of a clone, slices it, mutates the
                # alias and that slice with the op, and returns both.
                a = torch.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = torch.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                torch.ops.mylib.foo(a, d)
                return a, d

            orig_args = [torch.randn(2)]
            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        clone: "f32[s77][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = torch.ops.aten._to_copy.default(nonzero, dtype = torch.float32);  nonzero = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg0_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_1: "f32[s77][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        clone: "f32[2][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = torch.ops.aten._to_copy.default(nonzero, dtype = torch.float32);  nonzero = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg0_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_1: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(arg0_1)
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = torch.ops.prims.convert_element_type.default(nonzero, torch.float32);  nonzero = None
        alias_default: "f32[s77][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.alias.default(convert_element_type)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg0_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(arg0_1)
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = torch.ops.prims.convert_element_type.default(nonzero, torch.float32);  nonzero = None
        alias_default: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.alias.default(convert_element_type)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg0_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2_dynamic(self):
        # [FUNCTION DOC] Runs the alias-with-unbacked-size test in dynamic-shape mode, v2 flag on.
        self.test_alias2(_dynamic=True)

    # Test that the view regeneration optimizations do not result in recompilations. By comparing re-compilation in eager backend
    # with recompilation in inductor backend.
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_recompile(self):
        # [FUNCTION DOC] Checks that the v2 view-regeneration logic does not cause extra
        # recompilations relative to v1. For several caller shapes (two element views, two aliases,
        # last and first row, two slices, and an alias plus unbacked-shape tensor) it compiles each
        # with the compile counting backend using dynamic shapes over four input sizes, once with
        # v2 on and once with v2 off (dynamo reset between), and asserts both counts are equal to
        # an expected value. Duck shaping is disabled by config patch.
        # KNOWN ISSUE: a comment in the slice case cites a source line number of this file that is
        # stale; it is only informational.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); does nothing.
                pass

            def run_and_compare(func, expected=1):
                # [FUNCTION DOC] HELPER: Compiles the function with two separate counting inductor
                # backends, runs the v2 variant over four differently sized inputs, resets dynamo,
                # runs the v1 variant over the same inputs, then asserts both frame counts are
                # equal and equal to the expected count (default one).
                counter_v2 = CompileCounterWithBackend("inductor")
                counter_v1 = CompileCounterWithBackend("inductor")
                v1 = torch.compile(
                    func, backend=counter_v1, fullgraph=True, dynamic=True
                )

                v2 = torch.compile(
                    func, backend=counter_v2, fullgraph=True, dynamic=True
                )
                inputs = [
                    torch.rand(10, 10),
                    torch.rand(100, 100),
                    torch.rand(10, 2),
                    torch.rand(1000, 1000),
                ]

                with torch._inductor.config.patch(enable_auto_functionalized_v2=True):
                    for input in inputs:
                        v2(input)

                torch._dynamo.reset()

                with torch._inductor.config.patch(enable_auto_functionalized_v2=False):
                    for input in inputs:
                        v1(input)

                self.assertEqual(counter_v2.frame_count, counter_v1.frame_count)

                self.assertEqual(counter_v1.frame_count, expected)

            def func(x):
                # [FUNCTION DOC] HELPER: Case under test; mutates the first and second element
                # views of the input with the op.
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                # [FUNCTION DOC] HELPER: Case under test; mutates two alias views of the input with
                # the op.
                a = torch.ops.aten.alias.default(x)
                b = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                # [FUNCTION DOC] HELPER: Case under test; mutates the last row and the first row of
                # the input with the op.
                # last row
                a = x[x.size()[0] - 1]

                # first row
                b = x[0]
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                # [FUNCTION DOC] HELPER: Case under test; mutates two slices of the input (column
                # three, and rows one onward) with the op.
                a = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
                b = torch.ops.aten.slice.Tensor(x, 0, 1, 4)
                torch.ops.mylib.foo(a, b)

            # recompile here is not triggered by auto_functionalize
            # [__recompiles]     - 0/0: 4 <= L['x'].size()[1]  # a = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
            # test/inductor/test_auto_functionalize.py:1160 in func (_decomp/decompositions.py:781 in slice_forward)
            run_and_compare(func, 2)

            def func(x):
                # [FUNCTION DOC] HELPER: Case under test; mutates an alias of the input and a slice
                # of a tensor with an unbacked size, returns both.
                a = torch.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = torch.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                torch.ops.mylib.foo(a, d)
                return a, d

            with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
                run_and_compare(func, 1)

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    @unittest.skip(
        reason="This test fails because something else in inductor optimize out the alias. issue #137434"
    )
    def test_alias_id_input_to_custom_op(self):
        # [FUNCTION DOC] Skipped test (issue 137434: another inductor optimization removes the
        # alias). Intended to check that the alias form used for mutated args keeps the two op
        # arguments from being the same Python object. Defines an op whose CPU kernel asserts the
        # two argument ids differ, then compiles a caller passing an alias and the original input.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::not_eq",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::not_eq", "cpu", lib=lib)
            @torch._dynamo.disable
            def not_eq_impl(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); asserts the two
                # arguments are distinct objects.
                self.assertNotEqual(id(x), id(y))

            def func(x):
                # [FUNCTION DOC] HELPER: Caller under test; aliases the input and calls the op with
                # the alias and the input.
                a = torch.ops.aten.alias.default(x)
                torch.ops.mylib.not_eq(a, x)

            compiled = torch.compile(func, backend="inductor", fullgraph=True)
            compiled(torch.rand(2, 2))

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias_id_output(self):
        # [FUNCTION DOC] Checks that when the mutated arg is an alias of the input and the function
        # returns that alias, the compiled output is a different object from the input given to the
        # function. Compiles with inductor, calls it with a fresh random tensor, and asserts the
        # object ids of the output and a separately created tensor differ.
        # KNOWN ISSUE: the compared input is not the tensor that was passed to the compiled
        # function, so the assertion is trivially true.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); does nothing.
                pass

            def func(x):
                # [FUNCTION DOC] HELPER: Caller under test; aliases the input, calls the op on the
                # alias and the input, returns the alias.
                a = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, x)
                return a

            compiled = torch.compile(func, backend="inductor", fullgraph=True)
            input = torch.rand(2, 2)
            output = compiled(torch.rand(2, 2))
            self.assertNotEqual(id(output), id(input))

    def test_inference_mode_view(self):
        # [FUNCTION DOC] Regression test that a custom op declared to mutate its workspace
        # argument, whose output is viewed and passed through sine, compiles with aot_eager under
        # inference mode and produces the sine of the input.
        @torch.library.custom_op(
            "test_inference_mode_view::foo", mutates_args={"workspace"}
        )
        def foo(x: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation; returns a clone of x and leaves the
            # workspace untouched even though it is declared mutated.
            return x.clone()

        @foo.register_fake
        def _(x, workspace):
            # [FUNCTION DOC] HELPER: Fake kernel; returns a clone of x.
            return x.clone()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def f(x, w):
            # [FUNCTION DOC] HELPER: Compiled function; calls the op, flattens the result with a
            # view and returns its sine.
            y = foo(x, w)
            z = y.view(-1)
            return z.sin()

        x = torch.randn(2)
        w = torch.randn(2)
        with torch.inference_mode():
            y = f(x, w)
        self.assertEqual(y, x.sin())

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_scheduling_with_multiple_mutates(self):
        # [FUNCTION DOC] V2 scheduling test for an op mutating two tensors where one buffer is
        # reused across two calls. Compiles with inductor capturing the scheduler dependency log,
        # evaluates the logged text as a mapping from buffer names to users, and asserts the users
        # of the second empty buffer differ from those of the matmul output buffer, i.e. they were
        # not merged.
        with torch.library._scoped_library("mylib_scheduling", "FRAGMENT") as lib:
            torch.library.define(
                "mylib_scheduling::foo",
                "(Tensor! x, Tensor! y, Tensor z) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib_scheduling::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo(x, y, z):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); does nothing.
                pass

            def func(x, w):
                # [FUNCTION DOC] HELPER: Case under test; allocates two empty tensors, calls the op
                # on them and the input, multiplies the first by the weight, calls the op again
                # with that product, the second buffer and the input, and returns the product.
                a = torch.empty_like(x)  # buf0
                b = torch.empty_like(x)  # buf1
                torch.ops.mylib_scheduling.foo(a, b, x)  # buf2, buf3, buf4
                c = torch.mm(a, w)  # buf5
                torch.ops.mylib_scheduling.foo(c, b, x)  # buf6, buf7, buf8
                return c

            input = torch.rand(2, 2)
            weight = torch.rand(2, 2)
            [inductor_args, output, graph_inductor] = self.run_inductor(
                func,
                [input, weight],
                False,
                "torch._inductor.scheduler",
                "compute_dependencies",
            )
            name_to_users = eval(graph_inductor)
            self.assertNotEqual(name_to_users["buf1"], name_to_users["buf5"])

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dtype_view_clone_elimination(self):
        # [FUNCTION DOC] Checks the inductor pass that removes clones for dtype views (an int32
        # buffer viewed as float32) passed to a mutating op. Positive case: under inference mode
        # and under no-grad, the compiled result keeps int32 dtype, equals the eager result and the
        # pass counter is one. Negative cases (a full-range slice view, an intermediate buffer)
        # must match eager with the counter zero. Two mutations through one dtype view must match
        # eager with the counter one.
        """
        This test verifies that dtype views (tensors sharing storage but with
        different dtypes, e.g., int32 -> float32) have their clones eliminated
        when passed to mutating custom ops. The fix_auto_functionalized_dtype_views
        pass detects dtype views that alias graph inputs and removes unnecessary
        clone operations.

        Also tests negative cases where clones SHOULD be retained.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::mutate_inplace",
                "(Tensor(a!) x, Tensor y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::mutate_inplace", "cpu", lib=lib)
            @torch._dynamo.disable
            def mutate_inplace_cpu(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); copies y into
                # x.
                x.copy_(y)

            @torch.library.impl("mylib::mutate_inplace", "Meta", lib=lib)
            def mutate_inplace_meta(x, y):
                # [FUNCTION DOC] HELPER: Meta implementation of the op; does nothing.
                pass

            # Dtype view (int32 -> float32) - clone should be eliminated
            def f_dtype_view(cache_int32, data_float32):
                # [FUNCTION DOC] HELPER: Positive case; views an int32 input as float32, mutates it
                # with the op using the given data and returns the int32 input.
                cache_float = cache_int32.view(torch.float32)
                torch.ops.mylib.mutate_inplace(cache_float, data_float32)
                return cache_int32

            cache = torch.zeros((10, 10), dtype=torch.int32)
            data = torch.randn((10, 10), dtype=torch.float32)
            result_eager = f_dtype_view(cache.clone(), data)

            for mode_name, mode_ctx in [
                ("inference", torch.inference_mode()),
                ("no_grad", torch.no_grad()),
            ]:
                with mode_ctx:
                    counters.clear()
                    torch._dynamo.reset()
                    compiled_f = torch.compile(
                        f_dtype_view, fullgraph=True, backend="inductor"
                    )
                    result = compiled_f(cache.clone(), data)

                    self.assertEqual(result.dtype, torch.int32)
                    self.assertEqual(result, result_eager)
                    self.assertEqual(
                        counters["inductor"]["fix_auto_functionalized_dtype_views"], 1
                    )

            # Negative cases (slice view, intermediate tensor) - clone should be retained
            def f_slice(cache, data):
                # [FUNCTION DOC] HELPER: Negative case; slices the full range of the input, mutates
                # the slice with the op and returns the input.
                cache_slice = cache[0:10, 0:10]
                torch.ops.mylib.mutate_inplace(cache_slice, data)
                return cache

            def f_intermediate(data):
                # [FUNCTION DOC] HELPER: Negative case; allocates a zero int32 buffer, mutates its
                # float32 view with the op and returns the buffer.
                temp = torch.zeros((10, 10), dtype=torch.int32)
                torch.ops.mylib.mutate_inplace(temp.view(torch.float32), data)
                return temp

            for func, args in [
                (
                    f_slice,
                    [
                        torch.zeros((20, 20), dtype=torch.float32),
                        torch.randn((10, 10), dtype=torch.float32),
                    ],
                ),
                (f_intermediate, [torch.randn((10, 10), dtype=torch.float32)]),
            ]:
                result_eager_neg = func(*[a.clone() for a in args])

                with torch.no_grad():
                    counters.clear()
                    torch._dynamo.reset()
                    compiled_f = torch.compile(func, fullgraph=True, backend="inductor")
                    result = compiled_f(*[a.clone() for a in args])

                    self.assertEqual(result, result_eager_neg)
                    self.assertEqual(
                        counters["inductor"]["fix_auto_functionalized_dtype_views"], 0
                    )

            # Two mutations through the same dtype view: counter is 1, not 2 -
            # reinplace_inplaceable_ops already inplaces the second.
            def f_twice(cache, data1, data2):
                # [FUNCTION DOC] HELPER: Mutates the same float32 view of an int32 input with the
                # op twice, using two data tensors, and returns the input.
                v = cache.view(torch.float32)
                torch.ops.mylib.mutate_inplace(v, data1)
                torch.ops.mylib.mutate_inplace(v, data2)
                return cache

            data1 = torch.randn((10, 10), dtype=torch.float32)
            data2 = torch.randn((10, 10), dtype=torch.float32)
            cache_eager = cache.clone()
            f_twice(cache_eager, data1, data2)

            with torch.no_grad():
                counters.clear()
                torch._dynamo.reset()
                compiled_f = torch.compile(f_twice, fullgraph=True, backend="inductor")
                cache_compiled = cache.clone()
                compiled_f(cache_compiled, data1, data2)

                self.assertEqual(cache_compiled, cache_eager)
                self.assertEqual(
                    counters["inductor"]["fix_auto_functionalized_dtype_views"], 1
                )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dtype_view_clone_elimination_shared_storage(self):
        # [FUNCTION DOC] Checks the dtype-view pass when two graph inputs share one storage at
        # different offsets (slices of a pool), as in per-layer cache slices. Verifies the setup
        # shares storage with different offsets, compiles with inductor under no-grad and asserts
        # both outputs are int32 and match eager, the pool matches eager, and the pass counter is
        # two.
        """
        Two graph-input placeholders that share the same underlying storage
        (e.g. per-layer slices of a single KV pool) collide in the pass's
        ``storage_to_input`` map, which is keyed by
        ``untyped_storage()._cdata`` alone. Verify the pass still produces
        correct output for both mutations.

        This case is realistic for paged-KV deployments where each layer holds
        a slice of one big buffer with positive ``storage_offset``.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::mutate_inplace_pool",
                "(Tensor(a!) x, Tensor y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::mutate_inplace_pool", "cpu", lib=lib)
            @torch._dynamo.disable
            def mutate_inplace_pool_cpu(x, y):
                # [FUNCTION DOC] HELPER: CPU implementation (not traced by dynamo); copies y into
                # x.
                x.copy_(y)

            @torch.library.impl("mylib::mutate_inplace_pool", "Meta", lib=lib)
            def mutate_inplace_pool_meta(x, y):
                # [FUNCTION DOC] HELPER: Meta implementation of the op; does nothing.
                pass

            def f_pool_slices(slice_a, slice_b, data_a, data_b):
                # [FUNCTION DOC] HELPER: Views both int32 slices as float32, mutates each with the
                # op with its own data and returns the two int32 slices.
                a_view = slice_a.view(torch.float32)
                b_view = slice_b.view(torch.float32)
                torch.ops.mylib.mutate_inplace_pool(a_view, data_a)
                torch.ops.mylib.mutate_inplace_pool(b_view, data_b)
                return slice_a, slice_b

            pool = torch.zeros((4, 10, 10), dtype=torch.int32)
            # Two slices share `_cdata` but have different `storage_offset`.
            slice_a = pool[1]
            slice_b = pool[3]
            self.assertEqual(
                slice_a.untyped_storage()._cdata,
                slice_b.untyped_storage()._cdata,
            )
            self.assertNotEqual(slice_a.storage_offset(), slice_b.storage_offset())

            data_a = torch.randn((10, 10), dtype=torch.float32)
            data_b = torch.randn((10, 10), dtype=torch.float32)

            pool_eager = pool.clone()
            sa_eager, sb_eager = f_pool_slices(
                pool_eager[1], pool_eager[3], data_a, data_b
            )

            with torch.no_grad():
                counters.clear()
                torch._dynamo.reset()
                compiled_f = torch.compile(
                    f_pool_slices, fullgraph=True, backend="inductor"
                )
                pool_compiled = pool.clone()
                sa, sb = compiled_f(pool_compiled[1], pool_compiled[3], data_a, data_b)

                # Outputs must match eager for BOTH slices, regardless of the
                # pass's storage-collision identification order.
                self.assertEqual(sa.dtype, torch.int32)
                self.assertEqual(sb.dtype, torch.int32)
                self.assertEqual(sa, sa_eager)
                self.assertEqual(sb, sb_eager)
                # The underlying pool must reflect both mutations through the
                # shared storage.
                self.assertEqual(pool_compiled, pool_eager)
                # The pass must fire once per auto_functionalized_v2 node;
                # we have two dtype-view mutations so the counter is 2.
                self.assertEqual(
                    counters["inductor"]["fix_auto_functionalized_dtype_views"], 2
                )

    def test_reinplace_mutated_empty_no_self_edge(self):
        # [FUNCTION DOC] Regression test for a scheduler ancestor lookup failure seen with ops that
        # write into freshly allocated empty buffers which are then reshaped (like a flash
        # attention backward). Defines an op writing the reshaped gradient into three empty buffers
        # and returning a workspace, compiles a caller using dynamic shapes with a dynamic first
        # dim, and asserts the result equals the eager result.
        # Regression test for a Scheduler.compute_ancestors KeyError.
        #
        # An auto_functionalized custom op that mutates freshly allocated (empty)
        # output buffers whose results are then reshaped forces the reinplace pass
        # to fold a realization back onto the op's own buffer. That produced a
        # scheduler node with a non-weak MemoryDep on the buffer it also writes --
        # a self-edge -- and compute_ancestors looked up name_to_ancestors[op]
        # before it was populated, raising KeyError: 'opN'.
        #
        # Mirrors flash_attn._flash_attn_backward writing dq/dk/dv into empties.
        D = 512
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::attn_bwd",
                "(Tensor dout, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::attn_bwd", "cpu", lib=lib)
            def attn_bwd(dout, dq, dk, dv):
                # [FUNCTION DOC] HELPER: CPU implementation of the op; reshapes the first argument
                # to the shape of dq, copies it into dq, dk and dv, and returns a zeros tensor
                # shaped like dq.
                src = dout.reshape(dq.shape)
                dq.copy_(src)
                dk.copy_(src)
                dv.copy_(src)
                return torch.zeros_like(dq)  # workspace, discarded by caller

            @torch.library.register_fake("mylib::attn_bwd", lib=lib)
            def _(dout, dq, dk, dv):
                # [FUNCTION DOC] HELPER: Fake kernel; returns an empty tensor shaped like dq.
                return torch.empty_like(dq)

            def f(dout, weight, num_heads):
                # [FUNCTION DOC] HELPER: Caller under test; allocates three empty buffers for the
                # heads, calls the op, reshapes the buffers to the model width, and returns the
                # matrix product of their sum with the weight.
                head_dim = D // num_heads
                seqlen = dout.shape[0]
                dq = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                dk = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                dv = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                torch.ops.mylib.attn_bwd(dout, dq, dk, dv)  # return discarded
                a = dq.reshape(1, seqlen, D)
                b = dk.reshape(1, seqlen, D)
                c = dv.reshape(1, seqlen, D)
                return (a + b + c) @ weight

            dout = torch.randn(128, D)
            weight = torch.randn(D, D)
            torch._dynamo.mark_dynamic(dout, 0)

            expected = f(dout, weight, 8)
            got = torch.compile(f, fullgraph=True, dynamic=True)(dout, weight, 8)
            self.assertEqual(got, expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
