# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for torch.export.draft_export, which exports a program while recording
# (instead of raising on) problems such as missing or mismatched fake kernels, data dependent
# errors and added guards, and attaches a report to the result. The single test class here
# covers report rendering, retrying with automatic dims on constraint violations, custom op
# fake kernel profile generation, data dependent branches, side effects on inputs and buffers,
# torchbind inputs, AOTI packaging and CUDA peak memory. No import-time side effects besides
# imports; running the file starts the standard test runner.
import copy
import re
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.exc import UserError, UserErrorType
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Dim, draft_export, export
from torch.export._draft_export import DraftExportReport, FailureReport, FailureType
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)
from torch.utils._pytree import tree_leaves


class TestDraftExport(TestCase):
    # [CLASS DOC] Test case for draft_export. setUp initializes the torchbind test
    # implementations and stores the three torchbind queue operator handles in an attribute;
    # tearDown intentionally does nothing. Most tests build a small module (often with a
    # scoped custom op library), call draft_export, and inspect the report attached to the
    # result (its failure list, failure types, op profiles) and compare the exported module
    # against eager execution.
    def setUp(self):
        # [FUNCTION DOC] Initializes torchbind test implementations after the base setUp and
        # records the queue pop, push and size torchbind operators in an attribute list.
        super().setUp()
        init_torchbind_implementations()

        self.torch_bind_ops = [
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_size,
        ]

    def tearDown(self):
        # [FUNCTION DOC] Overrides tearDown to do nothing (skips the base class cleanup).
        return

    def test_report_warning_color_is_readable_on_light_background(self):
        # [FUNCTION DOC] Builds a report holding one missing-fake-kernel failure and checks
        # its string form: it must contain the red ANSI color code, must not contain the
        # bright yellow code, and the color reset code must appear before the first numbered
        # failure entry.
        report = DraftExportReport(
            [
                FailureReport(
                    FailureType.MISSING_FAKE_KERNEL,
                    {"op": "mylib.foo.default"},
                )
            ],
            {},
            {},
            {},
        )

        rendered = str(report)
        self.assertIn("\033[31m", rendered)
        self.assertNotIn("\033[93m", rendered)
        self.assertLess(
            rendered.index("\033[0m"),
            rendered.index("1. Missing fake kernel."),
        )

    def test_report_success_color_is_readable_on_light_background(self):
        # [FUNCTION DOC] Builds a report with no failures and checks that its string form uses
        # the green ANSI color and not the bright green one.
        report = DraftExportReport([], {}, {}, {})

        rendered = str(report)
        self.assertIn("\033[32m", rendered)
        self.assertNotIn("\033[92m", rendered)

    def test_retry_on_constraint_violation_uses_dim_auto(self):
        # [FUNCTION DOC] Verifies the retry path: the first call of the internal export
        # function is patched to raise a constraint violation user error, so draft_export must
        # call it again. Asserts exactly two calls were made, the second call's dynamic shape
        # for the input dimension is an automatic dim that keeps the original min and max of
        # the named Dim.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        dim = Dim("d0", min=2, max=16)
        dynamic_shapes = {"x": {0: dim}}
        calls = []

        import torch.export._draft_export as draft_export_mod

        real_export = draft_export_mod._export

        def patched_export(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper around the real internal export function. Records
            # a deep copy of the dynamic shapes it receives; on the first call it raises a
            # constraint violation user error, afterwards it forwards to the real function and
            # returns its result.
            calls.append(copy.deepcopy(kwargs["dynamic_shapes"]))
            if len(calls) == 1:
                raise UserError(
                    UserErrorType.CONSTRAINT_VIOLATION,
                    "mocked constraint violation",
                )
            return real_export(*args, **kwargs)

        with patch("torch.export._draft_export._export", side_effect=patched_export):
            draft_export(
                M(),
                (torch.randn(3),),
                dynamic_shapes=dynamic_shapes,
            )

        self.assertEqual(len(calls), 2)
        first_dim = calls[0]["x"][0]
        second_dim = calls[1]["x"][0]
        self.assertEqual(getattr(second_dim, "min", None), first_dim.min)
        self.assertEqual(getattr(second_dim, "max", None), first_dim.max)
        self.assertIn("AUTO", repr(second_dim))

    def test_shared_storage_parameters_with_offset(self):
        # [FUNCTION DOC] Checks draft_export on a module whose two parameters are views into
        # one shared storage at different offsets, with a dynamic batch dim. Asserts the
        # report is successful and the exported module matches eager output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters created as views of different
            # halves of one random tensor (shared storage with offset); forward adds the sum
            # of both to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the shared base tensor and the two parameter
                # views.
                super().__init__()
                base = torch.randn(12)
                self.p0 = torch.nn.Parameter(base[:6].view(2, 3))
                self.p1 = torch.nn.Parameter(base[6:].view(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the sum of both parameters.
                return x + self.p0.sum() + self.p1.sum()

        mod = M()
        x = torch.randn(2, 3)

        ep = draft_export(
            mod,
            (x,),
            dynamic_shapes={"x": {0: Dim("batch", min=1, max=4)}},
        )
        self.assertTrue(ep._report.successful())
        self.assertEqual(ep.module()(x), mod(x))

    def test_missing_meta_kernel_custom_op_basic(self):
        # [FUNCTION DOC] Uses a custom op defined without any fake kernel inside a scoped
        # library. Asserts the report has exactly one failure of the missing fake kernel type,
        # the exported module matches eager output on new inputs, and run_decompositions
        # succeeds when run under the context that generates fake kernels from the report op
        # profiles.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo2", mutates_args={})
            def foo2_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Eager implementation of the custom op; returns the
                # sum of its inputs.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the custom op once on its two inputs.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Returns the custom op applied to both inputs.
                    res = torch.ops.mylib.foo2(a, b)
                    return res

            inp = (torch.ones(3, 3), torch.ones(3, 3))

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_missing_meta_kernel_impl(self):
        # [FUNCTION DOC] Same missing-fake-kernel scenario as above but the op is defined with
        # explicit define and a CPU impl in a scoped library. The module calls the op twice.
        # Asserts one failure of the missing fake kernel type, matching eager output, exactly
        # one op profile entry for the op with a single profile (both calls share shapes),
        # prints the profiles, and checks the program after run_decompositions under generated
        # fake kernels.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: CPU implementation of the op; returns the sum of its
                # inputs.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying the op twice, feeding the first result
                # into the second.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Returns the op applied twice, the second time
                    # with the second input.
                    res = torch.ops.mylib.foo(a, b)
                    res = torch.ops.mylib.foo(res, b)
                    return res

            inp = (torch.ones(3, 3), torch.ones(3, 3))

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo.default"]), 1)
            print(report.op_profiles)

            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep = ep.run_decompositions()
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_missing_meta_kernel_custom_op_multiple_profiles(self):
        # [FUNCTION DOC] Calls the same fake-kernel-less custom op with two different input
        # shapes. Asserts one missing fake kernel failure, one op in the profiles and two
        # distinct profiles recorded for it, then runs run_decompositions under generated fake
        # kernels.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo3", mutates_args={})
            def foo3_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Eager implementation of the custom op; returns the
                # sum of inputs.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying the custom op to two pairs of inputs,
                # returning both.
                def forward(self, a, b, c, d):
                    # [FUNCTION DOC] HELPER: Returns a tuple with the op applied to the first
                    # pair and the second.
                    res1 = torch.ops.mylib.foo3(a, b)
                    res2 = torch.ops.mylib.foo3(c, d)
                    return res1, res2

            inp = (
                torch.ones(3, 4),
                torch.ones(3, 4),
                torch.ones(2, 3, 4),
                torch.ones(2, 3, 4),
            )

            ep = draft_export(M(), inp)
            report = ep._report

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo3.default"]), 2)

            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_missing_meta_kernel_custom_op_update_profile(self):
        # [FUNCTION DOC] Checks that profiles of a custom op are replaced by a later draft
        # export. After the first export there is one op profile. Under generated fake kernels
        # from that report, the op works on the original shapes and raises a RuntimeError
        # saying no profiles match for the new shapes; draft exporting with the new shapes
        # inside that context yields a single profile again, and afterwards the registration
        # accepts the new shapes and rejects the old ones.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo8", mutates_args={})
            def foo8_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Eager implementation of the custom op; returns the
                # sum of inputs.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying the custom op to its two inputs.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Returns the custom op applied to both inputs.
                    res = torch.ops.mylib.foo8(a, b)
                    return res

            inp = (
                torch.ones(3, 4),
                torch.ones(3, 4),
            )

            ep = draft_export(M(), inp)
            report = ep._report
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo8.default"]), 1)

            new_inp = (
                torch.ones(2, 3, 4),
                torch.ones(2, 3, 4),
            )

            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()):
                    torch.ops.mylib.foo8(*inp)
                    with self.assertRaisesRegex(
                        RuntimeError, "no profiles match the given inputs"
                    ):
                        torch.ops.mylib.foo8(*new_inp)

                ep = draft_export(M(), new_inp)

            report = ep._report
            self.assertEqual(len(report.op_profiles), 1)
            self.assertEqual(len(report.op_profiles["mylib.foo8.default"]), 1)

            with (
                torch._library.fake_profile.unsafe_generate_fake_kernels(
                    report.op_profiles
                ),
                FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()),
            ):
                torch.ops.mylib.foo8(*new_inp)

                # Existing registration has been updated to match the new
                # profile traced with draft-export
                with self.assertRaisesRegex(
                    RuntimeError, "no profiles match the given inputs"
                ):
                    torch.ops.mylib.foo8(*inp)

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_missing_meta_kernel_guard(self):
        # [FUNCTION DOC] Skipped without CUDA. Draft exports a custom op module with all dims
        # dynamic using CPU float inputs, then checks the exported module works on matching
        # inputs and that its input checks raise RuntimeError for CUDA inputs (device mismatch
        # message) and for half precision inputs (dtype mismatch message).
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo4", mutates_args={})
            def foo4_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Eager implementation of the custom op; returns the
                # sum of inputs.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying the custom op to its two inputs.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Returns the custom op applied to both inputs.
                    res1 = torch.ops.mylib.foo4(a, b)
                    return res1

            inp = (
                torch.ones(3, 4),
                torch.ones(3, 4),
            )

            ep = draft_export(
                M(),
                inp,
                dynamic_shapes={
                    "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                    "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                },
            )

            inp = (torch.randn(2, 3), torch.randn(2, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))
            m = ep.module()
            with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
                bad_device_inps = (
                    torch.randn(2, 3, device=torch.device("cuda")),
                    torch.randn(2, 3, device=torch.device("cuda")),
                )
                m(*bad_device_inps)

            with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
                bad_dtype_inps = (
                    torch.randn(2, 3, dtype=torch.float16),
                    torch.randn(2, 3, dtype=torch.float16),
                )
                m(*bad_dtype_inps)

    def test_fake_infer_dense_in_memory_check(self):
        # [FUNCTION DOC] Checks the dense-in-memory validation applied to custom op outputs
        # during draft export. Defines three ops (one returns a dense tensor, one a non-dense
        # slice, one a slice with nonzero storage offset) and a module choosing one by an
        # integer option. Dense outputs of several shapes export fine; the non-dense and the
        # offset cases raise RuntimeError with messages saying the return was not dense in
        # memory, respectively had a non-zero storage offset.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo5", mutates_args={})
            def foo5_impl(a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Op implementation returning the input doubled (a
                # dense tensor).
                return a * 2

            @torch.library.custom_op("mylib::foo6", mutates_args={})
            def foo6_impl(a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Op implementation returning a doubled slice that
                # drops the last row and column, so the result is not dense in memory.
                return (a * 2)[:, :-1, :-1]  # not dense in memory

            @torch.library.custom_op("mylib::foo7", mutates_args={})
            def foo7_impl(a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Op implementation returning a doubled slice that
                # skips the first and last entries of dim one, so the result has a non-zero
                # storage offset.
                return (a * 2)[:, 1:-1, :]  # non-zero storage offset

            class Foo(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that calls one of the three ops depending on an
                # integer argument (zero, one, or anything else).
                def forward(self, x, opt):
                    # [FUNCTION DOC] HELPER: Branches on the integer option and returns the
                    # matching op output.
                    if opt == 0:
                        return torch.ops.mylib.foo5(x)
                    elif opt == 1:
                        return torch.ops.mylib.foo6(x)
                    else:
                        return torch.ops.mylib.foo7(x)

            draft_export(Foo(), (torch.randn(80, 4, 4), 0))
            draft_export(Foo(), (torch.randn(80, 1, 4), 0))
            draft_export(Foo(), (torch.randn(1, 4, 1, 1, 4, 1, 4), 0))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return was not dense in memory",
            ):
                draft_export(Foo(), (torch.randn(4, 6, 8), 1))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return has a non-zero storage offset",
            ):
                draft_export(Foo(), (torch.randn(4, 6, 8), 2))

    def test_data_dependent_failure(self):
        # [FUNCTION DOC] Uses an op without fake kernel plus a branch on a scalar tensor item
        # to slice the result. Asserts the report has at least two failures, the first a
        # missing fake kernel and the second a data dependent error, and the exported module
        # matches eager output for other values.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            def foo_impl(a, b):
                # [FUNCTION DOC] HELPER: CPU implementation of the op; returns the sum of its
                # inputs.
                return a + b

            class M(torch.nn.Module):
                # [FUNCTION DOC] HELPER: Module that applies the op, reads an item from the
                # third input and, when positive, returns the result sliced up to that item.
                def forward(self, a, b, c):
                    # [FUNCTION DOC] HELPER: Applies the op, converts the third input to a
                    # Python number and returns a slice of the result when the number is
                    # positive (otherwise returns nothing).
                    res = torch.ops.mylib.foo1(a, b)

                    c_item = c.item()
                    if c_item > 0:
                        return res[:c_item]

            inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(3))

            ep = draft_export(M(), inp)
            report = ep._report
            self.assertTrue(len(report.failures) > 0)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[1].failure_type, FailureType.DATA_DEPENDENT_ERROR
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3), torch.tensor(2))
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_unbacked_div_mod_replacement(self):
        # [FUNCTION DOC] Draft exports a module building a zeros tensor sized by an input
        # item, repeating it and viewing it with an inferred dim and a fixed last dim. Asserts
        # the report has no failures, i.e. the unbacked division and modulo reasoning
        # succeeded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating a zeros vector from an input item, repeating
            # it and reshaping with one inferred dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the reshaped repeated zeros tensor.
                x = torch.zeros(x.item())
                x = x.unsqueeze(0).repeat(10, 2)
                return x.view(-1, 2, 2345)

        ep = draft_export(M(), (torch.tensor([938]),))
        report = ep._report
        self.assertEqual(len(report.failures), 0)

    def test_dedup_data_dependent_failure(self):
        # [FUNCTION DOC] Checks that repeated data dependent branches at the same source
        # location (a loop over two inputs) are reported as a single failure of the data
        # dependent error type. Also checks the exported module matches eager on other values
        # and that every placeholder's fake value in node metadata carries a real tensor.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module looping over two inputs, branching on each one's item
            # to pick multiplication or addition, then scaling the accumulated result by the
            # third input.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the third input times the accumulated result.
                res = 0
                for v in [x, y]:
                    b = v.item()
                    if b > 10:
                        res += v * b
                    else:
                        res += v + b

                return z * res

        inp = (torch.tensor(5), torch.tensor(3), torch.tensor(2))

        ep = draft_export(M(), inp)
        report = ep._report
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )

        inp = (torch.tensor(4), torch.tensor(2), torch.tensor(6))
        self.assertEqual(ep.module()(*inp), M()(*inp))

        # the fake tensors on node.meta["val"] should have real_tensor
        gm = ep.module()
        tensors = [
            node.meta.get("val").real_tensor
            for node in gm.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in tensors))

    def test_complex_data_dependent_expr(self):
        # [FUNCTION DOC] Exports a module that does arithmetic on an item (negate, floor
        # divide, add) then slices a concatenated tensor by it, with the second input's dim
        # dynamic. Asserts at least one failure and that the first is a data dependent error.
        # For both the program and its decomposed form, checks that exactly one unbacked
        # symbol appears across the node metadata bindings.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module deriving an index from an item by negation, floor
            # division and addition, and slicing a doubled-up second input by it when
            # positive.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the slice when the derived index is positive;
                # otherwise returns nothing.
                a = x.item()
                a = -a
                a = a // 3
                a = a + 5

                z = torch.cat([y, y])

                if a > 0:
                    return z[:a]

        ep = draft_export(
            M(),
            (torch.tensor(6), torch.randn(5)),
            dynamic_shapes={"x": None, "y": {0: Dim.DYNAMIC}},
        )
        report = ep._report
        self.assertTrue(len(report.failures) > 0)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )
        for _ep in [ep, ep.run_decompositions()]:
            # unbacked bindings
            unbacked_binding_symbols = set()
            for node in _ep.graph.nodes:
                if bindings := node.meta.get("unbacked_bindings"):
                    unbacked_binding_symbols.update(bindings.keys())
            self.assertEqual(len(unbacked_binding_symbols), 1)

    def test_offsets(self):
        # [FUNCTION DOC] Smoke test: draft exports a module that raises a RuntimeError only
        # when an item equals zero; the input is nonzero so export must complete without
        # error.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that raises RuntimeError when the item is zero,
            # otherwise returns the input times the item.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Raises on a zero item, else returns the scaled input.
                a = x.item()
                if a == 0:
                    raise RuntimeError("bad")
                return x * a

        inp = (torch.tensor(3),)
        draft_export(M(), inp)

    def test_shape_failure(self):
        # [FUNCTION DOC] Draft exports a module asserting its first dim equals three, with
        # that dim dynamic and deferred runtime asserts preferred over guards. Asserts a
        # single failure of the guard added type and eager match. Running the exported module
        # with a first dim of four raises AssertionError whose message says a guard on the
        # size failed.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that asserts the first dim is three and returns the
            # input squared elementwise.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Asserts the first dimension size is three, returns
                # input times itself.
                assert a.shape[0] == 3  # noqa: S101
                return a * a

        inp = (torch.ones(3, 3),)

        ep = draft_export(
            M(),
            inp,
            dynamic_shapes={"a": {0: Dim("a0")}},
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        report = ep._report

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].failure_type, FailureType.GUARD_ADDED)

        inp = (torch.randn(3, 3),)
        self.assertEqual(ep.module()(*inp), M()(*inp))

        inp = (torch.randn(4, 3),)
        with self.assertRaisesRegex(
            AssertionError,
            re.escape("Guard failed: a.size()[0] <= 3"),
        ):
            # expected <= 3, but got 4
            ep.module()(*inp)

    def test_side_effect1(self):
        # [FUNCTION DOC] Checks that draft export does not mutate module state. A buffer is
        # read as an item twice with branches around an in-place add. After draft export the
        # buffer still equals its initial value, and the exported graph code contains no add
        # operator calls.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer buffer that is branched on,
            # incremented in place, and branched on again to scale the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the integer buffer holding two.
                super().__init__()
                self.register_buffer("a", torch.tensor(2))

            def forward(self, b):
                # [FUNCTION DOC] HELPER: Reads the buffer item, scales the input based on its
                # value, increments the buffer in place, reads it again and scales the result
                # again.
                a_item = self.a.item()
                if a_item == 2:
                    res = a_item * b
                else:
                    res = (a_item + 1) * b

                self.a.add_(1)
                a_item = self.a.item()

                if a_item == 3:
                    res = a_item * res
                else:
                    res = (a_item + 1) * res
                return res

        inp = (torch.ones(3, 3),)
        mod = M()
        ep = draft_export(mod, inp)
        self.assertEqual(mod.a, torch.tensor(2))
        FileCheck().check_count("torch.ops.aten.add.default", 0, exactly=True).run(
            ep.graph_module.code
        )

    def test_side_effect_inps(self):
        # [FUNCTION DOC] Draft exports a module that applies an in-place sine to its input.
        # Asserts the report is successful and that the caller's input tensor is unchanged
        # afterward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward mutates its input in place with sine
            # and returns it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies in-place sine to the input and returns it.
                x.sin_()
                return x

        inp = (torch.ones(3, 3),)
        ep = draft_export(M(), inp)
        report = ep._report
        self.assertTrue(report.successful())
        self.assertEqual(inp[0], torch.ones(3, 3))

    def test_masked_linear(self):
        # [FUNCTION DOC] Draft exports a module that selects entries by a mask with a data
        # dependent result size and applies a linear layer, then compares with the regular
        # export. Both exported modules must give equal outputs for the original mask and
        # after setting more mask entries.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that boolean-indexes the input along the first dim
            # with a mask and applies a linear function with the given weight and bias.
            def forward(self, x, mask, weight, bias):
                # [FUNCTION DOC] HELPER: Returns the linear transform of the masked rows.
                masked = x[mask != 0, :, :]
                return torch.nn.functional.linear(masked, weight, bias)

        x = torch.zeros(10)
        x[0] += 1
        inp = (torch.randn(10, 8, 7), x, torch.randn(25, 7), torch.randn(25))
        draft_ep = draft_export(M(), inp)
        ep = export(M(), inp)
        self.assertEqual(draft_ep.module()(*inp), ep.module()(*inp))
        x[2] += 1
        x[3] += 1
        self.assertEqual(draft_ep.module()(*inp), ep.module()(*inp))

    def test_torchbind(self):
        # [FUNCTION DOC] Draft exports a module taking a torchbind tensor queue object. The
        # queue is deep copied before and after pushes. Asserts the report is successful and
        # the copies keep their sizes, with the original queue left at size two (not consumed
        # by tracing).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer that pops from the passed queue,
            # branches on whether it is empty, and returns two tensors plus the queue.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, tq, x):
                # [FUNCTION DOC] HELPER: Uses the queue pop, float size, size and emptiness
                # queries to build the outputs; returns the two computed outputs
                # and the queue.
                x_cos = tq.pop() + tq.float_size() + self.linear(x)
                if tq.is_empty():
                    x_sin = self.linear(tq.pop()) - tq.size() + x
                else:
                    x_sin = tq.pop() + tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = _empty_tensor_queue()
        tq2 = copy.deepcopy(tq)
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        tq.push(a)
        tq.push(b)
        tq3 = copy.deepcopy(tq)
        inp = (tq, torch.randn(2, 2))
        ep = draft_export(mod, inp)
        report = ep._report
        self.assertTrue(report.successful())
        self.assertEqual(tq2.size(), 0)
        self.assertEqual(tq3.size(), 2)
        self.assertEqual(tq.size(), 2)

    def test_override_size_and_dtype_mismatched_fake_kernels(self):
        # [FUNCTION DOC] Registers a custom op whose fake kernel is wrong in two outputs (a
        # size mismatch and a dtype mismatch) and checks that regular strict export with real
        # tensor propagation raises RuntimeError, while draft export succeeds. Exported
        # outputs match eager values and dtypes. The report has two mismatched fake kernel
        # failures with the expected reason strings (dtype and size), and run_decompositions
        # works under generated fake kernels.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the three-output custom op.
                def forward(self, a):
                    # [FUNCTION DOC] HELPER: Returns the custom op applied to the input.
                    return torch.ops.mylib.foo9(a)

            @torch.library.custom_op("mylib::foo9", mutates_args={})
            def foo(a: torch.Tensor) -> list[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Real kernel returning a list of three tensors: the
                # doubled input, the input repeated twice along both dims, and the input
                # converted to bfloat16.
                x = a * 2
                y = a.repeat(2, 2)
                z = a.to(torch.bfloat16)
                return [x, y, z]

            @torch.library.register_fake("mylib::foo9")
            def foo_fake_impl(a):
                # [FUNCTION DOC] HELPER: Fake kernel returning three tensors shaped and typed
                # like the input, which is wrong for the second (size) and third (dtype)
                # outputs.
                x = torch.empty_like(a)  # good
                y = torch.empty_like(a)  # size mismatch
                z = torch.empty_like(a)  # dtype mismatch
                return [x, y, z]

            mod = M()
            inputs = (torch.randn(3, 3),)
            with self.assertRaises(RuntimeError):
                with torch._functorch.config.patch(
                    fake_tensor_propagate_real_tensors=True
                ):
                    export(mod, inputs, strict=True)

            ep = draft_export(mod, inputs)
            report = ep._report
            for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
                self.assertTrue(torch.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 2)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[1].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                sorted([f.data["reason"] for f in report.failures]),
                [
                    "Dtypes torch.bfloat16 and torch.float32 are not equal!",
                    "mismatch between fake value 3 and real value 6 ",
                ],
            )

            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    def test_override_incorrectly_aliasing_kernel(self):
        # [FUNCTION DOC] Registers an op whose fake kernel returns its input itself as one
        # output (incorrect aliasing). Strict export with real tensor propagation must raise
        # RuntimeError about an aliasing mismatch; draft export succeeds with outputs matching
        # eager, and the report holds one mismatched fake kernel failure whose reason mentions
        # a mismatched aliasing spec between fake and real kernels.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo10", mutates_args={})
            def foo(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Real kernel returning two new tensors: the input
                # doubled and the input plus two.
                return a * 2, a + 2

            @torch.library.register_fake("mylib::foo10")
            def foo_fake_impl(a):
                # [FUNCTION DOC] HELPER: Fake kernel returning the input itself and an empty
                # tensor, which wrongly aliases the first output with the input.
                return a, torch.empty_like(a)  # incorrectly aliasing

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the two-output op.
                def forward(self, a):
                    # [FUNCTION DOC] HELPER: Returns the op output pair for the input.
                    return torch.ops.mylib.foo10(a)

            mod = M()
            inputs = (torch.randn(3, 3),)
            with self.assertRaisesRegex(
                RuntimeError,
                "Real tensor propagation found an aliasing mismatch",
            ):
                with torch._functorch.config.patch(
                    fake_tensor_propagate_real_tensors=True
                ):
                    export(mod, inputs, strict=True)

            ep = draft_export(mod, inputs)
            report = ep._report
            for ep_out, eager_out in zip(
                tree_leaves(ep.module()(*inputs)), tree_leaves(mod(*inputs))
            ):
                self.assertTrue(torch.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertTrue(
                "Mismatched aliasing spec between fake kernel and real kernel"
                in report.failures[0].data["reason"]
            )

    def test_override_mismatched_fake_kernel_with_unbacked_symbols(self):
        # [FUNCTION DOC] Registers an op whose real kernel indexes by an item and returns
        # bfloat16, while the fake kernel returns a float tensor with a new dynamic size.
        # Draft export must succeed, outputs match eager (values and dtype), and the report
        # has one mismatched fake kernel failure whose reason is the dtype mismatch message.
        # Then run_decompositions is run under generated fake kernels.
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo11", mutates_args={})
            def foo11(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Real kernel selecting a row of the first input by the
                # item of the second and converting it to bfloat16.
                return a[b.item()].to(torch.bfloat16)

            @torch.library.register_fake("mylib::foo11")
            def foo_fake_impl(a, b):
                # [FUNCTION DOC] HELPER: Fake kernel allocating a new dynamic size from the
                # library context and returning an empty tensor of that size by the input's
                # second dim and input dtype.
                ctx = torch.library.get_ctx()
                u = ctx.new_dynamic_size()
                return torch.empty(u, a.shape[1], dtype=a.dtype)

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the op on its two inputs.
                def forward(self, a, b):
                    # [FUNCTION DOC] HELPER: Returns the op applied to both inputs.
                    return torch.ops.mylib.foo11(a, b)

            mod = M()
            inputs = (torch.randn(100, 4), torch.tensor(10))

            ep = draft_export(mod, inputs)

            report = ep._report
            for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
                self.assertTrue(torch.allclose(ep_out, eager_out))
                self.assertEqual(ep_out.dtype, eager_out.dtype)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[0].data["reason"],
                "Dtypes torch.bfloat16 and torch.float32 are not equal!",
            )
            with torch._library.fake_profile.unsafe_generate_fake_kernels(
                report.op_profiles
            ):
                ep.run_decompositions()

    # https://github.com/pytorch/pytorch/issues/140625
    @unittest.skipIf(IS_WINDOWS, "aoti_compile_and_package not supported on Windows")
    def test_constantify_unbacked_symbol(self):
        # [FUNCTION DOC] Regression test for an issue about unbacked symbols becoming
        # constants (linked in the comment above). Skipped on Windows. Draft exports a module
        # that builds a tensor from a shape, takes an item and uses it as an arange length,
        # then compiles and packages the program with AOTI to a temporary file; passes if that
        # does not raise.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that turns the input shape into a tensor, reads its
            # first entry as a number and multiplies the second input by a range of that
            # length.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the second input times a range sized from the
                # first input's first dimension.
                xt = torch.tensor(x.shape)
                u0 = xt[0].item()
                return y * torch.arange(u0)

        mod = M()
        example_inputs = (torch.randn(3, 5), torch.randn(3))
        draft_ep = draft_export(mod, example_inputs)
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            torch._inductor.aoti_compile_and_package(
                draft_ep,
                package_path=f.name,
            )

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 2**28,
        "Requires 16 MB GPU memory to pass the test; setting it higher to catch violations",
    )
    def test_cuda_memory_usage(self):
        # [FUNCTION DOC] Skipped without CUDA or with under 256 MB GPU memory. Regression test
        # that draft export of a long chain of adds on a one-million element CUDA tensor stays
        # within four times the input's memory above the baseline, measured with peak
        # allocated bytes after resetting peak stats. Non-strict mode is used.
        # This used to OOM
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a small constant to the input one hundred
            # times in a loop.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the repeated add loop and returns the result.
                for _ in range(100):
                    x = x + 1e-3
                return x

        # measure base usage
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        base_usage = torch.cuda.memory_allocated(device)

        # usage with input tensor allocated
        x = torch.randn(2**10, 2**10).to(device)
        x_usage = torch.cuda.memory_allocated(device)

        # draft export peak memory usage
        draft_export(Foo(), (x,), strict=False)
        peak_mem_usage = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]

        # right now it's actually exactly 4x;
        # I guess original tensor, 2 tensors per add op, 1 for clone stored in node.meta["val"]
        self.assertTrue((peak_mem_usage - base_usage) <= (x_usage - base_usage) * 4.0)


if __name__ == "__main__":
    run_tests()
