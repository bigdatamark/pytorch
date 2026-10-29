# Owner(s): ["module: custom-operators"]
# [MODULE DOC] Tests for the inplace tag on custom operators: schema validation when
# defining an op tagged as inplace, detection through is_inplace, auto-generated fake
# kernels, and compile behavior (functionalization graphs) for an add_ op. Import-time side
# effect: defines a library _TestInplaceTag with op add_ and registers its eager and meta
# implementations. Defines TestInplaceTag.
import torch
from torch import Tensor
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._library.utils import is_inplace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


_test_lib = torch.library.Library("_TestInplaceTag", "DEF")  # noqa: SCOPED_LIBRARY

_test_lib.define(
    "add_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    tags=[torch.Tag.inplace],
)


def _add_inplace_impl(self_: Tensor, other: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: in-place add: adds other into self_ and returns self_ (registered
    # as the eager kernel).
    self_.add_(other)
    return self_


_test_lib.impl("add_", _add_inplace_impl, "CompositeExplicitAutograd")
_test_lib.impl("add_", lambda self_, other: self_, "Meta")


@skipIfTorchDynamo("custom operator tests not applicable to dynamo")
class TestInplaceTag(TestCase):
    # [CLASS DOC] Test cases for the inplace tag; each test gets a fresh fragment of the
    # _TestInplaceTag library as self.lib and destroys it afterwards. Skipped under dynamo.
    def setUp(self):
        # [FUNCTION DOC] HELPER: creates a FRAGMENT library handle for _TestInplaceTag
        # in self.lib.
        super().setUp()
        self.lib = torch.library.Library("_TestInplaceTag", "FRAGMENT")  # noqa: SCOPED_LIBRARY

    def tearDown(self):
        # [FUNCTION DOC] HELPER: destroys the library fragment so ops defined by the test are
        # removed, then calls the base.
        self.lib._destroy()
        super().tearDown()

    def test_basic_inplace(self):
        # [FUNCTION DOC] The add_ op defined at module level is detected as inplace.
        self.assertTrue(is_inplace(torch.ops._TestInplaceTag.add_.default))

    def test_auto_fake_kernel_inplace(self):
        # [FUNCTION DOC] Defines an inplace-tagged op with a CPU kernel that adds in place and
        # no explicit fake kernel; under fake tensor mode a call must return the very input
        # tensor object.
        self.lib.define(
            "auto_fake_add_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
            tags=[torch.Tag.inplace],
        )
        self.lib.impl(
            "auto_fake_add_",
            lambda self_, other: self_.add_(other),
            "CPU",
        )

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(3, 4)
            y = torch.randn(3, 4)
            result = torch.ops._TestInplaceTag.auto_fake_add_(x, y)
            self.assertIs(result, x)

    def test_is_inplace_native(self):
        # [FUNCTION DOC] is_inplace is true for aten abs_ and false for aten abs, the out
        # overload and the functional add overload.
        # Hand-written inplace op
        self.assertTrue(is_inplace(torch.ops.aten.abs_.default))
        self.assertFalse(is_inplace(torch.ops.aten.abs.default))
        # Out op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.abs.out))
        # Functional op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.add.Tensor))

    def test_no_positional_args(self):
        # [FUNCTION DOC] Defining an inplace-tagged op with only keyword-only arguments raises
        # ValueError requiring at least one positional argument.
        with self.assertRaisesRegex(ValueError, "at least one positional argument"):
            self.lib.define(
                "no_args(*, Tensor(a!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_mutable(self):
        # [FUNCTION DOC] Inplace-tagged op whose first positional argument is not mutable
        # raises ValueError.
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be mutable"
        ):
            self.lib.define(
                "not_mutable(Tensor self, Tensor other) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_tensor(self):
        # [FUNCTION DOC] Inplace-tagged op whose first positional argument is a list of tensors
        # rather than a Tensor raises ValueError.
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be a Tensor"
        ):
            self.lib.define(
                "not_tensor(Tensor(a!)[] self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count(self):
        # [FUNCTION DOC] Inplace-tagged op returning nothing raises ValueError (exactly one
        # return required).
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "no_return(Tensor(a!) self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count_multiple(self):
        # [FUNCTION DOC] Inplace-tagged op returning two values raises ValueError (exactly one
        # return required).
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "multi_return(Tensor(a!) self) -> (Tensor(a!), Tensor)",
                tags=[torch.Tag.inplace],
            )

    def test_return_not_aliased(self):
        # [FUNCTION DOC] Inplace-tagged op whose return is not an alias of the mutated argument
        # raises ValueError.
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "bad_alias(Tensor(a!) self) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_return_not_tensor(self):
        # [FUNCTION DOC] Inplace-tagged op whose return type is not a Tensor raises ValueError.
        with self.assertRaisesRegex(ValueError, "return must be a Tensor"):
            self.lib.define(
                "bad_return_type(Tensor(a!) self) -> int(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_return_not_mutable_alias(self):
        # [FUNCTION DOC] Inplace-tagged op whose return is a non-mutable alias
        # raises ValueError.
        with self.assertRaisesRegex(ValueError, "return must be a mutable alias"):
            self.lib.define(
                "bad_return_mutability(Tensor(a!) self) -> Tensor(a)",
                tags=[torch.Tag.inplace],
            )

    def test_return_wrong_alias(self):
        # [FUNCTION DOC] Inplace-tagged op returning the second mutable argument instead of the
        # first raises ValueError.
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "wrong_alias(Tensor(a!) self, Tensor(b!) other) -> Tensor(b!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_positional_arg(self):
        # [FUNCTION DOC] Inplace-tagged op with a second mutable positional argument
        # raises ValueError.
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_mut(Tensor(a!) self, Tensor(b!) other) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_kwarg(self):
        # [FUNCTION DOC] Inplace-tagged op with a mutable keyword-only argument
        # raises ValueError.
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_kwarg(Tensor(a!) self, *, Tensor(b!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_compile_inplace(self, backend):
        # [FUNCTION DOC] Parametrized by backend (aot_eager, inductor). Compiles a call of
        # add_ with full graph, and checks both the returned value and the mutated input
        # equal the sum.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: calls the add_ op on x and y.
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        expected = x + y

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)
        self.assertEqual(result, expected)
        self.assertEqual(x_compiled, expected)

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_compile_inplace_duplicated_base(self, backend):
        # [FUNCTION DOC] Parametrized by backend. Compiles add_ where both arguments are the
        # same tensor and checks the result and mutated input equal twice the input.
        # add_(x, x): self and other are the same tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: calls add_ with x as both arguments.
            return torch.ops._TestInplaceTag.add_(x, x)

        x = torch.randn(3, 4)
        expected = x * 2

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled)
        self.assertEqual(result, expected)
        self.assertEqual(x_compiled, expected)

    def test_compile_inplace_functionalized_graph(self):
        # [FUNCTION DOC] Compiles add_ with the aot-eager graph recorder and checks results,
        # then compares the forward graph text with the expected auto-functionalized call
        # followed by a copy back to the input.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: calls the add_ op on x and y.
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)

        self.assertEqual(result, x + y)
        self.assertEqual(x_compiled, x + y)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestInplaceTag.add_.default, other = arg1_1, _self_base_index = 0, _all_bases = [arg0_1]);  arg1_1 = None
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
    return (getitem_1,)""",
        )

    def test_compile_inplace_view_functionalized_graph(self):
        # [FUNCTION DOC] Compiles add_ applied to a slice view of a tensor; checks the slice is
        # updated, the rest is unchanged, and the forward graph text uses slice dim, start and
        # end arguments.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: calls add_ on rows one to three of x and y.
            return torch.ops._TestInplaceTag.add_(x[1:3], y)

        x = torch.randn(5, 4)
        y = torch.randn(2, 4)

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)

        self.assertEqual(result, x[1:3] + y)
        self.assertEqual(x_compiled[1:3], x[1:3] + y)
        # Unsliced parts are unchanged
        self.assertEqual(x_compiled[0], x[0])
        self.assertEqual(x_compiled[3:], x[3:])

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestInplaceTag.add_.default, other = arg1_1, _self_base_index = 0, _self_slice_dim = 0, _self_slice_start = 1, _self_slice_end = 3, _all_bases = [arg0_1]);  arg1_1 = None
    getitem = auto_functionalized_v2[0]
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
    return (getitem,)""",
        )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_compile_inplace_uses_v2_hop_when_config_off(self):
        # [FUNCTION DOC] With auto functionalization version two disabled by config, the
        # recorded forward graph still uses the v2 higher-order op and not the v1 one; results
        # are also checked.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: calls the add_ op on x and y.
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)

        backend = AotEagerAndRecordGraphs()
        x_compiled = x.clone()
        result = torch.compile(fn, backend=backend, fullgraph=True)(x_compiled, y)

        self.assertEqual(result, x + y)
        self.assertEqual(x_compiled, x + y)
        self.assertIn("auto_functionalized_v2", backend.fw_graphs[0].code)
        self.assertNotIn(
            "torch.ops.higher_order.auto_functionalized(",
            backend.fw_graphs[0].code,
        )


instantiate_parametrized_tests(TestInplaceTag)

if __name__ == "__main__":
    run_tests()
