# Owner(s): ["module: dynamo"]
# [MODULE DOC]
# Dynamo tests for Python enum support under torch.compile: enums as dict keys, identity and
# equality, membership, length, iteration, lookup by name or value, IntEnum and Flag
# behavior, enums with custom constructors or metaclasses, and pybind11 enums (ReduceOp,
# DispatchKey). Each test compares compiled output with eager output, some also checking
# frame or op counts. All tests live in EnumTests; no import-time side effects beyond imports.

import enum
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same, skipIfNotPy312


class EnumTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC]
    # Dynamo test case collecting enum-related compile tests. Holds no state; each test
    # defines its own local enum classes and functions, compiles with a simple backend and
    # asserts equality with eager execution.
    """Tests for enum support in torch.compile."""

    def test_enum_as_dict_key(self):
        # [FUNCTION DOC]
        # Compiles a function that builds a dict keyed by enum members, a string and an int,
        # forces a graph break, then looks values up afterwards. Over ten random inputs the
        # compiled result must match eager, and exactly two frames must be compiled.
        class MyEnum(enum.Enum):
            # [CLASS DOC] HELPER: Plain enum with members FOO=10 and BAR=20 used as dict keys.
            FOO = 10
            BAR = 20

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 2 to x, builds the mixed-key dict, breaks the
            # graph, then returns the sum of two dict values and x plus 2 doubled.
            y = x + 2
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            torch._dynamo.graph_break()
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            b = y * 2
            return a, b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_as_dict_key_with_overloaded_str(self):
        # [FUNCTION DOC]
        # Same scenario as the plain dict-key test, but the enum overrides its string
        # conversion. The dict also has a plain string key equal to the default enum string,
        # so the test checks Dynamo does not confuse them. Compiled must match eager over ten
        # inputs with exactly two compiled frames.
        class MyEnum(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members FOO=10 and BAR=20 and a custom string
            # conversion method.
            FOO = 10
            BAR = 20

            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the member value directly.
                # KNOWN ISSUE: the value is an int, so real string conversion would raise a
                # type error; the test never triggers it.
                return self.value

        def fn(x):
            # [FUNCTION DOC] HELPER: Same body as in the plain dict-key test: build the
            # mixed-key dict, break the graph, return a dict-value sum and doubled x plus 2.
            y = x + 2
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            torch._dynamo.graph_break()
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            b = y * 2
            return a, b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_no_graphbreaks(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that branches on identity with an enum member.
        # With the first member the add and mul ops are recorded (two ops); after a reset,
        # with the second member only the mul remains (one op), via the op count.
        class Foo(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members FOO=0 and BAR=1.
            FOO = 0
            BAR = 1

        def fn(x, foo):
            # [FUNCTION DOC] HELPER: Adds 1 to x only if foo is the FOO member, then
            # multiplies by 1 and returns.
            if foo is Foo.FOO:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.FOO)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.BAR)
        self.assertEqual(cnts.op_count, 1)

    def test_enum_membership_check(self):
        # [FUNCTION DOC]
        # Checks the in and not-in operators against an Enum class with an enum member as
        # the operand, compiled with fullgraph and the eager backend; each compiled result
        # must equal the eager one.
        class Reduction(enum.Enum):
            # [CLASS DOC] HELPER: Enum with string-valued members SUM, MAX and MIN.
            SUM = "sum"
            MAX = "max"
            MIN = "min"

        def fn_enum_in(x, reduction):
            # [FUNCTION DOC] HELPER: Adds 1 to x if reduction is in the Reduction class.
            if reduction in Reduction:
                x = torch.add(x, 1.0)
            return x

        def fn_enum_not_in(x, reduction):
            # [FUNCTION DOC] HELPER: Raises ValueError if reduction is not in Reduction,
            # otherwise returns x multiplied by 2.
            if reduction not in Reduction:
                raise ValueError("Unknown reduction")
            x = torch.mul(x, 2.0)
            return x

        x = torch.randn(4)
        reduction = Reduction.SUM

        # Test `in` operator
        opt_fn = torch.compile(fn_enum_in, backend="eager", fullgraph=True)
        ref = fn_enum_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

        # Test `not in` operator
        opt_fn = torch.compile(fn_enum_not_in, backend="eager", fullgraph=True)
        ref = fn_enum_not_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

    @skipIfNotPy312
    def test_enum_membership_check_constant(self):
        # [FUNCTION DOC]
        # Runs only on Python 3.12 or newer. Same as the member membership test, but the
        # operand is the plain string equal to a member value, which on 3.12 counts as a
        # member. Compiled results with fullgraph must equal eager for both operators.
        class Reduction(enum.Enum):
            # [CLASS DOC] HELPER: Enum with string-valued members SUM, MAX and MIN.
            SUM = "sum"
            MAX = "max"
            MIN = "min"

        def fn_enum_in(x, reduction):
            # [FUNCTION DOC] HELPER: Adds 1 to x if reduction is in the Reduction class.
            if reduction in Reduction:
                x = torch.add(x, 1.0)
            return x

        def fn_enum_not_in(x, reduction):
            # [FUNCTION DOC] HELPER: Raises ValueError if reduction is not in Reduction,
            # otherwise returns x multiplied by 2.
            if reduction not in Reduction:
                raise ValueError("Unknown reduction")
            x = torch.mul(x, 2.0)
            return x

        x = torch.randn(4)
        reduction = "sum"

        # Test `in` operator for constants
        opt_fn = torch.compile(fn_enum_in, backend="eager", fullgraph=True)
        ref = fn_enum_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

        # Test `not in` operator for constants
        opt_fn = torch.compile(fn_enum_not_in, backend="eager", fullgraph=True)
        ref = fn_enum_not_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

    def test_enum_guards(self):
        # [FUNCTION DOC]
        # Compiles a function that compares its argument with an enum member using equality
        # and passes the other member; the compiled result must match eager.
        class MyEnum(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members FOO=10 and BAR=20.
            FOO = 10
            BAR = 20

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if y equals FOO, else x minus 1.
            if y == MyEnum.FOO:
                return x + 1
            else:
                return x - 1

        x = torch.rand(3)
        y = MyEnum.BAR
        ref = fn(x, y)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_enum_method(self):
        # [FUNCTION DOC]
        # Verifies Dynamo traces into a method defined on an IntEnum with fullgraph. The
        # method also adds 1 to its tensor argument so the result proves it was traced, not
        # just called. Compiled output must match eager.
        class Bool(enum.IntEnum):
            # [CLASS DOC] HELPER: IntEnum with auto-numbered members TRUE and FALSE and one
            # method.
            TRUE = enum.auto()
            FALSE = enum.auto()

            def is_true(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair: whether self is TRUE, and x plus 1.
                # Return `x + 1` to make sure Dynamo actually traced into this,
                # rather than invoking it.
                return self == Bool.TRUE, x + 1

        def f(x, e):
            # [FUNCTION DOC] HELPER: Calls the enum method; returns y plus 2 if it reports
            # true, else y minus 2, where y is x plus 1.
            cond, y = e.is_true(x)
            if cond:
                return y + 2
            else:
                return y - 2

        opt_f = torch.compile(fullgraph=True, backend="eager")(f)
        args = [torch.zeros(1), Bool.TRUE]
        ref_out = f(*args)
        opt_out = opt_f(*args)
        self.assertTrue(same(ref_out, opt_out))

    def test_enum_subclass(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that looks up a member of an IntEnum with a
        # custom constructor (modeled on the standard library parameter-kind enum) by
        # passing a member to the enum class, then returns cos of x. Result must equal eager.
        # Copied from inspect.py

        class _ParameterKind(enum.IntEnum):
            # [CLASS DOC] HELPER: IntEnum whose members take a description and get integer
            # values from the current member count; string conversion yields the name.
            POSITIONAL_ONLY = "positional-only"

            def __new__(cls, description):
                # [FUNCTION DOC] HELPER: Creates an int member whose value is the number of
                # existing members and stores the description on it; returns the member.
                value = len(cls.__members__)
                member = int.__new__(cls, value)
                member._value_ = value
                member.description = description
                return member

            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the member name.
                return self.name

        _POSITIONAL_ONLY = _ParameterKind.POSITIONAL_ONLY

        def fn(x):
            # [FUNCTION DOC] HELPER: Looks up the enum by member (result discarded) and
            # returns cos of x.
            _ParameterKind(_POSITIONAL_ONLY)
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_user_function_variable_supports_enum_argument(self):
        # [FUNCTION DOC]
        # Checks a function with an enum member as a default argument value can be called
        # from a compiled function with fullgraph, using the default; result must be close
        # to eager.
        class Foo(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members FOO=0 and BAR=1.
            FOO = 0
            BAR = 1

        def gn(x, y=Foo.FOO):
            # [FUNCTION DOC] HELPER: Returns x unchanged if y is FOO, else x plus 1.
            if y is Foo.FOO:
                return x
            else:
                return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the helper with only x so its default is used.
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_enum_membership_custom_metaclass(self):
        # [FUNCTION DOC]
        # Defines an enum whose metaclass overrides the containment check to always say no.
        # Asserts eager behavior first, then that the compiled function (fullgraph) honors
        # the override, so the else branch runs and the result equals x minus 1.
        """Test that custom metaclass __contains__ override is respected."""

        class CustomEnumMeta(enum.EnumMeta):
            # [CLASS DOC] HELPER: Enum metaclass overriding containment to always be false.
            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Ignores the item and always returns False.
                # Custom behavior: always return False
                return False

        class MyEnum(enum.Enum, metaclass=CustomEnumMeta):
            # [CLASS DOC] HELPER: Enum with members A=1 and B=2 using the custom metaclass.
            A = 1
            B = 2

        # Verify eager behavior
        self.assertFalse(MyEnum.A in MyEnum)
        self.assertFalse(1 in MyEnum)

        def fn(x, member):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if member is contained in MyEnum,
            # else x minus 1.
            if member in MyEnum:
                return x + 1
            return x - 1

        x = torch.randn(4)
        # With custom metaclass, membership should always be False
        ref = fn(x, MyEnum.A)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, MyEnum.A)
        self.assertEqual(ref, res)
        # Should return x - 1 since custom __contains__ returns False
        self.assertEqual(res, x - 1)

    def test_enum_len(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function adding the number of members of an enum class
        # to x; result must equal eager.
        """Test len() on Enum class."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus the member count of Color.
            return x + len(Color)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_iter(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that sums member values by iterating the enum
        # class two ways (direct loop and explicit iterator method); result must equal eager.
        """Test iterating over Enum class."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds all member values twice (two iteration styles)
            # into a total and returns x plus that total.
            total = 0
            for color in Color:
                total += color.value
            for color in Color.__iter__():
                total += color.value
            return x + total

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_list(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that converts an enum class to a list and adds
        # its length to x; result must equal eager.
        """Test list(Enum) to get all members."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the member list and returns x plus its length.
            colors = list(Color)
            return x + len(colors)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_getitem(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that fetches an enum member by its name string
        # and branches on it; result must equal eager.
        """Test Enum.__getitem__ (access by name)."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Looks up RED by name; returns x plus 1 if it equals the
            # RED member (always), else x.
            color = Color["RED"]
            if color == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_reversed(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that lists the enum members in reverse order
        # and branches on the first and last; result must equal eager.
        """Test reversed() on Enum class."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if the reversed member list starts with
            # BLUE and ends with RED, else x.
            colors = list(reversed(Color))
            # Check order is reversed: BLUE, GREEN, RED
            if colors[0] == Color.BLUE and colors[2] == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_call(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that gets an enum member by its value and
        # branches on it; result must equal eager.
        """Test Enum(value) to get member by value."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Looks up the member with value 1 (RED); returns x plus 1
            # if it is RED, else x.
            color = Color(1)
            if color == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_name_value_properties(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function reading the name and value attributes of an
        # enum member passed in as argument; result must equal eager.
        """Test accessing .name and .value properties of enum members."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED=1, GREEN=2, BLUE=3.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x, color):
            # [FUNCTION DOC] HELPER: If the member name is RED returns x plus its value,
            # otherwise x.
            if color.name == "RED":
                return x + color.value
            return x

        x = torch.randn(4)
        ref = fn(x, Color.RED)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Color.RED)
        self.assertEqual(ref, res)

    def test_int_enum_arithmetic(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function adding an IntEnum member to a tensor; result
        # must equal eager.
        """Test IntEnum arithmetic operations."""

        class Priority(enum.IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members LOW=1, MEDIUM=2, HIGH=3.
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x, priority):
            # [FUNCTION DOC] HELPER: Returns x plus the integer-valued priority member.
            return x + priority

        x = torch.randn(4)
        ref = fn(x, Priority.HIGH)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.HIGH)
        self.assertEqual(ref, res)

    def test_flag_enum(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function testing whether one flag is set within a
        # combined flag value (the containment check on a flag value, not on the class);
        # called with READ and WRITE combined. Result must equal eager.
        """Test Flag enum operations."""

        # It checks if a flag is set in a combined value, not if it's a member of the Flag class.
        # This requires different handling in dynamo.

        class Permission(enum.Flag):
            # [CLASS DOC] HELPER: Flag enum with READ=1, WRITE=2, EXECUTE=4.
            READ = 1
            WRITE = 2
            EXECUTE = 4

        def fn(x, perm):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if READ is set in perm, else x.
            if Permission.READ in perm:
                return x + 1
            return x

        x = torch.randn(4)
        combined = Permission.READ | Permission.WRITE
        ref = fn(x, combined)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, combined)
        self.assertEqual(ref, res)

    def test_flag_enum_membership_combined(self):
        # [FUNCTION DOC]
        # Calls a compiled function with two different combined flag values, resetting
        # Dynamo between them. With MEAN set it must return the mean, otherwise the sum.
        """Test Flag enum membership check with combined flags."""

        class LocalReduction(enum.Flag):
            # [CLASS DOC] HELPER: Flag enum with auto-valued members MEAN, SUM, MAX.
            MEAN = enum.auto()
            SUM = enum.auto()
            MAX = enum.auto()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, local_reduction):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph, eager backend). Returns the mean of
            # x if MEAN is set in local_reduction, otherwise the sum.
            if LocalReduction.MEAN in local_reduction:
                return x.mean()
            return x.sum()

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(fn(x, LocalReduction.MEAN | LocalReduction.SUM), x.mean())
        torch._dynamo.reset()
        self.assertEqual(fn(x, LocalReduction.SUM | LocalReduction.MAX), x.sum())

    def test_enum_comparison(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that compares two enum members by equality and
        # identity, run first with the same member twice and, after a reset, with different
        # members. Each compiled result must equal eager.
        """Test enum comparison operations."""

        class Priority(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members LOW=1, MEDIUM=2, HIGH=3.
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if a equals b, else x plus 2 if they
            # are identical (unreachable), else x minus 1.
            if a == b:
                return x + 1
            elif a is b:
                return x + 2
            else:
                return x - 1

        x = torch.randn(4)

        # Same member
        ref = fn(x, Priority.LOW, Priority.LOW)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.LOW, Priority.LOW)
        self.assertEqual(ref, res)

        # Different members
        torch._dynamo.reset()
        ref = fn(x, Priority.LOW, Priority.HIGH)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.LOW, Priority.HIGH)
        self.assertEqual(ref, res)

    @unittest.expectedFailure
    def test_metaclass_custom_call(self):
        # [FUNCTION DOC]
        # Known-failing test (marked as an expected failure). A class with a metaclass whose
        # call hook skips the initializer is built inside a compiled function; Python scales
        # the value by the class factor, but Dynamo currently runs the default construction
        # path, so compiled output differs from eager and the assertion fails.
        # When a class has a metaclass that overrides __call__, Python invokes
        # the metaclass's __call__ instead of the default type.__call__
        # (which does __new__ + __init__). Dynamo currently bypasses the
        # metaclass __call__ and uses instantiate_user_defined_class_object
        # (which mimics type.__call__), producing wrong results.
        class ScalingMeta(type):
            # [CLASS DOC] HELPER: Metaclass overriding instance creation to bypass the
            # initializer.
            def __call__(cls, val):
                # [FUNCTION DOC] HELPER: Creates an instance without running the initializer,
                # sets its scaled attribute to val times the class scale factor, returns it.
                obj = cls.__new__(cls)
                obj.scaled = val * cls.scale_factor
                return obj

        class Scaled(metaclass=ScalingMeta):
            # [CLASS DOC] HELPER: Class using the scaling metaclass, with scale factor 10.
            scale_factor = 10

            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores val unscaled; never run by the metaclass
                # creation hook, so only a wrong construction path would call it.
                # type.__call__ would call this, but ScalingMeta.__call__
                # deliberately skips it and sets scaled directly.
                self.scaled = val

        def fn(x, val):
            # [FUNCTION DOC] HELPER: Builds a Scaled object from val and returns x times its
            # scaled attribute.
            obj = Scaled(val)
            return x * obj.scaled

        x = torch.tensor([1.0, 2.0])
        # Python: ScalingMeta.__call__ sets scaled = 3 * 10 = 30
        ref = fn(x, 3)
        compiled_fn = torch.compile(fn, backend="eager")
        res = compiled_fn(x, 3)
        self.assertEqual(ref, res)

    def test_enum_construction_no_extra_init(self):
        # [FUNCTION DOC]
        # Checks that looking up an enum member by value inside a compiled function does not
        # run the enum initializer (a counter list tracks calls). Eager gives a factor of
        # one; the compiled result must match. Not marked expected-failure here.
        # Real-world instance of the metaclass __call__ issue above.
        # EnumMeta.__call__ only calls __new__ (value lookup), NOT __init__.
        # Dynamo's instantiate_user_defined_class_object calls both, so the
        # extra __init__ call produces wrong results.
        init_count = [0]

        class MyEnum(enum.Enum):
            # [CLASS DOC] HELPER: Enum with custom constructor and initializer; members A=1,
            # B=2. The initializer increments a shared counter.
            def __new__(cls, val):
                # [FUNCTION DOC] HELPER: Creates a bare member object and sets its value.
                obj = object.__new__(cls)
                obj._value_ = val
                return obj

            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Increments the shared counter list element.
                init_count[0] += 1

            A = 1
            B = 2

        init_count[0] = 0

        def fn(x, val):
            # [FUNCTION DOC] HELPER: Looks up the member for val, discards it, and returns x
            # times one plus the counter value.
            MyEnum(val)
            return x * (1 + init_count[0])

        x = torch.tensor([1.0, 2.0])
        # Python: MyEnum(1) is a value lookup via EnumMeta.__call__, which
        # only calls Enum.__new__ (lookup) — __init__ is NOT called.
        # So init_count stays 0 and fn returns x * 1.
        ref = fn(x, 1)
        init_count[0] = 0

        compiled_fn = torch.compile(fn, backend="eager")
        res = compiled_fn(x, 1)
        self.assertEqual(ref, res)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "torch.distributed not available",
    )
    def test_pybind11_enum_as_dict_key(self):
        # [FUNCTION DOC]
        # Skipped without torch.distributed. Maps three ReduceOp values to integers in a
        # dict, and for each key compiles (fullgraph, with a reset first) a lookup function;
        # compiled must equal eager.
        """Test pybind11 enum (RedOpType) works as dict key without graph break."""
        import torch.distributed as dist

        op_mapping = {
            dist.ReduceOp.SUM: 0,
            dist.ReduceOp.MAX: 1,
            dist.ReduceOp.AVG: 2,
        }

        def fn(x, op):
            # [FUNCTION DOC] HELPER: Returns x plus the integer mapped to op.
            return x + op_mapping[op]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for op in (dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.AVG):
            torch._dynamo.reset()
            ref = fn(x, op)
            res = opt_fn(x, op)
            self.assertEqual(ref, res)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "torch.distributed not available",
    )
    def test_pybind11_enum_equality(self):
        # [FUNCTION DOC]
        # Skipped without torch.distributed. Compiles with fullgraph an equality test of a
        # ReduceOp argument with the premul-sum op, run with SUM and then, after a reset,
        # with PREMUL_SUM; each compiled result must equal eager.
        """Test pybind11 enum comparison without graph break."""
        import torch.distributed as dist

        def fn(x, op):
            # [FUNCTION DOC] HELPER: Returns x plus 2 if op equals PREMUL_SUM, else x plus 1.
            if op == dist.ReduceOp.PREMUL_SUM:
                return x + 2
            return x + 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x, dist.ReduceOp.SUM)
        res = opt_fn(x, dist.ReduceOp.SUM)
        self.assertEqual(ref, res)

        torch._dynamo.reset()
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, dist.ReduceOp.PREMUL_SUM)
        res = opt_fn(x, dist.ReduceOp.PREMUL_SUM)
        self.assertEqual(ref, res)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "torch.distributed not available",
    )
    def test_pybind11_enum_identity(self):
        # [FUNCTION DOC]
        # Skipped without torch.distributed. Compiles with fullgraph an identity test of a
        # ReduceOp argument against the SUM op and checks the result equals eager.
        """Test pybind11 enum identity check without graph break."""
        import torch.distributed as dist

        def fn(x, op):
            # [FUNCTION DOC] HELPER: Returns x times 2 if op is the SUM op, else x plus 1.
            if op is dist.ReduceOp.SUM:
                return x * 2
            return x + 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, dist.ReduceOp.SUM)
        res = opt_fn(x, dist.ReduceOp.SUM)
        self.assertEqual(ref, res)

    def test_flag_enum_invert(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that inverts a flag value and tests a member
        # in the result, passing READ; compiled output must equal eager.
        class Permission(enum.Flag):
            # [CLASS DOC] HELPER: Flag enum with READ=1, WRITE=2, EXECUTE=4.
            READ = 1
            WRITE = 2
            EXECUTE = 4

        def fn(x, perm):
            # [FUNCTION DOC] HELPER: Inverts perm; returns x plus 1 if WRITE is in the
            # inverted flags, else x minus 1.
            inverted = ~perm
            if Permission.WRITE in inverted:
                return x + 1
            return x - 1

        x = torch.randn(4)
        ref = fn(x, Permission.READ)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Permission.READ)
        self.assertEqual(ref, res)

    def test_dispatch_key_as_dict_key(self):
        # [FUNCTION DOC]
        # Builds a dict keyed by two DispatchKey values (a pybind11 enum) and compiles with
        # fullgraph a function indexing it with a key argument; result must equal eager.
        """Test DispatchKey (also a pybind11 enum) works as dict key."""
        d = {
            torch.DispatchKey.CPU: 0,
            torch.DispatchKey.CUDA: 1,
        }

        def fn(x, key):
            # [FUNCTION DOC] HELPER: Returns x plus the integer mapped to the key.
            return x + d[key]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, torch.DispatchKey.CPU)
        res = opt_fn(x, torch.DispatchKey.CPU)
        self.assertEqual(ref, res)

    def test_dispatch_key_sourceless(self):
        # [FUNCTION DOC]
        # Compiles with fullgraph a function that creates DispatchKey values inside its own
        # body (no external source) based on the device type and branches on them; result
        # must equal eager.
        """Test DispatchKey used as a literal inside the compiled function."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Picks the CPU key for CPU tensors else the CUDA key,
            # then returns x plus 1 if the key is CPU, else x minus 1.
            if x.device.type == "cpu":
                key = torch.DispatchKey.CPU
            else:
                key = torch.DispatchKey.CUDA
            # Use the key in a branch to ensure it's traced, not eliminated.
            if key == torch.DispatchKey.CPU:
                return x + 1
            return x - 1

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
