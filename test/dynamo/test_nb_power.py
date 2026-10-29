# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the power operator (nb_power slot) run through make_dynamo_test.
# Covers ints, floats, complex, bools, error cases, in-place power, user-defined forward, reflected
# and in-place power methods, int subclass priority, NotImplemented fallbacks, the three-argument
# builtin pow with user-defined and subclass operands, and compiled tensor/symbolic-size power. No
# import-time side effects beyond decorator application.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbPower(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for power under Dynamo. The class decorators enable tracing of unittest
    # and of class construction (load build class) for every test, so classes defined inside test
    # bodies are traced. Tests cover builtin numbers, user-defined classes, subclass dispatch,
    # ternary pow and compiled SymNode and tensor power.
    # --- Integer power ---

    @make_dynamo_test
    def test_pow_integers(self):
        # [FUNCTION DOC] Integer powers (including zero to the zero power) give the expected ints.
        self.assertEqual(2**10, 1024)
        self.assertEqual(3**3, 27)
        self.assertEqual(1**100, 1)
        self.assertEqual(0**0, 1)

    @make_dynamo_test
    def test_pow_negative_exponent(self):
        # [FUNCTION DOC] Negative integer exponents give float results.
        self.assertEqual(2**-1, 0.5)
        self.assertEqual(4**-1, 0.25)

    @make_dynamo_test
    def test_pow_chained(self):
        # [FUNCTION DOC] Chained power is right-associative, so two to the power of two to the
        # power of three is 256.
        self.assertEqual(2**2**3, 256)

    # --- Floats ---

    @make_dynamo_test
    def test_pow_floats(self):
        # [FUNCTION DOC] Float powers including a square root via exponent one half.
        self.assertEqual(2.0**3.0, 8.0)
        self.assertEqual(9.0**0.5, 3.0)

    @make_dynamo_test
    def test_pow_mixed_int_float(self):
        # [FUNCTION DOC] Mixed int and float operands give float results.
        self.assertEqual(2**3.0, 8.0)
        self.assertEqual(2.0**3, 8.0)

    # --- Complex ---

    @make_dynamo_test
    def test_pow_complex(self):
        # [FUNCTION DOC] Complex number powers with int exponents.
        self.assertEqual((1 + 1j) ** 2, 2j)
        self.assertEqual((2 + 0j) ** 3, (8 + 0j))

    # --- Booleans ---

    @make_dynamo_test
    def test_pow_bools(self):
        # [FUNCTION DOC] Power with bool operands behaves like int power.
        self.assertEqual(True**True, 1)
        self.assertEqual(False**True, 0)
        self.assertEqual(True**False, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_pow_zero_base_negative_exp_raises(self):
        # [FUNCTION DOC] Zero raised to a negative exponent raises ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            0**-1

    @make_dynamo_test
    def test_pow_str_raises(self):
        # [FUNCTION DOC] Power with a string as either operand raises TypeError.
        with self.assertRaises(TypeError):
            2 ** "a"
        with self.assertRaises(TypeError):
            "a" ** 2

    # --- Inplace **= ---

    @make_dynamo_test
    def test_inplace_pow_integers(self):
        # [FUNCTION DOC] In-place power on an int variable.
        x = 2
        x **= 8
        self.assertEqual(x, 256)

    @make_dynamo_test
    def test_inplace_pow_chained(self):
        # [FUNCTION DOC] Two successive in-place power operations on an int variable.
        x = 2
        x **= 3
        x **= 2
        self.assertEqual(x, 64)

    # --- User-defined __pow__ ---

    @make_dynamo_test
    def test_user_defined_pow_basic(self):
        # [FUNCTION DOC] Defines a local class with a forward power method that wraps the result of
        # raising the values to each other, plus equality; checks power between two instances.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value; its forward power takes another
            # instance and its equality compares class and value.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: Forward power: returns a new instance whose value is own
                # value to the power of the other instance's value.
                return type(self)(self.value**other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: other is the same exact type and has equal
                # value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(2) ** C(10), C(1024))

    @make_dynamo_test
    def test_user_defined_pow_with_integer(self):
        # [FUNCTION DOC] Defines a local class with a forward power method accepting a raw int
        # exponent; checks instance power int.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward power taking a raw
            # exponent.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: Forward power: returns a new instance whose value is
                # raised to the raw exponent.
                return type(self)(self.value**other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(2) ** 10, C(1024))

    @make_dynamo_test
    def test_reversed_pow_with_integer(self):
        # [FUNCTION DOC] Defines a local class with only a reflected power method; checks int power
        # instance uses it.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with a reflected power method.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: Reflected power: returns a new instance with the raw base
                # raised to own value.
                return type(self)(other**self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(2 ** C(10), C(1024))

    @make_dynamo_test
    def test_inplace_user_defined_pow(self):
        # [FUNCTION DOC] Defines a local class with an in-place power method that mutates and
        # returns self; checks the in-place operator updates the instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with an in-place power method.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __ipow__(self, other):
                # [FUNCTION DOC] HELPER: In-place power: raises own value to the other instance's
                # value, mutating self, and returns self.
                self.value **= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        x = C(2)
        x **= C(10)
        self.assertEqual(x, C(1024))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding forward and reflected power to return marker
        # strings; checks the override is used both as left and right operand.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose power methods return marker strings.
            def __pow__(self, other, modulo=None):
                # [FUNCTION DOC] HELPER: Forward power: returns the forward marker string; accepts
                # an optional modulo argument that is ignored.
                return "IntSub.__pow__"

            def __rpow__(self, other, modulo=None):
                # [FUNCTION DOC] HELPER: Reflected power: returns the reflected marker string;
                # accepts an optional modulo argument that is ignored.
                return "IntSub.__rpow__"

        self.assertEqual(IntSub(2) ** 10, "IntSub.__pow__")
        self.assertEqual(2 ** IntSub(10), "IntSub.__rpow__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline plain class with forward and reflected power returning markers,
        # used on either side of an int.
        class Base:
            # [CLASS DOC] HELPER: Local class whose power methods return marker strings.
            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: Forward power: returns the forward marker string.
                return "Base.__pow__"

            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: Reflected power: returns the reflected marker string.
                return "Base.__rpow__"

        self.assertEqual(Base() ** 1, "Base.__pow__")
        self.assertEqual(1 ** Base(), "Base.__rpow__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_pow_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] A class whose forward and reflected power return NotImplemented raises
        # TypeError when raised to itself.
        class C:
            # [CLASS DOC] HELPER: Local class whose power methods both return NotImplemented.
            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: Forward power: always NotImplemented.
                return NotImplemented

            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: Reflected power: always NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a**a

    @make_dynamo_test
    def test_pow_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Left forward power declines, so the right operand's reflected power result
        # string is returned.
        class A:
            # [CLASS DOC] HELPER: Local class whose forward power returns NotImplemented.
            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: Forward power: always NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Local class that defines only a reflected power returning a
            # marker.
            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: Reflected power: returns the marker string.
                return "B.__rpow__ called"

        result = A() ** B()
        self.assertEqual(result, "B.__rpow__ called")

    # --- 3-arg pow ---

    @make_dynamo_test
    def test_three_arg_pow_constants(self):
        # [FUNCTION DOC] Three-argument builtin pow with ints computes modular exponentiation.
        self.assertEqual(pow(2, 10, 1000), 24)
        self.assertEqual(pow(3, 4, 7), 4)

    @make_dynamo_test
    def test_three_arg_pow_user_defined(self):
        # [FUNCTION DOC] Three-argument pow with user-defined instances: forward power receives the
        # modulus as third argument and the result is reduced by it.
        # __pow__ receives z as third argument
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with a forward power taking an
            # optional modulus.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: Forward power: with a modulus, returns the plain int
                # result of the modular power; without it returns a new instance.
                if mod is not None:
                    return (self.value**other.value) % mod.value
                return type(self)(self.value**other.value)

        self.assertEqual(pow(C(2), C(10), C(1000)), 24)

    @make_dynamo_test
    def test_three_arg_pow_subtype_as_base(self):
        # [FUNCTION DOC] Int subclass as the base of ternary pow: its forward power gets the
        # modulus and returns a marker string.
        # Sub is the BASE (left operand); __pow__ is called with z.
        # Python does NOT call Sub.__pow__ when Sub is the exponent (right operand)
        # for ternary pow — no __rpow__ for 3-arg form.
        class Sub(int):
            # [CLASS DOC] HELPER: Int subclass whose forward power handles only the ternary form.
            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: Forward power: returns a marker string when a modulus is
                # given, else NotImplemented.
                if mod is not None:
                    return "Sub handled"
                return NotImplemented

        y = Sub(3)
        self.assertEqual(pow(y, 5, 7), "Sub handled")

    @make_dynamo_test
    def test_three_arg_pow_subtype_as_exponent(self):
        # [FUNCTION DOC] Int subclass as the exponent with plain int base and modulus: asserts the
        # result is computed as base to exponent mod modulus (equal to one), not with swapped
        # operands.
        # Sub is the EXPONENT (right operand), base is a plain int.
        # type(Sub) is a subtype of type(int) so ternary_op tries Sub's slot
        # first (subtype priority) with reverse=True: must compute pow(2, Sub(3), 7)
        # = 1, not pow(Sub(3), 2, 7) = 2.
        class Sub(int):
            # [CLASS DOC] HELPER: Empty int subclass used as the exponent.
            pass

        self.assertEqual(pow(2, Sub(3), 7), 1)

    @make_dynamo_test
    def test_three_arg_pow_udo_as_modulus(self):
        # [FUNCTION DOC] User-defined object as the modulus: the base's forward power receives it
        # and uses its value to reduce the result.
        # UDOV as modulus: base's __pow__ receives the UDOV modulus as its third arg.
        class Mod:
            # [CLASS DOC] HELPER: Local class holding a value; used only as modulus.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

        class A:
            # [CLASS DOC] HELPER: Local class holding a value with a forward power taking an
            # optional modulus.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: Forward power: with a modulus returns the plain int
                # modular result, else a new instance.
                if mod is not None:
                    return (self.value**other.value) % mod.value
                return type(self)(self.value**other.value)

        self.assertEqual(pow(A(2), A(10), Mod(1000)), 24)

    @make_dynamo_test
    def test_three_arg_pow_udo_not_implemented_raises(self):
        # [FUNCTION DOC] Ternary pow where the user class returns NotImplemented for the three-
        # argument form raises TypeError.
        # UDOV __pow__ returns NotImplemented for 3-arg form; must raise TypeError.
        class C:
            # [CLASS DOC] HELPER: Local class whose forward power returns NotImplemented for the
            # ternary form and zero otherwise.
            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: Forward power: NotImplemented when a modulus is given,
                # else zero.
                if mod is not None:
                    return NotImplemented
                return 0

        with self.assertRaises(TypeError):
            pow(C(), C(), C())

    @make_dynamo_test
    def test_three_arg_pow_z_slot(self):
        # [FUNCTION DOC] Ternary pow where both bases are an int subclass that declines; the plain
        # int modulus slot is then tried and the builtin result is expected.
        # v and w are an int subclass whose __pow__ returns NotImplemented,
        # so v's and w's nb_power slots are exhausted.  z is a plain int whose
        # C nb_power slot (type(z)->nb_power) is then tried and succeeds.
        class MyInt(int):
            # [CLASS DOC] HELPER: Int subclass whose forward power always declines.
            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: Forward power: always NotImplemented.
                return NotImplemented

        self.assertEqual(pow(MyInt(2), MyInt(3), 5), 3)

    # --- SymNode power ---

    def test_pow_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a nested function with fullgraph and dynamic first dimension
        # using the symbolic size squared as a shape and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor sized by the square
            # of the first dimension of the input.
            s = x.shape[0]
            return x.new_zeros(s**2)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_pow_symnode_negative_exponent(self):
        # [FUNCTION DOC] Compiles (without fullgraph) a nested function raising the symbolic size
        # to minus one and compares with eager; result is a float.
        # int ** negative_int produces float; shape expressions hit this
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning the size of the first dimension to
            # the power of minus one.
            s = x.shape[0]
            return s**-1

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(x), fn(x))

    def test_pow_tensor(self):
        # [FUNCTION DOC] Compiles tensor to tensor power with fullgraph and compares with eager
        # using positive bases.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the elementwise power of its two
            # tensor arguments.
            return x**y

        x = torch.randn(4).abs() + 1
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_pow_inplace_tensor(self):
        # [FUNCTION DOC] Compiles in-place tensor power with fullgraph, comparing against eager run
        # on a cloned input.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function that applies in-place power of x by y and
            # returns x.
            x **= y
            return x

        x = torch.randn(4).abs() + 1
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x_clone = x.clone()
        self.assertEqual(opt_fn(x, y), fn(x_clone, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
