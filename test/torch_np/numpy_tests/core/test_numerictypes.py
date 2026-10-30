# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for numeric type utilities: common-type promotion, subdtype checks, bit
# names, scalar type docstrings and uniqueness of scalar type names. Uses real numpy under
# TorchDynamo, else torch._numpy. Import time defines a skip helper.

import functools
import itertools
import sys
from unittest import skipIf as skipif

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_


skip = functools.partial(skipif, True)


@xpassIfTorchDynamo_np  # (
#    reason="We do not distinguish between scalar and array types."
#    " Thus, scalars can upcast arrays."
# )
class TestCommonType(TestCase):
    # [CLASS DOC] Test case for the common-type search where scalar types may or may not upcast
    # array types. The whole class is expected to fail under dynamo mode because scalars and
    # arrays are not distinguished.
    def test_scalar_loses1(self):
        # [FUNCTION DOC] Checks that a float32 array type combined with a float64 scalar type
        # resolves to f4: a scalar of the same kind does not upcast.
        res = np.find_common_type(["f4", "f4", "i2"], ["f8"])
        assert_(res == "f4")

    def test_scalar_loses2(self):
        # [FUNCTION DOC] Checks that float32 array types combined with an int64 scalar type still
        # resolve to f4.
        res = np.find_common_type(["f4", "f4"], ["i8"])
        assert_(res == "f4")

    def test_scalar_wins(self):
        # [FUNCTION DOC] Checks that float32 and int16 array types combined with a complex64
        # scalar type resolve to c8: a scalar of a higher kind wins.
        res = np.find_common_type(["f4", "f4", "i2"], ["c8"])
        assert_(res == "c8")

    def test_scalar_wins2(self):
        # [FUNCTION DOC] Checks that uint32 and int32 array types combined with a float32 scalar
        # type resolve to f8.
        res = np.find_common_type(["u4", "i4", "i4"], ["f4"])
        assert_(res == "f8")

    def test_scalar_wins3(self):  # doesn't go up to 'f16' on purpose
        # [FUNCTION DOC] Checks that uint64 and int64 array types combined with a float64 scalar
        # type resolve to f8 and do not go up to a wider float.
        res = np.find_common_type(["u8", "i8", "i8"], ["f8"])
        assert_(res == "f8")


class TestIsSubDType(TestCase):
    # [CLASS DOC] Test case for issubdtype. The class attribute wrappers holds two converters
    # (dtype constructor and identity) so each check runs for both a dtype object and a bare
    # scalar type.
    # scalar types can be promoted into dtypes
    wrappers = [np.dtype, lambda x: x]

    def test_both_abstract(self):
        # [FUNCTION DOC] Checks that floating is a subtype of inexact but inexact is not a subtype
        # of floating.
        assert_(np.issubdtype(np.floating, np.inexact))
        assert_(not np.issubdtype(np.inexact, np.floating))

    def test_same(self):
        # [FUNCTION DOC] For float32 and int32, checks a type is a subdtype of itself under every
        # combination of wrappers on both arguments.
        for cls in (np.float32, np.int32):
            for w1, w2 in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    def test_subclass(self):
        # [FUNCTION DOC] Checks float32 and float64, wrapped either way, are subtypes of the
        # abstract floating type.
        # note we cannot promote floating to a dtype, as it would turn into a
        # concrete type
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    def test_subclass_backwards(self):
        # [FUNCTION DOC] Checks that the abstract floating type is not a subtype of the concrete
        # float32 or float64 (wrapped either way).
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    def test_sibling_class(self):
        # [FUNCTION DOC] Checks float32 and float64 are not subtypes of each other for every
        # wrapper combination.
        for w1, w2 in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))

    def test_nondtype_nonscalartype(self):
        # [FUNCTION DOC] Checks that string and Python-builtin second arguments are handled as in
        # numpy: names like float64 or f8 match their exact type only, builtin int, float and
        # complex are not matched by sized types, while abstract classes match. Raises
        # AssertionError for each wrong answer.
        # See gh-14619 and gh-9505 which introduced the deprecation to fix
        # this. These tests are directly taken from gh-9505
        if np.issubdtype(np.float32, "float64"):
            raise AssertionError("np.float32 should not be subtype of float64")
        if np.issubdtype(np.float32, "f8"):
            raise AssertionError("np.float32 should not be subtype of f8")
        if np.issubdtype(np.int32, "int64"):
            raise AssertionError("np.int32 should not be subtype of int64")
        # for the following the correct spellings are
        # np.integer, np.floating, or np.complexfloating respectively:
        if np.issubdtype(np.int8, int):  # np.int8 is never np.int_
            raise AssertionError("np.int8 should not be subtype of int")
        if np.issubdtype(np.float32, float):
            raise AssertionError("np.float32 should not be subtype of float")
        if np.issubdtype(np.complex64, complex):
            raise AssertionError("np.complex64 should not be subtype of complex")
        if np.issubdtype(np.float32, "float"):
            raise AssertionError("np.float32 should not be subtype of 'float'")
        if np.issubdtype(np.float64, "f"):
            raise AssertionError("np.float64 should not be subtype of 'f'")

        # Test the same for the correct first datatype and abstract one
        # in the case of int, float, complex:
        if not np.issubdtype(np.float64, "float64"):
            raise AssertionError("np.float64 should be subtype of float64")
        if not np.issubdtype(np.float64, "f8"):
            raise AssertionError("np.float64 should be subtype of f8")
        if not np.issubdtype(np.int64, "int64"):
            raise AssertionError("np.int64 should be subtype of int64")
        if not np.issubdtype(np.int8, np.integer):
            raise AssertionError("np.int8 should be subtype of np.integer")
        if not np.issubdtype(np.float32, np.floating):
            raise AssertionError("np.float32 should be subtype of np.floating")
        if not np.issubdtype(np.complex64, np.complexfloating):
            raise AssertionError("np.complex64 should be subtype of np.complexfloating")
        if not np.issubdtype(np.float64, "float"):
            raise AssertionError("np.float64 should be subtype of 'float'")
        if not np.issubdtype(np.float32, "f"):
            raise AssertionError("np.float32 should be subtype of 'f'")


@xpassIfTorchDynamo_np  # (
#    reason="We do not have (or need) np.core.numerictypes."
#    " Our type aliases are in _dtypes.py."
# )
class TestBitName(TestCase):
    # [CLASS DOC] Test case for the bit-name helper of the numerictypes module. Expected to fail
    # under dynamo mode since that module is absent here.
    def test_abstract(self):
        # [FUNCTION DOC] Checks that the bitname helper raises ValueError when given the abstract
        # floating type.
        assert_raises(ValueError, np.core.numerictypes.bitname, np.floating)


@skip(reason="Docstrings for scalar types, not yet.")
@skipif(
    sys.flags.optimize > 1,
    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1",
)
class TestDocStrings(TestCase):
    # [CLASS DOC] Test case for scalar type docstrings; skipped unconditionally (docstrings not
    # available yet) and also skipped when Python runs with optimization level above one.
    def test_platform_dependent_aliases(self):
        # [FUNCTION DOC] Checks that the docstring of whichever of int_ or longlong aliases int64
        # mentions int64. Never runs because the class is skipped.
        if np.int64 is np.int_:
            assert_("int64" in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_("int64" in np.longlong.__doc__)


@instantiate_parametrized_tests
class TestScalarTypeNames(TestCase):
    # [CLASS DOC] Test case for scalar type names. The class attribute numeric_types lists the
    # scalar types checked; longlong is left out because torch does not provide it separately.
    # gh-9799

    numeric_types = [
        np.byte,
        np.short,
        np.intc,
        np.int_,  # , np.longlong, NB: torch does not properly have longlong
        np.ubyte,
        np.half,
        np.single,
        np.double,
        np.csingle,
        np.cdouble,
    ]

    def test_names_are_unique(self):
        # [FUNCTION DOC] Checks the listed scalar types contain no duplicates and that their name
        # attributes are all distinct; raises AssertionError otherwise.
        # none of the above may be aliases for each other
        if len(set(self.numeric_types)) != len(self.numeric_types):
            raise AssertionError("numeric_types contains duplicates")

        # names must be unique
        names = [t.__name__ for t in self.numeric_types]
        if len(set(names)) != len(names):
            raise AssertionError("numeric_type names are not unique")

    @parametrize("t", numeric_types)
    def test_names_reflect_attributes(self, t):
        # [FUNCTION DOC] Parametrized over the listed scalar types: checks the numpy namespace
        # attribute named like the type is the same type object.
        """Test that names correspond to where the type is under ``np.``"""
        if getattr(np, t.__name__) is not t:
            raise AssertionError(f"np.{t.__name__} is not {t}")

    @skipIfTorchDynamo()  # XXX: weird, some names are not OK
    @parametrize("t", numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        # [FUNCTION DOC] Parametrized over the listed scalar types: checks that building a dtype
        # from the type's name gives a dtype whose scalar type is the original type. Skipped under
        # dynamo because some names are not accepted.
        """Test the dtype constructor maps names back to the type"""
        if np.dtype(t.__name__).type is not t:
            raise AssertionError(f"np.dtype({t.__name__!r}).type is not {t}")


if __name__ == "__main__":
    run_tests()
