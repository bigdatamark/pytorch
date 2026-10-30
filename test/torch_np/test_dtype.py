# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that the dtype constructor of torch._numpy converts dtype specifiers
# (names and NumPy dtype objects) into its own dtype objects. At import time it builds a
# module-level list of parameter sets (name, specifier) for bool, int, uint, float and
# complex widths: the plain string name is a normal case, while NumPy namespaced types and
# NumPy dtype objects are expected-failure cases because they are not supported.

from unittest import expectedFailure as xfail

import numpy

import torch._numpy as tnp
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


dtype_names = [
    "bool_",
    *[f"int{w}" for w in [8, 16, 32, 64]],
    *[f"uint{w}" for w in [8, 16, 32, 64]],
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]

np_dtype_params = [
    subtest(("bool", "bool"), name="bool"),
    subtest(
        ("bool", numpy.dtype("bool")),
        name="numpy_dtype('bool')",
        decorators=[xfail],  # reason="XXX: np.dtype() objects not supported"),
    ),
]


for name in dtype_names:
    np_dtype_params.append(subtest((name, name), name=repr(name)))

    np_dtype_params.append(
        subtest((name, getattr(numpy, name)), name=f"numpy_{name}", decorators=[xfail])
    )  # numpy namespaced dtypes not supported
    np_dtype_params.append(
        subtest((name, numpy.dtype(name)), name=f"numpy_{name!r}", decorators=[xfail])
    )


@instantiate_parametrized_tests
class TestConvertDType(TestCase):
    # [CLASS DOC] Parametrized test case holding the dtype conversion test; instantiated once per
    # entry of the module-level parameter list by the instantiate_parametrized_tests decorator.
    @parametrize("name, np_dtype", np_dtype_params)
    def test_convert_np_dtypes(self, name, np_dtype):
        # [FUNCTION DOC] Converts the given specifier with the torch._numpy dtype constructor and
        # checks the result. Arguments are the expected dtype name and the specifier to convert.
        # For the name bool_ the result must equal the torch._numpy bool_ dtype; if the result is
        # named bool_ for any other input, the name must start with bool; otherwise the result name
        # must equal the expected name. Failures raise AssertionError. NumPy-object specifiers are
        # marked expected failures by the parameters.
        tnp_dtype = tnp.dtype(np_dtype)
        if name == "bool_":
            if tnp_dtype != tnp.bool_:
                raise AssertionError(
                    f"Expected tnp_dtype == tnp.bool_, got {tnp_dtype}"
                )
        elif tnp_dtype.name == "bool_":
            if not name.startswith("bool"):
                raise AssertionError(
                    f"Expected name to start with 'bool', got '{name}'"
                )
        else:
            if tnp_dtype.name != name:
                raise AssertionError(
                    f"Expected tnp_dtype.name == '{name}', got '{tnp_dtype.name}'"
                )


if __name__ == "__main__":
    run_tests()
