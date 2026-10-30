# Owner(s): ["module: fx"]

# [MODULE DOC] Tests that the _fast_bind helper from torch.utils._inspect behaves exactly
# like the standard signature bind method. Each test builds a signature and runs a helper
# that compares the bound arguments, args, kwargs and defaults, or the raised TypeError.
# Runs under the common_utils test runner.
import inspect

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._inspect import _fast_bind


class TestFastBind(TestCase):
    # [CLASS DOC] Test case with no state. Every test builds a signature (from a nested
    # function or by hand) and feeds argument combinations through the shared comparison
    # helper, covering defaults, missing/extra/duplicate arguments, keyword-only,
    # positional-only, variadic and mutable-default parameters.
    def _assert_fast_bind_matches_sig_bind(self, sig, args, kwargs):
        # [FUNCTION DOC] HELPER: Binds the given positional args and keyword kwargs
        # against signature sig with the reference bind method. If that raises TypeError,
        # asserts the fast bind raises TypeError too and returns. Otherwise asserts the
        # fast result has equal arguments, args and kwargs, then applies defaults to both
        # and compares again. Returns nothing; fails through assertions.
        try:
            ref = sig.bind(*args, **kwargs)
        except TypeError:
            # _fast_bind should raise the same error
            with self.assertRaises(TypeError):
                _fast_bind(sig, *args, **kwargs)
            return
        # Success path: compare BoundArguments
        got = _fast_bind(sig, *args, **kwargs)
        self.assertEqual(ref.arguments, got.arguments)
        self.assertEqual(ref.args, got.args)
        self.assertEqual(ref.kwargs, got.kwargs)

        # Also validate default population matches
        ref.apply_defaults()
        got.apply_defaults()
        self.assertEqual(ref.arguments, got.arguments)
        self.assertEqual(ref.args, got.args)
        self.assertEqual(ref.kwargs, got.kwargs)

    def test_positional_or_keyword_and_defaults(self):
        # [FUNCTION DOC] Signature with one required and two defaulted
        # positional-or-keyword parameters; checks a positional call plus a keyword, a
        # full positional call, and keyword only.
        def f(a, b=1, c=2):
            # [FUNCTION DOC] HELPER: Target function used only to obtain its signature;
            # never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (10,), {"c": 3})
        self._assert_fast_bind_matches_sig_bind(sig, (10, 20), {})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"a": 1})

    def test_missing_required(self):
        # [FUNCTION DOC] Checks that omitting required parameters (no args, or too few)
        # gives matching errors.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Signature donor with two required parameters;
            # never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})

    def test_too_many_positional(self):
        # [FUNCTION DOC] Checks that passing more positional arguments than parameters
        # gives matching errors.
        def f(a, b=1):
            # [FUNCTION DOC] HELPER: Signature donor with one required and one defaulted
            # parameter; never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2, 3), {})

    def test_multiple_values_for_argument(self):
        # [FUNCTION DOC] Checks that giving a parameter both positionally and by keyword
        # gives matching errors.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Signature donor with two required parameters;
            # never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"a": 2, "b": 3})

    def test_unexpected_keyword(self):
        # [FUNCTION DOC] Checks that an unknown keyword argument gives matching errors.
        def f(a, b=1):
            # [FUNCTION DOC] HELPER: Signature donor with one required and one defaulted
            # parameter; never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"c": 3})

    def test_keyword_only(self):
        # [FUNCTION DOC] Signature with a keyword-only required and a keyword-only
        # defaulted parameter; checks a valid call, an extra positional argument, and a
        # missing keyword-only one.
        def f(a, *, b, c=1):
            # [FUNCTION DOC] HELPER: Signature donor with keyword-only parameters;
            # never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"b": 2})
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})

    def test_positional_only(self):
        # [FUNCTION DOC] Builds a signature by hand with a positional-only, a
        # positional-or-keyword and a keyword-only defaulted parameter, then checks valid
        # calls and calls passing the positional-only name as a keyword (error expected,
        # or compared with reference).
        # def f(x, /, y, *, z=1): ...
        sig = inspect.Signature(
            [
                inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("z", inspect.Parameter.KEYWORD_ONLY, default=1),
            ]
        )
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2), {"z": 3})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"x": 1, "y": 2})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"x": 2, "y": 3})

    def test_from_keyword_positional_only_pattern(self):
        # [FUNCTION DOC] Hand-built signature mirroring the operator schema pattern where
        # the name from is positional-only with a default, followed by a defaulted
        # parameter and a keyword-only generator. Checks calls with growing positional
        # args, and a call using only keywords for the positional-only parameters.
        # Mirrors a common TorchScript schema normalization pattern in operator_schemas.py
        # where "from" is treated as positional-only.
        sig = inspect.Signature(
            [
                inspect.Parameter("input", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter(
                    "from", inspect.Parameter.POSITIONAL_ONLY, default=0.0
                ),
                inspect.Parameter(
                    "to", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1.0
                ),
                inspect.Parameter(
                    "generator",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        )
        self._assert_fast_bind_matches_sig_bind(sig, ("t",), {})
        self._assert_fast_bind_matches_sig_bind(sig, ("t", 0.25), {})
        self._assert_fast_bind_matches_sig_bind(
            sig, ("t", 0.25, 0.75), {"generator": None}
        )
        self._assert_fast_bind_matches_sig_bind(sig, (), {"input": "t", "from": 0.1})

    def test_varargs_and_varkw_fallback(self):
        # [FUNCTION DOC] Checks a signature with variadic positional and variadic keyword
        # parameters: extra positionals and unknown keywords must be collected the same
        # way as the reference.
        def f(a, *args, b=0, **kwargs):
            # [FUNCTION DOC] HELPER: Signature donor with variadic positional and keyword
            # collectors; never called.
            pass

        sig = inspect.signature(f)
        # Extra positional should be captured by *args; extra keywords by **kwargs
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2, 3), {"b": 4, "x": 5})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"a": 1, "x": 2})

    def test_zero_params(self):
        # [FUNCTION DOC] Signature with no parameters: valid empty call, plus calls with
        # extra positional, extra keyword, or both, which must raise matching errors.
        def f():
            # [FUNCTION DOC] HELPER: Signature donor taking no parameters; never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})  # too many args
        self._assert_fast_bind_matches_sig_bind(sig, (), {"x": 1})  # unexpected kwarg
        self._assert_fast_bind_matches_sig_bind(
            sig, (1,), {"b": 1}
        )  # unexpected arg & kwarg

    def test_mutable_defaults(self):
        # [FUNCTION DOC] Regression test for unhashable (mutable) default values, which
        # previously broke the fast bind because of caching on signature metadata. Binds
        # with and without the defaulted argument and compares against the reference.
        # This test case reproduces an issue where unhashable default values (like list)
        # caused _fast_bind to fail because of lru_cache on _signature_metadata
        def f(a, b=[]):  # noqa: B006
            # [FUNCTION DOC] HELPER: Signature donor whose default is a mutable list;
            # never called.
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1, [2]), {})


if __name__ == "__main__":
    run_tests()
