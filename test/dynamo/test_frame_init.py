# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo's frame evaluation hook can substitute replacement bytecode for
# functions with unusual signatures (positional-only, keyword-only, variadic positional and
# keyword arguments). Defines two target functions, each with two replacement variants that
# remove or introduce local variables while returning equivalent dictionaries. No import-time
# side effects beyond defining these functions and one TestCase class.

import torch
import torch._dynamo.test_case
from torch._C._dynamo.eval_frame import set_eval_frame
from torch._dynamo.types import ConvertFrameReturn, GuardedCode, wrap_guarded_code
from torch._guards import CompileId


def target_with_varkwargs(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # [FUNCTION DOC] HELPER: Returns a mapping of the local value one, and each argument
    # (including the keyword-only one and the extra keyword mapping), for testing frame
    # initialization.
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code1(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the keyword target; same signature and result but
    # the local variable is removed and the constant used directly.
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code2(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the keyword target; the result entry local is
    # computed from two newly introduced locals that sum to one, so the number of locals
    # differs.
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def target_with_varargs(arg1, /, positional_only_arg, *varargs, **kwargs):
    # [FUNCTION DOC] HELPER: Like the keyword target but with extra positional arguments
    # collected in a tuple instead of a keyword-only argument; returns a mapping including that
    # tuple.
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code1(arg1, /, positional_only_arg, *varargs, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the varargs target; local variable removed,
    # constant used directly in the result.
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code2(arg1, /, positional_only_arg, *varargs, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the varargs target; introduces two locals whose sum
    # (one) is the result entry for local.
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


class FrameInitTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase with one test of replacing the code objects of functions with
    # varargs and varkwargs signatures through a custom frame evaluation callback.
    def test_frame_init(self):
        # [FUNCTION DOC] Builds two maps from original code objects to replacement code objects,
        # and two callbacks that, for a mapped frame, return guarded replacement code with an
        # empty guard manager and a fixed compile id, and otherwise return a default no-op frame
        # result. Loops twice: resets Dynamo, computes expected outputs of both targets,
        # installs the callback through the eval-frame hook, computes outputs again, asserts
        # they equal the expected ones, and restores the previous hook.
        # KNOWN ISSUE: the loop variable is unused and the first callback is always installed,
        #     so the second callback and second code map are never exercised. If an assertion
        #     fails the previous eval-frame hook is not restored.
        code_map1 = {
            target_with_varargs.__code__: varargs_code1.__code__,
            target_with_varkwargs.__code__: varkwargs_code1.__code__,
        }
        code_map2 = {
            target_with_varargs.__code__: varargs_code2.__code__,
            target_with_varkwargs.__code__: varkwargs_code2.__code__,
        }

        empty_guard_manager = torch._dynamo.guards.GuardManagerWrapper()

        def callback1(frame, cache_entry, frame_state):
            # [FUNCTION DOC] HELPER: Frame callback. If the frame's code object is a key in the
            # first map, returns guarded code wrapping the mapped replacement code, an empty
            # guard manager, and compile id with no frame id and zero compile and autograd ids;
            # otherwise returns a default ConvertFrameReturn. The cache entry and frame state
            # arguments are unused.
            if frame.f_code in code_map1:
                transformed_code = code_map1[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        def callback2(frame, cache_entry, frame_state):
            # [FUNCTION DOC] HELPER: Same as the first callback but looks up the second code
            # map. Never installed by the test.
            if frame.f_code in code_map2:
                transformed_code = code_map2[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        for _ in [callback1, callback2]:
            torch._dynamo.reset()
            expected_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            expected_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            original = set_eval_frame(callback1)
            real_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            real_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            self.assertEqual(real_varargs_output, expected_varargs_output)
            self.assertEqual(real_kwargs_output, expected_kwargs_output)
            set_eval_frame(original)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
