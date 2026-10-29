# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that compiled functions returning a bounded collections.deque
# reconstruct it correctly regardless of whether this module's globals define, lack, or
# shadow the name deque. Uses the eager backend with fullgraph. No import-time side effects.

import collections
import contextlib

import torch
import torch._inductor.test_case


class TestDequeReconstruct(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Tests deque reconstruction after compilation under three states of the
    # module-global name deque (absent, the real class, or None). UNSET is a sentinel
    # meaning the name should be absent from module globals.
    UNSET = object()

    @contextlib.contextmanager
    def set_deque_in_globals(self, value):
        # [FUNCTION DOC] HELPER: Context manager that removes the global name deque from this
        # module (raising AssertionError if it remains), then sets it to the given value
        # unless the value is the UNSET sentinel. On exit the original state is restored:
        # removed if it was absent, otherwise reassigned to the previous value.
        prev = globals().pop("deque", self.UNSET)
        if "deque" in globals():
            raise AssertionError("Expected deque to not be in globals")

        try:
            if value is not self.UNSET:
                globals()["deque"] = value
            yield
        finally:
            if prev is self.UNSET:
                globals().pop("deque", None)
                if "deque" in globals():
                    raise AssertionError("Expected deque to not be in globals")
            else:
                globals()["deque"] = prev

    def test_deque_reconstruct_not_in_globals(self):
        # [FUNCTION DOC] With no global deque name, a compiled function that builds a deque
        # with maxlen 2 from three tensors must return a real deque with maxlen 2 holding
        # only the last two items.
        with self.set_deque_in_globals(self.UNSET):

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_in_globals(self):
        # [FUNCTION DOC] With the global deque name bound to the real class, checks plain
        # eager use works, then that the compiled function returns the same truncated deque.
        with self.set_deque_in_globals(collections.deque):
            # This does not emit a NameError
            dummy = deque([0, 1, 2], maxlen=2)  # noqa: F821
            self.assertIsInstance(dummy, collections.deque)
            self.assertEqual(list(dummy), [1, 2])

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_shallows_globals(self):
        # [FUNCTION DOC] With the global deque name bound to None, the compiled function must
        # still return a real deque (reconstruction must not rely on the global name).
        # The method name says "shallows" but means shadows.
        with self.set_deque_in_globals(None):
            # This does not emit a NameError
            self.assertIsNone(deque)  # noqa: F821

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
