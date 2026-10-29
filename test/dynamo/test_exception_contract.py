# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo test checking that compiled (eager backend) functions raise IndexError
# for out-of-range dimension arguments, as eager mode does. Defines a table OPS of named
# callables (softmax, squeeze, unsqueeze, amax, argmax, flip, cat, stack) each using a bad dim,
# and one TestCase class. No import-time side effects besides building the table.
"""
Tests that torch.compile preserves the IndexError exception contract for
out-of-range dimension/index arguments, matching eager-mode behavior.
"""

import torch
from torch._dynamo.test_case import run_tests, TestCase


OPS = [
    ("softmax", lambda t: torch.softmax(t, dim=10)),
    ("squeeze", lambda t: torch.squeeze(t, 99)),
    ("unsqueeze", lambda t: torch.unsqueeze(t, 99)),
    ("amax", lambda t: torch.amax(t, dim=99)),
    ("argmax", lambda t: torch.argmax(t, dim=99)),
    ("flip", lambda t: torch.flip(t, dims=[99])),
    ("cat", lambda t: torch.cat([t], dim=99)),
    ("stack", lambda t: torch.stack([t], dim=99)),
]


class TestIndexErrorContract(TestCase):
    # [CLASS DOC] Stateless test case holding two tests that compile small functions with the
    # eager backend and assert that an out-of-range dim still raises IndexError.
    def _make_tensor(self):
        # [FUNCTION DOC] HELPER: returns a fresh random float tensor of shape four (one dim).
        return torch.randn(4)

    def test_compile_preserves_index_error(self):
        # [FUNCTION DOC] For each entry in OPS (in a subTest), compiles the callable with the
        # eager backend, resets dynamo, and asserts calling it on a 1-D tensor raises IndexError.
        t = self._make_tensor()
        for name, op in OPS:
            with self.subTest(op=name):
                compiled = torch.compile(op, backend="eager")
                torch._dynamo.reset()
                with self.assertRaises(IndexError):
                    compiled(t)

    def test_size_negative_out_of_range(self):
        # [FUNCTION DOC] Asserts that asking for size along dim -99 raises IndexError both in
        # eager mode and in a function compiled with the eager backend (after a dynamo reset).
        t = self._make_tensor()
        with self.assertRaises(IndexError):
            t.size(-99)

        compiled = torch.compile(lambda x: x.size(-99), backend="eager")
        torch._dynamo.reset()
        with self.assertRaises(IndexError):
            compiled(t)


if __name__ == "__main__":
    run_tests()
