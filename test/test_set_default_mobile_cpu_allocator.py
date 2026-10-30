# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests the private mobile default CPU allocator set/unset hooks in torch._C.

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSetDefaultMobileCPUAllocator(TestCase):
    # [CLASS DOC] Tests the private mobile default-CPU-allocator toggles on torch._C.
    # The allocator switch is process-global state; setting twice or unsetting while unset
    # must raise, so each test leaves the state unset.
    def test_no_exception(self):
        # [FUNCTION DOC] Sets then unsets the mobile default CPU allocator once; both calls
        # must succeed without raising. Leaves the global allocator state unset.
        torch._C._set_default_mobile_cpu_allocator()
        torch._C._unset_default_mobile_cpu_allocator()

    def test_exception(self):
        # [FUNCTION DOC] Checks misuse raises: unset while not set, set twice in a row, and
        # set followed by two unsets. A manual unset between cases restores the unset state
        # after the double-set case left it set.
        with self.assertRaises(Exception):
            torch._C._unset_default_mobile_cpu_allocator()

        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._set_default_mobile_cpu_allocator()

        # Must reset to good state
        # For next test.
        torch._C._unset_default_mobile_cpu_allocator()

        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()

if __name__ == '__main__':
    run_tests()
