# Owner(s): ["module: inductor"]
# [MODULE DOC] Unit test for the inductor warp-specialization gating helpers
# (has_meta_ws and meta_ws_enabled from torch._inductor.autows_utils). No import-time side
# effects beyond imports; running as a script invokes the inductor test runner.
from torch._inductor.autows_utils import has_meta_ws, meta_ws_enabled
from torch._inductor.test_case import run_tests, TestCase


class AutoWSUtilsTests(TestCase):
    # [CLASS DOC] Test case holding the single check that meta_ws_enabled follows the Triton
    # use_meta_ws knob only when the Triton build supports meta warp specialization.
    def test_meta_ws_enabled_gating(self) -> None:
        # [FUNCTION DOC] Checks the gating of meta warp specialization. When has_meta_ws reports
        # no support, asserts the gate is off and returns early. Otherwise, inside a scoped Triton
        # nvidia knobs context (so the knob is restored afterwards), sets the knob true and asserts
        # the gate is on, then sets it false and asserts the gate is off.
        if not has_meta_ws():
            # Without fb-Triton the gate is off regardless of the knob.
            self.assertFalse(meta_ws_enabled())
            return
        from triton import knobs

        with knobs.nvidia.scope():
            knobs.nvidia.use_meta_ws = True
            self.assertTrue(meta_ws_enabled())
            knobs.nvidia.use_meta_ws = False
            self.assertFalse(meta_ws_enabled())


if __name__ == "__main__":
    run_tests()
