# Owner(s): ["module: sdpa"]
# [MODULE DOC] Tests for the flash attention implementation registry in torch.nn.attention.
# Covers registering and activating an implementation, and the error for unknown names. Uses a
# fake handle object and saves and restores the registry global state around each test.

import torch.nn.attention as attention
from torch.nn.attention import _registry
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class FakeHandle:
    # [CLASS DOC] HELPER: Minimal stand-in for a registration handle returned by a register
    # function.
    def remove(self):
        # [FUNCTION DOC] HELPER: No-op removal method, matching the handle interface.
        pass


class TestFlashAttentionRegistry(TestCase):
    # [CLASS DOC] Generic-hardware tests for the flash attention registry. Each test starts with
    # an emptied implementation table and no active implementation; the originals are restored on
    # teardown.
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves a copy of the registered implementations and the currently
        # active one, then clears the implementation table and resets the active implementation
        # to none.
        super().setUp()
        self._saved_impls = dict(_registry._FLASH_ATTENTION_IMPLS)
        self._saved_active = attention.current_flash_attention_impl()
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_ACTIVE = None

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the saved implementations table contents and active
        # implementation, then runs the parent teardown.
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_IMPLS.update(self._saved_impls)
        _registry._FLASH_ATTENTION_ACTIVE = self._saved_active
        super().tearDown()

    def test_register_and_activate_impl(self):
        # [FUNCTION DOC] Registers an implementation named TEST_FA with a register function that
        # records being called and returns a fake handle. Asserts it is listed, activates it,
        # then asserts the register function ran and the current implementation is TEST_FA.
        calls: dict[str, bool] = {}

        def fake_register():
            # [FUNCTION DOC] HELPER: Nested register function that marks the shared call record
            # and returns a fake handle.
            calls["called"] = True
            return FakeHandle()

        attention.register_flash_attention_impl("TEST_FA", register_fn=fake_register)
        self.assertIn("TEST_FA", attention.list_flash_attention_impls())

        attention.activate_flash_attention_impl("TEST_FA")

        self.assertTrue(calls.get("called", False))
        self.assertEqual("TEST_FA", attention.current_flash_attention_impl())

    def test_activate_unknown_impl_errors(self):
        # [FUNCTION DOC] Asserts that activating an unregistered name raises ValueError
        # mentioning an unknown flash attention implementation.
        with self.assertRaisesRegex(
            ValueError, "Unknown flash attention impl 'missing'"
        ):
            attention.activate_flash_attention_impl("missing")


if __name__ == "__main__":
    run_tests()
