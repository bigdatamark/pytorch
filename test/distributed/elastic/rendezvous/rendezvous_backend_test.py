# [MODULE DOC] Defines RendezvousBackendTestMixin, a reusable set of tests for any
# RendezvousBackend implementation (get_state and set_state semantics with tokens). Concrete
# test classes provide the backend and a way to corrupt stored state. No tests run from this
# module itself.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from torch.distributed.elastic.rendezvous import RendezvousStateError
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
)


class RendezvousBackendTestMixin(ABC):
    # [CLASS DOC] Abstract mixin to combine with a unittest TestCase. The subclass must set
    # _backend to the backend under test (empty at the start of each test) and implement
    # _corrupt_state. The class-level callable annotations only declare the assertion helpers
    # provided by TestCase for type checking. Tests cover reading missing, valid and corrupt
    # state, and compare-and-set behavior with current, old, missing and invalid tokens.
    _backend: RendezvousBackend

    # Type hints
    assertEqual: Callable
    assertNotEqual: Callable
    assertIsNone: Callable
    assertIsNotNone: Callable
    assertRaises: Callable

    @abstractmethod
    def _corrupt_state(self) -> None:
        # [FUNCTION DOC] Abstract hook that subclasses implement to put invalid data in the
        # backend so that the next get_state fails to decode it.
        """Corrupts the state stored in the backend."""

    def _set_state(
        self, state: bytes, token: Any | None = None
    ) -> tuple[bytes, Token, bool]:
        # [FUNCTION DOC] HELPER: Calls set_state on the backend with the state and optional
        # token, asserts the result is not None, and returns it as a tuple of state bytes, token
        # and a flag telling whether the write was applied.
        result = self._backend.set_state(state, token)

        self.assertIsNotNone(result)

        return cast(tuple[bytes, Token, bool], result)

    def test_get_state_returns_backend_state(self) -> None:
        # [FUNCTION DOC] After set_state of b"x", get_state must return that state and a token
        # that is not None.
        self._backend.set_state(b"x")

        result = self._backend.get_state()

        self.assertIsNotNone(result)

        state, token = cast(tuple[bytes, Token], result)

        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)

    def test_get_state_returns_none_if_backend_state_does_not_exist(self) -> None:
        # [FUNCTION DOC] On an empty backend, get_state must return None.
        result = self._backend.get_state()

        self.assertIsNone(result)

    def test_get_state_raises_error_if_backend_state_is_corrupt(self) -> None:
        # [FUNCTION DOC] After corrupting the stored state, get_state must raise
        # RendezvousStateError.
        self._corrupt_state()

        with self.assertRaises(RendezvousStateError):
            self._backend.get_state()

    def test_set_state_sets_backend_state_if_it_does_not_exist(self) -> None:
        # [FUNCTION DOC] Setting state with no token on an empty backend must succeed: the
        # returned state matches, the token is not None and the written flag is true.
        state, token, has_set = self._set_state(b"x")

        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)
        self.assertTrue(has_set)

    def test_set_state_sets_backend_state_if_token_is_current(self) -> None:
        # [FUNCTION DOC] Setting state with the token from the previous write must succeed,
        # return the new state and a different token, and report written true both times.
        _, token1, has_set1 = self._set_state(b"x")

        state2, token2, has_set2 = self._set_state(b"y", token1)

        self.assertEqual(b"y", state2)
        self.assertNotEqual(token1, token2)
        self.assertTrue(has_set1)
        self.assertTrue(has_set2)

    def test_set_state_returns_current_backend_state_if_token_is_old(self) -> None:
        # [FUNCTION DOC] Setting state with a stale token (not the latest) must not overwrite:
        # the call returns the current state and token from the latest successful write, with
        # written false.
        _, token1, _ = self._set_state(b"x")

        state2, token2, _ = self._set_state(b"y", token1)

        state3, token3, has_set = self._set_state(b"z", token1)

        self.assertEqual(state2, state3)
        self.assertEqual(token2, token3)
        self.assertFalse(has_set)

    def test_set_state_returns_current_backend_state_if_token_is_none(self) -> None:
        # [FUNCTION DOC] Setting state without a token when state already exists must not
        # overwrite; it returns the existing state and token with written false.
        state1, token1, _ = self._set_state(b"x")

        state2, token2, has_set = self._set_state(b"y")

        self.assertEqual(state1, state2)
        self.assertEqual(token1, token2)
        self.assertFalse(has_set)

    def test_set_state_returns_current_backend_state_if_token_is_invalid(self) -> None:
        # [FUNCTION DOC] Setting state with an invalid token (a string that was never issued)
        # must not overwrite; it returns the existing state and token with written false.
        state1, token1, _ = self._set_state(b"x")

        state2, token2, has_set = self._set_state(b"y", token="invalid")

        self.assertEqual(state1, state2)
        self.assertEqual(token1, token2)
        self.assertFalse(has_set)
