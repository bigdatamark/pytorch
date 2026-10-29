# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Unit tests for torch.distributed.elastic.rendezvous API types: validation and
# typed accessors (get, get_as_bool, get_as_int) of RendezvousParameters, and registration and
# creation behavior of RendezvousHandlerRegistry using a dummy RendezvousHandler. Uses plain
# unittest.TestCase with no run_tests entry point.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, SupportsInt
from unittest import TestCase

from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousHandlerRegistry,
    RendezvousInfo,
    RendezvousParameters,
)


class RendezvousParametersTest(TestCase):
    # [CLASS DOC] Tests RendezvousParameters. State: private fields (backend, endpoint,
    # run_id, min_nodes 3, max_nodes 6, extra kwargs mapping) that tests mutate before
    # calling _create_params.
    def setUp(self) -> None:
        # [FUNCTION DOC] Resets the default constructor inputs (dummy backend, endpoint and
        # run id, 3 to 6 nodes, empty extra options).
        super().setUp()
        self._backend = "dummy_backend"
        self._endpoint = "dummy_endpoint"
        self._run_id = "dummy_run_id"
        self._min_nodes = 3
        self._max_nodes = 6
        self._kwargs: dict[str, Any] = {}

    def _create_params(self) -> RendezvousParameters:
        # [FUNCTION DOC] HELPER: Builds RendezvousParameters from the current fields, passing
        # the extra options as keyword config entries; may raise ValueError on bad input.
        return RendezvousParameters(
            backend=self._backend,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            **self._kwargs,
        )

    def test_init_initializes_params(self) -> None:
        # [FUNCTION DOC] Checks all constructor values are exposed as attributes and an extra
        # option is retrievable through get.
        self._kwargs["dummy_param"] = "x"

        params = self._create_params()

        self.assertEqual(params.backend, self._backend)
        self.assertEqual(params.endpoint, self._endpoint)
        self.assertEqual(params.run_id, self._run_id)
        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

        self.assertEqual(params.get("dummy_param"), "x")

    def test_init_initializes_params_if_min_nodes_equals_to_1(self) -> None:
        # [FUNCTION DOC] A minimum node count of 1 is accepted and stored.
        self._min_nodes = 1

        params = self._create_params()

        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

    def test_init_initializes_params_if_min_and_max_nodes_are_equal(self) -> None:
        # [FUNCTION DOC] Equal minimum and maximum node counts (3 and 3) are accepted.
        self._max_nodes = 3

        params = self._create_params()

        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

    def test_init_raises_error_if_backend_is_none_or_empty(self) -> None:
        # [FUNCTION DOC] For a backend of None or an empty string, construction must raise
        # ValueError with the message that the backend name must be a non-empty string.
        for backend in [None, ""]:
            with self.subTest(backend=backend):
                self._backend = backend  # type: ignore[assignment]

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous backend name must be a non-empty string.$",
                ):
                    self._create_params()

    def test_init_raises_error_if_min_nodes_is_less_than_1(self) -> None:
        # [FUNCTION DOC] For minimum node counts 0, -1 and -5, construction must raise
        # ValueError saying the minimum (with the value) must be greater than zero.
        for min_nodes in [0, -1, -5]:
            with self.subTest(min_nodes=min_nodes):
                self._min_nodes = min_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of rendezvous nodes \({min_nodes}\) must be greater "
                    rf"than zero.$",
                ):
                    self._create_params()

    def test_init_raises_error_if_max_nodes_is_less_than_min_nodes(self) -> None:
        # [FUNCTION DOC] For maximum node counts 2, 1 and -2 (below the minimum of 3),
        # construction must raise ValueError saying the maximum must be at least the minimum.
        for max_nodes in [2, 1, -2]:
            with self.subTest(max_nodes=max_nodes):
                self._max_nodes = max_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The maximum number of rendezvous nodes \({max_nodes}\) must be greater "
                    "than or equal to the minimum number of rendezvous nodes "
                    rf"\({self._min_nodes}\).$",
                ):
                    self._create_params()

    def test_get_returns_none_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get on a missing option returns None.
        params = self._create_params()

        self.assertIsNone(params.get("dummy_param"))

    def test_get_returns_default_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get on a missing option returns the supplied default.
        params = self._create_params()

        self.assertEqual(params.get("dummy_param", default="x"), "x")

    def test_get_as_bool_returns_none_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get_as_bool on a missing option returns None.
        params = self._create_params()

        self.assertIsNone(params.get_as_bool("dummy_param"))

    def test_get_as_bool_returns_default_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get_as_bool on a missing option returns the supplied default (true).
        params = self._create_params()

        self.assertTrue(params.get_as_bool("dummy_param", default=True))

    def test_get_as_bool_returns_true_if_value_represents_true(self) -> None:
        # [FUNCTION DOC] Case-insensitive strings 1, true, t, yes, y and the values integer 1
        # and boolean true are all read as true by get_as_bool.
        for value in ["1", "True", "tRue", "T", "t", "yEs", "Y", 1, True]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertTrue(params.get_as_bool("dummy_param"))

    def test_get_as_bool_returns_false_if_value_represents_false(self) -> None:
        # [FUNCTION DOC] Case-insensitive strings 0, false, f, no, n and the values integer 0
        # and boolean false are all read as false by get_as_bool.
        for value in ["0", "False", "faLse", "F", "f", "nO", "N", 0, False]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertFalse(params.get_as_bool("dummy_param"))

    def test_get_as_bool_raises_error_if_value_is_invalid(self) -> None:
        # [FUNCTION DOC] Misspelled words, other strings and integers other than 0 or 1
        # make get_as_bool raise ValueError saying the option is not a valid boolean.
        for value in [
            "01",
            "Flse",  # codespell:ignore
            "Ture",  # codespell:ignore
            "g",
            "4",
            "_",
            "truefalse",
            2,
            -1,
        ]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'dummy_param' does not represent a "
                    r"valid boolean value.$",
                ):
                    params.get_as_bool("dummy_param")

    def test_get_as_int_returns_none_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get_as_int on a missing option returns None.
        params = self._create_params()

        self.assertIsNone(params.get_as_int("dummy_param"))

    def test_get_as_int_returns_default_if_key_does_not_exist(self) -> None:
        # [FUNCTION DOC] get_as_int on a missing option returns the supplied default (5).
        params = self._create_params()

        self.assertEqual(params.get_as_int("dummy_param", default=5), 5)

    def test_get_as_int_returns_integer_if_value_represents_integer(self) -> None:
        # [FUNCTION DOC] Integers and integer strings, including negative values and ones
        # with surrounding whitespace, are converted by get_as_int to the matching integer.
        for value in ["0", "-10", "5", "  4", "4  ", " 4 ", 0, -4, 3]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertEqual(
                    params.get_as_int("dummy_param"), int(cast(SupportsInt, value))
                )

    def test_get_as_int_raises_error_if_value_is_invalid(self) -> None:
        # [FUNCTION DOC] Non-numeric strings make get_as_int raise ValueError saying the
        # option is not a valid integer.
        for value in ["a", "0a", "3b", "abc"]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'dummy_param' does not represent a "
                    r"valid integer value.$",
                ):
                    params.get_as_int("dummy_param")


class _DummyRendezvousHandler(RendezvousHandler):
    # [CLASS DOC] HELPER: Minimal RendezvousHandler stub that keeps the parameters it was
    # created with; every method returns a fixed neutral value (or raises, for
    # next_rendezvous).
    def __init__(self, params: RendezvousParameters) -> None:
        # [FUNCTION DOC] HELPER: Stores the given parameters in the params attribute.
        self.params = params

    def get_backend(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the fixed backend name "dummy_backend".
        return "dummy_backend"

    def next_rendezvous(self) -> RendezvousInfo:
        # [FUNCTION DOC] HELPER: Not implemented; always raises NotImplementedError.
        raise NotImplementedError

    def is_closed(self) -> bool:
        # [FUNCTION DOC] HELPER: Always reports the rendezvous as not closed.
        return False

    def set_closed(self) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        pass

    def num_nodes_waiting(self) -> int:
        # [FUNCTION DOC] HELPER: Always reports zero waiting nodes.
        return 0

    def get_run_id(self) -> str:
        # [FUNCTION DOC] HELPER: Returns an empty run id.
        return ""

    def shutdown(self) -> bool:
        # [FUNCTION DOC] HELPER: No-op that reports the shutdown as not performed (false).
        return False


class RendezvousHandlerRegistryTest(TestCase):
    # [CLASS DOC] Tests RendezvousHandlerRegistry. State: a parameters object for backend
    # "dummy_backend" and a fresh registry per test.
    def setUp(self) -> None:
        # [FUNCTION DOC] Creates the dummy RendezvousParameters (one node) and an empty
        # registry.
        super().setUp()
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
        )

        self._registry = RendezvousHandlerRegistry()

    @staticmethod
    def _create_handler(params: RendezvousParameters) -> RendezvousHandler:
        # [FUNCTION DOC] HELPER: Creator function registered with the registry; returns a
        # dummy handler wrapping the given parameters.
        return _DummyRendezvousHandler(params)

    def test_register_registers_once_if_called_twice_with_same_creator(self) -> None:
        # [FUNCTION DOC] Registering the same backend with the same creator twice must not
        # raise (it is idempotent).
        self._registry.register("dummy_backend", self._create_handler)
        self._registry.register("dummy_backend", self._create_handler)

    def test_register_raises_error_if_called_twice_with_different_creators(
        self,
    ) -> None:
        # [FUNCTION DOC] Registering an already registered backend with a different creator
        # must raise ValueError whose message names both creators.
        self._registry.register("dummy_backend", self._create_handler)

        other_create_handler = lambda p: _DummyRendezvousHandler(p)  # noqa: E731

        with self.assertRaisesRegex(
            ValueError,
            r"^The rendezvous backend 'dummy_backend' cannot be registered with "
            rf"'{other_create_handler}' as it is already registered with '{self._create_handler}'.$",
        ):
            self._registry.register("dummy_backend", other_create_handler)

    def test_create_handler_returns_handler(self) -> None:
        # [FUNCTION DOC] After registering a creator, create_handler returns the dummy handler
        # built from the very same parameters object.
        self._registry.register("dummy_backend", self._create_handler)

        handler = self._registry.create_handler(self._params)

        self.assertIsInstance(handler, _DummyRendezvousHandler)

        self.assertIs(handler.params, self._params)

    def test_create_handler_raises_error_if_backend_is_not_registered(self) -> None:
        # [FUNCTION DOC] Creating a handler for an unregistered backend raises ValueError
        # advising that register may not have been called.
        with self.assertRaisesRegex(
            ValueError,
            r"^The rendezvous backend 'dummy_backend' is not registered. Did you forget to call "
            r"`register`\?$",
        ):
            self._registry.create_handler(self._params)

    def test_create_handler_raises_error_if_backend_names_do_not_match(self) -> None:
        # [FUNCTION DOC] Registers a creator under dummy_backend_2 whose handler reports
        # dummy_backend; creating a handler for dummy_backend_2 must raise RuntimeError that
        # the handler's backend does not match the requested one.
        self._registry.register("dummy_backend_2", self._create_handler)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^The rendezvous backend 'dummy_backend' does not match the requested backend "
            r"'dummy_backend_2'.$",
        ):
            self._params.backend = "dummy_backend_2"

            self._registry.create_handler(self._params)
