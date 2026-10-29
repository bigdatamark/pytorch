# Owner(s): ["module: dynamo"]
# [MODULE DOC] Unit tests for the Dynamo side effect replay handler registry: highest priority
# wins regardless of registration order, equal priority matches raise an ambiguity error, the
# built-in handlers keep their expected names and priorities, and the cell handler outranks the
# generic attribute mutation handler. Uses mock side effect and codegen objects; no compilation
# is performed.

from unittest.mock import Mock

from torch._dynamo.side_effects import (
    _side_effect_replay_registry,
    _SideEffectReplayRegistry,
    SideEffectReplayContext,
    SideEffectReplayHandler,
    SideEffects,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.variables import CellVariable
from torch._dynamo.variables.base import (
    AttributeMutation,
    AttributeMutationNew,
    VariableTracker,
)


class SideEffectReplayRegistryTests(TestCase):
    # [CLASS DOC] Test case exercising the replay registry class and the global registry of
    # built-in handlers. Tests build fresh registries or read the global one without modifying
    # it permanently.
    @staticmethod
    def make_context(var: VariableTracker | None = None) -> SideEffectReplayContext:
        # [FUNCTION DOC] HELPER: Static factory building a replay context for tests. Side
        # effects is a mock restricted to the real class whose attribute mutation check reports
        # true when the item's mutation type is an attribute mutation. Codegen and log are plain
        # mocks and the suffix list is empty. The tracked variable is the given one, or a fresh
        # base tracker when none is given.
        side_effects = Mock(spec=SideEffects)
        side_effects.is_attribute_mutation.side_effect = lambda item: isinstance(
            item.mutation_type, AttributeMutation
        )
        return SideEffectReplayContext(
            side_effects=side_effects,
            codegen=Mock(),
            var=var if var is not None else VariableTracker(),
            suffixes=[],
            log=Mock(),
        )

    def test_priority_wins_independent_of_registration_order(self) -> None:
        # [FUNCTION DOC] Registers a lower and a higher priority handler that both always match,
        # with the lower registered first. Asserts lookup returns the higher one; after the
        # higher handler's registration handle is destroyed, lookup returns the lower one.
        registry = _SideEffectReplayRegistry()
        ctx = self.make_context()
        lower = SideEffectReplayHandler("lower", lambda _: True, Mock(), priority=1)
        higher = SideEffectReplayHandler("higher", lambda _: True, Mock(), priority=2)

        registry.register_handler(lower)
        higher_handle = registry.register_handler(higher)
        self.assertIs(registry.lookup_handler(ctx), higher)

        higher_handle.destroy()
        self.assertIs(registry.lookup_handler(ctx), lower)

    def test_equal_priority_matches_are_ambiguous(self) -> None:
        # [FUNCTION DOC] Registers two always-matching handlers with the same priority and
        # asserts lookup raises a runtime error whose message says the handler is ambiguous for
        # the variable tracker class and lists both handler names in order.
        registry = _SideEffectReplayRegistry()
        ctx = self.make_context()
        registry.register_handler(
            SideEffectReplayHandler("first", lambda _: True, Mock(), priority=1)
        )
        registry.register_handler(
            SideEffectReplayHandler("second", lambda _: True, Mock(), priority=1)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Ambiguous side effect replay handler for VariableTracker: "
            r"\['first', 'second'\]",
        ):
            registry.lookup_handler(ctx)

    def test_builtin_priorities_preserve_legacy_dispatch_order(self) -> None:
        # [FUNCTION DOC] Asserts the global registry holds exactly the nine built-in handlers,
        # in registration order, with descending priorities: list, deque, constant dict or set,
        # torch function mode stack, cell, attribute, list iterator, count iterator and random
        # mutation handlers, from 90 down to 10 in steps of ten.
        self.assertEqual(
            [
                (handler.name, handler.priority)
                for handler in _side_effect_replay_registry.handlers
            ],
            [
                ("list_mutation", 90),
                ("deque_mutation", 80),
                ("const_dict_or_set_mutation", 70),
                ("torch_function_mode_stack_mutation", 60),
                ("cell_mutation", 50),
                ("attribute_mutation", 40),
                ("list_iterator_mutation", 30),
                ("count_iterator_mutation", 20),
                ("random_mutation", 10),
            ],
        )

    def test_cell_handler_wins_attribute_mutation_overlap(self) -> None:
        # [FUNCTION DOC] Builds a new cell variable with an attribute-mutation-new mutation type
        # and a local name, then looks up its handler in the global registry. Asserts a handler
        # is found and it is the cell mutation one, showing it takes precedence over the
        # overlapping attribute mutation handler.
        cell = CellVariable(mutation_type=AttributeMutationNew())
        cell.local_name = "cell"

        handler = _side_effect_replay_registry.lookup_handler(self.make_context(cell))

        self.assertIsNotNone(handler)
        self.assertEqual(handler.name, "cell_mutation")


if __name__ == "__main__":
    run_tests()
