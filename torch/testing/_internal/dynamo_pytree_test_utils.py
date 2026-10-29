# [MODULE DOC]
# Test helper providing PytreeRegisteringTestCase, a Dynamo test case that tracks pytree
# registrations made during a test and removes them afterwards. No import-time side effects.

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree


class PytreeRegisteringTestCase(torch._dynamo.test_case.TestCase):
    # [CLASS DOC]
    # Dynamo TestCase that records classes registered as pytree nodes or constants through
    # its helper methods, in two lists (_registered_pytree_nodes, _registered_constant_nodes),
    # and deregisters them in reverse order on tearDown, which also resets Dynamo.
    """TestCase that prunes all temporary pytree registrations and resets Dynamo."""

    def setUp(self) -> None:
        # [FUNCTION DOC]
        # Runs base setUp and initializes the two empty lists of registered classes.
        super().setUp()
        self._registered_pytree_nodes: list[type] = []
        self._registered_constant_nodes: list[type] = []

    def tearDown(self) -> None:
        # [FUNCTION DOC]
        # Deregisters every recorded pytree node and constant class (most recent first),
        # resets Dynamo, then runs base tearDown.
        # KNOWN ISSUE: constants are registered via register_constant but deregistered with the
        # generic pytree node deregistration; this works only if that is the right inverse.
        for cls in reversed(self._registered_pytree_nodes):
            pytree._deregister_pytree_node(cls)
        for cls in reversed(self._registered_constant_nodes):
            pytree._deregister_pytree_node(cls)
        torch._dynamo.reset()
        super().tearDown()

    def register_pytree_node(self, cls, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # Registers cls as a pytree node, forwarding extra arguments, and records it so
        # tearDown removes the registration. Raises whatever pytree registration raises.
        pytree.register_pytree_node(cls, *args, **kwargs)
        self._registered_pytree_nodes.append(cls)

    def register_constant(self, cls: type) -> None:
        # [FUNCTION DOC]
        # Registers cls as a pytree constant and records it for cleanup in tearDown.
        pytree.register_constant(cls)
        self._registered_constant_nodes.append(cls)
