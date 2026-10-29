# Owner(s): ["module: fx"]
# [MODULE DOC] Test of the graph module node hooks (create, erase and replace callbacks).
# Registers two hooks of each kind on a traced module, performs a node copy, replace-all-
# uses and erase on a placeholder, and checks every hook ran and that unregistering empties
# the hook lists. Running the file directly raises an error stating the test is not
# currently used and must be enabled in the test discovery list.
import torch
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase


class TestFXNodeHook(TestCase):
    # [CLASS DOC] Test case for FX node-update hooks. Holds one test; no state.
    def test_hooks_for_node_update(self):
        # [FUNCTION DOC] Registers two create, two erase and two replace hooks on a traced
        # function module. Copies the first placeholder node, redirects all its uses to the
        # copy and erases the original, then raises AssertionError unless all six hooks were
        # called, unregisters them and raises AssertionError if the module's private hook
        # lists are not empty afterward. Uses module-level global flags to record which
        # hooks were called.
        global create_node_hook1_called
        global create_node_hook2_called
        global erase_node_hook1_called
        global erase_node_hook2_called
        global replace_node_hook1_called
        global replace_node_hook2_called
        create_node_hook1_called = False
        create_node_hook2_called = False
        erase_node_hook1_called = False
        erase_node_hook2_called = False
        replace_node_hook1_called = False
        replace_node_hook2_called = False

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Traced function: a linear transform of the first two
            # inputs, plus the third, then cosine.
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        def create_node_hook1(node):
            # [FUNCTION DOC] HELPER: Create hook; sets its module-level called flag to true.
            # Argument is ignored.
            global create_node_hook1_called
            create_node_hook1_called = True

        def create_node_hook2(node):
            # [FUNCTION DOC] HELPER: Create hook; sets its module-level called flag to true.
            # Argument is ignored.
            global create_node_hook2_called
            create_node_hook2_called = True

        def erase_node_hook1(node):
            # [FUNCTION DOC] HELPER: Erase hook; sets its module-level called flag to true.
            # Argument is ignored.
            global erase_node_hook1_called
            erase_node_hook1_called = True

        def erase_node_hook2(node):
            # [FUNCTION DOC] HELPER: Erase hook; sets its module-level called flag to true.
            # Argument is ignored.
            global erase_node_hook2_called
            erase_node_hook2_called = True

        def replace_node_hook1(old, new, user):
            # [FUNCTION DOC] HELPER: Replace hook with assertions. Checks the replaced node
            # is named a, the new value is the name a_1 and the user node is named linear,
            # then sets its called flag.
            global replace_node_hook1_called
            self.assertEqual(old.name, "a")
            self.assertEqual(new, "a_1")
            self.assertEqual(user.name, "linear")
            replace_node_hook1_called = True

        def replace_node_hook2(old, new, user):
            # [FUNCTION DOC] HELPER: Replace hook; sets its called flag without checking its
            # arguments.
            global replace_node_hook2_called
            replace_node_hook2_called = True

        gm = symbolic_trace(fn)
        gm._register_create_node_hook(create_node_hook1)
        gm._register_create_node_hook(create_node_hook2)
        gm._register_erase_node_hook(erase_node_hook1)
        gm._register_erase_node_hook(erase_node_hook2)
        gm._register_replace_node_hook(replace_node_hook1)
        gm._register_replace_node_hook(replace_node_hook2)

        graph = gm.graph
        node_a = None
        for node in graph.find_nodes(op="placeholder"):
            node_a = node
            break
        if node_a is None:
            raise AssertionError("Expected to find a placeholder node")
        # This will create a new node
        node_a_copy = graph.node_copy(node_a)
        node_a.replace_all_uses_with(node_a_copy)
        graph.erase_node(node_a)

        if not (
            create_node_hook1_called
            and create_node_hook2_called
            and erase_node_hook1_called
            and erase_node_hook2_called
            and replace_node_hook1_called
            and replace_node_hook2_called
        ):
            raise AssertionError("Expected all node hooks to be called")

        gm._unregister_create_node_hook(create_node_hook1)
        gm._unregister_create_node_hook(create_node_hook2)
        gm._unregister_erase_node_hook(erase_node_hook1)
        gm._unregister_erase_node_hook(erase_node_hook2)
        gm._unregister_replace_node_hook(replace_node_hook1)
        gm._unregister_replace_node_hook(replace_node_hook2)

        if gm._create_node_hooks != []:
            raise AssertionError("Expected gm._create_node_hooks to be empty")
        if gm._erase_node_hooks != []:
            raise AssertionError("Expected gm._erase_node_hooks to be empty")
        if gm._replace_hooks != []:
            raise AssertionError("Expected gm._replace_hooks to be empty")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
