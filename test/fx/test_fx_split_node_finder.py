# Owner(s): ["module: fx"]

# pyre-strict
# [MODULE DOC] Tests for the node event tracker used by the FX accelerator node finder. Builds a
# small traced module with wrapped "supported" calls, runs the node finder with a NodeEventTracker
# in four modes (no dump, dump all events, dump tracked nodes, dump nodes with multiple events)
# and checks recorded events and the dumped event/node files. Import-time side effect: registers
# sup_f as an FX leaf function via the wrap decorator.
import os
import shutil
import sys
import tempfile

import torch
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.splitter_base import (
    ALL_SUFFIX,
    ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES,
    FxNetAccNodesFinder,
    NodeEventTracker,
    NODES_SUFFIX,
    ShapeProp,
)
from torch.testing._internal.common_utils import TestCase


# Wrappepr function to make it supported
@torch.fx.wrap
def sup_f(x):
    # [FUNCTION DOC] HELPER: Identity function on its argument. It is registered with FX wrap so
    # symbolic tracing records it as a call_function node (named sup_f, sup_f_1, ...) that the
    # test operator-support check can recognize by name.
    return x


class TestFxSplitNodeFinder(TestCase):
    # [CLASS DOC] Tests NodeEventTracker through FxNetAccNodesFinder. Each test gets a temporary
    # directory (self.tmpdir, also placed on sys.path) used as the dump location for tracker
    # output files; the mode tests differ only in the tracker mode and the expected files.
    def setUp(self):
        # [FUNCTION DOC] Saves sys.path, creates a temporary directory stored in self.tmpdir and
        # prepends it to sys.path.
        super().setUp()
        self.save_path = sys.path[:]
        self.tmpdir = tempfile.mkdtemp()
        sys.path.insert(0, self.tmpdir)

    def tearDown(self):
        # [FUNCTION DOC] Restores sys.path to the saved copy and removes the temporary directory,
        # ignoring removal errors. Does not call the parent tearDown.
        sys.path[:] = self.save_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _testTrackerBasics(self, tracker):
        # [FUNCTION DOC] HELPER: Installs the given tracker on a node finder built over a traced
        # three-call module, runs the finder and checks the result. The middle sup_f call
        # (sup_f_1) must be dropped from the accelerator nodes with exactly two events (init_acc,
        # then acc_del for a non-tensor output with a cpu user); the other sup_f nodes stay with
        # one init_acc event; all remaining nodes must not be accelerator nodes. Fails via test
        # assertions.
        """
        Test the basic functionalities of the tracker by putting it into a
        node finder and examine the events generated after the finder is called.
        """

        def getEvents(tracker, node):
            # [FUNCTION DOC] HELPER: Returns the list of tracker events recorded for the given
            # node, by looking up the event indices stored under the node name in the tracker.
            return [tracker.events[idx] for idx in tracker.node_events[node.name]]

        class IsNodeSupported(OperatorSupportBase):
            # [CLASS DOC] HELPER: Operator support policy that treats a node as supported when its
            # name contains the text sup_.
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                # [FUNCTION DOC] HELPER: Returns True when the node name contains sup_, ignoring
                # submodules.
                return "sup_" in node.name

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two inputs: wraps each input with sup_f, adds them
            # (the unsupported operation that splits the graph) and wraps the sum with sup_f
            # again.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies sup_f to x and y, adds the results and returns
                # sup_f of the sum.
                x = sup_f(x)
                y = sup_f(y)
                b = x + y  # non-supported to break graph
                return sup_f(b)

        gm = torch.fx.symbolic_trace(TestModule())
        ShapeProp(gm).propagate(*(torch.rand((2, 2)), 3))

        finder = FxNetAccNodesFinder(gm, IsNodeSupported(), False)
        finder.tracker = tracker
        acc_nodes = finder()
        # check that acc nodes events are as expected
        for node in gm.graph.nodes:
            if node.name == "sup_f_1":
                # this node should be removed from acc nodes.
                self.assertFalse(node in acc_nodes)
                events = getEvents(tracker, node)
                # 2 events.
                self.assertEqual(len(events), 2)
                # 1st event is init_acc as supported operator
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
                # 2nd event is del_acc as non-tensor output with cpu user
                self.assertTrue(
                    events[1].desc.startswith("acc_del|non_tensor_output_with_cpu_user")
                )
            elif node.name.startswith("sup_f"):
                # other supported nodes should remain in acc nodes.
                self.assertTrue(node in acc_nodes)
                events = getEvents(tracker, node)
                self.assertEqual(len(events), 1)
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
            else:
                # other nodes are on cpu.
                self.assertFalse(node in acc_nodes)

    def _validate_file_content(self, filepath, expected_lines):
        # [FUNCTION DOC] HELPER: Reads a text file line by line and asserts each line (without the
        # trailing newline) equals the corresponding expected line, and that the file has exactly
        # as many lines as expected_lines. Returns None; fails via test assertions.
        """
        Validate the content of the file.
        Args:
            filepath: the path of the file to be validated.
            expected_lines: the expected lines of the file.
        Returns:
            None
        """
        with open(filepath) as f:
            idx = 0
            for line in f:
                self.assertEqual(line.rstrip("\n"), expected_lines[idx])
                idx += 1
                self.assertTrue(idx <= len(expected_lines))
            self.assertEqual(idx, len(expected_lines))

    def _assert_events_file(self, events_file):
        # [FUNCTION DOC] HELPER: Asserts that the events dump file has the expected per-node event
        # listing for the traced module (x, y, sup_f, sup_f_1 with its acc_del event, add,
        # sup_f_2, output).
        self._validate_file_content(
            events_file,
            [
                "Node: x:",
                "  x: init_cpu|not_callable #",
                "Node: y:",
                "  y: init_cpu|not_callable #",
                "Node: sup_f:",
                "  sup_f: init_acc|callable_and_operator_supported #",
                "Node: sup_f_1:",
                "  sup_f_1: init_acc|callable_and_operator_supported #",
                "  sup_f_1: acc_del|non_tensor_output_with_cpu_user add",
                "Node: add:",
                "  add: init_cpu|operator_support #",
                "Node: sup_f_2:",
                "  sup_f_2: init_acc|callable_and_operator_supported #",
                "Node: output:",
                "  output: init_cpu|not_callable #",
            ],
        )

    def _testTrackerMode(self, mode):
        # [FUNCTION DOC] HELPER: Runs the tracker basics check with a NodeEventTracker of the
        # given mode (0 no dump, 1 all events file, 2 plus tracked-nodes file, 3 plus nodes with
        # multiple events) using dump paths under the temp directory. Verifies the dump files do
        # not exist beforehand and that the expected files and contents exist afterwards for that
        # mode.
        """
        Test the tracker with different modes.
        Args:
            mode: the mode to be tested.
            - 0: no local dump
            - 1: dump all events to file
            - 2: dump specific nodes in recursive manner
            - 3: dump all nodes with more than 1 event in recursive manner.
        """
        tmp_dump_base_path = self.tmpdir + "/" + str(mode)
        tracker = NodeEventTracker(
            mode,  # mode: just enable the tracker without dumping
            tmp_dump_base_path,  # dump path
        )
        events_file = tmp_dump_base_path + ALL_SUFFIX
        nodes_file = tmp_dump_base_path + NODES_SUFFIX
        self.assertFalse(os.path.exists(events_file))
        self.assertFalse(os.path.exists(nodes_file))
        self._testTrackerBasics(tracker)

        if mode == 0:
            # Make sure there are no files dumped
            self.assertFalse(os.path.exists(events_file))
            self.assertFalse(os.path.exists(nodes_file))
        elif mode == 1:
            self._assert_events_file(events_file)
            self.assertFalse(os.path.exists(nodes_file))
        elif mode == 2:
            self._assert_events_file(events_file)
            self._validate_file_content(
                nodes_file,
                [
                    "===== Tracking node sup_f_2 =====",
                    "|-sup_f_2: init_acc|callable_and_operator_supported #",
                    "===== End of tracking node sup_f_2 =====",
                ],
            )
        elif mode == 3:
            self._assert_events_file(events_file)
            self._validate_file_content(
                nodes_file,
                [
                    "===== Tracking node sup_f_1 =====",
                    "|-sup_f_1: init_acc|callable_and_operator_supported #",
                    "|-sup_f_1: acc_del|non_tensor_output_with_cpu_user add",
                    "| |-add: init_cpu|operator_support #",
                    "===== End of tracking node sup_f_1 =====",
                ],
            )

    def testMode0(self):
        # [FUNCTION DOC] Mode 0: tracker enabled without any file dump.
        self._testTrackerMode(0)

    def testMode1(self):
        # [FUNCTION DOC] Mode 1: all events are dumped to the events file only.
        self._testTrackerMode(1)

    def testMode2(self):
        # [FUNCTION DOC] Mode 2: sets the tracked-nodes environment variable to sup_f_2 and checks
        # the nodes file tracks that node. KNOWN ISSUE: the environment variable is never
        # restored, so it leaks into later tests in the same process.
        os.environ[ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES] = "sup_f_2"
        self._testTrackerMode(2)

    def testMode3(self):
        # [FUNCTION DOC] Mode 3: dumps events plus the tree for nodes having more than one event.
        self._testTrackerMode(3)
