# [MODULE DOC] Defines GraphTransformObserver, a context manager that observes node creation, erasure and
# replacement on a graph module during a pass, annotates node provenance metadata, optionally writes
# before/after dot graphs to a log directory, and can run a pass function with timing and disabling
# support. Imports the graph drawer at import time; no other import-time side effects.
import os
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.traceback import NodeSource, NodeSourceAction


T = TypeVar("T")


from .graph_drawer import FxGraphDrawer


__all__ = ["GraphTransformObserver"]


@compatibility(is_backward_compatible=False)
class GraphTransformObserver:
    # [CLASS DOC] Context manager wrapping one graph pass. State: the module, pass name, optional
    # subsystem, log directory, an active flag (true when a log directory is set or provenance tracking
    # level is one) and, when active, sets of created and erased node names, a name to node mapping, the
    # list of deep-copied modules and the registered hooks. A class-level counter numbers logged passes.
    # Use via the with statement or the apply helpers; when inactive it does nothing.
    __pass_count = 0

    def __init__(
        self,
        gm: GraphModule,
        passname: str,
        subsystem: str | None = None,
        log_url: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the arguments, takes the log directory from inductor configuration when
        # none is given, and decides whether the observer is active. When active, creates the bookkeeping
        # containers and the four hook callbacks. If a log directory is set, increments the global pass
        # counter and renders the input graph to a dot graph (ignoring get_attr, parameters and buffers).
        # Requires the pydot package in that case (the drawer raises otherwise).
        """
        log_url is inferred to be torch._inductor.config.trace.log_url_for_graph_xform unless otherwise specified
        """
        from torch._inductor import config as inductor_config

        self.gm = gm
        self.passname = passname
        self.subsystem = subsystem

        if log_url is None:
            log_url = inductor_config.trace.log_url_for_graph_xform

        self.log_url = log_url

        self.active = (
            self.log_url is not None
            or inductor_config.effective_provenance_tracking_level() == 1
        )

        if self.active:
            self.erased_nodes: set[str] = set()
            self.created_nodes: set[str] = set()
            self.name_to_node: dict[str, Node] = {}
            # record graph modules deepcopied from self.gm, so we can remove hooks on them when exiting the context
            self.copied_gms: list[GraphModule] = []

            self._node_creation_hook = self.get_node_creation_hook()
            self._node_erase_hook = self.get_node_erase_hook()
            self._node_replace_hook = self.get_node_replace_hook()
            self._deepcopy_hook = self.get_deepcopy_hook()

        # If log_url is None, we don't log anything
        if self.log_url is None:
            return
        GraphTransformObserver.__pass_count += 1

        self.input_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

    @classmethod
    def get_current_pass_count(cls) -> int:
        # [FUNCTION DOC] Returns how many logged passes have been constructed so far (class-wide counter).
        return cls.__pass_count

    def apply_gm_pass(self, pass_fn: Callable[[GraphModule], T]) -> T | None:
        # [FUNCTION DOC] Inside the observer context, returns none if the pass is disabled (by name or
        # by the compiler bisector); otherwise calls the function on the graph module under a timing
        # scope named after the subsystem and pass, and returns its result.
        from torch._dynamo.utils import dynamo_timed

        with self:
            if self._check_disable_pass():
                return None
            with dynamo_timed(
                f"pass.{self.subsystem}.{self.passname}"
                if self.subsystem
                else f"pass.{self.passname}"
            ):
                return pass_fn(self.gm)

    def apply_graph_pass(self, pass_fn: Callable[[Graph], T]) -> T | None:
        # [FUNCTION DOC] Same as apply_gm_pass but passes the graph of the module to the function.
        from torch._dynamo.utils import dynamo_timed

        with self:
            if self._check_disable_pass():
                return None
            with dynamo_timed(
                f"pass.{self.subsystem}.{self.passname}"
                if self.subsystem
                else f"pass.{self.passname}"
            ):
                return pass_fn(self.gm.graph)

    def _check_disable_pass(self) -> bool:
        # [FUNCTION DOC] HELPER: True if the pass name appears (case-insensitively, as a substring) in
        # the inductor disabled passes setting, or if the compiler bisector disables the subsystem for
        # this pass. Without a subsystem, only the setting is consulted.
        from torch._inductor import config as inductor_config

        if self.passname.upper() in inductor_config.disabled_passes.upper():
            return True

        if self.subsystem is None:
            return False

        from torch._inductor.compiler_bisector import CompilerBisector

        return CompilerBisector.disable_subsystem(
            "inductor", self.subsystem, lambda: self.passname
        )

    def __enter__(self) -> "GraphTransformObserver":
        # [FUNCTION DOC] When active, registers the create, erase, replace and deepcopy hooks on the
        # module, clears all bookkeeping and indexes the current nodes by name. Returns self.
        if not self.active:
            return self
        self.gm._register_create_node_hook(self._node_creation_hook)
        self.gm._register_erase_node_hook(self._node_erase_hook)
        self.gm._register_replace_node_hook(self._node_replace_hook)
        self.gm._register_deepcopy_hook(self._deepcopy_hook)

        self.erased_nodes.clear()
        self.created_nodes.clear()
        self.name_to_node.clear()
        self.copied_gms.clear()

        for node in self.gm.graph.nodes:
            self.name_to_node[node.name] = node

        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] When active, unregisters the hooks from the module and its deep copies. If a
        # log directory is set and nodes were created or erased, colors the saved input graph (erased
        # nodes yellow, others grey), writes it to a dot file named with the pass counter and pass name,
        # draws the output graph, colors created nodes yellow, and writes that file too. Exception
        # arguments are ignored and exceptions are not suppressed.
        if not self.active:
            return
        for gm in self.copied_gms + [self.gm]:
            gm._unregister_create_node_hook(self._node_creation_hook)
            gm._unregister_erase_node_hook(self._node_erase_hook)
            gm._unregister_replace_node_hook(self._node_replace_hook)
            gm._unregister_deepcopy_hook(self._deepcopy_hook)

        if self.log_url is None:
            return

        if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
            for e in self.input_dot_graph.get_node_list():
                if e.get_name() in self.erased_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            if self.log_url is None:
                raise AssertionError("log_url is not set")
            self.input_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.dot",
                )
            )

            output_dot_graph = FxGraphDrawer(
                self.gm,
                self.passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            for e in output_dot_graph.get_node_list():
                if e.get_name() in self.created_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            output_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.dot",
                )
            )

    def get_node_creation_hook(self) -> Callable[[Node], None]:
        # [FUNCTION DOC] Returns a callback that records a new node name as created, indexes it, and
        # appends a creation source entry for this pass to the from_node metadata (creating the list
        # when absent). Only valid when active (the sets exist).
        # We have to return a function instead of using a class method directly
        # to avoid max recursion issue when deepcopy a graph module within the context manager.
        def on_node_creation(node: Node) -> None:
            # [FUNCTION DOC] HELPER: Hook body for node creation; see get_node_creation_hook.
            self.created_nodes.add(node.name)
            self.name_to_node[node.name] = node
            source = NodeSource(None, self.passname, NodeSourceAction.CREATE)
            if "from_node" not in node.meta:
                node.meta["from_node"] = [source]
            else:
                node.meta["from_node"].append(source)

        return on_node_creation

    def get_node_erase_hook(self) -> Callable[[Node], None]:
        # [FUNCTION DOC] Returns a callback that records an erased node name and drops it from the index.
        def on_node_erase(node: Node) -> None:
            # [FUNCTION DOC] HELPER: Hook body for node erasure; see get_node_erase_hook.
            self.erased_nodes.add(node.name)
            self.name_to_node.pop(node.name, None)

        return on_node_erase

    def get_node_replace_hook(self) -> Callable[[Node, str, Node], None]:
        # [FUNCTION DOC] Returns a callback that updates provenance metadata when one node replaces another.
        def on_node_replace(old: Node, new: str, user: Node) -> None:
            # [FUNCTION DOC] HELPER: Hook body. Looks up the replacement node by name (ignoring unknown
            # names), skips it if the old node is already listed in its provenance, otherwise drops
            # creation entries made by this pass and appends a replace entry (plus create if the node
            # was created in this pass) naming the old node. The user argument is unused.
            # Update node meta when replacing old node with new node
            new_node = self.name_to_node.get(new, None)

            if not new_node:
                return

            if not isinstance(new_node, Node):
                raise AssertionError(f"Expected Node, got {type(new_node)}")

            # replace hook is called once for each user of old
            # this avoids adding duplicated source nodes
            added_nodes = {s.name for s in new_node.meta.get("from_node", [])}
            if old.name in added_nodes:
                return

            action = [NodeSourceAction.REPLACE]
            if new_node.name in self.created_nodes:
                action.append(NodeSourceAction.CREATE)

            def created_this_pass(source: NodeSource) -> bool:
                # [FUNCTION DOC] HELPER: True for a provenance entry from this pass whose action is
                # exactly a single creation.
                return source.pass_name == self.passname and source.action == [
                    NodeSourceAction.CREATE
                ]

            # remove redundant source added on node creation
            new_from_node = new_node.meta.get("from_node", [])
            new_from_node = [
                source for source in new_from_node if not created_this_pass(source)
            ]

            # add new source
            new_node_source = NodeSource(old, self.passname, action)
            new_from_node.append(new_node_source)
            new_node.meta["from_node"] = new_from_node

        return on_node_replace

    def get_deepcopy_hook(self) -> Callable[[GraphModule], None]:
        # [FUNCTION DOC] Returns a callback that remembers deep-copied modules so hooks can be removed later.
        def on_deepcopy(gm: GraphModule) -> None:
            # [FUNCTION DOC] HELPER: Hook body; appends the copied module to the tracked list.
            self.copied_gms.append(gm)

        return on_deepcopy
