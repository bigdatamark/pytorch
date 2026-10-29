# [MODULE DOC] Provides Interpreter, which executes an fx Graph node by node with overridable per-opcode methods,
# and Transformer, an Interpreter that re-traces a GraphModule symbolically into a new GraphModule.
# Also defines a debug-format helper for nodes. Import-time side effects: creates the module logger only.
import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
from torch._logging import LazyString, trace_structured
from torch.hub import tqdm

from . import config
from ._compatibility import compatibility
from ._lazy_graph_module import _make_graph_module
from ._symbolic_trace import Tracer
from .graph import Graph
from .graph_module import GraphModule
from .node import Argument, map_aggregate, map_arg, Node, Target
from .proxy import Proxy


log = logging.getLogger(__name__)

__all__ = ["Interpreter", "Transformer"]


def _format_fx_node(n: Node) -> str:
    # [FUNCTION DOC] HELPER: Builds a one-line debug string for node n of the form name = module.target(args, kwargs).
    # The module prefix comes from the target's module attribute when present; the target name falls back to the target itself.
    # Args are stringified positionally then as key=value pairs. Pure; no side effects.
    """
    Format a torch.fx.Node into a human-readable string for debug logging.

    Args:
        n (torch.fx.Node): The FX node being executed.

    Returns:
        str: A formatted string describing the node operation, including its
        name, target, positional arguments, and keyword arguments.
    """
    module_prefix = getattr(n.target, "__module__", "")
    module_prefix = f"{module_prefix}." if module_prefix else ""

    # Handle positional and keyword arguments
    args = ", ".join(map(str, n.args))
    kwargs = ", ".join(f"{k}={v}" for k, v in n.kwargs.items())
    joined = ", ".join(filter(None, [args, kwargs]))

    return (
        f"{n.name} = {module_prefix}{getattr(n.target, '__name__', n.target)}({joined})"
    )


@compatibility(is_backward_compatible=True)
class Interpreter:
    # [CLASS DOC] Executes an fx Graph node by node with concrete values.
    # State: module (source of attributes and submodules), submodules (mapping from qualified name to module), graph (the graph run),
    # env (mapping from Node to computed value), name, garbage_collect_values, extra_traceback, and user_to_last_uses
    # (when garbage collection is on: mapping from a node to the nodes whose values die after it runs) plus args_iter during a run.
    # Dispatch: run walks the nodes and run_node calls the method named by node.op (placeholder, get_attr, call_function,
    # call_method, call_module, output); subclasses override these to customize execution.
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overridable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(
                    self, target: Target, args: Tuple, kwargs: Dict
                ) -> Any:
                    if target is torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())

    Args:
        module (torch.nn.Module): The module to be executed
        garbage_collect_values (bool): Whether to delete values after their last
            use within the Module's execution. This ensures optimal memory usage during
            execution. This can be disabled to, for example, examine all of the intermediate
            values in the execution by looking at the ``Interpreter.env`` attribute.
        graph (Optional[Graph]): If passed, the interpreter will execute this
            graph instead of `module.graph`, using the provided `module`
            argument to satisfy any requests for state.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        module: torch.nn.Module,
        garbage_collect_values: bool = True,
        graph: Graph | None = None,
    ) -> None:
        # [FUNCTION DOC] Sets up the interpreter for module, using graph if given else module.graph.
        # Builds the submodule name mapping, empty env and flags. If garbage_collect_values is true, scans nodes in reverse
        # to record, for each node, the last user node so values can be freed after that user executes.
        # Side effects: stores state on self only. Raises AttributeError if no graph is given and module has no graph attribute.
        self.module = module
        self.submodules = dict(self.module.named_modules())
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.module.graph  # type: ignore[assignment]
        self.env: dict[Node, Any] = {}
        self.name = "Interpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True

        if self.garbage_collect_values:
            # Run through reverse nodes and record the first instance of a use
            # of a given node. This represents the *last* use of the node in the
            # execution order of the program, which we will use to free unused
            # values
            node_to_last_use: dict[Node, Node] = {}
            self.user_to_last_uses: dict[Node, list[Node]] = {}

            def register_last_uses(n: Node, user: Node) -> None:
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    self.user_to_last_uses.setdefault(user, []).append(n)

            for node in reversed(self.graph.nodes):
                for n in node._input_nodes:
                    register_last_uses(n, node)

    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        # [FUNCTION DOC] Runs the graph on the positional args and returns the output value.
        # initial_env pre-populates node values (those nodes are skipped); enable_io_processing applies graph.process_inputs to the
        # args and graph.process_outputs to the result. Shows a progress bar, executes each node via run_node, stores the result in env,
        # and frees dead values if garbage collection is on. On an exception, augments the message with the node description,
        # original stack trace and a tlparse hint (logging a structured trace artifact when module is a GraphModule); KeyError becomes
        # RuntimeError, others are re-raised. Returns None implicitly if the graph has no output node.
        # KNOWN ISSUE: self.env is set to initial_env itself (not a copy), so the caller's mapping is mutated, including by garbage collection.
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)
        pbar = tqdm(
            total=len(self.graph.nodes),
            desc=f"{self.name}: {str(list(self.graph.nodes)) if config.verbose_progress else ''}",
            initial=0,
            position=0,
            leave=True,
            disable=config.disable_progress,
            delay=0,
        )

        for node in self.graph.nodes:
            pbar.update(1)
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                if self.extra_traceback:
                    msg = f"While executing {node.format_node()}"
                    msg = f"{e.args[0]}\n\n{msg}" if e.args else str(msg)
                    msg += f"\nOriginal traceback:\n{node.stack_trace}"
                    if (
                        isinstance(self.module, GraphModule)
                        and self.module.graph is not None
                        and isinstance(self.module.graph, torch.fx.Graph)
                    ):
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "fx_interpreter_error",
                                "encoding": "string",
                            },
                            payload_fn=lambda: (
                                f"{msg}\nGraphModule: "
                                f"{self.module.print_readable(print_output=False, include_stride=True)}"  # type: ignore[operator]
                            ),
                        )

                    msg += "\nUse tlparse to see full graph. "
                    msg += "(https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)"
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                return (
                    self.graph.process_outputs(output_val)
                    if enable_io_processing
                    else output_val
                )

    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list: list[Any]) -> Any:
        # [FUNCTION DOC] Runs the graph using the boxed calling convention: args_list holds one value per placeholder node, in order.
        # Raises RuntimeError if the count differs from the number of placeholders. Binds the values to placeholders, clears args_list
        # in place so inputs can be freed early, then runs with that initial environment and returns the result.
        """
        Run `module` via interpretation and return the result.  This uses the "boxed"
        calling convention, where you pass a list of arguments, which will be cleared
        by the interpreter.  This ensures that input tensors are promptly deallocated.
        """
        # Collect placeholder nodes first
        placeholder_nodes = [n for n in self.graph.nodes if n.op == "placeholder"]

        # Check argument count
        if len(args_list) != len(placeholder_nodes):
            detail = (
                "extra arguments"
                if len(args_list) > len(placeholder_nodes)
                else "missing arguments"
            )
            raise RuntimeError(
                f"Interpreter.boxed_run expected {len(placeholder_nodes)} arguments for placeholders "
                f"but received {len(args_list)} ({detail})"
            )

        # Assign arguments to placeholders
        env = dict(zip(placeholder_nodes, args_list))
        args_list.clear()
        return self.run(initial_env=env)

    @contextmanager
    def _set_current_node(self, node: Node) -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Context manager that marks node as the current fx traceback metadata source (named after the
        # Interpreter subclass) for the duration of the with block.
        with fx_traceback.set_current_meta(
            node, f"Interpreter_{self.__class__.__name__}"
        ):
            yield

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Executes node n: logs it at debug level, fetches concrete args and kwargs from env, then calls the method
        # named by n.op with target, args and kwargs and returns its result. Raises AssertionError if fetched args are not a tuple
        # or kwargs not a dict.
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        log.debug("run_node %s", LazyString(lambda: _format_fx_node(n)))
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            if not isinstance(args, tuple):
                raise AssertionError(f"Expected args to be tuple, got {type(args)}")
            if not isinstance(kwargs, dict):
                raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
            return getattr(self, n.op)(n.target, args, kwargs)

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Returns the value for a placeholder node. A target starting with an asterisk consumes all remaining inputs as a list;
        # otherwise the next input is taken. If inputs are exhausted, returns the default in args if present, else raises RuntimeError.
        # Raises AssertionError if target is not a string. Advances args_iter.
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        if target.startswith("*"):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            try:
                return next(self.args_iter)
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(
                        f"Expected positional argument for parameter {target}, but one was not passed in!"
                    ) from si

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Returns the attribute named by target (dotted path) looked up on self.module via fetch_attr.
        # Raises AssertionError if target is not a string; RuntimeError if the path does not exist.
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Calls the callable target with args and kwargs and returns its result. Exceptions from the callee propagate.
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        if isinstance(target, str):
            raise AssertionError("target should not be a string for call_function")

        # Execute the function and return the result
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Calls the method named target on the first positional arg (the receiver) with the remaining args and kwargs
        # and returns the result. Raises AssertionError if target is not a string.
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return getattr(self_obj, target)(*args_tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Looks up the submodule named by target via fetch_attr, calls it with args and kwargs and returns the result.
        # Raises AssertionError if target is not a string.
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        submod = self.fetch_attr(target)

        return submod(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Returns the value referenced by an output node, which is the first positional arg.
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]

    # Helper methods
    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Any:
        # [FUNCTION DOC] Resolves the dotted attribute path target by walking from self.module and returns the value.
        # Raises RuntimeError naming the missing prefix if any step is absent.
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split(".")
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[: i + 1])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(
        self, n: Node
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC] Returns the (args, kwargs) of node n with every Node replaced by its value in env (via map_nodes_to_values).
        # Raises AssertionError if the mapped args is not a tuple or kwargs not a dict; RuntimeError for missing values.
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        args = self.map_nodes_to_values(n.args, n)
        if not isinstance(args, tuple):
            raise AssertionError(f"Expected args to be tuple, got {type(args)}")
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        if not isinstance(kwargs, dict):
            raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
        return args, kwargs

    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        # [FUNCTION DOC] Recursively replaces each Node in the nested structure args with its value from env. n is used only in the error text.
        # Raises RuntimeError if a referenced node has no value in env.
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

        def load_arg(n_arg: Node) -> Any:
            # [FUNCTION DOC] HELPER: Returns env value for n_arg or raises RuntimeError suggesting Graph.lint.
            if n_arg not in self.env:
                raise RuntimeError(
                    f"Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() "
                    f"to diagnose such issues"
                )
            return self.env[n_arg]

        return map_arg(args, load_arg)


@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    # [CLASS DOC] An Interpreter that runs symbolically to build a new GraphModule rather than computing values.
    # State: new_graph (graph being built, sharing the source codegen) and tracer (a leaf-everything Tracer writing into new_graph with root set to the module).
    # Overrides placeholder, get_attr, call_module and call_function to emit nodes and return Proxy objects; transform returns the result.
    """
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target is torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)

            transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, module: GraphModule) -> None:
        # [FUNCTION DOC] Initializes the base Interpreter, creates an empty new_graph using the codegen of module's graph, and
        # creates a tracer bound to it whose root is module.
        super().__init__(module)
        self.new_graph = Graph()
        self.new_graph.set_codegen(module.graph._codegen)

        class TransformerTracer(Tracer):
        # [CLASS DOC] HELPER: Tracer that writes into a supplied graph, treats every module as a leaf, and starts with empty tensor_attrs.
            def __init__(self, graph: Graph) -> None:
            # [FUNCTION DOC] HELPER: Initializes the Tracer and sets its graph to the supplied one.
                super().__init__()
                self.graph = graph
                self.tensor_attrs: dict[torch.Tensor, str] = {}  # type: ignore[assignment]

            def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            # [FUNCTION DOC] HELPER: Always returns True so modules are never inlined.
                return True

        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] Adds a placeholder named target to new_graph (default taken from the first arg if given) and returns a Proxy for it.
        # Raises AssertionError if target is not a string.
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        default_value = next(iter(args)) if args else inspect.Signature.empty
        return Proxy(
            self.new_graph.placeholder(target, default_value=default_value), self.tracer
        )

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] Adds a get_attr node for target to new_graph via the tracer and returns its Proxy.
        # Raises AssertionError if target is not a string.
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return self.tracer.create_proxy("get_attr", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Fetches the submodule and delegates to the tracer's call_module so its leaf-module policy applies (here always a leaf).
        # Raises AssertionError if target is not a string.
        # Override so that the leaf module policy from `self.tracer` is respected.
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Emits a call_function node for target through the tracer and returns the resulting Proxy, preserving wrapped functions.
        # Override so that functions that were wrapped are still wrapped.
        return self.tracer.create_proxy("call_function", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule:
        # [FUNCTION DOC] Runs the interpretation symbolically (preserving node metadata, without IO processing) to fill new_graph,
        # adds an output node from the result (unwrapping Proxy objects), copies metadata from the old output node, and returns a new GraphModule
        # built from self.module and new_graph. Raises AssertionError if the last node of the old graph is not an output node.
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Argument | Proxy) -> Any:
                # [FUNCTION DOC] HELPER: Returns the underlying node for a Proxy, otherwise the value unchanged.
                return a.node if isinstance(a, Proxy) else a

            new_output_node = self.new_graph.output(map_aggregate(result, strip_proxy))
            # also preserve the metadata from the old output node, if it exists
            old_output_node = list(self.graph.nodes)[-1]
            if old_output_node.op != "output":
                raise AssertionError(
                    f"Expected output node, got op={old_output_node.op}"
                )
            for k, v in old_output_node.meta.items():
                new_output_node.meta[k] = v

        return _make_graph_module(self.module, self.new_graph)
