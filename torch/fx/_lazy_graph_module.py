# [MODULE DOC] Implements a lazily-recompiling GraphModule subclass (_LazyGraphModule) plus the
# global switches and context managers that select it. Recompilation (code generation and
# forward installation) is deferred until the code or forward is actually used, saving time
# for many small graphs (e.g. inductor pattern matching). Import-time side effects: defines two
# module-level boolean flags (use lazy, force skip lazy), both False.
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from torch.fx.graph_module import (
    _format_import_block,
    GraphModule,
    reduce_graph_module,
    reduce_package_graph_module,
)
from torch.package import PackageExporter, sys_importer

from ._compatibility import compatibility


if TYPE_CHECKING:
    from torch.fx.graph import PythonCode


_use_lazy_graph_module_flag = False
_force_skip_lazy_graph_module_flag = False


@compatibility(is_backward_compatible=False)
@contextmanager
def _force_skip_lazy_graph_module() -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the global force-skip flag for the body,
    # so lazy graph modules are not used regardless of the use-lazy setting, and restores the
    # prior flag value on exit (even on error). Mutates module-level state; not thread safe.
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
    try:
        global _force_skip_lazy_graph_module_flag
        prior = _force_skip_lazy_graph_module_flag
        _force_skip_lazy_graph_module_flag = True
        yield
    finally:
        _force_skip_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
@contextmanager
def _use_lazy_graph_module(should_use: bool) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the global use-lazy flag to should_use
    # (forced to false while the force-skip flag is set) for the body, then restores the prior
    # value on exit, even on error. Mutates module-level state; not thread safe.
    try:
        global _use_lazy_graph_module_flag
        prior = _use_lazy_graph_module_flag
        _use_lazy_graph_module_flag = (
            should_use and not _force_skip_lazy_graph_module_flag
        )
        yield
    finally:
        _use_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
def _get_graph_module_cls() -> type[GraphModule]:
    # [FUNCTION DOC] HELPER: Return the lazy graph module class if the use-lazy flag is set,
    # otherwise the regular GraphModule class.
    return _LazyGraphModule if _use_lazy_graph_module_flag else GraphModule


def _make_graph_module(
    *args: Any, graph_module_cls: type[GraphModule] | None = None, **kwargs: Any
) -> GraphModule:
    # [FUNCTION DOC] HELPER: Construct a graph module, forwarding all positional and keyword
    # arguments to the class given by graph_module_cls, or the class chosen by the current
    # global setting when it is None. Returns the new instance.
    if graph_module_cls is None:
        graph_module_cls = _get_graph_module_cls()

    return graph_module_cls(*args, **kwargs)


@compatibility(is_backward_compatible=False)
def _unwrap_lazy_graph_module(gm: GraphModule) -> GraphModule:
    # [FUNCTION DOC] HELPER: If gm is a lazy graph module, build and return a plain GraphModule
    # from it and its graph (needed for scripting); otherwise return gm unchanged.
    # torch.jit.script cannot script _LazyGraphModule, so rebuild a plain one
    if isinstance(gm, _LazyGraphModule):
        return GraphModule(gm, gm.graph)
    return gm


@compatibility(is_backward_compatible=False)
class _LazyGraphModule(GraphModule):
    # [CLASS DOC] HELPER: GraphModule whose recompile call only marks that recompilation is
    # needed, by pointing the class-level forward at a lazy forward. The real code generation
    # runs on first access of code, string conversion, serialization or forward call.
    # Invariant: recompilation is pending exactly when the class forward is the lazy forward.
    # Relies on each graph module instance having its own class, so marking is per instance.
    """
    The main difference between _LazyGraphModule and GraphModule is how recompile happens.
    GraphModule will do a 'recompile' call to generate python code and the forward method when it's
    constructed. Later on if the graph get updated, recompile method can be called again to refresh
    the saved python code and forward method.

    However in some cases especially in inductor, the recompilation can be a waste since we never
    check the python code for the graph module or call its forward method. A few more concreate
    examples regarding pattern matching fx passes in inductor:
    1. some passes will update the graph to be compiled and then call recompile on the GraphModule.
    2. some passes will trace small pattern function to search it in the graph being compiled and
       replace the match with the traced graph of a replacement function. The pattern graph and
       replacement graph are quite small but there are large amount of them. Doing GraphModule.recompile
       for them in GraphModule.__init__ is also a waste of time.

    However simply skip calling GraphModule.recompile in these scenarios is also dangeruous.
    People may want to check the python code or call the GraphModule's forward method for debugging purposes.

    The way _LazyGraphModule solves it is, we override the recompile method to just mark the
    need for recompilation but does not do the actual recompilation. Later on if people really
    access the compiled python code or call the GraphModule's forward method, we do the real
    recompilation.
    """

    @classmethod
    def from_graphmodule(cls, gm: GraphModule) -> GraphModule:
        # [FUNCTION DOC] Return gm itself if already lazy; otherwise build a new lazy graph
        # module from gm and its graph.
        if isinstance(gm, _LazyGraphModule):
            return gm
        else:
            return _LazyGraphModule(gm, gm.graph)

    @staticmethod
    def force_recompile(gm: GraphModule) -> None:
        # [FUNCTION DOC] If gm is lazy, perform any pending recompilation now; no-op for other
        # graph modules. Returns nothing.
        """
        Sometimes we need force a recompile as a workaround
        - we want to do the real recompilation before symbolic_trace to avoid error:
            https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
        """
        if isinstance(gm, _LazyGraphModule):
            gm.real_recompile()

    def real_recompile(self) -> None:
        # [FUNCTION DOC] Run the actual recompilation only if one is pending; otherwise do
        # nothing. Makes repeated calls cheap.
        if self._needs_recompile():
            self._real_recompile()

    @classmethod
    def _needs_recompile(cls) -> bool:
        # [FUNCTION DOC] HELPER: True when the class forward is still the lazy forward, i.e.
        # real recompilation has not yet happened.
        return cls.forward is cls._lazy_forward

    def _lazy_forward(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Placeholder forward. Performs pending recompilation, raises an
        # assertion error if recompilation is still pending afterwards, then calls the module
        # through its call operator with the same arguments and returns the result.
        # Call self.real_recompile() rather than self._real_recompile() here.
        # The _lazy_forward method may be saved and call repeatedly.
        # Calling self.real_recompile can make sure we skip recompilation if
        # we have already done so.
        self.real_recompile()
        if self._needs_recompile():
            raise AssertionError("Recompilation required after real_recompile()")

        # call `__call__` rather than 'forward' since recompilation may
        # install a wrapper for `__call__` to provide a customized error
        # message.
        return self(*args, **kwargs)

    forward = _lazy_forward

    def __reduce_package__(
        self, exporter: PackageExporter
    ) -> tuple[Any, tuple[Any, str]]:
        # [FUNCTION DOC] Package-export hook: force real recompilation, copy the instance
        # state without the graph (recording the class name), save generated source (imports
        # plus code) into the exporter under a unique generated module name, and return the
        # package rebuild function with the state and that module name. Side effect: writes
        # a source string into the exporter.
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        python_code = self._real_recompile()
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        import_block = _format_import_block(python_code.globals, exporter.importer)
        module_code = import_block + self.code
        exporter.save_source_string(generated_module_name, module_code)
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, str]]:
        # [FUNCTION DOC] Pickle hook: force real recompilation, copy instance state without the
        # graph, and return the graph-module rebuild function with that state and the import
        # block text for the generated code's globals.
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        python_code = self._real_recompile()
        dict_without_graph = self.__dict__.copy()
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        return (reduce_graph_module, (dict_without_graph, import_block))

    def _real_recompile(self) -> "PythonCode":
        # [FUNCTION DOC] HELPER: Unconditionally call the parent recompile, generating code and
        # installing the real forward; returns the generated code object.
        return super().recompile()

    @classmethod
    def recompile(cls) -> None:  # pyrefly: ignore[bad-override]
        # [FUNCTION DOC] Override that only marks recompilation as pending by setting the class
        # forward to the lazy forward. Does no code generation and returns nothing (unlike the
        # parent, which returns the generated code).
        cls.forward = cls._lazy_forward

    @property
    def code(self) -> str:
        # [FUNCTION DOC] Property: ensure pending recompilation is done, then return the
        # generated source text from the parent.
        self.real_recompile()
        return super().code

    def __str__(self) -> str:
        # [FUNCTION DOC] Ensure pending recompilation is done, then return the parent's text.
        """
        str(GraphModule) will access the _code attribute. Make sure recompile
        happens so _code attribute is available.
        """
        self.real_recompile()
        return super().__str__()
