# mypy: allow-untyped-defs
# [MODULE DOC]
# Shared infrastructure for FX graph passes in the FX-based ONNX exporter: the abstract
# Transform base class, PackageInfo and GraphModuleOnnxMeta metadata records, and helpers
# that render FX graphs as a table or a full-context unified diff. No import-time side
# effects beyond imports.
from __future__ import annotations

import abc
import contextlib
import dataclasses
import difflib
import io
import sys
from typing import Any, TYPE_CHECKING

import torch
import torch.fx
from torch._subclasses.fake_tensor import unset_fake_temporarily


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._subclasses import fake_tensor


@dataclasses.dataclass
class PackageInfo:
    # [CLASS DOC]
    # Dataclass identifying the Python package that defined an op or module: package name,
    # optional version and optional commit hash (never filled in currently). Used to build
    # ONNX domain strings.
    package_name: str
    version: str | None
    commit_hash: str | None

    def to_onnx_domain_string(self) -> str:
        # [FUNCTION DOC]
        # Returns a dot-joined domain string starting with "pkg" followed by the package
        # name, version and commit hash, skipping any that are empty or None.
        return ".".join(
            filter(None, ("pkg", self.package_name, self.version, self.commit_hash))
        )

    @classmethod
    def from_python_class(cls, python_class_name: type | str) -> PackageInfo:
        # [FUNCTION DOC]
        # Builds a PackageInfo from a class or a dotted module name. A class is replaced
        # by its module name; the top-level package is imported and its version attribute
        # (if any) recorded. The commit hash is always None. Side effect: imports the
        # package; an unimportable name raises ImportError.
        if isinstance(python_class_name, type):
            python_class_name = python_class_name.__module__
        package_name = python_class_name.split(".")[0]
        package = __import__(package_name)
        version = getattr(package, "__version__", None)
        # TODO: Figure out how to retrieve commit hash.
        commit_hash = None
        return cls(package_name, version, commit_hash)


@dataclasses.dataclass
class GraphModuleOnnxMeta:
    # [CLASS DOC] Dataclass holding ONNX-related metadata for a graph module; currently only
    # the PackageInfo of the originating package.
    package_info: PackageInfo


@contextlib.contextmanager
def _patch_difflib_sequence_matcher_init():
    # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the SequenceMatcher
    # constructor in the difflib module so autojunk is always disabled, and restores the
    # original on exit (even on error). Global side effect: not thread safe while active.
    """Context patching `difflib.SequenceMatcher` for fx readable graph.

    Under this context, the `autojunk` argument of `difflib.SequenceMatcher` will always
    be considered as `False`. This is to prevent `difflib.SequenceMatcher` recognizing
    stacktrace messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `difflib.SequenceMatcher` is used underneath by all sorts of diffing functions
    in `difflib`, including `difflib.unified_diff`, `difflib.ndiff`, `difflib.context_diff`.
    Unfortunately, there is no way to pass `autojunk` argument to these functions, and
    they all default to `True`. This context patching will affect all of them.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """
    original_init = difflib.SequenceMatcher.__init__

    def patched_init(self, isjunk=None, a="", b="", autojunk=True) -> None:
        # [FUNCTION DOC] HELPER: Calls the original constructor, forcing autojunk off and
        # ignoring the caller supplied autojunk value.
        original_init(self, isjunk, a, b, autojunk=False)

    difflib.SequenceMatcher.__init__ = patched_init  # type: ignore[assignment]
    try:
        yield
    finally:
        difflib.SequenceMatcher.__init__ = original_init  # type: ignore[assignment]


def _unified_diff(a: str, b: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the unified diff of two strings split into lines, with
    # unlimited context so the whole text is shown and autojunk disabled via the patch
    # context manager. Returns the text "<no diff>" when the inputs are identical.
    """Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    `_patch_difflib_sequence_matcher_init` function.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The unified diff of the two strings. If there is no diff, return "<no diff>".

    Example::

        >>> a = '''class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... '''
        >>> b = '''class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... '''
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    """

    a_list = a.splitlines(keepends=True)
    b_list = b.splitlines(keepends=True)

    with _patch_difflib_sequence_matcher_init():
        # Set `n` to `sys.maxsize` to show entire graph when there is a diff.
        diff = "".join(difflib.unified_diff(a_list, b_list, n=sys.maxsize))

    if not diff:
        return "<no diff>"
    return diff


def _transform_diagnose_call_message_formatter(
    run: Callable,
    self: Transform,
    *args: Any,
    **kwargs: Any,
) -> str:
    # [FUNCTION DOC] HELPER: Message formatter for a diagnostic call wrapper; ignores the
    # wrapped function and arguments and returns a text naming the Transform subclass
    # being run.
    return f"Running {self.__class__.__name__} pass. "


def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> str | None:
    # [FUNCTION DOC]
    # Captures what the graph tabular print writes to standard output and returns it as a
    # string; returns None when the optional tabulate dependency is missing (ImportError).
    # Side effect: temporarily redirects standard output.
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.
    If `tabulate` is not installed, return `None`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format. None if `tabulate` is not installed.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            graph.print_tabular()
        except ImportError:
            return None
    return f.getvalue()


class Transform(abc.ABC):
    # [CLASS DOC]
    # Abstract base for FX graph passes. Holds the module to transform and the fake tensor
    # mode detected from node metadata (None if none). Subclasses implement the abstract
    # _run; the public run delegates to it and may mutate the module or return a new one.
    # Passes that change the model signature need input/output adapt steps elsewhere.
    """Base class for FX graph transformations to be used by FX-ONNX exporter.

    Similar to `FX Interpreter <https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter>`_,
    specializations of this class execute the FX graph Node-by-Node.
    Methods in the `Transform` class can be overridden to customize the behavior of the model.
    This pattern can be useful for many things, including writing code transformations as well as analysis passes.

    The following methods can be overridden::

        _run()
            +-- run_node()
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    One important aspect to note is that if the transformation modifies the model input and/or output signature,
    (e.g. additional inputs/outputs are added to the model), :class:`InputAdaptStep` and/or :class:`OutputAdaptStep`
    are needed to reconcile :attr:`ONNXProgram.model_proto`.
    That is, the model signature and the model representation must match.

    TODO(bowbao): Add more overridable methods in call hierarchy
    TODO(bowbao): Create an example once more overridable methods are added.
    """

    module: torch.fx.GraphModule
    """The module to be transformed."""

    fake_mode: fake_tensor.FakeTensorMode | None
    """The existing fake mode detected from `self.module`."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
    ) -> None:
        # [FUNCTION DOC] Stores the module and detects its fake tensor mode.
        """Initialize the transform.

        Args:
            module: The module to be transformed.
        """
        self.module = module
        self.fake_mode = self._detect_fake_mode()

    def _detect_fake_mode(self) -> fake_tensor.FakeTensorMode | None:
        # [FUNCTION DOC] HELPER: Collects the val metadata of every graph node and asks dynamo
        # to detect a fake mode from them, with any active fake mode temporarily unset.
        # Returns the mode or None.
        """Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
        fake_tensors = [node.meta.get("val") for node in self.module.graph.nodes]
        with unset_fake_temporarily():
            return torch._dynamo.utils.detect_fake_mode(fake_tensors)

    def _maybe_fakefy_args(
        self, fake_mode: fake_tensor.FakeTensorMode | None, *args: Any
    ) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns the args unchanged when fake_mode is None; otherwise
        # converts each tensor argument to a fake tensor in that mode (cached conversions
        # are reused) and passes non-tensors through, returning a tuple.
        if fake_mode is None:
            return args
        # NB: This should hit the cache if tensors were fakefied before.
        # E.g., when the fx graph is produced by Dynamo.
        return tuple(
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args
        )

    # [FUNCTION DOC] HELPER: Abstract hook that subclasses implement to execute the pass on
    # self.module with the given args and kwargs and return the resulting GraphModule.
    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule: ...

    def run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Public entry point: forwards args and kwargs to the subclass _run
        # and returns its GraphModule (either the mutated self.module or a new one).
        """Run the transform on `self.module`.

        Note that this method may or may not mutate `self.module`, and the returned
        `GraphModule` could be either `self.module` or a new `GraphModule`.

        Args:
            *args: Positional arguments for `self.module` to run.
            **kwargs: Keyword arguments for `self.module` to run.
        """
        return self._run(*args, **kwargs)
