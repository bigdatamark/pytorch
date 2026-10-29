# mypy: allow-untyped-defs
# [MODULE DOC] The writing side of torch.package. Defines PackageExporter, which builds a
# zip archive holding source code of modules, pickled objects, and text/binary resources,
# using a dependency graph and user-declared intern/extern/mock/deny patterns to decide how
# each referenced module is handled. Also defines the error reasons, the error types
# (PackagingError, EmptyMatchError), the hook type alias and the set of stdlib modules that
# may not be implicitly externed. Import-time side effects: none beyond definitions.
import collections
import importlib.machinery
import io
import linecache
import os
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, cast, IO

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import FileLike, Storage
from torch.utils.hooks import RemovableHandle

from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer


__all__ = [
    "PackagingErrorReason",
    "EmptyMatchError",
    "PackagingError",
    "PackageExporter",
]

_gate_torchscript_serialization = True

ActionHook = Callable[["PackageExporter", str], None]


class _ModuleProviderAction(Enum):
    # [CLASS DOC] HELPER: Enumeration of what the exporter does with a module: intern (copy
    # source into the archive), extern (record as an external dependency), mock (replace with
    # stubs), deny (error if required), plus two special cases: an already repackaged mock
    # module and the skip action for the package importer's special module.
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """

    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4
    # Special case: when a module is mocked, PackageExporter writes out a
    # `_mock` module that implements our mocking stubs. If we re-package code,
    # we may encounter a `_mock` module from the original package. If we do,
    # just ignore it and write a `_mock` module once.
    REPACKAGED_MOCK_MODULE = 5
    # Special case: PackageImporter adds a fake module
    # (`torch_package_importer`) that allows packaged code to access it. Don't
    # re-export this.
    SKIP = 6


class PackagingErrorReason(Enum):
    # [CLASS DOC] Enumeration of reasons a module can fail to package. Each member's value is
    # the human readable explanation text used in PackagingError messages. Members mark graph
    # nodes in the dependency graph via an error attribute.
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):
        # [FUNCTION DOC] Returns a compact string of the form class name dot member name.
        return f"<{self.__class__.__name__}.{self.name}>"

    IS_EXTENSION_MODULE = (
        "Module is a C extension module. torch.package supports Python modules only."
    )
    NO_DUNDER_FILE = "Module had no __file__ defined."
    SOURCE_FILE_NOT_FOUND = (
        "Module had a __file__, but we could not find it in your filesystem."
    )
    DEPENDENCY_RESOLUTION_FAILED = "Dependency resolution failed."
    NO_ACTION = (
        "Module did not match against any action pattern. Extern, mock, or intern it."
    )
    DENIED = "Module was denied by a pattern."
    MOCKED_BUT_STILL_USED = (
        "Module was mocked out, but is still being used in the package. "
        "Please intern or extern the mocked modules if objects are supposed to be in "
        "the package."
    )


@dataclass
class _PatternInfo:
    # [CLASS DOC] HELPER: Record kept per registered glob pattern: the action to apply to
    # matching modules, whether the user allows the pattern to match nothing, and a flag set
    # when the pattern matched some module during packaging (checked at close time).
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""

    # What action to take on a module that matches this pattern.
    action: _ModuleProviderAction
    # The value of `allow_empty` the user gave when specifying the pattern.
    allow_empty: bool
    # Whether this pattern has been matched during packaging.
    was_matched: bool

    def __init__(self, action, allow_empty):
        # [FUNCTION DOC] HELPER: Stores the action and the allow-empty flag and starts with the
        # matched flag cleared.
        self.action = action
        self.allow_empty = allow_empty
        self.was_matched = False


class EmptyMatchError(Exception):
    # [CLASS DOC] Error raised at close when an extern or mock pattern declared with
    # allow_empty false never matched any module.
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """


class PackagingError(Exception):
    # [CLASS DOC] Aggregate error listing every module in the dependency graph that is marked
    # broken, grouped by reason, with optional context and (in debug mode) a dependency path.
    # Keeps the graph in the public dependency_graph attribute for tooling.
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """

    def __init__(self, dependency_graph: DiGraph, debug=False):
        # [FUNCTION DOC] Builds the error message from the graph: groups nodes carrying an
        # error attribute by reason, then lists each module with optional context, a security
        # note for disallowed stdlib modules, and (if debug) the first path to it; otherwise
        # appends a hint to enable debug. Stores the graph and initializes the base exception
        # with the message. Raises AssertionError if a no-action node also has an action.
        # Group errors by reason.
        broken: dict[PackagingErrorReason, list[str]] = defaultdict(list)
        for module_name, attrs in dependency_graph.nodes.items():
            error = attrs.get("error")
            if error is None:
                continue
            if error == PackagingErrorReason.NO_ACTION:
                if "action" in attrs:
                    raise AssertionError(
                        f"module {module_name} has NO_ACTION error but action is set"
                    )
            broken[error].append(module_name)

        message = io.StringIO()
        message.write("\n")

        for reason, module_names in broken.items():
            message.write(f"* {reason.value}\n")
            for module_name in module_names:
                message.write(f"    {module_name}\n")

                # Print additional context if it's provided.
                error_context = dependency_graph.nodes[module_name].get("error_context")
                if error_context is not None:
                    message.write(f"      Context: {error_context}\n")
                if module_name in _DISALLOWED_MODULES:
                    message.write(
                        "      Note: While we usually use modules in the python standard library "
                        f"from the local environment, `{module_name}` has a lot of system "
                        "level access and therefore can pose a security risk. We heavily "
                        f"recommend removing `{module_name}` from your packaged code. However, if that "
                        "is not possible, add it to the extern list by calling "
                        f'PackageExporter.extern("`{module_name}`")\n'
                    )
                if debug:
                    module_path = dependency_graph.first_path(module_name)
                    message.write(
                        f"      A path to {module_name}: {' -> '.join(module_path)}\n"
                    )
        if not debug:
            message.write("\n")
            message.write(
                "Set debug=True when invoking PackageExporter for a visualization of where "
                "broken modules are coming from!\n"
            )
        # Save the dependency graph so that tooling can get at it.
        self.dependency_graph = dependency_graph
        super().__init__(message.getvalue())


class PackageExporter:
    # [CLASS DOC] Writer of torch.package archives, usable as a context manager (close on
    # exit). State: the zip file writer and optional output buffer, the dependency graph
    # (nodes are module names or pickle names, edges mean depends-on, node attributes hold
    # action, source, errors), the importer used to find modules, registered glob patterns
    # with actions, extern/mock/intern hook tables, set of written files, a unique id
    # counter, and a table of already serialized reduce results. Key methods: save_* to add
    # content, intern/extern/mock/deny to declare policies, close to validate and write all.
    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """

    """A importer that will be searched in order to find the modules referenced by other modules or by
    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.
    """
    importer: Importer

    def __init__(
        self,
        f: FileLike,
        importer: Importer | Sequence[Importer] = sys_importer,
        debug: bool = False,
    ) -> None:
        # [FUNCTION DOC] Creates the exporter: logs API usage, opens the zip writer on the
        # path or on the given binary buffer, sets the minimum archive version, and initializes
        # empty graph, patterns, hook tables and counters. A single Importer is used as is; a
        # sequence is wrapped in an OrderedImporter. Raises TypeError if importer is neither.
        # Side effects: creates or truncates the output file.
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passed, an ``OrderedImporter`` will be constructed out of them.
            debug: If set to True, add path of broken modules to PackagingErrors.
        """
        torch._C._log_api_usage_once("torch.package.PackageExporter")
        self.debug = debug
        if isinstance(f, (str, os.PathLike)):
            f = os.fspath(f)
            self.buffer: IO[bytes] | None = None
        else:  # is a byte buffer
            self.buffer = f

        self.zip_file = torch._C.PyTorchFileWriter(f)
        self.zip_file.set_min_version(6)
        self._written_files: set[str] = set()

        self.serialized_reduces: dict[int, Any] = {}

        # A graph tracking all the modules and pickle objects added to this
        # package and the dependencies between them.
        # - Each node is a module name (or a pickle name that looks like '<foo.obj.pkl>')
        # - Each directed edge (u, v) means u depends on v.
        # - Nodes may contain metadata that describe how to write the thing to the zipfile.
        self.dependency_graph = DiGraph()
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file)
        self.storage_context = self.script_module_serializer.storage_context()

        # These are OrderedDicts for compatibility with RemovableHandle.
        # Generic OrderedDict type annotations are not present until 3.7.
        # The real type signature is OrderedDict[int, Callable[[PackageExporter, str], None]]
        self._extern_hooks: OrderedDict = OrderedDict()
        self._mock_hooks: OrderedDict = OrderedDict()
        self._intern_hooks: OrderedDict = OrderedDict()

        if isinstance(importer, Importer):
            self.importer = importer
        else:
            if not isinstance(importer, collections.abc.Sequence):
                raise TypeError(
                    "importer arg should be an Importer or a sequence of Importers, "
                    f"got {type(importer)} instead."
                )
            self.importer = OrderedImporter(*importer)

        self.patterns: dict[GlobGroup, _PatternInfo] = {}
        self._unique_id = 0

    def save_source_file(
        self, module_name: str, file_or_directory: str, dependencies=True
    ):
        # [FUNCTION DOC] Adds code from a local file or directory as the source of the named
        # module. For a directory, walks all Python files recursively, derives submodule names
        # from relative paths (init files mark packages), and only after collecting all of
        # them saves each through save_source_string, so overwritten modules are not pulled
        # in as dependencies. For a file, saves its content, treating an init file as a
        # package. Reads files; mutates the dependency graph.
        """Adds the local file system ``file_or_directory`` to the source package to provide the code
        for ``module_name``.

        Args:
            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.
            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory
                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated
                as a package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        path = Path(file_or_directory)
        if path.is_dir():
            to_save = []  # list of tuples with arguments to save_source_string
            module_path = module_name.replace(".", "/")
            for filename in path.glob("**/*.py"):
                relative_path = filename.relative_to(path).as_posix()
                archivename = module_path + "/" + relative_path
                submodule_name = None
                if filename.name == "__init__.py":
                    submodule_name = archivename[: -len("/__init__.py")].replace(
                        "/", "."
                    )
                    is_package = True
                else:
                    submodule_name = archivename[: -len(".py")].replace("/", ".")
                    is_package = False

                # we delay the call to save_source_string so that we record all the source files
                # being provided by this directory structure _before_ attempting to resolve the dependencies
                # on the source. This makes sure we don't try to copy over modules that will just get
                # overwritten by this directory blob
                to_save.append(
                    (
                        submodule_name,
                        _read_file(str(filename)),
                        is_package,
                        dependencies,
                    )
                )

            for item in to_save:
                self.save_source_string(*item)
        else:
            is_package = path.name == "__init__.py"
            self.save_source_string(
                module_name,
                _read_file(file_or_directory),
                is_package,
                dependencies,
            )

    def get_unique_id(self) -> str:
        # [FUNCTION DOC] Returns the next value of an internal counter as a string and
        # increments it, so each call on this exporter yields a distinct id.
        """Get an id. This id is guaranteed to only be handed out once for this package."""
        ret = str(self._unique_id)
        self._unique_id += 1
        return ret

    def _get_dependencies(
        self, src: str, module_name: str, is_package: bool
    ) -> list[str]:
        # [FUNCTION DOC] HELPER: Scans the source for import references (relative to the module
        # itself if it is a package, else its parent) and returns an ordered, unique list of
        # module names that can actually be imported through this exporter's importer. A
        # from-import of a name that is a real submodule yields the submodule only. If scanning
        # fails, records a resolution-failed error on the module's graph node and returns an
        # empty list.
        """Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
        package_name = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        try:
            dep_pairs = find_files_source_depends_on(src, package_name)
        except Exception as e:
            self.dependency_graph.add_node(
                module_name,
                error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED,
                error_context=str(e),
            )
            return []

        # Use a dict to get uniquing but also deterministic order
        dependencies = {}
        for dep_module_name, dep_module_obj in dep_pairs:
            # handle the case where someone did something like `from pack import sub`
            # where `sub` is a submodule. In this case we don't have to save pack, just sub.
            # this ensures we don't pick up additional dependencies on pack.
            # However, in the case where `sub` is not a submodule but an object, then we do have
            # to save pack.
            if dep_module_obj is not None:
                possible_submodule = f"{dep_module_name}.{dep_module_obj}"
                if self._module_exists(possible_submodule):
                    dependencies[possible_submodule] = True
                    # we don't need to save `pack`
                    continue
            if self._module_exists(dep_module_name):
                dependencies[dep_module_name] = True

        return list(dependencies.keys())

    def save_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
        dependencies: bool = True,
    ):
        # [FUNCTION DOC] Registers the given text as the source of the module, as an interned
        # and provided node in the dependency graph (flagged as a package if requested). If
        # dependencies is true, finds its imports, adds graph edges from the module to each, and
        # runs add_dependency on each so patterns decide their handling. Replaces any previous
        # attributes of the node.
        """Adds ``src`` as the source code for ``module_name`` in the exported package.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code for this package.
            src (str): The Python source code to save for this package.
            is_package (bool, optional): If ``True``, this module is treated as a package. Packages are allowed to have submodules
                (e.g. ``my_package.my_subpackage.my_subsubpackage``), and resources can be saved inside them. Defaults to ``False``.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        self.dependency_graph.add_node(
            module_name,
            source=src,
            is_package=is_package,
            provided=True,
            action=_ModuleProviderAction.INTERN,
        )

        if dependencies:
            deps = self._get_dependencies(src, module_name, is_package)

            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def _write_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Writes the source into the archive at the path derived from the
        # module name (slashes for dots; init file inside a directory for packages, else a
        # py suffix). Delegates to the internal write, which raises if the path was already
        # written.
        """Write ``src`` as the source code for ``module_name`` in the zip archive.

        Arguments are otherwise the same as for :meth:`save_source_string`.
        """
        extension = "/__init__.py" if is_package else ".py"
        filename = module_name.replace(".", "/") + extension

        self._write(filename, src)

    def _import_module(self, module_name: str):
        # [FUNCTION DOC] HELPER: Imports the module through the exporter's importer. For a
        # mangled name that is not found, re-raises ModuleNotFoundError with a message saying
        # the originating PackageImporter must be in the importer list; unmangled misses
        # propagate unchanged.
        try:
            return self.importer.import_module(module_name)
        except ModuleNotFoundError:
            if not is_mangled(module_name):
                raise
            msg = (
                f"Module not found: '{module_name}'. Make sure the PackageImporter that "
                "created this module is present in `self.importer`"
            )
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if importing the module through the exporter
        # succeeds, False on any exception. Importing may run module code (side effects).
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _get_source_of_module(self, module: types.ModuleType) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the module's source text, or None if unavailable.
        # Finds the file name via a source file loader on the module spec when possible, else
        # via the file attribute; only names ending in a py suffix are read, through the line
        # cache (which may use module globals to find the source).
        filename = None
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            loader = getattr(spec, "loader", None)
            if loader is not None and isinstance(loader, SourceFileLoader):
                try:
                    filename = loader.get_filename(module.__name__)
                except ImportError:
                    pass
        if filename is None:
            filename = getattr(module, "__file__", None)
        if isinstance(filename, str) and filename.endswith(".py"):
            return "".join(linecache.getlines(filename, module.__dict__))
        return None

    def add_dependency(self, module_name: str, dependencies=True):
        # [FUNCTION DOC] Decides how the named module is handled and records it in the graph.
        # Does nothing if already provided. Special names (the package importer module and the
        # mock module) get skip or repackaged-mock actions; implicitly externable modules get
        # extern. Otherwise the first matching registered pattern sets the action (marking the
        # pattern matched; deny also records an error; intern also loads the module source and
        # its dependencies). With no match, records a no-action error on the node.
        """Given a module, add it to the dependency graph according to patterns
        specified by the user.
        """
        if (
            module_name in self.dependency_graph
            and self.dependency_graph.nodes[module_name].get("provided") is True
        ):
            return

        # Special case: PackageImporter provides a special module called
        # `torch_package_importer` that allows packaged modules to reference
        # their PackageImporter. We don't want to re-export this.
        if module_name == "torch_package_importer":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.SKIP,
                provided=True,
            )
            return

        if module_name == "_mock":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.REPACKAGED_MOCK_MODULE,
                provided=True,
            )
            return

        if self._can_implicitly_extern(module_name):
            self.dependency_graph.add_node(
                module_name, action=_ModuleProviderAction.EXTERN, provided=True
            )
            return

        for pattern, pattern_info in self.patterns.items():
            if pattern.matches(module_name):
                pattern_info.was_matched = True
                self.dependency_graph.add_node(
                    module_name, action=pattern_info.action, provided=True
                )

                if pattern_info.action == _ModuleProviderAction.DENY:
                    # Requiring a denied module just adds an error to the graph.
                    self.dependency_graph.add_node(
                        module_name, error=PackagingErrorReason.DENIED
                    )

                # If we are interning this module, we need to retrieve its
                # dependencies and package those as well.
                if pattern_info.action == _ModuleProviderAction.INTERN:
                    self._intern_module(module_name, dependencies)
                return

        # No patterns have matched. Explicitly add this as an error.
        self.dependency_graph.add_node(
            module_name, error=PackagingErrorReason.NO_ACTION
        )

    def save_module(self, module_name: str, dependencies=True):
        # [FUNCTION DOC] Public entry to add a module's code to the package: validates that the
        # name is a string (raises TypeError otherwise), then interns the module, finding its
        # source through the importer and optionally its dependencies.
        """Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the
        module object, and then using its ``__file__`` attribute to find the source code.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code
                for this package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        if not isinstance(module_name, str):
            raise TypeError(
                "save_module() expects a string input, did you perhaps mean to pass `__name__`?"
            )

        self._intern_module(module_name, dependencies)

    def _intern_module(
        self,
        module_name: str,
        dependencies: bool,
    ):
        # [FUNCTION DOC] HELPER: Imports the module via the importer, demangles its name, and
        # adds an interned node holding its source and package flag. If no source exists,
        # the node gets an error instead (no file, extension module, or file not found with
        # context), and nothing else happens. If dependencies is true, finds imports, adds edges
        # and calls add_dependency for each. Raises ModuleNotFoundError if import fails.
        """Adds the module to the dependency graph as an interned module,
        along with any metadata needed to write it out to the zipfile at serialization time.
        """
        module_obj = self._import_module(module_name)
        # Subtle: if the import above succeeded, either:
        #   1. The module name is not mangled, and this was just a regular import, or
        #   2. The module name is mangled, but one of the importers was able to
        #      recognize the mangling and import it.
        # Either way, it is now safe to demangle this name so that we don't
        # serialize the mangled version to the package.
        module_name = demangle(module_name)

        # Find dependencies of this module and require them as well.
        is_package = hasattr(module_obj, "__path__")
        source = self._get_source_of_module(module_obj)
        if source is None:
            # Couldn't find a source!  Add it to our dependency graph as broken
            # and continue.
            filename = getattr(module_obj, "__file__", None)
            error_context = None
            if filename is None:
                packaging_error = PackagingErrorReason.NO_DUNDER_FILE
            elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
            else:
                packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
                error_context = f"filename: {filename}"
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.INTERN,
                is_package=is_package,
                error=packaging_error,
                error_context=error_context,
                provided=True,
            )
            return

        self.dependency_graph.add_node(
            module_name,
            action=_ModuleProviderAction.INTERN,
            is_package=is_package,
            source=source,
            provided=True,
        )

        if dependencies:
            deps = self._get_dependencies(source, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def save_pickle(
        self,
        package: str,
        resource: str,
        obj: Any,
        dependencies: bool = True,
        pickle_protocol: int = 3,
    ):
        # [FUNCTION DOC] Pickles obj with a package-aware pickler (using the exporter importer
        # and a persistent-id hook for storages and objects with a package reduce method) and
        # writes the bytes under the package and resource path. Adds a pickle node to the
        # dependency graph. If dependencies is true, scans the pickle opcodes for global
        # references (protocol 4: string pairs before a stack-global opcode; protocol 3:
        # global opcode), adds an edge per referenced module and calls add_dependency, except
        # that modules matching a mock pattern become error nodes (mocked but still used).
        # Raises AssertionError for protocols other than 3 or 4 and for repeated file paths.
        # KNOWN ISSUE: the docstring does not document pickle_protocol, and the protocol 4
        # opcode scan is a heuristic that infers module and name from the last two strings
        # seen, so unrelated strings could be misread as globals.
        """Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into
        the archive rather than a stand-alone file. Standard pickle does not save the code, only the objects.
        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required
        to reconstruct them and save the relevant code.

        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,
        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that
        have previously been packaged, the importer's ``import_module`` method will need to be present in the ``importer`` list
        for this to work.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to save, must be picklable.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """

        if pickle_protocol not in (3, 4):
            raise AssertionError(
                f"torch.package only supports pickle protocols 3 and 4, got {pickle_protocol}"
            )

        filename = self._filename(package, resource)
        # Write the pickle data for `obj`
        data_buf = io.BytesIO()
        pickler = create_pickler(data_buf, self.importer, protocol=pickle_protocol)
        pickler.persistent_id = self._persistent_id
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        mocked_modules = defaultdict(list)
        name_in_dependency_graph = f"<{package}.{resource}>"
        self.dependency_graph.add_node(
            name_in_dependency_graph,
            action=_ModuleProviderAction.INTERN,
            provided=True,
            is_pickle=True,
        )

        def _check_mocked_error(module: str | None, field: str | None):
            # [FUNCTION DOC] HELPER: Given a module and attribute name seen in the pickle,
            # records the attribute in the enclosing mocked-modules mapping if the first
            # matching user pattern for that module is a mock action. Implicitly externed
            # modules are ignored. Raises AssertionError if either argument is not a string.
            """
            checks if an object (field) comes from a mocked module and then adds
            the pair to mocked_modules which contains mocked modules paired with their
            list of mocked objects present in the pickle.

            We also hold the invariant that the first user defined rule that applies
            to the module is the one we use.
            """

            if not isinstance(module, str):
                raise AssertionError(f"module must be str, got {type(module).__name__}")
            if not isinstance(field, str):
                raise AssertionError(f"field must be str, got {type(field).__name__}")
            if self._can_implicitly_extern(module):
                return
            for pattern, pattern_info in self.patterns.items():
                if pattern.matches(module):
                    if pattern_info.action == _ModuleProviderAction.MOCK:
                        mocked_modules[module].append(field)
                    return

        if dependencies:
            all_dependencies = []
            module = None
            field = None
            memo: defaultdict[int, str] = defaultdict(None)
            memo_count = 0
            # pickletools.dis(data_value)
            # pyrefly: ignore [bad-assignment]
            for opcode, arg, _pos in pickletools.genops(data_value):
                if pickle_protocol == 4:
                    if (
                        opcode.name == "SHORT_BINUNICODE"
                        or opcode.name == "BINUNICODE"
                        or opcode.name == "BINUNICODE8"
                    ):
                        if not isinstance(arg, str):
                            raise AssertionError(
                                f"expected str arg for {opcode.name}, got {type(arg).__name__}"
                            )
                        module = field
                        field = arg
                        memo[memo_count] = arg
                    elif (
                        opcode.name == "LONG_BINGET"
                        or opcode.name == "BINGET"
                        or opcode.name == "GET"
                    ):
                        if not isinstance(arg, int):
                            raise AssertionError(
                                f"expected int arg for {opcode.name}, got {type(arg).__name__}"
                            )
                        module = field
                        field = memo.get(arg, None)
                    elif opcode.name == "MEMOIZE":
                        memo_count += 1
                    elif opcode.name == "STACK_GLOBAL":
                        if module is None:
                            # If no module was passed on in the entries preceding this one, continue.
                            continue
                        if not isinstance(module, str):
                            raise AssertionError(
                                f"module must be str, got {type(module).__name__}"
                            )
                        if module not in all_dependencies:
                            all_dependencies.append(module)
                        _check_mocked_error(module, field)
                elif (
                    pickle_protocol == 3 and opcode.name == "GLOBAL"
                ):  # a global reference
                    if not isinstance(arg, str):
                        raise AssertionError(
                            f"expected str arg for GLOBAL, got {type(arg).__name__}"
                        )
                    module, field = arg.split(" ")
                    if module not in all_dependencies:
                        all_dependencies.append(module)
                    _check_mocked_error(module, field)
            for module_name in all_dependencies:
                self.dependency_graph.add_edge(name_in_dependency_graph, module_name)

                """ If an object happens to come from a mocked module, then we collect these errors and spit them
                    out with the other errors found by package exporter.
                """
                if module_name in mocked_modules:
                    if not isinstance(module_name, str):
                        raise AssertionError(
                            f"module_name must be str, got {type(module_name).__name__}"
                        )
                    fields = mocked_modules[module_name]
                    self.dependency_graph.add_node(
                        module_name,
                        action=_ModuleProviderAction.MOCK,
                        error=PackagingErrorReason.MOCKED_BUT_STILL_USED,
                        error_context=f"Object(s) '{fields}' from module `{module_name}` was mocked out during packaging "
                        f"but is being used in resource - `{resource}` in package `{package}`. ",
                        provided=True,
                    )
                else:
                    self.add_dependency(module_name)

        self._write(filename, data_value)

    def save_text(self, package: str, resource: str, text: str):
        # [FUNCTION DOC] Saves text into the archive at the path for package and resource,
        # encoded as UTF-8, by delegating to save_binary.
        """Save text data to the package.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            text (str): The contents to save.
        """
        return self.save_binary(package, resource, text.encode("utf-8"))

    def save_binary(self, package, resource, binary: bytes):
        # [FUNCTION DOC] Writes raw bytes into the archive at the path built from package and
        # resource (package dots become slashes, resource path normalized). Raises
        # AssertionError if that path was already written.
        # KNOWN ISSUE: the docstring types the data argument as str, but bytes are expected.
        """Save raw bytes to the package.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            binary (str): The data to save.
        """
        filename = self._filename(package, resource)
        self._write(filename, binary)

    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        # [FUNCTION DOC] Adds the hook to the extern hook table (called with the exporter and
        # module name for each externed module at close, in registration order) and returns a
        # handle whose remove method unregisters it.
        """Registers an extern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`extern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._extern_hooks)
        self._extern_hooks[handle.id] = hook
        return handle

    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        # [FUNCTION DOC] Adds the hook to the mock hook table (called with the exporter and
        # module name for each mocked module at close, in registration order) and returns a
        # handle whose remove method unregisters it.
        """Registers a mock hook on the exporter.

        The hook will be called each time a module matches against a :meth:`mock` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._mock_hooks)
        self._mock_hooks[handle.id] = hook
        return handle

    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        # [FUNCTION DOC] Adds the hook to the intern hook table (called with the exporter and
        # module name for each interned module at close, in registration order) and returns a
        # handle whose remove method unregisters it.
        """Registers an intern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`intern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._intern_hooks)
        self._intern_hooks[handle.id] = hook
        return handle

    def intern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        # [FUNCTION DOC] Registers a glob group (include and exclude patterns) with the intern
        # action in the pattern table, recording the allow-empty flag. Modules matching it are
        # copied into the package and their dependencies processed. Later registrations of an
        # equal-looking pattern are separate entries; earlier patterns win on matching.
        """Specify modules that should be packaged. A module must match some ``intern`` pattern in order to be
        included in the package and have its dependencies processed recursively.

        Args:
            include (list[str] | str): A string e.g. "my_package.my_subpackage", or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.

            allow_empty (bool): An optional flag that specifies whether the intern modules specified by this call
                to the ``intern`` method must be matched to some module during packaging. If an ``intern`` module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``)
                before any modules match that pattern, an exception is thrown. If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.INTERN, allow_empty
        )

    def mock(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        # [FUNCTION DOC] Registers a glob group with the mock action in the pattern table,
        # recording the allow-empty flag. Matching modules are written as stub modules that
        # return mocked objects for any attribute. If allow_empty is false and nothing
        # matches, closing raises EmptyMatchError.
        """Replace some required modules with a mock implementation.  Mocked modules will return a fake
        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes
        find files that are imported by model files but whose functionality is never used
        (e.g. custom serialization code or training helpers).
        Use this function to mock this functionality out without having to modify the original code.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be mocked out. Strings can also be a glob-style pattern
                string that may match multiple modules. Any required dependencies that match this pattern
                string will be mocked out automatically.

                Examples :
                    ``'torch.**'`` -- matches ``torch`` and all submodules of torch, e.g. ``'torch.nn'``
                    and ``'torch.nn.functional'``

                    ``'torch.*'`` -- matches ``'torch.nn'`` or ``'torch.functional'``, but not
                    ``'torch.nn.functional'``

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.
                e.g. ``include='torch.**', exclude='torch.foo'`` will mock all torch packages except ``'torch.foo'``,
                Default: is ``[]``.

            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call
                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with
                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has
                not been matched to a module used by the package being exported, an exception is thrown.
                If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.MOCK, allow_empty
        )

    def extern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        # [FUNCTION DOC] Registers a glob group with the extern action in the pattern table,
        # recording the allow-empty flag. Matching modules are not copied but listed in the
        # archive as external dependencies. If allow_empty is false and nothing matches,
        # closing raises EmptyMatchError.
        """Include ``module`` in the list of external modules the package can import.
        This will prevent dependency discovery from saving
        it in the package. The importer will load an external module directly from the standard import system.
        Code for extern modules must also exist in the process loading the package.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as
                described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the
                include string.

            allow_empty (bool): An optional flag that specifies whether the extern modules specified by this call
                to the ``extern`` method must be matched to some module during packaging. If an extern module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via
                ``__exit__``) before any modules match that pattern, an exception is thrown. If ``allow_empty=True``,
                no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.EXTERN, allow_empty
        )

    def deny(self, include: "GlobPattern", *, exclude: "GlobPattern" = ()):
        # [FUNCTION DOC] Registers a glob group with the deny action (always allowing empty
        # matches). Any required module that matches is marked with a denied error, so closing
        # raises PackagingError.
        """Blocklist modules whose names match the given glob patterns from the list of modules the package can import.
        If a dependency on any matching packages is found, a :class:`PackagingError` is raised.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.
        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.DENY, allow_empty=True
        )

    def _persistent_id(self, obj):
        # [FUNCTION DOC] HELPER: Pickler hook deciding how to save special objects. For tensor
        # storages (typed or untyped), writes the storage bytes into the archive once (moving
        # to CPU if needed) and returns a tuple of tag, storage type, storage id, location tag
        # and element count. For objects with a package reduce method, returns (and caches by
        # object identity) a tuple of tag, identity and the method's results; script modules
        # are refused while the beta gate flag is on. Returns None for everything else.
        # Raises RuntimeError for unrecognized storage kinds and a bare Exception for gated
        # script modules. Side effects: writes archive records, updates the reduce cache.
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            storage: Storage
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, we can
                # remove this case
                untyped_storage = obj._untyped_storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage = cast(Storage, untyped_storage)
                storage_numel = obj.size()

            elif isinstance(obj, torch.UntypedStorage):
                untyped_storage = obj
                storage = cast(Storage, untyped_storage)
                storage_type = normalize_storage_type(type(storage))
                storage_numel = storage.nbytes()
            else:
                raise RuntimeError(f"storage type not recognized: {type(obj)}")

            location = location_tag(storage)

            # serialize storage if not already written
            storage_present = self.storage_context.has_storage(storage)
            storage_id = self.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage, num_bytes
                )
            return ("storage", storage_type, storage_id, location, storage_numel)

        if hasattr(obj, "__reduce_package__"):
            if _gate_torchscript_serialization and isinstance(
                obj, torch.jit.RecursiveScriptModule
            ):
                raise Exception(  # noqa: TRY002
                    "Serializing ScriptModules directly into a package is a beta feature. "
                    "To use, set global "
                    "`torch.package.package_exporter._gate_torchscript_serialization` to `False`."
                )
            if self.serialized_reduces.get(id(obj)) is None:
                self.serialized_reduces[id(obj)] = (
                    "reduce_package",
                    id(obj),
                    *obj.__reduce_package__(self),
                )

            return self.serialized_reduces[id(obj)]

        return None

    def __enter__(self):
        # [FUNCTION DOC] Context manager entry; returns the exporter itself.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] Context manager exit. If the block raised, only finalizes the zip
        # (so a buffer stays valid) and lets the exception propagate; otherwise calls close,
        # which may itself raise PackagingError or EmptyMatchError. Never suppresses errors.
        # If __exit__ was called because an exception was raised, we do not
        # attempt to finalize the package. Instead, control is returned to the
        # caller to continue raising the exception.
        if exc_type is not None:
            # Do the bare minimum to leave the open buffer in a valid state.
            self._finalize_zip()
            return

        self.close()

    def _write(self, filename, str_or_bytes):
        # [FUNCTION DOC] HELPER: Writes one record to the archive. Text is encoded as UTF-8.
        # Raises AssertionError if the file name was already written in this archive or if it
        # looks like a mangled module name. Side effects: updates the written-files set and
        # writes to the zip.
        if filename in self._written_files:
            raise AssertionError(
                f"Tried to write file '{filename}', but it already exists in this archive. "
                "Please file a bug."
            )
        self._written_files.add(filename)

        if is_mangled(filename):
            raise AssertionError(
                f"Tried to save a torch.package'd module as '{filename}'. "
                "Directly saving torch.package'd modules is not allowed."
            )
        if isinstance(str_or_bytes, str):
            str_or_bytes = str_or_bytes.encode("utf-8")
        self.zip_file.write_record(filename, str_or_bytes, len(str_or_bytes))

    def _validate_dependency_graph(self):
        # [FUNCTION DOC] HELPER: Raises PackagingError if any graph node carries an error, and
        # raises EmptyMatchError if a pattern registered with allow-empty false never matched.
        # Does not mutate anything.
        # 1. Check the graph for any errors inserted during dependency analysis.
        for attrs in self.dependency_graph.nodes.values():
            if "error" in attrs:
                raise PackagingError(self.dependency_graph, debug=self.debug)

        # 2. Check that all patterns for which allow_empty=False have been matched at least once.
        for pattern, pattern_info in self.patterns.items():
            if not pattern_info.allow_empty and not pattern_info.was_matched:
                raise EmptyMatchError(
                    f"Exporter did not match any modules to {pattern}, which was marked as allow_empty=False"
                )

    def _write_mock_file(self):
        # [FUNCTION DOC] HELPER: Writes the mock support module source (copied from the file
        # next to this module) into the archive, only once per archive.
        if "_mock.py" not in self._written_files:
            mock_file = str(Path(__file__).parent / "_mock.py")
            self._write_source_string("_mock", _read_file(mock_file), is_package=False)

    def _execute_dependency_graph(self):
        # [FUNCTION DOC] HELPER: Validates the graph, then walks every node and acts on its
        # recorded action: extern nodes run extern hooks and are collected; mock nodes run
        # hooks, ensure the mock support module is written, and write a stub module (package
        # or plain, as the real module is); intern nodes run hooks and write their source
        # (pickle nodes write nothing); repackaged-mock nodes write the mock support module;
        # skip nodes are ignored. Finally writes the list of extern modules to a data file.
        # Raises AssertionError for unprovided interned nodes or unknown actions, plus the
        # validation errors. Side effects: writes archive records, calls user hooks.
        """Takes a finalized dependency graph describing how to package all
        modules and executes it, writing to the ZIP archive.
        """
        self._validate_dependency_graph()

        extern_modules = []
        for module_name, attrs in self.dependency_graph.nodes.items():
            action = attrs["action"]

            if action == _ModuleProviderAction.EXTERN:
                for hook in self._extern_hooks.values():
                    hook(self, module_name)

                extern_modules.append(module_name)

            elif action == _ModuleProviderAction.MOCK:
                for hook in self._mock_hooks.values():
                    hook(self, module_name)

                self._write_mock_file()

                is_package = hasattr(self._import_module(module_name), "__path__")
                self._write_source_string(module_name, _MOCK_IMPL, is_package)

            elif action == _ModuleProviderAction.INTERN:
                for hook in self._intern_hooks.values():
                    hook(self, module_name)

                # The node in the dependency graph contains metadata that tells us
                # how to intern the module.
                if "provided" not in attrs:
                    raise AssertionError(
                        f"Module was marked `intern` but not provided: {module_name}"
                    )

                if attrs.get("is_pickle") is True:
                    # This node came from save_pickle, we don't need to write any source for it.
                    continue

                is_package = attrs["is_package"]
                source = attrs["source"]
                self._write_source_string(module_name, source, is_package)

            elif action == _ModuleProviderAction.REPACKAGED_MOCK_MODULE:
                self._write_mock_file()
            elif action == _ModuleProviderAction.SKIP:
                continue
            else:
                raise AssertionError(
                    f"Invalid action: {module_name}, {action}. Please report a bug to PyTorch."
                )

        extern_file_contents = "\n".join(extern_modules) + "\n"
        self._write(".data/extern_modules", extern_file_contents)

    def _write_python_version(self):
        # [FUNCTION DOC] HELPER: Records the running interpreter version string in the archive's
        # version data file.
        """Writes the python version that the package was created with to .data/python_version"""
        self._write(".data/python_version", platform.python_version())

    def close(self):
        # [FUNCTION DOC] Finishes the package: executes the dependency graph (validating and
        # writing all modules), writes the python version, flushes script module files, and
        # finalizes the zip. Raises PackagingError or EmptyMatchError on invalid state. The
        # exporter must not be used afterwards (its zip writer is deleted).
        """Write the package to the filesystem. Any calls after :meth:`close` are now invalid.
        It is preferable to use resource guard syntax instead::

            with PackageExporter("file.zip") as e:
                ...
        """
        self._execute_dependency_graph()
        self._write_python_version()

        self.script_module_serializer.write_files()
        self._finalize_zip()

    def _finalize_zip(self):
        # [FUNCTION DOC] HELPER: Deletes the zip writer attribute (which closes the archive) and
        # flushes the user buffer if one was given.
        """Called at the very end of packaging to leave the zipfile in a closed but valid state."""
        del self.zip_file
        if self.buffer:
            self.buffer.flush()

    def _filename(self, package, resource):
        # [FUNCTION DOC] HELPER: Builds the archive path for a resource: package name with dots
        # turned into slashes, a slash, then the normalized resource path.
        package_path = package.replace(".", "/")
        resource = _normalize_path(resource)
        return f"{package_path}/{resource}"

    def _can_implicitly_extern(self, module_name: str):
        # [FUNCTION DOC] HELPER: Returns True if the top-level package is torch, or is a
        # standard library module that is not in the disallowed list; such modules are
        # externed automatically without a user pattern.
        top_level_package_name = module_name.partition(".")[0]
        return top_level_package_name == "torch" or (
            top_level_package_name not in _DISALLOWED_MODULES
            and is_stdlib_module(top_level_package_name)
        )

    def dependency_graph_string(self) -> str:
        # [FUNCTION DOC] Returns the dependency graph rendered in dot (graphviz) text.
        """Returns digraph string representation of dependencies in package.

        Returns:
            A string representation of dependencies in package.
        """
        return self.dependency_graph.to_dot()

    def _nodes_with_action_type(
        self, action: _ModuleProviderAction | None
    ) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the sorted names of non-pickle graph nodes whose
        # recorded action equals the given one (None selects nodes without an action).
        result = []
        for name, node_dict in self.dependency_graph.nodes.items():
            node_action = node_dict.get("action", None)
            if node_action == action and "is_pickle" not in node_dict:
                result.append(name)
        result.sort()
        return result

    def externed_modules(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted names of modules currently marked as extern.
        """Return all modules that are currently externed.

        Returns:
            A list containing the names of modules which will be
            externed in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.EXTERN)

    def interned_modules(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted names of modules currently marked as intern.
        """Return all modules that are currently interned.

        Returns:
            A list containing the names of modules which will be
            interned in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.INTERN)

    def mocked_modules(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted names of modules currently marked as mock.
        """Return all modules that are currently mocked.

        Returns:
            A list containing the names of modules which will be
            mocked in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.MOCK)

    def denied_modules(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted names of modules currently marked as deny.
        """Return all modules that are currently denied.

        Returns:
            A list containing the names of modules which will be
            denied in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.DENY)

    def get_rdeps(self, module_name: str) -> list[str]:
        # [FUNCTION DOC] Returns the names of modules that have an edge to the given module in
        # the dependency graph (its predecessors), or an empty list if it is not in the graph.
        """Return a list of all modules which depend on the module ``module_name``.

        Returns:
            A list containing the names of modules which depend on ``module_name``.
        """
        if module_name in self.dependency_graph._pred:
            return list(self.dependency_graph._pred[module_name].keys())
        else:
            return []

    def all_paths(self, src: str, dst: str) -> str:
        # [FUNCTION DOC] Returns dot text of the subgraph containing all paths from src to dst,
        # computed by the dependency graph.
        """Return a dot representation of the subgraph
           that has all paths from src to dst.

        Returns:
            A dot representation containing all paths from src to dst.
            (https://graphviz.org/doc/info/lang.html)
        """
        return self.dependency_graph.all_paths(src, dst)


# even though these are in the standard library, we do not allow them to be
# automatically externed since they offer a lot of system level access
_DISALLOWED_MODULES = ["sys", "io"]

_MOCK_IMPL = """\
from _mock import MockedObject
def __getattr__(attr: str):
    return MockedObject(__name__ + '.' + attr, _suppress_err=True)
"""


def _read_file(filename: str) -> str:
    # [FUNCTION DOC] HELPER: Reads the file in binary mode and returns its content decoded as
    # UTF-8. Raises OSError or UnicodeDecodeError on failure.
    with open(filename, "rb") as f:
        b = f.read()
        return b.decode("utf-8")
