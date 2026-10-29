# mypy: allow-untyped-defs
# [MODULE DOC] Static discovery of the modules a piece of Python source imports. Defines an
# AST visitor that records import statements and dynamic import calls, and exposes it as
# find_files_source_depends_on. No import-time side effects.
import ast

from ._importlib import _resolve_name


class _ExtractModuleReferences(ast.NodeVisitor):
    # [CLASS DOC] HELPER: AST visitor collecting module references of a source file. State:
    # the enclosing package name (for resolving relative imports) and an insertion-ordered
    # mapping whose keys are pairs of module name and optional attribute or submodule name.
    # Key entry point is the run class method.
    # KNOWN ISSUE: the docstring speaks of global variables read and written, but the class
    # only records imported modules.
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> list[tuple[str, str | None]]:
        # [FUNCTION DOC] HELPER: Parses the source text, visits the tree with a new visitor
        # for the given package, and returns the list of unique (module, attribute-or-None)
        # reference pairs in first-seen order. Raises SyntaxError if the source is invalid.
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        # [FUNCTION DOC] HELPER: Stores the package name used for relative import resolution
        # and starts with an empty references mapping.
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        # [FUNCTION DOC] HELPER: Converts a possibly relative module name to an absolute one
        # using the stored package when level is positive; otherwise returns it unchanged.
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        # [FUNCTION DOC] HELPER: Records each imported module name of a plain import
        # statement as a reference with no attribute.
        for alias in node.names:
            self.references[(alias.name, None)] = True

    def visit_ImportFrom(self, node):
        # [FUNCTION DOC] HELPER: Resolves the source module of a from-import to an absolute
        # name, then records a reference per imported name (pairing the module with that
        # name, which may turn out to be a submodule) or, for a star import, the module alone.
        # KNOWN ISSUE: a relative import with no module part (module is None) is passed to
        # name resolution as is and may fail.
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        for alias in node.names:
            # from my_package import foo
            # foo may be a module, so we have to add it to the list of
            # potential references, if import of it fails, we will ignore it
            if alias.name != "*":
                self.references[(name, alias.name)] = True
            else:
                self.references[(name, None)] = True

    def _grab_node_int(self, node):
        # [FUNCTION DOC] HELPER: Returns the constant value held by a literal AST node.
        return node.value

    def _grab_node_str(self, node):
        # [FUNCTION DOC] HELPER: Returns the constant value held by a literal AST node
        # (same as the integer variant; no type checking).
        return node.value

    def visit_Call(self, node):
        # [FUNCTION DOC] HELPER: Handles calls to the builtin import function with literal
        # arguments: reads the module name, the optional fromlist and level (positional or
        # keyword). With an empty fromlist records the full name and its parent package; with
        # a fromlist records one reference per entry (star gives the module alone). Any
        # failure to read non-literal arguments is swallowed silently and nothing is recorded.
        # KNOWN ISSUE: the parent package is recorded by stripping the last name component,
        # which the code comments call the top-level package; and the unresolved name is
        # recorded for level-relative calls with an empty fromlist.
        # __import__ calls aren't routed to the visit_Import/From nodes
        if hasattr(node.func, "id") and node.func.id == "__import__":
            try:
                name = self._grab_node_str(node.args[0])
                fromlist: list[str] = []
                level = 0
                if len(node.args) > 3:
                    fromlist.extend(self._grab_node_str(v) for v in node.args[3].elts)
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "fromlist":
                            fromlist.extend(
                                self._grab_node_str(v) for v in keyword.value.elts
                            )
                if len(node.args) > 4:
                    level = self._grab_node_int(node.args[4])
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "level":
                            level = self._grab_node_int(keyword.value)
                if fromlist == []:
                    # the top-level package (the name up till the first dot) is returned
                    # when the fromlist argument is empty in normal import system,
                    # we need to include top level package to match this behavior and last
                    # level package to capture the intended dependency of user
                    self.references[(name, None)] = True
                    top_name = name.rsplit(".", maxsplit=1)[0]
                    if top_name != name:
                        top_name = self._absmodule(top_name, level)
                        self.references[(top_name, None)] = True
                else:
                    name = self._absmodule(name, level)
                    for alias in fromlist:
                        # fromlist args may be submodules, so we have to add the fromlist args
                        # to the list of potential references. If import of an arg fails we
                        # will ignore it, similar to visit_ImportFrom
                        if alias != "*":
                            self.references[(name, alias)] = True
                        else:
                            self.references[(name, None)] = True
            except Exception:
                return


find_files_source_depends_on = _ExtractModuleReferences.run
