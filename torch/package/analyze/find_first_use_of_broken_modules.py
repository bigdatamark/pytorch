# [MODULE DOC] Defines find_first_use_of_broken_modules, which inspects the dependency graph
# of a PackagingError. Importing this module imports the package exporter.
from torch.package.package_exporter import PackagingError


__all__ = ["find_first_use_of_broken_modules"]


def find_first_use_of_broken_modules(exc: PackagingError) -> dict[str, list[str]]:
    # [FUNCTION DOC] Given a PackagingError, collects the graph nodes flagged with an error and
    # returns a mapping from each broken module name to the first dependency path (list of
    # module names) that reached it. Raises AssertionError if the argument is not a
    # PackagingError. Does not mutate the graph.
    """
    Find all broken modules in a PackagingError, and for each one, return the
    dependency path in which the module was first encountered.

    E.g. broken module m.n.o was added to a dependency graph while processing a.b.c,
    then re-encountered while processing d.e.f. This method would return
    {'m.n.o': ['a', 'b', 'c']}

    Args:
        exc: a PackagingError

    Returns: A dict from broken module names to lists of module names in the path.
    """

    if not isinstance(exc, PackagingError):
        raise AssertionError(
            f"exception must be a PackagingError, got {type(exc).__name__}"
        )
    uses = {}
    broken_module_names = [
        m for m, attr in exc.dependency_graph.nodes.items() if attr.get("error", False)
    ]
    for module_name in broken_module_names:
        path = exc.dependency_graph.first_path(module_name)
        uses[module_name] = path
    return uses
