# mypy: allow-untyped-defs
# [MODULE DOC] File tree representation of a package's contents. Defines Directory (a node
# in a tree of directories and files, printable as a tree) and a helper that builds a tree
# from a list of file paths filtered by glob patterns. No import-time side effects.

from .glob_group import GlobGroup, GlobPattern


__all__ = ["Directory"]


class Directory:
    # [CLASS DOC] Node of a file tree. Attributes: name (this node's own name), is_dir (True
    # for directories, False for files) and children (mapping from child name to child node;
    # empty for files). Paths use forward slashes. Key methods: has_file for lookup and
    # string conversion, which renders an ASCII-art style tree with directories listed first.
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""

    def __init__(self, name: str, is_dir: bool):
        # [FUNCTION DOC] Creates a node with the given name and directory flag and an empty
        # children mapping.
        self.name = name
        self.is_dir = is_dir
        self.children: dict[str, Directory] = {}

    def _get_dir(self, dirs: list[str]) -> "Directory":
        # [FUNCTION DOC] HELPER: Walks down the list of directory names recursively, creating
        # missing directory children as it goes, and returns the node at the end (self if the
        # list is empty). Mutates the tree.
        """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
        if len(dirs) == 0:
            return self
        dir_name = dirs[0]
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        return self.children[dir_name]._get_dir(dirs[1:])

    def _add_file(self, file_path: str):
        # [FUNCTION DOC] HELPER: Splits the slash-separated path, creates the intermediate
        # directories, and adds the last component as a file node (replacing any existing
        # child with that name). Mutates the tree.
        """Adds a file to a Directory.

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories.
        """
        *dirs, file = file_path.split("/")
        dir = self._get_dir(dirs)
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str) -> bool:
        # [FUNCTION DOC] Returns True if the slash-separated path names an existing child
        # (file or directory) below this node, looking up one component at a time
        # recursively; False otherwise. Pure.
        """Checks if a file is present in a :class:`Directory`.

        Args:
            filename (str): Path of file to search for.
        Returns:
            bool: If a :class:`Directory` contains the specified file.
        """
        lineage = filename.split("/", maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        if child in self.children:
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        # [FUNCTION DOC] Returns the multi-line tree rendering of this node and its
        # descendants, built by the recursive stringify helper.
        str_list: list[str] = []
        self._stringify_tree(str_list)
        return "".join(str_list)

    def _stringify_tree(
        self,
        str_list: list[str],
        preamble: str = "",
        dir_ptr: str = "\u2500\u2500\u2500 ",
    ):
        # [FUNCTION DOC] HELPER: Appends to str_list one line for this node (prefixed by the
        # preamble and pointer glyph), then recursively appends its children: subdirectories
        # sorted by name first, then files sorted by name, using branch glyphs that mark the
        # last entry differently. Mutates str_list.
        """Recursive method to generate print-friendly version of a Directory."""
        space = "    "
        branch = "\u2502   "
        tee = "\u251c\u2500\u2500 "
        last = "\u2514\u2500\u2500 "

        # add this directory's representation
        str_list.append(f"{preamble}{dir_ptr}{self.name}\n")

        # add directory's children representations
        if dir_ptr == tee:
            preamble = preamble + branch
        else:
            preamble = preamble + space

        file_keys: list[str] = []
        dir_keys: list[str] = []
        for key, val in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)

        for index, key in enumerate(sorted(dir_keys)):
            if (index == len(dir_keys) - 1) and len(file_keys) == 0:
                self.children[key]._stringify_tree(str_list, preamble, last)
            else:
                self.children[key]._stringify_tree(str_list, preamble, tee)
        for index, file in enumerate(sorted(file_keys)):
            pointer = last if (index == len(file_keys) - 1) else tee
            str_list.append(f"{preamble}{pointer}{file}\n")


def _create_directory_from_file_list(
    filename: str,
    file_list: list[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Directory:
    # [FUNCTION DOC] HELPER: Builds a Directory tree named filename from the list of file
    # paths, keeping only those matching the include glob patterns and not the exclude
    # ones (slash separator). Returns the root node. Pure apart from building new nodes.
    """Return a :class:`Directory` file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (list[str] | str): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern.

        exclude (list[str] | str): An optional pattern that excludes files whose name match the pattern.

    Returns:
            :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.
    """
    glob_pattern = GlobGroup(include, exclude=exclude, separator="/")

    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir._add_file(file)
    return top_dir
