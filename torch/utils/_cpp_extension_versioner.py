# mypy: allow-untyped-defs

# [MODULE DOC] Tracks per-extension version numbers that increase when a hash of the
# sources and build settings changes. Provides hash helpers and ExtensionVersioner.
import collections


Entry = collections.namedtuple("Entry", "version, hash")


def update_hash(seed, value):
    # [FUNCTION DOC] HELPER: Combines the seed with the hash of value in the style of the
    # boost hash-combine routine and returns the new integer.
    # Good old boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    return seed ^ (hash(value) + 0x9E3779B9 + (seed << 6) + (seed >> 2))


def hash_source_files(hash_value, source_files):
    # [FUNCTION DOC] HELPER: Folds the binary content of each listed file into the hash.
    # Raises OSError if a file cannot be read.
    for filename in source_files:
        with open(filename, "rb") as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_arguments(hash_value, build_arguments):
    # [FUNCTION DOC] HELPER: Folds every argument of every non-empty group of build
    # arguments into the hash and returns it.
    for group in build_arguments:
        if group:
            for argument in group:
                hash_value = update_hash(hash_value, argument)
    return hash_value


class ExtensionVersioner:
    # [CLASS DOC] Mapping from extension name to an entry of version number and last hash;
    # versions start at 0 and bump by one each time the hash changes.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates an empty entries mapping.
        self.entries = {}

    def get_version(self, name):
        # [FUNCTION DOC] Returns the current version of the named extension or None if unknown.
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    def bump_version_if_changed(
        self,
        name,
        source_files,
        build_arguments,
        build_directory,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
    ):
        # [FUNCTION DOC] Hashes the source file contents, build arguments, build directory
        # and the CUDA/SYCL/python-module/standalone flags. First sight of a name records
        # version 0; a changed hash bumps the version and stores the new hash. Returns the
        # version. Side effect: updates the entries mapping; may raise OSError reading files.
        hash_value = 0
        hash_value = hash_source_files(hash_value, source_files)
        hash_value = hash_build_arguments(hash_value, build_arguments)
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, with_sycl)
        hash_value = update_hash(hash_value, is_python_module)
        hash_value = update_hash(hash_value, is_standalone)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version
