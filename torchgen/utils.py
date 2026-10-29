# [MODULE DOC] Shared utilities for the torchgen code generator.
# Provides path constants (TORCHGEN_ROOT, REPO_ROOT), the Target enum, schema name splitting,
# lazy map helpers, an error-context manager, a stable string hash, FileManager (template
# based, optionally sharded file writer that only touches changed files), dataclass pretty
# printing helpers, NamespaceHelper for C++ namespaces and an insertion-ordered OrderedSet.
# Import time side effects: none besides computing the path constants.
from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import is_dataclass
from enum import auto, Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, Self

from torchgen.code_template import CodeTemplate


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterable, Iterator, Sequence


TORCHGEN_ROOT = Path(__file__).absolute().parent
REPO_ROOT = TORCHGEN_ROOT.parent


# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Literal[Target.XXX] or Literal[Target.XXX, Target.YYY]
# what targets are valid for your use.
class Target(Enum):
    # [CLASS DOC] Enum selecting which kind of generated C++ code a shared codegen function
    # should emit: definition, declaration, registration, or anonymous/namespaced variants.
    # Open enum; callers should restrict valid members explicitly.
    # top level namespace (not including at)
    DEFINITION = auto()
    DECLARATION = auto()
    # TORCH_LIBRARY(...) { ... }
    REGISTRATION = auto()
    # namespace { ... }
    ANONYMOUS_DEFINITION = auto()
    # namespace cpu { ... }
    NAMESPACED_DEFINITION = auto()
    NAMESPACED_DECLARATION = auto()


# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r"(^|\W){}($|\W)"


# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> tuple[str, list[str]]:
    # [FUNCTION DOC] Splits a function schema string of the form name(params) (optional
    # overload suffix allowed) into the bare name and the list of parameter strings, split
    # on comma-space. Raises RuntimeError if the text does not match.
    # KNOWN ISSUE: regex based and splits on every comma-space, so nested types containing
    # such separators are split wrongly (see the TODO above); an empty parameter list yields
    # a list holding one empty string.
    m = re.match(r"(\w+)(\.\w+)?\((.*)\)", schema)
    if m is None:
        raise RuntimeError(f"Unsupported function schema: {schema}")
    name, _, params = m.groups()
    return name, params.split(", ")


T = TypeVar("T")
S = TypeVar("S")

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them


# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], S | None], xs: Iterable[T]) -> Iterator[S]:
    # [FUNCTION DOC] Lazily applies func to each element of xs and yields only the results
    # that are not None. Returns a generator; exceptions from func propagate on iteration.
    for x in xs:
        r = func(x)
        if r is not None:
            yield r


# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    # [FUNCTION DOC] Lazily applies func to each element of xs and yields every item of each
    # returned sequence, flattening one level. Returns a generator.
    for x in xs:
        yield from func(x)


# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that adds error context. If the managed block raises
    # any Exception, msg_fn is called (only then) to build a message, which is indented and
    # appended on a new line to the first exception argument (or becomes the only argument
    # if there were none); the same exception is then re-raised. Mutates the exception args.
    # KNOWN ISSUE: per the TODO, KeyError messages are repr-formatted so this misformats them.
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = msg_fn()
        msg = textwrap.indent(msg, "  ")
        msg = f"{e.args[0]}\n{msg}" if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise


@functools.cache
def _read_template(template_fn: str) -> CodeTemplate:
    # [FUNCTION DOC] HELPER: Loads a CodeTemplate from the given file path; results are
    # cached per path for the process lifetime, so file edits are not re-read.
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    # [FUNCTION DOC] Returns a deterministic integer for a string: the SHA-1 digest of its
    # latin1 encoding read as a little-endian integer. Stable across runs, unlike the builtin
    # hash. Raises UnicodeEncodeError for characters outside latin1.
    sha1 = hashlib.sha1(s.encode("latin1"), usedforsecurity=False).digest()
    return int.from_bytes(sha1, byteorder="little")


# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    # [CLASS DOC] Renders template files into an install directory and records which output
    # files were requested. State: install_dir and template_dir (Path), files (set of output
    # paths scheduled so far, each may be written only once) and dry_run (when true, paths
    # are recorded but nothing is rendered or written). Writes skip unchanged content so
    # timestamps are preserved. Key methods: write, write_with_template, write_sharded,
    # write_sharded_with_template, write_outputs.
    def __init__(
        self,
        install_dir: str | Path,
        template_dir: str | Path,
        dry_run: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the install and template directories as Path objects, starts
        # with an empty set of output files, and remembers the dry_run flag. No I/O.
        self.install_dir = Path(install_dir)
        self.template_dir = Path(template_dir)
        self.files: set[Path] = set()
        self.dry_run = dry_run

    @property
    def filenames(self) -> frozenset[str]:
        # [FUNCTION DOC] Property returning the recorded output files as an immutable set of
        # forward-slash path strings.
        return frozenset({file.as_posix() for file in self.files})

    def _write_if_changed(self, filename: str | Path, contents: str) -> None:
        # [FUNCTION DOC] HELPER: Writes contents (UTF-8) to filename only if the file is
        # missing, unreadable, or differs; creates parent directories as needed. Side effect:
        # file system write. Read errors (OSError) are treated as "no old contents".
        file = Path(filename)
        old_contents: str | None = None
        try:
            old_contents = file.read_text(encoding="utf-8")
        except OSError:
            pass
        if contents != old_contents:
            # Create output directory if it doesn't exist
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(contents, encoding="utf-8")

    # Read from template file and replace pattern with callable (type could be dict or str).
    def substitute_with_template(
        self,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> str:
        # [FUNCTION DOC] Produces the text for a template (path relative to template_dir).
        # Calls env_callable once; a string result is returned unchanged. A dict result gets
        # a default generated_comment entry (naming the running script relative to the repo
        # root, or just its name, plus the template) if absent, is substituted into the cached
        # template, and then blank lines are normalized between a closing triple quote and a
        # following class/def (with optional decorators). Raises AssertionError if
        # template_fn is absolute. Does not write files.
        if Path(template_fn).is_absolute():
            raise AssertionError(f"template_fn must be relative: {template_fn}")
        template_path = self.template_dir / template_fn
        env = env_callable()
        if isinstance(env, dict):
            if "generated_comment" not in env:
                generator_default = TORCHGEN_ROOT / "gen.py"
                try:
                    generator = Path(
                        sys.modules["__main__"].__file__ or generator_default
                    ).absolute()
                except (KeyError, AttributeError):
                    generator = generator_default.absolute()

                try:
                    generator_path = generator.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    generator_path = generator.name

                env = {
                    **env,  # copy the original dict instead of mutating it
                    "generated_comment": (
                        "@" + f"generated by {generator_path} from {template_fn}"
                    ),
                }
            template = _read_template(template_path)
            substitute_out = template.substitute(env)
            # Ensure an extra blank line between the class/function definition
            # and the docstring of the previous class/function definition.
            # NB: It is generally not recommended to have docstrings in pyi stub
            #     files. But if there are any, we need to ensure that the file
            #     is properly formatted.
            return re.sub(
                r'''
                (""")\n+             # match triple quotes
                (
                    (\s*@.+\n)*     # match decorators if any
                    \s*(class|def)  # match class/function definition
                )
                ''',
                r"\g<1>\n\n\g<2>",
                substitute_out,
                flags=re.VERBOSE,
            )
        if isinstance(env, str):
            return env
        assert_never(env)

    def write_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        # [FUNCTION DOC] Schedules one output file under install_dir, rendered from the given
        # template. Raises AssertionError for an absolute filename or if the same output was
        # already scheduled. Records the path; unless dry_run, renders the template and writes
        # it only if content changed.
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        file = self.install_dir / filename
        if file in self.files:
            raise AssertionError(f"duplicate file write {file}")
        self.files.add(file)
        if not self.dry_run:
            substitute_out = self.substitute_with_template(
                template_fn=template_fn,
                env_callable=env_callable,
            )
            self._write_if_changed(filename=file, contents=substitute_out)

    def write(
        self,
        filename: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        # [FUNCTION DOC] Convenience form of write_with_template where the template has the
        # same relative name as the output file.
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
        self,
        filename: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        # [FUNCTION DOC] Convenience form of write_sharded_with_template using the output
        # filename as the template name; all other arguments are forwarded unchanged.
        self.write_sharded_with_template(
            filename,
            filename,
            items,
            key_fn=key_fn,
            env_callable=env_callable,
            num_shards=num_shards,
            base_env=base_env,
            sharded_keys=sharded_keys,
        )

    def write_sharded_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        # [FUNCTION DOC] Splits items across num_shards output files plus one aggregate
        # "Everything" file, all rendered from one template. Each shard starts as an
        # environment holding shard_id (and a copy of base_env); every key in sharded_keys is
        # a list (copied from base_env or empty). Each item is routed to a shard by the stable
        # hash of key_fn(item), and the lists from env_callable(item) are appended to that
        # shard and to Everything. Files are named by inserting the shard id into the stem.
        # The Everything file is removed from the tracked file set (not meant to be compiled).
        # Raises AssertionError for an absolute filename, a non-list sharded base_env value,
        # or an env key not declared in sharded_keys. In dry_run mode items are ignored.
        file = Path(filename)
        if file.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        everything: dict[str, Any] = {"shard_id": "Everything"}
        shards: list[dict[str, Any]] = [
            {"shard_id": f"_{i}"} for i in range(num_shards)
        ]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    if not isinstance(shard[key], list):
                        raise AssertionError("sharded keys in base_env must be a list")
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []

        def merge_env(into: dict[str, list[str]], from_: dict[str, list[str]]) -> None:
            # [FUNCTION DOC] HELPER: Appends each list in from_ to the same-named list in
            # into, raising AssertionError for a key not in sharded_keys. Mutates into.
            for k, v in from_.items():
                if k not in sharded_keys:
                    raise AssertionError(f"undeclared sharded key {k}")
                into[k] += v

        if self.dry_run:
            # Dry runs don't write any templates, so incomplete environments are fine
            items = ()

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        for shard in all_shards:
            shard_id = shard["shard_id"]
            self.write_with_template(
                file.with_stem(f"{file.stem}{shard_id}"),
                template_fn,
                lambda: shard,
            )

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.files.discard(self.install_dir / file.with_stem(f"{file.stem}Everything"))

    def write_outputs(self, variable_name: str, filename: str | Path) -> None:
        # [FUNCTION DOC] Writes a CMake file defining a list variable named variable_name
        # holding every recorded output path (sorted, POSIX style, quoted), only if changed.
        # Side effect: file system write.
        """Write a file containing the list of all outputs which are generated by this script."""
        content = "\n".join(
            (
                "set(",
                variable_name,
                # Use POSIX paths to avoid invalid escape sequences on Windows
                *(f'    "{file.as_posix()}"' for file in sorted(self.files)),
                ")",
            )
        )
        self._write_if_changed(filename, content)

    def template_dir_for_comments(self) -> str:
        # [FUNCTION DOC] Returns the template directory as a path relative to this module's
        # directory, so generated comments do not embed build-specific absolute paths.
        """
        This needs to be deterministic. The template dir is an absolute path
        that varies across builds. So, just use the path relative to this file,
        which will point to the codegen source but will be stable.
        """
        return os.path.relpath(self.template_dir, os.path.dirname(__file__))


# Helper function to generate file manager
def make_file_manager(
    options: Namespace,
    install_dir: str | Path | None = None,
) -> FileManager:
    # [FUNCTION DOC] Builds a FileManager from parsed command line options: templates come
    # from the templates folder under options.source_path, output goes to install_dir if
    # given and non-empty else options.install_dir, and dry_run comes from options.dry_run.
    template_dir = os.path.join(options.source_path, "templates")
    install_dir = install_dir if install_dir else options.install_dir
    return FileManager(
        install_dir=install_dir,
        template_dir=template_dir,
        dry_run=options.dry_run,
    )


# Helper function to create a pretty representation for dataclasses
def dataclass_repr(
    obj: Any,
    indent: int = 0,
    width: int = 80,
) -> str:
    # [FUNCTION DOC] Returns a pretty-printed string for obj (typically a dataclass) by
    # delegating to the standard pformat with the given indent and width.
    return pformat(obj, indent, width)


def _format_dict(
    attr: dict[Any, Any],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    # [FUNCTION DOC] HELPER: Formats a mapping as brace-delimited "key: value" entries.
    # Dataclass values are pretty printed with pformat, others use repr. Delegates layout
    # (single line vs one entry per line) to _format with an indent increased by 3.
    curr_indent += indent + 3
    dict_repr = []
    for k, v in attr.items():
        k_repr = repr(k)
        v_str = (
            pformat(v, indent, width, curr_indent + len(k_repr))
            if is_dataclass(v)
            else repr(v)
        )
        dict_repr.append(f"{k_repr}: {v_str}")

    return _format(dict_repr, indent, width, curr_indent, "{", "}")


def _format_list(
    attr: list[Any] | set[Any] | tuple[Any, ...],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    # [FUNCTION DOC] HELPER: Formats a list, set or tuple of elements (dataclasses via
    # pformat, others via repr). Uses square brackets for lists and parentheses otherwise
    # (so a set is shown with parentheses); layout is delegated to _format.
    curr_indent += indent + 1
    list_repr = [
        pformat(l, indent, width, curr_indent) if is_dataclass(l) else repr(l)
        for l in attr
    ]
    start, end = ("[", "]") if isinstance(attr, list) else ("(", ")")
    return _format(list_repr, indent, width, curr_indent, start, end)


def _format(
    fields_str: list[str],
    indent: int,
    width: int,
    curr_indent: int,
    start: str,
    end: str,
) -> str:
    # [FUNCTION DOC] HELPER: Joins already formatted field strings between the start and end
    # delimiters. If the repr of the field list is at least width characters, fields are
    # separated by a newline plus curr_indent spaces; otherwise they stay on one line.
    # The indent spaces are placed right after the start delimiter.
    delimiter, curr_indent_str = "", ""
    # if it exceed the max width then we place one element per line
    if len(repr(fields_str)) >= width:
        delimiter = "\n"
        curr_indent_str = " " * curr_indent

    indent_str = " " * indent
    body = f", {delimiter}{curr_indent_str}".join(fields_str)
    return f"{start}{indent_str}{body}{end}"


class NamespaceHelper:
    # [CLASS DOC] Immutable-by-convention helper holding a C++ namespace path (colon-separated)
    # and an optional entity name, with precomputed text to open (prologue) and close
    # (epilogue) the nested namespaces. Rejects nesting deeper than max_level.
    """A helper for constructing the namespace open and close strings for a nested set of namespaces.

    e.g. for namespace_str torch::lazy,

    prologue:
    namespace torch {
    namespace lazy {

    epilogue:
    } // namespace lazy
    } // namespace torch
    """

    def __init__(
        self,
        namespace_str: str,
        entity_name: str = "",
        max_level: int = 2,
    ) -> None:
        # [FUNCTION DOC] Splits namespace_str on the C++ scope separator and stores the
        # pieces, the opening lines (one per namespace, outermost first) and the closing
        # lines (innermost first). Raises AssertionError if there are more than max_level
        # namespaces. KNOWN ISSUE: an empty namespace_str yields one empty namespace name,
        # producing a prologue of an unnamed namespace rather than no output.
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        if len(cpp_namespaces) > max_level:
            raise AssertionError(
                f"Codegen doesn't support more than {max_level} level(s) of "
                f"custom namespace. Got {namespace_str}."
            )
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str,
        max_level: int = 2,
    ) -> NamespaceHelper:
        # [FUNCTION DOC] Static factory: treats the last scope-separated piece of the input
        # as the entity name and the rest as the namespace, then builds a NamespaceHelper.
        # Raises AssertionError via the constructor if the namespace is too deep.
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

    @property
    def prologue(self) -> str:
        # [FUNCTION DOC] Property: text of the namespace-opening lines.
        return self.prologue_

    @property
    def epilogue(self) -> str:
        # [FUNCTION DOC] Property: text of the namespace-closing lines, innermost first.
        return self.epilogue_

    @property
    def entity_name(self) -> str:
        # [FUNCTION DOC] Property: the entity (class or function) name, empty if none given.
        return self.entity_name_

    # Only allow certain level of namespaces
    def get_cpp_namespace(self, default: str = "") -> str:
        # [FUNCTION DOC] Returns the full colon-joined namespace string, or default when the
        # namespace is empty.
        """
        Return the namespace string from joining all the namespaces by "::" (hence no leading "::").
        Return default if namespace string is empty.
        """
        return self.cpp_namespace_ if self.cpp_namespace_ else default


class OrderedSet(Generic[T]):
    # [CLASS DOC] Minimal insertion-ordered set. State: storage, a dict whose keys are the
    # members (values unused), so iteration follows first insertion order. Supports
    # membership, iteration, add, update, copy, union, | and |=, and equality with other
    # OrderedSets (order-insensitive) or plain sets. Not hashable and no removal or length.
    storage: dict[T, None]

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        # [FUNCTION DOC] Creates an empty set, or one holding the iterable's items with
        # duplicates removed, keeping first-seen order.
        if iterable is None:
            self.storage = {}
        else:
            self.storage = dict.fromkeys(iterable)

    def __contains__(self, item: T) -> bool:
        # [FUNCTION DOC] True if item is a member.
        return item in self.storage

    def __iter__(self) -> Iterator[T]:
        # [FUNCTION DOC] Iterates members in insertion order.
        return iter(self.storage.keys())

    def update(self, items: OrderedSet[T]) -> None:
        # [FUNCTION DOC] Adds all members of another OrderedSet in place, keeping existing
        # positions and appending new ones in the other set's order.
        self.storage.update(items.storage)

    def add(self, item: T) -> None:
        # [FUNCTION DOC] Adds item as a member; an existing member keeps its position.
        self.storage[item] = None

    def copy(self) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a new OrderedSet with a shallow copy of the storage.
        ret: OrderedSet[T] = OrderedSet()
        ret.storage = self.storage.copy()
        return ret

    @staticmethod
    def union(*args: OrderedSet[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Static: returns a new set with the members of all arguments, in
        # first-seen order; inputs are not mutated. Raises IndexError when called with no
        # arguments.
        ret = args[0].copy()
        for s in args[1:]:
            ret.update(s)
        return ret

    def __or__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Union operator: returns a new set of self followed by other.
        return OrderedSet.union(self, other)

    def __ior__(self, other: OrderedSet[T]) -> Self:
        # [FUNCTION DOC] In-place union operator: updates self with other and returns self.
        self.update(other)
        return self

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equality: with another OrderedSet compares the storage mappings
        # (order ignored); with anything else compares the member set to it, so equality with
        # a plain set works. KNOWN ISSUE: defining equality without a hash method makes
        # instances unhashable.
        if isinstance(other, OrderedSet):
            return self.storage == other.storage
        else:
            return set(self.storage.keys()) == other
