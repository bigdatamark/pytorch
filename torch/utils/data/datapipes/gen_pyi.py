# mypy: allow-untyped-defs
# [MODULE DOC] Build-time script that generates the datapipe type stub from its template.
# It scans the iter and map datapipe source files with simple text parsing for classes
# decorated as functional datapipes, extracts constructor signatures and class docstrings,
# and writes method definitions into the template to produce the stub file. Also holds the
# per-kind configuration constants. Import-time side effect: if torchgen cannot be imported
# it temporarily puts the repository root on the module search path to import it. Exports
# nothing publicly. Running as a script calls main.
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing_extensions import deprecated


try:
    from torchgen.api.python import format_function_signature
    from torchgen.utils import FileManager
except ImportError:
    import sys

    REPO_ROOT = Path(__file__).absolute().parents[4]
    sys.path.insert(0, str(REPO_ROOT))

    from torchgen.api.python import format_function_signature
    from torchgen.utils import FileManager

    if len(sys.path) > 0 and sys.path[0] == str(REPO_ROOT):
        del sys.path[0]


__all__: list[str] = []  # not intended to expose any symbols


def __dir__() -> list[str]:
    # [FUNCTION DOC] HELPER: Module attribute listing hook; returns an empty list so the
    # public API test sees no public names.
    return []  # appease public API test


@deprecated(
    "`torch.utils.data.datapipes.gen_pyi.materialize_lines` is deprecated and will be removed in the future.",
    category=FutureWarning,
)
def materialize_lines(lines: list[str], indentation: int) -> str:
    # [FUNCTION DOC] Deprecated (emits FutureWarning). Joins lines with newlines followed by
    # indentation spaces, and also indents newlines embedded inside a line; the first line
    # is not indented. Returns the combined text.
    output = ""
    new_line_with_indent = "\n" + " " * indentation
    for i, line in enumerate(lines):
        if i != 0:
            output += new_line_with_indent
        output += line.replace("\n", new_line_with_indent)
    return output


@deprecated(
    "`torch.utils.data.datapipes.gen_pyi.gen_from_template` is deprecated and will be removed in the future.",
    category=FutureWarning,
)
def gen_from_template(
    dir: str,
    template_name: str,
    output_name: str,
    replacements: list[tuple[str, Any, int]],
) -> None:
    # [FUNCTION DOC] Deprecated (emits FutureWarning). Reads a template file in dir, and for
    # each (placeholder, lines, indentation) replaces the placeholder with the materialized
    # lines, rewriting the output file in dir with the cumulative content after each
    # replacement. Side effect: writes the output file.
    # KNOWN ISSUE: if replacements is empty the output file is never written.
    template_path = os.path.join(dir, template_name)
    output_path = os.path.join(dir, output_name)

    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    for placeholder, lines, indentation in replacements:
        with open(output_path, "w", encoding="utf-8") as f:
            content = content.replace(
                placeholder, materialize_lines(lines, indentation)
            )
            f.write(content)


def find_file_paths(dir_paths: list[str], files_to_exclude: set[str]) -> set[str]:
    # [FUNCTION DOC] Lists each given directory (not recursive) and returns the set of joined
    # paths of files whose names end in .py and are not in files_to_exclude. Reads the
    # filesystem; raises OSError if a directory cannot be listed.
    """
    When given a path to a directory, returns the paths to the relevant files within it.

    This function does NOT recursive traverse to subdirectories.
    """
    paths: set[str] = set()
    for dir_path in dir_paths:
        all_files = os.listdir(dir_path)
        python_files = {fname for fname in all_files if ".py" == fname[-3:]}
        filter_files = {
            fname for fname in python_files if fname not in files_to_exclude
        }
        paths.update({os.path.join(dir_path, fname) for fname in filter_files})
    return paths


def extract_method_name(line: str) -> str:
    # [FUNCTION DOC] Returns the quoted name inside a decorator line (text between the
    # opening and closing parenthesis plus quote, either double or single quotes). Raises
    # RuntimeError if neither quoting form is found.
    """Extract method name from decorator in the form of "@functional_datapipe({method_name})"."""
    if '("' in line:
        start_token, end_token = '("', '")'
    elif "('" in line:
        start_token, end_token = "('", "')"
    else:
        raise RuntimeError(
            f"Unable to find appropriate method name within line:\n{line}"
        )
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def extract_class_name(line: str) -> str:
    # [FUNCTION DOC] Returns the text between the class keyword and the first opening
    # parenthesis of a class definition line. No validation if either is missing.
    """Extract class name from class definition in the form of "class {CLASS_NAME}({Type}):"."""
    start_token = "class "
    end_token = "("
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def parse_datapipe_file(
    file_path: str,
) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]:
    # [FUNCTION DOC] Line-by-line text parser of one source file. For each functional
    # datapipe decorator it records the method name, the class name that follows, and the
    # constructor (or new-instance method) parameter text, cleaned by process_signature.
    # Lines inside triple-quoted blocks are collected as that method's docstring text.
    # Returns four items: method name to signature token list, method name to class name,
    # the set of method names defined through the new-instance method (needing a special
    # output type), and method name to docstring lines. Raises RuntimeError on unbalanced
    # parentheses; reads the file.
    # KNOWN ISSUE: docstring lines seen while no method is active are stored under the
    # empty name, and the parser relies on a triple-double-quote convention.
    """Given a path to file, parses the file and returns a dictionary of method names to function signatures."""
    method_to_signature, method_to_class_name, special_output_type = {}, {}, set()
    doc_string_dict = defaultdict(list)
    with open(file_path, encoding="utf-8") as f:
        open_paren_count = 0
        method_name, class_name, signature = "", "", ""
        skip = False
        for line in f:
            if line.count('"""') % 2 == 1:
                skip = not skip
            if skip or '"""' in line:  # Saving docstrings
                doc_string_dict[method_name].append(line)
                continue
            if "@functional_datapipe" in line:
                method_name = extract_method_name(line)
                doc_string_dict[method_name] = []
                continue
            if method_name and "class " in line:
                class_name = extract_class_name(line)
                continue
            if method_name and ("def __init__(" in line or "def __new__(" in line):
                if "def __new__(" in line:
                    special_output_type.add(method_name)
                open_paren_count += 1
                start = line.find("(") + len("(")
                line = line[start:]
            if open_paren_count > 0:
                open_paren_count += line.count("(")
                open_paren_count -= line.count(")")
                if open_paren_count == 0:
                    end = line.rfind(")")
                    signature += line[:end]
                    method_to_signature[method_name] = process_signature(signature)
                    method_to_class_name[method_name] = class_name
                    method_name, class_name, signature = "", "", ""
                elif open_paren_count < 0:
                    raise RuntimeError(
                        "open parenthesis count < 0. This shouldn't be possible."
                    )
                else:
                    signature += line.strip()
    return (
        method_to_signature,
        method_to_class_name,
        special_output_type,
        doc_string_dict,
    )


def parse_datapipe_files(
    file_paths: set[str],
) -> tuple[dict[str, list[str]], dict[str, str], set[str], dict[str, list[str]]]:
    # [FUNCTION DOC] Runs the single-file parser over every path and merges the four
    # results (signatures, class names, special-output method names, docstrings); later
    # files override earlier ones on name clashes. Returns the merged four items.
    methods_and_signatures = {}
    methods_and_class_names = {}
    methods_with_special_output_types = set()
    methods_and_doc_strings = {}
    for path in file_paths:
        (
            method_to_signature,
            method_to_class_name,
            methods_needing_special_output_types,
            doc_string_dict,
        ) = parse_datapipe_file(path)
        methods_and_signatures.update(method_to_signature)
        methods_and_class_names.update(method_to_class_name)
        methods_with_special_output_types.update(methods_needing_special_output_types)
        methods_and_doc_strings.update(doc_string_dict)
    return (
        methods_and_signatures,
        methods_and_class_names,
        methods_with_special_output_types,
        methods_and_doc_strings,
    )


def split_outside_bracket(line: str, delimiter: str = ",") -> list[str]:
    # [FUNCTION DOC] Splits text on the delimiter (default comma) except where the delimiter
    # appears inside square brackets. Brackets themselves are kept in the pieces. Returns
    # the list of pieces (a trailing delimiter yields a final empty piece).
    """Given a line of text, split it on comma unless the comma is within a bracket '[]'."""
    bracket_count = 0
    curr_token = ""
    res = []
    for char in line:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        elif char == delimiter and bracket_count == 0:
            res.append(curr_token)
            curr_token = ""
            continue
        curr_token += char
    res.append(curr_token)
    return res


def process_signature(line: str) -> list[str]:
    # [FUNCTION DOC] Splits a raw parameter text into tokens and cleans them: strips spaces,
    # renames cls to self, blanks the parameter right after self unless it starts with a star,
    # replaces defaults on callable-typed parameters with an ellipsis, and drops empty
    # tokens. Returns the token list.
    # KNOWN ISSUE: indexing the first character of an empty token (such as one produced by
    # a trailing comma right after self) would raise IndexError; cls detection uses the
    # unstripped token.
    """
    Clean up a given raw function signature.

    This includes removing the self-referential datapipe argument, default
    arguments of input functions, newlines, and spaces.
    """
    tokens: list[str] = split_outside_bracket(line)
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(" ")
        if token == "cls":
            tokens[i] = "self"
        elif i > 0 and ("self" == tokens[i - 1]) and (tokens[i][0] != "*"):
            # Remove the datapipe after 'self' or 'cls' unless it has '*'
            tokens[i] = ""
        elif "Callable =" in token:  # Remove default argument if it is a function
            head = token.rpartition("=")[0]
            tokens[i] = head.strip(" ") + " = ..."
    tokens = [t for t in tokens if t != ""]
    return tokens


def get_method_definitions(
    file_path: str | list[str],
    files_to_exclude: set[str],
    deprecated_files: set[str],
    default_output_type: str,
    method_to_special_output_type: dict[str, str],
    root: str = "",
) -> list[str]:
    # [FUNCTION DOC] Produces the stub text for every functional datapipe found in the given
    # directory(ies) under root (default: this file's directory). Finds and parses source
    # files excluding the excluded and deprecated sets, picks the special output type for
    # methods in that mapping (or those defined via the new-instance method) and the default
    # output type otherwise, formats each signature, appends the class docstring or an
    # ellipsis, prefixes a comment naming the class, and sorts the result by method name.
    # Returns a list of definition strings. Reads the filesystem.
    # KNOWN ISSUE: a method defined through the new-instance method but missing from
    # method_to_special_output_type causes a KeyError.
    """
    #.pyi generation for functional DataPipes Process.

    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    """
    if root == "":
        root = str(Path(__file__).parent.resolve())
    file_path = [file_path] if isinstance(file_path, str) else file_path
    file_path = [os.path.join(root, path) for path in file_path]
    file_paths = find_file_paths(
        file_path, files_to_exclude=files_to_exclude.union(deprecated_files)
    )
    (
        methods_and_signatures,
        methods_and_class_names,
        methods_w_special_output_types,
        methods_and_doc_strings,
    ) = parse_datapipe_files(file_paths)

    for fn_name in method_to_special_output_type:
        if fn_name not in methods_w_special_output_types:
            methods_w_special_output_types.add(fn_name)

    method_definitions = []
    for method_name, arguments in methods_and_signatures.items():
        class_name = methods_and_class_names[method_name]
        if method_name in methods_w_special_output_types:
            output_type = method_to_special_output_type[method_name]
        else:
            output_type = default_output_type
        doc_string = "".join(methods_and_doc_strings[method_name])
        if doc_string == "":
            doc_string = " ..."
        else:
            doc_string = "\n" + doc_string
        definition = format_function_signature(method_name, arguments, output_type)
        method_definitions.append(
            f"# Functional form of '{class_name}'\n"
            + definition.removesuffix("...").rstrip()  # remove "..."
            + doc_string,
        )
    method_definitions.sort(
        key=lambda s: s.split("\n")[1]
    )  # sorting based on method_name

    return method_definitions


# Defined outside of main() so they can be imported by TorchData
iterDP_file_path: str = "iter"
iterDP_files_to_exclude: set[str] = {"__init__.py", "utils.py"}
iterDP_deprecated_files: set[str] = set()
iterDP_method_to_special_output_type: dict[str, str] = {
    "demux": "list[IterDataPipe]",
    "fork": "list[IterDataPipe]",
}

mapDP_file_path: str = "map"
mapDP_files_to_exclude: set[str] = {"__init__.py", "utils.py"}
mapDP_deprecated_files: set[str] = set()
mapDP_method_to_special_output_type: dict[str, str] = {"shuffle": "IterDataPipe"}


def main() -> None:
    # [FUNCTION DOC] Generates method definitions for iterable and map datapipes using the
    # module configuration, then renders the stub template next to this file into the
    # stub file, with the two method lists substituted. Side effect: writes the stub file.
    """
    # Inject file into template datapipe.pyi.in.

    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    """
    iter_method_definitions = get_method_definitions(
        iterDP_file_path,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
    )

    map_method_definitions = get_method_definitions(
        mapDP_file_path,
        mapDP_files_to_exclude,
        mapDP_deprecated_files,
        "MapDataPipe",
        mapDP_method_to_special_output_type,
    )

    path = Path(__file__).absolute().parent
    fm = FileManager(install_dir=path, template_dir=path, dry_run=False)
    fm.write_with_template(
        "datapipe.pyi",
        "datapipe.pyi.in",
        lambda: {
            "IterDataPipeMethods": iter_method_definitions,
            "MapDataPipeMethods": map_method_definitions,
        },
    )


if __name__ == "__main__":
    main()
