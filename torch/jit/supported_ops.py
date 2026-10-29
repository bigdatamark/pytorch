# mypy: allow-untyped-defs
# [MODULE DOC] Generates, for the Sphinx docs, a reStructuredText listing of the operators and
# functions TorchScript supports (tensor methods, torch.nn.functional and torch functions,
# TorchScript builtins, Python builtins, math module). Import-time side effect: builds the
# whole text by querying operator schemas and scripting nn.functional functions (slow), and
# assigns it to the module __doc__.
import inspect
import textwrap

import torch.jit
from torch.jit._builtins import _find_builtin


# this file is for generating documentation using sphinx autodoc
# > help(torch.jit.supported_ops) will also give a nice list of the
# supported ops programmatically


def _hidden(name):
    # [FUNCTION DOC] HELPER: True if the name starts with a single underscore (private) but not
    # a double underscore (dunder names are kept).
    return name.startswith("_") and not name.startswith("__")


def _emit_type(type):
    # [FUNCTION DOC] HELPER: Return the string form of a schema type.
    return str(type)


def _emit_arg(indent, i, arg):
    # [FUNCTION DOC] HELPER: Format one schema argument as "name : type", appending an equals sign
    # and the default value when one exists. Arguments after the first (i > 0) are prefixed by a
    # newline plus indent spaces for aligned multi-line output.
    v = f"{arg.name} : {_emit_type(arg.type)}"
    default = arg.default_value
    if default is not None:
        v = f"{v}={str(default)}"
    if i > 0:
        v = f"\n{' ' * indent}{v}"
    return v


def _emit_args(indent, arguments):
    # [FUNCTION DOC] HELPER: Format all arguments with _emit_arg and join them with commas.
    return ",".join(_emit_arg(indent, i, arg) for i, arg in enumerate(arguments))


def _emit_ret(ret):
    # [FUNCTION DOC] HELPER: Return the string form of a return value's type.
    return _emit_type(ret.type)


def _emit_rets(returns):
    # [FUNCTION DOC] HELPER: Format the return list: the single type if exactly one return,
    # otherwise a Tuple[...] string of the comma-joined types (also used for zero returns,
    # giving an empty Tuple).
    if len(returns) == 1:
        return _emit_ret(returns[0])
    return f"Tuple[{', '.join(_emit_ret(r) for r in returns)}]"


def _emit_schema(mod, name, schema, arg_start=0, padding=4):
    # [FUNCTION DOC] HELPER: Render an operator schema as a signature string "qualified_name(args)
    # -> rets". The qualified name is mod.name, or just name if mod is None. Arguments before
    # arg_start are skipped (used to drop self); continuation lines are indented to align under
    # the opening parenthesis plus padding.
    if mod is None:
        qualified_name = name
    else:
        qualified_name = f"{mod}.{name}"
    schema_str = (
        f"{qualified_name}"
        f"({_emit_args(len(qualified_name) + 1 + padding, schema.arguments[arg_start:])}) "
        f"-> {_emit_rets(schema.returns)}"
    )
    return schema_str


def _get_tensor_ops():
    # [FUNCTION DOC] HELPER: For each non-hidden attribute of torch.Tensor, look up the aten
    # schemas of the same name and keep those whose first argument is named self and is a Tensor
    # subtype. Returns the section title and a list of rendered signatures (self omitted).
    def is_tensor_method(schema) -> bool:
        # [FUNCTION DOC] HELPER: True if the schema has a first argument named self whose type is
        # a subtype of Tensor.
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != "self":
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True

    methods = []
    # discover methods
    for elem in dir(torch.Tensor):
        if not _hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    methods.append(_emit_schema("Tensor", elem, schema, arg_start=1))

    return "Supported Tensor Methods", methods


def _get_nn_functional_ops():
    # [FUNCTION DOC] HELPER: Collect signatures of supported PyTorch functions. First, every
    # function defined in torch.nn.functional is scripted and its schema rendered; functions that
    # fail to script are silently skipped (bare except). Second, for modules listed as containing
    # builtins, each attribute that maps to a builtin op has all its non-hidden schemas rendered.
    # Returns the section title and the list of signatures. Raises RuntimeError if a function's
    # module cannot be determined. Side effect: scripts many functions.
    # KNOWN ISSUE: the hidden check in the first loop is applied to only the first character of
    # the name, so it only filters names that are exactly one underscore.
    functions = []

    # Iterate over torch.nn.functional
    mod = torch.nn.functional
    name = mod.__name__
    for elem in dir(torch.nn.functional):
        attr = getattr(mod, elem)
        if not inspect.isfunction(attr) or _hidden(elem[0]):
            # Ignore non-functions and internal methods
            continue

        attr_module = inspect.getmodule(attr)
        if not attr_module:
            raise RuntimeError(f"Module for {attr} not found")

        if "torch.nn.functional" not in attr_module.__name__:
            # Ignore functions from outside torch.nn.functional
            continue

        try:
            # compile fn, get schema
            scripted = torch.jit.script(attr)
            scripted_schema = scripted.schema
            functions.append(_emit_schema(name, elem, scripted_schema))
        except:  # noqa: E722
            # Skip interpolate / boolean dispatched things
            pass

    # Iterate over modules that we know contain a lot of builtins
    for mod in torch.jit._builtins._modules_containing_builtins:
        name = mod.__name__
        for elem in dir(mod):
            builtin = _find_builtin(getattr(mod, elem))
            if builtin is not None:
                schemas = torch._C._jit_get_schemas_for_operator(builtin)
                for schema in schemas:
                    # remove _tan but not __and__
                    if not _hidden(elem):
                        functions.append(_emit_schema(name, elem, schema))
    return "Supported PyTorch Functions", functions


def _get_builtins_helper():
    # [FUNCTION DOC] HELPER: Filter the registered builtin ops table to a list of (function,
    # builtin name) pairs, skipping entries with no name (typing classes), no module, hidden
    # names, or modules under torch._C.
    builtins = []
    for fn, _builtin_name in torch.jit._builtins._builtin_ops:
        mod = inspect.getmodule(fn)

        if not hasattr(fn, "__name__"):
            # typing classes
            continue
        if not mod:
            continue
        if _hidden(fn.__name__) or _hidden(fn.__qualname__) or _hidden(mod.__name__):
            # skip internal-only methods
            continue

        if "torch._C" in mod.__name__:
            continue

        builtins.append((fn, _builtin_name))

    return builtins


def _is_math_fn(fn):
    # [FUNCTION DOC] HELPER: True if the function's defining module is the math module. Raises
    # RuntimeError if the module cannot be determined.
    mod = inspect.getmodule(fn)
    if not mod:
        raise RuntimeError(f"Module for {fn} not found")

    return mod.__name__ == "math"


def _get_torchscript_builtins():
    # [FUNCTION DOC] HELPER: Render the schemas of all specially registered builtins that are not
    # from the math module. Returns the section title and list of signatures. Raises RuntimeError
    # if a function's module is unknown.
    functions = []
    builtins = filter(lambda fn: not _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    # Iterate over the specially added builtins
    for fn, _builtin_name in builtins_list:
        mod = inspect.getmodule(fn)
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        builtin = _find_builtin(fn)
        if builtin is not None:
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            for schema in schemas:
                functions.append(_emit_schema(mod.__name__, fn.__name__, schema))

    return "TorchScript Builtin Functions", functions


def _get_math_builtins():
    # [FUNCTION DOC] HELPER: Like the TorchScript builtins collector but only for math module
    # functions, skipping any schema whose rendered text mentions Tensor. Returns the section
    # title and a list. Raises RuntimeError if a function's module is unknown.
    # KNOWN ISSUE: the list receives the raw schema objects instead of the rendered strings, so
    # the downstream formatting prints schema reprs rather than the emitted signatures.
    functions = []
    builtins = filter(lambda fn: _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    # Iterate over the specially added builtins
    for fn, _builtin_name in builtins_list:
        mod = inspect.getmodule(fn)
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        builtin = _find_builtin(fn)
        if builtin is not None:
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            for schema in schemas:
                schema_str = _emit_schema(mod.__name__, fn.__name__, schema)
                if "Tensor" in schema_str:
                    # Skip Tensor ops that have the same name as math functions
                    # (they will show up in the tensor methods section)
                    continue
                functions.append(schema)

    return "``math`` Module", functions


def _get_global_builtins():
    # [FUNCTION DOC] HELPER: Build the RST for Python builtin functions supported by TorchScript
    # (hard-coded list mirroring the globals map in the C++ ir_emitter). Ops with a schema are
    # rendered as signatures in a code block; ops without one become CSV rows with an
    # explanation; a table of builtin-to-magic-method rows is also included. Returns the section
    # title and the full RST string (not a list).
    # KNOWN ISSUE: a builtin with no schema and no entry in the explanations map would raise a
    # lookup error.
    # Taken from the 'globals' map in torch/csrc/jit/frontend/ir_emitter.cpp
    supported_builtins = [
        "print",
        "tuple",
        "float",
        "complex",
        "int",
        "bool",
        "str",
        "getattr",
        "hasattr",
        "isinstance",
        "len",
        "hex",
        "oct",
        "round",
        "hash",
        "min",
        "max",
        "abs",
        "all",
        "divmod",
        "list",
        "ord",
        "chr",
        "bin",
        "range",
        "zip",
        "enumerate",
        "sorted",
    ]

    op_renames = {
        "bool": "aten::Bool",
        "int": "aten::Int",
        "float": "aten::Float",
        "complex": "aten::Complex",
        "abs": "prim::abs",
        "max": "prim::max",
        "min": "prim::min",
        "range": "fake::does_not_exist",
    }

    schemaless_op_explanations = {
        "print": "Print any value",
        "tuple": "Lists cannot be converted to tuples with this method since their size is not statically known",
        "getattr": "Attribute name must be a literal string",
        "hasattr": "Attribute name must be a literal string",
        "isinstance": "Result is static",
        "zip": "Arguments must be iterable.",
        "enumerate": "Arguments must be iterable.",
        "range": "Can only be used as an iterator in a for loop",
    }

    magic_methods = [
        ("complex", "__complex__"),
        ("float", "__float__"),
        ("int", "__int__"),
        ("bool", "__bool__"),
        ("str", "__str__"),
        ("len", "__len__"),
        ("hex", "__hex__"),
        ("oct", "__oct__"),
    ]

    magic_methods_rows = []
    for fn, magic_method in magic_methods:
        # pyrefly: ignore [bad-argument-type]
        magic_methods_rows.append(f'"{fn}", "``{magic_method}``"')

    schematized_ops = []
    schemaless_ops = []

    for fn in supported_builtins:
        op_name = f"aten::{fn}"
        if fn in op_renames:
            op_name = op_renames[fn]
        schemas = torch._C._jit_get_schemas_for_operator(op_name)
        for s in schemas:
            schematized_ops.append(_emit_schema(None, fn, s, padding=0))
        if len(schemas) > 0:
            schematized_ops.append("")
        else:
            table_row = (
                f'":external+python:py:obj:`{fn}`", "{schemaless_op_explanations[fn]}"'
            )
            # pyrefly: ignore [bad-argument-type]
            schemaless_ops.append(table_row)

    schematized_ops_str = "\n".join(schematized_ops)
    schemaless_ops_str = "\n".join(schemaless_ops)
    magic_methods_rows_str = "\n".join(magic_methods_rows)
    schematized_ops_str = textwrap.indent(schematized_ops_str, "\t")
    schemaless_ops_str = textwrap.indent(schemaless_ops_str, "\t")
    magic_methods_rows_str = textwrap.indent(magic_methods_rows_str, "\t")
    section = f"""
The functions in the following table are supported but do not have a static schema

.. csv-table::
    :header: "Function", "Note"

{schemaless_ops_str}

The following functions will use the corresponding magic method on TorchScript classes

.. csv-table::
    :header: "Function", "Magic Method"

{magic_methods_rows_str}

These built-in functions use the schema

.. rst-class:: codeblock-height-limiter

::

{schematized_ops_str}
    """

    return "Python Built-in Functions", section


def _list_supported_ops():
    # [FUNCTION DOC] HELPER: Run each section collector in order, and concatenate the sections
    # into one RST string, each preceded by a link target derived from its title and followed by
    # a title underline and either the raw string body or a code block of the items. Returns it.
    def emit_block(decls):
        # [FUNCTION DOC] HELPER: Format a list of declarations as a literal code block, each
        # indented and separated by a blank line.
        return "\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{}\n".format(
            "".join(f"    {d}\n\n" for d in decls)
        )

    body = ""
    op_gathering_fns = (
        _get_tensor_ops,
        _get_nn_functional_ops,
        _get_torchscript_builtins,
        _get_global_builtins,
        _get_math_builtins,
    )
    for fn in op_gathering_fns:
        header, items = fn()
        link_target = header.replace("`", "").replace("-", "").lower().replace(" ", "-")
        if isinstance(items, str):
            section = f"{header}\n{'~' * len(header)}\n{items}\n"
        else:
            section = f"{header}\n{'~' * len(header)}\n{emit_block(items)}"
        section = f".. _{link_target}:" + "\n\n" + section
        body += section

    return body


__doc__ = _list_supported_ops()
