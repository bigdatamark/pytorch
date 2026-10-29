# mypy: allow-untyped-defs
# [MODULE DOC] Builds, at import time, reStructuredText listing the torch.Tensor methods and
# properties that TorchScript cannot compile, and stores it as this module's __doc__ (used by
# the docs build). Import-time side effects: compiles one tiny TorchScript function per public
# Tensor attribute (slow) and runs exec on generated code.
from textwrap import dedent
from typing import Any

import torch.jit


def execWrapper(code, glob, loc) -> None:
    # [FUNCTION DOC] HELPER: Run the given source string with exec using the supplied global and
    # local namespaces; definitions land in loc. Returns nothing.
    exec(code, glob, loc)


def _gen_unsupported_methods_properties():
    # [FUNCTION DOC] HELPER: For every public Tensor attribute (names not starting with an
    # underscore, minus a hard-coded deprecated set), generate and try to compile a function
    # that calls it. If compilation fails with a message mentioning a nonexistent attribute,
    # classify the attribute as a method (its repr mentions a bound/built-in method) or a
    # property. Other failures are silently skipped. Returns a pair of strings: RST bullet lines
    # for methods and for properties. Side effects: exec of generated code, JIT compilation.
    # KNOWN ISSUE: the exec of the generated source is done before the try block, so an exec
    # failure would propagate; also attributes that compile successfully are simply dropped.
    tensor_attrs = set(filter(lambda x: x[0] != "_", dir(torch.Tensor)))
    tensor = torch.tensor([2])
    funcs_template = dedent(
        """
    def func(x):
        return x.{op}()
    """
    )

    deprecated_apis = {
        "volatile",
        "resize",
        "reinforce",
        "new",
        "name",
        "map2_",
        "grad_fn",
        "resize_as",
    }
    tensor_attrs = tensor_attrs - deprecated_apis

    properties = []
    methods = []
    sorted_tensor_attrs = sorted(tensor_attrs, key=lambda x: x.lower())
    for attr in sorted_tensor_attrs:
        funcs_str = funcs_template.format(op=attr)
        scope: dict[str, Any] = {}
        execWrapper(funcs_str, globals(), scope)
        try:
            torch.jit.CompilationUnit(funcs_str)
        except Exception as e:
            if "nonexistent attribute" not in repr(e):
                continue
            attr_repr = repr(getattr(tensor, attr))
            if "bound method" in attr_repr or "built-in method" in attr_repr:
                methods.append(attr)
            else:
                properties.append(attr)

    mapped_methods = ("\t*  :meth:`~torch.Tensor." + x + r"`" for x in methods)
    mapped_properties = ("\t*  :attr:`~torch.Tensor." + x + r"`" for x in properties)
    return "\n".join(mapped_methods), "\n".join(mapped_properties)


def _list_unsupported_tensor_ops():
    # [FUNCTION DOC] HELPER: Assemble the full RST document: a section header and bullet list of
    # unsupported Tensor methods followed by a header and list of unsupported properties.
    # Returns the string; used to set the module __doc__.
    header = """\n\n
Unsupported Tensor Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    methods, properties = _gen_unsupported_methods_properties()
    return (
        header
        + "\n"
        + methods
        + """

Unsupported Tensor Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
        + "\n"
        + properties
    )


__doc__ = _list_unsupported_tensor_ops()
