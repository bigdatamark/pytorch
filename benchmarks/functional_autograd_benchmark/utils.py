# [MODULE DOC] Helpers for the functional autograd benchmark. Defines type aliases for the
# benchmark "getter" protocol (a getter maps a device to a callable plus its inputs) and for the
# timing-result mapping, utilities that turn an nn.Module into a pure function of its parameters
# (extract_weights / load_weights), markdown table (de)serialization of timing results, and a
# functorch availability probe. No import-time side effects besides imports.
from collections import defaultdict
from collections.abc import Callable

import torch
from torch import nn, Tensor


# Type helpers
InputsType = Tensor | tuple[Tensor, ...]
# A Getter takes in a device and returns a callable and the inputs to that callable
GetterReturnType = tuple[Callable[..., Tensor], InputsType]
GetterType = Callable[[torch.device], GetterReturnType]
# V here refers to the v in either vjp, jvp, vhp or hvp
VType = None | Tensor | tuple[Tensor, ...]
# Type used to store timing results. The first key is the model name, the second key
# is the task name, the result is a Tuple of: speedup, mean_before, var_before, mean_after, var_after.
TimingResultType = dict[str, dict[str, tuple[float, ...]]]


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Recursively delete a possibly dotted attribute from a module. The
    # names list is the attribute path; all but the last entry are followed with getattr and the
    # final entry is removed with delattr on the owning submodule. Raises AttributeError if any
    # name along the path is missing. Mutates obj in place and returns nothing.
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: list[str], value: Tensor) -> None:
    # [FUNCTION DOC] HELPER: Recursively set a possibly dotted attribute on a module. Follows
    # all but the last entry of names via getattr, then assigns value on the owning submodule
    # with setattr. Raises AttributeError if an intermediate attribute is missing. Mutates obj
    # in place.
    # KNOWN ISSUE: the docstring example calls the delete helper instead of this setter.
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod: nn.Module) -> tuple[tuple[Tensor, ...], list[str]]:
    # [FUNCTION DOC] Strip every parameter out of a module so it can be driven functionally.
    # Returns a pair: a tuple of detached tensors that require grad (one per original parameter,
    # in parameter iteration order) and the list of dotted parameter names in the order they
    # were deleted. Side effect: the module is mutated in place and its parameter list is empty
    # afterwards, so load_weights must be called before the next forward pass.
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, names: list[str], params: tuple[Tensor, ...]) -> None:
    # [FUNCTION DOC] Reinstall parameters on a module, in the order given by names. Each entry
    # of params is a plain tensor (possibly with autograd history) set as a regular attribute at
    # the dotted path, so it does not become an nn.Parameter. Pairs are consumed with zip, so
    # extra items in the longer sequence are silently ignored. Mutates mod in place,
    # returns nothing.
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# Utilities to read/write markdown table-like content.
def to_markdown_table(
    res: TimingResultType, header: tuple[str, ...] | None = None
) -> str:
    # [FUNCTION DOC] Render timing results as a markdown table string. res maps model name to a
    # mapping of task name to a tuple of numbers; each pair becomes one row consisting of the
    # model, the task, and the numbers. header defaults to model, task, mean, var. The header
    # row and a separator row are emitted first. Returns the text, one line per row, each
    # terminated by a newline. The number of columns is not checked against the row lengths.
    if header is None:
        header = ("model", "task", "mean", "var")
    out = ""

    def write_line(*args):
        # [FUNCTION DOC] HELPER: Append one table row to the enclosing out string, joining the
        # stringified arguments with a vertical bar separator and a trailing newline. Uses a
        # nonlocal binding to rebind the enclosing variable.
        nonlocal out
        out += f"| {' | '.join(str(a) for a in args)} |\n"

    # Make it a markdown table
    write_line(*header)
    write_line(*["--"] * len(header))
    for model, tasks in res.items():
        for task, line in tasks.items():
            write_line(*(model, task) + line)

    return out


def from_markdown_table(data: str) -> TimingResultType:
    # [FUNCTION DOC] Parse text produced by to_markdown_table back into a nested mapping of
    # model name to task name to a tuple of two floats (mean, var). The first two lines (header
    # and separator) are dropped; each remaining line is split on the bar character, empty
    # fields are discarded and the four remaining fields are stripped. Raises ValueError if a
    # row does not have exactly four fields or the numbers do not parse. Returns a defaultdict
    # of defaultdicts.
    # KNOWN ISSUE: the result type alias documents five-number tuples, but only two numbers are
    # produced, and any speedup column written by the writer would break parsing.
    out = data.strip().split("\n")
    out = out[2:]  # Ignore the header lines

    res: TimingResultType
    res = defaultdict(defaultdict)

    for line in out:
        model, task, mean, var = (f.strip() for f in line.strip().split("|") if f)
        res[model][task] = (float(mean), float(var))

    return res


def check_for_functorch():
    # [FUNCTION DOC] Report whether the functorch package can be imported. Returns True on
    # success and False if the import raises ImportError. The import itself is a side effect
    # only in that it loads the package.
    try:
        import functorch  # noqa: F401

        return True
    except ImportError:
        return False
