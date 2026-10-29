# [MODULE DOC]
# Attaches documentation strings to methods of the C-level torch.Size class.
# Import-time side effect: sets the docstring of Size.numel.
"""Adds docstrings to torch.Size functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


def add_docstr_all(method: str, docstr: str) -> None:
    # [FUNCTION DOC] HELPER: Sets the docstring of the named torch.Size method;
    # raises AttributeError if the method does not exist.
    add_docstr(getattr(torch._C.Size, method), docstr)


add_docstr_all(
    "numel",
    """
numel() -> int

Returns the number of elements a :class:`torch.Tensor` with the given size would contain.

More formally, for a tensor ``x = tensor.ones(10, 10)`` with size ``s = torch.Size([10, 10])``,
``x.numel() == x.size().numel() == s.numel() == 100`` holds true.

Example::

    >>> x=torch.ones(10, 10)
    >>> s=x.size()
    >>> s
    torch.Size([10, 10])
    >>> s.numel()
    100
    >>> x.numel() == s.numel()
    True


.. warning::

    This function does not return the number of dimensions described by :class:`torch.Size`, but instead the number
    of elements a :class:`torch.Tensor` with that size would contain.

""",
)
