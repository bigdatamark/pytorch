# [MODULE DOC] Debug helpers that dump the lazy IR graph of tensors as text, dot, or backend IR.
from collections.abc import Sequence

import torch
import torch._C._lazy


def render_ir_graph(tensors: Sequence[torch.Tensor]) -> str:
    # [FUNCTION DOC] Returns the LTC IR graph of the given tensors as dot-format text.
    """Return a text dump of the LTC IR graph in dot format for the tensors.
    The text can be processed by tools like dot to be rendered in pdf,png etc."""
    return torch._C._lazy._get_tensors_dot(list(tensors))


def dump_ir(tensors: Sequence[torch.Tensor], ir_format: str) -> str:
    # [FUNCTION DOC] Returns an IR dump of the tensors: ir_format text gives LTC IR, backend gives the active
    # backend IR. Raises RuntimeError for any other format.
    """Return a dump of the tensors in the specified format.
    Valid format are
    - text: for LTC IR
    - backend: for the active backend IR
    """
    if ir_format == "text":
        return torch._C._lazy._get_tensors_text(list(tensors))
    elif ir_format == "backend":
        return torch._C._lazy._get_tensors_backend(list(tensors))
    else:
        raise RuntimeError(f"Unrecognized IR format: {ir_format}")
