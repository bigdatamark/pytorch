# [MODULE DOC] Debugging helper for FX: provides set_trace, which injects a pdb breakpoint into the
# generated Python code of a GraphModule. Public export list holds only set_trace. No import-time
# side effects.
from collections.abc import Sequence

import torch.fx as fx


__all__ = ["set_trace"]


def set_trace(gm: fx.GraphModule) -> fx.GraphModule:
    # [FUNCTION DOC] Insert a pdb breakpoint at the start of the generated forward code of the graph
    # module and recompile it so that running the module drops into the debugger. Chains with any
    # existing code transformer of the graph. Mutates the module (recompiled) and returns the same
    # module. The generated code is changed only while recompiling under the code-generation hook.
    """
    Sets a breakpoint in `gm`'s generated python code. It drops into pdb when
    `gm` gets run.

    Args:
        gm: graph module to insert breakpoint. It is then recompiled for it to
            take effect.

    Returns:
        the `gm` with breakpoint inserted.
    """

    def insert_pdb(body: Sequence[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Return a new list of code lines with a line that imports pdb and starts the
        # trace placed before the given body lines.
        return ["import pdb; pdb.set_trace()\n", *body]

    with gm.graph.on_generate_code(
        make_transformer=lambda cur_transform: (
            # new code transformer to register
            lambda body: (insert_pdb(cur_transform(body) if cur_transform else body))
        )
    ):
        gm.recompile()

    return gm
