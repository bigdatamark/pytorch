# [PACKAGE DOC] torch.fx.passes: collection of reusable FX graph analysis and transformation passes
# (graph drawing, shape propagation, splitting, reinplacing, runtime asserts, pass manager, etc).
# Importing the package eagerly imports the listed submodules and the pass_manager module so they are
# reachable as attributes; it defines no functions or classes of its own and has no other side effects.
from . import (
    graph_drawer,
    graph_manipulation,
    net_min_base,
    operator_support,
    param_fetch,
    regional_inductor,
    reinplace,
    runtime_assert,
    shape_prop,
    split_module,
    split_utils,
    splitter_base,
    tools_common,
)
