# [PACKAGE DOC] Utility subpackage of torch.fx.passes for subgraph matching, fusion and lifting.
# Re-exports compare_graphs, HolderModule and lift_subgraph_as_module from the common module.
# Importing it also imports the matcher utilities that common depends on.
from .common import compare_graphs, HolderModule, lift_subgraph_as_module
