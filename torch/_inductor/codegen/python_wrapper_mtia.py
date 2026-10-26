# [MODULE DOC] Python wrapper code generator variant for the MTIA backend.
# Defines PythonWrapperMtia, a thin PythonWrapperCodegen subclass that adds an MTIA
# dynamic-library import to generated wrapper headers. No import-time side effects.
from typing_extensions import override

from torch._inductor import ir

from .wrapper import PythonWrapperCodegen


class PythonWrapperMtia(PythonWrapperCodegen):
    # [CLASS DOC] PythonWrapperCodegen specialization for MTIA devices; overrides header emission
    # and the create factory. Holds no extra state.
    """
    A thin wrapper of PythonWrapperCodegen with MTIA specific logic
    """

    @override
    def write_header(self) -> None:
        # [FUNCTION DOC] Writes the standard wrapper header via the base class, then splices an extra
        # import of the MTIA host-runtime dynamic library into the imports buffer. Mutates self.imports.
        super().write_header()

        # MITA specific imports
        self.imports.splice("import mtia.host_runtime.torch_mtia.dynamic_library")

    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> PythonWrapperCodegen:
        # [FUNCTION DOC] Static factory. For subgraphs it delegates to the base class create with the same
        # arguments; otherwise returns a fresh PythonWrapperMtia. Args: is_subgraph, subgraph_name,
        # parent_wrapper, partition_signatures (optional). Returns a PythonWrapperCodegen instance.
        if is_subgraph:
            # Delegate to the parent class to handle the case of subgraph
            return PythonWrapperCodegen.create(
                is_subgraph, subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperMtia()
