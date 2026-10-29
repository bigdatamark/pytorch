# [MODULE DOC] Defines LoweredBackendModule, a module placeholder representing a program lowered to a backend.
import torch
from torch.export import ExportedProgram


class LoweredBackendModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping the original exported program together with a backend id and optional
    # module name. Its forward delegates to the executorch call-delegate higher-order op passing itself.
    def __init__(
        self,
        original_exported_program: ExportedProgram,
        backend_id: str,
        *,
        module_name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the backend id, optional module name and the original exported program.
        super().__init__()
        self._backend_id = backend_id
        self._module_name = module_name
        self._original_exported_program = original_exported_program

    @property
    def backend_id(self) -> str:
        # [FUNCTION DOC] Property: the backend identifier string.
        return self._backend_id

    @property
    def module_name(self) -> str | None:
        # [FUNCTION DOC] Property: the optional module name, or None.
        return self._module_name

    @property
    def original_module(self) -> ExportedProgram:
        # [FUNCTION DOC] Property: the original exported program (despite the name, not a module).
        return self._original_exported_program

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Calls the call-delegate higher-order op with this module and the given arguments and returns its result.
        return torch._higher_order_ops.executorch_call_delegate(self, *args, **kwargs)
