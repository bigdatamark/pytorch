# [MODULE DOC] Utilities for lowering an exported program to an AOTInductor-compiled delegate and
# packaging it for the native runtime. Provides flattening of program inputs/outputs, lowering into a delegate
# program that calls the executorch call-delegate higher-order op, and writing the pt2 archive.
import types

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import AOTI_FILES, package_pt2
from torch.types import FileLike

from ._lowered_aoti_module import LoweredBackendModule


def get_new_ep_with_flat_inputs_outputs(ep: ExportedProgram) -> ExportedProgram:
    # [FUNCTION DOC] Returns a new exported program whose inputs and outputs are flat. Wraps the original module so flat
    # inputs are unflattened with the input spec, the module is called, and outputs are flattened to a tuple;
    # then re-exports it using the flattened example inputs of the original.
    class FlattenedModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module adapter holding the original module and its input/output specs; its forward
        # accepts flat inputs and returns a flat tuple of outputs.
        def __init__(
            self,
            original_module: torch.fx.GraphModule,
            in_spec: pytree.TreeSpec,
            out_spec: pytree.TreeSpec,
        ) -> None:
            # [FUNCTION DOC] Stores the original module and the input and output tree specs.
            super().__init__()
            self.original_module = original_module
            self.in_spec = in_spec
            self.out_spec = out_spec

        def forward(self, *flat_inputs):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] Rebuilds args and kwargs from flat inputs via the input spec, calls the original module and returns the flattened outputs as a tuple.
            # Unflatten inputs to original structure
            inputs = pytree.tree_unflatten(flat_inputs, self.in_spec)
            args, kwargs = inputs
            outputs = self.original_module(*args, **kwargs)
            # Flatten outputs
            flat_outputs, _ = pytree.tree_flatten(outputs)
            return tuple(flat_outputs)

    flattened_module = FlattenedModule(
        ep.module(), ep.call_spec.in_spec, ep.call_spec.out_spec
    )
    args, kwargs = ep.example_inputs
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))
    flat_ep = torch.export.export(flattened_module, tuple(flat_inputs))

    return flat_ep


def lower_exported_program(
    exported_program: ExportedProgram, model_name: str, backend_id: str
) -> tuple[ExportedProgram, AOTI_FILES]:
    # [FUNCTION DOC] Compiles the flattened program ahead of time with AOTInductor in packaging mode, wraps it in a
    # LoweredBackendModule, patches its forward to flatten inputs, invoke the call-delegate op and restore the
    # original output structure, and re-exports it. Returns the delegate program and the list of AOTI files.
    # Raises an assertion error if the compile result is not a list. Side effects: runs compilation.
    """
    Lower an exported program to AOTInductor and return a delegate ExportedProgram
    with the `executorch_call_delegate` HOP
    """
    args, kwargs = exported_program.example_inputs
    out_spec = exported_program.call_spec.out_spec
    flat_ep = get_new_ep_with_flat_inputs_outputs(exported_program)
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))

    aoti_files = torch._inductor.aot_compile(
        flat_ep.module(), tuple(flat_inputs), options={"aot_inductor.package": True}
    )
    if not isinstance(aoti_files, list):
        raise AssertionError(
            f"aoti_files must be a list, got {type(aoti_files).__name__}"
        )

    lowered_aoti_module = LoweredBackendModule(
        flat_ep, backend_id, module_name=model_name
    )

    def patched_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Replacement forward bound to the lowered module: flattens inputs, calls the delegate op,
        # and unflattens outputs with the saved output spec when available.
        flat_inputs, _ = pytree.tree_flatten((args, kwargs))
        flat_outputs = torch._higher_order_ops.executorch_call_delegate(
            self, *flat_inputs
        )
        if out_spec is not None and flat_outputs is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        else:
            return flat_outputs

    lowered_aoti_module.forward = types.MethodType(patched_forward, lowered_aoti_module)  # type: ignore[method-assign]

    aoti_delegate_ep = torch.export.export(lowered_aoti_module, args, kwargs)

    return aoti_delegate_ep, aoti_files


def package_nativert_with_aoti_delegate(
    f: FileLike,
    model_name: str,
    backend_id: str,
    original_ep: ExportedProgram,
    delegate_ep: ExportedProgram,
    delegate_files: AOTI_FILES,
) -> None:
    # [FUNCTION DOC] Writes a pt2 archive to f containing the original program under model_name and the delegate
    # program under model_name plus backend id, along with the delegate AOTI files. Returns nothing.
    """
    Package a pt2 archive file that can be consumed by NativeRT with AOTI Delegate
    """
    package_pt2(
        f,
        exported_programs={
            model_name: original_ep,
            f"{model_name}-{backend_id}": delegate_ep,
        },
        aoti_files={f"{model_name}-{backend_id}": delegate_files},  # type: ignore[dict-item]
    )
    return
