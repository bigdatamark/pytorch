# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC] Entry points for converting a TorchScript model to an NNAPI-backed module.
# Provides the NNAPI compilation-preference constants, NnapiModule (runs a serialized
# model through a torchbind NNAPI Compilation, fixing memory formats), process_for_nnapi
# (serializes the model and code-generates a shape-computation module) and
# convert_to_nnapi wrapper convert_model_to_nnapi. No import-time side effects beyond imports.

from typing import Optional

import torch
from torch.backends._nnapi.serializer import _NnapiSerializer


ANEURALNETWORKS_PREFER_LOW_POWER = 0
ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1
ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2


class NnapiModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping a lazily created NNAPI Compilation. Holds the shape-compute module, the serialized
    # model tensor, weights, per-input/output memory format codes (0 contiguous, 1 channels-last, 2 scalar/vector),
    # compilation preference and the f32-to-f16 relaxation flag. comp is None until init runs (first forward call);
    # out_templates holds empty output tensors computed from the first inputs' shapes.
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    # _nnapi.Compilation is defined
    comp: Optional[torch.classes._nnapi.Compilation]  # type: ignore[name-defined]  # noqa: UP045
    weights: list[torch.Tensor]
    out_templates: list[torch.Tensor]

    def __init__(
        self,
        shape_compute_module: torch.nn.Module,
        ser_model: torch.Tensor,
        weights: list[torch.Tensor],
        inp_mem_fmts: list[int],
        out_mem_fmts: list[int],
        compilation_preference: int,
        relax_f32_to_f16: bool,
    ):
        super().__init__()
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: list[torch.Tensor]):
        # [FUNCTION DOC] Create the NNAPI compilation from example inputs. Runs the shape-compute module's prepare
        # (which patches the serialized model for concrete shapes and returns output templates), makes weights
        # contiguous, builds and initializes a Compilation, and stores it in comp. Raises AssertionError if
        # already initialized.
        if self.comp is not None:
            raise AssertionError("comp must be None before initialization")
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)  # type: ignore[operator]
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init2(
            self.ser_model,
            self.weights,
            self.compilation_preference,
            self.relax_f32_to_f16,
        )

        self.comp = comp

    def forward(self, args: list[torch.Tensor]) -> list[torch.Tensor]:
        # [FUNCTION DOC] Run the model on a list of input tensors, initializing lazily on first call. Inputs with
        # format 1 are permuted from NCHW to NHWC and made contiguous; format 0 inputs are made contiguous.
        # Outputs are allocated like the templates, filled by the compilation, and format 1 outputs are permuted back
        # to NCHW. Returns the list of outputs. Raises ValueError on an unknown format and AssertionError on length mismatches.
        if self.comp is None:
            self.init(args)
        comp = self.comp
        if comp is None:
            raise AssertionError("comp must not be None")
        outs = [torch.empty_like(out) for out in self.out_templates]

        if len(args) != len(self.inp_mem_fmts):
            raise AssertionError(
                f"args length {len(args)} != inp_mem_fmts length {len(self.inp_mem_fmts)}"
            )
        fixed_args = []
        for idx in range(len(args)):
            fmt = self.inp_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                raise ValueError("Invalid mem_fmt")
        comp.run(fixed_args, outs)
        if len(outs) != len(self.out_mem_fmts):
            raise AssertionError(
                f"outs length {len(outs)} != out_mem_fmts length {len(self.out_mem_fmts)}"
            )
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt in (0, 2):
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise ValueError("Invalid mem_fmt")
        return outs


def convert_model_to_nnapi(
    model,
    inputs,
    serializer=None,
    return_shapes=None,
    use_int16_for_qint16=False,
    compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    relax_f32_to_f16=False,
):
    # [FUNCTION DOC] Convert a model plus example inputs into a scripted NNAPI module. Serializes via
    # process_for_nnapi, wraps the result in NnapiModule and a scripted wrapper whose generated forward accepts
    # separate positional arguments (arg_N) and returns one tensor (negative retval_count) or a tuple.
    # Args: serializer and return_shapes are optional overrides; compilation_preference and relax_f32_to_f16
    # configure NNAPI. Returns the scripted wrapper module.
    (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    ) = process_for_nnapi(
        model, inputs, serializer, return_shapes, use_int16_for_qint16
    )

    nnapi_model = NnapiModule(
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        compilation_preference,
        relax_f32_to_f16,
    )

    class NnapiInterfaceWrapper(torch.nn.Module):
        # [CLASS DOC] HELPER: Wrapper adapting NnapiModule's list-in/list-out interface; its forward is generated afterwards via define.
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            # [FUNCTION DOC] HELPER: Store the wrapped NnapiModule as mod.
            super().__init__()
            self.mod = mod

    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    wrapper_model = torch.jit.script(wrapper_model_py)
    # TODO: Maybe make these names match the original.
    arg_list = ", ".join(f"arg_{idx}" for idx in range(len(inputs)))
    if retval_count < 0:
        ret_expr = "retvals[0]"
    else:
        ret_expr = "".join(f"retvals[{idx}], " for idx in range(retval_count))
    wrapper_model.define(
        f"def forward(self, {arg_list}):\n"
        f"    retvals = self.mod([{arg_list}])\n"
        f"    return {ret_expr}\n"
    )
    return wrapper_model


def process_for_nnapi(
    model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False
):
    # [FUNCTION DOC] Freeze the model, serialize it to NNAPI and generate a shape-compute module. A single tensor
    # input is wrapped in a list; a default _NnapiSerializer is created if none is given. Returns a 6-tuple:
    # shape compute module, serialized model as an int32 tensor, used weights, input and output memory formats,
    # and return value count. A fresh class is defined per call because define mutates the class.
    model = torch.jit.freeze(model)

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    serializer = serializer or _NnapiSerializer(
        config=None, use_int16_for_qint16=use_int16_for_qint16
    )
    (
        ser_model,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        shape_compute_lines,
        retval_count,
    ) = serializer.serialize_model(model, inputs, return_shapes)
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    # We have to create a new class here every time this function is called
    # because module.define adds a method to the *class*, not the instance.
    class ShapeComputeModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Empty module whose prepare method is code-generated from the serializer's shape computation lines.
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """

    shape_compute_module = torch.jit.script(ShapeComputeModule())
    real_shape_compute_lines = [
        "def prepare(self, ser_model: torch.Tensor, args: List[torch.Tensor]) -> List[torch.Tensor]:\n",
    ] + [f"    {line}\n" for line in shape_compute_lines]
    shape_compute_module.define("".join(real_shape_compute_lines))

    return (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    )
