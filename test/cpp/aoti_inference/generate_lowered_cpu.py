# [MODULE DOC] Command line tool that lowers an exported program for the AOTI CPU C++ tests.
# Loads an ExportedProgram, traces the original module, AOT-compiles it with inductor into a shared
# library, runs the reference output through the FX interpreter, and saves everything (script
# module, library path, inputs, outputs) as one scripted container file. Uses click for options.
import copy

import click

import torch


class Serializer(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose attributes are set from a mapping so the data can be
    # scripted and saved; each key of the data mapping becomes an attribute.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Sets each key of data as a module attribute holding its value.
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


@click.command()
@click.option(
    "--input-path",
    type=str,
    default="",
    required=True,
    help="path to the ExportedProgram",
)
@click.option(
    "--output-path",
    type=str,
    default="",
    required=True,
)
def main(
    input_path: str = "",
    output_path: str = "",
) -> None:
    # [FUNCTION DOC] Click command entry point. Args: input_path, file of the exported program to
    # load; output_path, where the scripted data container is written. Under no-grad, traces a copy
    # of the module, AOT-compiles the module via inductor (writing a shared library), computes
    # reference outputs (normalized to a list) and saves script_module, model_so_path, inputs and
    # outputs. Returns nothing; raises on load or compile failures.
    data = {}
    ep = torch.export.load(input_path)
    with torch.no_grad():
        example_inputs = ep.example_inputs[0]
        # Get scripted original module.
        module = torch.jit.trace(copy.deepcopy(ep.module()), example_inputs)

        # Get aot compiled module.
        so_path = torch._inductor.aot_compile(ep.module(), example_inputs)
        runner = torch.fx.Interpreter(ep.module())
        output = runner.run(example_inputs)
        if isinstance(output, (list, tuple)):
            output = list(output)
        else:
            output = [output]

        data.update(
            {
                "script_module": module,
                "model_so_path": so_path,
                "inputs": list(example_inputs),
                "outputs": output,
            }
        )

    torch.jit.script(Serializer(data)).save(output_path)


if __name__ == "__main__":
    main()
