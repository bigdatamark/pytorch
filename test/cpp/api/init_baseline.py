# [MODULE DOC] Script that prints a C++ header of baseline weight values for initializers.
# For each initializer (Xavier and Kaiming, uniform and normal) it seeds the RNG, builds three
# linear layers (7->15, 15->15, 15->2), initializes their weights and prints C++ source defining
# functions in namespace expected_parameters that return the values as tensors. Output goes to
# stdout and progress to stderr; run as a script. No import-time side effects beyond definitions.
"""Script to generate baseline values from PyTorch initialization algorithms"""

import sys

import torch


HEADER = """
#include <torch/types.h>

#include <vector>

namespace expected_parameters {
"""

FOOTER = "} // namespace expected_parameters"

PARAMETERS = "inline std::vector<std::vector<torch::Tensor>> {}() {{"

INITIALIZERS = {
    "Xavier_Uniform": lambda w: torch.nn.init.xavier_uniform(w),
    "Xavier_Normal": lambda w: torch.nn.init.xavier_normal(w),
    "Kaiming_Normal": lambda w: torch.nn.init.kaiming_normal(w),
    "Kaiming_Uniform": lambda w: torch.nn.init.kaiming_uniform(w),
}


def emit(initializer_parameter_map):
    # [FUNCTION DOC] HELPER: Prints the C++ header text for a mapping from initializer name to a
    # list of weight arrays: a generated-file marker line, then one C++ function per initializer
    # returning nested vectors of tensors built from the flattened values. Writes to stdout only.
    # Don't write generated with an @ in front, else this file is recognized as generated.
    print("// @{} from {}".format("generated", __file__))
    print(HEADER)
    for initializer_name, weights in initializer_parameter_map.items():
        print(PARAMETERS.format(initializer_name))
        print("  return {")
        for sample in weights:
            print("    {")
            for parameter in sample:
                parameter_values = "{{{}}}".format(", ".join(map(str, parameter)))
                print(f"      torch::tensor({parameter_values}),")
            print("    },")
        print("  };")
        print("}\n")
    print(FOOTER)


def run(initializer):
    # [FUNCTION DOC] HELPER: Seeds the RNG with 0, creates three linear layers, applies the named
    # initializer (key in INITIALIZERS) to each weight and returns the three weights as numpy
    # arrays. Raises KeyError for an unknown name.
    torch.manual_seed(0)

    layer1 = torch.nn.Linear(7, 15)
    INITIALIZERS[initializer](layer1.weight)

    layer2 = torch.nn.Linear(15, 15)
    INITIALIZERS[initializer](layer2.weight)

    layer3 = torch.nn.Linear(15, 2)
    INITIALIZERS[initializer](layer3.weight)

    weight1 = layer1.weight.data.numpy()
    weight2 = layer2.weight.data.numpy()
    weight3 = layer3.weight.data.numpy()

    return [weight1, weight2, weight3]


def main():
    # [FUNCTION DOC] HELPER: Entry point. Runs every initializer, logging progress to stderr, then
    # emits the C++ baseline text to stdout.
    initializer_parameter_map = {}
    for initializer in INITIALIZERS:
        sys.stderr.write(f"Evaluating {initializer} ...\n")
        initializer_parameter_map[initializer] = run(initializer)

    emit(initializer_parameter_map)


if __name__ == "__main__":
    main()
