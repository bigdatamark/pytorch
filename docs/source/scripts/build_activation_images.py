# [MODULE DOC] Script that renders an input/output line plot of each listed activation
# function over the range from -7 to 7 and saves it as a png in an activation_images
# directory next to this file. Running it has side effects at import time: switches
# matplotlib to the Agg backend, creates the output directory if missing, then plots
# and prints a message per function. Plots whose png already exists are skipped.
"""
This script will generate input-out plots for all of the activation
functions. These are for use in the documentation, and potentially in
online tutorials.
"""

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import torch


matplotlib.use("Agg")


# Create a directory for the images, if it doesn't exist
ACTIVATION_IMAGE_PATH = Path(__file__).parent / "activation_images"

if not ACTIVATION_IMAGE_PATH.exists():
    ACTIVATION_IMAGE_PATH.mkdir()

# In a refactor, these ought to go into their own module or entry
# points so we can generate this list programmatically
functions = [
    torch.nn.ELU(),
    torch.nn.Hardshrink(),
    torch.nn.Hardtanh(),
    torch.nn.Hardsigmoid(),
    torch.nn.Hardswish(),
    torch.nn.LeakyReLU(negative_slope=0.1),
    torch.nn.LogSigmoid(),
    torch.nn.PReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU6(),
    torch.nn.RReLU(),
    torch.nn.SELU(),
    torch.nn.SiLU(),
    torch.nn.Mish(),
    torch.nn.CELU(),
    torch.nn.GELU(),
    torch.nn.Sigmoid(),
    torch.nn.Softplus(),
    torch.nn.Softshrink(),
    torch.nn.Softsign(),
    torch.nn.Tanh(),
    torch.nn.Tanhshrink(),
    torch.nn.Threshold(0, 0.5),
    # Note: GLU is not included because it requires splitting the input tensor
    # into two halves (a, b) and computing a * sigmoid(b). A simple 1D input-output
    # plot doesn't meaningfully represent this behavior.
]


def plot_function(function, **args):
    # [FUNCTION DOC] Evaluates the given activation module on a grid of inputs from -7 to
    # 7 (step 0.01), detaches the result and draws it on the current matplotlib plot.
    # Extra keyword arguments (color, alpha and so on) are passed to the plot call.
    # Returns nothing.
    """
    Plot a function on the current plot. The additional arguments may
    be used to specify color, alpha, etc.
    """
    xrange = torch.arange(-7.0, 7.0, 0.01)  # We need to go beyond 6 for ReLU6
    x = xrange.numpy()
    y = function(xrange).detach().numpy()
    plt.plot(x, y, **args)


# Step through all the functions
for function in functions:
    function_name = function._get_name()
    plot_path = ACTIVATION_IMAGE_PATH / f"{function_name}.png"
    if not plot_path.exists():
        # Start a new plot
        plt.clf()
        plt.grid(color="k", alpha=0.2, linestyle="--")

        # Plot the current function
        plot_function(function)

        plt.title(function)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        # And save it
        plt.savefig(plot_path)
        print(f"Saved activation image for {function_name} at {plot_path}")
