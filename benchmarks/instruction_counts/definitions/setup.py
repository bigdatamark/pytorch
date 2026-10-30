"""Define some common setup blocks which benchmarks can reuse."""

# mypy: ignore-errors
# [MODULE DOC] Reusable setup snippets for the benchmark definitions: trivial all-ones tensors in
# two, three and four dimensions, and a small training setup with inputs and weights that require
# gradients, each given as Python and C++ code in a GroupedSetup. The Setup enum exposes them by
# name. Import-time work: the training setup is parsed from a side-by-side table.

import enum

from core.api import GroupedSetup
from core.utils import parse_stmts


_TRIVIAL_2D = GroupedSetup(r"x = torch.ones((4, 4))", r"auto x = torch::ones({4, 4});")


_TRIVIAL_3D = GroupedSetup(
    r"x = torch.ones((4, 4, 4))", r"auto x = torch::ones({4, 4, 4});"
)


_TRIVIAL_4D = GroupedSetup(
    r"x = torch.ones((4, 4, 4, 4))", r"auto x = torch::ones({4, 4, 4, 4});"
)


_TRAINING = GroupedSetup(
    *parse_stmts(
        r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # Inputs                                 | // Inputs
        x = torch.ones((1,))                     | auto x = torch::ones({1});
        y = torch.ones((1,))                     | auto y = torch::ones({1});
                                                 |
        # Weights                                | // Weights
        w0 = torch.ones(                         | auto w0 = torch::ones({1});
            (1,), requires_grad=True)            | w0.set_requires_grad(true);
        w1 = torch.ones(                         | auto w1 = torch::ones({1});
            (1,), requires_grad=True)            | w1.set_requires_grad(true);
        w2 = torch.ones(                         | auto w2 = torch::ones({2});
            (2,), requires_grad=True)            | w2.set_requires_grad(true);
    """
    )
)


class Setup(enum.Enum):
    # [CLASS DOC] Enum whose members wrap the predefined setup snippets (trivial tensors of 2, 3
    # and 4 dimensions and the training setup). Member values are GroupedSetup objects.
    TRIVIAL_2D = _TRIVIAL_2D
    TRIVIAL_3D = _TRIVIAL_3D
    TRIVIAL_4D = _TRIVIAL_4D
    TRAINING = _TRAINING
