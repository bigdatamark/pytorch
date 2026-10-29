# Owner(s): ["module: unknown"]

# [MODULE DOC]
# Test fixtures for sparsity/pruning tests. Contains a minimal BaseSparsifier subclass, a mock
# sparse Linear for convert tests, a row-subset checker, and a family of small nn.Module models
# (Linear, Conv2d, pooling, flatten, LSTM variants) whose layer patterns exercise the pruned
# structured fusion cases. No import-time side effects.

from typing import Any
from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn

class ImplementedSparsifier(BaseSparsifier):
    # [CLASS DOC] HELPER: Minimal concrete BaseSparsifier used by sparsifier tests. It adds no
    # state of its own; its update_mask zeroes a fixed mask row and counts calls in the
    # sparsifier state, so tests can verify step and bookkeeping behavior.
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Forwards the keyword arguments to the base class as its
        # defaults mapping. Returns nothing.
        super().__init__(defaults=kwargs)

    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Test update hook. Sets the first row of the mask of the first
        # parametrization on the module's weight to zero, then increments a step counter
        # (starting from zero) stored in the sparsifier state under the linear1 weight key.
        # Ignores tensor_name and extra kwargs. Mutates the mask and self.state.
        # Raises an error if the module has no weight parametrization with a mask.
        # KNOWN ISSUE: the state key is hardcoded to linear1 regardless of the module passed.
        module.parametrizations.weight[0].mask[0] = 0  # type: ignore[index, union-attr]
        linear_state = self.state['linear1.weight']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class MockSparseLinear(nn.Linear):
    # [CLASS DOC] HELPER: Stand-in sparse Linear for convert tests. Behaves exactly like a
    # Linear layer but is a distinct type, and offers a from_dense constructor so the convert
    # machinery can swap dense modules for it.
    """
    This class is a MockSparseLinear class to check convert functionality.
    It is the same as a normal Linear layer, except with a different type, as
    well as an additional from_dense method.
    """
    @classmethod
    def from_dense(cls, mod: nn.Linear) -> 'MockSparseLinear':
        # [FUNCTION DOC] HELPER: Builds a new instance with the same in_features and
        # out_features as the given Linear module. Weights are freshly initialized, not copied.
        """
        """
        linear = cls(mod.in_features,
                     mod.out_features)
        return linear


def rows_are_subset(subset_tensor: torch.Tensor, superset_tensor: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if every row of the first tensor appears in the second
    # tensor, with matching rows found in the same relative order (a single forward scan over
    # the superset; each superset row is consumed once). Rows are compared for exact equality.
    # Returns False as soon as a row cannot be found. No side effects.
    """
    Checks to see if all rows in subset tensor are present in the superset tensor
    """
    i = 0
    for row in subset_tensor:
        while i < len(superset_tensor):
            if not torch.equal(row, superset_tensor[i]):
                i += 1
            else:
                break
        else:
            return False
    return True


class SimpleLinear(nn.Module):
    # [CLASS DOC] HELPER: Test model of bias-free Linear layers: a Sequential of three (7->5,
    # 5->6, 6->4) followed by two more (4->4, 4->10). Used for Linear-Linear pruned fusion tests.
    r"""Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the Sequential seq and the layers linear1 and linear2
        # (all without bias). Takes no arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 4, bias=False),
        )
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs the input through seq, linear1, then linear2 in order and
        # returns the result (last dimension 10).
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LinearBias(nn.Module):
    # [CLASS DOC] HELPER: Test model that is a single Sequential of five Linear layers
    # (7->5->6->3->3->10) with biases on alternating layers. Used for Linear-Bias-Linear tests.
    r"""Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, a Sequential of five Linear layers where layers
        # 1, 3 and 4 have bias and layers 2 and 5 do not. No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 3, bias=True),
            nn.Linear(3, 3, bias=True),
            nn.Linear(3, 10, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq to the input and returns the result.
        x = self.seq(x)
        return x


class LinearActivation(nn.Module):
    # [CLASS DOC] HELPER: Test model of Linear layers with activation modules (ReLU, Tanh)
    # between them: a Sequential (7->5->6->4) then linear1 (4->3) and linear2 (3->10), each
    # followed by an activation module. Used for Linear(Bias)-Activation-Linear fusion tests.
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq (Linear/ReLU/Linear/Tanh/Linear), linear1, act1
        # (ReLU), linear2 and act2 (Tanh). No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.Tanh(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(3, 10, bias=False)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, then linear1 and act1, then linear2 and act2, and
        # returns the result.
        x = self.seq(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x


class LinearActivationFunctional(nn.Module):
    # [CLASS DOC] HELPER: Like LinearActivation, but the layers outside the Sequential are
    # separated by functional ReLU calls in forward rather than activation modules (except
    # act1, which is defined but unused). Used for Linear(Bias)-Activation-Linear fusion tests.
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activations are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq (Linear/ReLU/Linear/ReLU/Linear), linear1..linear3
        # and an act1 ReLU module. No arguments, returns nothing.
        # KNOWN ISSUE: act1 is created but never used in forward.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear3 = nn.Linear(8, 10, bias=False)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, then linear1, linear2 and linear3 with a
        # functional ReLU after each, and returns the result.
        x = self.seq(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x


class SimpleConv2d(nn.Module):
    # [CLASS DOC] HELPER: Test model of bias-free Conv2d layers: a Sequential of two (1->32,
    # 32->64) then conv2d1 (64->48) and conv2d2 (48->52). Used for Conv2d-Conv2d fusion tests.
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1 and conv2d2 (3x3 kernels, stride 1, no
        # bias). No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=False),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1 and conv2d2 in order and returns the
        # result (52 channels).
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dBias(nn.Module):
    # [CLASS DOC] HELPER: Test model of Conv2d layers with biases on some of them: a Sequential
    # of three (1->32, 32->32, 32->64) then conv2d1 (64->48) and conv2d2 (48->52).
    # Used for Conv2d-Bias-Conv2d fusion tests.
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1 and conv2d2 (3x3 kernels, stride 1);
        # biases are present on the first two Sequential layers and on conv2d1 only.
        # No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.Conv2d(32, 32, 3, 1, bias=True),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1 and conv2d2 in order and returns the
        # result (52 channels).
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dActivation(nn.Module):
    # [CLASS DOC] HELPER: Test model of Conv2d layers: a Sequential interleaving convolutions
    # with ReLU/Tanh/ReLU modules, followed by conv2d1 and conv2d2 separated and followed by
    # functional activations in forward. Used for Conv2d-Bias-Activation-Conv2d fusion tests.
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq (three Conv2d layers, each followed by an
        # activation module), conv2d1 (no bias) and conv2d2 (with bias). No arguments,
        # returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 1, bias=False),
            nn.ReLU(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1, functional ReLU, conv2d2, then
        # functional hardtanh, and returns the result.
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x


class Conv2dPadBias(nn.Module):
    # [CLASS DOC] HELPER: Test model of biased Conv2d layers, several with padding 1, with
    # activation modules after every layer (a Sequential of five convolutions, then conv2d1
    # and conv2d2 each followed by act1 and act2). Used to test bias propagation in pruned
    # Conv2d-Bias-(Activation)-Conv2d fusion when the following convolution has padding.
    r"""Model with only Conv2d layers, all with bias and some with padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1, act1 (ReLU), conv2d2 and act2 (Tanh).
        # One Sequential convolution has no bias; others have bias. No arguments, returns
        # nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, padding=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1, act1, conv2d2, act2 in order and
        # returns the result.
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPool(nn.Module):
    # [CLASS DOC] HELPER: Test model of biased Conv2d layers with pooling and activations in
    # between: a Sequential of two convolutions with MaxPool2d, ReLU, Tanh, AvgPool2d, then
    # three more convolutions with module and functional pooling/activations in forward.
    # Used for Conv2d-Pool2d-Conv2d fusion tests.
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1, a maxpool module, an af1 ReLU, conv2d2
        # and conv2d3 (3x3 kernels, padding 1, all with bias). No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, kernel_size=3, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(52, 52, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1, maxpool, af1, conv2d2, a functional
        # average pool (kernel 2, stride 2, padding 1), functional ReLU, then conv2d3, and
        # returns the result.
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.maxpool(x)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = F.relu(x)
        x = self.conv2d3(x)
        return x


class Conv2dPoolFlattenFunctional(nn.Module):
    # [CLASS DOC] HELPER: Test model of biased Conv2d layers with pooling and activations,
    # ending in an adaptive average pool to 1x1, a functional flatten and a Linear layer fc
    # (11->13). Used for Conv2d-Pool2d-Flatten-Linear fusion tests with functional flatten.
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1, af1 (ReLU), conv2d2, avg_pool (adaptive
        # average pool to 1x1) and fc (Linear 11->13). No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(11, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1, functional max pool, af1, conv2d2 and
        # avg_pool, flattens from dimension 1 with the functional flatten, applies fc and
        # returns the (batch, 13) result.
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # test functional flatten
        x = self.fc(x)
        return x


class Conv2dPoolFlatten(nn.Module):
    # [CLASS DOC] HELPER: Same as Conv2dPoolFlattenFunctional but pools to 2x2 and uses a
    # Flatten module before the Linear layer fc (44->13). Used for Conv2d-Pool2d-Flatten-Linear
    # fusion tests with a flatten module.
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a Flatten module followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq, conv2d1, af1, conv2d2, avg_pool (adaptive average
        # pool to 2x2), a flatten module and fc (Linear 44->13). No arguments, returns nothing.
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(44, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq, conv2d1, functional max pool, af1, conv2d2,
        # avg_pool, the flatten module and fc, and returns the (batch, 13) result.
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class LSTMLinearModel(nn.Module):
    # [CLASS DOC] HELPER: Test model with an LSTM (lstm) followed by a Linear decoder (linear)
    # mapping hidden_dim to output_dim. Used for sparsifying recurrent plus linear models.
    """Container module with an encoder, a recurrent module, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        # [FUNCTION DOC] HELPER: Builds an LSTM with the given input size, hidden size and
        # number of layers, and a Linear layer from hidden_dim to output_dim. Returns nothing.
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Runs the LSTM on the input (initial state defaulted), applies
        # the Linear layer to the LSTM output, and returns the pair (decoded, LSTM output).
        output, _hidden = self.lstm(input)
        decoded = self.linear(output)
        return decoded, output


class LSTMLayerNormLinearModel(nn.Module):
    # [CLASS DOC] HELPER: Test model with an LSTM, a LayerNorm over the hidden dimension and a
    # Linear layer (hidden_dim to output_dim), applied in that order.
    """Container module with an LSTM, a LayerNorm, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        # [FUNCTION DOC] HELPER: Builds lstm, norm (LayerNorm of size hidden_dim) and linear
        # (hidden_dim to output_dim). Returns nothing.
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Runs the LSTM, then the LayerNorm and the Linear layer on its
        # output, and returns the pair (result, LSTM state tuple).
        x, state = self.lstm(x)
        x = self.norm(x)
        x = self.linear(x)
        return x, state
