# mypy: allow-untyped-defs
# [MODULE DOC] LinearAPoT: a reference quantized linear module whose weight is APoT quantized and whose matrix
# multiplication is emulated with bit shifts on uniformly quantized activations. Pure Python (slow, loop based); no import-time side effects.
import numpy as np
import numpy.typing as npt

import torch
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT


class LinearAPoT(WeightedQuantizedModule):
    # [CLASS DOC] Quantized linear layer for APoT. State: b (total bits), k (bits per block), n (b divided by k), the qparams
    # alpha, gamma, quantization_levels and level_indices computed by an APoTObserver on the weight, weight (integer APoT data
    # of the quantized weight) and weight_transposed. Key methods: decompose_APoT, bitshift_mul, matmul, forward. from_reference is not implemented.
    r"""
    A quantized linear module with quantized tensor as inputs and outputs
    to support APoT quantization.
    We adopt the same interface as `torch.nn.Linear`, see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        alpha: `alpha` qparam of output Quantized Tensor, type: Tensor
        gamma: `gamma` qparam of output Quantized Tensor, type: Tensor
        quantization_levels: `quantization_levels` qparam of output Quantized Tensor, type: Tensor
        level_indices: `level_indices` qparam of output Quantized Tensor, type: Tensor
        weight: APoT quantized tensor from weight2quantize
        weight_transposed: transposed weight tensor, used in linear transformation calculation (y = x * A^T + b)
    """

    def __init__(self, weight2quantize: torch.Tensor, b: int, k: int):
        # [FUNCTION DOC] Build the module from a float 2-D weight. Raises AssertionError if the weight is not 2-D or b is not divisible by k.
        # Observes the weight with an APoTObserver (b, k), stores the unsigned qparams, quantizes the weight with quantize_APoT,
        # and keeps the integer data plus its transpose. Side effect: runs the observer on the weight.
        if weight2quantize.dim() != 2:
            raise AssertionError(
                f"weight2quantize must be a 2-D tensor, got dim={weight2quantize.dim()}"
            )
        if b % k != 0:
            raise AssertionError(f"b must be divisible by k, got b={b}, k={k}")

        super().__init__()

        self.b = b
        self.k = k
        self.n = self.b // self.k

        observer = APoTObserver(b=self.b, k=self.k)

        observer(weight2quantize)

        (
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        ) = observer.calculate_qparams(signed=False)

        quantized_weight = quantize_APoT(
            weight2quantize,
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        )
        self.weight = quantized_weight.data
        self.weight_transposed = torch.transpose(self.weight, 0, 1)

    def decompose_APoT(self, x):
        # [FUNCTION DOC] Split a binary string (as produced by the builtin binary conversion) into a list of k-sized blocks:
        # the two character prefix is dropped and the remaining digits are cut left to right in chunks of k (last chunk may be shorter).
        # KNOWN ISSUE: a negative value's string has a minus sign before the prefix, which this does not handle.
        r"""
        Decompose binary representation of APoT values into list of k-sized blocks
        Args:
            x (Tensor): binary representation of APoT quantized tensor
        """
        # remove "0b" prefix from binary representation
        x = x[2:]

        # initialize list of blocks
        blocks = []

        while x:
            blocks.append(x[0 : self.k])
            x = x[self.k :]

        return blocks

    def bitshift_mul(self, weight_val, r):
        # [FUNCTION DOC] Multiply an integer activation r by an APoT weight given as a list of digit blocks, using shifts and additions.
        # Blocks are processed from last to first with digits reversed inside each block; a running place counter increases per digit
        # and r shifted left by the place is added for each digit equal to one. Returns the integer product.
        r"""
        Compute multiplication of weight_val * r using bitshifting
        method discussed in APoT paper: https://arxiv.org/pdf/1909.13144.pdf
        Args:
            weight_val: list of binary digits representing APoT quantized weight value
            r: int representing uniformly quantized activation value
        """
        product = 0

        idx = len(weight_val) - 1
        place = 0

        while idx >= 0:
            block = weight_val[idx]

            # reverse digits in block
            block = block[::-1]

            curr_block_result = 0

            for ele in block:
                if int(ele):
                    curr_block_result += r << place
                place += 1

            idx -= 1
            product += curr_block_result

        return product

    def matmul(self, decomposed_weight, activation):
        # [FUNCTION DOC] Matrix product of an activation matrix with a 2-D object array of decomposed weights using triple nested loops.
        # For each output cell it sums bitshift_mul over the inner dimension (activation entries are cast to int). Returns a float
        # zeros-initialised tensor of shape (activation rows, weight columns), filled in place.
        r"""
        Perform matrix multiplication between decomposed_weight and
        activation by calling bitshift_mul function for each value
        Args:
            decomposed_weight (Tensor): APoT quantized weight decomposed into binary
            activation (Tensor): uniformly quantized activation
        """
        rows1 = activation.size(dim=0)
        rows2 = decomposed_weight.shape[0]
        cols2 = decomposed_weight.shape[1]

        result = torch.zeros(rows1, cols2)

        # compute matrix multiplication with bitshifts
        for i in range(rows1):
            for j in range(cols2):
                for k in range(rows2):
                    weight_val = decomposed_weight[k][j]
                    r = int(activation[i][k])

                    product = self.bitshift_mul(weight_val, r)

                    result[i][j] += product

        return result

    def forward(self, activation: torch.Tensor) -> torch.FloatTensor:
        # [FUNCTION DOC] Compute the layer output. Raises AssertionError if the activation is not 2-D. Converts every entry of the transposed
        # weight to a binary string and decomposes it into blocks, then calls matmul and casts the result to float32. No bias is applied.
        r"""
        Multiply APoT quantized weight and uniformly quantized activation (dtype: quint8)
        with bitshifting instead of matrix multiplication.
        Result has dtype torch.float32
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        if activation.dim() != 2:
            raise AssertionError(
                f"activation must be a 2-D tensor, got dim={activation.dim()}"
            )

        weight_rows = self.weight_transposed.size()[0]
        weight_cols = self.weight_transposed.size()[1]

        decomposed_weight: npt.NDArray = np.empty(
            shape=(weight_rows, weight_cols), dtype=object
        )
        for row in range(weight_rows):
            for col in range(weight_cols):
                decomposed_weight[row][col] = self.decompose_APoT(
                    bin(self.weight_transposed[row][col])
                )

        result = self.matmul(decomposed_weight, activation).type(torch.FloatTensor)

        return result

    @classmethod
    def from_reference(  # type: ignore[override]
        cls,
        ref_qlinear,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor,
    ):
        # [FUNCTION DOC] Class method conversion from a reference quantized linear; not implemented and always raises NotImplementedError.
        raise NotImplementedError
