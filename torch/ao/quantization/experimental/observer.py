# mypy: allow-untyped-defs
# [MODULE DOC] Nonuniform (APoT) observer: APoTObserver tracks the running min/max of observed values and computes the
# Additive Powers-of-Two quantization parameters alpha, gamma, quantization levels and level indices. Can plot the levels (needs matplotlib).
# Bare string literals between methods are not docstrings.
"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import itertools

import torch
from torch.ao.quantization.experimental.apot_utils import apot_to_float, float_to_apot
from torch.ao.quantization.observer import ObserverBase


# TODO: Consider adding NonUniformQuantizationObserverBase class
# when more than one non-uniform method is implemented


class APoTObserver(ObserverBase):
    # [CLASS DOC] Observer for APoT quantization. Attributes: b (total bits), k (bits per additive term), n (terms count, set when
    # qparams are computed), min_val and max_val (empty tensors until values are observed). Main methods: forward (update running
    # min/max, pass-through), calculate_qparams, quant_levels_visualization.
    b: int
    k: int
    n: int
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, b, k, dtype=torch.quint8) -> None:
        # [FUNCTION DOC] Initialise with b, k and dtype (default quint8) through the observer base class; min_val and max_val start as empty tensors.
        super().__init__(dtype)
        self.b = b
        self.k = k

        self.min_val = torch.tensor([])
        self.max_val = torch.tensor([])

    # min_val and max_val are optional args to override
    # the min_val and max_val observed by forward
    def calculate_qparams(self, signed):  # type:ignore[override]
        # [FUNCTION DOC] Compute APoT qparams from the currently observed min_val and max_val. Arg signed selects signed levels.
        # Returns the tuple (alpha, gamma, quantization_levels, level_indices) from the internal helper.
        return self._calculate_qparams(signed, self.min_val, self.max_val)

    r""" Calculates nonuniform quantization parameters according to APoT paper:
    https://arxiv.org/pdf/1909.13144.pdf.
    Arg:
        signed: specifies whether to include signed values in quantization level calculations
        min_val: optional arg that can override min_val internal attribute
        max_val: optional arg that can override max_val internal attribute
    Returns:
        alpha: alpha quantization parameter, max of abs value of observed values
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels (fp representation)
        level_indices: int representation of quantization_levels indices
    """

    def _calculate_qparams(self, signed: bool, min_val=None, max_val=None):
        # [FUNCTION DOC] HELPER: Compute the APoT parameters; optional min_val and max_val override and overwrite the stored ones.
        # alpha is the larger of the negated min and the max. Raises AssertionError if k is zero/missing or b is not divisible by k. Sets n to b divided by k.
        # Builds n lists of base levels (zero plus powers of two 2^-(i + j*n), plus negatives when signed, sorted descending if signed);
        # gamma is alpha divided by the sum of each list's largest element (index 0 if signed, index 1 otherwise).
        # Takes the Cartesian product of the lists, sums each combination, scales by gamma and sorts.
        # Returns (alpha, gamma, sorted quantization_levels, the sort permutation as level_indices). Mutates min_val, max_val and n.
        # KNOWN ISSUE: level_indices is the argsort permutation of the levels, not an independent index encoding, and fails for empty observers.
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val

        # compute alpha
        alpha = torch.max(-self.min_val, self.max_val)

        # check for valid inputs of b, k
        if not self.k or self.k == 0:
            raise AssertionError(f"k must be a non-zero integer, got k={self.k}")
        if self.b % self.k != 0:
            raise AssertionError(
                f"b must be divisible by k, got b={self.b}, k={self.k}"
            )

        # compute n and store as member variable
        self.n = self.b // self.k

        # store a tensor of subtensors (all levels)
        p_all = []

        # create levels
        for i in range(self.n):
            p_curr = torch.tensor([0])

            for j in range((2**self.k - 2) + 1):
                curr_ele = 2 ** (-(i + j * self.n))
                p_append = torch.tensor([curr_ele])
                p_curr = torch.cat((p_curr, p_append))
                # introduce signed numbers
                if signed:
                    p_curr = torch.cat((p_curr, torch.tensor([-curr_ele])))

            if signed:
                # sort tensor in reverse order before adding to list if signed
                sorted, _indices = torch.sort(p_curr, descending=True)
                p_all.append(sorted)
            else:
                p_all.append(p_curr)

        # gamma calculation:
        # loop through all tensors
        # if signed, add element at index 0 for each tensor
        # else, add element at index 1 for each tensor
        # gamma defined to ensure alpha is at max of range
        p_sum = 0.0
        for tens in p_all:
            if signed:
                p_sum += float(tens[0])
            else:
                p_sum += float(tens[1])

        # assign gamma
        gamma = alpha / p_sum

        # calculate cartesian product
        cartesian_product = list(itertools.product(*p_all))

        quantization_levels_list = []

        # calculate sum of each row
        for row in cartesian_product:
            sum = 0.0
            for ele in row:
                sum += ele
            quantization_levels_list.append(sum)

        quantization_levels_gamma = [
            float(gamma) * ele for ele in quantization_levels_list
        ]
        quantization_levels = torch.tensor(quantization_levels_gamma)
        level_indices = torch.tensor([])
        quantization_levels, level_indices = quantization_levels.sort()

        return (alpha, gamma, quantization_levels, level_indices)

    r"""Records the running minimum and maximum of ``x``.
        Args:
            x_orig: Tensor to be observed for min and max val"""

    def forward(self, x_orig):
        # [FUNCTION DOC] Update running min and max with the extrema of x_orig (merged with existing values when present) and return x_orig unchanged.
        # Empty inputs are returned without updating. Mutates min_val and max_val.
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val, max_val = torch.aminmax(x)
        if self.min_val.numel():
            min_val = torch.min(min_val, self.min_val)
        if self.max_val.numel():
            max_val = torch.max(max_val, self.max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    r"""Displays visualization of APoT quantization levels
        Args:
            observer: APoTObserver to calculate qparams
            signed: bool to indicate if qparams should be signed/unsigned
    """

    def quant_levels_visualization(self, signed=False):
        # [FUNCTION DOC] Plot the quantized value for 1000 sample inputs in [0, 1) against the full-precision value using matplotlib.
        # Imports matplotlib lazily (ImportError if missing), computes qparams for the signed flag, and shows a figure (blocking side effect).
        # matplotlib is optional dep
        import matplotlib.pyplot as plt

        alpha, _gamma, quantization_levels, level_indices = self.calculate_qparams(
            signed
        )

        xs = [float(x) / 1000.0 for x in range(1000)]
        ys = [
            apot_to_float(
                float_to_apot(x, quantization_levels, level_indices, alpha),
                quantization_levels,
                level_indices,
            ).item()
            for x in xs
        ]

        plt.figure(figsize=(15, 10))

        plt.plot(xs, ys)
        plt.title("APoT Quantization Plot")
        plt.xlabel("Full Precision")
        plt.ylabel("Quantized")
        plt.show()
