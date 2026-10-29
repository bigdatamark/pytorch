# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Shared utilities for quantized modules: the abstract base
# WeightedQuantizedModule (modules lowerable from reference modules), weight
# observer helpers, weight quantization with range clamping (_quantize_weight),
# an n-tuple parser (_ntuple_from_first, plus the module-level _pair_from_first
# built at import time with n=2) and a repr function that hides packed-params
# children.
import abc
import collections
import itertools

import torch
from torch.nn.modules.module import _addindent


__all__ = [
    "WeightedQuantizedModule",
]


class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    # [CLASS DOC] Abstract base class (ABC metaclass) for quantized modules that can
    # be lowered from reference modules; subclasses must implement the
    # from_reference classmethod.
    """Wrapper for quantized modules that can be lowered from reference modules."""

    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        # [FUNCTION DOC] Abstract classmethod: build a quantized module from a
        # reference module plus output scale and zero point. The base
        # implementation raises NotImplementedError.
        raise NotImplementedError


def _get_weight_observer(observer):
    # [FUNCTION DOC] HELPER: If the given observer is a fake-quantize wrapper
    # (has an activation_post_process attribute), returns that inner observer;
    # otherwise returns the observer itself.
    # FakeQuantize observer
    if hasattr(observer, "activation_post_process"):
        observer = observer.activation_post_process
    # UniformQuantizationObserverBase observer
    return observer


def _needs_weight_clamping(observer, dtype):
    # [FUNCTION DOC] HELPER: Returns True when dtype is qint8, quint8 or qint32
    # and the observer's quant_min/quant_max range is narrower than the full
    # integer range of the dtype (so weights need clamping); False otherwise.
    observer = _get_weight_observer(observer)
    if dtype in [torch.qint8, torch.quint8, torch.qint32]:
        info = torch.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False


def _clamp_weights(qweight, observer, scale, zp):
    # [FUNCTION DOC] HELPER: Clamps the integer representation of a quantized
    # weight to the observer's quant_min/quant_max and rebuilds a quantized
    # tensor with the given scale and zero point(s), per tensor or per channel
    # (using the observer's channel axis). Returns the weight unchanged if no
    # clamping is needed. Raises ValueError for an unexpected qscheme.
    # KNOWN ISSUE: the error message concatenates a string with the qscheme
    # object, which would raise a TypeError instead of the intended ValueError.
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight

    observer = _get_weight_observer(observer)
    min_, max_ = observer.quant_min, observer.quant_max

    # Doing this because can't use torch.ops.quantized.clamp() with per_channel qscheme yet.
    qw_int_max = torch.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = torch.clone(qweight.int_repr()).fill_(min_)
    qw_int = torch.minimum(torch.maximum(qweight.int_repr(), qw_int_min), qw_int_max)

    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch._make_per_tensor_quantized_tensor(
            qw_int, scale.item(), zp.item()
        )
    elif observer.qscheme in [
        torch.per_channel_symmetric,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        qweight = torch._make_per_channel_quantized_tensor(
            qw_int, scale, zp, axis=observer.ch_axis
        )
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _quantize_weight(float_wt, observer):
    # [FUNCTION DOC] HELPER: Quantizes a float weight using the observer's
    # calculated qparams and qscheme: per tensor, per channel (qint8) or per
    # channel with float qparams (observer dtype); then clamps the result to the
    # observer range. Returns the quantized weight. Raises ValueError for an
    # unexpected qscheme.
    # KNOWN ISSUE: the error message concatenates a string with the qscheme
    # object, which would raise a TypeError instead of the intended ValueError.
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.qint8
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double),
            wt_zp.to(torch.int64),
            wt_axis,
            torch.qint8,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme == torch.per_channel_affine_float_qparams:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float),
            wt_zp.to(torch.float),
            observer.ch_axis,
            observer.dtype,
        )
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


def _ntuple_from_first(n):
    # [FUNCTION DOC] HELPER: Returns a parser function that converts its
    # argument into a tuple of length n with the first element repeated.
    """Converts the argument to a tuple of size n
    with the first element repeated."""

    def parse(x):
        # [FUNCTION DOC] HELPER: Reduces x while it is a sequence whose length is not n
        # by taking its first element (stops early if the length equals n), then
        # returns a tuple of n repetitions of the resulting value.
        while isinstance(x, collections.abc.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(itertools.repeat(x, n))

    return parse


def _hide_packed_params_repr(self, params):
    # [FUNCTION DOC] HELPER: Builds a module repr string like the standard
    # module repr, but skips child modules that are instances of the given
    # params class. Returns the formatted string (name, extra repr lines and
    # child reprs indented).
    # We don't want to show `PackedParams` children, hence custom
    # `__repr__`. This is the same as nn.Module.__repr__, except the check
    # for the `params module`.
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in self._modules.items():
        if isinstance(module, params):
            continue
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = self._get_name() + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"
    return main_str


_pair_from_first = _ntuple_from_first(2)
