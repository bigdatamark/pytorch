# [MODULE DOC] Quantized LSTM module. Defines LSTM, a subclass of the
# quantizable (observed) LSTM that is produced by converting an observed LSTM
# during the quantization convert step. No import-time side effects beyond class
# definition.
from typing import Any

import torch


__all__ = [
    "LSTM",
]


class LSTM(torch.ao.nn.quantizable.LSTM):
    # [CLASS DOC] Quantized LSTM built from an observed quantizable LSTM by
    # from_observed; it reuses the observed module's layers (state: the
    # converted layers). from_float is intentionally unsupported because the
    # flow is float -> observed -> quantized.
    r"""A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """

    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM  # type: ignore[assignment]

    def _get_name(self) -> str:
        # [FUNCTION DOC] HELPER: returns the display name string QuantizedLSTM.
        return "QuantizedLSTM"

    @classmethod
    def from_float(cls, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Always raises NotImplementedError telling the caller
        # that a non-observed LSTM cannot be converted directly; arguments are
        # ignored.
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )

    @classmethod
    def from_observed(cls: type["LSTM"], other: torch.ao.nn.quantizable.LSTM) -> "LSTM":
        # [FUNCTION DOC] Converts an observed quantizable LSTM into a quantized
        # LSTM. Raises AssertionError if other is not an instance of the
        # float/observed module class. Runs the quantization convert on it (not
        # in place, qconfig removed) and then rebinds the result's class to this
        # class; returns it.
        if not isinstance(other, cls._FLOAT_MODULE):  # type: ignore[has-type]
            raise AssertionError(
                f"Expected module type {cls._FLOAT_MODULE}, got {type(other)}"
            )
        converted = torch.ao.quantization.convert(
            other, inplace=False, remove_qconfig=True
        )
        converted.__class__ = cls
        return converted
