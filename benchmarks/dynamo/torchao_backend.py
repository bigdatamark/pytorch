# [MODULE DOC] Hooks for running dynamo benchmarks with torchao quantization: a baseline setup
# function and a factory that wraps a model iteration function to quantize the module first.
# torchao is imported lazily inside the functions.
from collections.abc import Callable
from typing import Any

import torch


def setup_baseline():
    # [FUNCTION DOC] Applies the recommended torchao inductor config, disables automatic dynamic
    # shapes, and raises the dynamo recompile limit to 10000. All effects are global config
    # mutations.
    from torchao.quantization.utils import recommended_inductor_config_setter

    recommended_inductor_config_setter()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 10000


def torchao_optimize_ctx(quantization: str):
    # [FUNCTION DOC] Returns a decorator-style function that wraps an iteration function so the
    # module is quantized in place on first use. The quantization argument selects int8 dynamic
    # activation with int8 weights, int8 weight-only, or int4 weight-only; an unrecognized name
    # applies no quantization but still marks the module as quantized.
    from torchao.quantization.quant_api import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    def inner(model_iter_fn: Callable):
        # [FUNCTION DOC] HELPER: Takes a model iteration function and returns the wrapper that
        # applies quantization and then delegates.
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            # [FUNCTION DOC] HELPER: If the module lacks a quantized marker, quantizes it in
            # place per the chosen scheme, unwraps tensor subclasses, and sets the marker. Then
            # calls the wrapped iteration function with the module and example inputs; its
            # return value is discarded.
            if getattr(module, "_quantized", None) is None:
                if quantization == "int8dynamic":
                    quantize_(
                        module,
                        int8_dynamic_activation_int8_weight(),
                        set_inductor_config=False,
                    )
                elif quantization == "int8weightonly":
                    quantize_(module, int8_weight_only(), set_inductor_config=False)
                elif quantization == "int4weightonly":
                    quantize_(module, int4_weight_only(), set_inductor_config=False)
                unwrap_tensor_subclass(module)
                setattr(module, "_quantized", True)  # noqa: B010
            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner
