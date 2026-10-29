# mypy: allow-untyped-defs
# [MODULE DOC] Post-training quantization entry points for TorchScript (JIT) models.
# Offers fuse_conv_bn_jit, prepare_jit / prepare_dynamic_jit (insert observers), convert_jit /
# convert_dynamic_jit (replace observers with quant/dequant), the one-call quantize_jit and
# quantize_dynamic_jit, plus script_qconfig helpers and private on-device PTQ variants.
# All heavy work is delegated to torch._C JIT passes. No import-time side effects.

import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module


__all__ = [
    "script_qconfig",
    "script_qconfig_dict",
    "fuse_conv_bn_jit",
    "prepare_jit",
    "prepare_dynamic_jit",
    "convert_jit",
    "convert_dynamic_jit",
    "quantize_jit",
    "quantize_dynamic_jit",
]


def _check_is_script_module(model):
    # [FUNCTION DOC] HELPER: Raise ValueError unless the model is a TorchScript ScriptModule.
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError("input must be a script module, got: " + str(type(model)))


def _check_forward_method(model):
    # [FUNCTION DOC] HELPER: Raise ValueError if the script module has no forward method.
    if not model._c._has_method("forward"):
        raise ValueError("input script module does not have forward method")


def script_qconfig(qconfig):
    # [FUNCTION DOC] Instantiate the activation and weight observer factories of a qconfig, script each
    # one, and return a new QConfig holding the underlying C++ modules (deep-copied later in prepare).
    r"""Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c,
    )


def script_qconfig_dict(qconfig_dict):
    # [FUNCTION DOC] HELPER: Map every entry of the qconfig dict to its scripted qconfig, keeping None for
    # falsy values. Returns a new dict.
    r"""Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}


def fuse_conv_bn_jit(model, inplace=False):
    # [FUNCTION DOC] Fold batch norm into preceding conv using the JIT pass; intended for eval-mode models.
    # Logs API usage. If inplace, reconstructs the given model from the result and returns it; otherwise
    # returns a new wrapped module.
    r"""Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.fuse_conv_bn_jit")
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def _prepare_jit(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    # [FUNCTION DOC] HELPER: Shared prepare step for static/dynamic quantization. Validates script module,
    # forward method and that qconfig_dict keys are strings (ValueError otherwise), scripts the qconfigs,
    # fuses conv-bn, then inserts observers on forward via the JIT pass. Returns the model (in place or new).
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict):
        raise ValueError("qconfig_dict should only contain names(str) as keys.")
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observers(
        model._c, "forward", scripted_qconfig_dict, inplace, quant_type
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def _prepare_ondevice_jit(
    model,
    qconfig_dict,
    method_name="forward",
    inplace=False,
    quant_type=QuantType.STATIC,
):
    # [FUNCTION DOC] HELPER: On-device PTQ prepare step. Validates script module and string keys (ValueError),
    # scripts qconfigs, inlines the named method's graph (mutating it), fuses conv-bn and inserts observer
    # methods for on-device PTQ. Returns the model (in place or a new wrapped module).
    _check_is_script_module(model)
    if not all(isinstance(x, str) for x in qconfig_dict):
        raise ValueError("qconfig_dict should only contain names(str) as keys.")
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    method_graph = model._c._get_method(method_name).graph
    torch._C._jit_pass_inline(method_graph)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observer_method_for_ondevice_ptq(
        model._c, method_name, scripted_qconfig_dict, inplace, quant_type
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def prepare_jit(model, qconfig_dict, inplace=False):
    # [FUNCTION DOC] Public static-quantization prepare: log usage and insert observers into a script model.
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)


def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    # [FUNCTION DOC] Public dynamic-quantization prepare: log usage and insert observers for dynamic quant.
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_dynamic_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)


def _prepare_ondevice_dynamic_jit(
    model, qconfig_dict, method_name="forward", inplace=False
):
    # [FUNCTION DOC] HELPER: On-device dynamic prepare; forwards to the on-device prepare with dynamic quant type.
    return _prepare_ondevice_jit(
        model, qconfig_dict, method_name, inplace, quant_type=QuantType.DYNAMIC
    )


def _convert_jit(
    model, inplace=False, debug=False, quant_type=QuantType.STATIC, preserved_attrs=None
):
    # [FUNCTION DOC] HELPER: Shared convert step. Validates script module, puts it in eval mode (mutating it),
    # inserts quant/dequant, and unless debug is set moves parameters to CPU (skipped when all are on XPU)
    # and finalizes with preserved_attrs (default empty). Reconstructs in place or wraps a new module, then
    # runs constant propagation and dead-code elimination on its graph. Returns the model.
    _check_is_script_module(model)
    model.eval()
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant(
        model_c, "forward", inplace, debug, quant_type
    )
    if not debug:
        is_xpu = all(p.device.type == "xpu" for p in model.parameters())
        if not is_xpu:
            # Moving model parameters to CPU since quantized operators
            # are only supported on CPU and XPU right now
            model.cpu()
        if preserved_attrs is None:
            preserved_attrs = []
        model_c = torch._C._jit_pass_quant_finalize(
            model_c, quant_type, preserved_attrs
        )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model


def _convert_ondevice_jit(
    model, method_name, inplace=False, debug=False, quant_type=QuantType.STATIC
):
    # [FUNCTION DOC] HELPER: On-device convert step. Requires a script module, dynamic quant type and a method
    # name not starting with observe_ (AssertionError otherwise). Inserts quant/dequant for the observe_ method
    # and finalizes into a quantize_ method. Returns the model (in place or new).
    _check_is_script_module(model)
    if quant_type != QuantType.DYNAMIC:
        raise AssertionError(
            "This API, while should work for static quant, is only tested for dynamic quant."
        )
    if method_name.startswith("observe_"):
        raise AssertionError("Pass in valid method to be quantized, e.g. forward")
    observe_method_name = "observe_" + method_name
    quantize_method_name = "quantize_" + method_name
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq(
        model._c, observe_method_name, inplace, debug, QuantType.DYNAMIC
    )
    model_c = torch._C._jit_pass_quant_finalize_for_ondevice_ptq(
        model_c, QuantType.DYNAMIC, quantize_method_name
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    # [FUNCTION DOC] Public static convert: log usage and convert an observed script model to quantized.
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_jit")
    return _convert_jit(
        model,
        inplace,
        debug,
        quant_type=QuantType.STATIC,
        preserved_attrs=preserved_attrs,
    )


def convert_dynamic_jit(model, inplace=False, debug=False, preserved_attrs=None):
    # [FUNCTION DOC] Public dynamic convert: log usage and convert with dynamic quant type.
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_dynamic_jit")
    return _convert_jit(
        model,
        inplace,
        debug,
        quant_type=QuantType.DYNAMIC,
        preserved_attrs=preserved_attrs,
    )


def _convert_ondevice_dynamic_jit(model, method_name, inplace=False, debug=False):
    # [FUNCTION DOC] HELPER: On-device dynamic convert; forwards with dynamic quant type.
    return _convert_ondevice_jit(
        model, method_name, inplace, debug, quant_type=QuantType.DYNAMIC
    )


def _quantize_ondevice_dynamic_jit_impl(
    model, qconfig_dict, method_name, inplace=False
):
    # [FUNCTION DOC] HELPER: Prepare then convert for on-device dynamic PTQ; returns the resulting model.
    model = _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name, inplace)
    model = _convert_ondevice_dynamic_jit(model, method_name, inplace)
    return model


def _quantize_jit(
    model,
    qconfig_dict,
    run_fn=None,
    run_args=None,
    inplace=False,
    debug=False,
    quant_type=QuantType.STATIC,
):
    # [FUNCTION DOC] HELPER: Shared prepare+convert pipeline. Dynamic: prepare then convert in place. Static:
    # requires run_fn and run_args (AssertionError otherwise), prepares, runs calibration run_fn(model, *run_args),
    # converts in place. Finally runs constant propagation and dead-code elimination. Returns the model.
    # Always do inplace convert because the Tensor is already
    # copied in prepare_jit when inplace is False
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        if not run_fn:
            raise AssertionError(
                "Must provide calibration function for post training static quantization"
            )
        if not run_args:
            raise AssertionError(
                "Must provide calibration dataset for post training static quantization"
            )
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)

    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model


def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    # [FUNCTION DOC] Post-training static quantization of a float script model: prepare, calibrate with
    # run_fn and run_args, convert. Returns the quantized script model; inplace and debug are passed through.
    r"""Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it's overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchScript model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_jit")
    return _quantize_jit(
        model,
        qconfig_dict,
        run_fn,
        run_args,
        inplace,
        debug,
        quant_type=QuantType.STATIC,
    )


def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    # [FUNCTION DOC] Post-training dynamic quantization of a float script model (no calibration needed).
    # Returns the quantized script model.
    # KNOWN ISSUE: the docstring example calls it with calibration arguments this function does not accept.
    r"""Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchScript model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_dynamic_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_dynamic_jit")
    return _quantize_jit(
        model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC
    )


def _quantize_ondevice_dynamic_jit(
    model, qconfig_dict, method_name="forward", inplace=False
):
    # [FUNCTION DOC] HELPER: Prepare a script model for on-device dynamic PTQ of the named method; returns the
    # model with observe_/quantize_ methods added. Delegates to the private implementation.
    # KNOWN ISSUE: docstring Args ordering mixes the method_name and qconfig_dict descriptions.
    r"""Prepares the input float TorchScript model with
    *on-device* post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        `method_name`: Name of the method within the model, to be prepared for quantization
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated

    Return:
        TorchScript model that is ready for on device quantization.
        This means that the returned
        model has:
        - Method is inlined.
        - Model has observer modules inserted in the model.
        - Model has packed params inserted in the model. However they are empty as in they don't
          contain valid quantized weights.
        - observe_<method_name> is added that observe the values to be quantized.
        - reset_observers_<method_name> to reset observers.
        - quantize_<method_name> is added to the model.
          - This method extract scale, zero points.
          - Quantizes observed weights.
          - Creates packed params from it and update the attribute of the model with the new values
            for the packed params.
          - Reset the original fp32 weights with empty tensor using SetAttr.
        - quantized_<method_name> is added to the model.
          - This method uses quantized weights and quantized linear ops instead of fp32 op.
          - This method should be used for inference post PTQ.
        - Note that all method's signatures should be the same as method_name.

        Later on device:
        - Run reset_observers_<method_name>
        - Run observe_<method_name>
        - Run quantize_<method_name>
        - Now model can be saved and loaded later.
        - Run model with quantized_<method_name>

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization.quantize_jit import _quantize_ondevice_dynamic_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")
    quant_ready_model = _quantize_ondevice_dynamic_jit(
        ts_model, {"": qconfig}, "forward", True
    )
    ```
    """
    return _quantize_ondevice_dynamic_jit_impl(
        model, qconfig_dict, method_name, inplace=inplace
    )
