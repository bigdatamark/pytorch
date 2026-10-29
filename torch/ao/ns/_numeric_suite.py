# mypy: allow-untyped-defs
# [MODULE DOC] Eager-mode Numeric Suite: utilities to compare weights and activations
# between a float model and its quantized counterpart. Provides weight comparison, logger
# modules (Logger, ShadowLogger, OutputLogger), the Shadow wrapper module, and functions to
# attach shadows/loggers and collect their stats. Defines NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST
# (module types that get loggers even though they are non-leaf). No import-time side effects.
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.quantization import prepare
from torch.ao.quantization.quantization_mappings import (
    get_default_compare_output_module_list,
)


NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}


def _find_match(
    str_list: dict[str, Any] | list[str],
    key_str: str,
    postfix: str,
) -> str | None:
    # [FUNCTION DOC] HELPER: Finds the entry of str_list that corresponds to key_str when
    # key_str ends with the given postfix. Compares the dot-joined prefix of key_str
    # (without the last component) against the prefix of each candidate with the last one
    # or two components removed; returns the first match. For the packed-params postfix
    # it retries with two components removed from key_str. Returns None if the postfix
    # does not match or no candidate matches.
    # KNOWN ISSUE: components are joined with an empty string, not a dot, so distinct names can collide.
    split_str = key_str.split(".")
    if split_str[-1] == postfix:
        match_string = "".join(key_str.split(".")[0:-1])
        for s2 in str_list:
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        if postfix == "_packed_params":
            match_string = "".join(key_str.split(".")[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None


def compare_weights(
    float_dict: dict[str, Any], quantized_dict: dict[str, Any]
) -> dict[str, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Pairs float and quantized weights from two state dicts. For each
    # quantized key, looks for a matching float weight (plain weight keys, or packed-params
    # keys where the first element of the packed tuple is the quantized weight, or LSTM
    # per-layer packed weights). Returns a mapping from quantized key to a mapping with
    # entries float and quantized. Logs API usage.
    # KNOWN ISSUE: in the LSTM branch the float and quantized entries are assigned twice,
    # so only the hidden-to-hidden weight pair is kept and the input-to-hidden pair is overwritten.
    # Also the packed-params branch does not skip the LSTM check after a match (no continue).
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]["float"],
                    wt_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_weights")
    weight_dict: dict[str, dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key][0]

        # For LSTM
        split_str = key.split(".")
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            layer = split_str[-2]
            module_name = ".".join(split_str[:-3])
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    return weight_dict


def _get_logger_dict_helper(
    mod: nn.Module,
    target_dict: dict[str, Any],
    prefix: str = "",
) -> None:
    # [FUNCTION DOC] HELPER: Recursively walks child modules; when a module has a direct
    # Logger child, stores that logger's stats in target_dict under the dotted module path
    # plus "stats". Mutates target_dict; returns nothing.
    r"""This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """

    def get_prefix(prefix):
        # [FUNCTION DOC] HELPER: Returns the prefix with a trailing dot, or empty if empty.
        return prefix if prefix == "" else prefix + "."

    for child in mod.children():
        if isinstance(child, Logger):
            target_dict[get_prefix(prefix) + "stats"] = child.stats
            break

    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)


def get_logger_dict(mod: nn.Module, prefix: str = "") -> dict[str, dict]:
    # [FUNCTION DOC] Collects the stats of all Logger modules found under mod into a new
    # mapping keyed by module path plus "stats". Logs API usage. Returns the mapping.
    r"""Traverse the modules and save all logger stats into target dict.
    This is mainly used for quantization accuracy debug.

    Type of loggers supported:
        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,
        OutputLogger: used to log the outputs of the modules

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module

    Return:
        target_dict: the dictionary used to save all logger stats

    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.get_logger_dict")

    target_dict: dict[str, dict] = {}
    _get_logger_dict_helper(mod, target_dict, prefix)
    return target_dict


class Logger(nn.Module):
    # [CLASS DOC] Base class for stats-logging modules. Holds a stats mapping and a dtype
    # (quint8) so it can be attached as an observer in FX mode. Its forward does nothing.
    r"""Base class for stats logging"""

    def __init__(self):
        # [FUNCTION DOC] Initializes an empty stats mapping and sets dtype to quint8.
        super().__init__()
        self.stats = {}
        # We only insert observer if the op is quantized with static quantization,
        # which is identified by activation_observer.dtype == quint8.  This is needed
        # when attaching Logger as observer for FX mode
        self.dtype = torch.quint8

    def forward(self, x):
        # [FUNCTION DOC] No-op placeholder; returns None and logs nothing.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on


class ShadowLogger(Logger):
    # [CLASS DOC] Logger used inside Shadow; stats has two lists, quantized and float,
    # that record the outputs of the original (quantized) and shadow (float) modules.
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        # [FUNCTION DOC] Creates empty float and quantized output lists in stats.
        super().__init__()
        self.stats["float"] = []
        self.stats["quantized"] = []

    def forward(self, x, y):  # type: ignore[override]
        # [FUNCTION DOC] Appends detached x to the quantized list and detached y to the float
        # list. If an argument has more than one element (a tuple/sequence), its first
        # element is used. Returns None.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats["quantized"].append(x.detach())
        self.stats["float"].append(y.detach())


class OutputLogger(Logger):
    # [CLASS DOC] Logger that records every tensor passing through it in the tensor_val
    # list of stats and passes the tensor on unchanged.
    r"""Class used to log the outputs of the module"""

    def __init__(self):
        # [FUNCTION DOC] Creates an empty tensor_val list in stats.
        super().__init__()
        self.stats["tensor_val"] = []

    def forward(self, x):
        # [FUNCTION DOC] Appends x to tensor_val and returns x unchanged.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        self.stats["tensor_val"].append(x)
        return x


def _convert_tuple_to_list(t: Any) -> Any:
    # [FUNCTION DOC] HELPER: Recursively converts tuples (exact tuple type) into lists; other values unchanged.
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t


def _dequantize_tensor_list(t: Any) -> Any:
    # [FUNCTION DOC] HELPER: Recursively dequantizes tensors inside (nested) lists; quantized
    # tensors are dequantized, other values returned as is.
    return (
        [_dequantize_tensor_list(x) for x in t]
        if type(t) is list
        else t.dequantize()
        if t.is_quantized
        else t
    )


class Shadow(nn.Module):
    # [CLASS DOC] Wraps a quantized module (orig_module) together with its float
    # counterpart (shadow_module) and a logger instance. Each call runs both on the same
    # input (dequantized for the float one), logs both outputs, and returns the quantized
    # output. Also mirrors the quantized functional-op methods (add, mul, cat, ...).
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    def __init__(self, q_module, float_module, logger_cls):
        # [FUNCTION DOC] Stores the quantized and float modules, a DeQuantize module, and a
        # new instance of logger_cls as the logger.
        super().__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = logger_cls()

    def forward(self, *x) -> torch.Tensor:
        # [FUNCTION DOC] Runs the quantized module on the inputs (tuples turned into lists),
        # runs the float module on the dequantized inputs, logs both outputs, and returns the
        # quantized module output.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        xl = _convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = _dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return output

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Quantized add on the original module; float add on dequantized inputs
        # on the shadow; logs both; returns the quantized result.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add(x, y)
        self.logger(output, shadow_output)
        return output

    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # [FUNCTION DOC] Same as add but with a scalar second operand; only x is dequantized.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.add_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Same as add but for multiplication of two tensors.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.mul(x, y)
        self.logger(output, shadow_output)
        return output

    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # [FUNCTION DOC] Same as add_scalar but for multiplication by a scalar.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.mul_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def cat(self, x: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        # [FUNCTION DOC] Quantized concatenation along dim on the original module; float
        # concatenation of the dequantized tensors on the shadow; logs both; returns the
        # quantized result.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.cat(x, dim)
        x = [y.dequantize() for y in x]
        shadow_output = self.shadow_module.cat(x, dim)
        self.logger(output, shadow_output)
        return output

    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Same as add but using the fused add-then-relu operation.
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_relu(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add_relu(x, y)
        self.logger(output, shadow_output)
        return output


def prepare_model_with_stubs(
    float_module: nn.Module,
    q_module: nn.Module,
    module_swap_list: set[type],
    logger_cls: Callable,
) -> None:
    # [FUNCTION DOC] Recursively walks the children of q_module in parallel with float_module
    # (matched by name). Children whose float type is not in module_swap_list are recursed
    # into; children whose float type is in the list and whose module-type structure differs
    # from the quantized one get replaced in q_module by a Shadow(quantized, float, logger_cls).
    # Mutates q_module in place after iteration; returns None. Logs API usage.
    r"""Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_with_stubs"
    )

    float_module_children = dict(float_module.named_children())

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)

        # Insert shadow module only if the module is not of the same type as
        # the floating point module
        if type(float_mod) in module_swap_list and not _is_identical_module_type(
            mod, float_mod
        ):
            reassign[name] = Shadow(mod, float_mod, logger_cls)

    for key, value in reassign.items():
        q_module._modules[key] = value


def _is_identical_module_type(mod1, mod2):
    # [FUNCTION DOC] HELPER: Returns True if both modules' submodule type lists (including themselves) are equal.
    # Compare if two modules have the same dtype
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types


def compare_model_stub(
    float_model: nn.Module,
    q_model: nn.Module,
    module_swap_list: set[type],
    *data,
    logger_cls=ShadowLogger,
) -> dict[str, dict]:
    # [FUNCTION DOC] Attaches shadow modules to q_model via prepare_model_with_stubs, runs
    # q_model on data, and returns the logger dict (module path -> stats with float and
    # quantized outputs). Mutates q_model; logs API usage.
    r"""Compare quantized module in a model with its floating point counterpart,
    feeding both of them the same input. Return a dict with key corresponding to
    module names and each entry being a dictionary with two keys 'float' and
    'quantized', containing the output tensors of quantized and its matching
    float shadow module. This dict can be used to compare and compute the module
    level quantization error.

    This function first call prepare_model_with_stubs() to swap the quantized
    module that we want to compare with the Shadow module, which takes quantized
    module, corresponding float module and logger as input, and creates a forward
    path inside to make the float module to shadow quantized module sharing the
    same input. The logger can be customizable, default logger is ShadowLogger
    and it will save the outputs of the quantized module and float module that
    can be used to compute the module level quantization error.

    Example usage::

        module_swap_list = [
            torchvision.models.quantization.resnet.QuantizableBasicBlock
        ]
        ob_dict = compare_model_stub(float_model, qmodel, module_swap_list, data)
        for key in ob_dict:
            print(
                key,
                compute_error(
                    ob_dict[key]["float"], ob_dict[key]["quantized"].dequantize()
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        module_swap_list: list of float module types at which shadow modules will
            be attached.
        data: input data used to run the prepared q_model
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_model_stub")
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict


def get_matching_activations(
    float_module: nn.Module,
    q_module: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Reads logger stats of both modules, skips quantized entries with no
    # recorded outputs, matches each remaining quantized logger key to a float logger key,
    # and returns a mapping from quantized key to a mapping with float and quantized recorded
    # tensor lists. Logs API usage.
    r"""Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.get_matching_activations"
    )
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: dict[str, dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]["tensor_val"]) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, "stats")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key]["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key]["tensor_val"]
    return act_dict


def prepare_model_outputs(
    float_module: nn.Module,
    q_module: nn.Module,
    logger_cls=OutputLogger,
    allow_list=None,
) -> None:
    # [FUNCTION DOC] Sets a debug qconfig whose activation observer is logger_cls on both
    # modules and runs the standard prepare in place (the quantized one also treats
    # selected non-leaf modules as leaves) so loggers get attached to modules in allow_list
    # (default: the default compare-output module list). Mutates both modules; logs API usage.
    r"""Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()

    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    float_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={}
    )
    q_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        q_module,
        inplace=True,
        allow_list=allow_list,
        observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
        prepare_custom_config_dict={},
    )


def compare_model_outputs(
    float_model: nn.Module,
    q_model: nn.Module,
    *data,
    logger_cls=OutputLogger,
    allow_list=None,
) -> dict[str, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Prepares both models with output loggers, runs each on data, and returns
    # the matching float and quantized activations per quantized module name. Mutates both
    # models; logs API usage.
    r"""Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    'float' and 'quantized', containing the activations of quantized model and
    float model at matching locations. This dict can be used to compare and
    compute the propagation quantization error.

    Example usage::

        act_compare_dict = compare_model_outputs(float_model, qmodel, data)
        for key in act_compare_dict:
            print(
                key,
                compute_error(
                    act_compare_dict[key]["float"],
                    act_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys 'float' and 'quantized',
        containing the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.compare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict
