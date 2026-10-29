# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Utilities shared by the Lightning data sparsity callbacks: attach a model's parameters
# to a data sparsifier, make dot-free data names, and log per-layer sparsity. Creates the module logger.
import logging

from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import (
    SUPPORTED_TYPES,
)


logger: logging.Logger = logging.getLogger(__name__)


def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None):
    # [FUNCTION DOC] HELPER: Add every parameter of module whose exact type is a supported data type to the data
    # sparsifier under a dot-free name, using per-name overrides from config (default: none) as keyword config.
    # Mutates the sparsifier; returns nothing.
    """Attaches a data sparsifier to all the layers of the module.
    Essentially, loop over all the weight parameters in the module and
    attach it to the data sparsifier.
    Note::
        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)
        before attaching to the sparsifier. This is because, the data
        sparsifier uses a dummy model inside to store the weight parameters.
    """
    if config is None:
        config = {}
    for name, parameter in module.named_parameters():
        if type(parameter) in SUPPORTED_TYPES:
            valid_name = _get_valid_name(name)
            # will be defaulted to default configs
            data_sparsifier.add_data(
                name=valid_name, data=parameter, **config.get(valid_name, {})
            )


def _get_valid_name(name):
    # [FUNCTION DOC] HELPER: Return name with dots replaced by underscores (dots are not allowed in container names).
    return name.replace(".", "_")  # . is not allowed as a name


def _log_sparsified_level(model, data_sparsifier) -> None:
    # [FUNCTION DOC] HELPER: For each supported parameter of model, fetch its mask from the sparsifier and log
    # the sparsity (one minus the mean of the mask) at info level. Returns nothing.
    # KNOWN ISSUE: the log format string ends with an invalid percent conversion, so formatting the message
    # fails inside the logging machinery (an error report is printed) instead of showing the percentage.
    # Show the level of sparsity AFTER step:
    for name, parameter in model.named_parameters():
        if type(parameter) not in SUPPORTED_TYPES:
            continue
        valid_name = _get_valid_name(name)
        mask = data_sparsifier.get_mask(name=valid_name)
        sparsity_level = 1.0 - mask.float().mean()
        logger.info("Sparsity in layer %s = % .2%", name, sparsity_level)
