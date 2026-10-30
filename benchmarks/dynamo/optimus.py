# [MODULE DOC] Helpers returning torch.compile wrappers with the Inductor backend for benchmark
# harnesses: a plain baseline and an Optimus variant that first sets Inductor's pre-grad fusion
# pass options. No import-time side effects.
import functools

import torch


def get_baseline_ctx(nopython, inductor_compile_mode):
    # [FUNCTION DOC] Return a partially applied torch.compile preconfigured with the Inductor
    # backend, full graph mode equal to the nopython flag, and the given Inductor compile mode.
    # The caller applies it to a function or module. Does not change any config.
    return functools.partial(
        torch.compile,
        backend="inductor",
        fullgraph=nopython,
        mode=inductor_compile_mode,
    )


def get_optimus_optimize_ctx(config, nopython, inductor_compile_mode):
    # [FUNCTION DOC] Return a decorator-style function that compiles with Inductor after
    # installing a named set of pre-grad fusion options. The config name selects vertical_opt
    # (normalization, split/cat and unbind passes), horizontal_opt (normalization, batch linear,
    # batch layernorm) or all (union of both); any other name raises RuntimeError at once. The
    # nopython flag becomes full graph mode and the compile mode is forwarded.
    if config == "vertical_opt":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "merge_splits_pass": {},
                "split_cat_pass": {},
                "unbind_stack_pass": {},
                "unbind_cat_to_view_pass": {},
            }
        }
    elif config == "horizontal_opt":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "batch_linear": {},
                "batch_layernorm": {},
            },
        }
    elif config == "all":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "batch_linear": {},
                "batch_layernorm": {},
                "merge_splits_pass": {},
                "split_cat_pass": {},
                "unbind_stack_pass": {},
                "unbind_cat_to_view_pass": {},
            },
        }
    else:
        raise RuntimeError(f"Unknown optimus config: {config}")

    def _inner(fn):
        # [FUNCTION DOC] HELPER: Apply the chosen options to the global Inductor config (pre-
        # grad always; post-grad only if the preset has such options, which none does) and
        # return the Inductor-compiled version of fn. Mutates global config each time it is
        # called.
        # KNOWN ISSUE: the post-grad branch is dead code because no preset defines post-grad
        #     options, and the global config is changed as a side effect so later compiles in
        #     the process also see these options.
        if "pre_grad_fusion_options" in optimus_inductor_config:
            torch._inductor.config.pre_grad_fusion_options = optimus_inductor_config[
                "pre_grad_fusion_options"
            ]
        if "post_grad_fusion_options" in optimus_inductor_config:
            torch._inductor.config.post_grad_fusion_options = optimus_inductor_config[
                "post_grad_fusion_options"
            ]
        return torch.compile(
            fn, backend="inductor", fullgraph=nopython, mode=inductor_compile_mode
        )

    return _inner
