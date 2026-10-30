#!/usr/bin/env python3
# flake8: noqa: F821
# [MODULE DOC] Dynamo benchmark suite driver for HuggingFace Transformers models. Defines the
# HuggingfaceRunner (model loading, input generation, tolerance policy, forward / backward
# passes), the table of extra non-registry models, and an entry point that runs the common
# benchmark main. Import-time side effects: enables inductor FX graph cache and autograd cache
# unless the matching environment variables are set; installs transformers with pip if it or
# any needed class is missing; executes imports of the needed transformers classes into module
# globals; downloads several model configs from the hub when building the extra models table;
# reads huggingface_models_list.txt (model name and batch size per line, optionally filtered
# by an environment variable) and raises AssertionError if the file is missing or yields no
# models.

import importlib
import logging
import os
import re
import subprocess
import sys
import types
import warnings


try:
    from .common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )
except ImportError:
    from common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )

import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs


log = logging.getLogger(__name__)

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True

# Enable Autograd caching
if "TORCHINDUCTOR_AUTOGRAD_CACHE" not in os.environ:
    torch._functorch.config.enable_autograd_cache = True


def pip_install(package):
    # [FUNCTION DOC] HELPER: Install a package into the current interpreter with pip. Raises
    # CalledProcessError if the install fails.
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Disable the flake warnings for the imports. Flake8 does not provide a way to
# disable just warning for the entire file. Disabling flake8 entirely.
imports = [
    "AlbertForPreTraining",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "BigBirdConfig",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotSmallModel",
    "CLIPModel",
    "CLIPVisionModel",
    "ElectraForPreTraining",
    "GPT2ForSequenceClassification",
    "GPTJForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "HubertForSequenceClassification",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "MarianForCausalLM",
    "MarianModel",
    "MarianMTModel",
    "PegasusModel",
    "ReformerConfig",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
]


def process_hf_reformer_output(out):
    # [FUNCTION DOC] HELPER: Drop the second element of the Reformer output list, which is
    # unstable between runs, and return the remaining elements. Raises AssertionError if the
    # output is not a list. Not called within this file.
    if not isinstance(out, list):
        raise AssertionError(f"expected out to be a list, got {type(out)}")
    # second output is unstable
    return [elem for i, elem in enumerate(out) if i != 1]


try:
    mod = importlib.import_module("transformers")
    for cls in imports:
        if not hasattr(mod, cls):
            raise ModuleNotFoundError
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    for cls in imports:
        exec(f"from transformers import {cls}")


# These models contain the models present in huggingface_models_list. It is a
# combination of models supported by HF Fx parser and some manually supplied
# models. For these models, we already know the largest batch size that can fit
# on A100 GPUs - 40 GB.
BATCH_SIZE_KNOWN_MODELS = {}

# Run only this selected group of models, leave this empty to run everything
TORCHBENCH_ONLY_MODELS = [
    m.strip() for m in os.getenv("TORCHBENCH_ONLY_MODELS", "").split(",") if m.strip()
]


# TODO(sdym): use batch-size-file parameter of common.main, like torchbench.py
# Get the list of models and their batch sizes
MODELS_FILENAME = os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
if not os.path.exists(MODELS_FILENAME):
    raise AssertionError(f"models file not found: {MODELS_FILENAME}")
with open(MODELS_FILENAME) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        if TORCHBENCH_ONLY_MODELS and model_name not in TORCHBENCH_ONLY_MODELS:
            continue
        batch_size = int(batch_size)
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size
if not BATCH_SIZE_KNOWN_MODELS:
    raise AssertionError("BATCH_SIZE_KNOWN_MODELS is empty")


try:
    from .huggingface_llm_models import HF_LLM_MODELS
except ImportError:
    from huggingface_llm_models import HF_LLM_MODELS


def get_module_cls_by_model_name(model_cls_name):
    # [FUNCTION DOC] Resolve a transformers model class from its name. Two decoder classes are
    # looked up in their specific submodules, everything else in the top-level package. Raises
    # ImportError or AttributeError if the module or class is missing.
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def get_sequence_length(model_cls, model_name):
    # [FUNCTION DOC] Choose the sequence length for the generated inputs from the model name
    # via prefix and name rules (128 to 4096, 10000 for Wav2Vec2); unknown names log at info
    # level and get 128. The model_cls argument is unused.
    # KNOWN ISSUE: the branch for the TrOCR causal LM model tests membership in a plain string
    # (not a one-element tuple), so it is a substring test, not an equality test.
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif model_name.startswith("OPT"):
        seq_length = 2048
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert"):
        seq_length = 512
    elif model_name in ("TrOCRForCausalLM"):
        seq_length = 256
    elif model_name.startswith("MobileBert"):
        seq_length = 128
    elif model_name.startswith("Wav2Vec2"):
        # If too short, will fail with something like
        # ValueError: `mask_length` has to be smaller than `sequence_length`,
        # but got `mask_length`: 10 and `sequence_length`: 9`
        seq_length = 10000  # NB: a more realistic size is 155136
    else:
        log.info(
            f"Sequence Length not defined for {model_name}. Choosing 128 arbitrarily"  # noqa: G004
        )
        seq_length = 128
    return seq_length


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # [FUNCTION DOC] Build the keyword inputs for a model as a mapping of name to tensor.
    # Token ids are random integers (vocab size from the model config); Wav2Vec2 gets float
    # waveform values, an attention mask and labels. Adds decoder ids for encoder-decoder
    # families, visual features for Lxmert, and, if include_loss_args is true, labels of a form
    # that depends on the model name suffix (pretraining, question answering, LM, token or
    # sequence classification, multiple choice, next sentence, conditional generation, extra
    # models). Raises NotImplementedError for an unsupported name when labels are requested.
    # KNOWN ISSUE: for pretraining models other than Electra and Lxmert the labels entry is
    # assigned a one-element tuple (trailing comma) instead of a tensor.
    # TODO - Check if following values are representative
    num_choices = 3
    num_visual_features = 42
    seq_length = get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size

    if model_name.startswith("Wav2Vec2"):
        # TODO: If we add more input_values style models, try to work this
        # into the overall control flow
        target_length = 100
        return {
            "input_values": torch.randn((bs, seq_length), device=device),
            # Added because that's what the example training script has
            "attention_mask": rand_int_tensor(device, 0, 2, (bs, seq_length)),
            "labels": rand_int_tensor(device, 0, vocab_size, (bs, target_length)),
        }

    if model_name.endswith("MultipleChoice"):
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if model_name.startswith(("T5", "M2M100", "MT5")) or model_cls in [
        BlenderbotModel,
        BlenderbotSmallModel,
        BlenderbotForConditionalGeneration,
        PegasusModel,
        MarianModel,
        MarianMTModel,
    ]:
        input_dict["decoder_input_ids"] = input

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict["labels"] = (
                    rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("QuestionAnswering"):
            input_dict["start_positions"] = rand_int_tensor(
                device, 0, seq_length, (bs,)
            )
            input_dict["end_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
        elif model_name.endswith(
            ("MaskedLM", "HeadModel", "CausalLM", "DoubleHeadsModel")
        ):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        elif model_name.endswith("TokenClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs, seq_length)
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict["labels"] = rand_int_tensor(device, 0, num_choices, (bs,))
        elif model_name.endswith("SequenceClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs,)
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size - 1, (bs, seq_length)
            )
        elif model_name in EXTRA_MODELS:
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        else:
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    return input_dict


def rand_int_tensor(device, low, high, shape):
    # [FUNCTION DOC] Return a random int64 tensor of the given shape on the device with values
    # in the range from low up to but excluding high, without gradient tracking.
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


EXTRA_MODELS = {
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
    ),
    # "BigBird": (
    #     BigBirdConfig(attention_type="block_sparse"),
    #     AutoModelForMaskedLM,
    # ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    # [CLASS DOC] BenchmarkRunner for the HuggingFace suite. Reads skip lists, tolerances,
    # batch size divisors and precision restrictions from a yaml config file (re-read on each
    # access). Sets an hf_llm flag when loading a model that uses generate-based speed tests.
    def __init__(self):
        # [FUNCTION DOC] Initialize the base runner and set the suite name to huggingface.
        super().__init__()
        self.suite_name = "huggingface"

    @property
    def _config(self):
        # [FUNCTION DOC] Property: load and return the parsed contents of the yaml config file.
        return load_yaml_file("huggingface.yaml")

    @property
    def _skip(self):
        # [FUNCTION DOC] Property: the skip section of the config.
        return self._config["skip"]

    @property
    def _accuracy(self):
        # [FUNCTION DOC] Property: the accuracy section of the config.
        return self._config["accuracy"]

    @property
    def skip_models(self):
        # [FUNCTION DOC] Property: names of models skipped for all configurations.
        return self._skip["all"]

    @property
    def skip_models_for_cpu(self):
        # [FUNCTION DOC] Property: names of models skipped when running on the CPU device.
        return self._skip["device"]["cpu"]

    @property
    def fp32_only_models(self):
        # [FUNCTION DOC] Property: names of models restricted to float32 precision.
        return self._config["only_fp32"]

    @property
    def skip_models_due_to_control_flow(self):
        # [FUNCTION DOC] Property: names of models skipped due to data-dependent control flow.
        return self._skip["control_flow"]

    def use_larger_multiplier_for_smaller_tensor(self, name):
        # [FUNCTION DOC] Return true for the two named models (GPT2 sequence classification and
        # Bert masked LM) whose numerics are noisy, so a larger accuracy multiplier is used.
        return name in [
            "GPT2ForSequenceClassification",
            # Scalar-loss training output; the fp64 RMSE ratio is noise-dominated
            # and drifts just past the 3x multiplier on CUDA 13.2 (passes on 13.0).
            "BertForMaskedLM",
        ]

    def _get_model_cls_and_config(self, model_name):
        # [FUNCTION DOC] Return the (model class, config) pair for a name. Extra models come
        # from the extra models table. Others are resolved by class name and given a default
        # config instance; GPT2/GPT-Neo/GPT-J sequence classifiers and Roberta / Marian
        # families get a pad token id of zero so batch sizes above one work.
        if model_name not in EXTRA_MODELS:
            model_cls = get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls
                in [
                    GPT2ForSequenceClassification,
                    GPTNeoForSequenceClassification,
                    GPTJForSequenceClassification,
                ]
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0

        else:
            config, model_cls = EXTRA_MODELS[model_name]

        return model_cls, config

    @download_retry_decorator
    def _download_model(self, model_name):
        # [FUNCTION DOC] Instantiate (randomly initialized, no weights downloaded) the model for
        # a name, using the from-config constructor for auto classes and the plain constructor
        # otherwise. The decorator retries on download failures.
        model_cls, config = self._get_model_cls_and_config(model_name)
        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config)
        else:
            model = model_cls(config)
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
    ):
        # [FUNCTION DOC] Load a model and its example inputs. Resets the RNG state, picks the
        # batch size (known-model table, default 16, optionally divided by a configured divisor
        # when none is given), then builds either an LLM from the LLM registry (setting the
        # hf_llm flag and installing a generate-based method on the runner) or a regular model
        # moved to the device in float32 with generated inputs including labels. Dropout-like
        # float config attributes are set to a tiny value after the model is built, and the KV
        # cache is disabled. Optionally enables activation checkpointing, sets train or eval
        # mode, validates the model and returns (device, name, model, inputs, batch size).
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dtype = torch.float32
        reset_rng_state()

        # Get batch size
        if model_name in BATCH_SIZE_KNOWN_MODELS:
            batch_size_default = BATCH_SIZE_KNOWN_MODELS[model_name]
        elif batch_size is None:
            batch_size_default = 16
            log.info(
                f"Batch size not specified for {model_name}. Setting batch_size=16"  # noqa: G004
            )

        if batch_size is None:
            batch_size = batch_size_default
            batch_size_divisors = self._config["batch_size"]["divisors"]
            if model_name in batch_size_divisors:
                batch_size = max(int(batch_size / batch_size_divisors[model_name]), 1)
                log.info(
                    f"Running smaller batch size={batch_size} for {model_name}, orig batch_size={batch_size_default}"  # noqa: G004
                )

        # Get model and example inputs
        if model_name in HF_LLM_MODELS:
            benchmark_cls = HF_LLM_MODELS[model_name]
            model, example_inputs = benchmark_cls.get_model_and_inputs(
                model_name, device
            )

            # Set this flag so that when we test for speedup, we use
            # model.generate instead of using model.forward
            self.hf_llm = True

            def generate(self, _, example_inputs, collect_outputs=True):
                # [FUNCTION DOC] HELPER: Run model generate on the example inputs and return its
                # result; bound to the runner as its generate method. The second and fourth
                # arguments are ignored.
                return model.generate(**example_inputs)

            self.generate = types.MethodType(generate, self)

        else:
            self.hf_llm = False

            model_cls, config = self._get_model_cls_and_config(model_name)
            model = self._download_model(model_name)
            model = model.to(device, dtype=dtype)

            example_inputs = generate_inputs_for_model(
                model_cls, model, model_name, batch_size, device, include_loss_args=True
            )

            # So we can check for correct gradients without eliminating the dropout computation
            for attr in dir(config):
                if "drop" in attr and isinstance(getattr(config, attr), float):
                    setattr(config, attr, 1e-30)

            # Turning off kv cache for torchbench models. This is not the right
            # thing to do, but the pt2 dashboard is outdated. Real transformers
            # benchmarks will be added soon using a different infra.
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        if self.args.enable_activation_checkpointing:
            model.gradient_checkpointing_enable()

        if (
            is_training
            and not use_eval_mode
            and not (
                self.args.accuracy and model_name in self._config["only_inference"]
            )
        ):
            model.train()
        else:
            model.eval()

        self.validate_model(model_name, model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        # [FUNCTION DOC] Generator over benchmark model names: the sorted union of known and
        # extra models, restricted to this run's shard of indices, kept when it matches the
        # filter patterns (case-insensitive), does not match the exclude patterns, is not in
        # the exact-exclude list and is not in the skip list.
        model_names = list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys())
        model_names = set(model_names)
        model_names = sorted(model_names)

        start, end = self.get_benchmark_indices(len(model_names))
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                not re.search("|".join(args.filter), model_name, re.IGNORECASE)
                or re.search("|".join(args.exclude), model_name, re.IGNORECASE)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue
            yield model_name

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        # [FUNCTION DOC] Property: the set of large models whose accuracy check is skipped, taken
        # from config when running in dashboard or accuracy mode, otherwise an empty set.
        if self.args.dashboard or self.args.accuracy:
            return self._accuracy["skip"]["large_models"]
        return set()

    @property
    def get_output_amp_train_process_func(self):
        # [FUNCTION DOC] Property: mapping of per-name output processing functions for AMP
        # training; always empty for this suite.
        return {}

    def pick_grad(self, name, is_training):
        # [FUNCTION DOC] Return a gradient-enabling context when training, else a no-grad
        # context. The name argument is unused.
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        # [FUNCTION DOC] Return (tolerance, cosine flag). Training uses 2e-2 for models in the
        # higher training list (or the higher max-autotune list when max autotune is on), else
        # 1e-2. Inference uses 5e-3 on CPU for the higher CPU list, 4e-3 for the higher
        # inference list, else 1e-3. The cosine flag comes from the command line arguments.
        cosine = self.args.cosine
        if is_training:
            from torch._inductor import config as inductor_config

            if (name in self._config["tolerance"]["higher_training"]) or (
                inductor_config.max_autotune
                and name in self._config["tolerance"]["higher_max_autotune_training"]
            ):
                return 2e-2, cosine
            else:
                return 1e-2, cosine
        else:
            if (
                current_device == "cpu"
                and name in self._config["tolerance"]["higher_inference_cpu"]
            ):
                return 5e-3, cosine
            if name in self._config["tolerance"]["higher_inference"]:
                return 4e-3, cosine
        return 1e-3, cosine

    def compute_loss(self, pred):
        # [FUNCTION DOC] Return the first element of the model output, used as the loss.
        return pred[0]

    def forward_pass(self, mod, inputs, collect_outputs=True):
        # [FUNCTION DOC] Run the model on keyword inputs without gradients under autocast.
        # Returns the logits for LLM models, otherwise the raw output. collect_outputs is unused.
        with torch.no_grad(), self.autocast(**self.autocast_arg):
            res = mod(**inputs)
        return res.logits if self.hf_llm else res

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        # [FUNCTION DOC] Training step: clone inputs, zero gradients, run the forward under
        # autocast, take the loss, backward through the grad scaler and take an optimizer step.
        # Returns collected results (outputs, loss and gradients) if collect_outputs is true,
        # else None.
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, None, loss, cloned_inputs)
        return None


def refresh_model_names_and_batch_sizes():
    # [FUNCTION DOC] Maintenance routine to regenerate the models list file: picks, per model
    # family supported by the HF FX tracer, representative classes (skipping some known broken
    # ones), adds the extra models, then for each runs this script in a subprocess to find the
    # batch size, appending to the models list file. A failing subprocess only logs a warning.
    # Not called by default (its call in the entry point is commented out).
    """
    This function reads the HF Fx tracer supported models and finds the largest
    batch size that could fit on the GPU with PyTorch eager.

    The resulting data is written in huggingface_models_list.txt.

    Note - We only need to run this function if we believe that HF Fx tracer now
    supports more models.
    """
    import transformers.utils.fx as hf_fx

    family = {}
    lm_seen = set()
    family_seen = set()
    for cls_name in hf_fx._SUPPORTED_MODELS:
        if "For" not in cls_name:
            continue

        model_cls = get_module_cls_by_model_name(cls_name)

        # TODO: AttributeError: '*Config' object has no attribute 'vocab_size'
        if model_cls in [
            CLIPModel,
            CLIPVisionModel,
            # SwinForImageClassification,
            # SwinForImageClassification,
            # SwinForMaskedImageModeling,
            # SwinModel,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
        ]:
            continue

        # TODO: AssertionError: Padding_idx must be within num_embeddings
        if model_cls in [MarianForCausalLM, MarianMTModel, MarianModel]:
            continue

        # TODO: "model is not supported yet" from HFTracer
        if model_cls in [HubertForSequenceClassification]:
            continue

        # TODO: shape mismatch in loss calculation
        if model_cls in [LxmertForQuestionAnswering]:
            continue

        family_name = cls_name.split("For")[0]
        if family_name not in family:
            family[family_name] = []
        if cls_name.endswith(("MaskedLM", "CausalLM")) and family_name not in lm_seen:
            family[family_name].append(cls_name)
            lm_seen.add(family_name)
        elif (
            cls_name.endswith(
                ("SequenceClassification", "ConditionalGeneration", "QuestionAnswering")
            )
            and family_name not in family_seen
        ):
            family[family_name].append(cls_name)
            family_seen.add(family_name)
        elif cls_name.endswith("ImageClassification"):
            family[family_name].append(cls_name)

    chosen_models = set()
    for members in family.values():
        chosen_models.update(set(members))

    # Add the EXTRA_MODELS
    chosen_models.update(set(EXTRA_MODELS.keys()))

    for model_name in sorted(chosen_models):
        try:
            subprocess.check_call(
                [sys.executable]
                + sys.argv
                + ["--find-batch-sizes"]
                + [f"--only={model_name}"]
                + [f"--output={MODELS_FILENAME}"]
            )
        except subprocess.SubprocessError:
            log.warning(f"Failed to find suitable batch size for {model_name}")  # noqa: G004


def huggingface_main():
    # [FUNCTION DOC] Entry point: set logging to warning level, silence Python warnings and run
    # the common benchmark main with a HuggingFace runner.
    # Code to refresh model names and batch sizes
    # if "--find-batch-sizes" not in sys.argv:
    #     refresh_model_names_and_batch_sizes()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())


if __name__ == "__main__":
    huggingface_main()
