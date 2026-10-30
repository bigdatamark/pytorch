#!/usr/bin/env python3
# [MODULE DOC] Dynamo benchmark suite driver for the timm (PyTorch Image Models) vision models.
# Defines a runner class built on the shared BenchmarkRunner and an entry function that runs it.
# Import-time side effects: enables the inductor FX graph cache unless the environment variable
# for it is already set; installs timm from git without dependencies if importing timm fails
# (printing a message); reads the models list text file next to this file (each line holds a
# model name and a batch size separated by one space) into the TIMM_MODELS mapping, optionally
# restricted by the comma-separated TORCHBENCH_ONLY_MODELS environment variable.

import importlib
import logging
import os
import re
import subprocess
import sys
import warnings


try:
    from .common import BenchmarkRunner, download_retry_decorator, load_yaml_file, main
except ImportError:
    from common import BenchmarkRunner, download_retry_decorator, load_yaml_file, main

import torch
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs


log = logging.getLogger(__name__)

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True


def pip_install(package, *, no_deps=False):
    # [FUNCTION DOC] Installs a package with pip using the current interpreter, adding the
    # option to skip dependency resolution when no_deps is true. Raises CalledProcessError if
    # pip fails.
    command = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        command.append("--no-deps")
    command.append(package)
    subprocess.check_call(command)


try:
    importlib.import_module("timm")
except ModuleNotFoundError:
    print("Installing PyTorch Image Models...")
    pip_install("git+https://github.com/rwightman/pytorch-image-models", no_deps=True)
finally:
    from timm import __version__ as timmversion
    from timm.data import resolve_data_config
    from timm.models import create_model

TIMM_MODELS = {}

# Run only this selected group of models, leave this empty to run everything
TORCHBENCH_ONLY_MODELS = [
    m.strip() for m in os.getenv("TORCHBENCH_ONLY_MODELS", "").split(",") if m.strip()
]

filename = os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
with open(filename) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(" ")
        if TORCHBENCH_ONLY_MODELS and model_name not in TORCHBENCH_ONLY_MODELS:
            continue
        TIMM_MODELS[model_name] = int(batch_size)


def refresh_model_names():
    # [FUNCTION DOC] HELPER: Maintenance utility, not used by the benchmark run. Collects model
    # names from the timm docs markdown files (relative path to a sibling checkout), lists all
    # pretrained timm models excluding the in21k variants, groups both by family, drops families
    # that appear in the docs from the full set, picks the first model of each remaining family
    # plus of each docs family, and writes the sorted names, one per line, to the models list
    # file (under the benchmarks directory if that exists in the working directory). Imports
    # glob and timm lazily.
    # KNOWN ISSUE: it writes only model names, while the module-level loader expects each line
    #     to hold a name and a batch size, so its output cannot be read back without editing;
    #     also the removal of docs families raises a key error if a docs family is absent from
    #     the full list.
    import glob

    from timm.models import list_models

    def read_models_from_docs():
        # [FUNCTION DOC] HELPER: Scans every markdown file under the docs models directory and
        # returns a set of model names taken from lines that start with the create-model
        # example, using the text between the first pair of single quotes on that line.
        models = set()
        # TODO - set the path to pytorch-image-models repo
        for fn in glob.glob("../pytorch-image-models/docs/models/*.md"):
            with open(fn) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.startswith("model = timm.create_model("):
                        continue

                    model = line.split("'")[1]
                    # print(model)
                    models.add(model)
        return models

    def get_family_name(name):
        # [FUNCTION DOC] HELPER: Maps a model name to a family name. Returns the first entry of
        # a fixed list of known family names contained in the model name; otherwise for names
        # starting with the gluon prefix returns the prefix plus the next underscore-separated
        # token; otherwise returns the token before the first underscore.
        known_families = [
            "darknet",
            "densenet",
            "dla",
            "dpn",
            "ecaresnet",
            "halo",
            "regnet",
            "efficientnet",
            "deit",
            "mobilevit",
            "mnasnet",
            "convnext",
            "resnet",
            "resnest",
            "resnext",
            "selecsls",
            "vgg",
            "xception",
        ]

        for known_family in known_families:
            if known_family in name:
                return known_family

        if name.startswith("gluon_"):
            return "gluon_" + name.split("_")[1]
        return name.split("_")[0]

    def populate_family(models):
        # [FUNCTION DOC] HELPER: Groups an iterable of model names into a mapping from family
        # name to the list of names in that family, preserving iteration order within each
        # family.
        family = {}
        for model_name in models:
            family_name = get_family_name(model_name)
            if family_name not in family:
                family[family_name] = []
            family[family_name].append(model_name)
        return family

    docs_models = read_models_from_docs()
    all_models = list_models(pretrained=True, exclude_filters=["*in21k"])

    all_models_family = populate_family(all_models)
    docs_models_family = populate_family(docs_models)

    for key in docs_models_family:
        del all_models_family[key]

    chosen_models = set()
    chosen_models.update(value[0] for value in docs_models_family.values())

    chosen_models.update(value[0] for key, value in all_models_family.items())

    filename = "timm_models_list.txt"
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    with open(filename, "w") as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + "\n")


class TimmRunner(BenchmarkRunner):
    # [CLASS DOC] Benchmark runner for timm models. Configuration (skip lists, batch size
    # divisors, tolerances, accuracy exceptions, dtype rules) is read from the timm models yaml
    # file through properties. Sets suite_name to timm_models and keeps a lazily captured
    # baseline of the inductor emulate-precision-casts setting. Provides model loading, model
    # name iteration (partition aware), tolerance selection, loss computation and
    # forward/backward passes.
    def __init__(self):
        # [FUNCTION DOC] Initializes the base runner, sets the suite name, and leaves the saved
        # baseline of the inductor emulate-precision-casts flag unset so that it is captured on
        # the first model load, after command line config overrides have been applied.
        super().__init__()
        self.suite_name = "timm_models"
        # Sentinel; captured lazily on first load_model call (which runs
        # AFTER main() in common.py has applied any --inductor-config CLI
        # overrides). Capturing eagerly here would always see the unmodified
        # default and silently override the user-specified value.
        self._orig_emulate_precision_casts = None

    @property
    def _config(self):
        # [FUNCTION DOC] HELPER: Loads and returns the parsed timm models yaml configuration
        # (re-read on every access).
        return load_yaml_file("timm_models.yaml")

    @property
    def _skip(self):
        # [FUNCTION DOC] HELPER: Returns the skip section of the configuration.
        return self._config["skip"]

    @property
    def _batch_size(self):
        # [FUNCTION DOC] HELPER: Returns the batch size section of the configuration (divisors
        # and optional CI accuracy sizes).
        return self._config["batch_size"]

    @property
    def _tolerance(self):
        # [FUNCTION DOC] HELPER: Returns the tolerance section of the configuration.
        return self._config["tolerance"]

    @property
    def _accuracy(self):
        # [FUNCTION DOC] HELPER: Returns the accuracy section of the configuration.
        return self._config["accuracy"]

    @property
    def _require_larger_multiplier_for_smaller_tensor(self):
        # [FUNCTION DOC] HELPER: Returns the configured collection of models that need a larger
        # multiplier for smaller tensors.
        return self._config["require_larger_multiplier_for_smaller_tensor"]

    @property
    def _emulate_precision_casts(self):
        # [FUNCTION DOC] HELPER: Returns the configured collection of models for which inductor
        # should emulate precision casts.
        return self._config["emulate_precision_casts"]

    @property
    def skip_models_for_cpu(self):
        # [FUNCTION DOC] Returns the configured collection of model names to skip on cpu.
        return self._skip["device"]["cpu"]

    @property
    def skip_models_for_cpu_aarch64(self):
        # [FUNCTION DOC] Returns the configured collection of model names to skip on aarch64
        # cpu.
        return self._skip["device"]["cpu_aarch64"]

    @property
    def skip_models(self):
        # [FUNCTION DOC] Returns the configured collection of model names skipped on all
        # devices.
        return self._skip["all"]

    @property
    def force_amp_for_fp16_bf16_models(self):
        # [FUNCTION DOC] Returns the configured models that must use automatic mixed precision
        # when run in fp16 or bf16.
        return self._config["dtype"]["force_amp_for_fp16_bf16_models"]

    @property
    def force_fp16_for_bf16_models(self):
        # [FUNCTION DOC] Returns an empty set: no models are forced to fp16 when bf16 is
        # requested.
        return set()

    @property
    def get_output_amp_train_process_func(self):
        # [FUNCTION DOC] Returns an empty mapping: no custom output processing for amp training.
        return {}

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        # [FUNCTION DOC] Returns the configured set of models whose eager output is non-
        # deterministic when running accuracy checks in training mode; otherwise an empty set.
        if self.args.accuracy and self.args.training:
            return self._accuracy["skip"]["eager_not_deterministic"]
        return set()

    @property
        # [FUNCTION DOC] Returns an empty mapping: no models are configured to guard on nn
        # module state.
    def guard_on_nn_module_models(self):
        return {}

    @download_retry_decorator
    def _download_model(self, model_name):
        # [FUNCTION DOC] Creates a pretrained timm model with three input channels, no
        # scripting, no classifier head (no classes) and zero dropout rates for drop, drop path
        # and drop block. If the factory raises a TypeError complaining about an unexpected
        # keyword argument, logs a warning and retries without the dropout arguments; other
        # TypeErrors propagate. The retry decorator repeats the whole download on failure.
        # Returns the model.
        kwargs = dict(in_chans=3, scriptable=False, num_classes=None, pretrained=True)
        try:
            model = create_model(
                model_name,
                **kwargs,
                drop_rate=0.0,
                drop_path_rate=None,
                drop_block_rate=None,
            )
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            log.warning(
                "Model %s does not support drop_rate kwargs, loading without them",
                model_name,
            )
            model = create_model(model_name, **kwargs)
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
    ):
        # [FUNCTION DOC] Builds one benchmark case. Raises NotImplementedError if activation
        # checkpointing is requested and RuntimeError if the model could not be created. Moves
        # the model to the device (channels-last layout if requested) and resolves the timm data
        # config to get the input size; the batch size is the explicit argument, else the
        # recorded size divided by the configured divisor (at least one). For cuda (non-HIP)
        # inductor CI accuracy training runs the configured CI batch size overrides it. Inputs
        # are random integer values converted to float, standardized by mean and std, with a
        # fixed seed, and wrapped in a one-element list. Creates the cross-entropy loss,
        # switches to the scaled loss for configured models, and sets the inductor emulate-
        # precision-casts flag to the saved baseline or, if the baseline is off, to whether the
        # model is listed (a global config mutation). Puts the model in train or eval mode,
        # validates it, and returns the device, name, model, example inputs and batch size.
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Timm models"
            )

        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode

        channels_last = self._args.channels_last
        model = self._download_model(model_name)

        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        model.to(
            device=device,
            memory_format=torch.channels_last if channels_last else None,
        )

        data_config = resolve_data_config(
            vars(self._args) if timmversion >= "0.8.0" else self._args,
            model=model,
            use_test_size=not is_training,
        )
        input_size = data_config["input_size"]
        recorded_batch_size = TIMM_MODELS[model_name]

        batch_size_divisors = self._batch_size["divisors"]
        if model_name in batch_size_divisors:
            recorded_batch_size = max(
                int(recorded_batch_size / batch_size_divisors[model_name]), 1
            )
        batch_size = batch_size or recorded_batch_size
        if (
            device == "cuda"
            and torch.version.hip is None
            and self.args.backend == "inductor"
            and self.args.ci
            and self.args.accuracy
            and self.args.training
        ):
            ci_accuracy_batch_sizes = self._batch_size.get("ci_accuracy", {})
            batch_size = ci_accuracy_batch_sizes.get(model_name, batch_size)

        torch.manual_seed(1337)
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=device
        ).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev

        if channels_last:
            example_inputs = example_inputs.contiguous(
                memory_format=torch.channels_last
            )
        example_inputs = [
            example_inputs,
        ]

        self.loss = torch.nn.CrossEntropyLoss().to(device)

        if model_name in self._config["scaled_compute_loss"]:
            self.compute_loss = self.scaled_compute_loss

        # See yaml note for emulate_precision_casts. This preserves the
        # autocast downcast-upcast pairs (bf16 and fp16; mobilenetv2_100 is
        # the fp16 case) that inductor would otherwise elide when fusing
        # across mixed-precision boundaries (e.g. bf16 conv-bias-add fused
        # into an fp32 cat-prep store), which is what eager autocast does.
        #
        # Baseline is captured on the FIRST load_model call so that any
        # --inductor-config emulate_precision_casts=... CLI override (applied
        # by main() in common.py between __init__ and this point) is included
        # in the baseline. Per-call assignment then sets the flag to
        # (baseline OR model-in-yaml-list), which both preserves the user
        # override across every model and toggles the flag back to baseline
        # for non-listed models (no carry-over across iter_models).
        if self._orig_emulate_precision_casts is None:
            self._orig_emulate_precision_casts = (
                torch._inductor.config.emulate_precision_casts
            )
        if self._orig_emulate_precision_casts:
            torch._inductor.config.emulate_precision_casts = True
        else:
            torch._inductor.config.emulate_precision_casts = (
                model_name in self._emulate_precision_casts
            )

        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        self.validate_model(model_name, model, example_inputs)

        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        # [FUNCTION DOC] Generator over model names from TIMM_MODELS, sorted, restricted to this
        # partition's index range. Yields names matching the filter patterns (case-insensitive)
        # that do not match the exclude patterns, are not in the exact-exclude list and are not
        # in the skip list.
        # for model_name in list_models(pretrained=True, exclude_filters=["*in21k"]):
        model_names = sorted(TIMM_MODELS.keys())
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

    def pick_grad(self, name, is_training):
        # [FUNCTION DOC] Returns a gradient-enabled context when training and a no-grad context
        # otherwise; the model name is unused.
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def use_larger_multiplier_for_smaller_tensor(self, name):
        # [FUNCTION DOC] Returns whether the model name is in the configured set needing a
        # larger multiplier for smaller tensors.
        return name in self._require_larger_multiplier_for_smaller_tensor

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        # [FUNCTION DOC] Returns the tolerance and the cosine flag from the arguments. Defaults
        # to a small tolerance; freezing runs on listed models use a larger one. In training
        # mode the value is chosen from the configured tiers (highest, even higher including a
        # max-autotune case, higher including amp and fp16-on-xpu cases), else a medium default;
        # note training always replaces any freezing value. The current device is only used for
        # the xpu case.
        cosine = self.args.cosine
        tolerance = 1e-3

        if self.args.freezing and name in self._tolerance["freezing"]:
            # the conv-batchnorm fusion used under freezing may cause relatively
            # large numerical difference. We need a larger tolerance.
            # Check https://github.com/pytorch/pytorch/issues/120545 for context
            tolerance = 8 * 1e-2

        if is_training:
            from torch._inductor import config as inductor_config

            if name in self._tolerance["highest_training"]:
                tolerance = 16 * 1e-2
            elif name in self._tolerance["even_higher"] or (
                inductor_config.max_autotune
                and name in self._tolerance["even_higher_max_autotune"]
            ):
                tolerance = 8 * 1e-2
            elif name in self._tolerance["higher_training"] or (
                self.args.amp and name in self._tolerance["higher_amp"]
            ):
                tolerance = 4 * 1e-2
            elif (
                name in self._tolerance["higher_fp16_xpu"]
                and self.args.float16
                and current_device == "xpu"
            ):
                tolerance = 4 * 1e-2
            else:
                tolerance = 1e-2
        return tolerance, cosine

    def compute_loss(self, pred):
        # [FUNCTION DOC] Reduces a prediction to a scalar loss using the dynamo testing helper.
        # High loss values make gradient checking harder, as small changes in
        # accumulation order upsets accuracy checks.
        return reduce_to_scalar_loss(pred)

    def scaled_compute_loss(self, pred):
        # [FUNCTION DOC] Same as the unscaled loss but divided by one thousand, used for models
        # with large loss values.
        # Loss values need zoom out further.
        return reduce_to_scalar_loss(pred) / 1000.0

    def forward_pass(self, mod, inputs, collect_outputs=True):
        # [FUNCTION DOC] Runs the model on the inputs under the autocast context and returns its
        # output. The collect_outputs argument is unused.
        with self.autocast(**self.autocast_arg):
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        # [FUNCTION DOC] Clones the inputs, clears gradients, then under autocast runs the model
        # (taking the first element if it returns a tuple) and computes the loss. Backpropagates
        # the scaled loss and steps the optimizer if any. Returns collected results when
        # collect_outputs is true, otherwise None.
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            pred = mod(*cloned_inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, None, loss, cloned_inputs)
        return None


def timm_main():
    # [FUNCTION DOC] Entry point: sets the logging level to warning, silences all warnings, and
    # runs the shared benchmark main with a timm runner.
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TimmRunner())


if __name__ == "__main__":
    timm_main()
